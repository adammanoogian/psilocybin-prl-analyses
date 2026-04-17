"""Post-hoc trajectory export for the PAT-RL / dcm_pytorch integration surface.

For each fitted subject, this module runs a single forward pass of the PAT-RL
HGF at posterior-mean parameters, extracts per-trial belief trajectories
``(mu2, sigma2, mu3, sigma3, delta1, epsilon2, epsilon3, psi2)``, joins them
with trial metadata, and writes one CSV per subject.  A separate function
emits a per-subject parameter-summary CSV with posterior mean + 94% HDI for
each free parameter.

These CSVs constitute the **producer side** of the PRL.4 integration surface
with dcm_pytorch's bilinear-DCM modulator channel.  The ``outcome_time_s``
column feeds directly into ``stimulus["times"]`` (absolute seconds from
session start); the HGF belief columns feed into ``stimulus["values"][:, j]``
for whichever modulator channels the HEART2ADAPT caller selects.

Schema finalized in ``.planning/phases/18-pat-rl-task-adaptation/
18-05-dcm-interface-notes.md`` after reading
``dcm_pytorch/src/pyro_dcm/simulators/task_simulator.py`` and
``forward_models/neural_state.py``.

Do **not** modify ``src/prl_hgf/analysis/__init__.py``; callers import
directly::

    from prl_hgf.analysis.export_trajectories import (
        export_subject_trajectories,
        export_subject_parameters,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az

from prl_hgf.models.hgf_2level_patrl import (
    BELIEF_NODE,
    INPUT_NODE,
    build_2level_network_patrl,
)
from prl_hgf.models.hgf_3level_patrl import (
    VOLATILITY_NODE,
    build_3level_network_patrl,
)

__all__ = [
    "_safe_temp",
    "export_subject_trajectories",
    "export_subject_parameters",
]

# ---------------------------------------------------------------------------
# Column order for the per-trial trajectory CSV (frozen schema)
# ---------------------------------------------------------------------------

_TRAJECTORY_COLUMNS: list[str] = [
    "participant_id",
    "trial_idx",
    "run_idx",
    "trial_in_run",
    "regime",
    "outcome_time_s",
    "state",
    "choice",
    "reward_mag",
    "shock_mag",
    "delta_hr",
    "mu2",
    "sigma2",
    "mu3",
    "sigma3",
    "delta1",
    "epsilon2",
    "epsilon3",
    "psi2",
]


# ---------------------------------------------------------------------------
# pyhgf attribute safety helper
# ---------------------------------------------------------------------------


def _safe_temp(traj_node: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """Extract a per-trial array from a pyhgf node temp sub-dict.

    Returns the array if the key exists; otherwise returns an array of
    ``np.nan`` with the given *shape*.  This protects the export against
    future pyhgf version drift where temp-dict keys may be renamed.

    Parameters
    ----------
    traj_node : dict
        A single node's trajectory dict (e.g. ``net.node_trajectories[1]``).
    key : str
        Name of the key inside the ``"temp"`` sub-dict
        (e.g. ``"value_prediction_error"``).
    shape : tuple of int
        Expected shape of the result.  Used only when the key is absent.

    Returns
    -------
    np.ndarray
        Array of shape *shape* and dtype float64.  Contains the real values
        when the key is present, otherwise ``np.nan`` throughout.
    """
    temp = traj_node.get("temp", {})
    if key in temp:
        return np.asarray(temp[key], dtype=np.float64)
    return np.full(shape, np.nan, dtype=np.float64)


# ---------------------------------------------------------------------------
# Primary export: per-trial belief trajectories
# ---------------------------------------------------------------------------


def export_subject_trajectories(
    participant_id: str,
    idata: az.InferenceData,
    trials: list[Any],
    choices: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Export one subject's per-trial HGF trajectory CSV.

    Runs a single forward pass of the PAT-RL Network with posterior-mean
    parameters, extracts per-trial belief arrays, joins with trial metadata,
    and writes a CSV.

    Parameters
    ----------
    participant_id : str
        Subject identifier.  Must match a ``participant_id`` coordinate in
        ``idata.posterior``.
    idata : az.InferenceData
        ArviZ InferenceData object from MCMC fitting.  Must have a
        ``posterior`` group with a ``participant_id`` coordinate.
    trials : list[PATRLTrial]
        Trial list from ``generate_session_patrl`` (length ``n_trials``).
        Provides trial metadata: ``trial_idx``, ``run_idx``,
        ``trial_in_run``, ``regime``, ``state``, ``reward_mag``,
        ``shock_mag``, ``delta_hr``, ``outcome_time_s``.
    choices : np.ndarray
        Shape ``(n_trials,)`` int array of recorded choices (0=avoid,
        1=approach).
    model_name : str
        Either ``"hgf_2level_patrl"`` or ``"hgf_3level_patrl"``.  Controls
        which Network builder is used and which belief columns are populated
        (3-level fills ``mu3``, ``sigma3``, ``epsilon3``; 2-level sets those
        to NaN).
    output_dir : Path
        Directory where the CSV is written.  Created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the written CSV file
        (``output_dir / f"{participant_id}_trajectories.csv"``).

    Raises
    ------
    ValueError
        If ``model_name`` is not one of the two recognised strings, or if
        ``participant_id`` is not in ``idata.posterior.participant_id``.
    """
    if model_name not in {"hgf_2level_patrl", "hgf_3level_patrl"}:
        raise ValueError(
            f"export_subject_trajectories: model_name must be "
            f"'hgf_2level_patrl' or 'hgf_3level_patrl', "
            f"got {model_name!r}."
        )

    post = idata.posterior  # type: ignore[attr-defined]
    pid_coord = post.coords.get("participant_id", None)
    if pid_coord is None or participant_id not in pid_coord.values:
        raise ValueError(
            f"export_subject_trajectories: participant_id={participant_id!r} "
            f"not found in idata.posterior participant_id coordinate."
        )

    # --- 1. Extract posterior means for this participant --------------------
    def _post_mean(var: str) -> float:
        return float(
            post[var]
            .sel(participant_id=participant_id)
            .mean()
            .values
        )

    omega_2 = _post_mean("omega_2")
    is_3level = model_name == "hgf_3level_patrl"

    # --- 2. Build Network at posterior means --------------------------------
    if is_3level:
        omega_3 = _post_mean("omega_3")
        kappa = _post_mean("kappa")
        mu3_0 = _post_mean("mu3_0")
        net = build_3level_network_patrl(
            omega_2=omega_2,
            omega_3=omega_3,
            kappa=kappa,
            mu3_0=mu3_0,
        )
    else:
        net = build_2level_network_patrl(omega_2=omega_2)

    # --- 3. Build input array: binary state observations --------------------
    n_trials = len(trials)
    u = np.array([t.state for t in trials], dtype=np.float64)  # (n_trials,)

    # --- 4. Run forward pass -------------------------------------------------
    # pyhgf 0.2.8: input_data expects (n_trials, 1) and time_steps 1D np.ones
    net.input_data(
        input_data=u[:, None],
        time_steps=np.ones(n_trials, dtype=np.float64),
    )

    # --- 5. Extract per-trial trajectories from node_trajectories ------------
    traj_node1 = net.node_trajectories[BELIEF_NODE]
    traj_node0 = net.node_trajectories[INPUT_NODE]

    mu2 = np.asarray(traj_node1["mean"], dtype=np.float64)
    sigma2 = 1.0 / np.asarray(traj_node1["precision"], dtype=np.float64)

    psi2 = _safe_temp(traj_node1, "effective_precision", (n_trials,))
    epsilon2 = _safe_temp(traj_node1, "value_prediction_error", (n_trials,))
    delta1 = _safe_temp(traj_node0, "value_prediction_error", (n_trials,))

    if is_3level:
        traj_node2 = net.node_trajectories[VOLATILITY_NODE]
        mu3 = np.asarray(traj_node2["mean"], dtype=np.float64)
        sigma3 = 1.0 / np.asarray(traj_node2["precision"], dtype=np.float64)
        epsilon3 = _safe_temp(traj_node1, "volatility_prediction_error", (n_trials,))
    else:
        mu3 = np.full(n_trials, np.nan, dtype=np.float64)
        sigma3 = np.full(n_trials, np.nan, dtype=np.float64)
        epsilon3 = np.full(n_trials, np.nan, dtype=np.float64)

    # --- 6. Build DataFrame with frozen schema column order -----------------
    choices_arr = np.asarray(choices, dtype=np.int32)

    df = pd.DataFrame(
        {
            "participant_id": participant_id,
            "trial_idx": np.array([t.trial_idx for t in trials], dtype=np.int32),
            "run_idx": np.array([t.run_idx for t in trials], dtype=np.int32),
            "trial_in_run": np.array([t.trial_in_run for t in trials], dtype=np.int32),
            "regime": [t.regime for t in trials],
            "outcome_time_s": np.array([t.outcome_time_s for t in trials], dtype=np.float64),
            "state": np.array([t.state for t in trials], dtype=np.int32),
            "choice": choices_arr,
            "reward_mag": np.array([t.reward_mag for t in trials], dtype=np.float64),
            "shock_mag": np.array([t.shock_mag for t in trials], dtype=np.float64),
            "delta_hr": np.array([t.delta_hr for t in trials], dtype=np.float64),
            "mu2": mu2,
            "sigma2": sigma2,
            "mu3": mu3,
            "sigma3": sigma3,
            "delta1": delta1,
            "epsilon2": epsilon2,
            "epsilon3": epsilon3,
            "psi2": psi2,
        }
    )

    # Enforce column order
    df = df[_TRAJECTORY_COLUMNS]

    # --- 7. Write CSV --------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{participant_id}_trajectories.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Secondary export: per-subject parameter summary
# ---------------------------------------------------------------------------

_PARAMS_2LEVEL: list[str] = ["omega_2", "beta"]
_PARAMS_3LEVEL: list[str] = ["omega_2", "omega_3", "kappa", "beta", "mu3_0"]


def export_subject_parameters(
    idata: az.InferenceData,
    model_name: str,
    output_dir: Path,
    filename: str = "parameter_summary.csv",
) -> Path:
    """Export per-subject posterior mean + 94% HDI for each parameter.

    Produces a long-format CSV with one row per ``(participant_id,
    parameter)`` combination.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData object with a ``posterior`` group containing a
        ``participant_id`` coordinate.
    model_name : str
        Either ``"hgf_2level_patrl"`` or ``"hgf_3level_patrl"``.  Determines
        which parameters are exported (2-level: ``omega_2, beta``; 3-level:
        ``omega_2, omega_3, kappa, beta, mu3_0``).
    output_dir : Path
        Directory where the CSV is written.  Created if it does not exist.
    filename : str, optional
        File name for the output CSV.  Default ``"parameter_summary.csv"``.

    Returns
    -------
    Path
        Absolute path to the written CSV file.

    Raises
    ------
    ValueError
        If ``model_name`` is not one of the two recognised strings.

    Notes
    -----
    ``az.hdi`` (ArviZ 0.22+) returns a Dataset with shape
    ``(participant_id, hdi)`` where the ``hdi`` coordinate has values
    ``"lower"`` and ``"higher"``.  The function pivots this into long format.
    """
    import arviz as az  # noqa: PLC0415

    if model_name not in {"hgf_2level_patrl", "hgf_3level_patrl"}:
        raise ValueError(
            f"export_subject_parameters: model_name must be "
            f"'hgf_2level_patrl' or 'hgf_3level_patrl', "
            f"got {model_name!r}."
        )

    params = _PARAMS_3LEVEL if model_name == "hgf_3level_patrl" else _PARAMS_2LEVEL

    post = idata.posterior  # type: ignore[attr-defined]
    participant_ids = list(post.coords["participant_id"].values)

    # Compute posterior means for all params at once
    # az.hdi returns Dataset; shape per var is (participant_id, hdi)
    hdi_ds = az.hdi(post[params], hdi_prob=0.94)

    rows: list[dict[str, object]] = []
    for param in params:
        mean_da = post[param].mean(dim=["chain", "draw"])  # (participant_id,)
        hdi_da = hdi_ds[param]  # (participant_id, hdi)

        for pid in participant_ids:
            mean_val = float(mean_da.sel(participant_id=pid).values)
            hdi_low = float(hdi_da.sel(participant_id=pid, hdi="lower").values)
            hdi_high = float(hdi_da.sel(participant_id=pid, hdi="higher").values)
            rows.append(
                {
                    "participant_id": pid,
                    "parameter": param,
                    "posterior_mean": mean_val,
                    "hdi_low": hdi_low,
                    "hdi_high": hdi_high,
                }
            )

    summary_df = pd.DataFrame(
        rows,
        columns=["participant_id", "parameter", "posterior_mean", "hdi_low", "hdi_high"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    summary_df.to_csv(out_path, index=False)
    return out_path
