"""Batch fitting orchestrator for the PRL pick_best_cue pipeline.

# Frozen for v1.1 reproducibility — DO NOT MODIFY.
# See src/prl_hgf/fitting/hierarchical.py for the v1.2+ implementation.

Fits all participant-sessions in a simulation DataFrame via NUTS MCMC and
aggregates results into a single tidy DataFrame.

The output has one row per (participant, session, parameter) and includes
convergence diagnostics and a ``flagged`` column indicating whether R-hat
or ESS thresholds were exceeded.

JIT pre-warm
------------
Before the main loop the function builds a minimal Op once with dummy data
and evaluates it to trigger JAX JIT compilation.  This pays the one-time
compilation cost upfront so the first real participant-session does not pay
the overhead.

Error handling
--------------
If a single fit raises an exception (e.g. PyMC SamplingError from
divergences), the exception is caught and logged as a warning, and NaN-filled
rows are appended so the participant is still represented in the output.  The
batch continues without crashing.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from prl_hgf.fitting.legacy.single import fit_participant

__all__ = ["fit_batch"]

log = logging.getLogger(__name__)

# FIT-04 output column order
_RESULT_COLUMNS = [
    "participant_id",
    "group",
    "session",
    "model",
    "parameter",
    "mean",
    "sd",
    "hdi_3%",
    "hdi_97%",
    "r_hat",
    "ess",
    "flagged",
]


# ---------------------------------------------------------------------------
# JIT pre-warm helper
# ---------------------------------------------------------------------------


def _prewarm_jit(model_name: str) -> None:
    """Trigger JAX JIT compilation before the main batch loop.

    Builds a minimal Op for the requested model variant and evaluates it once
    with dummy data.  This pays the one-time JAX compilation cost upfront.

    Parameters
    ----------
    model_name : str
        ``"hgf_2level"`` or ``"hgf_3level"``.
    """
    from prl_hgf.fitting.ops import (
        build_logp_ops_2level,
        build_logp_ops_3level,
    )

    dummy_input = np.zeros((5, 3), dtype=float)
    dummy_obs = np.zeros((5, 3), dtype=int)
    dummy_obs[:, 0] = 1
    dummy_choices = np.zeros(5, dtype=int)

    if model_name == "hgf_2level":
        op, _ = build_logp_ops_2level(dummy_input, dummy_obs, dummy_choices)
        # Evaluate once to trigger JIT
        import pytensor.tensor as pt

        o2 = pt.as_tensor_variable(-3.0)
        b = pt.as_tensor_variable(3.0)
        z = pt.as_tensor_variable(0.5)
        _ = op(o2, b, z).eval()
    else:
        op, _ = build_logp_ops_3level(dummy_input, dummy_obs, dummy_choices)
        import pytensor.tensor as pt

        o2 = pt.as_tensor_variable(-3.0)
        o3 = pt.as_tensor_variable(-6.0)
        k = pt.as_tensor_variable(1.0)
        b = pt.as_tensor_variable(3.0)
        z = pt.as_tensor_variable(0.5)
        _ = op(o2, o3, k, b, z).eval()


# ---------------------------------------------------------------------------
# Input array construction
# ---------------------------------------------------------------------------


def _build_arrays(
    subset: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (input_data_arr, observed_arr, choices_arr) from trial subset.

    Implements the partial-feedback protocol: only the chosen cue receives a
    reward signal on each trial.  Unchosen cues have ``observed=0``.

    Parameters
    ----------
    subset : pandas.DataFrame
        Rows for one participant-session with columns ``cue_chosen`` and
        ``reward``.  Must be sorted by trial order (ascending trial index).

    Returns
    -------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.
    """
    n_trials = len(subset)
    choices = subset["cue_chosen"].to_numpy(dtype=int)
    rewards = subset["reward"].to_numpy(dtype=float)

    input_data_arr = np.zeros((n_trials, 3), dtype=float)
    observed_arr = np.zeros((n_trials, 3), dtype=int)

    for t in range(n_trials):
        cue = choices[t]
        input_data_arr[t, cue] = rewards[t]
        observed_arr[t, cue] = 1

    return input_data_arr, observed_arr, choices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_batch(
    sim_df: pd.DataFrame,
    model_name: str = "hgf_2level",
    n_chains: int = 4,
    n_draws: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42,
    cores: int = 1,
    output_path: Path | None = None,
    log_every: int = 10,
    return_idata: bool = False,
    sampler: str = "pymc",
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple, object]]:
    """Fit all participant-sessions in a simulation DataFrame via NUTS MCMC.

    Groups the trial-level DataFrame by ``(participant_id, group, session)``
    and calls :func:`~prl_hgf.fitting.legacy.single.fit_participant` for each
    unique participant-session.  Results are aggregated into a single tidy
    DataFrame in the FIT-04 schema.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Tidy trial-level DataFrame from
        :func:`~prl_hgf.simulation.batch.simulate_batch`.  Required columns:
        ``participant_id``, ``group``, ``session``, ``cue_chosen``,
        ``reward``.
    model_name : str, optional
        Model variant to fit.  One of ``"hgf_2level"`` (default) or
        ``"hgf_3level"``.
    n_chains : int, optional
        Number of MCMC chains per participant.  Default ``4``.
    n_draws : int, optional
        Posterior draws per chain after tuning.  Default ``1000``.
    n_tune : int, optional
        Tuning (warm-up) steps per chain.  Default ``1000``.
    target_accept : float, optional
        NUTS step-size adaptation target acceptance rate.  Default ``0.9``.
    random_seed : int, optional
        Base RNG seed.  Each participant gets ``random_seed + flat_idx`` to
        ensure independent but reproducible seeds across the batch.
    cores : int, optional
        Number of parallel chains.  Use ``1`` on Windows to avoid JAX
        process-isolation issues.  Default ``1``.
    sampler : str, optional
        MCMC backend.  ``"pymc"`` (default) or ``"numpyro"`` (JAX-native
        NUTS, bypasses PyTensor compilation).  Passed to
        :func:`~prl_hgf.fitting.legacy.single.fit_participant`.
    output_path : Path or None, optional
        If provided, the results DataFrame is saved as CSV at this path.
    log_every : int, optional
        Print progress every this many participant-sessions.  Default ``10``.
    return_idata : bool, optional
        If ``True``, also return a dict mapping
        ``(participant_id, group, session)`` tuples to
        :class:`arviz.InferenceData` objects (``None`` for failed fits).
        Default ``False`` preserves backward-compatible single-return behaviour.

    Returns
    -------
    pandas.DataFrame
        Tidy results DataFrame with one row per (participant, session,
        parameter).  Columns: ``participant_id``, ``group``, ``session``,
        ``model``, ``parameter``, ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
        ``r_hat``, ``ess``, ``flagged``.

        When ``return_idata=True``, returns a ``(DataFrame, idata_dict)``
        tuple where ``idata_dict`` is keyed by
        ``(participant_id, group, session)``.

    Notes
    -----
    Failed individual fits (e.g. PyMC SamplingError) are caught and logged.
    NaN-filled rows are added for failed participants so they appear in the
    output with missing values rather than being silently dropped.

    The ``flagged`` column is ``True`` when any parameter has
    ``r_hat > 1.05`` or ``ess < 400`` (configurable via
    :func:`~prl_hgf.fitting.legacy.single.flag_fit`).

    Examples
    --------
    >>> import pandas as pd
    >>> from prl_hgf.fitting.batch import fit_batch
    >>> # See tests/test_fitting.py for a minimal runnable example.
    """
    # Validate required columns
    required_cols = {"participant_id", "group", "session", "cue_chosen", "reward"}
    missing_cols = required_cols - set(sim_df.columns)
    if missing_cols:
        raise ValueError(
            f"sim_df is missing required columns: {sorted(missing_cols)}. "
            f"Got columns: {sorted(sim_df.columns)}"
        )

    # JIT pre-warm
    print(f"Pre-warming JAX JIT compilation for {model_name}...")
    try:
        _prewarm_jit(model_name)
        print("JIT pre-warm complete.")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"JIT pre-warm failed (non-fatal): {exc}",
            stacklevel=2,
        )

    # Group by participant-session
    group_keys = ["participant_id", "group", "session"]
    groups = list(sim_df.groupby(group_keys, sort=False))
    n_total = len(groups)
    print(f"Fitting {n_total} participant-sessions with model={model_name!r}")

    all_rows: list[dict] = []
    idata_dict: dict[tuple, object] = {}
    t_start = time.time()

    for flat_idx, ((participant_id, group, session), subset) in enumerate(groups):
        # Sort by trial index if column exists
        if "trial" in subset.columns:
            subset = subset.sort_values("trial")

        input_data_arr, observed_arr, choices_arr = _build_arrays(subset)

        # Use per-participant seed derived from base seed for reproducibility
        participant_seed = random_seed + flat_idx

        try:
            idata, summary_rows, flagged = fit_participant(
                input_data_arr=input_data_arr,
                observed_arr=observed_arr,
                choices_arr=choices_arr,
                participant_id=participant_id,
                group=group,
                session=session,
                model_name=model_name,
                n_chains=n_chains,
                n_draws=n_draws,
                n_tune=n_tune,
                target_accept=target_accept,
                random_seed=participant_seed,
                cores=cores,
                sampler=sampler,
            )
            # Add flagged column to each row
            for row in summary_rows:
                row["flagged"] = flagged
            all_rows.extend(summary_rows)

            if return_idata:
                idata_dict[(participant_id, group, session)] = idata

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Fit FAILED for %s (%s / %s): %s",
                participant_id,
                group,
                session,
                exc,
            )
            # Add NaN-filled rows so participant appears in output
            nan_rows = _make_nan_rows(
                participant_id=participant_id,
                group=group,
                session=session,
                model_name=model_name,
            )
            all_rows.extend(nan_rows)

            if return_idata:
                idata_dict[(participant_id, group, session)] = None

        # Progress logging
        if (flat_idx + 1) % log_every == 0 or (flat_idx + 1) == n_total:
            elapsed = time.time() - t_start
            rate = (flat_idx + 1) / elapsed if elapsed > 0 else float("inf")
            remaining = (n_total - flat_idx - 1) / rate if rate > 0 else float("inf")
            print(
                f"[{flat_idx + 1}/{n_total}] {participant_id} "
                f"({group} / {session}) done "
                f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    if not all_rows:
        results_df = pd.DataFrame(columns=_RESULT_COLUMNS)
    else:
        results_df = pd.DataFrame(all_rows)[_RESULT_COLUMNS]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Saved batch results to: {output_path}")

    n_flagged = int(results_df["flagged"].sum()) if "flagged" in results_df.columns else 0
    n_rows = len(results_df)
    print(
        f"Batch complete: {n_rows} result rows, "
        f"{n_flagged} flagged rows, "
        f"{time.time() - t_start:.1f}s total"
    )

    if return_idata:
        return results_df, idata_dict
    return results_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_nan_rows(
    participant_id: str,
    group: str,
    session: str,
    model_name: str,
) -> list[dict]:
    """Build NaN-filled result rows for a failed fit.

    Returns one NaN row per expected parameter so the participant appears
    in the output DataFrame with identifiable metadata but missing values.

    Parameters
    ----------
    participant_id : str
        Participant identifier.
    group : str
        Group label.
    session : str
        Session label.
    model_name : str
        Model name used to determine the expected parameter list.

    Returns
    -------
    list[dict]
        NaN-filled rows, one per expected parameter.
    """
    if model_name == "hgf_3level":
        params = ["omega_2", "omega_3", "kappa", "beta", "zeta"]
    else:
        params = ["omega_2", "beta", "zeta"]

    rows = []
    for param in params:
        rows.append(
            {
                "participant_id": participant_id,
                "group": group,
                "session": session,
                "model": model_name,
                "parameter": param,
                "mean": float("nan"),
                "sd": float("nan"),
                "hdi_3%": float("nan"),
                "hdi_97%": float("nan"),
                "r_hat": float("nan"),
                "ess": float("nan"),
                "flagged": True,
            }
        )
    return rows
