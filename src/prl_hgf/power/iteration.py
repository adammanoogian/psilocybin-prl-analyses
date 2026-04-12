"""Core power iteration pipeline: simulate -> fit -> BF -> BMS -> diagnostics.

Orchestrates a single power sweep cell: generates synthetic data, fits both
HGF model variants, computes Bayes Factor contrasts, runs random-effects BMS,
and extracts parameter recovery diagnostics.  Returns result dicts (one per
contrast type) conforming to :data:`~prl_hgf.power.schema.POWER_SCHEMA`.

Two iteration strategies are provided:

- :func:`run_power_iteration` — fixed-grid BFDA: one (N, d, iter) cell per
  call. Fits at exactly the requested N.
- :func:`run_sbf_iteration` — Sequential Bayes Factor: simulates at max N,
  fits once, then subsamples posteriors at each N level. 3.8x fewer MCMC
  fits.

Two fitting backends are available inside :func:`run_sbf_iteration`:

- Default (``use_legacy=False``): calls :func:`~prl_hgf.fitting.hierarchical\
.fit_batch_hierarchical`, the v1.2 batched JAX path.  Returns a single joint
  ``arviz.InferenceData`` for all participants.
- Legacy (``use_legacy=True``): calls the original v1.1
  :func:`~prl_hgf.fitting.batch.fit_batch` sequential path.  Preserved for
  reproducibility and debugging (VALID-05).

Memory strategy
---------------
The legacy BMS path (:func:`_compute_bms_power`) processes idata
incrementally and deletes 3-level idata after computing its WAIC to limit
peak memory.  The SBF path (:func:`_compute_waic_table`) keeps both idata
dicts alive until all N levels are processed, then deletes them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prl_hgf.analysis.bms import compute_subject_waic, run_group_bms
from prl_hgf.analysis.recovery import build_recovery_df, compute_recovery_metrics
from prl_hgf.fitting.batch import fit_batch
from prl_hgf.power.config import PowerConfig, make_power_config
from prl_hgf.power.contrasts import compute_all_contrasts
from prl_hgf.simulation.batch import simulate_batch

__all__ = ["run_power_iteration", "run_sbf_iteration", "build_arrays_from_sim"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array builder (duplicates fitting.batch._build_arrays without importing it)
# ---------------------------------------------------------------------------


def build_arrays_from_sim(
    subset: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build trial arrays from a simulation DataFrame subset.

    Duplicates the logic of ``fitting.batch._build_arrays`` for use in the
    power analysis pipeline.  The ``power/`` package does not import private
    functions from other subpackages.

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
    observed_arr : numpy.ndarray, shape (n_trials, 3)
        Binary observed mask (int).
    choices_arr : numpy.ndarray, shape (n_trials,)
        Chosen cue index for each trial (int).
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
# BMS power helper
# ---------------------------------------------------------------------------


def _compute_bms_power(
    sim_df: pd.DataFrame,
    fit_df_3: pd.DataFrame,
    fit_df_2: pd.DataFrame,
    idata_3level: dict[tuple, object],
    idata_2level: dict[tuple, object],
) -> tuple[float, bool]:
    """Compute BMS exceedance probability for the 3-level model.

    Uses incremental WAIC to limit peak memory: processes 3-level idata
    first, deletes it, then processes 2-level idata.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level simulation DataFrame.
    fit_df_3 : pandas.DataFrame
        Fit results for the 3-level model.
    fit_df_2 : pandas.DataFrame
        Fit results for the 2-level model.
    idata_3level : dict[tuple, object]
        Mapping ``(participant_id, group, session) -> InferenceData`` for the
        3-level model.  ``None`` values indicate failed fits.
    idata_2level : dict[tuple, object]
        Mapping ``(participant_id, group, session) -> InferenceData`` for the
        2-level model.  ``None`` values indicate failed fits.

    Returns
    -------
    tuple[float, bool]
        ``(xp_3level, bms_correct)`` where ``bms_correct`` is True when
        ``xp_3level > 0.75``.
    """
    # Get unique participant-session keys
    keys = (
        sim_df[["participant_id", "group", "session"]]
        .drop_duplicates()
        .values.tolist()
    )

    waic_rows: list[dict] = []

    # Phase A: 3-level WAIC (while idata_3level is in memory)
    for pid, grp, sess in keys:
        idata = idata_3level.get((pid, grp, sess))
        if idata is None:
            continue

        mask = (
            (sim_df["participant_id"] == pid)
            & (sim_df["group"] == grp)
            & (sim_df["session"] == sess)
        )
        subset = sim_df.loc[mask].sort_values("trial")
        input_arr, obs_arr, choices_arr = build_arrays_from_sim(subset)

        try:
            elpd = compute_subject_waic(
                input_arr, obs_arr, choices_arr, idata, "hgf_3level"
            )
            waic_rows.append(
                {
                    "participant_id": pid,
                    "model": "hgf_3level",
                    "elpd_waic": elpd,
                }
            )
        except Exception:
            log.exception(
                "WAIC failed for %s (%s/%s) model=hgf_3level", pid, grp, sess
            )

    # Free 3-level idata before Phase B
    del idata_3level

    # Phase B: 2-level WAIC (only idata_2level in memory now)
    for pid, grp, sess in keys:
        idata = idata_2level.get((pid, grp, sess))
        if idata is None:
            continue

        mask = (
            (sim_df["participant_id"] == pid)
            & (sim_df["group"] == grp)
            & (sim_df["session"] == sess)
        )
        subset = sim_df.loc[mask].sort_values("trial")
        input_arr, obs_arr, choices_arr = build_arrays_from_sim(subset)

        try:
            elpd = compute_subject_waic(
                input_arr, obs_arr, choices_arr, idata, "hgf_2level"
            )
            waic_rows.append(
                {
                    "participant_id": pid,
                    "model": "hgf_2level",
                    "elpd_waic": elpd,
                }
            )
        except Exception:
            log.exception(
                "WAIC failed for %s (%s/%s) model=hgf_2level", pid, grp, sess
            )

    if not waic_rows:
        return 0.5, False

    waic_df = pd.DataFrame(waic_rows)

    # Average elpd_waic across sessions per (participant_id, model)
    avg = (
        waic_df.groupby(["participant_id", "model"])["elpd_waic"]
        .mean()
        .reset_index()
    )

    # Pivot to (n_subjects, 2) matrix: columns [hgf_2level, hgf_3level]
    model_names = ["hgf_2level", "hgf_3level"]
    pivot = avg.pivot(
        index="participant_id", columns="model", values="elpd_waic"
    )
    missing_models = [m for m in model_names if m not in pivot.columns]
    if missing_models:
        log.warning("BMS: models %s missing from WAIC results", missing_models)
        return 0.5, False

    pivot = pivot[model_names].dropna()
    if len(pivot) < 3:
        return 0.5, False

    matrix = pivot.to_numpy(dtype=float)
    bms_result = run_group_bms(matrix, model_names)

    # Index 1 = hgf_3level (critical: model_names order is [2level, 3level])
    xp_3level = float(bms_result["xp"][1])
    return xp_3level, xp_3level > 0.75


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------


def _extract_diagnostics(
    sim_df: pd.DataFrame,
    fit_df: pd.DataFrame,
) -> tuple[float, int, float]:
    """Extract recovery, divergence, and convergence diagnostics.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level simulation DataFrame with ``true_*`` columns.
    fit_df : pandas.DataFrame
        Fit results DataFrame from ``fit_batch``.

    Returns
    -------
    tuple[float, int, float]
        ``(recovery_r, n_divergences, mean_rhat)`` where ``recovery_r`` is
        the Pearson correlation for ``omega_2`` between true and fitted
        values, ``n_divergences`` is always 0 (placeholder — ``fit_batch``
        does not currently produce this column), and ``mean_rhat`` is the
        average R-hat across all fitted parameters.
    """
    # Recovery r for omega_2
    recovery_r = float("nan")
    try:
        recovery_df = build_recovery_df(
            sim_df, fit_df, exclude_flagged=True, min_n=0
        )
        if len(recovery_df) > 0:
            metrics_df = compute_recovery_metrics(recovery_df)
            omega_2_row = metrics_df[metrics_df["parameter"] == "omega_2"]
            if len(omega_2_row) > 0:
                recovery_r = float(omega_2_row.iloc[0]["r"])
    except Exception:
        log.exception("Recovery metric computation failed")

    # n_divergences: placeholder (fit_batch doesn't produce this column)
    n_divergences = 0
    if "n_divergences" in fit_df.columns:
        n_divergences = int(fit_df["n_divergences"].sum())

    # mean_rhat: NaN-safe average
    mean_rhat = float("nan")
    if "r_hat" in fit_df.columns:
        mean_rhat = float(np.nanmean(fit_df["r_hat"].to_numpy(dtype=float)))

    return recovery_r, n_divergences, mean_rhat


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_power_iteration(
    base_config: object,
    n_per_group: int,
    effect_size_delta: float,
    iteration: int,
    child_seed: int,
    power_config: PowerConfig,
    n_chains: int = 2,
    n_draws: int = 500,
    n_tune: int = 500,
    sampler: str = "pymc",
) -> list[dict]:
    """Run one full simulate-fit-BF-BMS iteration for a power sweep cell.

    Orchestrates the full pipeline for a single ``(N, d, iteration)`` cell:

    1. Build frozen config with overridden sample size and effect.
    2. Simulate a synthetic cohort.
    3. Fit the 3-level model and compute BF contrasts + diagnostics.
    4. Fit the 2-level model for BMS comparison.
    5. Compute BMS exceedance probability (incremental WAIC).
    6. Return 3 result dicts (one per contrast type).

    Parameters
    ----------
    base_config : AnalysisConfig
        Base analysis configuration loaded from YAML.
    n_per_group : int
        Number of participants per group for this cell.
    effect_size_delta : float
        Additive shift to psilocybin group's omega_2 deltas.
    iteration : int
        Iteration index within this grid cell.
    child_seed : int
        RNG seed for this specific iteration.
    power_config : PowerConfig
        Power analysis grid configuration (provides ``bf_threshold``).
    n_chains : int, optional
        Number of MCMC chains.  Default ``2``.
    n_draws : int, optional
        Posterior draws per chain.  Default ``500``.
    n_tune : int, optional
        Tuning steps per chain.  Default ``500``.
    sampler : str, optional
        MCMC backend: ``"pymc"`` (default) or ``"numpyro"``.

    Returns
    -------
    list[dict]
        Three dicts conforming to :data:`~prl_hgf.power.schema.POWER_SCHEMA`
        (13 columns each), one per contrast type: ``did_postdose``,
        ``did_followup``, ``linear_trend``.
    """
    # Step 1: Build frozen config
    cfg = make_power_config(
        base_config, n_per_group, effect_size_delta, child_seed
    )

    # Step 2: Simulate
    sim_df = simulate_batch(cfg)

    # Step 3: Fit 3-level model (primary for BF path)
    fit_df_3, idata_3level = fit_batch(
        sim_df,
        "hgf_3level",
        return_idata=True,
        random_seed=child_seed,
        cores=1,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        sampler=sampler,
    )

    # Step 4: Compute contrasts + BF (uses fit_df_3 for omega_2 posteriors)
    contrast_results = compute_all_contrasts(
        fit_df_3,
        parameter="omega_2",
        bf_threshold=power_config.bf_threshold,
    )

    # Step 5: Extract diagnostics (uses fit_df_3 + sim_df)
    recovery_r, n_divergences, mean_rhat = _extract_diagnostics(
        sim_df, fit_df_3
    )

    # Step 6: Fit 2-level model (for BMS path)
    fit_df_2, idata_2level = fit_batch(
        sim_df,
        "hgf_2level",
        return_idata=True,
        random_seed=child_seed + 1,
        cores=1,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        sampler=sampler,
    )

    # Step 7: Compute BMS power (incremental WAIC — deletes idata_3level)
    bms_xp, bms_correct = _compute_bms_power(
        sim_df, fit_df_3, fit_df_2, idata_3level, idata_2level
    )
    # idata_3level was deleted inside _compute_bms_power.
    # Delete idata_2level here to release memory.
    del idata_2level

    # Step 8: Build 3 result dicts (one per contrast type)
    results: list[dict] = []
    for contrast in contrast_results:
        results.append(
            {
                "sweep_type": contrast["sweep_type"],
                "effect_size": effect_size_delta,
                "n_per_group": n_per_group,
                "trial_count": cfg.task.n_trials_total,
                "iteration": iteration,
                "parameter": "omega_2",
                "bf_value": contrast["bf_value"],
                "bf_exceeds": contrast["bf_exceeds"],
                "bms_xp": bms_xp,
                "bms_correct": bms_correct,
                "recovery_r": recovery_r,
                "n_divergences": n_divergences,
                "mean_rhat": mean_rhat,
            }
        )

    return results


# ---------------------------------------------------------------------------
# SBF WAIC helpers (compute once, subsample many)
# ---------------------------------------------------------------------------


def _compute_waic_table(
    sim_df: pd.DataFrame,
    idata_3level: dict[tuple, object],
    idata_2level: dict[tuple, object],
) -> pd.DataFrame:
    """Compute per-(participant, model) WAIC values for all participants.

    Unlike :func:`_compute_bms_power`, this function does **not** delete any
    idata dicts — the caller retains ownership for subsampling across N levels.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level simulation DataFrame.
    idata_3level : dict[tuple, object]
        Mapping ``(participant_id, group, session) -> InferenceData``
        for the 3-level model.
    idata_2level : dict[tuple, object]
        Mapping ``(participant_id, group, session) -> InferenceData``
        for the 2-level model.

    Returns
    -------
    pandas.DataFrame
        Columns: ``participant_id``, ``model``, ``elpd_waic``.  WAIC is
        averaged across sessions per (participant_id, model).
    """
    keys = (
        sim_df[["participant_id", "group", "session"]]
        .drop_duplicates()
        .values.tolist()
    )

    waic_rows: list[dict] = []

    for model_name, idata_dict in [
        ("hgf_3level", idata_3level),
        ("hgf_2level", idata_2level),
    ]:
        for pid, grp, sess in keys:
            idata = idata_dict.get((pid, grp, sess))
            if idata is None:
                continue

            mask = (
                (sim_df["participant_id"] == pid)
                & (sim_df["group"] == grp)
                & (sim_df["session"] == sess)
            )
            subset = sim_df.loc[mask].sort_values("trial")
            input_arr, obs_arr, choices_arr = build_arrays_from_sim(subset)

            try:
                elpd = compute_subject_waic(
                    input_arr, obs_arr, choices_arr, idata, model_name
                )
                waic_rows.append(
                    {
                        "participant_id": pid,
                        "model": model_name,
                        "elpd_waic": elpd,
                    }
                )
            except Exception:
                log.exception(
                    "WAIC failed for %s (%s/%s) model=%s",
                    pid, grp, sess, model_name,
                )

    if not waic_rows:
        return pd.DataFrame(
            columns=["participant_id", "model", "elpd_waic"]
        )

    waic_df = pd.DataFrame(waic_rows)

    # Average across sessions per (participant_id, model)
    avg = (
        waic_df.groupby(["participant_id", "model"])["elpd_waic"]
        .mean()
        .reset_index()
    )
    return avg


def _bms_from_waic_table(
    waic_avg: pd.DataFrame,
    participant_ids: list,
) -> tuple[float, bool]:
    """Run BMS on a subset of participants from a pre-computed WAIC table.

    Parameters
    ----------
    waic_avg : pandas.DataFrame
        Session-averaged WAIC table with columns ``participant_id``,
        ``model``, ``elpd_waic`` (output of :func:`_compute_waic_table`).
    participant_ids : list
        Participant IDs to include in the BMS.

    Returns
    -------
    tuple[float, bool]
        ``(xp_3level, bms_correct)`` where ``bms_correct`` is True when
        ``xp_3level > 0.75``.
    """
    sub = waic_avg[waic_avg["participant_id"].isin(participant_ids)]

    model_names = ["hgf_2level", "hgf_3level"]
    pivot = sub.pivot(
        index="participant_id", columns="model", values="elpd_waic"
    )
    missing_models = [m for m in model_names if m not in pivot.columns]
    if missing_models:
        log.warning("BMS: models %s missing from WAIC results", missing_models)
        return 0.5, False

    pivot = pivot[model_names].dropna()
    if len(pivot) < 3:
        return 0.5, False

    matrix = pivot.to_numpy(dtype=float)
    bms_result = run_group_bms(matrix, model_names)

    # Index 1 = hgf_3level (model_names order is [2level, 3level])
    xp_3level = float(bms_result["xp"][1])
    return xp_3level, xp_3level > 0.75


# ---------------------------------------------------------------------------
# Batched InferenceData conversion helpers (v1.2 path)
# ---------------------------------------------------------------------------


def _idata_to_fit_df(
    idata: object,
    model_name: str,
) -> pd.DataFrame:
    """Convert batched ``InferenceData`` to the legacy ``fit_batch`` schema.

    Extracts per-participant posterior statistics from a joint
    ``arviz.InferenceData`` (produced by :func:`~prl_hgf.fitting.hierarchical\
.fit_batch_hierarchical`) and returns a DataFrame conforming to the schema
    produced by the legacy :func:`~prl_hgf.fitting.batch.fit_batch`.

    Parameters
    ----------
    idata : arviz.InferenceData
        Joint posterior for all participants.  Must have a ``participant``
        dimension on every parameter variable, plus ``participant_group`` and
        ``participant_session`` coordinates.
    model_name : str
        Model label to populate the ``model`` column (e.g.
        ``"hgf_3level"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``participant_id``, ``group``, ``session``, ``model``,
        ``parameter``, ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
        ``r_hat``, ``ess``, ``flagged``.  One row per
        (participant, parameter) combination.
    """
    import arviz as az  # deferred — heavy import

    posterior = idata.posterior  # type: ignore[union-attr]

    # Participant ordering from coords is the ground truth
    participant_ids = list(posterior.coords["participant"].values)
    participant_groups = list(posterior.coords["participant_group"].values)
    participant_sessions = list(posterior.coords["participant_session"].values)

    # Parameter variables: all data_vars in the posterior
    coord_names = set(posterior.coords)
    param_names = [
        v for v in posterior.data_vars
        if v not in coord_names
    ]

    rows: list[dict] = []
    for param in param_names:
        for i, (pid, grp, sess) in enumerate(
            zip(
                participant_ids,
                participant_groups,
                participant_sessions,
                strict=True,
            )
        ):
            da = posterior[param].isel(participant=i)
            # Flatten to (chain * draw,) then compute stats
            flat = da.values.flatten()
            mean_val = float(np.mean(flat))
            sd_val = float(np.std(flat, ddof=1))

            # az.rhat / az.ess expect DataArrays with (chain, draw) dims
            rhat_val = float(az.rhat(da).item())
            ess_val = float(az.ess(da).item())

            # HDI via az.hdi — returns Dataset with hdi_prob-specific keys
            hdi_result = az.hdi(da, hdi_prob=0.94)
            # hdi returns Dataset; variable name is param
            hdi_lower = float(
                hdi_result[param].values[0]
                if hasattr(hdi_result, "data_vars")
                else hdi_result.values[0]
            )
            hdi_upper = float(
                hdi_result[param].values[1]
                if hasattr(hdi_result, "data_vars")
                else hdi_result.values[1]
            )

            flagged = rhat_val > 1.05 or ess_val < 400

            rows.append(
                {
                    "participant_id": pid,
                    "group": grp,
                    "session": sess,
                    "model": model_name,
                    "parameter": param,
                    "mean": mean_val,
                    "sd": sd_val,
                    "hdi_3%": hdi_lower,
                    "hdi_97%": hdi_upper,
                    "r_hat": rhat_val,
                    "ess": ess_val,
                    "flagged": flagged,
                }
            )

    return pd.DataFrame(rows)


def _split_idata(
    joint_idata: object,
    participant_idx: int,
) -> object:
    """Slice a joint ``InferenceData`` to a single participant.

    Parameters
    ----------
    joint_idata : arviz.InferenceData
        Batched posterior with a ``participant`` dimension.
    participant_idx : int
        Zero-based position along the ``participant`` dimension to extract.

    Returns
    -------
    arviz.InferenceData
        Single-participant ``InferenceData`` whose posterior has no
        ``participant`` dimension.  Compatible with
        :func:`~prl_hgf.analysis.bms.compute_subject_waic`.
    """
    import arviz as az  # deferred — heavy import

    posterior_slice = joint_idata.posterior.isel(  # type: ignore[union-attr]
        participant=participant_idx
    )
    return az.InferenceData(posterior=posterior_slice)


def _build_idata_dict(
    joint_idata: object,
    participant_ids: list[str],
    participant_groups: list[str],
    participant_sessions: list[str],
) -> dict[tuple, object]:
    """Build a per-participant ``InferenceData`` mapping from a joint object.

    Converts the joint ``InferenceData`` returned by
    :func:`~prl_hgf.fitting.hierarchical.fit_batch_hierarchical` into the
    ``dict[tuple, InferenceData]`` format expected by
    :func:`_compute_waic_table` and :func:`_compute_bms_power`.

    Parameters
    ----------
    joint_idata : arviz.InferenceData
        Batched posterior with a ``participant`` dimension.
    participant_ids : list[str]
        Participant ID for each position in the ``participant`` dimension.
    participant_groups : list[str]
        Group label for each participant.
    participant_sessions : list[str]
        Session label for each participant.

    Returns
    -------
    dict[tuple, arviz.InferenceData]
        Mapping ``(participant_id, group, session) -> single-participant
        InferenceData``.
    """
    result: dict[tuple, object] = {}
    for i, (pid, grp, sess) in enumerate(
        zip(
            participant_ids,
            participant_groups,
            participant_sessions,
            strict=True,
        )
    ):
        result[(pid, grp, sess)] = _split_idata(joint_idata, i)
    return result


# ---------------------------------------------------------------------------
# SBF iteration: simulate at max N, fit once, subsample at each N
# ---------------------------------------------------------------------------


def run_sbf_iteration(
    base_config: object,
    effect_size_delta: float,
    iteration: int,
    child_seed: int,
    n_per_group_grid: list[int],
    power_config: PowerConfig,
    n_chains: int = 2,
    n_draws: int = 500,
    n_tune: int = 500,
    sampler: str = "pymc",
    use_legacy: bool = False,
) -> list[dict]:
    """Run one SBF iteration: simulate at max N, fit once, subsample at each N.

    This is 3.8x more efficient than the fixed-grid approach because MCMC
    fitting is performed only once at ``max(n_per_group_grid)`` and then
    posteriors are subsampled for each smaller N level.

    Two fitting backends are available:

    - ``use_legacy=False`` (default): uses
      :func:`~prl_hgf.fitting.hierarchical.fit_batch_hierarchical`, the v1.2
      batched JAX path.  Fits all participants in a single NUTS call.
    - ``use_legacy=True``: uses the original v1.1
      :func:`~prl_hgf.fitting.batch.fit_batch` sequential path.  Preserved
      for reproducibility and debugging (VALID-05).

    Steps
    -----
    1. Simulate ``max(n_per_group_grid)`` participants per group (both
       groups, 3 sessions).
    2. Fit 3-level model to ALL participants.
    3. Fit 2-level model to ALL participants.
    4. Compute WAIC table once for all participants (both models).
    5. For each N in ``sorted(n_per_group_grid)``:

       a. Subsample: take first N participant_ids per group from sim_df.
       b. Compute BF contrasts on subsampled fit_df_3.
       c. Compute BMS power from pre-computed WAIC table.
       d. Extract diagnostics (recovery_r, mean_rhat) from subsampled
          fit_df_3.
       e. Append result rows.

    6. Return all result rows (3 contrast types x len(n_per_group_grid)
       N-levels).

    Parameters
    ----------
    base_config : AnalysisConfig
        Base analysis configuration loaded from YAML.
    effect_size_delta : float
        Interaction effect size (psilocybin minus placebo) for omega_2.
    iteration : int
        Iteration index within this grid cell.
    child_seed : int
        RNG seed for this specific iteration.
    n_per_group_grid : list[int]
        Sample sizes per group to evaluate via subsampling.
    power_config : PowerConfig
        Power analysis grid configuration (provides ``bf_threshold``).
    n_chains : int, optional
        Number of MCMC chains.  Default ``2``.
    n_draws : int, optional
        Posterior draws per chain.  Default ``500``.
    n_tune : int, optional
        Tuning steps per chain.  Default ``500``.
    sampler : str, optional
        MCMC backend: ``"pymc"`` (default) or ``"numpyro"``.
    use_legacy : bool, optional
        If ``True``, use the v1.1 sequential :func:`~prl_hgf.fitting.batch\
.fit_batch` path.  If ``False`` (default), use the v1.2 batched
        :func:`~prl_hgf.fitting.hierarchical.fit_batch_hierarchical` path.

    Returns
    -------
    list[dict]
        ``3 * len(n_per_group_grid)`` dicts conforming to
        :data:`~prl_hgf.power.schema.POWER_SCHEMA` (13 columns each).
    """
    max_n = max(n_per_group_grid)

    # Step 1: Build frozen config at max N
    cfg = make_power_config(
        base_config, max_n, effect_size_delta, child_seed
    )

    # Step 2: Simulate at max N (same for both paths — simulate_batch already
    # uses JAX vmap internally since Phase 13)
    sim_df = simulate_batch(cfg)

    if use_legacy:
        # ------------------------------------------------------------------
        # Legacy v1.1 path: sequential per-participant MCMC
        # ------------------------------------------------------------------
        # Step 3 (legacy): Fit 3-level model to ALL participants
        fit_df_3, idata_3level = fit_batch(
            sim_df,
            "hgf_3level",
            return_idata=True,
            random_seed=child_seed,
            cores=1,
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            sampler=sampler,
        )

        # Step 4 (legacy): Fit 2-level model to ALL participants
        fit_df_2, idata_2level = fit_batch(
            sim_df,
            "hgf_2level",
            return_idata=True,
            random_seed=child_seed + 1,
            cores=1,
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            sampler=sampler,
        )
    else:
        # ------------------------------------------------------------------
        # Batched v1.2 path: one joint NUTS call for all participants
        # ------------------------------------------------------------------
        from prl_hgf.fitting.hierarchical import (  # noqa: PLC0415
            fit_batch_hierarchical,
        )

        # Step 3 (batched): Fit 3-level model — single NUTS call
        idata_3 = fit_batch_hierarchical(
            sim_df,
            "hgf_3level",
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=0.9,
            random_seed=child_seed,
            sampler=sampler,
            progressbar=False,
        )

        # Step 4 (batched): Fit 2-level model — single NUTS call
        idata_2 = fit_batch_hierarchical(
            sim_df,
            "hgf_2level",
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=0.9,
            random_seed=child_seed + 1,
            sampler=sampler,
            progressbar=False,
        )

        # Extract participant metadata from coords (ground truth ordering)
        pids = idata_3.posterior.coords["participant"].values.tolist()
        pgrps = idata_3.posterior.coords["participant_group"].values.tolist()
        psess = idata_3.posterior.coords["participant_session"].values.tolist()

        # Convert to legacy DataFrame schema for BF contrasts + diagnostics.
        # Only fit_df_3 is used in the subsampling loop (BF contrasts and
        # recovery diagnostics are computed on the 3-level model).
        fit_df_3 = _idata_to_fit_df(idata_3, "hgf_3level")

        # Build per-participant idata dicts for WAIC computation
        idata_3level = _build_idata_dict(idata_3, pids, pgrps, psess)
        idata_2level = _build_idata_dict(idata_2, pids, pgrps, psess)

    # Step 5: Compute WAIC table once (does NOT delete idata)
    waic_table = _compute_waic_table(sim_df, idata_3level, idata_2level)

    # Build sorted participant_id lists per group
    pid_by_group: dict[str, list] = {}
    for grp in sim_df["group"].unique():
        pids = sorted(
            sim_df.loc[sim_df["group"] == grp, "participant_id"].unique()
        )
        pid_by_group[grp] = pids

    # Step 6: Subsample at each N level
    results: list[dict] = []
    trial_count = cfg.task.n_trials_total

    for n_per_group in sorted(n_per_group_grid):
        # Select first N participants per group
        selected_pids: list = []
        for _grp, pids in pid_by_group.items():
            selected_pids.extend(pids[:n_per_group])
        selected_set = set(selected_pids)

        # Filter fit_df and sim_df to selected participants
        sub_fit_3 = fit_df_3[
            fit_df_3["participant_id"].isin(selected_set)
        ]
        sub_sim = sim_df[sim_df["participant_id"].isin(selected_set)]

        # (a) BF contrasts on subsampled fit_df_3
        contrast_results = compute_all_contrasts(
            sub_fit_3,
            parameter="omega_2",
            bf_threshold=power_config.bf_threshold,
        )

        # (b) BMS from pre-computed WAIC table
        bms_xp, bms_correct = _bms_from_waic_table(
            waic_table, list(selected_set)
        )

        # (c) Diagnostics from subsampled data
        recovery_r, n_divergences, mean_rhat = _extract_diagnostics(
            sub_sim, sub_fit_3
        )

        # (d) Build result rows
        for contrast in contrast_results:
            results.append(
                {
                    "sweep_type": contrast["sweep_type"],
                    "effect_size": effect_size_delta,
                    "n_per_group": n_per_group,
                    "trial_count": trial_count,
                    "iteration": iteration,
                    "parameter": "omega_2",
                    "bf_value": contrast["bf_value"],
                    "bf_exceeds": contrast["bf_exceeds"],
                    "bms_xp": bms_xp,
                    "bms_correct": bms_correct,
                    "recovery_r": recovery_r,
                    "n_divergences": n_divergences,
                    "mean_rhat": mean_rhat,
                }
            )

    # Clean up idata to free memory
    del idata_3level
    del idata_2level

    return results
