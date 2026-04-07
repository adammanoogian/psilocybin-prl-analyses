"""Recovery precheck gate for the BFDA power analysis pipeline.

Implements PRE-01 (50-participant recovery run), PRE-02 (confound matrix),
PRE-03 (eligibility table with reasons), and PRE-06 (MCMC convergence gating).

Purpose: gate which HGF parameters advance to the expensive Phase 10 power
sweep.  Running the precheck on 50 baseline-only participants (~1/3 the full
study cost) establishes recoverability before committing cluster resources.

Notes
-----
- :func:`make_trial_config` scales phase ``n_trials`` proportionally without
  touching the transfer phase ``n_trials``.
- :func:`run_recovery_precheck` filters to ``session == "baseline"`` before
  fitting to avoid 3x compute cost.
- ``omega_3`` is always labelled ``"exploratory -- upper bound"`` in
  :func:`build_eligibility_table` regardless of its actual r value. This is a
  locked project decision (see STATE.md / ROADMAP.md).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prl_hgf.analysis.plots import plot_correlation_matrix, plot_recovery_scatter
from prl_hgf.analysis.recovery import (
    build_recovery_df,
    compute_correlation_matrix,
    compute_recovery_metrics,
)
from prl_hgf.env.task_config import AnalysisConfig, PhaseConfig, TaskConfig
from prl_hgf.fitting.batch import fit_batch
from prl_hgf.power.config import make_power_config
from prl_hgf.simulation.batch import simulate_batch

__all__ = [
    "PrecheckResult",
    "make_trial_config",
    "run_recovery_precheck",
    "build_eligibility_table",
]

log = logging.getLogger(__name__)

# Locked project decision: omega_3 always exploratory regardless of r value.
_OMEGA3_STATUS = "exploratory -- upper bound"
_OMEGA3_RATIONALE = (
    "Project decision (locked): omega_3 recovery is known to be poor with "
    "binary PRL data (literature r ~ 0.67). BFDA power estimates for omega_3 "
    "would be inflated 20-40 percentage points. Included as upper bound only."
)

# Threshold for passing recovery gate.
_R_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# PrecheckResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecheckResult:
    """Container for recovery precheck outputs.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-parameter recovery metrics from
        :func:`~prl_hgf.analysis.recovery.compute_recovery_metrics`.
        Columns: ``parameter``, ``r``, ``p``, ``bias``, ``rmse``, ``n``,
        ``passes_threshold``.
    corr_df : pandas.DataFrame
        Square inter-parameter Pearson correlation matrix from
        :func:`~prl_hgf.analysis.recovery.compute_correlation_matrix`.
    eligibility_df : pandas.DataFrame
        Eligibility table from :func:`build_eligibility_table`.
        Columns: ``parameter``, ``r``, ``status``, ``reason``.
    n_flagged : int
        Number of participants excluded due to R-hat > 1.05 or ESS < 400.
    n_total : int
        Total number of participants fitted (before convergence exclusion).
    recovery_df : pandas.DataFrame
        Wide-form recovery DataFrame from
        :func:`~prl_hgf.analysis.recovery.build_recovery_df`.
    """

    metrics_df: pd.DataFrame
    corr_df: pd.DataFrame
    eligibility_df: pd.DataFrame
    n_flagged: int
    n_total: int
    recovery_df: pd.DataFrame


# ---------------------------------------------------------------------------
# make_trial_config
# ---------------------------------------------------------------------------


def make_trial_config(
    base: AnalysisConfig,
    target_total_trials: int,
) -> AnalysisConfig:
    """Return a frozen AnalysisConfig with scaled per-phase trial counts.

    Computes a scale factor relative to the sum of the four main phase
    ``n_trials`` values and applies ``max(1, round(factor * phase.n_trials))``
    to each phase.  The transfer phase ``n_trials`` is **not** changed.

    The approach preserves the stable/volatile 1:1 ratio because all four
    phases currently have equal ``n_trials`` in the YAML config.

    This function performs no file I/O and does not mutate ``base``.

    Parameters
    ----------
    base : AnalysisConfig
        The baseline frozen config from which to derive the trial variant.
    target_total_trials : int
        Desired ``task.n_trials_total`` for one session. Because n_trials are
        integers, the actual total may differ by a few trials due to rounding.
        Must be >= ``n_sets * (n_phases + transfer_n_trials)`` so that each
        phase gets at least 1 trial.

    Returns
    -------
    AnalysisConfig
        A new frozen :class:`~prl_hgf.env.task_config.AnalysisConfig` with
        scaled phase ``n_trials``. ``base.simulation`` and ``base.fitting`` are
        carried over unchanged.

    Examples
    --------
    >>> from prl_hgf.env.task_config import load_config
    >>> from prl_hgf.power.precheck import make_trial_config
    >>> base = load_config()
    >>> variant = make_trial_config(base, target_total_trials=200)
    >>> abs(variant.task.n_trials_total - 200) <= 4
    True
    """
    task = base.task
    n_sets = task.n_sets
    transfer_n_trials = task.transfer.n_trials

    # target_per_set is the total within one set (phases + transfer)
    target_per_set = target_total_trials / n_sets

    # Sum of only the phase trials (excluding transfer)
    sum_phase_trials = sum(p.n_trials for p in task.phases)

    # Trials available for phases within one set
    phase_budget_per_set = target_per_set - transfer_n_trials

    # Scale factor relative to current phase trial total
    scale = phase_budget_per_set / sum_phase_trials

    new_phases: list[PhaseConfig] = [
        dataclasses.replace(p, n_trials=max(1, round(scale * p.n_trials)))
        for p in task.phases
    ]

    new_task: TaskConfig = dataclasses.replace(base.task, phases=new_phases)
    new_config = dataclasses.replace(base, task=new_task)

    actual_total = new_config.task.n_trials_total
    log.info(
        "make_trial_config: target=%d, actual=%d (delta=%d)",
        target_total_trials,
        actual_total,
        actual_total - target_total_trials,
    )
    return new_config


# ---------------------------------------------------------------------------
# run_recovery_precheck
# ---------------------------------------------------------------------------


def run_recovery_precheck(
    config: AnalysisConfig,
    n_participants: int = 50,
    model_name: str = "hgf_3level",
    seed: int = 42,
    output_dir: Path | None = None,
) -> PrecheckResult:
    """Run the PRE-01/PRE-02/PRE-06 recovery precheck on baseline data only.

    Simulates ``n_participants`` per group at ``effect_size_delta=0.0``, then
    fits only the ``"baseline"`` session to avoid 3x compute cost. Convergence
    diagnostics gate participants before computing recovery metrics.

    Parameters
    ----------
    config : AnalysisConfig
        Base analysis configuration loaded via
        :func:`~prl_hgf.env.task_config.load_config`.
    n_participants : int, optional
        Number of synthetic participants per group. Default 50.
    model_name : str, optional
        HGF model variant to fit. Default ``"hgf_3level"``.
    seed : int, optional
        Master RNG seed passed to :func:`~prl_hgf.power.config.make_power_config`
        and :func:`~prl_hgf.fitting.batch.fit_batch`. Default 42.
    output_dir : Path or None, optional
        If provided, CSVs and PNG plots are saved here.  Directory is created
        if it does not exist.

    Returns
    -------
    PrecheckResult
        Frozen dataclass containing metrics, correlation matrix, eligibility
        table, exclusion counts, and raw recovery DataFrame.

    Notes
    -----
    PRE-06 exclusion: participants where R-hat > 1.05 or ESS < 400 on any
    parameter are excluded before computing recovery metrics. The count is
    printed to console and recorded in the returned :class:`PrecheckResult`.
    """
    # --- Step 1: build power config with no effect (null hypothesis recovery)
    power_cfg = make_power_config(
        config,
        n_per_group=n_participants,
        effect_size_delta=0.0,
        master_seed=seed,
    )

    # --- Step 2: simulate
    log.info(
        "Simulating %d participants/group for recovery precheck...",
        n_participants,
    )
    sim_df = simulate_batch(power_cfg)

    # --- Step 3: filter to baseline only (PRE-01 cost reduction)
    sim_df_pre = sim_df[sim_df["session"] == "baseline"].copy()
    log.info(
        "Baseline-only filter: %d trial rows (%d sessions removed)",
        len(sim_df_pre),
        len(sim_df) - len(sim_df_pre),
    )

    # --- Step 4: fit baseline sessions
    log.info("Fitting baseline participants (model=%s)...", model_name)
    fit_df = fit_batch(
        sim_df_pre,
        model_name=model_name,
        cores=1,
    )

    # --- Step 5: PRE-06 convergence exclusion count
    n_total = int(fit_df["participant_id"].nunique())
    # flagged is per parameter row — a participant is flagged if ANY row is True
    n_flagged = int(
        fit_df.groupby("participant_id")["flagged"].any().sum()
    )
    print(
        f"PRE-06: {n_flagged}/{n_total} participants excluded "
        f"(R-hat>1.05 or ESS<400)"
    )

    # --- Step 6: build recovery DataFrame
    recovery_df = build_recovery_df(
        sim_df_pre, fit_df, exclude_flagged=True, min_n=30
    )

    # --- Step 7: compute metrics and correlation matrix
    metrics_df = compute_recovery_metrics(recovery_df)
    corr_df = compute_correlation_matrix(recovery_df)

    # --- Step 8: build eligibility table
    eligibility_df = build_eligibility_table(metrics_df)

    # --- Step 9: save outputs if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        metrics_path = output_dir / "recovery_metrics_precheck.csv"
        metrics_df.to_csv(metrics_path, index=False)
        log.info("Saved: %s", metrics_path)

        eligibility_path = output_dir / "power_eligible_params.csv"
        eligibility_df.to_csv(eligibility_path, index=False)
        log.info("Saved: %s", eligibility_path)

        scatter_path = output_dir / f"recovery_scatter_precheck_{model_name}.png"
        fig = plot_recovery_scatter(recovery_df, metrics_df, save_path=scatter_path)
        plt.close(fig)
        log.info("Saved: %s", scatter_path)

        corr_path = output_dir / f"correlation_matrix_precheck_{model_name}.png"
        fig = plot_correlation_matrix(corr_df, save_path=corr_path)
        plt.close(fig)
        log.info("Saved: %s", corr_path)

    return PrecheckResult(
        metrics_df=metrics_df,
        corr_df=corr_df,
        eligibility_df=eligibility_df,
        n_flagged=n_flagged,
        n_total=n_total,
        recovery_df=recovery_df,
    )


# ---------------------------------------------------------------------------
# build_eligibility_table
# ---------------------------------------------------------------------------


def build_eligibility_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build parameter eligibility table from recovery metrics.

    Classifies each parameter as power-eligible, exploratory, or excluded
    based on its Pearson r against the ``_R_THRESHOLD`` (0.7), with a locked
    exception for ``omega_3`` which is always marked exploratory regardless of
    its actual r value.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Output of :func:`~prl_hgf.analysis.recovery.compute_recovery_metrics`.
        Must contain columns ``parameter``, ``r``, and ``passes_threshold``.

    Returns
    -------
    pandas.DataFrame
        One row per parameter with columns:
        ``["parameter", "r", "status", "reason"]``.

        Status values:

        - ``"power-eligible"`` — ``|r| >= 0.7`` and not omega_3.
        - ``"exploratory -- upper bound"`` — parameter is ``omega_3``
          (locked project decision; see module-level notes).
        - ``"excluded"`` — ``|r| < 0.7`` and not omega_3.

    Examples
    --------
    >>> import pandas as pd
    >>> from prl_hgf.power.precheck import build_eligibility_table
    >>> metrics = pd.DataFrame({
    ...     "parameter": ["omega_2", "omega_3"],
    ...     "r": [0.85, 0.90],
    ...     "passes_threshold": [True, True],
    ... })
    >>> elig = build_eligibility_table(metrics)
    >>> elig.loc[elig["parameter"] == "omega_3", "status"].iloc[0]
    'exploratory -- upper bound'
    """
    rows: list[dict] = []
    for _, row in metrics_df.iterrows():
        param = str(row["parameter"])
        r_val = float(row["r"])
        passes = bool(row["passes_threshold"])

        if param == "omega_3":
            status = _OMEGA3_STATUS
            reason = (
                f"r={r_val:.2f} (actual). "
                + _OMEGA3_RATIONALE
            )
        elif passes:
            status = "power-eligible"
            reason = f"r={r_val:.2f} >= {_R_THRESHOLD}"
        else:
            status = "excluded"
            reason = f"r={r_val:.2f} < {_R_THRESHOLD} threshold"

        rows.append(
            {
                "parameter": param,
                "r": r_val,
                "status": status,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows, columns=["parameter", "r", "status", "reason"])
