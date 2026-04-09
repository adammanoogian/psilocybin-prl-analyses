"""Recovery precheck gate for the BFDA power analysis pipeline.

Implements PRE-01 (50-participant recovery run), PRE-02 (confound matrix),
PRE-03 (eligibility table with reasons), PRE-04/PRE-05 (trial count sweep),
and PRE-06 (MCMC convergence gating).

Purpose: gate which HGF parameters advance to the expensive Phase 10 power
sweep.  Running the precheck on 50 baseline-only participants (~1/3 the full
study cost) establishes recoverability before committing cluster resources.

The trial count sweep (:func:`run_trial_sweep`) identifies the minimum trial
count where all power-eligible parameters achieve r >= 0.7, using reduced
MCMC settings (2 chains, 500 draws, 500 tune) to manage compute cost.

Notes
-----
- :func:`make_trial_config` scales phase ``n_trials`` proportionally without
  touching the transfer phase ``n_trials``.
- :func:`run_recovery_precheck` filters to ``session == "baseline"`` before
  fitting to avoid 3x compute cost.
- ``omega_3`` is always labelled ``"exploratory -- upper bound"`` in
  :func:`build_eligibility_table` regardless of its actual r value. This is a
  locked project decision (see STATE.md / ROADMAP.md).
- :func:`find_minimum_trial_count` excludes ``omega_3`` from the "all must
  pass" requirement by default (exploratory parameter).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

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
    "SweepPoint",
    "SurfacePoint",
    "make_trial_config",
    "make_sets_config",
    "run_recovery_precheck",
    "build_eligibility_table",
    "run_trial_sweep",
    "run_recoverability_surface",
    "plot_trial_sweep",
    "find_minimum_trial_count",
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
# make_sets_config
# ---------------------------------------------------------------------------


def make_sets_config(
    base: AnalysisConfig,
    n_sets: int,
) -> AnalysisConfig:
    """Return a frozen AnalysisConfig with overridden n_sets.

    Unlike :func:`make_trial_config` which scales per-phase trial counts
    proportionally, this preserves the full phase structure (30 trials per
    phase, 20 transfer) and varies only the number of complete blocks.
    Each set adds 2 reversals (volatile phases), which is the primary
    driver of HGF parameter recovery.

    Parameters
    ----------
    base : AnalysisConfig
        The baseline frozen config.
    n_sets : int
        Number of sets per session (must be >= 1).

    Returns
    -------
    AnalysisConfig
        New config with ``task.n_sets`` overridden.
    """
    new_task = dataclasses.replace(base.task, n_sets=n_sets)
    return dataclasses.replace(base, task=new_task)


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
    n_flagged = int(fit_df.groupby("participant_id")["flagged"].any().sum())
    print(
        f"PRE-06: {n_flagged}/{n_total} participants excluded (R-hat>1.05 or ESS<400)"
    )

    # --- Step 6: build recovery DataFrame
    recovery_df = build_recovery_df(sim_df_pre, fit_df, exclude_flagged=True, min_n=30)

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
    >>> metrics = pd.DataFrame(
    ...     {
    ...         "parameter": ["omega_2", "omega_3"],
    ...         "r": [0.85, 0.90],
    ...         "passes_threshold": [True, True],
    ...     }
    ... )
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
            reason = f"r={r_val:.2f} (actual). " + _OMEGA3_RATIONALE
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


# ---------------------------------------------------------------------------
# SweepPoint dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """Container for one trial-count grid point in a recovery sweep.

    Parameters
    ----------
    trial_count : int
        Actual total trial count achieved after integer rounding (may differ
        slightly from the requested target).
    metrics_df : pandas.DataFrame
        Per-parameter recovery metrics from
        :func:`~prl_hgf.analysis.recovery.compute_recovery_metrics`.
        Columns: ``parameter``, ``r``, ``p``, ``bias``, ``rmse``, ``n``,
        ``passes_threshold``.
    n_flagged : int
        Participants excluded due to R-hat > 1.05 or ESS < 400.
    n_total : int
        Total participants fitted at this grid point (before exclusion).
    """

    trial_count: int
    metrics_df: pd.DataFrame
    n_flagged: int
    n_total: int


# ---------------------------------------------------------------------------
# run_trial_sweep
# ---------------------------------------------------------------------------

_DEFAULT_TRIAL_GRID = [150, 200, 250, 300, 420]


def run_trial_sweep(
    config: AnalysisConfig,
    trial_grid: list[int] | None = None,
    n_per_group: int = 30,
    model_name: str = "hgf_3level",
    seed: int = 42,
    output_dir: Path | None = None,
) -> list[SweepPoint]:
    """Run a trial-count sweep (PRE-04/PRE-05) with reduced MCMC settings.

    For each target trial count in ``trial_grid``, simulates ``n_per_group``
    participants per group, fits only the baseline session, and computes
    parameter recovery metrics.  Each grid point uses an independent RNG seed
    (``seed + idx``) so participants are fresh at every point.

    Reduced MCMC settings (2 chains, 500 draws, 500 tune) are used to keep
    compute cost manageable.  These are adequate for recovery estimation but
    not inference-grade.

    Parameters
    ----------
    config : AnalysisConfig
        Base analysis configuration loaded via
        :func:`~prl_hgf.env.task_config.load_config`.
    trial_grid : list[int] or None, optional
        Total trial counts to evaluate. Default ``[150, 200, 250, 300, 420]``.
    n_per_group : int, optional
        Participants per group at each grid point. Default 30.
    model_name : str, optional
        HGF model variant to fit. Default ``"hgf_3level"``.
    seed : int, optional
        Base RNG seed. Grid point ``idx`` uses ``seed + idx`` to ensure
        independent participants across grid points. Default 42.
    output_dir : Path or None, optional
        If provided, saves ``trial_sweep_results.csv`` (long-form) here.
        Directory is created if it does not exist.

    Returns
    -------
    list[SweepPoint]
        One :class:`SweepPoint` per grid point, in ascending trial-count order.

    Notes
    -----
    The stable/volatile ratio is preserved at each grid point because
    :func:`make_trial_config` scales all phases proportionally.

    ``min_n=0`` is passed to :func:`~prl_hgf.analysis.recovery.build_recovery_df`
    because small trial counts may lose many participants to convergence
    failures; downstream callers can apply their own minimum-n filter.
    """
    if trial_grid is None:
        trial_grid = _DEFAULT_TRIAL_GRID

    results: list[SweepPoint] = []

    for idx, target_trials in enumerate(trial_grid):
        point_seed = seed + idx

        # Scale trial count
        trial_cfg = make_trial_config(config, target_total_trials=target_trials)
        actual_total = trial_cfg.task.n_trials_total

        # Build power config — null effect, fresh participants per grid point
        power_cfg = make_power_config(
            trial_cfg,
            n_per_group=n_per_group,
            effect_size_delta=0.0,
            master_seed=point_seed,
        )

        # Simulate
        sim_df = simulate_batch(power_cfg)

        # Baseline-only filter
        sim_df_pre = sim_df[sim_df["session"] == "baseline"].copy()

        # Fit with reduced MCMC settings (PRE-05)
        print(
            f"[{idx + 1}/{len(trial_grid)}] trials={actual_total}: "
            f"fitting {len(sim_df_pre['participant_id'].unique())} participants..."
        )
        fit_df = fit_batch(
            sim_df_pre,
            model_name=model_name,
            cores=1,
            n_chains=2,
            n_draws=500,
            n_tune=500,
        )

        # Convergence exclusion count
        n_total = int(fit_df["participant_id"].nunique())
        n_flagged = int(fit_df.groupby("participant_id")["flagged"].any().sum())
        print(
            f"[{idx + 1}/{len(trial_grid)}] trials={actual_total}: "
            f"{n_flagged} flagged, computing recovery..."
        )

        # Build recovery DataFrame (min_n=0 for small trial counts)
        recovery_df = build_recovery_df(
            sim_df_pre, fit_df, exclude_flagged=True, min_n=0
        )

        # Compute metrics
        metrics_df = compute_recovery_metrics(recovery_df)

        results.append(
            SweepPoint(
                trial_count=actual_total,
                metrics_df=metrics_df,
                n_flagged=n_flagged,
                n_total=n_total,
            )
        )

    # Save long-form CSV if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for pt in results:
            for _, mrow in pt.metrics_df.iterrows():
                rows.append(
                    {
                        "trial_count": pt.trial_count,
                        "parameter": mrow["parameter"],
                        "r": mrow["r"],
                        "p": mrow["p"],
                        "bias": mrow["bias"],
                        "rmse": mrow["rmse"],
                        "n": mrow["n"],
                        "passes_threshold": mrow["passes_threshold"],
                    }
                )
        sweep_csv = output_dir / "trial_sweep_results.csv"
        pd.DataFrame(rows).to_csv(sweep_csv, index=False)
        log.info("Saved: %s", sweep_csv)

    return results


# ---------------------------------------------------------------------------
# plot_trial_sweep
# ---------------------------------------------------------------------------

# Parameter-specific styles for the sweep plot.
_PARAM_COLORS = {
    "omega_2": "#1f77b4",
    "omega_3": "#ff7f0e",
    "kappa": "#2ca02c",
    "beta": "#9467bd",
    "zeta": "#d62728",
}
_PARAM_MARKERS = {
    "omega_2": "o",
    "omega_3": "s",
    "kappa": "^",
    "beta": "D",
    "zeta": "v",
}


def plot_trial_sweep(
    sweep_results: list[SweepPoint],
    r_threshold: float = 0.7,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot parameter recovery r vs total trial count (VIZ-01).

    Produces a single figure with one line per parameter.  ``omega_3`` is
    rendered with a dashed/dotted line style and labelled "(exploratory)".
    A horizontal dashed red reference line marks ``r_threshold``.

    Parameters
    ----------
    sweep_results : list[SweepPoint]
        Output of :func:`run_trial_sweep`, one point per trial-count grid
        value.
    r_threshold : float, optional
        Reference line position on the Y axis. Default 0.7.
    save_path : Path or None, optional
        If provided, saves the figure as PNG at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    if not sweep_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Parameter recovery vs trial count (no data)")
        return fig

    # Collect all parameter names across all grid points
    all_params: list[str] = []
    for pt in sweep_results:
        for param in pt.metrics_df["parameter"].tolist():
            if param not in all_params:
                all_params.append(param)

    trial_counts = [pt.trial_count for pt in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    for param in all_params:
        r_vals = []
        for pt in sweep_results:
            row = pt.metrics_df[pt.metrics_df["parameter"] == param]
            if len(row) > 0:
                r_vals.append(float(row.iloc[0]["r"]))
            else:
                r_vals.append(float("nan"))

        color = _PARAM_COLORS.get(param)
        marker = _PARAM_MARKERS.get(param, "o")

        if param == "omega_3":
            linestyle = "--"
            label = f"{param} (exploratory)"
        else:
            linestyle = "-"
            label = param

        ax.plot(
            trial_counts,
            r_vals,
            linestyle=linestyle,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.8,
            markersize=6,
        )

    # Reference line at r_threshold
    ax.axhline(
        y=r_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"r = {r_threshold} (threshold)",
    )

    ax.set_xlabel("Total trials per session")
    ax.set_ylabel("Pearson r (true vs recovered)")
    ax.set_title("Parameter recovery vs trial count")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# find_minimum_trial_count
# ---------------------------------------------------------------------------


def find_minimum_trial_count(
    sweep_results: list[SweepPoint],
    r_threshold: float = 0.7,
    eligible_params: list[str] | None = None,
) -> int | None:
    """Return the smallest trial count where all eligible parameters pass r >= threshold.

    Examines ``sweep_results`` in ascending ``trial_count`` order.  ``omega_3``
    is excluded from the "all must pass" requirement by default because it is
    marked exploratory (locked project decision).

    Parameters
    ----------
    sweep_results : list[SweepPoint]
        Output of :func:`run_trial_sweep`.
    r_threshold : float, optional
        Minimum Pearson r required per parameter. Default 0.7.
    eligible_params : list[str] or None, optional
        Parameters to include in the "all must pass" check.  If ``None``,
        all parameters except ``omega_3`` are included.

    Returns
    -------
    int or None
        Smallest ``trial_count`` where every parameter in ``eligible_params``
        achieves ``r >= r_threshold``, or ``None`` if no grid point satisfies
        the condition.

    Examples
    --------
    >>> from prl_hgf.power.precheck import find_minimum_trial_count, SweepPoint
    >>> import pandas as pd
    >>> pts = [
    ...     SweepPoint(
    ...         150, pd.DataFrame({"parameter": ["omega_2"], "r": [0.60]}), 0, 10
    ...     ),
    ...     SweepPoint(
    ...         250, pd.DataFrame({"parameter": ["omega_2"], "r": [0.75]}), 0, 10
    ...     ),
    ... ]
    >>> find_minimum_trial_count(pts)
    250
    """
    # Sort ascending by trial count so we return the *minimum*
    sorted_points = sorted(sweep_results, key=lambda pt: pt.trial_count)

    for pt in sorted_points:
        params_in_point = pt.metrics_df["parameter"].tolist()

        # Determine which params to check
        if eligible_params is not None:
            check_params = eligible_params
        else:
            check_params = [p for p in params_in_point if p != "omega_3"]

        if not check_params:
            # Nothing to check — treat as satisfied
            return pt.trial_count

        all_pass = True
        for param in check_params:
            row = pt.metrics_df[pt.metrics_df["parameter"] == param]
            if len(row) == 0:
                # Parameter not present at this grid point — treat as fail
                all_pass = False
                break
            if float(row.iloc[0]["r"]) < r_threshold:
                all_pass = False
                break

        if all_pass:
            return pt.trial_count

    return None


# ---------------------------------------------------------------------------
# _compute_contrast_recovery helper
# ---------------------------------------------------------------------------


def _compute_contrast_recovery(
    sim_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    parameter: str,
) -> float:
    """Compute r(true_DiD, recovered_DiD) for a parameter.

    The DiD contrast per participant:
        DiD = (param_post_dose - param_baseline) for psilocybin
            - (param_post_dose - param_baseline) for placebo

    For individual-level recovery, we compute DiD per participant
    (within-group) and correlate true vs recovered across participants.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulation DataFrame with true_* columns.
    fit_df : pandas.DataFrame
        Fit results with mean column for recovered parameters.
    parameter : str
        Parameter name (e.g., "omega_2").

    Returns
    -------
    float
        Pearson r between true and recovered DiD contrasts.
        NaN if insufficient data.
    """
    true_col = f"true_{parameter}"

    # Get true parameter per (participant, session) from sim_df
    true_params = (
        sim_df.groupby(["participant_id", "group", "session"])[true_col]
        .first()
        .reset_index()
    )

    # Get recovered parameter per (participant, session) from fit_df
    rec_params = fit_df[fit_df["parameter"] == parameter][
        ["participant_id", "group", "session", "mean"]
    ].copy()
    rec_params = rec_params.rename(columns={"mean": "recovered"})

    # Merge
    merged = true_params.merge(
        rec_params, on=["participant_id", "group", "session"], how="inner"
    )

    if len(merged) == 0:
        return float("nan")

    # Compute per-participant delta: post_dose - baseline
    deltas = []
    for pid in merged["participant_id"].unique():
        p = merged[merged["participant_id"] == pid]
        base = p[p["session"] == "baseline"]
        post = p[p["session"] == "post_dose"]
        if len(base) == 0 or len(post) == 0:
            continue
        true_delta = float(post[true_col].iloc[0]) - float(base[true_col].iloc[0])
        rec_delta = float(post["recovered"].iloc[0]) - float(base["recovered"].iloc[0])
        deltas.append({"true_delta": true_delta, "rec_delta": rec_delta})

    if len(deltas) < 5:
        return float("nan")

    delta_df = pd.DataFrame(deltas)
    r, _ = stats.pearsonr(delta_df["true_delta"], delta_df["rec_delta"])
    return float(r)


# ---------------------------------------------------------------------------
# SurfacePoint dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfacePoint:
    """One cell in the joint recoverability surface.

    Parameters
    ----------
    n_sets : int
        Number of task sets per session.
    trial_count : int
        Actual total trials (``n_sets * n_trials_per_set``).
    n_per_group : int
        Sample size per group.
    effect_size : float
        Effect size delta applied to psilocybin group.
    parameter : str
        HGF parameter name.
    r_individual : float
        Mean r(true_param, recovered_param) across iterations.
    r_contrast : float
        Mean r(true_DiD, recovered_DiD) across iterations. NaN for d=0.
    bf_rate : float
        Proportion of iterations where BF > threshold.
    n_iterations : int
        Number of iterations completed for this cell.
    n_flagged : int
        Mean flagged participants across iterations.
    n_total : int
        Mean total participants across iterations.
    """

    n_sets: int
    trial_count: int
    n_per_group: int
    effect_size: float
    parameter: str
    r_individual: float
    r_contrast: float
    bf_rate: float
    n_iterations: int
    n_flagged: int
    n_total: int


# ---------------------------------------------------------------------------
# run_recoverability_surface
# ---------------------------------------------------------------------------


def run_recoverability_surface(
    config: AnalysisConfig,
    sets_grid: list[int] | None = None,
    n_per_group_grid: list[int] | None = None,
    effect_size_grid: list[float] | None = None,
    n_iterations: int = 30,
    model_name: str = "hgf_3level",
    seed: int = 42,
    bf_threshold: float = 10.0,
    output_dir: Path | None = None,
) -> list[SurfacePoint]:
    """Run joint (N x sets x d) recoverability surface.

    For each (n_sets, d) combination, simulates at max(n_per_group_grid)
    participants per group, fits ALL sessions (not just baseline), then
    subsamples at each N level. Computes individual parameter recovery
    and DiD contrast recovery.

    Uses reduced MCMC (2 chains, 500 draws, 500 tune) and fits only
    the 3-level model.

    Parameters
    ----------
    config : AnalysisConfig
        Base analysis configuration.
    sets_grid : list[int] or None
        Number of sets to sweep. Default ``[1, 2, 3, 4, 5]``.
    n_per_group_grid : list[int] or None
        Sample sizes to evaluate via subsampling.
        Default ``[10, 15, 20, 25, 30, 40, 50]``.
    effect_size_grid : list[float] or None
        Effect sizes. Default ``[0.0, 0.5]``.
    n_iterations : int
        Iterations per (sets, d) cell. Default 30.
    model_name : str
        HGF model variant. Default ``"hgf_3level"``.
    seed : int
        Base RNG seed.
    bf_threshold : float
        BF threshold for power/false positive rate. Default 10.0.
    output_dir : Path or None
        If provided, saves ``recoverability_surface.csv`` here.

    Returns
    -------
    list[SurfacePoint]
        One entry per (n_sets, n_per_group, effect_size, parameter)
        combination.
    """
    from prl_hgf.power.contrasts import compute_all_contrasts

    if sets_grid is None:
        sets_grid = [1, 2, 3, 4, 5]
    if n_per_group_grid is None:
        n_per_group_grid = [10, 15, 20, 25, 30, 40, 50]
    if effect_size_grid is None:
        effect_size_grid = [0.0, 0.5]

    max_n = max(n_per_group_grid)
    all_points: list[SurfacePoint] = []

    for sets_idx, n_sets in enumerate(sets_grid):
        sets_cfg = make_sets_config(config, n_sets)
        actual_trials = sets_cfg.task.n_trials_total

        for d_idx, d in enumerate(effect_size_grid):
            # Storage keyed by n_per_group, accumulates across iterations
            iter_results: dict[int, list[dict]] = {n: [] for n in n_per_group_grid}

            for it in range(n_iterations):
                it_seed = seed + sets_idx * 1000 + d_idx * 100 + it

                # Build config: sets override + effect size + max N
                power_cfg = make_power_config(sets_cfg, max_n, d, it_seed)

                # Simulate all sessions
                sim_df = simulate_batch(power_cfg)

                # Fit ALL sessions with reduced MCMC
                fit_df = fit_batch(
                    sim_df,
                    model_name=model_name,
                    cores=1,
                    n_chains=2,
                    n_draws=500,
                    n_tune=500,
                )

                # Get sorted participant IDs per group
                pid_by_group: dict[str, list] = {}
                for grp in sim_df["group"].unique():
                    pids = sorted(
                        sim_df[sim_df["group"] == grp]["participant_id"].unique()
                    )
                    pid_by_group[grp] = pids

                # For each N, subsample and compute metrics
                for n in sorted(n_per_group_grid):
                    # Take first N per group
                    selected_pids: list = []
                    for _grp, pids in pid_by_group.items():
                        selected_pids.extend(pids[:n])

                    sub_sim = sim_df[sim_df["participant_id"].isin(selected_pids)]
                    sub_fit = fit_df[fit_df["participant_id"].isin(selected_pids)]

                    # Compute individual recovery (r per parameter)
                    recovery_df = build_recovery_df(
                        sub_sim,
                        sub_fit,
                        exclude_flagged=True,
                        min_n=0,
                    )
                    if len(recovery_df) > 0:
                        metrics = compute_recovery_metrics(recovery_df)
                    else:
                        metrics = pd.DataFrame(
                            columns=[
                                "parameter",
                                "r",
                                "passes_threshold",
                            ]
                        )

                    # Compute DiD contrast recovery (meaningful when d>0)
                    contrast_r = float("nan")
                    if d > 0 and len(sub_fit) > 0:
                        contrast_r = _compute_contrast_recovery(
                            sub_sim, sub_fit, "omega_2"
                        )

                    # Compute BF on the interaction contrast
                    bf_exceeds = False
                    if len(sub_fit) > 0:
                        try:
                            contrasts = compute_all_contrasts(
                                sub_fit,
                                parameter="omega_2",
                                bf_threshold=bf_threshold,
                            )
                            for c in contrasts:
                                if c["sweep_type"] == "did_postdose":
                                    bf_exceeds = c["bf_exceeds"]
                                    break
                        except Exception:
                            pass

                    # Store per-iteration results
                    n_total = int(sub_fit["participant_id"].nunique())
                    n_flagged = 0
                    if "flagged" in sub_fit.columns:
                        n_flagged = int(
                            sub_fit.groupby("participant_id")["flagged"].any().sum()
                        )

                    iter_results[n].append(
                        {
                            "metrics": metrics,
                            "contrast_r": contrast_r,
                            "bf_exceeds": bf_exceeds,
                            "n_flagged": n_flagged,
                            "n_total": n_total,
                        }
                    )

                # Progress
                print(f"[sets={n_sets}, d={d:.1f}] iter {it + 1}/{n_iterations}")

            # Aggregate across iterations for each N
            for n in sorted(n_per_group_grid):
                results_at_n = iter_results[n]
                n_iter_actual = len(results_at_n)

                # Average r per parameter across iterations
                param_rs: dict[str, list[float]] = {}
                for res in results_at_n:
                    for _, row in res["metrics"].iterrows():
                        param = str(row["parameter"])
                        r_val = float(row["r"])
                        if param not in param_rs:
                            param_rs[param] = []
                        param_rs[param].append(r_val)

                # Average contrast r
                contrast_rs = [
                    res["contrast_r"]
                    for res in results_at_n
                    if not (
                        isinstance(res["contrast_r"], float)
                        and np.isnan(res["contrast_r"])
                    )
                ]
                mean_contrast_r = (
                    float(np.mean(contrast_rs)) if contrast_rs else float("nan")
                )

                # BF rate
                bf_rate = (
                    sum(1 for res in results_at_n if res["bf_exceeds"]) / n_iter_actual
                )

                # Mean flagged
                mean_flagged = int(np.mean([res["n_flagged"] for res in results_at_n]))
                mean_total = int(np.mean([res["n_total"] for res in results_at_n]))

                for param, rs in param_rs.items():
                    all_points.append(
                        SurfacePoint(
                            n_sets=n_sets,
                            trial_count=actual_trials,
                            n_per_group=n,
                            effect_size=d,
                            parameter=param,
                            r_individual=float(np.mean(rs)),
                            r_contrast=mean_contrast_r,
                            bf_rate=bf_rate,
                            n_iterations=n_iter_actual,
                            n_flagged=mean_flagged,
                            n_total=mean_total,
                        )
                    )

    # Save CSV
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [dataclasses.asdict(pt) for pt in all_points]
        csv_path = output_dir / "recoverability_surface.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return all_points
