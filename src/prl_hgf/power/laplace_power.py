"""Fast VB-Laplace design sweep for experiment planning.

Replaces the MCMC-based BFDA power analysis with a Laplace-based sweep
that runs locally in minutes instead of hours on a GPU cluster.  Produces
the same outputs (recovery metrics, BF, power curves) with comparable
point estimates for well-identified parameters.

Workflow
--------
For each (n_per_group, effect_size_delta, iteration):

1. Override config with ``make_power_config()``
2. Simulate cohort with group differences
3. Fit both models with ``fit_vb_laplace_prl()``
4. Compute parameter recovery, DiD Bayes Factor, and model comparison

Returns a tidy DataFrame suitable for power curve plotting.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from prl_hgf.env.task_config import load_config
from prl_hgf.fitting.fit_vb_laplace_prl import (
    compare_models_laplace,
    fit_vb_laplace_prl,
    idata_to_fit_df,
)
from prl_hgf.power.config import make_power_config
from prl_hgf.power.contrasts import compute_did_contrast, compute_jzs_bf
from prl_hgf.simulation.batch import simulate_batch

logger = logging.getLogger(__name__)


def run_design_sweep(
    n_per_group_grid: list[int] | None = None,
    effect_size_grid: list[float] | None = None,
    n_iterations: int = 20,
    target_params: list[str] | None = None,
    bf_threshold: float = 6.0,
    seed: int = 42,
    sessions_to_fit: str = "baseline",
) -> pd.DataFrame:
    """Run a fast VB-Laplace design optimisation sweep.

    Parameters
    ----------
    n_per_group_grid : list[int] or None
        Sample sizes to sweep.  Default ``[10, 20, 30, 50]``.
    effect_size_grid : list[float] or None
        Effect size deltas (omega_2 units) to sweep.  Default ``[0.3, 0.5, 0.7]``.
    n_iterations : int, default 20
        Number of simulated datasets per grid cell.
    target_params : list[str] or None
        Parameters to track recovery for.  Default ``["omega_2", "beta", "zeta"]``.
    bf_threshold : float, default 6.0
        Bayes Factor threshold for declaring evidence.
    seed : int, default 42
        Master RNG seed.
    sessions_to_fit : str, default "baseline"
        Which session(s) to fit.  ``"baseline"`` = fast single-session;
        ``"all"`` = full multi-session (slower but tests DiD contrasts).

    Returns
    -------
    pd.DataFrame
        Tidy results with columns: ``n_per_group``, ``effect_size``,
        ``iteration``, ``parameter``, ``recovery_r``, ``bias``, ``rmse``,
        ``bf_omega2``, ``bms_xp_3level``, ``n_diverged``.
    """
    if n_per_group_grid is None:
        n_per_group_grid = [10, 20, 30, 50]
    if effect_size_grid is None:
        effect_size_grid = [0.3, 0.5, 0.7]
    if target_params is None:
        target_params = ["omega_2", "beta", "zeta"]

    base_config = load_config()
    rng = np.random.default_rng(seed)
    results: list[dict] = []

    total_cells = len(n_per_group_grid) * len(effect_size_grid) * n_iterations
    cell_i = 0
    t_start = time.perf_counter()

    for n_per_group in n_per_group_grid:
        for effect_size in effect_size_grid:
            for iteration in range(n_iterations):
                cell_i += 1
                child_seed = int(rng.integers(0, 2**31))

                try:
                    row = _run_single_cell(
                        base_config=base_config,
                        n_per_group=n_per_group,
                        effect_size_delta=effect_size,
                        child_seed=child_seed,
                        target_params=target_params,
                        bf_threshold=bf_threshold,
                        sessions_to_fit=sessions_to_fit,
                    )
                    for r in row:
                        r["iteration"] = iteration
                        r["n_per_group"] = n_per_group
                        r["effect_size"] = effect_size
                    results.extend(row)
                except Exception:
                    logger.exception(
                        "Cell %d/%d failed (N=%d, d=%.2f, iter=%d)",
                        cell_i,
                        total_cells,
                        n_per_group,
                        effect_size,
                        iteration,
                    )

                elapsed = time.perf_counter() - t_start
                rate = cell_i / elapsed if elapsed > 0 else 0
                eta = (total_cells - cell_i) / rate if rate > 0 else 0
                if cell_i % 5 == 0 or cell_i == total_cells:
                    print(
                        f"  [{cell_i}/{total_cells}] "
                        f"N={n_per_group} d={effect_size:.1f} "
                        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    )

    return pd.DataFrame(results)


def _run_single_cell(
    base_config,
    n_per_group: int,
    effect_size_delta: float,
    child_seed: int,
    target_params: list[str],
    bf_threshold: float,
    sessions_to_fit: str,
) -> list[dict]:
    """Simulate, fit, and evaluate one (N, d) cell."""
    from prl_hgf.analysis.recovery import (
        build_recovery_df,
        compute_recovery_metrics,
    )

    # 1. Override config
    cell_config = make_power_config(
        base_config, n_per_group, effect_size_delta, child_seed
    )

    # 2. Simulate
    sim_df = simulate_batch(cell_config)

    # 3. Filter sessions
    if sessions_to_fit == "baseline":
        sim_df = sim_df[sim_df["session"] == "baseline"].copy()

    # 4. Fit 3-level
    idata_3 = fit_vb_laplace_prl(
        sim_df,
        model_name="hgf_3level",
        n_pseudo_draws=200,
        random_seed=child_seed,
    )

    # 5. Fit 2-level
    idata_2 = fit_vb_laplace_prl(
        sim_df,
        model_name="hgf_2level",
        n_pseudo_draws=200,
        random_seed=child_seed + 1,
    )

    # 6. Recovery
    fit_df = idata_to_fit_df(idata_3, target_params + ["omega_3"])
    recovery_df = build_recovery_df(sim_df, fit_df, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    # 7. Model comparison
    try:
        bms = compare_models_laplace(
            {"hgf_3level": idata_3, "hgf_2level": idata_2}
        )
        bms_xp_3level = float(bms["xp"][0])
    except Exception:
        bms_xp_3level = float("nan")

    # 8. BF on omega_2 group contrast (if multi-session)
    bf_omega2 = float("nan")
    if sessions_to_fit == "all" and "session" in fit_df.columns:
        try:
            did_psi, did_plc = compute_did_contrast(
                fit_df, "omega_2", "baseline", "post_dose"
            )
            bf_omega2 = compute_jzs_bf(did_psi, did_plc)
        except Exception:
            pass

    # 9. Build result rows (one per parameter)
    n_diverged = int(sim_df.get("diverged", pd.Series(dtype=bool)).sum())
    rows: list[dict] = []
    for _, m in metrics_df.iterrows():
        rows.append(
            {
                "parameter": m["parameter"],
                "recovery_r": m["r"],
                "bias": m["bias"],
                "rmse": m["rmse"],
                "passes_threshold": m["r"] >= 0.7,
                "bf_omega2": bf_omega2,
                "bf_exceeds": bf_omega2 > bf_threshold if np.isfinite(bf_omega2) else False,
                "bms_xp_3level": bms_xp_3level,
                "n_diverged": n_diverged,
            }
        )

    return rows


def summarize_power(
    sweep_df: pd.DataFrame,
    power_threshold: float = 0.80,
) -> pd.DataFrame:
    """Summarize a design sweep into a power table.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of :func:`run_design_sweep`.
    power_threshold : float, default 0.80
        Power level to find minimum N for.

    Returns
    -------
    pd.DataFrame
        Per (parameter, effect_size): mean_r, p_recoverable (proportion
        passing r >= 0.7), min_n_for_recovery (smallest N where
        p_recoverable >= power_threshold), mean_bms_xp.
    """
    grouped = sweep_df.groupby(["parameter", "effect_size", "n_per_group"])
    summary_rows: list[dict] = []

    for (param, es, n), grp in grouped:
        summary_rows.append(
            {
                "parameter": param,
                "effect_size": es,
                "n_per_group": n,
                "mean_r": float(grp["recovery_r"].mean()),
                "p_recoverable": float(grp["passes_threshold"].mean()),
                "mean_bias": float(grp["bias"].mean()),
                "mean_rmse": float(grp["rmse"].mean()),
                "mean_bms_xp": float(grp["bms_xp_3level"].mean()),
                "n_iterations": len(grp),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Find minimum N for each (parameter, effect_size)
    min_n_rows: list[dict] = []
    for (param, es), grp in summary.groupby(["parameter", "effect_size"]):
        grp_sorted = grp.sort_values("n_per_group")
        passing = grp_sorted[grp_sorted["p_recoverable"] >= power_threshold]
        min_n = int(passing["n_per_group"].iloc[0]) if len(passing) > 0 else None
        min_n_rows.append(
            {
                "parameter": param,
                "effect_size": es,
                "min_n_for_recovery": min_n,
            }
        )

    min_n_df = pd.DataFrame(min_n_rows)
    return summary.merge(min_n_df, on=["parameter", "effect_size"])


def print_design_report(
    sweep_df: pd.DataFrame,
    power_threshold: float = 0.80,
) -> None:
    """Print a human-readable design recommendation."""
    summary = summarize_power(sweep_df, power_threshold)

    print("\n" + "=" * 65)
    print("EXPERIMENT DESIGN REPORT")
    print("=" * 65)

    for param in sorted(summary["parameter"].unique()):
        print(f"\n--- {param} ---")
        param_df = summary[summary["parameter"] == param]

        for es in sorted(param_df["effect_size"].unique()):
            es_df = param_df[param_df["effect_size"] == es].sort_values(
                "n_per_group"
            )
            min_n = es_df["min_n_for_recovery"].iloc[0]
            min_n_str = str(min_n) if min_n is not None else ">max tested"

            print(f"\n  Effect size d = {es:.1f}:")
            print(f"    Min N for {power_threshold:.0%} recovery power: {min_n_str}")
            print(f"    {'N':>5s}  {'mean_r':>7s}  {'P(r≥0.7)':>8s}  {'bias':>7s}")
            for _, row in es_df.iterrows():
                print(
                    f"    {row['n_per_group']:5.0f}  "
                    f"{row['mean_r']:7.3f}  "
                    f"{row['p_recoverable']:8.1%}  "
                    f"{row['mean_bias']:+7.3f}"
                )

    print("\n" + "=" * 65)
    print("RECOMMENDATIONS")
    print("=" * 65)

    for param in sorted(summary["parameter"].unique()):
        param_df = summary[summary["parameter"] == param]
        best = param_df.sort_values(
            ["p_recoverable", "mean_r"], ascending=False
        ).iloc[0]
        min_n = best["min_n_for_recovery"]
        if min_n is not None:
            print(f"  {param}: N >= {min_n:.0f} per group (d = {best['effect_size']:.1f})")
        else:
            print(f"  {param}: not recoverable at tested N values")

    print("=" * 65)
