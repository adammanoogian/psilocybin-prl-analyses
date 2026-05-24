#!/usr/bin/env python
"""One-click HGF quickstart demo: simulate, fit (VB-Laplace), recover, compare.

Runs the full analysis pipeline on a small synthetic cohort (5 per group,
baseline session only) using Variational Laplace — the same algorithm as
MATLAB TAPAS ``tapas_fitModel``.  Completes in ~2 minutes on CPU.

Usage:
    python scripts/demo_quickstart.py
    # or: make demo
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────
from prl_hgf.env.task_config import load_config
from prl_hgf.fitting.fit_vb_laplace_prl import (
    compare_models_laplace,
    fit_vb_laplace_prl,
    idata_to_fit_df,
)
from prl_hgf.simulation.batch import simulate_batch

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

N_PER_GROUP = 5


def main() -> None:
    """Run the full quickstart pipeline."""
    print("=" * 60)
    print("HGF Quickstart Demo")
    print("  Simulate -> Fit (VB-Laplace) -> Recover -> Compare")
    print("=" * 60)

    # ── 1. Simulate ─────────────────────────────────────────────────
    print("\n[Step 1] Simulating synthetic cohort...")
    t0 = time.perf_counter()
    config = load_config()
    sim_df = simulate_batch(config)

    sim_df = sim_df[sim_df["session"] == "baseline"].copy()
    pids = sorted(sim_df["participant_id"].unique())[:N_PER_GROUP * 2]
    sim_df = sim_df[sim_df["participant_id"].isin(pids)].copy()

    n_participants = sim_df["participant_id"].nunique()
    n_trials = len(sim_df) // n_participants
    print(
        f"  Cohort: {n_participants} participants, "
        f"{n_trials} trials each"
    )
    print(f"  Simulation: {time.perf_counter() - t0:.1f}s")

    sim_path = DEMO_DIR / "demo_simulated.csv"
    sim_df.to_csv(sim_path, index=False)

    # ── 2. Fit 3-level HGF ──────────────────────────────────────────
    print("\n[Step 2] Fitting 3-level HGF via VB-Laplace...")
    t0 = time.perf_counter()
    idata_3 = fit_vb_laplace_prl(
        sim_df,
        model_name="hgf_3level",
        n_pseudo_draws=500,
    )
    print(f"  3-level fit: {time.perf_counter() - t0:.1f}s")

    # ── 3. Fit 2-level HGF ──────────────────────────────────────────
    print("\n[Step 3] Fitting 2-level HGF via VB-Laplace...")
    t0 = time.perf_counter()
    idata_2 = fit_vb_laplace_prl(
        sim_df,
        model_name="hgf_2level",
        n_pseudo_draws=500,
    )
    print(f"  2-level fit: {time.perf_counter() - t0:.1f}s")

    # ── 4. Parameter recovery ────────────────────────────────────────
    print("\n[Step 4] Parameter recovery (3-level)...")

    fit_df = idata_to_fit_df(
        idata_3, ["omega_2", "beta", "zeta", "omega_3"]
    )

    from prl_hgf.analysis.recovery import (
        build_recovery_df,
        compute_recovery_metrics,
    )

    recovery_df = build_recovery_df(sim_df, fit_df, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    print("\n  Recovery metrics:")
    print("  " + "-" * 50)
    for _, row in metrics_df.iterrows():
        status = "PASS" if row.get("passes_threshold", row["r"] >= 0.7) else "FAIL"
        print(
            f"  {row['parameter']:>10s}:  r = {row['r']:.3f}  "
            f"bias = {row['bias']:+.3f}  [{status}]"
        )
    print("  " + "-" * 50)

    # Recovery scatter plot
    from prl_hgf.analysis.plots import plot_recovery_scatter

    fig = plot_recovery_scatter(recovery_df, metrics_df)
    fig_path = DEMO_DIR / "demo_recovery_scatter.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {fig_path}")

    # ── 5. Model comparison (BMS via Laplace LME) ────────────────────
    print("\n[Step 5] Model comparison (random-effects BMS)...")

    bms = compare_models_laplace({
        "hgf_3level": idata_3,
        "hgf_2level": idata_2,
    })

    print("\n  Log Model Evidence (LME) summary:")
    print(bms["lme_summary"].to_string(index=False))
    print(f"\n  Exceedance probability:           {dict(zip(bms['model_names'], bms['xp'], strict=False))}")
    print(f"  Protected exceedance probability: {dict(zip(bms['model_names'], bms['pxp'], strict=False))}")
    print(f"  Bayesian Omnibus Risk:            {bms['bor']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    lme_3 = float(np.sum(idata_3.attrs["lme"]))
    lme_2 = float(np.sum(idata_2.attrs["lme"]))
    winner = "3-level" if lme_3 > lme_2 else "2-level"

    print("\n" + "=" * 60)
    print(f"Demo complete!  Winner: {winner} HGF (sum LME: {max(lme_3, lme_2):.1f})")
    print(f"  Outputs in: {DEMO_DIR}")
    print("  Next: open notebooks/quickstart_hgf.ipynb for the full tutorial")
    print("=" * 60)


if __name__ == "__main__":
    main()
