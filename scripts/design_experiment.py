#!/usr/bin/env python
"""Design experiment helper: find the N and trials needed for your hypothesis.

Runs a fast VB-Laplace sweep to answer:
  1. Are the parameters recoverable at your sample size?
  2. What N do you need for 80% power?
  3. Can you distinguish groups at a given effect size?
  4. Which model (2-level vs 3-level) is supported?

Usage:
    # Quick sweep (5 min): 3 N values, 2 effect sizes, 10 iterations
    python scripts/design_experiment.py --quick

    # Standard sweep (30 min): 4 N values, 3 effect sizes, 20 iterations
    python scripts/design_experiment.py

    # Custom sweep
    python scripts/design_experiment.py --n-grid 10 20 30 40 50 \
        --effect-grid 0.3 0.5 0.8 --n-iter 30

    # or: make design
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prl_hgf.power.laplace_power import (
    print_design_report,
    run_design_sweep,
    summarize_power,
)

DESIGN_DIR = Path(__file__).resolve().parent.parent / "results" / "design"


def plot_power_curves(
    summary_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Plot recovery power curves: P(r >= 0.7) vs N for each parameter."""
    params = sorted(summary_df["parameter"].unique())
    effect_sizes = sorted(summary_df["effect_size"].unique())
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params, strict=False):
        for es in effect_sizes:
            mask = (summary_df["parameter"] == param) & (
                summary_df["effect_size"] == es
            )
            sub = summary_df[mask].sort_values("n_per_group")
            ax.plot(
                sub["n_per_group"],
                sub["p_recoverable"],
                "o-",
                label=f"d = {es:.1f}",
            )

        ax.axhline(0.80, color="gray", linestyle="--", alpha=0.5, label="80% power")
        ax.set_xlabel("N per group")
        ax.set_title(param)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("P(recovery r >= 0.7)")
    fig.suptitle("Recovery Power Curves (VB-Laplace)", fontweight="bold")
    plt.tight_layout()

    save_path = save_dir / "power_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Power curves saved to {save_path}")


def plot_recovery_by_n(
    summary_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Plot mean recovery r vs N for each parameter and effect size."""
    params = sorted(summary_df["parameter"].unique())
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params, strict=False):
        for es in sorted(summary_df["effect_size"].unique()):
            mask = (summary_df["parameter"] == param) & (
                summary_df["effect_size"] == es
            )
            sub = summary_df[mask].sort_values("n_per_group")
            ax.plot(sub["n_per_group"], sub["mean_r"], "o-", label=f"d = {es:.1f}")

        ax.axhline(0.70, color="red", linestyle="--", alpha=0.5, label="r = 0.7")
        ax.set_xlabel("N per group")
        ax.set_title(param)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean recovery r")
    fig.suptitle("Parameter Recovery vs Sample Size", fontweight="bold")
    plt.tight_layout()

    save_path = save_dir / "recovery_by_n.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recovery curves saved to {save_path}")


def main() -> None:
    """Run the design experiment sweep."""
    parser = argparse.ArgumentParser(
        description="HGF experiment design helper (VB-Laplace)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 N values, 2 effect sizes, 10 iterations (~5 min)",
    )
    parser.add_argument("--n-grid", nargs="+", type=int, default=None)
    parser.add_argument("--effect-grid", nargs="+", type=float, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        n_grid = args.n_grid or [10, 20, 30]
        effect_grid = args.effect_grid or [0.3, 0.7]
        n_iter = args.n_iter or 10
    else:
        n_grid = args.n_grid or [10, 20, 30, 50]
        effect_grid = args.effect_grid or [0.3, 0.5, 0.7]
        n_iter = args.n_iter or 20

    DESIGN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HGF EXPERIMENT DESIGN HELPER")
    print("=" * 65)
    print(f"  N per group:   {n_grid}")
    print(f"  Effect sizes:  {effect_grid}")
    print(f"  Iterations:    {n_iter}")
    total = len(n_grid) * len(effect_grid) * n_iter
    print(f"  Total cells:   {total}")
    print()

    t0 = time.perf_counter()
    sweep_df = run_design_sweep(
        n_per_group_grid=n_grid,
        effect_size_grid=effect_grid,
        n_iterations=n_iter,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Save raw results
    sweep_path = DESIGN_DIR / "design_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Raw results saved to {sweep_path}")

    # Summary and report
    summary_df = summarize_power(sweep_df)
    summary_path = DESIGN_DIR / "design_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print_design_report(sweep_df)

    # Plots
    plot_power_curves(summary_df, DESIGN_DIR)
    plot_recovery_by_n(summary_df, DESIGN_DIR)

    print(f"\n  All outputs in: {DESIGN_DIR}")


if __name__ == "__main__":
    main()
