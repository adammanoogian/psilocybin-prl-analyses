#!/usr/bin/env python
"""End-to-end smoke test for the full v1.1 power analysis pipeline.

Runs every stage with minimal settings (n_per_group=3, 2 chains, 100 draws)
and reports wall-clock and CPU time per stage.

Usage:
    conda run -n ds_env python scripts/smoke_test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as _cfg
from prl_hgf.env.task_config import load_config
from prl_hgf.power.config import PowerConfig, load_power_config, make_power_config
from prl_hgf.power.curves import compute_power_a, compute_power_b
from prl_hgf.power.iteration import run_power_iteration
from prl_hgf.power.schema import write_parquet_batch

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

_timings: list[dict] = []


def timed(stage_name: str):
    """Decorator to time a function and record results."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            result = fn(*args, **kwargs)
            wall_end = time.perf_counter()
            cpu_end = time.process_time()
            wall_s = wall_end - wall_start
            cpu_s = cpu_end - cpu_start
            _timings.append(
                {
                    "stage": stage_name,
                    "wall_s": wall_s,
                    "cpu_s": cpu_s,
                }
            )
            print(
                f"  {stage_name}: wall={wall_s:.1f}s  cpu={cpu_s:.1f}s"
            )
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

N_PER_GROUP = 5
N_CHAINS = 2
N_DRAWS = 50
N_TUNE = 50
EFFECT_SIZE = 0.3
SEED = 42
# Set True to skip MCMC (dry-run wiring test only)
# On cluster with GPU: set False for real timing
DRY_RUN = "--dry-run" in sys.argv


@timed("1. Load config")
def stage_load_config():
    """Load base and power configs."""
    base = load_config()
    power = load_power_config()
    return base, power


@timed("2a. Simulate only")
def stage_simulate(base_config):
    """Time just simulation (no fitting)."""
    from prl_hgf.simulation.batch import simulate_batch

    cfg = make_power_config(
        base_config, N_PER_GROUP, EFFECT_SIZE, SEED
    )
    sim_df = simulate_batch(cfg)
    return sim_df, cfg


@timed("2b. Fit + BF + BMS (1 iteration)")
def stage_power_iteration(base_config, power_config):
    """Run one full power iteration with minimal settings."""
    results = run_power_iteration(
        base_config=base_config,
        n_per_group=N_PER_GROUP,
        effect_size_delta=EFFECT_SIZE,
        iteration=0,
        child_seed=SEED,
        power_config=power_config,
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
        n_tune=N_TUNE,
    )
    return results


def stage_dry_run_results():
    """Create fake results for pipeline wiring test."""
    rows = []
    for sweep in ["did_postdose", "did_followup", "linear_trend"]:
        rows.append(
            {
                "sweep_type": sweep,
                "effect_size": EFFECT_SIZE,
                "n_per_group": N_PER_GROUP,
                "trial_count": 420,
                "iteration": 0,
                "parameter": "omega_2",
                "bf_value": 3.5 if sweep == "did_postdose" else 1.2,
                "bf_exceeds": False,
                "bms_xp": 0.65,
                "bms_correct": False,
                "recovery_r": 0.72,
                "n_divergences": 0,
                "mean_rhat": 1.02,
            }
        )
    return rows


@timed("3. Write parquet")
def stage_write_parquet(results, output_dir):
    """Write results to parquet."""
    out_path = output_dir / "smoke_test.parquet"
    write_parquet_batch(results, out_path)
    return out_path


@timed("4. Aggregate parquets")
def stage_aggregate(output_dir):
    """Read parquets and compute power summaries."""
    files = sorted(output_dir.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    master_df = pd.concat(frames, ignore_index=True)
    master_path = output_dir / "power_master.csv"
    master_df.to_csv(master_path, index=False)

    power_a = compute_power_a(master_df)
    power_b = compute_power_b(master_df)
    power_a.to_csv(output_dir / "power_a_summary.csv", index=False)
    power_b.to_csv(output_dir / "power_b_summary.csv", index=False)
    return master_df, power_a, power_b


@timed("5. Generate figures")
def stage_figures(master_df, power_a_df, power_b_df, output_dir):
    """Generate all 4 figure types."""
    # Import the plotting script
    spec = importlib.util.spec_from_file_location(
        "plot_power",
        str(Path(__file__).parent / "10_plot_power_curves.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Minimal precheck data (empty — just testing the pipeline)
    precheck_df = pd.DataFrame(
        columns=["trial_count", "parameter", "r", "p", "bias", "rmse", "n",
                 "passes_threshold"]
    )

    fig_a = mod.plot_power_a(
        power_a_df, master_df, bf_threshold=6.0,
        save_path=output_dir / "power_a.png",
    )
    plt.close(fig_a)

    fig_b = mod.plot_power_b(
        power_b_df, save_path=output_dir / "power_b.png",
    )
    plt.close(fig_b)

    fig_h = mod.plot_sensitivity_heatmap(
        master_df, bf_threshold=6.0,
        save_path=output_dir / "sensitivity.png",
    )
    plt.close(fig_h)

    fig_c = mod.plot_combined_figure(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=precheck_df,
        bf_threshold=6.0,
        save_path=output_dir / "combined",
    )
    plt.close(fig_c)

    return True


@timed("6. Generate recommendation")
def stage_recommendation(master_df, power_a_df, power_b_df, output_dir):
    """Generate recommendation.md."""
    spec = importlib.util.spec_from_file_location(
        "write_rec",
        str(Path(__file__).parent / "11_write_recommendation.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    md = mod.generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )
    rec_path = output_dir / "recommendation.md"
    rec_path.write_text(md, encoding="utf-8")
    return rec_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full pipeline smoke test."""
    print("=" * 70)
    print("SMOKE TEST: Full v1.1 Power Analysis Pipeline")
    print("=" * 70)
    print(f"  n_per_group  = {N_PER_GROUP} ({N_PER_GROUP * 2} total participants)")
    print(f"  sessions     = 3 (baseline, post_dose, follow_up)")
    print(f"  fits/iter    = {N_PER_GROUP * 2 * 3 * 2} "
          f"({N_PER_GROUP * 2} x 3 sessions x 2 models)")
    print(f"  MCMC         = {N_CHAINS} chains x {N_DRAWS} draws x {N_TUNE} tune")
    print(f"  effect_size  = {EFFECT_SIZE}")
    print(f"  seed         = {SEED}")
    print("=" * 70)

    output_dir = Path(tempfile.mkdtemp(prefix="prl_smoke_"))
    print(f"\nOutput: {output_dir}\n")

    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    # Stage 1: Load config
    base_config, power_config = stage_load_config()

    # Stage 2a: Simulate only (always — fast, no MCMC)
    sim_df, _ = stage_simulate(base_config)
    n_sessions = sim_df[["participant_id", "session"]].drop_duplicates().shape[0]
    print(f"    -> {len(sim_df)} trial rows, {n_sessions} participant-sessions")

    # Stage 2b: Full iteration (skip in dry-run)
    if DRY_RUN:
        print("  [DRY RUN] Skipping MCMC — using placeholder results")
        results = stage_dry_run_results()
    else:
        results = stage_power_iteration(base_config, power_config)
    print(f"    -> {len(results)} result rows (3 contrasts)")
    for r in results:
        print(
            f"    -> {r['sweep_type']}: BF={r['bf_value']:.2f}, "
            f"BMS_xp={r['bms_xp']:.3f}, rhat={r['mean_rhat']:.3f}"
        )

    # Stage 3: Write parquet
    parquet_path = stage_write_parquet(results, output_dir)

    # Stage 4: Aggregate
    master_df, power_a_df, power_b_df = stage_aggregate(output_dir)
    print(f"    -> master: {len(master_df)} rows")

    # Stage 5: Figures
    stage_figures(master_df, power_a_df, power_b_df, output_dir)
    figs = list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf"))
    print(f"    -> {len(figs)} figure files generated")

    # Stage 6: Recommendation
    rec_path = stage_recommendation(
        master_df, power_a_df, power_b_df, output_dir
    )
    print(f"    -> {rec_path.stat().st_size} bytes")

    total_wall = time.perf_counter() - total_wall_start
    total_cpu = time.process_time() - total_cpu_start

    # Summary
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<45} {'Wall (s)':>10} {'CPU (s)':>10}")
    print("-" * 70)
    for t in _timings:
        print(f"{t['stage']:<45} {t['wall_s']:>10.1f} {t['cpu_s']:>10.1f}")
    print("-" * 70)
    print(f"{'TOTAL':<45} {total_wall:>10.1f} {total_cpu:>10.1f}")
    print("=" * 70)

    # Extrapolation
    iter_match = [t for t in _timings if "iteration" in t["stage"].lower()]
    if iter_match:
        iter_wall = iter_match[0]["wall_s"]
    else:
        print("\n  [DRY RUN] No MCMC timing — extrapolation not available.")
        print("  Run on cluster for real timing.")
        return
    n_participants_this = N_PER_GROUP * 2

    print("\nEXTRAPOLATION TO FULL SWEEP")
    print("-" * 70)
    print(f"Per-iteration wall time (N={N_PER_GROUP}/group): {iter_wall:.1f}s")
    print(f"Per-participant-session fit (approx): "
          f"{iter_wall / (n_participants_this * 3 * 2):.1f}s")
    print()

    grid_n = [10, 15, 20, 25, 30, 40, 50]
    grid_d = [0.3, 0.5, 0.7]
    k = 200
    total_iters = len(grid_n) * len(grid_d) * k

    print(f"Full grid: {len(grid_n)} N x {len(grid_d)} d x {k} iters "
          f"= {total_iters} iterations")
    print()

    # Scale by participant count (fits scale linearly with N)
    per_fit_s = iter_wall / (n_participants_this * 3 * 2)
    for n in grid_n:
        n_fits = n * 2 * 3 * 2  # participants x sessions x models
        est_iter_s = per_fit_s * n_fits
        est_total_h = est_iter_s * len(grid_d) * k / 3600
        est_per_chunk_h = est_total_h / 3  # 3 chunks
        print(
            f"  N={n:>2}/group: ~{est_iter_s:>6.0f}s/iter, "
            f"{est_total_h:>7.1f}h total, "
            f"{est_per_chunk_h:>7.1f}h/chunk (3 chunks)"
        )

    total_fits = sum(
        n * 2 * 3 * 2 * len(grid_d) * k for n in grid_n
    )
    total_est_h = per_fit_s * total_fits / 3600
    print(f"\n  TOTAL ESTIMATED: {total_est_h:.1f}h wall (single core)")
    print(f"  With 3 GPU chunks: ~{total_est_h / 3:.1f}h/chunk")
    print(f"  (GPU expected 5-10x faster than CPU)")

    # Cleanup note
    print(f"\nSmoke test outputs in: {output_dir}")
    print("(Delete manually when done inspecting)")


if __name__ == "__main__":
    main()
