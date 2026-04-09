#!/usr/bin/env python
"""Entry point for chunk-based SBF power analysis on SLURM.

Each invocation processes a contiguous chunk of the SBF power sweep grid.
A single process reuses the JAX-compiled HGF model across all iterations
in the chunk, avoiding redundant JIT compilation.

SBF grid layout: ``effect_size x iteration`` (row-major, 2-D).
Sample size (N) is handled inside each iteration by simulating at
``max(n_per_group_grid)`` and subsampling posteriors at each N level.

With ``n_chunks=3`` and the default grid (3 d x 200 iter = 600 tasks),
each chunk handles 200 iterations (one effect size per chunk) and writes
one combined parquet file.

Chunk assignment:
    chunk 0 → task_ids [0, 200)   (d=0.3)
    chunk 1 → task_ids [200, 400) (d=0.5)
    chunk 2 → task_ids [400, 600) (d=0.7)

In ``--dry-run`` mode, placeholder rows are written without running
the full simulate-fit pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work on cluster
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as _cfg
from prl_hgf.env.task_config import load_config
from prl_hgf.power.config import load_power_config
from prl_hgf.power.grid import (
    chunk_task_ids,
    decode_sbf_task_id,
    sbf_grid_size,
)
from prl_hgf.power.iteration import run_sbf_iteration
from prl_hgf.power.schema import write_parquet_batch
from prl_hgf.power.seeds import make_chunk_rngs

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a chunk of power analysis iterations (SLURM job).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        required=True,
        help="Zero-based chunk index (SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="SLURM_ARRAY_JOB_ID (used in output filename).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Write placeholder parquet rows instead of running the full "
            "pipeline. Used to verify infrastructure without MCMC."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help=(
            "Benchmark mode: time JAX compilation/caching, then fit a "
            "single participant to measure post-cache per-fit time. "
            "Writes timing results to stdout and a benchmark JSON file."
        ),
    )
    parser.add_argument(
        "--fit-chains",
        type=int,
        default=2,
        help="Number of MCMC chains for power sweep fits.",
    )
    parser.add_argument(
        "--fit-draws",
        type=int,
        default=500,
        help="Posterior draws per chain.",
    )
    parser.add_argument(
        "--fit-tune",
        type=int,
        default=500,
        help="Tuning steps per chain.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["pymc", "numpyro"],
        default="pymc",
        help="MCMC backend: pymc (PyTensor NUTS) or numpyro (JAX NUTS).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. Defaults to RESULTS_DIR / 'power'. "
            "Use this in tests to redirect output to a tmp directory."
        ),
    )
    return parser.parse_args()


def _run_benchmark(
    base_config: object,
    power_config: object,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Time JAX compilation and a single real MCMC fit.

    Measures three phases:

    1. JAX JIT compilation / cache load (via ``fit_batch._prewarm_jit``).
    2. Simulation of a small cohort (N=10 per group, baseline only).
    3. A single-participant MCMC fit (post-compilation).

    Writes a JSON report to ``output_dir / benchmark.json`` and prints
    a human-readable summary to stdout.

    Parameters
    ----------
    base_config : AnalysisConfig
        Base analysis configuration loaded from YAML.
    power_config : PowerConfig
        Power analysis grid configuration.
    output_dir : Path
        Directory for the benchmark JSON output.
    args : argparse.Namespace
        Parsed CLI arguments (provides fit-chains, fit-draws, etc.).
    """
    from prl_hgf.fitting.batch import _prewarm_jit
    from prl_hgf.fitting.single import fit_participant
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 60)
    print("BENCHMARK MODE")
    print("=" * 60)

    # --- Phase 0: GPU device info ---
    import subprocess

    import jax

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        dev = gpu_devices[0]
        results["gpu_device"] = str(dev)
        print(f"\nGPU: {dev}")
    else:
        results["gpu_device"] = "none (CPU only)"
        print("\nWARNING: No GPU detected — timings will reflect CPU performance")

    # Capture full nvidia-smi snapshot
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0:
            gpu_info = smi.stdout.strip()
            results["gpu_nvidia_smi"] = gpu_info
            print(f"  nvidia-smi: {gpu_info}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # --- Phase 1: JAX JIT compilation for both models ---
    for model_name in ("hgf_3level", "hgf_2level"):
        print(f"\nPhase 1: JIT compilation ({model_name})...")
        t0 = time.perf_counter()
        _prewarm_jit(model_name)
        dt = time.perf_counter() - t0
        key = f"jit_compile_{model_name}_s"
        results[key] = round(dt, 2)
        print(f"  {model_name} JIT: {dt:.2f}s")

    # --- Phase 2: Simulate a small cohort ---
    print("\nPhase 2: Simulate cohort (N=10/group, 3 sessions)...")
    n_bench = 10
    d_bench = power_config.effect_size_grid[0]
    cfg = make_power_config(base_config, n_bench, d_bench, 99999)

    t0 = time.perf_counter()
    sim_df = simulate_batch(cfg)
    dt_sim = time.perf_counter() - t0
    results["simulate_n10_s"] = round(dt_sim, 2)

    n_participant_sessions = sim_df.groupby(
        ["participant_id", "group", "session"]
    ).ngroups
    print(f"  Simulated {n_participant_sessions} participant-sessions in {dt_sim:.2f}s")

    # --- Phase 3: Fit ONE participant-session (post-cache) per model ---
    import numpy as np

    first_key = (
        sim_df[["participant_id", "group", "session"]]
        .drop_duplicates()
        .iloc[0]
    )
    pid, grp, sess = first_key["participant_id"], first_key["group"], first_key["session"]
    subset = sim_df[
        (sim_df["participant_id"] == pid)
        & (sim_df["group"] == grp)
        & (sim_df["session"] == sess)
    ].sort_values("trial")

    n_trials = len(subset)
    choices = subset["cue_chosen"].to_numpy(dtype=int)
    rewards = subset["reward"].to_numpy(dtype=float)
    input_arr = np.zeros((n_trials, 3), dtype=float)
    observed_arr = np.zeros((n_trials, 3), dtype=int)
    for t in range(n_trials):
        input_arr[t, choices[t]] = rewards[t]
        observed_arr[t, choices[t]] = 1

    def _gpu_vram_mb() -> dict[str, float] | None:
        """Query nvidia-smi for current VRAM usage in MB."""
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if smi.returncode == 0:
                used, total = smi.stdout.strip().split(", ")
                return {"used_mb": float(used), "total_mb": float(total)}
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    for model_name in ("hgf_3level", "hgf_2level"):
        print(f"\nPhase 3: Fit 1 participant ({model_name}, "
              f"{args.fit_chains} chains × {args.fit_draws} draws)...")
        vram_before = _gpu_vram_mb()
        t0 = time.perf_counter()
        fit_participant(
            input_data_arr=input_arr,
            observed_arr=observed_arr,
            choices_arr=choices,
            participant_id=pid,
            group=grp,
            session=sess,
            model_name=model_name,
            n_chains=args.fit_chains,
            n_draws=args.fit_draws,
            n_tune=args.fit_tune,
            target_accept=0.9,
            random_seed=42,
            cores=1,
            sampler=args.sampler,
        )
        dt_fit = time.perf_counter() - t0
        vram_after = _gpu_vram_mb()

        key = f"fit_single_{model_name}_s"
        results[key] = round(dt_fit, 2)
        print(f"  {model_name} single fit: {dt_fit:.2f}s")

        if vram_after:
            vram_key = f"vram_after_{model_name}_mb"
            results[vram_key] = vram_after["used_mb"]
            results["vram_total_mb"] = vram_after["total_mb"]
            print(f"  VRAM: {vram_after['used_mb']:.0f} / {vram_after['total_mb']:.0f} MB")
            if vram_before:
                delta = vram_after["used_mb"] - vram_before["used_mb"]
                results[f"vram_delta_{model_name}_mb"] = round(delta, 1)
                print(f"  VRAM delta: +{delta:.0f} MB")

    # --- Projections (SBF design: fit at max N only, subsample rest) ---
    fit_3 = results["fit_single_hgf_3level_s"]
    fit_2 = results["fit_single_hgf_2level_s"]
    fit_both = fit_3 + fit_2

    n_grid = power_config.n_per_group_grid
    d_grid = power_config.effect_size_grid
    n_iter = power_config.n_iterations
    max_n = max(n_grid)

    # SBF: fit at max N only — 2 groups × max_n × 3 sessions × 2 models
    fits_per_d_iter = 2 * max_n * 3 * 2
    total_fits = fits_per_d_iter * len(d_grid) * n_iter
    total_hours = (total_fits * fit_both / 2) / 3600  # /2: fit_both covers 2 models

    results["projection_sbf_design"] = {
        "total_fits": total_fits,
        "total_gpu_hours": round(total_hours, 1),
        "per_chunk_hours": round(total_hours / power_config.n_chunks, 1),
    }
    results["grid"] = {
        "n_per_group": n_grid,
        "max_n": max_n,
        "effect_sizes": d_grid,
        "n_iterations": n_iter,
        "n_chunks": power_config.n_chunks,
        "sbf_grid_tasks": len(d_grid) * n_iter,
    }
    results["mcmc_settings"] = {
        "chains": args.fit_chains,
        "draws": args.fit_draws,
        "tune": args.fit_tune,
        "sampler": args.sampler,
    }

    # --- Report ---
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  GPU:                   {results.get('gpu_device', 'unknown')}")
    if "vram_total_mb" in results:
        print(f"  VRAM total:            {results['vram_total_mb']:.0f} MB")
    if "vram_after_hgf_3level_mb" in results:
        print(f"  VRAM after 3-level:    {results['vram_after_hgf_3level_mb']:.0f} MB")
    if "vram_after_hgf_2level_mb" in results:
        print(f"  VRAM after 2-level:    {results['vram_after_hgf_2level_mb']:.0f} MB")
    print()
    print(f"  JIT compile (3-level): {results['jit_compile_hgf_3level_s']:.2f}s")
    print(f"  JIT compile (2-level): {results['jit_compile_hgf_2level_s']:.2f}s")
    print(f"  Simulate (N=10):       {results['simulate_n10_s']:.2f}s")
    print(f"  Single fit (3-level):  {fit_3:.2f}s")
    print(f"  Single fit (2-level):  {fit_2:.2f}s")
    print(f"  Both models / participant-session: {fit_both:.2f}s")
    print()
    print("PROJECTIONS (SBF design — fit max N, subsample at each N):")
    print(f"  Max N per group:       {max_n}")
    print(f"  SBF grid tasks:        {len(d_grid) * n_iter:,}")
    print(f"  Total MCMC fits:       {total_fits:,}")
    print(f"  Estimated GPU-hours:   {total_hours:.1f}h")
    n_chunks = power_config.n_chunks
    print(f"  Per chunk ({n_chunks} chunks): {total_hours / n_chunks:.1f}h")
    if "vram_total_mb" in results:
        peak = max(
            results.get("vram_after_hgf_3level_mb", 0),
            results.get("vram_after_hgf_2level_mb", 0),
        )
        print()
        print("GPU SIZING:")
        print(f"  Peak VRAM (single fit): {peak:.0f} MB")
        print(f"  GPU assigned:           {results['vram_total_mb']:.0f} MB")
        headroom = results["vram_total_mb"] - peak
        print(f"  Headroom:               {headroom:.0f} MB")
        if peak < 8000:
            print(f"  Recommendation:         T4 (16 GB) sufficient")
        elif peak < 20000:
            print(f"  Recommendation:         A40 (48 GB) or A100 (40 GB)")
        else:
            print(f"  Recommendation:         A100 (80 GB)")
    print("=" * 60)

    bench_path = output_dir / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to: {bench_path}")


def main() -> None:
    """Execute one chunk of the power sweep.

    Raises
    ------
    SystemExit
        On argument parse error (via argparse).
    """
    args = parse_args()

    base_config = load_config()
    power_config = load_power_config()

    grid_size = sbf_grid_size(
        power_config.effect_size_grid,
        power_config.n_iterations,
    )
    task_ids = chunk_task_ids(args.chunk_id, power_config.n_chunks, grid_size)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _cfg.RESULTS_DIR / "power"
    )

    out_path = output_dir / f"job_{args.job_id}_chunk_{args.chunk_id:04d}.parquet"

    if args.benchmark:
        _run_benchmark(base_config, power_config, output_dir, args)
        return

    if args.dry_run:
        rows: list[dict] = []
        for tid in task_ids:
            effect_size, iteration = decode_sbf_task_id(
                tid,
                power_config.effect_size_grid,
                power_config.n_iterations,
            )
            for n_per_group in sorted(power_config.n_per_group_grid):
                rows.append(
                    {
                        "sweep_type": "smoke_test",
                        "effect_size": effect_size,
                        "n_per_group": n_per_group,
                        "trial_count": base_config.task.n_trials_total,
                        "iteration": iteration,
                        "parameter": "omega_2",
                        "bf_value": 1.0,
                        "bf_exceeds": False,
                        "bms_xp": 0.5,
                        "bms_correct": False,
                        "recovery_r": 0.0,
                        "n_divergences": 0,
                        "mean_rhat": 1.0,
                    }
                )
        write_parquet_batch(rows, out_path)
        print(
            f"Dry run: wrote {len(rows)} placeholder rows to {out_path}"
        )
        return

    # Build independent RNGs for all task IDs in this chunk (spawn once)
    rngs = make_chunk_rngs(power_config.master_seed, grid_size, task_ids)

    all_results: list[dict] = []
    for i, tid in enumerate(task_ids):
        effect_size_delta, iteration = decode_sbf_task_id(
            tid,
            power_config.effect_size_grid,
            power_config.n_iterations,
        )

        log.info(
            "Chunk %d: task %d/%d (d=%.1f, iter=%d, N_grid=%s)",
            args.chunk_id,
            i + 1,
            len(task_ids),
            effect_size_delta,
            iteration,
            power_config.n_per_group_grid,
        )

        results = run_sbf_iteration(
            base_config=base_config,
            effect_size_delta=effect_size_delta,
            iteration=iteration,
            child_seed=int(rngs[i].integers(0, 2**31)),
            n_per_group_grid=power_config.n_per_group_grid,
            power_config=power_config,
            n_chains=args.fit_chains,
            n_draws=args.fit_draws,
            n_tune=args.fit_tune,
            sampler=args.sampler,
        )
        all_results.extend(results)

        if (i + 1) % 50 == 0:
            print(
                f"  Progress: {i + 1}/{len(task_ids)} iterations complete"
            )

    write_parquet_batch(all_results, out_path)
    print(
        f"Wrote {len(all_results)} rows ({len(task_ids)} iterations) "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
