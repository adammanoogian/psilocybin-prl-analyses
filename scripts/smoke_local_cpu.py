#!/usr/bin/env python
"""Minimal CPU-only driver for BlackJAX-NUTS HGF sampling diagnostics.

Forces ``JAX_PLATFORMS=cpu`` and runs three successive fits (cold /
warm1 / warm2) at a tiny cohort size with a handful of draws, then
prints per-draw NUTS diagnostics (integration steps, acceptance,
divergence) for each.  The goal is to answer: is the cluster-observed
~1174 s warm sampling time dominated by compile, by Python retracing,
or by the sampler itself hitting ``max_tree_depth``?

Use ``--enable-x64`` to compare fp32 vs fp64.  With 2 participants /
group at 1 set (140 trials), fp32 on CPU should finish cold+warm1+warm2
in well under a minute if the math is healthy.

Usage
-----
    python scripts/smoke_local_cpu.py
    python scripts/smoke_local_cpu.py --enable-x64
    python scripts/smoke_local_cpu.py --n-per-group 3 --draws 20 --tune 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# --- JAX config must happen before jax is imported ---
if "--enable-x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

import numpy as np  # noqa: E402

from prl_hgf.env.task_config import load_config  # noqa: E402
from prl_hgf.fitting.hierarchical import fit_batch_hierarchical  # noqa: E402
from prl_hgf.power.config import make_power_config  # noqa: E402
from prl_hgf.simulation.batch import simulate_batch  # noqa: E402


def _summary(label: str, idata: object, wall_s: float) -> None:
    """Print aggregate NUTS diagnostics for one fit."""
    stats = idata.sample_stats
    steps = np.asarray(stats["num_integration_steps"]).ravel()
    accept = np.asarray(stats["acceptance_rate"]).ravel()
    div = np.asarray(stats["diverging"]).ravel()
    expansions = (
        np.asarray(stats["num_trajectory_expansions"]).ravel()
        if "num_trajectory_expansions" in stats
        else None
    )
    print(f"\n[{label}] wall={wall_s:.2f}s, n_draws_total={steps.size}")
    print(
        f"  integration_steps: mean={np.mean(steps):.1f}  p50={np.percentile(steps, 50):.0f}  "
        f"p95={np.percentile(steps, 95):.0f}  max={np.max(steps)}"
    )
    if expansions is not None:
        print(
            f"  trajectory_expansions: mean={np.mean(expansions):.2f}  "
            f"max={np.max(expansions)} (10 = hit max_tree_depth cap)"
        )
    print(
        f"  acceptance_rate mean={np.mean(accept):.3f}  divergence_rate={np.mean(div):.3f}"
    )


def main() -> None:
    """Run cold/warm1/warm2 timing and NUTS diagnostics on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-x64", action="store_true", default=False)
    parser.add_argument("--n-per-group", type=int, default=2)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--tune", type=int, default=10)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--model",
        choices=["hgf_2level", "hgf_3level"],
        default="hgf_3level",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Local CPU BlackJAX NUTS smoke test")
    print("=" * 60)
    print(f"  JAX {jax.__version__}, devices={jax.devices()}")
    print(f"  jax_enable_x64: {jax.config.jax_enable_x64}")
    print(f"  model: {args.model}")
    print(
        f"  n_per_group={args.n_per_group}  chains={args.chains}  "
        f"draws={args.draws}  tune={args.tune}"
    )

    base = load_config()
    # Build a minimal power config to feed simulate_batch.
    cfg = make_power_config(base, args.n_per_group, 0.3, master_seed=12345)
    t0 = time.perf_counter()
    sim_df = simulate_batch(cfg)
    print(f"\n  simulate_batch: {time.perf_counter() - t0:.2f}s  rows={len(sim_df)}")

    n_ps = sim_df[["participant_id", "session"]].drop_duplicates().shape[0]
    n_trials = (
        sim_df.groupby(["participant_id", "session"]).size().iloc[0]
    )
    print(f"  participant-sessions={n_ps}  trials/session={n_trials}")

    # --- Cold: full warmup + sample ---
    t0 = time.perf_counter()
    idata_cold, adapted = fit_batch_hierarchical(
        sim_df,
        args.model,
        n_chains=args.chains,
        n_draws=args.draws,
        n_tune=args.tune,
        target_accept=0.9,
        random_seed=42,
        progressbar=False,
    )
    cold_s = time.perf_counter() - t0
    _summary("cold (warmup+sample)", idata_cold, cold_s)

    # --- Warm 1: skip warmup, new seed ---
    t0 = time.perf_counter()
    idata_warm1 = fit_batch_hierarchical(
        sim_df,
        args.model,
        n_chains=args.chains,
        n_draws=args.draws,
        n_tune=args.tune,
        target_accept=0.9,
        random_seed=43,
        progressbar=False,
        warmup_params=adapted,
    )
    warm1_s = time.perf_counter() - t0
    _summary("warm1 (sample only)", idata_warm1, warm1_s)

    # --- Warm 2: trace-cache hit ---
    t0 = time.perf_counter()
    idata_warm2 = fit_batch_hierarchical(
        sim_df,
        args.model,
        n_chains=args.chains,
        n_draws=args.draws,
        n_tune=args.tune,
        target_accept=0.9,
        random_seed=44,
        progressbar=False,
        warmup_params=adapted,
    )
    warm2_s = time.perf_counter() - t0
    _summary("warm2 (sample only)", idata_warm2, warm2_s)

    print()
    print(f"cold  - warm1 = {cold_s - warm1_s:+.2f}s  (warmup cost estimate)")
    print(
        f"warm1 - warm2 = {warm1_s - warm2_s:+.2f}s  "
        "(python-retrace + XLA-compile cost estimate)"
    )
    print(f"warm2          = {warm2_s:.2f}s  (pure execute estimate)")


if __name__ == "__main__":
    main()
