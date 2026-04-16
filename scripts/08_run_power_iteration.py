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
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work on cluster
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax

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
        "--smoke-test",
        action="store_true",
        default=False,
        help=(
            "Smoke test mode: lightweight JIT timing at a fixed small N "
            "(N=5/group, 30 participant-sessions).  Tests cold JIT, warm "
            "JIT cache reuse, and simulation vmap.  Short enough for a "
            "2-hour wall time.  Writes smoke_test.json."
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
        default="numpyro",
        help=(
            "DEPRECATED: ignored for the batched path (always uses "
            "numpyro-direct MCMC). Retained for backward compatibility "
            "with existing SLURM scripts."
        ),
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
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help=(
            "Use the v1.1 legacy per-participant sequential fitting path "
            "instead of the v1.2 batched hierarchical path. Preserved for "
            "reproducibility and debugging (VALID-05)."
        ),
    )
    return parser.parse_args()


class _GpuMonitor:
    """Background-threaded nvidia-smi poller for GPU utilization metrics."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: list[dict] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    parts = out.stdout.strip().split(", ")
                    if len(parts) == 3:
                        self.samples.append(
                            {
                                "gpu_util_pct": float(parts[0]),
                                "vram_used_mb": float(parts[1]),
                                "vram_total_mb": float(parts[2]),
                            }
                        )
            except Exception:  # noqa: BLE001
                pass

    @property
    def peak_vram_mb(self) -> float:
        """Peak VRAM usage observed across all samples (MB)."""
        return max((s["vram_used_mb"] for s in self.samples), default=0.0)

    @property
    def mean_gpu_util_pct(self) -> float:
        """Mean GPU utilisation percentage across all samples."""
        if not self.samples:
            return 0.0
        return float(
            sum(s["gpu_util_pct"] for s in self.samples) / len(self.samples)
        )

    @property
    def vram_total_mb(self) -> float:
        """Total VRAM capacity reported by nvidia-smi (MB)."""
        return max((s["vram_total_mb"] for s in self.samples), default=0.0)


def _update_state_md(
    state_md_path: Path,
    decision: str,
    gpu_hours_per_chunk: float,
    per_iteration_s: float,
) -> None:
    """Append decision gate result to .planning/STATE.md Key Decisions table.

    Parameters
    ----------
    state_md_path : Path
        Absolute path to .planning/STATE.md.
    decision : str
        Gate decision: ``"gpu"`` or ``"cpu_comp"``.
    gpu_hours_per_chunk : float
        Projected GPU-hours per chunk.
    per_iteration_s : float
        Measured wall-clock time for one full iteration (seconds).
    """
    if not state_md_path.exists():
        print(
            f"WARNING: STATE.md not found at {state_md_path} — "
            "skipping decision gate append."
        )
        return

    text = state_md_path.read_text(encoding="utf-8")

    operator = ">" if decision == "cpu_comp" else "<="
    row = (
        f"| Benchmark decision gate: {decision} "
        f"({gpu_hours_per_chunk} GPU-hrs/chunk) "
        f"| BENCH-02: per_iter_s={round(per_iteration_s, 1)}, "
        f"formula=per_iter_s*600/3600 {operator} 50 | 14-02 |\n"
    )

    # Find the Key Decisions table and append before the next blank line /
    # section heading.  Locate the header row of the table to anchor insertion.
    table_header = "| Decision | Rationale | Phase |"
    if table_header not in text:
        print(
            "WARNING: Key Decisions table header not found in STATE.md — "
            "appending row at end of file."
        )
        text = text.rstrip("\n") + "\n\n" + row
    else:
        # Find the last row of the Key Decisions table: the last line starting
        # with '| ' that comes after the table header.
        header_pos = text.index(table_header)
        after_header = text[header_pos:]
        lines = after_header.split("\n")
        last_table_line_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith("| "):
                last_table_line_idx = idx
        insert_pos = header_pos + len(
            "\n".join(lines[: last_table_line_idx + 1])
        )
        text = text[: insert_pos + 1] + row + text[insert_pos + 1 :]

    state_md_path.write_text(text, encoding="utf-8")
    print(f"  Appended decision gate row to {state_md_path}")


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def _get_cache_stats(cache_dir: str | None) -> dict:
    """Inspect JAX compilation cache directory and return stats."""
    info: dict = {
        "cache_dir": cache_dir,
        "cache_exists": False,
        "n_files": 0,
        "total_size_mb": 0.0,
    }
    if not cache_dir:
        return info
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return info
    info["cache_exists"] = True
    files = list(cache_path.rglob("*"))
    real_files = [f for f in files if f.is_file()]
    info["n_files"] = len(real_files)
    info["total_size_mb"] = round(
        sum(f.stat().st_size for f in real_files) / (1024 * 1024), 2,
    )
    return info


def _query_gpu_table() -> list[dict]:
    """Query per-GPU info via nvidia-smi as structured list."""
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    rows = []
    for line in smi.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "vram_total_mb": int(parts[2]),
                "vram_used_mb": int(parts[3]),
                "vram_free_mb": int(parts[4]),
                "gpu_util_pct": int(parts[5]),
                "temp_c": int(parts[6]),
                "pci_bus": parts[7],
            })
    return rows


def _get_ptxas_release() -> str | None:
    """Return just the ptxas release string, e.g. '12.8, V12.8.93'."""
    try:
        result = subprocess.run(
            ["ptxas", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "release" in line:
                return line.strip()
        return result.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        return None


def _get_xla_env() -> dict:
    """Capture XLA/JAX-related environment variables."""
    import os
    keys = [
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_LOG_COMPILES",
        "JAX_PLATFORMS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
    ]
    return {k: os.environ.get(k, "(unset)") for k in keys}


def _print_gpu_table(gpus: list[dict]) -> None:
    """Pretty-print per-GPU info."""
    if not gpus:
        print("  nvidia-smi: unavailable")
        return
    print(f"\n  {'GPU':<5} {'Name':<20} {'VRAM Used/Total':>18} {'Util':>6} {'Temp':>6}")
    print(f"  {'-'*5} {'-'*20} {'-'*18} {'-'*6} {'-'*6}")
    for g in gpus:
        print(
            f"  {g['index']:<5} {g['name']:<20} "
            f"{g['vram_used_mb']:>6}/{g['vram_total_mb']:<6} MB "
            f"{g['gpu_util_pct']:>4}% "
            f"{g['temp_c']:>4}C"
        )


def _run_smoke_test(
    base_config: object,
    power_config: object,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Lightweight JIT smoke test at fixed small N.

    Tests three things at N=5/group (30 participant-sessions):

    1. Simulation vmap compiles and runs.
    2. Cold JIT for ``fit_batch_hierarchical`` (first MCMC call).
    3. Warm JIT cache reuse (second call, same shape, different data).

    Uses minimal MCMC settings (2 chains, 10 draws, 10 tune) to isolate
    compilation cost from sampling cost.  Writes ``smoke_test.json``.

    Parameters
    ----------
    base_config : AnalysisConfig
        Base analysis configuration loaded from YAML.
    power_config : PowerConfig
        Power analysis grid configuration.
    output_dir : Path
        Directory for the smoke test JSON output.
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    import os

    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    # Enable cache miss diagnostics (logs why tracing cache misses)
    jax.config.update("jax_explain_cache_misses", True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    n_smoke = 5  # participants per group
    n_chains_smoke = 2
    n_draws_smoke = 10
    n_tune_smoke = 10
    d_smoke = power_config.effect_size_grid[0]

    print("=" * 60)
    print("SMOKE TEST MODE (JIT compilation diagnostics)")
    print("=" * 60)

    # --- 1. GPU device table ---
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    n_gpus = len(gpu_devices)

    print(f"\nJAX devices ({len(devices)} total, {n_gpus} GPU):")
    for d in devices:
        print(f"  {d}")

    if gpu_devices:
        results["gpu_device"] = str(gpu_devices[0])
    else:
        results["gpu_device"] = "none (CPU only)"
        print("  WARNING: No GPU detected — running on CPU.")

    gpus = _query_gpu_table()
    _print_gpu_table(gpus)
    results["gpu_table"] = gpus

    # --- 2. ptxas (release line only) ---
    ptxas_release = _get_ptxas_release()
    results["ptxas_available"] = ptxas_release is not None
    results["ptxas_version"] = ptxas_release or "not found"
    if ptxas_release:
        print(f"\n  ptxas: {ptxas_release}")
    else:
        print("\n  WARNING: ptxas not found — XLA parallel compilation disabled")

    # --- 3. XLA / JAX environment ---
    xla_env = _get_xla_env()
    results["xla_env"] = xla_env
    print("\n  XLA/JAX environment:")
    for k, v in xla_env.items():
        print(f"    {k} = {v}")

    # --- 4. JAX compilation cache (before) ---
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_before = _get_cache_stats(cache_dir)
    results["cache_before"] = cache_before
    print(f"\n  JAX cache (before):")
    print(f"    dir:    {cache_before['cache_dir']}")
    print(f"    exists: {cache_before['cache_exists']}")
    print(f"    files:  {cache_before['n_files']}")
    print(f"    size:   {cache_before['total_size_mb']} MB")

    # --- 5. JAX/jaxlib version ---
    import jaxlib
    results["jax_version"] = jax.__version__
    results["jaxlib_version"] = jaxlib.__version__
    print(f"\n  JAX {jax.__version__}, jaxlib {jaxlib.__version__}")

    # --- Config summary ---
    n_participant_sessions = 2 * n_smoke * 3
    # Always vectorized: enables jit_model_args for trace cache reuse
    # and avoids confirmed L40S pmap bug (JAX #31626)
    chain_method = "vectorized"

    print(f"\nSmoke test config:")
    print(f"  N per group:            {n_smoke}")
    print(f"  Participant-sessions:   {n_participant_sessions}")
    print(f"  Chains:                 {n_chains_smoke}")
    print(f"  Draws/tune:             {n_draws_smoke}/{n_tune_smoke}")
    print(f"  Model:                  hgf_3level")
    print(f"  GPUs available:         {n_gpus}")
    print(f"  Chain method:           {chain_method} (jit_model_args=True)")

    # --- Step 1: Simulation vmap ---
    print(f"\nStep 1: Simulate cohort (N={n_smoke}/group)...")
    cfg_smoke = make_power_config(base_config, n_smoke, d_smoke, 77777)
    t0 = time.perf_counter()
    sim_smoke = simulate_batch(cfg_smoke)
    sim_s = time.perf_counter() - t0
    results["sim_vmap_s"] = round(sim_s, 2)
    print(f"  Simulation: {sim_s:.2f}s")

    # --- Step 2: Cold JIT ---
    print("\nStep 2: Cold JIT (first MCMC call — includes XLA compilation)...")
    gpus_pre_cold = _query_gpu_table()
    results["gpu_pre_cold_jit"] = gpus_pre_cold
    if gpus_pre_cold:
        for g in gpus_pre_cold:
            print(
                f"  GPU {g['index']} VRAM before: "
                f"{g['vram_used_mb']}/{g['vram_total_mb']} MB"
            )

    t0 = time.perf_counter()
    # Cold call: full warmup, returns (idata, adapted_params)
    _idata_cold, adapted_params = fit_batch_hierarchical(
        sim_smoke,
        "hgf_3level",
        n_chains=n_chains_smoke,
        n_draws=n_draws_smoke,
        n_tune=n_tune_smoke,
        target_accept=0.9,
        random_seed=42,
        progressbar=False,
    )
    jit_cold_s = time.perf_counter() - t0
    results["jit_cold_s"] = round(jit_cold_s, 2)
    print(f"  Cold JIT: {jit_cold_s:.2f}s")

    gpus_post_cold = _query_gpu_table()
    results["gpu_post_cold_jit"] = gpus_post_cold
    if gpus_post_cold:
        for g in gpus_post_cold:
            print(
                f"  GPU {g['index']} VRAM after:  "
                f"{g['vram_used_mb']}/{g['vram_total_mb']} MB"
            )

    # --- Cache after cold JIT ---
    cache_after_cold = _get_cache_stats(cache_dir)
    results["cache_after_cold"] = cache_after_cold
    new_files = cache_after_cold["n_files"] - cache_before["n_files"]
    new_mb = cache_after_cold["total_size_mb"] - cache_before["total_size_mb"]
    print(f"  Cache delta: +{new_files} files, +{new_mb:.1f} MB")

    # --- Step 3: Warm JIT (same shape, different data) ---
    print("\nStep 3: Warm JIT (same shape, different seed — tests cache reuse)...")
    cfg_smoke_2 = make_power_config(base_config, n_smoke, d_smoke, 88888)
    sim_smoke_2 = simulate_batch(cfg_smoke_2)

    gpus_pre_warm = _query_gpu_table()
    results["gpu_pre_warm_jit"] = gpus_pre_warm

    t0 = time.perf_counter()
    # Warm call: skip warmup by reusing adapted params from cold call
    fit_batch_hierarchical(
        sim_smoke_2,
        "hgf_3level",
        n_chains=n_chains_smoke,
        n_draws=n_draws_smoke,
        n_tune=n_tune_smoke,
        target_accept=0.9,
        random_seed=43,
        progressbar=False,
        warmup_params=adapted_params,
    )
    jit_warm_s = time.perf_counter() - t0
    results["jit_warm_s"] = round(jit_warm_s, 2)
    print(f"  Warm JIT: {jit_warm_s:.2f}s (warmup skipped via warmup_params)")

    gpus_post_warm = _query_gpu_table()
    results["gpu_post_warm_jit"] = gpus_post_warm
    if gpus_post_warm:
        for g in gpus_post_warm:
            print(
                f"  GPU {g['index']} VRAM after:  "
                f"{g['vram_used_mb']}/{g['vram_total_mb']} MB"
            )

    # --- Cache after warm JIT ---
    cache_after_warm = _get_cache_stats(cache_dir)
    results["cache_after_warm"] = cache_after_warm
    warm_new_files = (
        cache_after_warm["n_files"] - cache_after_cold["n_files"]
    )
    warm_new_mb = (
        cache_after_warm["total_size_mb"] - cache_after_cold["total_size_mb"]
    )
    print(f"  Cache delta: +{warm_new_files} files, +{warm_new_mb:.1f} MB")
    if warm_new_files > 0:
        print("  WARNING: warm JIT wrote NEW cache entries — cache miss detected")
    else:
        print("  OK: no new cache entries (cache hit)")

    cache_speedup = jit_cold_s / max(jit_warm_s, 0.001)
    results["cache_speedup"] = round(cache_speedup, 2)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  GPU:                {results.get('gpu_device', 'unknown')}")
    print(f"  ptxas:              {results['ptxas_version']}")
    print(f"  Chain method:       {chain_method}")
    print(f"  Sim vmap:           {results['sim_vmap_s']:.2f}s")
    print(f"  Cold JIT:           {results['jit_cold_s']:.2f}s")
    print(f"  Warm JIT:           {results['jit_warm_s']:.2f}s")
    print(f"  Cache speedup:      {cache_speedup:.1f}x")
    print(f"  Cache files before: {cache_before['n_files']}")
    print(f"  Cache files after:  {cache_after_warm['n_files']}")
    print()

    # Pass/fail gates
    cold_ok = jit_cold_s < 600  # 10 min max for cold JIT
    cache_ok = cache_speedup > 3.0  # warm should be >3x faster than cold
    warm_ok = jit_warm_s < 120  # warm JIT under 2 min

    results["gate_cold_jit_under_600s"] = cold_ok
    results["gate_cache_speedup_over_3x"] = cache_ok
    results["gate_warm_jit_under_120s"] = warm_ok
    all_pass = cold_ok and cache_ok and warm_ok
    results["all_gates_pass"] = all_pass

    print("GATES:")
    print(f"  Cold JIT < 600s:    {'PASS' if cold_ok else 'FAIL'} ({jit_cold_s:.0f}s)")
    print(f"  Cache speedup > 3x: {'PASS' if cache_ok else 'FAIL'} ({cache_speedup:.1f}x)")
    print(f"  Warm JIT < 120s:    {'PASS' if warm_ok else 'FAIL'} ({jit_warm_s:.0f}s)")
    print(f"  Overall:            {'ALL PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        print()
        if not cold_ok:
            print("  FIX: Cold JIT too slow. Check CUDA/PTX version match:")
            print("       pip install -r cluster/requirements-gpu.txt")
        if not cache_ok:
            print("  FIX: XLA cache not reusing compiled kernels.")
            print("       Verify JAX_COMPILATION_CACHE_DIR is set and writable.")
            print("       Verify data is passed as dynamic args (not closure).")
            if warm_new_files > 0:
                print("       DIAGNOSTIC: warm JIT wrote new cache entries,")
                print("       confirming cache miss (different trace shape?).")
            else:
                print("       DIAGNOSTIC: no new cache entries from warm JIT —")
                print("       cache hit occurred but compilation still slow.")
                print("       This suggests the model itself is slow, not the cache.")
        if not warm_ok:
            print("  FIX: Even cached JIT is slow. Check GPU memory pressure")
            print("       and CUDA driver version.")
            if gpus_post_warm:
                max_vram_pct = max(
                    g["vram_used_mb"] / g["vram_total_mb"] * 100
                    for g in gpus_post_warm
                )
                if max_vram_pct > 90:
                    print(f"       WARNING: VRAM at {max_vram_pct:.0f}% — "
                          "memory pressure likely causing slowdown.")

    print("=" * 60)

    # --- Write JSON ---
    results["smoke_test_config"] = {
        "n_per_group": n_smoke,
        "participant_sessions": n_participant_sessions,
        "chains": n_chains_smoke,
        "draws": n_draws_smoke,
        "tune": n_tune_smoke,
        "n_gpus": n_gpus,
        "chain_method": chain_method,
    }
    smoke_path = output_dir / "smoke_test.json"
    with open(smoke_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {smoke_path}")

    if not all_pass:
        sys.exit(1)


def _run_benchmark(
    base_config: object,
    power_config: object,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run one full batched SBF iteration and report timing + decision gate.

    Measures four phases:

    1. GPU device info query.
    2. JAX compilation cache test: two back-to-back small fits (BENCH-05).
    3. Full batched iteration timing at max N (BENCH-01).
    4. Decision gate application and result recording (BENCH-02).

    GPU utilisation is sampled via background-threaded nvidia-smi during the
    full iteration (BENCH-04).

    Writes ``benchmark_batched.json`` to ``output_dir`` and appends the
    decision gate result to ``.planning/STATE.md``.

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
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.power.config import make_power_config
    from prl_hgf.power.iteration import apply_decision_gate, run_sbf_iteration
    from prl_hgf.simulation.batch import simulate_batch

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=" * 60)
    print("BENCHMARK MODE (batched hierarchical path)")
    print("=" * 60)

    # --- Phase 0: GPU device info ---
    import os

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    n_gpus = len(gpu_devices)

    print(f"\nJAX devices ({len(devices)} total, {n_gpus} GPU):")
    for d in devices:
        print(f"  {d}")

    if gpu_devices:
        results["gpu_device"] = str(gpu_devices[0])
    else:
        results["gpu_device"] = "none (CPU only)"
        print(
            "\nWARNING: No GPU detected. Benchmark reflects CPU performance "
            "only. Decision gate results may not be meaningful."
        )

    gpus = _query_gpu_table()
    _print_gpu_table(gpus)
    results["gpu_table"] = gpus

    ptxas_release = _get_ptxas_release()
    results["ptxas_available"] = ptxas_release is not None
    results["ptxas_version"] = ptxas_release or "not found"
    if ptxas_release:
        print(f"\n  ptxas: {ptxas_release}")

    xla_env = _get_xla_env()
    results["xla_env"] = xla_env
    print("\n  XLA/JAX environment:")
    for k, v in xla_env.items():
        print(f"    {k} = {v}")

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_before = _get_cache_stats(cache_dir)
    results["cache_before"] = cache_before
    print(f"\n  JAX cache: {cache_before['n_files']} files, "
          f"{cache_before['total_size_mb']} MB in {cache_dir}")

    # Always vectorized: enables jit_model_args for trace cache reuse
    chain_method_bench = "vectorized"
    print(f"\n  GPUs available:  {n_gpus}")
    print(f"  Fit chains:      {args.fit_chains}")
    print(f"  Chain method:    {chain_method_bench} (jit_model_args=True)")

    # --- Phase 1: JAX compilation cache test (BENCH-05) ---
    print("\nPhase 1: JAX compilation cache test (BENCH-05)...")
    print("  Building tiny cohort for JIT warm-up (N=2/group, 2 chains, 10 draws)...")

    d_bench = power_config.effect_size_grid[0]
    cfg_tiny = make_power_config(base_config, 2, d_bench, 99999)
    sim_tiny = simulate_batch(cfg_tiny)

    t0 = time.perf_counter()
    fit_batch_hierarchical(
        sim_tiny,
        "hgf_3level",
        n_chains=2,
        n_draws=10,
        n_tune=10,
        target_accept=0.9,
        random_seed=42,
        progressbar=False,
    )
    jit_cold_s = time.perf_counter() - t0
    results["jit_cold_s"] = round(jit_cold_s, 2)
    print(f"  Cold JIT: {jit_cold_s:.2f}s")

    t0 = time.perf_counter()
    fit_batch_hierarchical(
        sim_tiny,
        "hgf_3level",
        n_chains=2,
        n_draws=10,
        n_tune=10,
        target_accept=0.9,
        random_seed=43,
        progressbar=False,
    )
    jit_warm_s = time.perf_counter() - t0
    results["jit_warm_s"] = round(jit_warm_s, 2)
    print(f"  Warm JIT: {jit_warm_s:.2f}s")
    cache_speedup = jit_cold_s / max(jit_warm_s, 0.001)
    print(f"  Cache speedup: {cache_speedup:.1f}x")

    # --- Phase 2: Full batched iteration at max N (BENCH-01) ---
    max_n = max(power_config.n_per_group_grid)
    # 2 groups x max_n x 3 sessions = participant-sessions
    benchmark_n_participant_sessions = 2 * max_n * 3
    print(
        f"\nPhase 2: Full batched iteration (BENCH-01) at max N={max_n} "
        f"({benchmark_n_participant_sessions} participant-sessions × 2 models)..."
    )

    monitor = _GpuMonitor(interval_s=2.0)
    monitor.start()

    t0 = time.perf_counter()
    run_sbf_iteration(
        base_config=base_config,
        effect_size_delta=d_bench,
        iteration=0,
        child_seed=99999,
        n_per_group_grid=power_config.n_per_group_grid,
        power_config=power_config,
        n_chains=args.fit_chains,
        n_draws=args.fit_draws,
        n_tune=args.fit_tune,
        use_legacy=False,
    )
    per_iteration_s = time.perf_counter() - t0

    monitor.stop()

    results["per_iteration_s"] = round(per_iteration_s, 2)
    results["peak_vram_mb"] = monitor.peak_vram_mb
    results["mean_gpu_util_pct"] = round(monitor.mean_gpu_util_pct, 1)
    results["vram_total_mb"] = monitor.vram_total_mb
    results["benchmark_n_participant_sessions"] = benchmark_n_participant_sessions

    print(f"  Full iteration: {per_iteration_s:.2f}s")
    if monitor.samples:
        print(f"  Peak VRAM: {monitor.peak_vram_mb:.0f} MB")
        print(f"  Mean GPU util: {monitor.mean_gpu_util_pct:.1f}%")

    # --- Phase 3: Decision gate (BENCH-02) ---
    print("\nPhase 3: Decision gate (BENCH-02)...")
    gate_result = apply_decision_gate(per_iteration_s)
    results.update(gate_result)

    print(f"  GPU-hours/chunk: {gate_result['gpu_hours_per_chunk']:.1f}h")
    print(f"  Decision: {gate_result['decision'].upper()}")

    # --- Grid and MCMC settings metadata ---
    n_grid = power_config.n_per_group_grid
    d_grid = power_config.effect_size_grid
    n_iter = power_config.n_iterations
    results["grid"] = {
        "n_per_group": n_grid,
        "max_n": max_n,
        "effect_sizes": d_grid,
        "n_iterations": n_iter,
        "n_chunks": power_config.n_chunks,
    }
    results["mcmc_settings"] = {
        "chains": args.fit_chains,
        "draws": args.fit_draws,
        "tune": args.fit_tune,
        "sampler": "numpyro-direct",
    }

    # --- Write benchmark_batched.json ---
    bench_path = output_dir / "benchmark_batched.json"
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to: {bench_path}")

    # --- Append decision to STATE.md ---
    state_md_path = (
        Path(__file__).resolve().parent.parent / ".planning" / "STATE.md"
    )
    _update_state_md(
        state_md_path,
        gate_result["decision"],
        gate_result["gpu_hours_per_chunk"],
        per_iteration_s,
    )

    # --- Human-readable summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  GPU:                      {results.get('gpu_device', 'unknown')}")
    print(f"  VRAM total:               {results['vram_total_mb']:.0f} MB")
    print(f"  Peak VRAM (full iter):    {results['peak_vram_mb']:.0f} MB")
    print(f"  Mean GPU util:            {results['mean_gpu_util_pct']:.1f}%")
    print()
    print(f"  JIT cold (first call):    {results['jit_cold_s']:.2f}s")
    print(f"  JIT warm (second call):   {results['jit_warm_s']:.2f}s")
    print()
    print(f"  Full iteration (N={max_n}): {per_iteration_s:.2f}s")
    print(f"  Participant-sessions:      {benchmark_n_participant_sessions} x 2 models")
    print()
    print("DECISION GATE (BENCH-02):")
    print("  Formula: per_iter_s * 600 / 3600 > 50")
    print(f"  {per_iteration_s:.1f}s * 600 / 3600 = {gate_result['gpu_hours_per_chunk']:.1f} GPU-hrs/chunk")
    if gate_result["decision"] == "gpu":
        print(f"  {gate_result['gpu_hours_per_chunk']:.1f} <= 50 → RECOMMEND GPU (mgpu partition)")
    else:
        print(f"  {gate_result['gpu_hours_per_chunk']:.1f} > 50 → RECOMMEND CPU (comp partition)")
    print("=" * 60)


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

    if args.smoke_test:
        _run_smoke_test(base_config, power_config, output_dir, args)
        return

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
            use_legacy=args.legacy,
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
