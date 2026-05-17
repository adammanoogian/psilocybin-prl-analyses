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
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work on cluster
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Pre-parse --enable-x64 before importing jax so JAX_ENABLE_X64 takes effect.
# Config must be set before any jnp array is created, so env var is the
# reliable channel.
if "--enable-x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"

import jax

import config as _cfg
from prl_hgf.env.task_config import load_config
from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig
from prl_hgf.fitting.priors import HGFPriorSpec
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
        "--fit-config",
        type=Path,
        default=None,
        help=(
            "Path to a FitConfig YAML file (configs/fit/*.yaml). When "
            "provided, sampler settings (n_chains, n_draws, n_warmup, "
            "max_tree_depth) are loaded from the YAML and override the "
            "legacy --fit-chains/--fit-draws/--fit-tune/--max-tree-depth "
            "CLI args."
        ),
    )
    # Legacy CLI args — kept for backward compatibility.  When --fit-config
    # is provided these are ignored (YAML values take precedence).
    parser.add_argument(
        "--fit-chains",
        type=int,
        default=2,
        help="[DEPRECATED: use --fit-config] Number of MCMC chains.",
    )
    parser.add_argument(
        "--fit-draws",
        type=int,
        default=500,
        help="[DEPRECATED: use --fit-config] Posterior draws per chain.",
    )
    parser.add_argument(
        "--fit-tune",
        type=int,
        default=500,
        help="[DEPRECATED: use --fit-config] Tuning steps per chain.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help=(
            "[DEPRECATED: use --fit-config] NUTS max tree depth "
            "(BlackJAX `max_num_doublings`).  Default 10."
        ),
    )
    parser.add_argument(
        "--participant-chunk-id",
        type=int,
        default=0,
        help=(
            "Phase 14.2 variant 4: zero-based participant chunk index for "
            "SLURM-array benchmark mode.  Combined with "
            "--participant-chunk-count, slices the simulated cohort to its "
            "K-th 1/N-th of unique (participant_id, group, session) tuples.  "
            "Default 0 (with chunk-count 1 = full cohort, no-op)."
        ),
    )
    parser.add_argument(
        "--participant-chunk-count",
        type=int,
        default=1,
        help=(
            "Phase 14.2 variant 4: total number of participant chunks the "
            "cohort is split across in SLURM-array benchmark mode.  At the "
            "production cohort shape (300 PS), --participant-chunk-count=6 "
            "gives 50 PS per chunk, matching Phase 21's P=50 scan.  Default "
            "1 (no chunking — full cohort)."
        ),
    )
    parser.add_argument(
        "--laplace-warmup",
        action="store_true",
        default=False,
        help=(
            "Phase 14.2 variant 2: skip BlackJAX window_adaptation and "
            "instead initialize NUTS from a Laplace approximation — "
            "jaxopt LBFGS to find the MAP, Hessian diagonal at the MAP "
            "via vmapped jvp, regularize and invert to get the inverse "
            "mass matrix.  Falls back to standard window_adaptation if "
            "the Laplace step fails (LBFGS doesn't converge, Hessian has "
            "non-finite entries, etc.)."
        ),
    )
    parser.add_argument(
        "--tight-omega3-prior",
        action="store_true",
        default=False,
        help=(
            "Phase 14.2 variant 3: tighten the ω₃ prior to "
            "Normal(-6, 1.0) (default is TruncatedNormal(-6, 2, high=0)). "
            "Collapses the (μ₃, ω₃) funnel arm at extreme-negative ω₃ "
            "where σ₃→0.  Field-defensible — TAPAS / pyhgf treat ω₃ as "
            "poorly-identified and constrain it via prior; cf. CLAUDE.md "
            "'ω₃ recovery is known to be poor in the literature'."
        ),
    )
    parser.add_argument(
        "--laplace-only",
        action="store_true",
        default=False,
        help=(
            "Phase 14.2 variant 5: run only the Laplace approximation and "
            "skip the NUTS sampling phase entirely (NUTS still runs but "
            "with n_tune=1 / n_draws=1 so the wall is dominated by the "
            "Laplace LBFGS + Hessian work).  Implies --laplace-warmup; "
            "overrides --fit-tune and --fit-draws to 1.  Provides the "
            "'Laplace as supplement, not replacement' baseline against "
            "which full NUTS variants are compared."
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["pymc", "numpyro", "blackjax"],
        default="blackjax",
        help=(
            "Sampler for fit_batch_hierarchical. 'blackjax' is the "
            "Phase 17 production sampler (STATE decision 98); "
            "'numpyro' is the fallback; 'pymc' is deprecated and "
            "falls through to numpyro (STATE decision 102)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. Defaults to MODELS_DIR / 'power'. "
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
    parser.add_argument(
        "--n-per-group-overrides",
        type=str,
        default=None,
        help=(
            "Comma-separated cohort-size override that REPLACES "
            "power.n_per_group_grid from the YAML config for both Phase 1 "
            "warmup (uses max of override) and Phase 2 measurement loop. "
            "Example: --n-per-group-overrides '5,10,17,25' produces a "
            "P_total sweep at [n_groups*n_sessions]*[5,10,17,25]. Used "
            "for cliff-edge characterization (capability-map work). Empty "
            "or omitted = use config grid unchanged."
        ),
    )
    parser.add_argument(
        "--benchmark-model",
        type=str,
        choices=["hgf_2level", "hgf_3level"],
        default="hgf_3level",
        help=(
            "HGF model variant for the --benchmark path. Defaults to "
            "hgf_3level for back-compat with Phase 14/21 benchmark history. "
            "Use --benchmark-model hgf_2level to fill the 2-level NUTS gap "
            "in docs/CAPABILITY_MAP.md. Has no effect outside --benchmark."
        ),
    )
    parser.add_argument(
        "--benchmark-out-suffix",
        type=str,
        default="",
        help=(
            "Suffix appended to benchmark_batched.json (before extension), "
            "so concurrent --benchmark jobs writing different (model, grid) "
            "configurations don't clobber each other's results. Example: "
            "--benchmark-out-suffix '_hgf_2level' produces "
            "benchmark_batched_hgf_2level.json. Empty/omitted = use the "
            "canonical filename (back-compat with downstream consumers)."
        ),
    )
    parser.add_argument(
        "--enable-x64",
        action="store_true",
        default=False,
        help=(
            "Enable JAX float64 precision (sets JAX_ENABLE_X64=1 before "
            "jax import).  Diagnostic flag for the smoke test: lets us "
            "compare fp32 vs fp64 stability and runtime for the HGF + "
            "BlackJAX NUTS path."
        ),
    )
    # -----------------------------------------------------------------
    # Phase 21 diagnostic flags (BENCH-DIAG-04).  Gated behind
    # --diagnostic-mode / --p-scan / --vb-laplace-probe; production
    # --benchmark path is unaffected when these are omitted.
    # -----------------------------------------------------------------
    parser.add_argument(
        "--p-scan",
        type=int,
        default=None,
        help=(
            "Phase 21 diagnostic: run one cold+warm PAT-RL fit at P "
            "participants and append a row to .planning/phases/"
            "21-benchmark-bottleneck-diagnosis/jit_scaling_sweep.csv. "
            "Mutually exclusive with --benchmark."
        ),
    )
    parser.add_argument(
        "--p-scan-early-stop",
        action="store_true",
        default=False,
        help=(
            "Phase 21 diagnostic: mark the P-scan row status as "
            "'early_stopped' (used by the SLURM dependency chain when a "
            "prior P value has already produced a clear verdict)."
        ),
    )
    parser.add_argument(
        "--vb-laplace-probe",
        action="store_true",
        default=False,
        help=(
            "Phase 21 diagnostic: run fit_vb_laplace_patrl at --probe-p "
            "and write .planning/phases/21-benchmark-bottleneck-diagnosis/"
            "vb_laplace_vs_nuts_jit.json per-stage compile events + "
            "per-call wall-clock."
        ),
    )
    parser.add_argument(
        "--vb-laplace-subproc-only",
        action="store_true",
        default=False,
        help=(
            "Phase 21 diagnostic: run ONE fit_vb_laplace_patrl call at "
            "--probe-p and exit.  Used internally by --vb-laplace-probe "
            "to spawn a fresh subprocess for the next-proc persistent-"
            "cache wall-clock measurement.  Do not invoke manually."
        ),
    )
    parser.add_argument(
        "--probe-p",
        type=int,
        default=5,
        help="Participants for --vb-laplace-probe (default 5).",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        default=False,
        help=(
            "Enable Phase 21 diagnostic instrumentation in _run_benchmark "
            "(JAX_LOG_COMPILES=1, jax_explain_cache_misses=True, cache "
            "delta tracking)."
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
        return float(sum(s["gpu_util_pct"] for s in self.samples) / len(self.samples))

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
        insert_pos = header_pos + len("\n".join(lines[: last_table_line_idx + 1]))
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
        sum(f.stat().st_size for f in real_files) / (1024 * 1024),
        2,
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
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(parts[2]),
                    "vram_used_mb": int(parts[3]),
                    "vram_free_mb": int(parts[4]),
                    "gpu_util_pct": int(parts[5]),
                    "temp_c": int(parts[6]),
                    "pci_bus": parts[7],
                }
            )
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
    print(
        f"\n  {'GPU':<5} {'Name':<20} {'VRAM Used/Total':>18} {'Util':>6} {'Temp':>6}"
    )
    print(f"  {'-' * 5} {'-' * 20} {'-' * 18} {'-' * 6} {'-' * 6}")
    for g in gpus:
        print(
            f"  {g['index']:<5} {g['name']:<20} "
            f"{g['vram_used_mb']:>6}/{g['vram_total_mb']:<6} MB "
            f"{g['gpu_util_pct']:>4}% "
            f"{g['temp_c']:>4}C"
        )


def _summarize_nuts_stats(
    idata: object,
    label: str,
) -> dict:
    """Print and return aggregate NUTS diagnostics from an idata object.

    Reports per-draw integrator-step counts, divergence rate, and mean
    acceptance.  When ``num_integration_steps`` saturates at the BlackJAX
    default cap (1024 = 2**10 doublings), this is the signature of a
    step size too small for the posterior curvature (the primary suspect
    when warm sampling is slow despite a JIT-cache hit).

    Parameters
    ----------
    idata : arviz.InferenceData
        Must have ``sample_stats`` with ``num_integration_steps``,
        ``acceptance_rate``, ``diverging``.
    label : str
        Short label printed next to the stats (``"cold"``, ``"warm1"``,
        ``"warm2"``).

    Returns
    -------
    dict
        Numeric summary fields suitable for the smoke_test.json blob.
    """
    import numpy as np

    stats = idata.sample_stats
    steps = np.asarray(stats["num_integration_steps"]).ravel()
    accept = np.asarray(stats["acceptance_rate"]).ravel()
    diverging = np.asarray(stats["diverging"]).ravel()
    expansions = (
        np.asarray(stats["num_trajectory_expansions"]).ravel()
        if "num_trajectory_expansions" in stats
        else None
    )

    n_draws_total = steps.size
    summary = {
        "n_draws_total": int(n_draws_total),
        "integration_steps_mean": float(np.mean(steps)),
        "integration_steps_p50": float(np.percentile(steps, 50)),
        "integration_steps_p95": float(np.percentile(steps, 95)),
        "integration_steps_max": int(np.max(steps)),
        "acceptance_mean": float(np.mean(accept)),
        "divergence_rate": float(np.mean(diverging)),
    }
    if expansions is not None:
        summary["trajectory_expansions_max"] = int(np.max(expansions))
        summary["trajectory_expansions_mean"] = float(np.mean(expansions))

    print(f"  NUTS stats ({label}):")
    print(
        f"    integration_steps: mean={summary['integration_steps_mean']:.0f} "
        f"p50={summary['integration_steps_p50']:.0f} "
        f"p95={summary['integration_steps_p95']:.0f} "
        f"max={summary['integration_steps_max']}"
    )
    if expansions is not None:
        print(
            f"    trajectory_expansions: mean="
            f"{summary['trajectory_expansions_mean']:.1f} "
            f"max={summary['trajectory_expansions_max']} "
            "(10 = hit max_tree_depth cap)"
        )
    print(
        f"    acceptance_rate: mean={summary['acceptance_mean']:.3f}  "
        f"divergence_rate: {summary['divergence_rate']:.3f}"
    )
    return summary


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

    # Enable cache miss diagnostics (logs why tracing cache misses).
    # Gated behind PRL_EXPLAIN_CACHE_MISSES because JAX 0.9.2 has an
    # upstream bug in partial_eval.diff_tracing_cache_keys that triggers
    # a misleading 3-vs-2 unpack ValueError inside lax.scan bodies that
    # contain a nested JIT'd function (e.g. pyhgf's scan_fn).  Worked
    # under JAX 0.4.x; opt-in only under ds_env_v10 (jax 0.9.2).
    if os.environ.get("PRL_EXPLAIN_CACHE_MISSES", "0") == "1":
        jax.config.update("jax_explain_cache_misses", True)

    # PyTensor's jax link silently flips jax_enable_x64 to True when it's
    # imported (via prl_hgf.fitting.hierarchical → jax_funcify).  If the
    # user did NOT pass --enable-x64, force it back to False before any
    # jnp array is created so the sampler actually runs in fp32 for the
    # precision comparison.  When --enable-x64 was passed, the env var
    # preparse at the top of the module already set it; no-op here.
    if not getattr(args, "enable_x64", False):
        jax.config.update("jax_enable_x64", False)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    # Defaults match the original JIT-cache smoke test (cheap: 10 tune / 10
    # draws).  Override via env for adaptation diagnostics — e.g., set
    # PRL_SMOKE_TUNE=500 PRL_SMOKE_DRAWS=500 to match the --fit-tune /
    # --fit-draws production argparse defaults and see whether
    # window_adaptation finds a step size that gets off the
    # max_tree_depth cap.
    n_smoke = int(os.environ.get("PRL_SMOKE_N_PER_GROUP", "5"))
    n_chains_smoke = int(os.environ.get("PRL_SMOKE_CHAINS", "2"))
    n_draws_smoke = int(os.environ.get("PRL_SMOKE_DRAWS", "10"))
    n_tune_smoke = int(os.environ.get("PRL_SMOKE_TUNE", "10"))
    # Periodic progress logging: emit NUTS diagnostics every PRL_SMOKE_LOG_EVERY
    # draws during sampling.  0 disables (default).  Intended for long runs
    # where we want to kill the job early once takeaway is clear.
    log_every_smoke = int(os.environ.get("PRL_SMOKE_LOG_EVERY", "0"))
    # Model selection: default to hgf_2level (primary target; no ω₃×κ
    # ridge).  Set PRL_SMOKE_MODEL=hgf_3level to exercise the κ-frozen
    # 3-level geometry.  Set to "both" to run 2-level THEN 3-level in
    # the same SLURM job so we get both diagnostics in one cluster call.
    model_smoke = os.environ.get("PRL_SMOKE_MODEL", "hgf_2level")
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
    print("\n  JAX cache (before):")
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

    if model_smoke == "both":
        msg = (
            "PRL_SMOKE_MODEL=both is handled at the SLURM level by running "
            "two separate invocations.  Set PRL_SMOKE_MODEL to either "
            "'hgf_2level' or 'hgf_3level' for a single-model smoke."
        )
        raise ValueError(msg)
    if model_smoke not in ("hgf_2level", "hgf_3level"):
        msg = (
            f"PRL_SMOKE_MODEL must be 'hgf_2level' or 'hgf_3level', got {model_smoke!r}"
        )
        raise ValueError(msg)
    results["model_name"] = model_smoke
    results["log_every"] = log_every_smoke
    # κ frozen at 1.0 (see prl_hgf.fitting.hierarchical._KAPPA_FIXED).  For
    # the 3-level smoke this is the geometry fix — logs should show
    # integration_steps dropping below the max_tree_depth cap.
    results["kappa_frozen_at"] = 1.0

    print("\nSmoke test config:")
    print(f"  N per group:            {n_smoke}")
    print(f"  Participant-sessions:   {n_participant_sessions}")
    print(f"  Chains:                 {n_chains_smoke}")
    print(f"  Draws/tune:             {n_draws_smoke}/{n_tune_smoke}")
    print(f"  Model:                  {model_smoke}")
    print("  κ frozen at:            1.0 (geometry fix — see _KAPPA_FIXED)")
    print(f"  log_every:              {log_every_smoke} (0 = off)")
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

    # Build FitConfig for smoke test
    _smoke_fit_config = FitConfig(
        model_name=model_smoke,
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=n_chains_smoke,
            n_draws=n_draws_smoke,
            n_warmup=n_tune_smoke,
            target_accept=0.9,
            random_seed=42,
            max_tree_depth=args.max_tree_depth,
        ),
        mitigation=MitigationConfig(
            use_laplace_warmup=args.laplace_warmup,
        ),
        log_every=log_every_smoke,
        progressbar=False,
    )
    _smoke_prior_spec = (
        HGFPriorSpec.tight_3level() if args.tight_omega3_prior else None
    )

    t0 = time.perf_counter()
    # Cold call: full warmup, returns (idata, adapted_params)
    idata_cold, adapted_params = fit_batch_hierarchical(
        sim_smoke,
        _smoke_fit_config,
        prior_spec=_smoke_prior_spec,
    )
    jit_cold_s = time.perf_counter() - t0
    results["jit_cold_s"] = round(jit_cold_s, 2)
    print(f"  Cold JIT: {jit_cold_s:.2f}s")
    results["nuts_stats_cold"] = _summarize_nuts_stats(idata_cold, "cold")

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
    # Warm call 1: skip warmup by reusing adapted params from cold call.
    # Still incurs Python-level retrace of sample_loop_vmap (new closure),
    # but XLA compile should hit the persistent on-disk cache.
    import dataclasses as _dc

    _warm1_config = _dc.replace(
        _smoke_fit_config,
        sampler=_dc.replace(_smoke_fit_config.sampler, random_seed=43),
    )
    idata_warm1 = fit_batch_hierarchical(
        sim_smoke_2,
        _warm1_config,
        prior_spec=_smoke_prior_spec,
        warmup_params=adapted_params,
    )
    jit_warm_s = time.perf_counter() - t0
    results["jit_warm_s"] = round(jit_warm_s, 2)
    print(f"  Warm JIT: {jit_warm_s:.2f}s (warmup skipped via warmup_params)")
    results["nuts_stats_warm1"] = _summarize_nuts_stats(idata_warm1, "warm1")

    # --- Step 3b: Warm JIT call #2 (compile vs execute split) ---
    # Second warm call with yet another seed.  If XLA trace cache is
    # healthy, this should be much faster than warm1 (pure execute, no
    # Python retrace overhead).  If warm2 ≈ warm1, sampling itself is
    # the bottleneck; if warm2 << warm1, tracing/compile dominates.
    print("\nStep 3b: Warm JIT #2 (pure execute — trace cache reuse)...")
    _warm2_config = _dc.replace(
        _smoke_fit_config,
        sampler=_dc.replace(_smoke_fit_config.sampler, random_seed=44),
    )
    t0 = time.perf_counter()
    idata_warm2 = fit_batch_hierarchical(
        sim_smoke_2,
        _warm2_config,
        prior_spec=_smoke_prior_spec,
        warmup_params=adapted_params,
    )
    jit_warm2_s = time.perf_counter() - t0
    results["jit_warm2_s"] = round(jit_warm2_s, 2)
    print(f"  Warm JIT #2: {jit_warm2_s:.2f}s")
    results["nuts_stats_warm2"] = _summarize_nuts_stats(idata_warm2, "warm2")

    warm1_minus_warm2 = jit_warm_s - jit_warm2_s
    results["warm1_minus_warm2_s"] = round(warm1_minus_warm2, 2)
    print(
        f"  warm1 - warm2 = {warm1_minus_warm2:.2f}s "
        "(estimate of Python retrace + XLA compile cost per call)"
    )
    print("  warm2 ≈ pure sampling execute time")

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
    warm_new_files = cache_after_warm["n_files"] - cache_after_cold["n_files"]
    warm_new_mb = cache_after_warm["total_size_mb"] - cache_after_cold["total_size_mb"]
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
    print(
        f"  Cache speedup > 3x: {'PASS' if cache_ok else 'FAIL'} ({cache_speedup:.1f}x)"
    )
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
                    g["vram_used_mb"] / g["vram_total_mb"] * 100 for g in gpus_post_warm
                )
                if max_vram_pct > 90:
                    print(
                        f"       WARNING: VRAM at {max_vram_pct:.0f}% — "
                        "memory pressure likely causing slowdown."
                    )

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
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "env_overrides": {
            k: os.environ[k]
            for k in (
                "PRL_SMOKE_TUNE",
                "PRL_SMOKE_DRAWS",
                "PRL_SMOKE_CHAINS",
                "PRL_SMOKE_N_PER_GROUP",
            )
            if k in os.environ
        },
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

    # Phase 21 diagnostic instrumentation (BENCH-DIAG-04).
    # Only activate when --diagnostic-mode is set OR JAX_LOG_COMPILES is
    # already set externally (SLURM-level env injection).  In production
    # benchmark mode without the flag, behaviour is unchanged — the env
    # var and jax.config update are no-ops when not requested.
    if (
        getattr(args, "diagnostic_mode", False)
        or os.environ.get("JAX_LOG_COMPILES") == "1"
    ):
        os.environ.setdefault("JAX_LOG_COMPILES", "1")
        # See note above: gated behind PRL_EXPLAIN_CACHE_MISSES to avoid
        # JAX 0.9.2 partial_eval.diff_tracing_cache_keys bug.
        if os.environ.get("PRL_EXPLAIN_CACHE_MISSES", "0") == "1":
            jax.config.update("jax_explain_cache_misses", True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=" * 60)
    print("BENCHMARK MODE (batched hierarchical path)")
    print("=" * 60)

    # --- Phase 0: GPU device info ---
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
    print(
        f"\n  JAX cache: {cache_before['n_files']} files, "
        f"{cache_before['total_size_mb']} MB in {cache_dir}"
    )

    # Always vectorized: enables jit_model_args for trace cache reuse
    chain_method_bench = "vectorized"
    print(f"\n  GPUs available:  {n_gpus}")
    print(f"  Fit chains:      {args.fit_chains}")
    print(f"  Chain method:    {chain_method_bench} (jit_model_args=True)")

    # --- Phase 1: JAX compilation cache test (BENCH-05) ---
    # Phase 21 VERDICT §c secondary patch (14.1-07): the former N=2/group warm-up
    # exercised a structurally different HLO scan body (hash 2f25aed94588c656,
    # 4,771,952 B) from the Phase-2 N=max_n/group measurement (hash
    # 62fbc95475cb0a32, 4,786,267 B; Jaccard 0.180 per 21-06 §c). The tracing
    # cache could not bridge the shape gap so Phase 1 contributed compile time
    # without accelerating Phase 2. Upgrading Phase 1 to production N means the
    # in-process tracing cache hits cleanly and Phase 2 runs warm-JIT-only.
    print("\nPhase 1: JAX compilation cache test (BENCH-05)...")
    print(
        "  Building production-shape cohort for JIT warm-up "
        "(N=max(n_per_group_grid), args.fit_chains chains, "
        "args.fit_tune tune / args.fit_draws draws)..."
    )

    d_bench = power_config.effect_size_grid[0]
    max_n_warm = max(power_config.n_per_group_grid)
    cfg_warm = make_power_config(base_config, max_n_warm, d_bench, 99999)
    sim_warm = simulate_batch(cfg_warm)

    # Variant 4: optionally subset the cohort to one chunk before fitting.
    # The simulation runs at full N first so chunk selection is deterministic
    # and reproducible regardless of chunk_count (same RNG-seeded cohort).
    if args.participant_chunk_count > 1:
        from prl_hgf.power.iteration import subset_cohort_by_chunk

        n_total_ps = (
            sim_warm[["participant_id", "group", "session"]].drop_duplicates().shape[0]
        )
        sim_warm = subset_cohort_by_chunk(
            sim_warm,
            args.participant_chunk_id,
            args.participant_chunk_count,
        )
        n_chunk_ps = (
            sim_warm[["participant_id", "group", "session"]].drop_duplicates().shape[0]
        )
        print(
            f"Variant 4 chunking: chunk {args.participant_chunk_id} of "
            f"{args.participant_chunk_count} -> {n_chunk_ps} of {n_total_ps} "
            "participant-sessions",
            flush=True,
        )
        results["participant_chunk_id"] = args.participant_chunk_id
        results["participant_chunk_count"] = args.participant_chunk_count
        results["chunk_n_participant_sessions"] = n_chunk_ps
        results["cohort_n_participant_sessions_full"] = n_total_ps

    # Build FitConfig for benchmark cold/warm calls
    import dataclasses as _dc

    _bench_fit_config = FitConfig(
        model_name=args.benchmark_model,
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=args.fit_chains,
            n_draws=args.fit_draws,
            n_warmup=args.fit_tune,
            target_accept=0.9,
            random_seed=42,
            max_tree_depth=args.max_tree_depth,
        ),
        mitigation=MitigationConfig(
            use_laplace_warmup=args.laplace_warmup,
        ),
        progressbar=False,
    )
    _bench_prior_spec = (
        HGFPriorSpec.tight_3level() if args.tight_omega3_prior else None
    )

    t0 = time.perf_counter()
    # Cold call: returns (idata, adapted_params) when warmup_params is None
    # (Phase 21 Patch C: capture adapted params so the warm call can skip
    # window adaptation and we get a real cold-vs-warm comparison).
    _cold_return = fit_batch_hierarchical(
        sim_warm,
        _bench_fit_config,
        prior_spec=_bench_prior_spec,
    )
    if isinstance(_cold_return, tuple):
        _idata_cold, adapted_params = _cold_return
    else:
        # Fallback: older signatures returned just an idata.  Pass None
        # through and the warm call will re-run warmup.
        _idata_cold = _cold_return
        adapted_params = None
    jit_cold_s = time.perf_counter() - t0
    results["jit_cold_s"] = round(jit_cold_s, 2)
    print(f"  Cold JIT: {jit_cold_s:.2f}s")

    # Phase 21 Patch B: cache stats AFTER cold call (parity with
    # _run_smoke_test).  Delta fields let downstream readers know how many
    # cache entries the cold compile produced.
    cache_after_cold = _get_cache_stats(cache_dir)
    results["cache_after_cold"] = cache_after_cold
    results["cache_delta_cold_n_files"] = (
        cache_after_cold["n_files"] - cache_before["n_files"]
    )
    results["cache_delta_cold_mb"] = round(
        cache_after_cold["total_size_mb"] - cache_before["total_size_mb"],
        2,
    )
    print(
        f"  Cache delta (cold): +{results['cache_delta_cold_n_files']} files, "
        f"+{results['cache_delta_cold_mb']:.1f} MB"
    )

    t0 = time.perf_counter()
    # Phase 21 Patch C: pass warmup_params to the warm call so it skips
    # NUTS window adaptation.  This eliminates the ~1100s warmup-duplication
    # confound observed in job-54899271 (1.1x speedup was because both cold
    # and warm re-ran warmup, not because the XLA cache was missing).
    _bench_warm_config = _dc.replace(
        _bench_fit_config,
        sampler=_dc.replace(_bench_fit_config.sampler, random_seed=43),
    )
    fit_batch_hierarchical(
        sim_warm,
        _bench_warm_config,
        prior_spec=_bench_prior_spec,
        warmup_params=adapted_params,
    )
    jit_warm_s = time.perf_counter() - t0
    results["jit_warm_s"] = round(jit_warm_s, 2)
    print(f"  Warm JIT: {jit_warm_s:.2f}s (warmup skipped via warmup_params)")
    cache_speedup = jit_cold_s / max(jit_warm_s, 0.001)
    print(f"  Cache speedup: {cache_speedup:.1f}x")

    # Phase 21 Patch B: cache stats AFTER warm call.  If new cache files
    # appeared, the warm call hit a cache miss (HLO changed between cold
    # and warm — closure-over-data signature drift, retrace, etc.).
    cache_after_warm = _get_cache_stats(cache_dir)
    results["cache_after_warm"] = cache_after_warm
    results["cache_delta_warm_n_files"] = (
        cache_after_warm["n_files"] - cache_after_cold["n_files"]
    )
    results["cache_delta_warm_mb"] = round(
        cache_after_warm["total_size_mb"] - cache_after_cold["total_size_mb"],
        2,
    )
    print(
        f"  Cache delta (warm): +{results['cache_delta_warm_n_files']} files, "
        f"+{results['cache_delta_warm_mb']:.1f} MB"
    )

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

    # Build FitConfig for the full iteration call
    _iter_fit_config = FitConfig(
        model_name=args.benchmark_model,
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=args.fit_chains,
            n_draws=args.fit_draws,
            n_warmup=args.fit_tune,
            target_accept=0.9,
            random_seed=99999,
            max_tree_depth=args.max_tree_depth,
        ),
        mitigation=MitigationConfig(
            use_laplace_warmup=args.laplace_warmup,
        ),
        progressbar=False,
    )

    t0 = time.perf_counter()
    run_sbf_iteration(
        base_config=base_config,
        effect_size_delta=d_bench,
        iteration=0,
        child_seed=99999,
        n_per_group_grid=power_config.n_per_group_grid,
        power_config=power_config,
        fit_config=_iter_fit_config,
        use_legacy=False,
        participant_chunk_id=args.participant_chunk_id,
        participant_chunk_count=args.participant_chunk_count,
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
    # Variant 4: when chunked, encode chunk in the filename so SLURM-array
    # tasks don't clobber each other.  Un-chunked path keeps the original
    # filename for backwards compatibility with downstream consumers.
    # Capability-map: --benchmark-out-suffix lets concurrent jobs writing
    # different (model, grid) configurations differentiate without colliding.
    _out_suffix = args.benchmark_out_suffix or ""
    if args.participant_chunk_count > 1:
        _chunk_tag = (
            f"_chunk_{args.participant_chunk_id:02d}"
            f"_of_{args.participant_chunk_count:02d}"
        )
        bench_path = output_dir / f"benchmark_batched{_chunk_tag}{_out_suffix}.json"
    else:
        bench_path = output_dir / f"benchmark_batched{_out_suffix}.json"
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to: {bench_path}")

    # --- Append decision to STATE.md ---
    state_md_path = Path(__file__).resolve().parent.parent.parent / ".planning" / "STATE.md"
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
    print(
        f"  {per_iteration_s:.1f}s * 600 / 3600 = {gate_result['gpu_hours_per_chunk']:.1f} GPU-hrs/chunk"
    )
    if gate_result["decision"] == "gpu":
        print(
            f"  {gate_result['gpu_hours_per_chunk']:.1f} <= 50 → RECOMMEND GPU (mgpu partition)"
        )
    else:
        print(
            f"  {gate_result['gpu_hours_per_chunk']:.1f} > 50 → RECOMMEND CPU (comp partition)"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Phase 21 diagnostic probe helpers (BENCH-DIAG-04).
# ---------------------------------------------------------------------------


def _run_p_scan_probe(
    base_config: object,
    power_config: object,
    output_dir: Path,
    P: int,
    args: argparse.Namespace,
) -> None:
    """Phase 21 diagnostic: single-P cold+warm PAT-RL JIT timing probe.

    Runs ``fit_batch_hierarchical_patrl`` twice in one process at N=P
    participants (hgf_3level_patrl, model_a, n_chains chains, n_draws
    draws, n_tune tune) and appends one row to
    ``.planning/phases/21-benchmark-bottleneck-diagnosis/jit_scaling_sweep.csv``.

    Named ``n_chains`` and ``n_draws`` constants at the top of the body
    ensure the ``draws_per_s`` computation stays consistent with the fit
    kwargs (M2: no hardcoded ``2 * 10``).

    Parameters
    ----------
    base_config : object
        Unused (pick_best_cue config); kept for signature parity with
        the other dispatched helpers.  The PAT-RL path loads its own
        config via ``load_pat_rl_config``.
    power_config : object
        Unused (pick_best_cue grid); kept for signature parity.
    output_dir : Path
        Unused (the probe writes to a fixed Phase 21 path); kept for
        signature parity.
    P : int
        Number of PAT-RL participants to simulate for the probe.
    args : argparse.Namespace
        Parsed CLI args (provides ``p_scan_early_stop`` flag).

    Notes
    -----
    - ``next_proc_warm_s`` column is written empty; filled by a different
      SLURM job per plan 21-05's submission chain.
    - ``status`` enum: ``completed | early_stopped | infeasible``.
    - Exceptions during the fit are caught so an infeasible status is
      recorded (a cold-compile OOM at P=50 should not block the CSV
      from gaining a row).
    """
    os.environ.setdefault("JAX_LOG_COMPILES", "1")
    # Gated behind PRL_EXPLAIN_CACHE_MISSES — JAX 0.9.2 partial_eval bug.
    if os.environ.get("PRL_EXPLAIN_CACHE_MISSES", "0") == "1":
        jax.config.update("jax_explain_cache_misses", True)

    from prl_hgf.env.pat_rl_config import load_pat_rl_config
    from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort
    from prl_hgf.fitting.hierarchical_patrl import (
        fit_batch_hierarchical_patrl,
    )

    # Probe config constants (M2: these must track the n_chains * n_draws
    # passed to fit_batch_hierarchical_patrl below; draws_per_s uses
    # n_total_draws).
    n_chains = 2
    n_draws = 10
    n_tune = 10
    n_total_draws = n_chains * n_draws

    csv_path = Path(
        ".planning/phases/21-benchmark-bottleneck-diagnosis/jit_scaling_sweep.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    patrl_config = load_pat_rl_config()
    # Spread P across as many phenotypes as possible (min 1 per phenotype).
    n_per_phenotype = max(1, P // 4)
    sim_df, _true_params, _trials = simulate_patrl_cohort(
        n_participants=n_per_phenotype,
        config=patrl_config,
        master_seed=99999,
    )
    # Trim to exactly P participants for shape fidelity.
    pids = sim_df["participant_id"].unique()[:P]
    sim_df = sim_df[sim_df["participant_id"].isin(pids)].reset_index(drop=True)

    status = "completed"
    vram_peak_mb = 0.0
    cold_s = float("nan")
    same_proc_warm_s = float("nan")
    try:
        t0 = time.perf_counter()
        fit_batch_hierarchical_patrl(
            sim_df,
            model_name="hgf_3level_patrl",
            response_model="model_a",
            config=patrl_config,
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=0.9,
            random_seed=42,
        )
        cold_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        fit_batch_hierarchical_patrl(
            sim_df,
            model_name="hgf_3level_patrl",
            response_model="model_a",
            config=patrl_config,
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=0.9,
            random_seed=43,
        )
        same_proc_warm_s = time.perf_counter() - t0

        gpus = _query_gpu_table()
        if gpus:
            vram_peak_mb = float(gpus[0].get("vram_used_mb", 0.0))

        if args.p_scan_early_stop:
            status = "early_stopped"
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: --p-scan {P} fit failed: {exc!r}",
            file=sys.stderr,
        )
        status = "infeasible"

    if status == "completed" and same_proc_warm_s > 0:
        draws_per_s = n_total_draws / same_proc_warm_s
    else:
        draws_per_s = float("nan")

    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "P,cold_jit_s,same_proc_warm_s,next_proc_warm_s,"
                "draws_per_s,vram_peak_mb,status\n"
            )
        f.write(
            f"{P},{cold_s:.2f},{same_proc_warm_s:.2f},,"
            f"{draws_per_s:.4f},{vram_peak_mb:.1f},{status}\n"
        )

    print(
        f"--p-scan {P} complete: status={status}, cold={cold_s:.1f}s, "
        f"warm={same_proc_warm_s:.1f}s, row appended to {csv_path}"
    )


def _run_vb_laplace_subproc_one_shot(P: int) -> None:
    """Phase 21 diagnostic: one-shot ``fit_vb_laplace_patrl`` call + exit.

    Invoked by :func:`_run_vb_laplace_probe` as a fresh subprocess to
    measure the next-proc persistent-cache wall-clock time.  Does NOT
    write to the shared ``vb_laplace_vs_nuts_jit.json``; only performs
    one fit so the parent's ``subprocess.run()`` wall-clock is the
    ``next_proc_s`` measurement.

    Parameters
    ----------
    P : int
        Number of PAT-RL participants to simulate (same shape as the
        parent call so the XLA persistent cache can land a hit).
    """
    os.environ.setdefault("JAX_LOG_COMPILES", "1")
    # Gated behind PRL_EXPLAIN_CACHE_MISSES — JAX 0.9.2 partial_eval bug.
    if os.environ.get("PRL_EXPLAIN_CACHE_MISSES", "0") == "1":
        jax.config.update("jax_explain_cache_misses", True)

    from prl_hgf.env.pat_rl_config import load_pat_rl_config
    from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort
    from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl

    patrl_config = load_pat_rl_config()
    n_per_phenotype = max(1, P // 4)
    sim_df, _true_params, _trials = simulate_patrl_cohort(
        n_participants=n_per_phenotype,
        config=patrl_config,
        master_seed=99999,
    )
    pids = sim_df["participant_id"].unique()[:P]
    sim_df = sim_df[sim_df["participant_id"].isin(pids)].reset_index(drop=True)

    fit_vb_laplace_patrl(
        sim_df,
        model_name="hgf_3level_patrl",
        response_model="model_a",
        config=patrl_config,
        random_seed=42,
    )
    print(f"--vb-laplace-subproc-only P={P} complete")


def _run_vb_laplace_probe(
    base_config: object,
    power_config: object,
    output_dir: Path,
    P: int,
    args: argparse.Namespace,
) -> None:
    """Phase 21 diagnostic: VB-Laplace JIT timing probe.

    Runs ``fit_vb_laplace_patrl`` end-to-end at P participants
    (``hgf_3level_patrl``, ``model_a``).  Captures ``JAX_LOG_COMPILES``
    event COUNTS (not timings — B4: the originally-proposed
    ``per_stage_compile_X`` (timings) field was replaced with
    ``per_stage_compile_events`` because JAX_LOG_COMPILES does not
    emit timings).  Captures full-pipeline
    wall-clock via ``time.perf_counter`` wrappers around three calls
    (cold, in-process warm, subprocess-spawned next-proc) into the
    ``per_call_wall_clock`` field.  Writes/appends to
    ``.planning/phases/21-benchmark-bottleneck-diagnosis/vb_laplace_vs_nuts_jit.json``.

    Parameters
    ----------
    base_config, power_config, output_dir : object
        Unused — kept for signature parity with the other probe helpers.
    P : int
        Number of PAT-RL participants to simulate for the probe.
    args : argparse.Namespace
        Parsed CLI args (unused in the current body but kept for parity).

    Notes
    -----
    - Captures ``JAX_LOG_COMPILES`` output in-process via a ``StringIO``
      ``logging`` handler (robust to subprocess details).
    - Fills ``per_stage_compile_events`` (counts),
      ``per_call_wall_clock`` (timings), and ``per_p_scaling``
      (aggregate) slices for the invoked P.  The ``bottleneck_verdict``
      and ``hlo_op_counts`` keys remain undetermined/empty; plan 21-07
      fills them at synthesis time.
    - Spawns a subprocess via :func:`_run_vb_laplace_subproc_one_shot`
      (wired through the ``--vb-laplace-subproc-only`` flag) so the
      next-proc wall-clock actually crosses a fresh Python invocation —
      exercises the persistent compilation cache.
    - Archives the raw log per-P for reproducibility.
    """
    del args  # unused — kept for signature parity

    import logging
    import re
    from io import StringIO

    os.environ.setdefault("JAX_LOG_COMPILES", "1")
    # Gated behind PRL_EXPLAIN_CACHE_MISSES — JAX 0.9.2 partial_eval bug.
    if os.environ.get("PRL_EXPLAIN_CACHE_MISSES", "0") == "1":
        jax.config.update("jax_explain_cache_misses", True)

    # Capture JAX logger output for the in-process calls.
    log_buffer = StringIO()
    jax_logger = logging.getLogger("jax")
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.WARNING)
    jax_logger.addHandler(handler)
    jax_logger.setLevel(logging.WARNING)

    from prl_hgf.env.pat_rl_config import load_pat_rl_config
    from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort
    from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl

    json_path = Path(
        ".planning/phases/21-benchmark-bottleneck-diagnosis/vb_laplace_vs_nuts_jit.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    patrl_config = load_pat_rl_config()
    n_per_phenotype = max(1, P // 4)
    sim_df, _true_params, _trials = simulate_patrl_cohort(
        n_participants=n_per_phenotype,
        config=patrl_config,
        master_seed=99999,
    )
    pids = sim_df["participant_id"].unique()[:P]
    sim_df = sim_df[sim_df["participant_id"].isin(pids)].reset_index(drop=True)

    # Call 1: cold (full-pipeline wall-clock).
    t0 = time.perf_counter()
    fit_vb_laplace_patrl(
        sim_df,
        model_name="hgf_3level_patrl",
        response_model="model_a",
        config=patrl_config,
        random_seed=42,
    )
    cold_s = time.perf_counter() - t0

    # Call 2: in-process warm (second call at same shape).
    t0 = time.perf_counter()
    fit_vb_laplace_patrl(
        sim_df,
        model_name="hgf_3level_patrl",
        response_model="model_a",
        config=patrl_config,
        random_seed=43,
    )
    warm_s = time.perf_counter() - t0

    jax_logger.removeHandler(handler)
    log_text = log_buffer.getvalue()

    # Call 3: subprocess-spawned next-proc warm (persistent-cache landing).
    # Reuses JAX_COMPILATION_CACHE_DIR from the parent env so a warm hit
    # is possible.  Time only the subprocess.run wall-clock; the child's
    # own internal timing is not needed.
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--vb-laplace-subproc-only",
                "--probe-p",
                str(P),
                # --chunk-id and --job-id are required args in parse_args,
                # so pass sentinel values that will never run production
                # logic (the subproc flag short-circuits main()).
                "--chunk-id",
                "0",
                "--job-id",
                "vb-laplace-subproc",
            ],
            check=True,
            timeout=1800,  # 30min safety cap (matches SC7)
        )
        next_proc_s = time.perf_counter() - t0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(
            f"WARNING: vb-laplace next-proc subproc failed: {exc!r}",
            file=sys.stderr,
        )
        next_proc_s = float("nan")

    # Parse JAX_LOG_COMPILES event counts (NOT timings — B4 rationale).
    # Message format: "Compiling <function_name> for args ..."
    compile_events = re.findall(r"Compiling (\S+) for args", log_text)
    cache_hits = re.findall(
        r"Persistent compilation cache hit for '(\S+)'",
        log_text,
    )
    cache_misses = re.findall(
        r"PERSISTENT COMPILATION CACHE MISS for '(\S+)'",
        log_text,
    )

    # Aggregate by stage: LBFGS body = any event with "log_posterior" or
    # "_clamped_step" or "logp_fn" in name; Hessian = any event with
    # "hessian" or "jvp" in name; NUTS kernel = N/A for this probe (plan
    # 21-04 captures that).
    def _matches_stage(name: str, keywords: list[str]) -> bool:
        return any(kw in name for kw in keywords)

    lbfgs_events = [
        e
        for e in compile_events
        if _matches_stage(e, ["log_posterior", "_clamped_step", "logp_fn"])
    ]
    hessian_events = [
        e for e in compile_events if _matches_stage(e, ["hessian", "jvp"])
    ]

    # Load existing JSON if present, else initialize with the 5-key
    # schema (B4: per_stage_compile_events + per_call_wall_clock replace
    # the originally-proposed single per-stage-timings field).
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("per_stage_compile_events", {})
        data.setdefault("per_call_wall_clock", {})
        data.setdefault("per_p_scaling", {})
        data.setdefault(
            "bottleneck_verdict",
            {
                "bottleneck_layer": "undetermined",
                "evidence": ("Filled by plan 21-07 after all probes complete."),
            },
        )
        data.setdefault("hlo_op_counts", {})
    else:
        data = {
            "per_stage_compile_events": {},
            "per_call_wall_clock": {},
            "per_p_scaling": {},
            "bottleneck_verdict": {
                "bottleneck_layer": "undetermined",
                "evidence": ("Filled by plan 21-07 after all probes complete."),
            },
            "hlo_op_counts": {},
        }

    # Fill per_stage_compile_events for this P (counts, not timings).
    data["per_stage_compile_events"][str(P)] = {
        "lbfgs_n_events": len(lbfgs_events),
        "hessian_n_events": len(hessian_events),
        "total_n_events": len(compile_events),
        "notes": (
            "JAX_LOG_COMPILES emits event counts, not timings; "
            "per-stage wall-clock timing would require harness-splitting "
            "which CONTEXT explicitly rejects.  See per_call_wall_clock "
            "for full-pipeline wall-clock times."
        ),
    }

    # Fill per_call_wall_clock for this P (full-pipeline timings).
    data["per_call_wall_clock"][str(P)] = {
        "cold_s": round(cold_s, 2),
        "warm_s": round(warm_s, 2),
        "next_proc_s": (
            round(next_proc_s, 2)
            if next_proc_s == next_proc_s  # NaN check
            else None
        ),
    }

    # Fill per_p_scaling for this P (aggregate slice).
    data["per_p_scaling"][str(P)] = {
        "total_s": round(cold_s + warm_s, 2),
        "n_compile_events": len(compile_events),
        "n_lbfgs_events": len(lbfgs_events),
        "n_hessian_events": len(hessian_events),
        "n_cache_hits": len(cache_hits),
        "n_cache_misses": len(cache_misses),
    }

    # Archive the raw JAX_LOG_COMPILES text for this P.
    log_archive = json_path.parent / f"vb_laplace_jax_compiles_P{P}.log"
    log_archive.write_text(log_text, encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        f"--vb-laplace-probe P={P} complete: cold={cold_s:.1f}s "
        f"warm={warm_s:.1f}s next_proc={next_proc_s:.1f}s, "
        f"{len(compile_events)} compile events, "
        f"{len(cache_hits)} cache hits, "
        f"{len(cache_misses)} cache misses.  "
        f"Log archived to {log_archive}"
    )


def main() -> None:
    """Execute one chunk of the power sweep.

    Raises
    ------
    SystemExit
        On argument parse error (via argparse).
    """
    args = parse_args()

    # -----------------------------------------------------------------
    # FitConfig YAML loading: when --fit-config is provided, populate the
    # legacy args.fit_chains / fit_draws / fit_tune / max_tree_depth from
    # the YAML so downstream code that reads those attrs keeps working.
    # -----------------------------------------------------------------
    if args.fit_config is not None:
        _loaded_cfg = FitConfig.from_yaml(args.fit_config)
        args.fit_chains = _loaded_cfg.sampler.n_chains
        args.fit_draws = _loaded_cfg.sampler.n_draws
        args.fit_tune = _loaded_cfg.sampler.n_warmup
        args.max_tree_depth = _loaded_cfg.sampler.max_tree_depth
        args.laplace_warmup = _loaded_cfg.mitigation.use_laplace_warmup
        print(
            f"[FitConfig] Loaded from {args.fit_config}: "
            f"chains={args.fit_chains} draws={args.fit_draws} "
            f"warmup={args.fit_tune} max_tree_depth={args.max_tree_depth}",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Phase 21 diagnostic mutual-exclusion guards (BENCH-DIAG-04).
    # -----------------------------------------------------------------
    diag_flags = (
        (args.p_scan is not None),
        bool(args.vb_laplace_probe),
        bool(args.vb_laplace_subproc_only),
    )
    if sum(diag_flags) > 1:
        print(
            "ERROR: --p-scan, --vb-laplace-probe, and "
            "--vb-laplace-subproc-only are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.benchmark and any(diag_flags):
        print(
            "ERROR: --benchmark cannot be combined with --p-scan, "
            "--vb-laplace-probe, or --vb-laplace-subproc-only.",
            file=sys.stderr,
        )
        sys.exit(2)

    # -----------------------------------------------------------------
    # Phase 21 probe dispatch: short-circuit BEFORE loading power config
    # or building the SBF grid.  The probes do not use chunk/grid logic
    # and write to their own Phase 21 artifact paths.
    # -----------------------------------------------------------------
    if args.vb_laplace_subproc_only:
        _run_vb_laplace_subproc_one_shot(args.probe_p)
        return

    base_config = load_config()
    power_config = load_power_config()

    if args.n_per_group_overrides is not None:
        from dataclasses import replace as _dc_replace  # noqa: PLC0415
        _override_grid = [
            int(v.strip()) for v in args.n_per_group_overrides.split(",") if v.strip()
        ]
        if not _override_grid:
            raise ValueError(
                f"--n-per-group-overrides parsed to empty list: "
                f"{args.n_per_group_overrides!r}"
            )
        if any(v <= 0 for v in _override_grid):
            raise ValueError(
                f"--n-per-group-overrides must be positive integers, got "
                f"{_override_grid}"
            )
        print(
            f"[capability-map] Overriding power_config.n_per_group_grid: "
            f"{list(power_config.n_per_group_grid)} -> {_override_grid}",
            flush=True,
        )
        power_config = _dc_replace(power_config, n_per_group_grid=_override_grid)

    if args.p_scan is not None:
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else _cfg.MODELS_DIR / "power"
        )
        _run_p_scan_probe(
            base_config,
            power_config,
            output_dir,
            args.p_scan,
            args,
        )
        return

    if args.vb_laplace_probe:
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else _cfg.MODELS_DIR / "power"
        )
        _run_vb_laplace_probe(
            base_config,
            power_config,
            output_dir,
            args.probe_p,
            args,
        )
        return

    # Phase 14.2 variant 5: --laplace-only short-circuits the NUTS sampling
    # work to a 1-tune / 1-draw token call so wall time is dominated by
    # the Laplace LBFGS + Hessian.  Implies --laplace-warmup so
    # window_adaptation is skipped (warmup_params come from Laplace).
    if args.laplace_only:
        args.laplace_warmup = True
        args.fit_tune = 1
        args.fit_draws = 1
        print(
            "[variant 5] --laplace-only: forcing --laplace-warmup, "
            "--fit-tune=1, --fit-draws=1",
            flush=True,
        )

    grid_size = sbf_grid_size(
        power_config.effect_size_grid,
        power_config.n_iterations,
    )
    task_ids = chunk_task_ids(args.chunk_id, power_config.n_chunks, grid_size)

    output_dir = (
        args.output_dir if args.output_dir is not None else _cfg.MODELS_DIR / "power"
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
        print(f"Dry run: wrote {len(rows)} placeholder rows to {out_path}")
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

        _loop_seed = int(rngs[i].integers(0, 2**31))
        _loop_fit_config = FitConfig(
            model_name="hgf_3level",  # overridden inside run_sbf_iteration
            sampler=SamplerConfig(
                backend="blackjax",
                n_chains=args.fit_chains,
                n_draws=args.fit_draws,
                n_warmup=args.fit_tune,
                target_accept=0.9,
                random_seed=_loop_seed,
                max_tree_depth=args.max_tree_depth,
            ),
            mitigation=MitigationConfig(
                use_laplace_warmup=args.laplace_warmup,
            ),
            progressbar=False,
        )
        results = run_sbf_iteration(
            base_config=base_config,
            effect_size_delta=effect_size_delta,
            iteration=iteration,
            child_seed=_loop_seed,
            n_per_group_grid=power_config.n_per_group_grid,
            power_config=power_config,
            fit_config=_loop_fit_config,
            use_legacy=args.legacy,
        )
        all_results.extend(results)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(task_ids)} iterations complete")

    write_parquet_batch(all_results, out_path)
    print(f"Wrote {len(all_results)} rows ({len(task_ids)} iterations) to {out_path}")


if __name__ == "__main__":
    main()
