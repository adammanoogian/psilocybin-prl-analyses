#!/bin/bash
# =============================================================================
# SLURM Job: Phase 21 — NUTS (PAT-RL) JIT Probe (BENCH-DIAG-02, plan 21-04)
# =============================================================================
# Runs the --p-scan entry point in scripts/03_pre_analysis/03_run_power_iteration.py
# sequentially for P ∈ {5, 20} inside a single 30-minute GPU job.
#
# This is the logp-graph-vs-sampler isolation test's OTHER HALF. Plan 21-03's
# cluster/21_diagnose_laplace.slurm measured VB-Laplace (LBFGS + Hessian, which
# exercises the logp graph but NOT the NUTS sampler); this script measures NUTS
# (exercises the logp graph AND the sampler window-adaptation / leapfrog
# integrator). Comparing the cold compile-times at matched P ∈ {5, 20} names
# whether the sampler is the marginal bottleneck on top of the logp graph.
#
# Captures:
#   - JAX_LOG_COMPILES=1 stdout+stderr → archived per-P under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/nuts_jax_compiles_P{5,20}.log
#   - XLA text-IR HLO dumps per-P under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/nuts_P{5,20}/
#   - vb_laplace_vs_nuts_jit.json with NEW top-level key `nuts_per_p_scaling`
#     (preserving the `per_p_scaling` Laplace rows written by plan 21-03)
#   - jit_scaling_sweep.csv same-proc rows at P=5 and P=20 (written by
#     _run_p_scan_probe; `next_proc_warm_s` column stays empty here — plan
#     21-05's cross-process chain fills it)
#
# Scope (per 21-04-PLAN.md):
#   - This script does NOT touch results/power/benchmark_batched.json (14.1-03 owns it).
#   - This script does NOT touch apply_decision_gate thresholds.
#   - P=50 is intentionally out of scope (early-stop-at-verdict policy, CONTEXT.md).
#   - This is DIAGNOSTIC ONLY — no fixes, no production code changes.
#
# Cross-job cache sharing:
#   The JAX_COMPILATION_CACHE_DIR is IDENTICAL to cluster/21_diagnose_laplace.slurm
#   (shared path: /scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared — Phase 21
#   VERDICT §c patch; ${SLURMD_NODENAME:-local} suffix dropped so cross-node reuse works).
#   If Laplace (21-03) ran first on any node, the NUTS probe's first scan-body
#   compile could land as a PERSISTENT CACHE HIT — evidence that the logp-graph HLO
#   hash is stable across the Laplace→NUTS transition (plan 21-06 analyses this).
#
# Usage:
#   sbatch cluster/21_diagnose_nuts.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_diag21_nuts
#SBATCH --output=logs/diag21_nuts_%j.out
#SBATCH --error=logs/diag21_nuts_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Shell hygiene: -u catches unbound variables early; deliberately NO -e so that
# a P=5 failure does NOT skip the P=20 iteration (and vice versa).
# =============================================================================
set -u

# =============================================================================
# Environment Setup (mirrors cluster/21_diagnose_laplace.slurm verbatim)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
mkdir -p .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps
export PYTHONUNBUFFERED=1

# Activate ds_env (identical block to 14_benchmark_gpu.slurm + 21_diagnose_laplace.slurm)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# =============================================================================
# Add pip-installed ptxas to PATH (identical to 14_benchmark_gpu.slurm)
# =============================================================================
PTXAS_BIN=$(python -c '
import os, site
sp = site.getsitepackages()[0]
for subdir in ("nvidia/cu13/bin", "nvidia/cuda_nvcc/bin"):
    d = os.path.join(sp, subdir)
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "ptxas")):
        print(d); break
' 2>/dev/null)
if [[ -n "$PTXAS_BIN" && -d "$PTXAS_BIN" ]]; then
    export PATH="$PTXAS_BIN:$PATH"
    echo "Added ptxas to PATH: $PTXAS_BIN"
fi

# =============================================================================
# CUDA / PTX Version Compatibility Check (identical to 14_benchmark_gpu.slurm)
# =============================================================================
check_cuda_compat() {
    DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [[ -z "$DRIVER_CUDA_VER" ]]; then
        echo "WARNING: nvidia-smi not available or no GPU detected"
        return 1
    fi

    PTXAS_VER=$(ptxas --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    if [[ -z "$PTXAS_VER" ]]; then
        echo "WARNING: ptxas not found in PATH"
        return 1
    fi

    echo "Driver CUDA version: $DRIVER_CUDA_VER"
    echo "ptxas CUDA version:  $PTXAS_VER"

    DRIVER_MAJOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f1)
    DRIVER_MINOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f2)
    PTXAS_MAJOR=$(echo "$PTXAS_VER" | cut -d. -f1)
    PTXAS_MINOR=$(echo "$PTXAS_VER" | cut -d. -f2)

    if (( PTXAS_MAJOR > DRIVER_MAJOR )) || \
       (( PTXAS_MAJOR == DRIVER_MAJOR && PTXAS_MINOR > DRIVER_MINOR )); then
        echo ""
        echo "ERROR: ptxas CUDA version ($PTXAS_VER) > driver CUDA version ($DRIVER_CUDA_VER)"
        echo "  XLA will disable parallel compilation, causing 2-5x slower JIT."
        echo "  FIX: pip install -r cluster/requirements-gpu.txt"
        return 1
    fi

    echo "OK: ptxas version <= driver version (XLA parallel compilation enabled)"
    return 0
}

echo ""
echo "=== CUDA Environment Check ==="
check_cuda_compat || echo "WARNING: CUDA compatibility issue detected (non-fatal, continuing)"
echo ""

# =============================================================================
# JAX cache + diagnostic env vars (Phase 21)
# IDENTICAL path formula to cluster/21_diagnose_laplace.slurm so 21-06 scan-body
# cache-reuse diffing has a shared persistent cache to analyze.
# Shared across nodes via Lustre scratch (Phase 21 VERDICT §c patch — dropped
# ${SLURMD_NODENAME:-local} suffix so cross-SLURM-job cache reuse works).
# =============================================================================
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Pre-warm ArviZ daily-warning file to defeat import-time race (14.1-07 follow-up)
# arviz/__init__.py _warn_once_per_day() skips _atomic_write_text ONLY IF the
# stamp file already contains today's ISO date. Writing today's date here makes
# the racey tmp.replace(path) code path unreachable under parallel SLURM launches.
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# Phase 21 addition: emit JIT compile events to stdout/stderr for forensic parsing.
export JAX_LOG_COMPILES=1

# Show actual VRAM usage, not preallocated pool (mirrors 14_benchmark_gpu.slurm)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 21 NUTS (PAT-RL) JIT Probe"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):  ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache:          ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "JAX_LOG_COMPILES:   ${JAX_LOG_COMPILES}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:   $(date)"
echo "============================================================"
echo ""

python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — probe results will not be meaningful!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
"
echo ""

# =============================================================================
# Per-P probe loop: P ∈ {5, 20}, SEQUENTIAL in one job.
# Each iteration:
#   1. Sets XLA_FLAGS with per-P HLO dump dir so P=5 and P=20 dumps are isolated.
#   2. Invokes --p-scan $P --diagnostic-mode (which calls
#      fit_batch_hierarchical_patrl twice: cold+warm in-process).
#   3. Redirects stdout+stderr to nuts_jax_compiles_P${P}.log for forensic archival
#      (JAX_LOG_COMPILES events land on stderr via Python's logging module; the
#      `2>&1` captures both streams into the same file for post-run parsing).
#   4. Tails the last 80 lines into the SLURM stdout so the .out log stays useful.
#   5. Unsets XLA_FLAGS to avoid cross-P dump contamination.
#
# The `|| echo` pattern preserves P=5 results if P=20 fails (OOM, timeout, etc.).
# =============================================================================
for P in 5 20; do
    echo "========================================"
    echo "NUTS (PAT-RL) probe: P=${P}"
    echo "========================================"
    echo "Start P=${P}: $(date)"

    HLO_DUMP_DIR=".planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/nuts_P${P}"
    NUTS_LOG=".planning/phases/21-benchmark-bottleneck-diagnosis/nuts_jax_compiles_P${P}.log"
    mkdir -p "${HLO_DUMP_DIR}"

    # Per-P XLA dump isolation. Set locally for this iteration only.
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_DIR}"
    echo "XLA_FLAGS=${XLA_FLAGS}"

    # _run_p_scan_probe writes to stdout AND stderr. JAX_LOG_COMPILES messages
    # go to stderr via Python's logging module. Capture both streams so the
    # log file has everything for post-hoc parsing.
    python scripts/03_pre_analysis/03_run_power_iteration.py \
        --p-scan "${P}" \
        --diagnostic-mode \
        --chunk-id 0 \
        --job-id "${SLURM_JOB_ID:-diag21_nuts}" \
        > "${NUTS_LOG}" 2>&1 \
        || echo "WARNING: NUTS probe at P=${P} exited non-zero; continuing"

    # Also tee the log into the SLURM stdout so it's visible in diag21_nuts_%j.out
    echo "--- NUTS probe P=${P} log (tail 80 lines) ---"
    tail -n 80 "${NUTS_LOG}" || echo "(log file missing)"
    echo "--- end P=${P} log ---"

    echo "End P=${P}: $(date)"
    echo ""

    # Clear XLA_FLAGS so the next iteration (or the post-loop block) doesn't
    # inherit the P=${P} dump directory.
    unset XLA_FLAGS
done

echo "============================================================"
echo "All NUTS probes complete"
echo "End: $(date)"
echo "============================================================"

# =============================================================================
# Post-probe JSON assembly: parse the per-P NUTS logs and merge NEW top-level
# `nuts_per_p_scaling` key into vb_laplace_vs_nuts_jit.json WITHOUT clobbering
# the Laplace-path `per_p_scaling` rows written by plan 21-03.
#
# Schema of new key (per 21-04-PLAN.md):
#   {
#     "nuts_per_p_scaling": {
#       "5":  {"cold_s": float|None, "warm_s": float|None,
#              "n_compile_events": int, "n_cache_hits": int, "n_cache_misses": int},
#       "20": {...}
#     }
#   }
# =============================================================================
echo ""
echo "========================================"
echo "Merging NUTS results into vb_laplace_vs_nuts_jit.json"
echo "========================================"

python <<'PYEOF'
import json
import re
from pathlib import Path

phase_dir = Path(".planning/phases/21-benchmark-bottleneck-diagnosis")
json_path = phase_dir / "vb_laplace_vs_nuts_jit.json"

if json_path.exists():
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
else:
    # Should not happen if plan 21-03 ran first; guard anyway so this SLURM
    # script is idempotent and does not crash on a fresh clone.
    data = {
        "per_stage_compile_events": {},
        "per_call_wall_clock": {},
        "per_p_scaling": {},
        "bottleneck_verdict": {
            "bottleneck_layer": "undetermined",
            "evidence": "Filled by plan 21-07.",
        },
        "hlo_op_counts": {},
    }

# Additive: do NOT overwrite Laplace's per_p_scaling key. Add NUTS rows under
# a new sibling key so 21-07 can compute the Laplace-vs-NUTS ratio at matched P.
data.setdefault("nuts_per_p_scaling", {})

for P in [5, 20]:
    log_path = phase_dir / f"nuts_jax_compiles_P{P}.log"
    if not log_path.exists():
        print(f"WARNING: {log_path} missing — skipping P={P}")
        continue
    text = log_path.read_text(encoding="utf-8", errors="replace")

    compile_events = re.findall(r"Compiling (\S+) for args", text)
    cache_hits = re.findall(r"Persistent compilation cache hit for '(\S+)'", text)
    cache_misses = re.findall(
        r"PERSISTENT COMPILATION CACHE MISS for '(\S+)'", text
    )

    # Extract the cold/warm wall-clock from the probe's own print statement:
    # "--p-scan P complete: status=..., cold=Xs, warm=Ys, row appended to ..."
    match = re.search(
        r"--p-scan\s+\d+\s+complete:.*?cold=([\d.]+)s,\s*warm=([\d.]+)s",
        text,
        re.DOTALL,
    )
    if match:
        cold_s = float(match.group(1))
        warm_s = float(match.group(2))
    else:
        cold_s = None
        warm_s = None

    data["nuts_per_p_scaling"][str(P)] = {
        "cold_s": cold_s,
        "warm_s": warm_s,
        "n_compile_events": len(compile_events),
        "n_cache_hits": len(cache_hits),
        "n_cache_misses": len(cache_misses),
    }

with json_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Merged NUTS rows for P={sorted(data['nuts_per_p_scaling'].keys())}")
PYEOF

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

# Exit 0 even if individual P iterations failed; any partial state is a data
# point per 21-04-PLAN.md (walltime overrun or P=20 OOM is acceptable).
exit 0
