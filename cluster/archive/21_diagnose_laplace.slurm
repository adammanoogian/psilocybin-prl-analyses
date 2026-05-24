#!/bin/bash
# =============================================================================
# SLURM Job: Phase 21 — VB-Laplace JIT Probe (BENCH-DIAG-02)
# =============================================================================
# Runs the --vb-laplace-probe entry point in scripts/03_pre_analysis/03_run_power_iteration.py
# sequentially for P ∈ {5, 20} inside a single 30-minute GPU job.
#
# Captures:
#   - JAX_LOG_COMPILES=1 stdout → archived per-P under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/vb_laplace_jax_compiles_P{5,20}.log
#   - XLA text-IR HLO dumps per-P under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/laplace_P{5,20}/
#   - vb_laplace_vs_nuts_jit.json with Laplace-path per_p_scaling rows
#
# Scope (per 21-03-PLAN.md):
#   - This script does NOT touch results/power/benchmark_batched.json (14.1-03 owns it).
#   - This script does NOT touch apply_decision_gate thresholds.
#   - NUTS-matched probe is plan 21-04; P=50 is intentionally out of scope here
#     (early-stop-at-verdict policy, CONTEXT.md).
#
# Usage:
#   sbatch cluster/21_diagnose_laplace.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_diag21_laplace
#SBATCH --output=logs/diag21_laplace_%j.out
#SBATCH --error=logs/diag21_laplace_%j.err
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
# Environment Setup (mirrors cluster/14_benchmark_gpu.slurm)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
mkdir -p .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps
export PYTHONUNBUFFERED=1

# Activate ds_env (identical block to 14_benchmark_gpu.slurm lines 37-43)
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
# Same JAX_COMPILATION_CACHE_DIR path as cluster/14_benchmark_gpu.slurm for
# apples-to-apples cache-warming across the two probes.
# Shared across nodes via Lustre scratch (Phase 21 VERDICT §c patch).
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

# Phase 21 addition: emit JIT compile events to stdout for forensic parsing.
export JAX_LOG_COMPILES=1

# Show actual VRAM usage, not preallocated pool (mirrors 14_benchmark_gpu.slurm)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 21 VB-Laplace JIT Probe"
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
#   2. Invokes --vb-laplace-probe --probe-p $P --diagnostic-mode.
#   3. Tees stdout (contains JAX_LOG_COMPILES events) to
#      vb_laplace_jax_compiles_P${P}.log for forensic archival.
#   4. Unsets XLA_FLAGS to avoid cross-P dump contamination.
#
# The `|| echo` pattern preserves P=5 results if P=20 fails (OOM, timeout, etc.).
# =============================================================================
for P in 5 20; do
    echo "========================================"
    echo "VB-Laplace probe: P=${P}"
    echo "========================================"
    echo "Start P=${P}: $(date)"

    HLO_DUMP_DIR=".planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/laplace_P${P}"
    COMPILE_LOG=".planning/phases/21-benchmark-bottleneck-diagnosis/vb_laplace_jax_compiles_P${P}.log"
    mkdir -p "${HLO_DUMP_DIR}"

    # Per-P XLA dump isolation. Set locally for this iteration only.
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_DIR}"
    echo "XLA_FLAGS=${XLA_FLAGS}"

    # Tee stdout+stderr so JAX_LOG_COMPILES events land in both the SLURM log
    # and the per-P archival log (consumed by 21-06 cache forensics).
    python scripts/03_pre_analysis/03_run_power_iteration.py \
        --vb-laplace-probe \
        --probe-p "${P}" \
        --diagnostic-mode \
        --chunk-id 0 \
        --job-id "${SLURM_JOB_ID:-diag21_laplace}" \
        2>&1 | tee "${COMPILE_LOG}"
    P_EXIT=${PIPESTATUS[0]}

    if [[ ${P_EXIT} -ne 0 ]]; then
        echo "WARNING: VB-Laplace probe at P=${P} exited non-zero (${P_EXIT}); continuing"
    fi

    echo "End P=${P}: $(date)"
    echo ""

    # Clear XLA_FLAGS so the next iteration (or the post-loop block) doesn't
    # inherit the P=${P} dump directory.
    unset XLA_FLAGS
done

echo "============================================================"
echo "All probes complete"
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

# Exit 0 even if individual P iterations failed; any partial state is a data
# point per 21-03-PLAN.md (walltime overrun or P=20 OOM is acceptable).
exit 0
