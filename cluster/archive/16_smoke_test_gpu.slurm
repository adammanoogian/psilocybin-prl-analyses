#!/bin/bash
# =============================================================================
# SLURM Job: Phase 16/17 — GPU Smoke Test (BlackJAX NUTS + JIT diagnostics)
# =============================================================================
# Lightweight test that verifies:
#   1. CUDA/PTX version match (XLA parallel compilation enabled)
#   2. Simulation vmap compiles on GPU
#   3. Cold JIT timing for BlackJAX NUTS MCMC (default sampler since Phase 17)
#   4. Warm JIT cache reuse (same shape, different data)
#
# BlackJAX compiles the NUTS step function once via jax.jit and reuses it
# across all MCMC steps — no per-call JIT recompilation (unlike NumPyro).
# This smoke test exercises the BlackJAX path by calling
# fit_batch_hierarchical (which defaults to sampler="blackjax").
#
# Runs at N=5/group (30 participant-sessions) with minimal MCMC
# (2 chains, 10 draws, 10 tune).  Should complete in <1 hour if
# CUDA versions are matched.  Exits non-zero if gates fail.
#
# Prerequisites:
#   pip install -r cluster/requirements-gpu.txt
#
# Usage:
#   sbatch cluster/16_smoke_test_gpu.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_smoke16
#SBATCH --output=logs/smoke16_%j.out
#SBATCH --error=logs/smoke16_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Run-commit provenance: pin every log to the commit it ran from
echo "============================================================"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Run job:    ${SLURM_JOB_ID:-<no-slurm>} on $(hostname)"
echo "============================================================"

mkdir -p cluster/logs models/power
export PYTHONUNBUFFERED=1

# Activate ds_env (or override via CONDA_ENV=ds_env_v10 at sbatch --export time)
_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV} conda environment"
    exit 1
}

# =============================================================================
# Add pip-installed ptxas to PATH (cuda13: nvidia.cu13, cuda12: nvidia.cuda_nvcc)
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
# CUDA / PTX Version Compatibility Check
# =============================================================================
check_cuda_compat() {
    DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [[ -z "$DRIVER_CUDA_VER" ]]; then
        echo "WARNING: nvidia-smi not available or no GPU detected"
        return 1
    fi

    PTXAS_VER=$(ptxas --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    if [[ -z "$PTXAS_VER" ]]; then
        echo "FAIL: ptxas not found in PATH"
        echo ""
        echo "  XLA parallel compilation is DISABLED without ptxas."
        echo "  This causes 3-5x slower JIT and is the #1 bottleneck."
        echo ""
        echo "  FIX: pip install -r cluster/requirements-gpu.txt"
        echo "       (installs nvidia-cuda-nvcc-cu12==12.8.93)"
        echo ""
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
        echo "FAIL: ptxas CUDA version ($PTXAS_VER) > driver CUDA version ($DRIVER_CUDA_VER)"
        echo "  XLA parallel compilation disabled — expect 3-5x slower JIT."
        echo "  FIX: pip install -r cluster/requirements-gpu.txt"
        return 1
    fi

    echo "OK: ptxas version <= driver version (XLA parallel compilation enabled)"
    return 0
}

echo ""
echo "=== CUDA Environment Check ==="
if ! check_cuda_compat; then
    echo ""
    echo "WARNING: CUDA compatibility issue detected."
    echo "  The smoke test will run but is expected to FAIL the JIT gates."
    echo "  Install the fix first: pip install -r cluster/requirements-gpu.txt"
    echo ""
fi

# JAX compilation cache (shared across nodes via Lustre scratch; cross-SLURM-job persistence — Phase 21 VERDICT §c)
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

# XLA autotuning cache (persist autotuning results across jobs)
_XLA_AUTOTUNE_DIR="/scratch/${_PROJECT}/${USER}/.xla_autotune_cache"
mkdir -p "$_XLA_AUTOTUNE_DIR" 2>/dev/null
_XLA_AUTOTUNE_FILE="${_XLA_AUTOTUNE_DIR}/results.pb"
export XLA_FLAGS="--xla_gpu_dump_autotune_results_to=${_XLA_AUTOTUNE_FILE}"
if [[ -f "$_XLA_AUTOTUNE_FILE" ]]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_load_autotune_results_from=${_XLA_AUTOTUNE_FILE}"
    echo "XLA autotune: loading cached results from ${_XLA_AUTOTUNE_FILE}"
else
    echo "XLA autotune: no cached results yet, will create on first run"
fi

# Diagnostics: log compilations + cache miss reasons
export JAX_LOG_COMPILES=1

# Show actual VRAM usage, not preallocated pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Verify GPU
# =============================================================================
# Note: BlackJAX supports pmap for multi-GPU chain parallelism, but this
# smoke test uses 1 GPU (--gres=gpu:1) to validate compilation and cache
# reuse.  For multi-GPU pmap, see 08_power_sweep.slurm comments.
echo ""
echo "============================================================"
echo "Phase 16/17 GPU Smoke Test (BlackJAX NUTS)"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):  ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Start:   $(date)"
echo "============================================================"
echo ""

python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — smoke test results will not be meaningful!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
"
echo ""

# =============================================================================
# Run Smoke Test
# =============================================================================
python scripts/03_pre_analysis/03_run_power_iteration.py \
    --smoke-test \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-smoke16}"

EXIT_CODE=$?

# Note: Smoke test gate thresholds (cold JIT < 600s, cache speedup > 3x,
# warm JIT < 120s) were originally set for the NumPyro sampler.  BlackJAX
# should be equal or faster — these thresholds may be tightened after
# BlackJAX-specific benchmarking confirms typical timings on L40S.
echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SMOKE TEST PASSED (BlackJAX NUTS)"
    echo "  JIT compilation and cache reuse are working."
    echo "  Safe to proceed to full benchmark (14_benchmark_gpu.slurm)."
else
    echo "SMOKE TEST FAILED (exit code: $EXIT_CODE)"
    echo "  Check: logs/smoke16_${SLURM_JOB_ID}.err"
    echo "  Most likely fix: pip install -r cluster/requirements-gpu.txt"
fi
echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
