#!/bin/bash
# =============================================================================
# SLURM Job: Phase 33-06 — Mode B Recovery Validation
# =============================================================================
# Runs both covariate conditions (orthogonal + near-collinear) for Mode B
# hierarchical recovery validation at P=200.
#
# Usage:
#   sbatch cluster/33_recovery_validation.slurm
#
# Debug solo run with auto-push:
#   bash cluster/submit_one.sh cluster/33_recovery_validation.slurm
# =============================================================================

#SBATCH --job-name=recov33
#SBATCH --output=logs/recov33_%j.out
#SBATCH --error=logs/recov33_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --account=fc37
#SBATCH --signal=B:TERM@120

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

mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

# Activate ds_env_v10
_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env_v10}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV} conda environment"
    exit 1
}

# =============================================================================
# Add pip-installed ptxas to PATH (nvidia-cuda-nvcc-cu12 puts it in site-packages)
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

# JAX compilation cache (shared across nodes via Lustre scratch)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Pre-warm ArviZ daily-warning file to defeat import-time race
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# XLA autotuning cache (persist autotuning results across jobs)
_XLA_AUTOTUNE_DIR="/scratch/${_PROJECT}/${USER}/.xla_autotune_cache"
mkdir -p "$_XLA_AUTOTUNE_DIR" 2>/dev/null
_XLA_AUTOTUNE_FILE="${_XLA_AUTOTUNE_DIR}/results.pb"
export XLA_FLAGS="--xla_gpu_dump_autotune_results_to=${_XLA_AUTOTUNE_FILE}"
if [[ -f "$_XLA_AUTOTUNE_FILE" ]]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_load_autotune_results_from=${_XLA_AUTOTUNE_FILE}"
fi

# Show actual VRAM usage, not preallocated pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# GPU Assertion
# =============================================================================
echo "============================================================"
echo "Phase 33-06 Mode B Recovery Validation"
echo "============================================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Conda env: ${_CONDA_ENV}"
echo "Start:     $(date)"
echo "============================================================"
echo ""

python -c "import jax; assert jax.devices()[0].platform == 'gpu', f'No GPU: {jax.devices()}'"
if [[ $? -ne 0 ]]; then
    echo "ERROR: GPU assertion failed on $(hostname) — cuSPARSE incompatibility."
    exit 1
fi

# =============================================================================
# Run Recovery Validation
# =============================================================================
echo "Running Mode B recovery validation (both conditions)..."
echo ""

python scripts/03_pre_analysis/06_recovery_validation_modeb.py --condition both --seed 123

EXIT_CODE=$?

echo ""
echo "============================================================"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "RECOVERY VALIDATION PASSED"
    if [[ -f "logs/33_recovery_results.json" ]]; then
        echo "Results:"
        cat logs/33_recovery_results.json | python -m json.tool 2>/dev/null || cat logs/33_recovery_results.json
    fi
else
    echo "RECOVERY VALIDATION FAILED (exit code: $EXIT_CODE)"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
