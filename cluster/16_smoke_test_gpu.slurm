#!/bin/bash
# =============================================================================
# SLURM Job: Phase 16 — GPU Smoke Test (JIT compilation diagnostics)
# =============================================================================
# Lightweight test that verifies:
#   1. CUDA/PTX version match (XLA parallel compilation enabled)
#   2. Simulation vmap compiles on GPU
#   3. Cold JIT timing for numpyro-direct MCMC
#   4. Warm JIT cache reuse (same shape, different data)
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
# =============================================================================

#SBATCH --job-name=prl_smoke16
#SBATCH --output=cluster/logs/smoke16_%j.out
#SBATCH --error=cluster/logs/smoke16_%j.err
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

mkdir -p cluster/logs results/power
export PYTHONUNBUFFERED=1

# Activate ds_env
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

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

# JAX compilation cache (persistent across jobs)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# =============================================================================
# Verify GPU
# =============================================================================
echo ""
echo "============================================================"
echo "Phase 16 GPU Smoke Test"
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
python scripts/08_run_power_iteration.py \
    --smoke-test \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-smoke16}"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SMOKE TEST PASSED"
    echo "  JIT compilation and cache reuse are working."
    echo "  Safe to proceed to full benchmark (14_benchmark_gpu.slurm)."
else
    echo "SMOKE TEST FAILED (exit code: $EXIT_CODE)"
    echo "  Check: cluster/logs/smoke16_${SLURM_JOB_ID}.err"
    echo "  Most likely fix: pip install -r cluster/requirements-gpu.txt"
fi
echo ""
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
