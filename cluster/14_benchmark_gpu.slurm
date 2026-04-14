#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — GPU Benchmark + Decision Gate (BENCH-01..05)
# =============================================================================
# Runs the v1.2 batched hierarchical benchmark on a GPU node:
#   1. JAX compilation cache warm/cold test (BENCH-05)
#   2. Full batched iteration timing (BENCH-01, BENCH-04)
#   3. Decision gate: GPU vs CPU comp recommendation (BENCH-02)
#   4. Writes results/power/benchmark_batched.json
#   5. Appends decision to .planning/STATE.md
#
# Usage:
#   sbatch cluster/14_benchmark_gpu.slurm
# =============================================================================

#SBATCH --job-name=prl_bench14
#SBATCH --output=cluster/logs/bench14_%j.out
#SBATCH --error=cluster/logs/bench14_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
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

# GPU deps: pip install -r cluster/requirements-gpu.txt

# =============================================================================
# Add pip-installed ptxas to PATH (nvidia-cuda-nvcc-cu12 puts it in site-packages)
# =============================================================================
PTXAS_BIN=$(python -c 'import nvidia.cuda_nvcc, os; print(os.path.join(os.path.dirname(nvidia.cuda_nvcc.__file__), "bin"))' 2>/dev/null)
if [[ -n "$PTXAS_BIN" && -d "$PTXAS_BIN" ]]; then
    export PATH="$PTXAS_BIN:$PATH"
    echo "Added ptxas to PATH: $PTXAS_BIN"
fi

# =============================================================================
# CUDA / PTX Version Compatibility Check (NPRO-03)
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

# JAX compilation cache (persistent across jobs for BENCH-05)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# =============================================================================
# Verify GPU
# =============================================================================
echo "============================================================"
echo "Phase 14 GPU Benchmark"
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
    print('ERROR: No GPU detected — benchmark results will not be meaningful!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
"
echo ""

# =============================================================================
# Run Benchmark (BENCH-01, BENCH-02, BENCH-04, BENCH-05)
# =============================================================================
echo "Running batched hierarchical benchmark..."
echo ""

python scripts/08_run_power_iteration.py \
    --benchmark \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-bench14}" \
    --fit-chains 4 \
    --fit-draws 500 \
    --fit-tune 500 \
    --sampler numpyro

EXIT_CODE=$?

echo ""
echo "============================================================"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "BENCHMARK COMPLETE"
    echo ""
    if [[ -f results/power/benchmark_batched.json ]]; then
        echo "Results:"
        cat results/power/benchmark_batched.json | python -m json.tool 2>/dev/null || \
            cat results/power/benchmark_batched.json
    fi
else
    echo "BENCHMARK FAILED (exit code: $EXIT_CODE)"
    echo "Check: cluster/logs/bench14_${SLURM_JOB_ID}.err"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
