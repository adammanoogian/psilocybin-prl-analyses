#!/bin/bash
# =============================================================================
# SLURM Job: Phase 34 — Mode B Grid Sweep (24 cells)
# =============================================================================
# Dispatches all 24 benchmark cells as independent SLURM array tasks.
# Each cell runs one (model x n_per_group x mitigation_combo) configuration.
#
# Grid: 2 models x 3 n_per_group x 4 mitigation_combos = 24 cells
#   Models:      hgf_2level, hgf_3level
#   N/group:     10, 17, 33
#   Mitigations: hier+M1, hier+M1+M2, hier+M1+M2+Laplace,
#                hier+M1+M2+Laplace+covariates
#
# Usage:
#   sbatch cluster/34_grid_sweep_gpu.slurm
#
# Debug solo run with auto-push:
#   bash cluster/submit_one.sh cluster/34_grid_sweep_gpu.slurm
# =============================================================================

#SBATCH --job-name=bench34_grid
#SBATCH --output=logs/bench34_%A_%a.out
#SBATCH --error=logs/bench34_%A_%a.err
#SBATCH --array=0-23%4
#SBATCH --time=24:00:00
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
echo "Run job:    ${SLURM_JOB_ID:-<no-slurm>} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "============================================================"

mkdir -p logs models/power/bench_mode_b_results
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

# Activate ds_env_v10 (Phase 34 requires v10 with JAX 0.9 + BlackJAX 1.5)
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
# GPU Assertion (safety net — m3g108 is known-good A40 node)
# =============================================================================
echo "============================================================"
echo "Phase 34 Grid Sweep — Cell ${SLURM_ARRAY_TASK_ID}"
echo "============================================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
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
    echo "  Node $(hostname) does not support ds_env_v10 JAX GPU. Writing CRASH result."
    CELL_ID=$(printf '%04d' ${SLURM_ARRAY_TASK_ID})
    python -c "
import json, time
from pathlib import Path
result = {
    'cell_id': ${SLURM_ARRAY_TASK_ID},
    'status': 'CRASH',
    'error': 'cuSPARSE incompatible on node $(hostname)',
    'walltime_s': 0,
    'job_id': '${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}',
    'node': '$(hostname)',
}
Path('models/power/bench_mode_b_results/cell_${CELL_ID}.json').write_text(json.dumps(result, indent=2))
"
    exit 1
fi

# =============================================================================
# Run Grid Cell
# =============================================================================
echo "Running benchmark cell ${SLURM_ARRAY_TASK_ID}..."
echo ""

python scripts/03_pre_analysis/16_benchmark_hierarchical.py \
    --cell-id ${SLURM_ARRAY_TASK_ID} \
    --job-id ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

EXIT_CODE=$?

echo ""
echo "============================================================"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "CELL ${SLURM_ARRAY_TASK_ID} COMPLETE"
    RESULT_FILE="models/power/bench_mode_b_results/cell_$(printf '%04d' ${SLURM_ARRAY_TASK_ID}).json"
    if [[ -f "$RESULT_FILE" ]]; then
        echo "Result:"
        cat "$RESULT_FILE" | python -m json.tool 2>/dev/null || cat "$RESULT_FILE"
    fi
else
    echo "CELL ${SLURM_ARRAY_TASK_ID} FAILED (exit code: $EXIT_CODE)"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
