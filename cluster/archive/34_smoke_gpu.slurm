#!/bin/bash
# =============================================================================
# SLURM Job: Phase 34 — Mode B Smoke Test (4 cells, tiny draws)
# =============================================================================
# Quick validation that all 4 mitigation combos complete without crash.
# Uses 2-level model, n_per_group=10 (P=60), n_draws=50, n_warmup=50.
# Cells 0-3 cover: hier+M1, hier+M1+M2, hier+M1+M2+Laplace, +covariates.
#
# Expected runtime: ~5-15 min per cell.
#
# Usage:
#   sbatch cluster/34_smoke_gpu.slurm
# =============================================================================

#SBATCH --job-name=smoke34
#SBATCH --output=logs/smoke34_%A_%a.out
#SBATCH --error=logs/smoke34_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --account=fc37
#SBATCH --signal=B:TERM@60

# =============================================================================
# Environment Setup (same as 34_grid_sweep_gpu.slurm)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "Smoke test cell ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "============================================================"

mkdir -p logs models/power/bench_mode_b_results
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env_v10}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV}"
    exit 1
}

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
fi

export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR

mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# GPU check
python -c "import jax; assert jax.devices()[0].platform == 'gpu', f'No GPU: {jax.devices()}'"
if [[ $? -ne 0 ]]; then
    echo "ERROR: No GPU on $(hostname)"
    exit 1
fi

# =============================================================================
# Run smoke cell with tiny draws
# =============================================================================
echo ""
echo "Running SMOKE cell ${SLURM_ARRAY_TASK_ID} (n_draws=50, n_warmup=50)..."
echo ""

python scripts/benchmarks/16_benchmark_hierarchical.py \
    --cell-id ${SLURM_ARRAY_TASK_ID} \
    --job-id ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --n-draws 50 \
    --n-warmup 50

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SMOKE CELL ${SLURM_ARRAY_TASK_ID} COMPLETE"
    RESULT_FILE="models/power/bench_mode_b_results/cell_$(printf '%04d' ${SLURM_ARRAY_TASK_ID}).json"
    if [[ -f "$RESULT_FILE" ]]; then
        cat "$RESULT_FILE" | python -m json.tool 2>/dev/null || cat "$RESULT_FILE"
    fi
else
    echo "SMOKE CELL ${SLURM_ARRAY_TASK_ID} FAILED (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
