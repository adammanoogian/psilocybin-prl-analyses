#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — VALID-03 GPU Fit
# =============================================================================
# Runs the same small batched fit on GPU, saves posterior means to JSON.
# After both CPU + GPU jobs complete, run the comparison.
#
# Usage:
#   sbatch cluster/14_valid03_gpu.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_v03gpu
#SBATCH --output=logs/valid03_gpu_%j.out
#SBATCH --error=logs/valid03_gpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cluster/logs models/power
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# JAX compilation cache
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null

echo "============================================================"
echo "VALID-03: GPU Fit"
echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-none}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:    $(date)"
echo "============================================================"
echo ""

python -c "import jax; d=jax.devices(); print(f'JAX: {d}')"
echo ""

python tests/scientific/test_valid03_cross_platform.py run \
    --output models/power/valid03_gpu.json \
    --n-per-group 3 \
    --seed 42 \
    --n-chains 2 \
    --n-draws 200 \
    --n-tune 200

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 && -f models/power/valid03_gpu.json ]]; then
    echo "GPU fit saved to models/power/valid03_gpu.json"
else
    echo "GPU fit FAILED (exit code: $EXIT_CODE)"
fi

echo "End: $(date)"
exit $EXIT_CODE
