#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — VALID-03 CPU Baseline Fit
# =============================================================================
# Runs a small batched fit on CPU, saves posterior means to JSON.
# Submit this AND 14_valid03_gpu.slurm, then compare results.
#
# Usage:
#   sbatch cluster/14_valid03_cpu.slurm
# =============================================================================

#SBATCH --job-name=prl_v03cpu
#SBATCH --output=cluster/logs/valid03_cpu_%j.out
#SBATCH --error=cluster/logs/valid03_cpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cluster/logs results/power
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# Force CPU platform
export JAX_PLATFORM_NAME=cpu

echo "============================================================"
echo "VALID-03: CPU Baseline Fit"
echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     ${SLURMD_NODENAME:-$(hostname)}"
echo "Platform: CPU (forced via JAX_PLATFORM_NAME=cpu)"
echo "Start:    $(date)"
echo "============================================================"
echo ""

python validation/valid03_cross_platform.py run \
    --output results/power/valid03_cpu.json \
    --n-per-group 3 \
    --seed 42 \
    --n-chains 2 \
    --n-draws 200 \
    --n-tune 200

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 && -f results/power/valid03_cpu.json ]]; then
    echo "CPU fit saved to results/power/valid03_cpu.json"
else
    echo "CPU fit FAILED (exit code: $EXIT_CODE)"
fi

echo "End: $(date)"
exit $EXIT_CODE
