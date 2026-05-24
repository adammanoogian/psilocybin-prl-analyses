#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — VALID-03 Compare CPU vs GPU Results
# =============================================================================
# Runs after both 14_valid03_cpu.slurm and 14_valid03_gpu.slurm complete.
# Asserts all per-participant posterior means agree within 1% relative error.
#
# Usage (after both fit jobs finish):
#   sbatch --dependency=afterok:${CPU_JOBID}:${GPU_JOBID} cluster/14_valid03_compare.slurm
#
# Or manually:
#   sbatch cluster/14_valid03_compare.slurm
# =============================================================================

#SBATCH --job-name=prl_v03cmp
#SBATCH --output=logs/valid03_compare_%j.out
#SBATCH --error=logs/valid03_compare_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

echo "============================================================"
echo "VALID-03: Cross-Platform Comparison"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "============================================================"
echo ""

# Check both files exist
if [[ ! -f models/power/valid03_cpu.json ]]; then
    echo "ERROR: models/power/valid03_cpu.json not found"
    echo "Run 14_valid03_cpu.slurm first"
    exit 1
fi
if [[ ! -f models/power/valid03_gpu.json ]]; then
    echo "ERROR: models/power/valid03_gpu.json not found"
    echo "Run 14_valid03_gpu.slurm first"
    exit 1
fi

python tests/scientific/test_valid03_cross_platform.py compare \
    models/power/valid03_cpu.json \
    models/power/valid03_gpu.json \
    --rtol 0.01

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "VALID-03 PASSED: CPU and GPU posterior means agree within 1%"
else
    echo "VALID-03 FAILED: posterior means differ by more than 1%"
fi

echo "============================================================"
exit $EXIT_CODE
