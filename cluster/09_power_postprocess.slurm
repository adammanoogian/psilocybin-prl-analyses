#!/bin/bash
# =============================================================================
# SLURM Job: Power Analysis Post-Processing (CPU-only)
# =============================================================================
# Runs after the power sweep array job completes. Executes sequentially:
#   1. Aggregate parquet files → summary CSVs   (09_aggregate_power.py)
#   2. Generate publication figures              (10_plot_power_curves.py)
#   3. Write N/group recommendation             (11_write_recommendation.py)
#
# Usage:
#   # Standalone:
#   sbatch cluster/09_power_postprocess.slurm
#
#   # As dependency (submitted by submit_power_pipeline.sh):
#   sbatch --dependency=afterok:12345 cluster/09_power_postprocess.slurm
# =============================================================================

#SBATCH --job-name=prl_postprocess
#SBATCH --output=cluster/logs/postprocess_%j.out
#SBATCH --error=cluster/logs/postprocess_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=comp
#SBATCH --cpus-per-task=4

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

# Activate ds_env (shared env across projects)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

echo "============================================================"
echo "Power Post-Processing Pipeline"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:   $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Aggregate parquet → CSV
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1/3: Aggregate Power Results"
echo "============================================================"

python scripts/09_aggregate_power.py
STEP1_EXIT=$?

if [[ $STEP1_EXIT -ne 0 ]]; then
    echo "*** ERROR: Aggregation failed (exit $STEP1_EXIT). Aborting pipeline. ***"
    exit $STEP1_EXIT
fi

echo "Step 1 complete."

# =============================================================================
# Step 2: Generate publication figures
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2/3: Generate Publication Figures"
echo "============================================================"

python scripts/10_plot_power_curves.py
STEP2_EXIT=$?

if [[ $STEP2_EXIT -ne 0 ]]; then
    echo "*** WARNING: Figure generation failed (exit $STEP2_EXIT). Continuing to recommendation. ***"
fi

echo "Step 2 complete."

# =============================================================================
# Step 3: Write recommendation
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3/3: Write Recommendation"
echo "============================================================"

python scripts/11_write_recommendation.py
STEP3_EXIT=$?

if [[ $STEP3_EXIT -ne 0 ]]; then
    echo "*** WARNING: Recommendation generation failed (exit $STEP3_EXIT). ***"
fi

echo "Step 3 complete."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Post-Processing Complete"
echo "============================================================"
echo "  Aggregate (09): exit $STEP1_EXIT"
echo "  Figures   (10): exit ${STEP2_EXIT:-skipped}"
echo "  Recommend (11): exit ${STEP3_EXIT:-skipped}"
echo ""
echo "Results in: results/power/"
ls -lh results/power/*.csv results/power/*.md results/power/*.pdf results/power/*.png 2>/dev/null || echo "  (no output files found)"
echo ""
echo "Completed: $(date)"
echo "============================================================"

# Exit with the aggregation status (the critical step)
exit $STEP1_EXIT
