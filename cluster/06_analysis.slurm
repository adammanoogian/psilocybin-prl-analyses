#!/bin/bash
# =============================================================================
# SLURM Job: Group-Level Analysis (CPU-only)
# =============================================================================
# Runs group-level HGF parameter analysis using bambi mixed-effects models.
# No GPU required — runs on comp partition. Uses ds_env for bambi.
#
# Usage:
#   sbatch cluster/06_analysis.slurm
#   sbatch --dependency=afterok:12345 cluster/06_analysis.slurm
#   sbatch --export=DRAWS=4000 cluster/06_analysis.slurm
#
# Runs analysis for both 2-level and 3-level models sequentially.
# Plots are skipped on cluster (--skip-plots); generate locally.
# =============================================================================

#SBATCH --job-name=prl_analysis
#SBATCH --output=cluster/logs/analysis_%j.out
#SBATCH --error=cluster/logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=comp
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration
# =============================================================================
DRAWS="${DRAWS:-2000}"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Group-Level Analysis - SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Draws: $DRAWS"
echo "Start: $(date)"
echo "============================================================"

# Activate ds_env (shared env across projects)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# =============================================================================
# Run Group Analysis for Both Models
# =============================================================================
ALL_SUCCESS=true

for model in 2level 3level; do
    echo ""
    echo "============================================================"
    echo "Analysing: hgf_${model}"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/06_group_analysis.py --model $model --skip-plots --draws $DRAWS
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "*** WARNING: hgf_${model} analysis exited with code $EXIT_CODE ***"
        ALL_SUCCESS=false
    fi

    echo "Completed hgf_${model}: $(date) (exit code: $EXIT_CODE)"
done

echo ""
echo "============================================================"
echo "Group analysis complete. End: $(date)"
echo "Results: results/group_analysis/"
echo "============================================================"

if [[ "$ALL_SUCCESS" != "true" ]]; then
    echo "WARNING: Some analyses had non-zero exit codes (see above)"
    exit 1
fi
