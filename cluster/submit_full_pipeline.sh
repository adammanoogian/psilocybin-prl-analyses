#!/bin/bash
# =============================================================================
# Wave Orchestrator: Submit Full PRL HGF Pipeline on M3 Cluster
# =============================================================================
# Submits all pipeline stages with SLURM dependency chains.
#
# Wave 1: Simulate (inline — fast, produces input CSV)
# Wave 2: Fit hgf_2level + hgf_3level (parallel GPU jobs)
# Wave 3: Validation (depends on both fit jobs)
# Wave 4: Group analysis (depends on validation, CPU-only)
# Wave 5: Push results (depends on analysis, optional)
#
# Usage:
#   bash cluster/submit_full_pipeline.sh
#   bash cluster/submit_full_pipeline.sh --skip-push
#   bash cluster/submit_full_pipeline.sh --skip-validation
#   bash cluster/submit_full_pipeline.sh --models="hgf_2level"
#   bash cluster/submit_full_pipeline.sh --skip-push --models="hgf_3level"
#
# Prerequisites:
#   - ds_env conda environment with jax[cuda12] installed
#   - On M3 login node with SLURM available
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SKIP_PUSH=false
SKIP_VALIDATION=false
MODELS="hgf_2level hgf_3level"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --models=*)
            MODELS="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: bash cluster/submit_full_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-push          Skip auto-push results to git"
            echo "  --skip-validation    Skip validation step (05)"
            echo "  --models=MODELS      Models to fit (default: \"hgf_2level hgf_3level\")"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs

echo "============================================================"
echo "PRL HGF Pipeline Orchestrator"
echo "============================================================"
echo "Project: $PROJECT_ROOT"
echo "Models: $MODELS"
echo "Skip push: $SKIP_PUSH"
echo "Skip validation: $SKIP_VALIDATION"
echo "Start: $(date)"
echo "============================================================"

# Load miniforge3 for simulation step
module load miniforge3

# Activate ds_env (shared env across projects)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# =============================================================================
# Wave 1: Simulate (inline — fast ~30s)
# =============================================================================
echo ""
echo "============================================================"
echo "Wave 1: Simulate Participants"
echo "============================================================"

python scripts/03_simulate_participants.py
if [[ $? -ne 0 ]]; then
    echo "ERROR: Simulation failed. Aborting pipeline."
    exit 1
fi

echo "Simulation complete."

# =============================================================================
# Wave 2: Fit Models (parallel GPU jobs)
# =============================================================================
echo ""
echo "============================================================"
echo "Wave 2: Submit MCMC Fitting Jobs"
echo "============================================================"

FIT_JOBIDS=()

for model in $MODELS; do
    JOBID=$(sbatch --parsable \
        --export="MODEL=$model" \
        --job-name="prl_fit_${model}" \
        cluster/04_fit_mcmc_gpu.slurm)
    FIT_JOBIDS+=("$JOBID")
    echo "  $model -> Job $JOBID"
done

FIT_DEPENDENCY=$(IFS=:; echo "${FIT_JOBIDS[*]}")
echo "Fit jobs submitted: ${FIT_JOBIDS[*]}"

# =============================================================================
# Wave 3: Validation (depends on all fit jobs)
# =============================================================================
VALID_JOBID=""

if [[ "$SKIP_VALIDATION" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "Wave 3: Submit Validation Job"
    echo "============================================================"

    VALID_JOBID=$(sbatch --parsable \
        --dependency=afterok:${FIT_DEPENDENCY} \
        cluster/05_validation_gpu.slurm)
    echo "  Validation -> Job $VALID_JOBID (depends on fit: ${FIT_DEPENDENCY})"
else
    echo ""
    echo "Wave 3: Validation SKIPPED (--skip-validation)"
fi

# =============================================================================
# Wave 4: Group Analysis (depends on validation or fit)
# =============================================================================
echo ""
echo "============================================================"
echo "Wave 4: Submit Group Analysis Job"
echo "============================================================"

if [[ -n "$VALID_JOBID" ]]; then
    ANALYSIS_DEP="--dependency=afterok:${VALID_JOBID}"
    echo "  Depends on: validation ($VALID_JOBID)"
else
    ANALYSIS_DEP="--dependency=afterok:${FIT_DEPENDENCY}"
    echo "  Depends on: fit jobs (${FIT_DEPENDENCY})"
fi

ANALYSIS_JOBID=$(sbatch --parsable \
    $ANALYSIS_DEP \
    cluster/06_analysis.slurm)
echo "  Analysis -> Job $ANALYSIS_JOBID"

# =============================================================================
# Wave 5: Push Results (depends on analysis, optional)
# =============================================================================
PUSH_JOBID=""

if [[ "$SKIP_PUSH" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "Wave 5: Submit Push Results Job"
    echo "============================================================"

    ALL_JOBIDS=("${FIT_JOBIDS[@]}")
    [[ -n "$VALID_JOBID" ]] && ALL_JOBIDS+=("$VALID_JOBID")
    ALL_JOBIDS+=("$ANALYSIS_JOBID")
    PARENT_LIST=$(IFS=' '; echo "${ALL_JOBIDS[*]}")

    PUSH_JOBID=$(sbatch --parsable \
        --dependency=afterany:${ANALYSIS_JOBID} \
        --export="ALL,PARENT_JOBS=${PARENT_LIST}" \
        cluster/99_push_results.slurm)
    echo "  Push -> Job $PUSH_JOBID (depends on analysis: ${ANALYSIS_JOBID})"
else
    echo ""
    echo "Wave 5: Push SKIPPED (--skip-push)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Pipeline Submitted Successfully!"
echo "============================================================"
echo ""
echo "Job Summary:"
echo "  Wave 1 (simulate):   inline (done)"
for i in "${!FIT_JOBIDS[@]}"; do
    model=$(echo $MODELS | cut -d' ' -f$((i+1)))
    echo "  Wave 2 (fit $model): ${FIT_JOBIDS[$i]}"
done
[[ -n "$VALID_JOBID" ]] && echo "  Wave 3 (validate):   $VALID_JOBID"
echo "  Wave 4 (analysis):   $ANALYSIS_JOBID"
[[ -n "$PUSH_JOBID" ]] && echo "  Wave 5 (push):       $PUSH_JOBID"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u $USER"
echo "  watch -n 30 'squeue -u $USER'"
echo ""
echo "View logs:"
for i in "${!FIT_JOBIDS[@]}"; do
    model=$(echo $MODELS | cut -d' ' -f$((i+1)))
    echo "  tail -f cluster/logs/fit_mcmc_${FIT_JOBIDS[$i]}.out   # $model"
done
[[ -n "$VALID_JOBID" ]] && echo "  tail -f cluster/logs/validation_${VALID_JOBID}.out"
echo "  tail -f cluster/logs/analysis_${ANALYSIS_JOBID}.out"
[[ -n "$PUSH_JOBID" ]] && echo "  tail -f cluster/logs/push_results_${PUSH_JOBID}.out"
echo ""
echo "Cancel all:"
ALL_IDS="${FIT_JOBIDS[*]}"
[[ -n "$VALID_JOBID" ]] && ALL_IDS="$ALL_IDS $VALID_JOBID"
ALL_IDS="$ALL_IDS $ANALYSIS_JOBID"
[[ -n "$PUSH_JOBID" ]] && ALL_IDS="$ALL_IDS $PUSH_JOBID"
echo "  scancel $ALL_IDS"
echo ""
echo "============================================================"
