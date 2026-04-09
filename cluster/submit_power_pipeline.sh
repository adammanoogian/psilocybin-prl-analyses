#!/bin/bash
# =============================================================================
# Submit Power Analysis Pipeline on M3 Cluster
# =============================================================================
# Single entry point for the power analysis workflow. Handles environment
# setup, benchmarking, dry-runs, and the full sweep with dependency chains.
#
# Usage:
#   bash cluster/submit_power_pipeline.sh --setup              # pull + env setup only
#   bash cluster/submit_power_pipeline.sh --setup --benchmark  # setup then benchmark
#   bash cluster/submit_power_pipeline.sh --benchmark          # benchmark (env must exist)
#   bash cluster/submit_power_pipeline.sh --dry-run            # placeholder parquet
#   bash cluster/submit_power_pipeline.sh                      # full sweep + post-process
#   bash cluster/submit_power_pipeline.sh --sampler=numpyro    # JAX NUTS backend
#
# Quick start on M3 (first time):
#   cd /scratch/fc37/$USER
#   git clone git@github.com:adammanoogian/psilocybin-prl-analyses.git
#   cd psilocybin-prl-analyses
#   bash cluster/submit_power_pipeline.sh --setup --benchmark
#
# Subsequent sessions:
#   cd /scratch/fc37/$USER/psilocybin-prl-analyses
#   bash cluster/submit_power_pipeline.sh --setup              # pull + update env
#   bash cluster/submit_power_pipeline.sh                      # submit sweep
# =============================================================================

set -euo pipefail

# =============================================================================
# Parse arguments
# =============================================================================
DRY_RUN=0
BENCHMARK=0
SETUP=false
SKIP_PUSH=false
SAMPLER="pymc"

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup|-s)
            SETUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --benchmark)
            BENCHMARK=1
            shift
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --sampler=*)
            SAMPLER="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: bash cluster/submit_power_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --setup, -s          Git pull + create/update prl_gpu env first"
            echo "  --dry-run            Placeholder parquet (no MCMC)"
            echo "  --benchmark          JIT compile + single fit timing (1 GPU, ~15 min)"
            echo "  --skip-push          Skip auto-push results to git"
            echo "  --sampler=BACKEND    pymc (default) or numpyro"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Examples:"
            echo "  --setup --benchmark  First time: pull, env, then benchmark"
            echo "  --setup              Update code + env, no job submitted"
            echo "  (no flags)           Full power sweep + post-processing"
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

# =============================================================================
# Setup: git pull + conda env
# =============================================================================
if [[ "$SETUP" == true ]]; then
    echo "============================================================"
    echo "Setup: Git Pull + Environment"
    echo "============================================================"

    # Git pull
    echo ""
    echo "--- Git pull ---"
    git fetch origin
    git pull --ff-only origin main
    echo "HEAD: $(git log --oneline -1)"

    # Env setup (delegates to 00_setup_env_gpu.sh --update)
    echo ""
    echo "--- Conda environment ---"
    bash cluster/00_setup_env_gpu.sh --update

    echo ""
    echo "Setup complete."

    # If no job mode requested, stop here
    if [[ "$BENCHMARK" == "0" && "$DRY_RUN" == "0" ]]; then
        echo ""
        echo "Next steps:"
        echo "  bash cluster/submit_power_pipeline.sh --benchmark  # time a single fit"
        echo "  bash cluster/submit_power_pipeline.sh              # full sweep"
        exit 0
    fi

    echo ""
    echo "Continuing to job submission..."
fi

# =============================================================================
# Job submission
# =============================================================================
mkdir -p cluster/logs results/power

echo "============================================================"
echo "Power Analysis Pipeline"
echo "============================================================"
echo "Project:   $PROJECT_ROOT"
echo "Sampler:   $SAMPLER"
echo "Dry run:   $DRY_RUN"
echo "Benchmark: $BENCHMARK"
echo "Start:     $(date)"
echo "============================================================"

# --- Wave 1: Power Sweep (GPU array job) ---
echo ""
echo "--- Wave 1: Power Sweep (GPU) ---"

SWEEP_EXTRA_ARGS=""
if [[ "$BENCHMARK" == "1" ]]; then
    SWEEP_EXTRA_ARGS="--array=0 --time=00:30:00"
    echo "(benchmark: single chunk, 30min walltime)"
fi

SWEEP_JOBID=$(sbatch --parsable \
    --export="ALL,DRY_RUN=${DRY_RUN},BENCHMARK=${BENCHMARK},SAMPLER=${SAMPLER}" \
    ${SWEEP_EXTRA_ARGS} \
    cluster/08_power_sweep.slurm)

if [[ "$BENCHMARK" == "1" ]]; then
    echo "  Benchmark -> Job $SWEEP_JOBID"
else
    echo "  Sweep -> Job $SWEEP_JOBID (array 0-2)"
fi

# --- Wave 2: Post-Processing (CPU, depends on sweep) ---
POSTPROC_JOBID=""
if [[ "$BENCHMARK" == "0" && "$DRY_RUN" == "0" ]]; then
    echo ""
    echo "--- Wave 2: Post-Processing (CPU, afterok:${SWEEP_JOBID}) ---"

    POSTPROC_JOBID=$(sbatch --parsable \
        --dependency=afterok:${SWEEP_JOBID} \
        cluster/09_power_postprocess.slurm)
    echo "  Post-process -> Job $POSTPROC_JOBID"
fi

# --- Wave 3: Push (optional) ---
PUSH_JOBID=""
if [[ "$SKIP_PUSH" == "false" && -n "$POSTPROC_JOBID" ]]; then
    echo ""
    echo "--- Wave 3: Push Results (afterany:${POSTPROC_JOBID}) ---"

    PUSH_JOBID=$(sbatch --parsable \
        --dependency=afterany:${POSTPROC_JOBID} \
        --export="ALL,PARENT_JOBS=${SWEEP_JOBID} ${POSTPROC_JOBID}" \
        cluster/99_push_results.slurm)
    echo "  Push -> Job $PUSH_JOBID"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Submitted"
echo "============================================================"
echo "  Wave 1: $SWEEP_JOBID (GPU)"
[[ -n "$POSTPROC_JOBID" ]] && echo "  Wave 2: $POSTPROC_JOBID (CPU)"
[[ -n "$PUSH_JOBID" ]] && echo "  Wave 3: $PUSH_JOBID (push)"
echo ""
echo "  squeue -u $USER"
echo "  tail -f cluster/logs/power_${SWEEP_JOBID}_*.out"
[[ -n "$POSTPROC_JOBID" ]] && echo "  tail -f cluster/logs/postprocess_${POSTPROC_JOBID}.out"
echo ""
if [[ "$BENCHMARK" == "1" ]]; then
    echo "After benchmark completes:"
    echo "  cat results/power/benchmark.json"
fi
echo "============================================================"
