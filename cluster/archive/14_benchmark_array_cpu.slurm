#!/bin/bash
# =============================================================================
# SLURM Array Job: Phase 14.2 variant 4 — CPU bench, P=50 chunks
# =============================================================================
# CPU partition equivalent of cluster/14_benchmark_array_gpu.slurm.  Same
# Python entry, same chunk-50 participant split.  Submit whichever queue
# is shorter that day; statistical answer is identical (BlackJAX path
# uses independent priors per PS — see _build_log_posterior in
# src/prl_hgf/fitting/hierarchical.py).
#
# Usage:
#   sbatch cluster/14_benchmark_array_cpu.slurm
#   sbatch --export=ALL,MAX_TREE_DEPTH=8 cluster/14_benchmark_array_cpu.slurm
#
# Manual flush after array completes:
#   git add logs/bench14arrcpu_${SLURM_ARRAY_JOB_ID}_*.{out,err} \
#           results/power/benchmark_batched_chunk_*.json
#   git commit -m "bench14 array cpu: ${SLURM_ARRAY_JOB_ID}"
#   git push origin main
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_bench14arrcpu
#SBATCH --output=logs/bench14arrcpu_%A_%a.out
#SBATCH --error=logs/bench14arrcpu_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Run-commit provenance: pin every log to the commit it ran from
echo "============================================================"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Run job:    ${SLURM_JOB_ID:-<no-slurm>} on $(hostname)"
echo "============================================================"

mkdir -p cluster/logs models/power
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV} conda environment"
    exit 1
}

# Force CPU platform — separate cache so cross-architecture HLO don't collide.
export JAX_PLATFORMS=cpu
unset CUDA_VISIBLE_DEVICES

export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null

mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# =============================================================================
# Variant knobs
# =============================================================================
export SAMPLER="${SAMPLER:-blackjax}"
export MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
export PARTICIPANT_CHUNK_COUNT="${PARTICIPANT_CHUNK_COUNT:-6}"

# Fitting configuration (FitConfig YAML — Phase 28)
FIT_CONFIG="${FIT_CONFIG:-configs/fit/default_2level.yaml}"

# variant 3: tight ω₃ prior
export TIGHT_OMEGA3="${TIGHT_OMEGA3:-0}"
_TIGHT_OMEGA3_FLAG=""
if [[ "${TIGHT_OMEGA3}" == "1" ]]; then
    _TIGHT_OMEGA3_FLAG="--tight-omega3-prior"
fi

# variant 5: Laplace-only
export LAPLACE_ONLY="${LAPLACE_ONLY:-0}"
_LAPLACE_ONLY_FLAG=""
if [[ "${LAPLACE_ONLY}" == "1" ]]; then
    _LAPLACE_ONLY_FLAG="--laplace-only"
fi

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 14.2 variant 4 — Array bench (CPU)"
echo "============================================================"
echo "Array job ID:   ${SLURM_ARRAY_JOB_ID:-local}"
echo "Array task ID:  ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Sampler:        ${SAMPLER}"
echo "Max tree depth: ${MAX_TREE_DEPTH}"
echo "Fit config:     ${FIT_CONFIG}"
echo "Chunk count:    ${PARTICIPANT_CHUNK_COUNT}"
echo "Chunk id:       ${SLURM_ARRAY_TASK_ID:-0}"
echo "Tight omega3:   ${TIGHT_OMEGA3}"
echo "Laplace only:   ${LAPLACE_ONLY}"
echo "Start:          $(date)"
echo "============================================================"
echo ""

python -c "import jax; print(f'JAX devices: {jax.devices()} backend: {jax.default_backend()}')"
echo ""

python scripts/03_pre_analysis/03_run_power_iteration.py \
    --benchmark \
    --chunk-id 0 \
    --job-id "${SLURM_ARRAY_JOB_ID:-bench14arrcpu}_${SLURM_ARRAY_TASK_ID:-0}" \
    --fit-config "${FIT_CONFIG}" \
    --sampler "${SAMPLER}" \
    --max-tree-depth "${MAX_TREE_DEPTH}" \
    --participant-chunk-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --participant-chunk-count "${PARTICIPANT_CHUNK_COUNT}" \
    ${_TIGHT_OMEGA3_FLAG} \
    ${_LAPLACE_ONLY_FLAG}

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID} COMPLETE"
else
    echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID} FAILED (exit $EXIT_CODE)"
fi
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
