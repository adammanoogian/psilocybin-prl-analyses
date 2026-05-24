#!/bin/bash
# =============================================================================
# SLURM Array Job: Phase 14.2 variant 4 — GPU bench, P=50 chunks
# =============================================================================
# Splits the production cohort (2 groups × 50 N × 3 sessions = 300
# participant-sessions) into 6 chunks of one (group, session) cell each
# (50 PS per chunk).  Each array task runs the bench at chunk shape
# matching Phase 21's clean P=50 scan (~225s warm).  6 tasks in parallel
# turns the previously 12+ hour monolithic bench into ~minutes wall time.
#
# Cohort split is deterministic via subset_cohort_by_chunk
# (src/prl_hgf/power/iteration.py): same simulated cohort at every chunk
# count, deterministic slice selection, IID-equivalent to the un-chunked
# fit per the BlackJAX path (independent priors per PS).
#
# Usage:
#   sbatch cluster/14_benchmark_array_gpu.slurm                                # default knobs
#   sbatch --export=ALL,MAX_TREE_DEPTH=8 cluster/14_benchmark_array_gpu.slurm  # variant 1+4 stack
#
# Auto-push is disabled inside the array tasks to avoid 6 concurrent
# `git push` races.  After the array completes, run:
#   git add logs/bench14arr_${SLURM_ARRAY_JOB_ID}_*.{out,err} \
#           results/power/benchmark_batched_chunk_*.json
#   git commit -m "bench14 array: ${SLURM_ARRAY_JOB_ID}"
#   git push origin main
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_bench14arr
#SBATCH --output=logs/bench14arr_%A_%a.out
#SBATCH --error=logs/bench14arr_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

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

# Add pip-installed ptxas to PATH
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

# JAX shared compilation cache
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null

# ArviZ daily-warning prewarm (race fix from 14.1-07)
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# XLA autotuning cache
_XLA_AUTOTUNE_DIR="/scratch/${_PROJECT}/${USER}/.xla_autotune_cache"
mkdir -p "$_XLA_AUTOTUNE_DIR" 2>/dev/null
_XLA_AUTOTUNE_FILE="${_XLA_AUTOTUNE_DIR}/results.pb"
export XLA_FLAGS="--xla_gpu_dump_autotune_results_to=${_XLA_AUTOTUNE_FILE}"
if [[ -f "$_XLA_AUTOTUNE_FILE" ]]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_load_autotune_results_from=${_XLA_AUTOTUNE_FILE}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Variant knobs — overridable from sbatch --export=ALL,VAR=val,...
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
echo "Phase 14.2 variant 4 — Array bench (GPU)"
echo "============================================================"
echo "Array job ID:   ${SLURM_ARRAY_JOB_ID:-local}"
echo "Array task ID:  ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):         ${CUDA_VISIBLE_DEVICES:-none}"
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

python scripts/03_pre_analysis/03_run_power_iteration.py \
    --benchmark \
    --chunk-id 0 \
    --job-id "${SLURM_ARRAY_JOB_ID:-bench14arr}_${SLURM_ARRAY_TASK_ID:-0}" \
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

# Auto-push intentionally omitted — see header docstring for manual flush command.
exit $EXIT_CODE
