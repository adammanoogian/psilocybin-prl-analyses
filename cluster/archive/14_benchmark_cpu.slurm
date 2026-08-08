#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14.2 — CPU Benchmark variant (mirror of bench14_gpu)
# =============================================================================
# Same Python entry point as cluster/14_benchmark_gpu.slurm but runs on the
# `comp` partition (CPU only) so the variant comparison rig can submit to
# whichever queue is shorter that day.  Phase 14 was originally supposed to
# settle the GPU-vs-CPU decision gate; with the production-N warm-up upgrade
# (commit 6b7e6ed) the bench has never reached the decision step on either
# partition.  Variant-4 (P=50 SLURM array) is designed to make the bench
# tractable on both, finally closing the gate empirically.
#
# Usage:
#   sbatch cluster/14_benchmark_cpu.slurm                                   # default knobs
#   sbatch --export=ALL,MAX_TREE_DEPTH=8 cluster/14_benchmark_cpu.slurm     # variant 1 cap
#   sbatch --export=ALL,SAMPLER=numpyro cluster/14_benchmark_cpu.slurm      # NumPyro baseline
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_bench14cpu
#SBATCH --output=logs/bench14cpu_%j.out
#SBATCH --error=logs/bench14cpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16

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

# Force CPU platform.  Cross-architecture cache contamination would invalidate
# every entry, so use a CPU-specific cache dir (mirrors 16_smoke_test_cpu.slurm).
export JAX_PLATFORMS=cpu
unset CUDA_VISIBLE_DEVICES

export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Pre-warm ArviZ daily-warning file to defeat import-time race (14.1-07 follow-up)
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# =============================================================================
# Sampler selection (default: blackjax — Phase 17 production sampler)
# =============================================================================
export SAMPLER="${SAMPLER:-blackjax}"

# =============================================================================
# Phase 14.2 variant knobs — overridable from sbatch --export=...
#   MAX_TREE_DEPTH=8  : variant 1, caps each NUTS draw at 256 leapfrogs
#                       (BlackJAX default is 10 = 1024).  Persistent JAX
#                       cache picks up a new HLO entry; default 10 keeps
#                       the existing cache valid.
# =============================================================================
export MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
export FIT_CHAINS="${FIT_CHAINS:-4}"

# Fitting configuration (FitConfig YAML — Phase 28)
FIT_CONFIG="${FIT_CONFIG:-configs/fit/default_2level.yaml}"

# Precision: JAX_X64=1 sets fp64 (numerically correct for HGF integrators;
# may eliminate the step_size→1e-10 underflow seen in fp32 runs).
export JAX_X64="${JAX_X64:-0}"
_X64_FLAG=""
if [[ "${JAX_X64}" == "1" ]]; then
    _X64_FLAG="--enable-x64"
fi

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
echo "Phase 14 CPU Benchmark"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Sampler:  ${SAMPLER}"
echo "Fit config: ${FIT_CONFIG}"
echo "Precision:  $([[ "${JAX_X64}" == "1" ]] && echo fp64 || echo fp32)  (JAX_X64=${JAX_X64})"
echo "Max tree depth: ${MAX_TREE_DEPTH}"
echo "Tight omega3:   ${TIGHT_OMEGA3}"
echo "Laplace only:   ${LAPLACE_ONLY}"
echo "Start:   $(date)"
echo "============================================================"
echo ""

python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
print(f'JAX backend: {jax.default_backend()}')
"
echo ""

# =============================================================================
# Run Benchmark
# =============================================================================
echo "Running batched hierarchical benchmark..."
echo ""

python scripts/03_pre_analysis/03_run_power_iteration.py \
    --benchmark \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-bench14cpu}" \
    --fit-config "${FIT_CONFIG}" \
    --sampler "${SAMPLER}" \
    --max-tree-depth "${MAX_TREE_DEPTH}" \
    ${_X64_FLAG} \
    ${_TIGHT_OMEGA3_FLAG} \
    ${_LAPLACE_ONLY_FLAG}

EXIT_CODE=$?

echo ""
echo "============================================================"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "BENCHMARK COMPLETE"
    echo ""
    if [[ -f models/power/benchmark_batched.json ]]; then
        echo "Results:"
        cat models/power/benchmark_batched.json | python -m json.tool 2>/dev/null || \
            cat models/power/benchmark_batched.json
    fi
else
    echo "BENCHMARK FAILED (exit code: $EXIT_CODE)"
    echo "Check: logs/bench14cpu_${SLURM_JOB_ID}.err"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
