#!/bin/bash
# =============================================================================
# SLURM Job: Phase 16/17 — CPU Smoke Test (BlackJAX NUTS diagnostics)
# =============================================================================
# CPU-only counterpart to 16_smoke_test_gpu.slurm.  Used to isolate whether
# the cluster-observed slow warm sampling (~1174 s for 10 draws on L40S) is
# driven by compile time, XLA/GPU overhead, or the sampler itself hitting
# max_tree_depth.
#
# Runs at the same cohort size (N=5/group, 30 participant-sessions, 2 chains,
# 10 draws / 10 tune) but on CPU, exercising the same fit_batch_hierarchical
# path.  Prints per-draw NUTS diagnostics (integration_steps mean/p50/p95/max,
# divergence rate, acceptance) from the new --smoke-test instrumentation.
#
# Precision toggle:
#   Set env PRL_ENABLE_X64=1 when submitting to force jax_enable_x64=True.
#   Default (unset/0) forces jax_enable_x64=False so the sampler runs in
#   fp32 — this is the diagnostic we want, because pytensor's jax link
#   silently enables x64 on import and the existing GPU smoke test has
#   been running in fp64 without anyone noticing.
#
#   sbatch cluster/16_smoke_test_cpu.slurm                 # fp32 (default)
#   sbatch --export=PRL_ENABLE_X64=1 cluster/16_smoke_test_cpu.slurm   # fp64
#
# Output:
#   results/power/smoke_test_cpu_${PRECISION}.json   (fp32 or fp64)
#   cluster/logs/smoke16cpu_${SLURM_JOB_ID}.out/.err
# =============================================================================

#SBATCH --job-name=prl_smoke16cpu
#SBATCH --output=cluster/logs/smoke16cpu_%j.out
#SBATCH --error=cluster/logs/smoke16cpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs results/power
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# Force CPU platform.  No GPU/ptxas checks — CPU path has no XLA autotune
# cache, no CUDA compat concerns.
export JAX_PLATFORMS=cpu
unset CUDA_VISIBLE_DEVICES

# Persistent JAX compilation cache (CPU-specific path so we don't poison
# cross-architecture with the GPU cache)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/${SLURMD_NODENAME:-local}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Log compilations (goes to stderr, matches GPU smoke test)
export JAX_LOG_COMPILES=1

# Precision selection: PRL_ENABLE_X64=1 enables fp64, anything else → fp32
if [[ "${PRL_ENABLE_X64:-0}" == "1" ]]; then
    EXTRA_ARGS="--enable-x64"
    PRECISION="fp64"
else
    EXTRA_ARGS=""
    PRECISION="fp32"
fi

# MCMC config: default to production-scale warmup + draws so the smoke
# test exercises the real adaptation budget (the original 10/10 was
# chosen for JIT-cache diagnostics, not sampler health).  Override via
# env at submission:  sbatch --export=ALL,PRL_SMOKE_TUNE=500,PRL_SMOKE_DRAWS=500 ...
export PRL_SMOKE_TUNE="${PRL_SMOKE_TUNE:-500}"
export PRL_SMOKE_DRAWS="${PRL_SMOKE_DRAWS:-500}"
export PRL_SMOKE_CHAINS="${PRL_SMOKE_CHAINS:-2}"
export PRL_SMOKE_N_PER_GROUP="${PRL_SMOKE_N_PER_GROUP:-5}"
MCMC_SUFFIX="tune${PRL_SMOKE_TUNE}_draws${PRL_SMOKE_DRAWS}"

# =============================================================================
# Run Smoke Test
# =============================================================================
echo ""
echo "============================================================"
echo "Phase 16/17 CPU Smoke Test (BlackJAX NUTS)"
echo "============================================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: comp (CPU only)"
echo "Precision: ${PRECISION}  (PRL_ENABLE_X64=${PRL_ENABLE_X64:-0})"
echo "MCMC:      tune=${PRL_SMOKE_TUNE} draws=${PRL_SMOKE_DRAWS} chains=${PRL_SMOKE_CHAINS} N/group=${PRL_SMOKE_N_PER_GROUP}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Start:     $(date)"
echo "============================================================"
echo ""

python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'jax_enable_x64 (pre-run): {jax.config.jax_enable_x64}')
"
echo ""

python scripts/08_run_power_iteration.py \
    --smoke-test \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-smoke16cpu}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

# Rename the JSON so fp32/fp64 × quick/production runs don't clobber each other
_RESULT_JSON="results/power/smoke_test_cpu_${PRECISION}_${MCMC_SUFFIX}.json"
if [[ -f "results/power/smoke_test.json" ]]; then
    mv "results/power/smoke_test.json" "${_RESULT_JSON}"
    echo "Renamed results → ${_RESULT_JSON}"
fi

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "CPU SMOKE TEST PASSED (BlackJAX NUTS, ${PRECISION})"
else
    echo "CPU SMOKE TEST FAILED (exit code: $EXIT_CODE, ${PRECISION})"
    echo "  Check: cluster/logs/smoke16cpu_${SLURM_JOB_ID}.err"
fi
echo "End:       $(date)"
echo "============================================================"

# =============================================================================
# Auto-push logs and results to git (mirrors the GPU smoke test)
# =============================================================================
echo ""
echo "Pushing CPU smoke test results to git..."
_SMOKE_LOG_OUT="cluster/logs/smoke16cpu_${SLURM_JOB_ID}.out"
_SMOKE_LOG_ERR="cluster/logs/smoke16cpu_${SLURM_JOB_ID}.err"

git add "${_SMOKE_LOG_OUT}" "${_SMOKE_LOG_ERR}" "${_RESULT_JSON}" 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "smoke cpu ${PRECISION} ${MCMC_SUFFIX} (job ${SLURM_JOB_ID})"
    if git push origin main 2>/dev/null; then
        echo "Pushed CPU smoke test results to origin/main"
    else
        echo "WARNING: git push failed — results committed locally only"
    fi
else
    echo "No new files to push"
fi

exit $EXIT_CODE
