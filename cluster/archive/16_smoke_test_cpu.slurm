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
#   models/power/smoke_test_cpu_${PRECISION}.json   (fp32 or fp64)
#   logs/smoke16cpu_${SLURM_JOB_ID}.out/.err
# =============================================================================

#SBATCH --job-name=prl_smoke16cpu
#SBATCH --output=logs/smoke16cpu_%j.out
#SBATCH --error=logs/smoke16cpu_%j.err
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

# Force CPU platform.  No GPU/ptxas checks — CPU path has no XLA autotune
# cache, no CUDA compat concerns.
export JAX_PLATFORMS=cpu
unset CUDA_VISIBLE_DEVICES

# Persistent JAX compilation cache (CPU-specific path so we don't poison
# cross-architecture with the GPU cache)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Pre-warm ArviZ daily-warning file to defeat import-time race (14.1-07 follow-up)
# arviz/__init__.py _warn_once_per_day() skips _atomic_write_text ONLY IF the
# stamp file already contains today's ISO date. Writing today's date here makes
# the racey tmp.replace(path) code path unreachable under parallel SLURM launches.
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

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
# Periodic progress logging: emit NUTS diagnostics every N draws during
# sampling.  Intended for long runs where we want to kill the job early
# once takeaway is clear.  0 disables.  Default 25 → ~20 log lines per
# 500-draw run.
export PRL_SMOKE_LOG_EVERY="${PRL_SMOKE_LOG_EVERY:-25}"
# Model sequence.  Default: run 2-level first (primary target, no ω₃×κ
# geometry issue), then 3-level with κ frozen (tests whether freezing
# κ collapses the NUTS tree-depth saturation).  Override with a single
# model name to run only one:  PRL_SMOKE_MODELS="hgf_2level"
export PRL_SMOKE_MODELS="${PRL_SMOKE_MODELS:-hgf_2level hgf_3level}"
MCMC_SUFFIX="tune${PRL_SMOKE_TUNE}_draws${PRL_SMOKE_DRAWS}"

# =============================================================================
# Run Smoke Test
# =============================================================================
echo ""
echo "============================================================"
echo "Phase 16/17 CPU Smoke Test (BlackJAX NUTS)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition:  comp (CPU only)"
echo "Precision:  ${PRECISION}  (PRL_ENABLE_X64=${PRL_ENABLE_X64:-0})"
echo "MCMC:       tune=${PRL_SMOKE_TUNE} draws=${PRL_SMOKE_DRAWS} chains=${PRL_SMOKE_CHAINS} N/group=${PRL_SMOKE_N_PER_GROUP}"
echo "log_every:  ${PRL_SMOKE_LOG_EVERY} (0=off; per-draw NUTS diagnostics)"
echo "Models:     ${PRL_SMOKE_MODELS}"
echo "κ:          FROZEN at 1.0 for 3-level (see _KAPPA_FIXED in hierarchical.py)"
echo "JAX cache:  ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Start:      $(date)"
echo "============================================================"
echo ""

python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'jax_enable_x64 (pre-run): {jax.config.jax_enable_x64}')
"
echo ""

# Loop over requested models (default runs 2-level then 3-level).  Each
# invocation writes models/power/smoke_test.json which we then rename
# to carry the model name alongside precision + MCMC scale, so multiple
# runs in one SLURM job don't clobber each other.
OVERALL_EXIT=0
_ALL_RESULT_JSONS=()
for _MODEL in ${PRL_SMOKE_MODELS}; do
    export PRL_SMOKE_MODEL="${_MODEL}"
    echo ""
    echo "############################################################"
    echo "# Running smoke for model: ${_MODEL}"
    echo "# (${PRECISION}, tune=${PRL_SMOKE_TUNE}, draws=${PRL_SMOKE_DRAWS},"
    echo "#  log_every=${PRL_SMOKE_LOG_EVERY})"
    echo "############################################################"
    echo ""

    python scripts/03_pre_analysis/03_run_power_iteration.py \
        --smoke-test \
        --chunk-id 0 \
        --job-id "${SLURM_JOB_ID:-smoke16cpu}" \
        ${EXTRA_ARGS}

    _MODEL_EXIT=$?
    if [[ ${_MODEL_EXIT} -ne 0 ]]; then
        OVERALL_EXIT=${_MODEL_EXIT}
    fi

    # Rename per-model so fp32/fp64 × scale × model don't clobber
    _RESULT_JSON="models/power/smoke_test_cpu_${PRECISION}_${_MODEL}_${MCMC_SUFFIX}.json"
    if [[ -f "models/power/smoke_test.json" ]]; then
        mv "models/power/smoke_test.json" "${_RESULT_JSON}"
        echo "Renamed results → ${_RESULT_JSON}"
        _ALL_RESULT_JSONS+=("${_RESULT_JSON}")
    fi

    echo ""
    echo "--- Finished ${_MODEL} (exit=${_MODEL_EXIT}) at $(date) ---"
    echo ""
done

EXIT_CODE=${OVERALL_EXIT}

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "CPU SMOKE TEST PASSED (BlackJAX NUTS, ${PRECISION}, models: ${PRL_SMOKE_MODELS})"
else
    echo "CPU SMOKE TEST FAILED (exit code: $EXIT_CODE, ${PRECISION})"
    echo "  Check: logs/smoke16cpu_${SLURM_JOB_ID}.err"
fi
echo "End:       $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
