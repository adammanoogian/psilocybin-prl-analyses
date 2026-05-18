#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — GPU Benchmark + Decision Gate (BENCH-01..05)
# =============================================================================
# Runs the v1.2 batched hierarchical benchmark on a GPU node:
#   1. JAX compilation cache warm/cold test (BENCH-05)
#   2. Full batched iteration timing (BENCH-01, BENCH-04)
#   3. Decision gate: GPU vs CPU comp recommendation (BENCH-02)
#   4. Writes models/power/benchmark_batched.json
#   5. Appends decision to .planning/STATE.md
#
# Usage:
#   sbatch cluster/14_benchmark_gpu.slurm                              # BlackJAX (default, Phase 17 production sampler)
#   sbatch --export=ALL,SAMPLER=numpyro cluster/14_benchmark_gpu.slurm # NumPyro baseline (legacy)
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_bench14
#SBATCH --output=logs/bench14_%j.out
#SBATCH --error=logs/bench14_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
# Override gres at submit time for multi-GPU pmap chains:
#   sbatch --gres=gpu:4 --export=ALL,FIT_CHAINS=4 cluster/14_benchmark_gpu.slurm
# When n_devices >= n_chains the code dispatches pmap (one chain per GPU)
# instead of vmap (all chains share one device).

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

# Activate ds_env (or override via CONDA_ENV=ds_env_v10 at sbatch --export time)
_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV} conda environment"
    exit 1
}

# GPU deps: pip install -r cluster/requirements-gpu.txt

# =============================================================================
# Add pip-installed ptxas to PATH (nvidia-cuda-nvcc-cu12 puts it in site-packages)
# =============================================================================
PTXAS_BIN=$(python -c '
import os
for mod in ("nvidia.cu13", "nvidia.cuda_nvcc"):
    try:
        m = __import__(mod, fromlist=["x"])
        d = os.path.join(os.path.dirname(m.__file__), "bin")
        if os.path.isdir(d):
            print(d); break
    except ImportError:
        pass
' 2>/dev/null)
if [[ -n "$PTXAS_BIN" && -d "$PTXAS_BIN" ]]; then
    export PATH="$PTXAS_BIN:$PATH"
    echo "Added ptxas to PATH: $PTXAS_BIN"
fi

# =============================================================================
# CUDA / PTX Version Compatibility Check (NPRO-03)
# =============================================================================
check_cuda_compat() {
    DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [[ -z "$DRIVER_CUDA_VER" ]]; then
        echo "WARNING: nvidia-smi not available or no GPU detected"
        return 1
    fi

    PTXAS_VER=$(ptxas --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    if [[ -z "$PTXAS_VER" ]]; then
        echo "WARNING: ptxas not found in PATH"
        return 1
    fi

    echo "Driver CUDA version: $DRIVER_CUDA_VER"
    echo "ptxas CUDA version:  $PTXAS_VER"

    DRIVER_MAJOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f1)
    DRIVER_MINOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f2)
    PTXAS_MAJOR=$(echo "$PTXAS_VER" | cut -d. -f1)
    PTXAS_MINOR=$(echo "$PTXAS_VER" | cut -d. -f2)

    if (( PTXAS_MAJOR > DRIVER_MAJOR )) || \
       (( PTXAS_MAJOR == DRIVER_MAJOR && PTXAS_MINOR > DRIVER_MINOR )); then
        echo ""
        echo "ERROR: ptxas CUDA version ($PTXAS_VER) > driver CUDA version ($DRIVER_CUDA_VER)"
        echo "  XLA will disable parallel compilation, causing 2-5x slower JIT."
        echo "  FIX: pip install -r cluster/requirements-gpu.txt"
        return 1
    fi

    echo "OK: ptxas version <= driver version (XLA parallel compilation enabled)"
    return 0
}

echo ""
echo "=== CUDA Environment Check ==="
check_cuda_compat || echo "WARNING: CUDA compatibility issue detected (non-fatal, continuing)"
echo ""

# JAX compilation cache (shared across nodes via Lustre scratch; cross-SLURM-job persistence — Phase 21 VERDICT §c)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
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

# XLA autotuning cache (persist autotuning results across jobs)
_XLA_AUTOTUNE_DIR="/scratch/${_PROJECT}/${USER}/.xla_autotune_cache"
mkdir -p "$_XLA_AUTOTUNE_DIR" 2>/dev/null
_XLA_AUTOTUNE_FILE="${_XLA_AUTOTUNE_DIR}/results.pb"
export XLA_FLAGS="--xla_gpu_dump_autotune_results_to=${_XLA_AUTOTUNE_FILE}"
if [[ -f "$_XLA_AUTOTUNE_FILE" ]]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_load_autotune_results_from=${_XLA_AUTOTUNE_FILE}"
fi

# Show actual VRAM usage, not preallocated pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Sampler selection (default: blackjax — Phase 17 production sampler)
# Set BEFORE the banner so echoed value reflects the actual invocation.
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

# Chain count — should match number of GPUs allocated (--gres=gpu:N) so the
# pmap path activates (one chain per device). Defaults to 4 to match the
# historical vmap-on-1-GPU configuration.
export FIT_CHAINS="${FIT_CHAINS:-4}"

# Fitting configuration (FitConfig YAML — Phase 28)
FIT_CONFIG="${FIT_CONFIG:-configs/fit/default_2level.yaml}"

# Precision: JAX_X64=1 sets fp64 (numerically correct for HGF integrators;
# may eliminate the step_size→1e-10 underflow seen in fp32 runs). Adds the
# --enable-x64 CLI flag, which the script picks up *before* importing jax.
export JAX_X64="${JAX_X64:-0}"
_X64_FLAG=""
if [[ "${JAX_X64}" == "1" ]]; then
    _X64_FLAG="--enable-x64"
fi

# variant 3: tight ω₃ prior — Normal(-6, 1.0) instead of TruncatedNormal(-6, 2)
export TIGHT_OMEGA3="${TIGHT_OMEGA3:-0}"
_TIGHT_OMEGA3_FLAG=""
if [[ "${TIGHT_OMEGA3}" == "1" ]]; then
    _TIGHT_OMEGA3_FLAG="--tight-omega3-prior"
fi

# variant 5: Laplace-only (skip NUTS sampling, just MAP + Hessian)
export LAPLACE_ONLY="${LAPLACE_ONLY:-0}"
_LAPLACE_ONLY_FLAG=""
if [[ "${LAPLACE_ONLY}" == "1" ]]; then
    _LAPLACE_ONLY_FLAG="--laplace-only"
fi

# Capability-map override: replace power.n_per_group_grid for cliff-edge sweep.
# Comma-separated ints, e.g. N_PER_GROUP_OVERRIDES="8,12,17,25". Phase 1 warmup
# uses MAX of override; Phase 2 measurement loop walks all values. Empty/unset
# = use the YAML config grid unchanged. See memory project_end_goal_capability_map.md.
export N_PER_GROUP_OVERRIDES="${N_PER_GROUP_OVERRIDES:-}"
_NPG_OVERRIDE_FLAG=""
if [[ -n "${N_PER_GROUP_OVERRIDES}" ]]; then
    _NPG_OVERRIDE_FLAG="--n-per-group-overrides ${N_PER_GROUP_OVERRIDES}"
fi

# Capability-map: model selector for the --benchmark path.
# BENCH_MODEL=hgf_2level fills the 2-level NUTS gap; default hgf_3level
# preserves Phase 14/21 benchmark history.
export BENCH_MODEL="${BENCH_MODEL:-hgf_3level}"

# Capability-map: output-filename suffix so concurrent jobs writing
# different (model, grid) configurations don't clobber each other on
# models/power/benchmark_batched.json. Default empty = canonical filename
# (back-compat). Example: BENCH_OUT_SUFFIX=_hgf_2level → benchmark_batched_hgf_2level.json.
export BENCH_OUT_SUFFIX="${BENCH_OUT_SUFFIX:-}"

# =============================================================================
# Verify GPU
# =============================================================================
echo "============================================================"
echo "Phase 14 GPU Benchmark"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):  ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Sampler:  ${SAMPLER}"
echo "Fit config: ${FIT_CONFIG}"
echo "Precision:  $([[ "${JAX_X64}" == "1" ]] && echo fp64 || echo fp32)  (JAX_X64=${JAX_X64})"
echo "Max tree depth: ${MAX_TREE_DEPTH}"
echo "Tight omega3:   ${TIGHT_OMEGA3}"
echo "Laplace only:   ${LAPLACE_ONLY}"
echo "N/group override: ${N_PER_GROUP_OVERRIDES:-(use config grid)}"
echo "Bench model:    ${BENCH_MODEL}"
echo "Out suffix:     ${BENCH_OUT_SUFFIX:-(none)}"
echo "Start:   $(date)"
echo "============================================================"
echo ""

python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — benchmark results will not be meaningful!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
"
echo ""

# =============================================================================
# Run Benchmark (BENCH-01, BENCH-02, BENCH-04, BENCH-05)
# =============================================================================
echo "Running batched hierarchical benchmark..."
echo ""

python scripts/03_pre_analysis/03_run_power_iteration.py \
    --benchmark \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-bench14}" \
    --fit-config "${FIT_CONFIG}" \
    --sampler "${SAMPLER}" \
    --max-tree-depth "${MAX_TREE_DEPTH}" \
    --benchmark-model "${BENCH_MODEL}" \
    --benchmark-out-suffix "${BENCH_OUT_SUFFIX}" \
    ${_X64_FLAG} \
    ${_TIGHT_OMEGA3_FLAG} \
    ${_LAPLACE_ONLY_FLAG} \
    ${_NPG_OVERRIDE_FLAG}

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
    echo "Check: logs/bench14_${SLURM_JOB_ID}.err"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
