#!/bin/bash
# =============================================================================
# SLURM Job: Phase 25-04 — HGF Prototype GPU Timing Benchmark
# =============================================================================
# Benchmarks parallel-scan vs sequential HGF log-likelihood at three
# (P, T) shapes across three init strategies (cold, linear, cached).
#
# PREREQUISITE: Repair Phase 14.1-03 pip-install issue on M3 before submitting:
#   cd $PROJECT_ROOT && pip install -e . --quiet
# Verify: python -c "import prl_hgf; print('OK')"
#
# Results written to: logs/phase25/proto_timing.json
# Incremental write: JSON is updated after each cell — partial results survive
# SLURM timeout (important for the T=2000 cells).
#
# Usage:
#   # From project root on M3:
#   sbatch cluster/25_proto_timing_gpu.slurm
#
# Expected speedups (per 25-RESEARCH.md §9 + 25-04 CPU baseline):
#   (P=50,  T=420)  cold init: 0.5-2x (marginal; T near breakeven on GPU)
#   (P=300, T=420)  cold init: possibly higher (better GPU occupancy)
#   (P=50,  T=2000) cold init: 5-20x  (firmly in win zone on GPU)
#   cached init:    lower K (1-3) across all shapes
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=25-proto-timing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/phase25/proto_timing_%j.out
#SBATCH --error=logs/phase25/proto_timing_%j.err

# =============================================================================
# Environment Setup (mirrors cluster/14_benchmark_gpu.slurm)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p logs/phase25
export PYTHONUNBUFFERED=1

# Activate ds_env
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# Repair editable install (Phase 14.1-03 pip-install regression guard)
echo "Verifying prl_hgf editable install..."
python -c "import prl_hgf" 2>/dev/null || {
    echo "Editable install missing — running pip install -e . ..."
    pip install -e . --quiet
    python -c "import prl_hgf" || {
        echo "ERROR: prl_hgf not importable after pip install -e ."
        exit 1
    }
}
echo "prl_hgf import OK"

# =============================================================================
# ptxas PATH (nvidia-cuda-nvcc-cu12 — mirrors cluster/14_benchmark_gpu.slurm)
# =============================================================================
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
    echo "Added ptxas to PATH: $PTXAS_BIN"
fi

# =============================================================================
# JAX configuration
# =============================================================================
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Surface every XLA compile event in stderr so we can see exactly where the
# wall-clock goes (compile vs run) under the 25-04-rerun benchmark.
export JAX_LOG_COMPILES=1
# Also surface compile cache hits/misses so we can verify the prototype
# refactor really collapses to one compile per (T, init) cell.
export JAX_DEBUG_LOG_MODULES="jax._src.compilation_cache,jax._src.dispatch"

# JAX compilation cache (Lustre scratch, shared across nodes).  Bumped to a
# fresh dir name so the post-refactor benchmark gets a clean cold-cache run
# (the prior shared_25proto dir holds artifacts keyed by the pre-refactor
# per-participant scan_fn closures and is no longer relevant).
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared_25proto_v2"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 25-04 HGF Prototype GPU Timing Benchmark"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):  ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:   $(date)"
echo "============================================================"
echo ""

# Verify GPU is present
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
# Run timing benchmark
# =============================================================================
echo "Running 9-cell timing benchmark (3 shapes x 3 init strategies)..."
echo "Results written incrementally to logs/phase25/proto_timing.json"
echo ""

python scripts/_maintenance/phase25_proto/proto_timing_benchmark.py

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "BENCHMARK COMPLETE"
    echo ""
    if [[ -f logs/phase25/proto_timing.json ]]; then
        echo "Results summary:"
        python -m json.tool logs/phase25/proto_timing.json 2>/dev/null || \
            cat logs/phase25/proto_timing.json
    fi
else
    echo "BENCHMARK FAILED (exit code: $EXIT_CODE)"
    echo "Check: logs/phase25/proto_timing_${SLURM_JOB_ID}.err"
    echo "Partial results (if any):"
    cat logs/phase25/proto_timing.json 2>/dev/null | python -m json.tool 2>/dev/null || true
fi

echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
