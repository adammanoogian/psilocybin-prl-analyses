#!/bin/bash
# =============================================================================
# SLURM Job: Phase 25-03 — ELK Reference Reproduction on GPU
# =============================================================================
# Runs the quasi-ELK coordinated-turn reproduction benchmark on a GPU node:
#   1. Verifies GPU device is available (fails early if CPU-only).
#   2. Runs reproduce_reference.py with JAX_ENABLE_X64=1 (mandatory).
#   3. Writes results JSON to:
#      logs/phase25/
#      reproduction_benchmark.json (overwrites CPU run; GPU JSON is authoritative).
#   4. Logs stdout to results/reproduction_gpu_<JOBID>.out.
#   5. Auto-commits and pushes results to git (mirrors Phase 14 pattern).
#
# Usage:
#   sbatch cluster/25_reproduce_reference_gpu.slurm
#
# Expected outcome (PASS gate):
#   rel_err_vs_sequential < 1e-5  (numerical agreement)
#   K_observed <= 15              (convergence rate)
#   GPU speedup informational only (not gating per plan override note)
#
# Derived from: cluster/14_benchmark_gpu.slurm (Phase 14 GPU benchmark template)
# Plan: 25-03, Phase 25 — Parallel-scan acceleration of HGF likelihood
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=25-reproduce-ref
#SBATCH --output=logs/phase25/reproduction_gpu_%j.out
#SBATCH --error=logs/phase25/reproduction_gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

# =============================================================================
# Environment Setup (mirrors 14_benchmark_gpu.slurm pattern)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
mkdir -p logs/phase25
export PYTHONUNBUFFERED=1

# Activate ds_env
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# =============================================================================
# Add pip-installed ptxas to PATH (from Phase 14 template — GPU compilation)
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
# JAX float64 mode (HIGH SEVERITY — must be set before jax import; Risk #6)
# =============================================================================
export JAX_ENABLE_X64=1

# JAX compilation cache (shared across nodes — Phase 21 VERDICT §c pattern)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Show actual VRAM usage, not preallocated pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 25-03: ELK Reference Reproduction GPU Benchmark"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):  ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "JAX_ENABLE_X64: ${JAX_ENABLE_X64}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:   $(date)"
echo "============================================================"
echo ""

# =============================================================================
# GPU Verification (fail early if no GPU)
# =============================================================================
python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — this script requires a GPU node!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
print(f'JAX version: {jax.__version__}')
"
if [[ $? -ne 0 ]]; then
    echo "GPU check failed. Exiting."
    exit 1
fi
echo ""

# =============================================================================
# Re-install ELK dependencies if needed (quasi-DEER uses only JAX;
# dynamax and flax are NOT required for reproduce_reference.py)
# =============================================================================
# The reproduce_reference.py script is a pure-JAX implementation of
# quasi-DEER; it does NOT import elk, dynamax, or flax.
# No additional pip installs needed beyond ds_env base JAX.

# =============================================================================
# Run Reproduction Script
# =============================================================================
echo "Running quasi-ELK coordinated-turn reproduction (T=500, D=4)..."
echo ""

python scripts/_maintenance/phase25_proto/reproduce_reference.py

EXIT_CODE=$?

echo ""
echo "============================================================"

RESULTS_JSON="logs/phase25/reproduction_benchmark.json"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "REPRODUCTION COMPLETE (PASS)"
    echo ""
    if [[ -f "$RESULTS_JSON" ]]; then
        echo "Results:"
        cat "$RESULTS_JSON" | python -m json.tool 2>/dev/null || cat "$RESULTS_JSON"
    fi
else
    echo "REPRODUCTION FAILED (exit code: $EXIT_CODE)"
    echo "Check: logs/phase25/reproduction_gpu_${SLURM_JOB_ID}.err"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
