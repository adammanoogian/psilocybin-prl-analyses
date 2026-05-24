#!/bin/bash
# =============================================================================
# SLURM Job: Phase 25-05 — Prototype vs Sequential NUTS Recovery (GPU)
# =============================================================================
# Runs BlackJAX NUTS recovery on a P=20 simulated cohort using BOTH the
# prototype parallel-scan likelihood and the sequential lax.scan baseline.
# Both backends use the SAME simulated cohort (seed=2505) so per-parameter
# RMSE/ESS/R-hat/divergence comparison is meaningful.
#
# Outputs (NetCDF chains):
#   logs/phase25/recovery_chains_sequential.nc
#   logs/phase25/recovery_chains_proto.nc
#
# PREREQUISITE
# ------------
# `proto_grad_check.py` must report PASS first.  Until the prototype's
# `_build_hgf_closures` is refactored to thread JAX-traced theta (mirroring
# `hierarchical.py::_single_logp_3level`), this job will exit with the
# blocker diagnostic from `proto_recovery_run.py::_check_differentiability_or_exit`.
#
# Usage:
#   sbatch cluster/25_proto_recovery_gpu.slurm
#
# Plan: 25-05 — Phase 25 Parallel-Scan Acceleration Research
# Derived from: cluster/14_benchmark_gpu.slurm (Phase 14 GPU template)
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=25-recovery-cmp
#SBATCH --output=logs/phase25/recovery_%j.out
#SBATCH --error=logs/phase25/recovery_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

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

# Repair editable install (Phase 14.1-03 regression guard)
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
# ptxas PATH (nvidia-cuda-nvcc-cu12 — mirrors 14_benchmark_gpu.slurm)
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

# Surface compile events in stderr so we can see whether NUTS compiles once
# per backend or once-per-(participant, chain).
export JAX_LOG_COMPILES=1

# JAX compilation cache (Lustre scratch, shared across nodes)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared_25recovery"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 25-05: NUTS Recovery Comparison (proto vs sequential)"
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

# Verify GPU
python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — recovery results will not be meaningful!')
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
# Pre-flight: differentiability gate (proto_grad_check.py must PASS)
# =============================================================================
echo "Pre-flight: running proto_grad_check.py to confirm differentiability..."
python scripts/_maintenance/phase25_proto/proto_grad_check.py
GRAD_EXIT=$?
if [[ $GRAD_EXIT -ne 0 ]]; then
    echo ""
    echo "ABORT: proto_grad_check.py failed (exit ${GRAD_EXIT})."
    echo "  See logs/phase25/proto_grad_check.log for the failure mode."
    echo "  Recovery run will not produce meaningful results until the prototype"
    echo "  is refactored to thread JAX-traced theta (see hierarchical.py::"
    echo "  _single_logp_3level for the production shallow-copy pattern)."
    exit ${GRAD_EXIT}
fi
echo "proto_grad_check.py PASS — proceeding to NUTS recovery."
echo ""

# =============================================================================
# Recovery run: sequential first (baseline), then proto
# =============================================================================
RESULTS_DIR="logs/phase25"
SEQ_NC="${RESULTS_DIR}/recovery_chains_sequential.nc"
PRO_NC="${RESULTS_DIR}/recovery_chains_proto.nc"

echo "[1/2] Sequential baseline NUTS..."
python scripts/_maintenance/phase25_proto/proto_recovery_run.py \
    --backend sequential \
    --seed 2505 \
    --n-participants 20 \
    --n-trials 420 \
    --num-warmup 1000 \
    --num-samples 1000 \
    --num-chains 4 \
    --out "${SEQ_NC}" \
    --allow-unblocked
SEQ_EXIT=$?

echo ""
echo "[2/2] Prototype parallel-scan NUTS..."
python scripts/_maintenance/phase25_proto/proto_recovery_run.py \
    --backend proto \
    --seed 2505 \
    --n-participants 20 \
    --n-trials 420 \
    --num-warmup 1000 \
    --num-samples 1000 \
    --num-chains 4 \
    --out "${PRO_NC}" \
    --allow-unblocked
PRO_EXIT=$?

echo ""
echo "============================================================"
if [[ $SEQ_EXIT -eq 0 && $PRO_EXIT -eq 0 ]]; then
    echo "RECOVERY COMPLETE (both backends)"
    ls -lh "${SEQ_NC}" "${PRO_NC}" 2>/dev/null || true
else
    echo "RECOVERY FAILED — sequential_exit=${SEQ_EXIT} proto_exit=${PRO_EXIT}"
fi

echo ""
echo "End: $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

# Overall exit: nonzero if either backend failed
if [[ $SEQ_EXIT -ne 0 ]]; then
    exit ${SEQ_EXIT}
fi
exit ${PRO_EXIT}
