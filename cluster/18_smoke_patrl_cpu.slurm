#!/bin/bash
# =============================================================================
# SLURM Job: Phase 18 — PAT-RL Foundation Smoke Test (CPU)
# =============================================================================
# Cluster entry point for the Phase 18 PAT-RL foundation smoke:
#   simulate -> fit -> export -> sanity-check
#
# Runs scripts/12_smoke_patrl_foundation.py with production-scale MCMC
# defaults on the CPU partition (mirrors cluster/16_smoke_test_cpu.slurm
# environment setup exactly).  Override MCMC scale at submission:
#
#   sbatch cluster/18_smoke_patrl_cpu.slurm                    # defaults
#   sbatch --export=ALL,PRL_PATRL_SMOKE_N=10 cluster/18_smoke_patrl_cpu.slurm
#
# Separate JAX compilation cache path (.jax_cache_cpu/patrl/...) avoids
# collision with the pick_best_cue CPU cache.
#
# Output:
#   output/patrl_smoke_${SLURM_JOB_ID}/   (trajectory CSVs + param summary)
#   cluster/logs/patrl18smoke_${SLURM_JOB_ID}.out/.err
# =============================================================================

#SBATCH --job-name=prl_patrl18smoke
#SBATCH --output=cluster/logs/patrl18smoke_%j.out
#SBATCH --error=cluster/logs/patrl18smoke_%j.err
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

mkdir -p cluster/logs output/patrl_smoke
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# Force CPU platform.  No GPU/ptxas checks.
export JAX_PLATFORMS=cpu
unset CUDA_VISIBLE_DEVICES

# Persistent JAX compilation cache (separate path from pick_best_cue cache)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/patrl/${SLURMD_NODENAME:-local}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Log compilations (goes to stderr)
export JAX_LOG_COMPILES=1

# MCMC and smoke scale — override via --export at submission time
export PRL_PATRL_SMOKE_LEVEL="${PRL_PATRL_SMOKE_LEVEL:-2}"
export PRL_PATRL_SMOKE_N="${PRL_PATRL_SMOKE_N:-5}"
export PRL_PATRL_SMOKE_TUNE="${PRL_PATRL_SMOKE_TUNE:-500}"
export PRL_PATRL_SMOKE_DRAWS="${PRL_PATRL_SMOKE_DRAWS:-500}"
export PRL_PATRL_SMOKE_SEED="${PRL_PATRL_SMOKE_SEED:-42}"

# =============================================================================
# Run Smoke Test
# =============================================================================
echo ""
echo "============================================================"
echo "Phase 18 PAT-RL Foundation Smoke Test (CPU)"
echo "============================================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: comp (CPU only)"
echo "MCMC:      tune=${PRL_PATRL_SMOKE_TUNE} draws=${PRL_PATRL_SMOKE_DRAWS} N=${PRL_PATRL_SMOKE_N}"
echo "Level:     ${PRL_PATRL_SMOKE_LEVEL}"
echo "Seed:      ${PRL_PATRL_SMOKE_SEED}"
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

python scripts/12_smoke_patrl_foundation.py \
    --level "${PRL_PATRL_SMOKE_LEVEL}" \
    --n-participants "${PRL_PATRL_SMOKE_N}" \
    --n-tune "${PRL_PATRL_SMOKE_TUNE}" \
    --n-draws "${PRL_PATRL_SMOKE_DRAWS}" \
    --seed "${PRL_PATRL_SMOKE_SEED}" \
    --output-dir "output/patrl_smoke_${SLURM_JOB_ID:-local}"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "PAT-RL SMOKE PASSED (level=${PRL_PATRL_SMOKE_LEVEL}, tune=${PRL_PATRL_SMOKE_TUNE}, draws=${PRL_PATRL_SMOKE_DRAWS})"
elif [[ $EXIT_CODE -eq 2 ]]; then
    echo "PAT-RL SMOKE BLOCKED: blackjax not installed (exit=2)"
    echo "  Install: pip install blackjax>=1.2.4"
else
    echo "PAT-RL SMOKE FAILED (exit code: $EXIT_CODE)"
    echo "  Check: cluster/logs/patrl18smoke_${SLURM_JOB_ID}.err"
fi
echo "End:       $(date)"
echo "============================================================"

# =============================================================================
# Auto-push logs and results to git (mirrors 16_smoke_test_cpu.slurm)
# =============================================================================
echo ""
echo "Pushing PAT-RL smoke results to git..."
_LOG_OUT="cluster/logs/patrl18smoke_${SLURM_JOB_ID}.out"
_LOG_ERR="cluster/logs/patrl18smoke_${SLURM_JOB_ID}.err"
_OUT_DIR="output/patrl_smoke_${SLURM_JOB_ID}"

git add "${_LOG_OUT}" "${_LOG_ERR}" 2>/dev/null
git add "${_OUT_DIR}"/*.csv 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "smoke patrl cpu level=${PRL_PATRL_SMOKE_LEVEL} tune=${PRL_PATRL_SMOKE_TUNE} draws=${PRL_PATRL_SMOKE_DRAWS} (job ${SLURM_JOB_ID})"
    if git push origin main 2>/dev/null; then
        echo "Pushed PAT-RL smoke results to origin/main"
    else
        echo "WARNING: git push failed — results committed locally only"
    fi
else
    echo "No new files to push"
fi

exit $EXIT_CODE
