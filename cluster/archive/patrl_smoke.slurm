#!/bin/bash
# =============================================================================
# SLURM Job: Plan 20-05 — PAT-RL SC5 Cohort Smoke (40 × 4 = 160 agents, CPU)
# =============================================================================
# SC5: 4 phenotypes × 40 agents = 160-agent cohort for PRL-V1/V2 formal gates
# (Phase 20-07). Invokes scripts/03_pre_analysis/08_smoke_patrl_foundation.py with the
# production-scale cohort defaults.
#
# Usage:
#   sbatch cluster/patrl_smoke.slurm                          # defaults
#   sbatch --export=ALL,PRL_PATRL_SMOKE_N=5 cluster/patrl_smoke.slurm
#   sbatch --export=ALL,PRL_PATRL_SMOKE_PHENOTYPES=healthy cluster/patrl_smoke.slurm
#
# Output (git-tracked under results/ so artifacts auto-push to origin):
#   results/patrl_smoke/${SLURM_JOB_ID}/   (trajectory CSVs, param summary,
#                                            idata_{laplace,nuts}.nc,
#                                            laplace_vs_nuts_diff.csv)
#   logs/patrl_smoke_${SLURM_JOB_ID}.out/.err
#
# Default fit method is 'both' (dual NUTS + Laplace) consistent with the
# 18_smoke_patrl_cpu.slurm policy set in Plans 18-06/19-05.
# Override via PRL_PATRL_SMOKE_METHOD at submission time.
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_patrl_smoke
#SBATCH --output=logs/patrl_smoke_%j.out
#SBATCH --error=logs/patrl_smoke_%j.err
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

mkdir -p cluster/logs results/patrl_smoke
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
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu/patrl/shared"
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

# Log compilations (goes to stderr)
export JAX_LOG_COMPILES=1

# MCMC and smoke scale — override via --export at submission time.
# SC5 defaults (Plan 20-05): PRL_PATRL_SMOKE_N=40 PRL_PATRL_SMOKE_PHENOTYPES=all
# 40 agents × 4 phenotypes = 160 cohort for PRL-V1/V2 gates (Phase 20-07)
export PRL_PATRL_SMOKE_N="${PRL_PATRL_SMOKE_N:-40}"
export PRL_PATRL_SMOKE_PHENOTYPES="${PRL_PATRL_SMOKE_PHENOTYPES:-all}"
export PRL_PATRL_SMOKE_LEVEL="${PRL_PATRL_SMOKE_LEVEL:-2}"
export PRL_PATRL_SMOKE_TUNE="${PRL_PATRL_SMOKE_TUNE:-500}"
export PRL_PATRL_SMOKE_DRAWS="${PRL_PATRL_SMOKE_DRAWS:-500}"
export PRL_PATRL_SMOKE_SEED="${PRL_PATRL_SMOKE_SEED:-42}"
# Default to dual fit (NUTS + Laplace) so idata_nuts.nc + idata_laplace.nc
# both land in results/ for the offline vbl06 comparison harness.
export PRL_PATRL_SMOKE_METHOD="${PRL_PATRL_SMOKE_METHOD:-both}"

# =============================================================================
# Run Smoke Test
# =============================================================================
echo ""
echo "============================================================"
echo "PAT-RL SC5 Cohort Smoke (Plan 20-05)"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition:     comp (CPU only)"
echo "N per phenotype: ${PRL_PATRL_SMOKE_N}  (total: $((PRL_PATRL_SMOKE_N * 4)) if all phenotypes)"
echo "Phenotypes:    ${PRL_PATRL_SMOKE_PHENOTYPES}"
echo "MCMC:          tune=${PRL_PATRL_SMOKE_TUNE} draws=${PRL_PATRL_SMOKE_DRAWS}"
echo "Level:         ${PRL_PATRL_SMOKE_LEVEL}"
echo "Seed:          ${PRL_PATRL_SMOKE_SEED}"
echo "Fit method:    ${PRL_PATRL_SMOKE_METHOD}"
echo "JAX cache:     ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:         $(date)"
echo "============================================================"
echo ""

python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'jax_enable_x64 (pre-run): {jax.config.jax_enable_x64}')
"
echo ""

python scripts/03_pre_analysis/08_smoke_patrl_foundation.py \
    --level "${PRL_PATRL_SMOKE_LEVEL}" \
    --n-participants "${PRL_PATRL_SMOKE_N}" \
    --phenotypes "${PRL_PATRL_SMOKE_PHENOTYPES}" \
    --n-tune "${PRL_PATRL_SMOKE_TUNE}" \
    --n-draws "${PRL_PATRL_SMOKE_DRAWS}" \
    --seed "${PRL_PATRL_SMOKE_SEED}" \
    --fit-method "${PRL_PATRL_SMOKE_METHOD}" \
    --output-dir "results/patrl_smoke/${SLURM_JOB_ID:-local}"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "PAT-RL SMOKE PASSED (N=${PRL_PATRL_SMOKE_N} phenotypes=${PRL_PATRL_SMOKE_PHENOTYPES} level=${PRL_PATRL_SMOKE_LEVEL})"
elif [[ $EXIT_CODE -eq 2 ]]; then
    echo "PAT-RL SMOKE BLOCKED: blackjax not installed (exit=2)"
    echo "  Install: pip install blackjax>=1.2.4"
else
    echo "PAT-RL SMOKE FAILED (exit code: $EXIT_CODE)"
    echo "  Check: logs/patrl_smoke_${SLURM_JOB_ID}.err"
fi
echo "End:           $(date)"
echo "============================================================"

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

exit $EXIT_CODE
