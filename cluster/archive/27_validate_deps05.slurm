#!/bin/bash
# =============================================================================
# SLURM Job: Phase 27-03 — DEPS-05 validation gate (BlackJAX 1.5 dense mass)
# =============================================================================
# Runs scripts/ci/validate_deps05_lowrank.py on a GPU node.
#
# Outputs:
#   .planning/phases/27-dependency-upgrade-chain/DEPS-05-evidence.json
#   logs/deps05_<JOBID>.out (full run log; .planning/ may not autopush)
#
# Usage:
#   sbatch cluster/27_validate_deps05.slurm
#   # override env (e.g. for parallel ds_env_v11 build):
#   sbatch --export=ALL,CONDA_ENV=ds_env_v10 cluster/27_validate_deps05.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=deps05_validate
#SBATCH --output=logs/deps05_%j.out
#SBATCH --error=logs/deps05_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
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

mkdir -p cluster/logs .planning/phases/27-dependency-upgrade-chain
export PYTHONUNBUFFERED=1

# Activate ds_env (or override via CONDA_ENV at sbatch --export time)
_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV} conda environment"
    exit 1
}

# =============================================================================
# Sanity check: GPU visible to JAX
# =============================================================================
echo ""
echo "Verifying GPU + JAX..."
python -c "
import jax
devs = jax.devices()
print(f'JAX devices: {devs}')
gpu = [d for d in devs if d.platform == 'gpu']
if not gpu:
    print('WARNING: No GPU detected; DEPS-05 cliff probe is meaningless on CPU.')
    raise SystemExit(2)
print(f'GPU(s): {gpu}')
print(f'jax={jax.__version__}')
import blackjax, numpyro, pyhgf
print(f'blackjax={blackjax.__version__} numpyro={numpyro.__version__} pyhgf={pyhgf.__version__}')
" || exit 2
echo ""

# =============================================================================
# DEPS-05 validation gate
# Legacy Phase 27 validation script — not migrated to FitConfig
# =============================================================================
N_WARMUP="${N_WARMUP:-1000}"
N_DRAWS="${N_DRAWS:-500}"
OUT_PATH=".planning/phases/27-dependency-upgrade-chain/DEPS-05-evidence.json"

echo "Running DEPS-05 validation: n_warmup=${N_WARMUP} n_draws=${N_DRAWS}"
echo "Output JSON: ${OUT_PATH}"
echo ""

python scripts/ci/validate_deps05_lowrank.py \
    --output "${OUT_PATH}" \
    --n-warmup "${N_WARMUP}" \
    --n-draws "${N_DRAWS}"

EXIT=$?

echo ""
echo "============================================================"
if [[ $EXIT -eq 0 && -f "${OUT_PATH}" ]]; then
    echo "DEPS-05 evidence written:"
    cat "${OUT_PATH}"
else
    echo "DEPS-05 FAILED (exit ${EXIT}); ${OUT_PATH} may not exist"
fi
echo "============================================================"

exit $EXIT
