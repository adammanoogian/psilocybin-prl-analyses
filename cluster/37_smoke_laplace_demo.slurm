#!/bin/bash
# =============================================================================
# SLURM Job: VB-Laplace Quickstart Demo Smoke Test
# =============================================================================
# Runs the quickstart demo script end-to-end on a GPU node to verify:
#   1. simulate_batch() works
#   2. fit_vb_laplace_prl() for both 2-level and 3-level completes
#   3. Parameter recovery pipeline produces valid output
#   4. compare_models_laplace() (BMS) runs cleanly
#
# Expected runtime: ~5 min on GPU, ~10 min on CPU.
#
# Usage:
#   bash cluster/submit_one.sh cluster/37_smoke_laplace_demo.slurm
# =============================================================================

#SBATCH --job-name=laplace_demo
#SBATCH --output=logs/laplace_demo_%j.out
#SBATCH --error=logs/laplace_demo_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:T4:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --account=fc37

# ── environment ──────────────────────────────────────────────────────
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "VB-Laplace Quickstart Demo Smoke Test"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================================"

mkdir -p logs data/demo
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

_PROJECT="${PROJECT:-fc37}"
_CONDA_ENV="${CONDA_ENV:-ds_env_v10}"
conda activate "${_CONDA_ENV}" 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/"${_CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Failed to activate ${_CONDA_ENV}"
    exit 1
}

# ptxas path for JAX GPU compilation
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
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# GPU check (soft — demo also works on CPU)
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# ── run demo ─────────────────────────────────────────────────────────
echo ""
python scripts/demo_quickstart.py
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "DEMO COMPLETE — all outputs in data/demo/"
    ls -la data/demo/ 2>/dev/null
else
    echo "DEMO FAILED (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
