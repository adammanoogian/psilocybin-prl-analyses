#!/bin/bash
# =============================================================================
# SLURM Job: Parameter Recovery Validation
# =============================================================================
# Runs parameter recovery metrics and BMS after MCMC fitting completes.
# Typically submitted as a dependent job after 04_fit_mcmc_gpu.slurm.
#
# Usage:
#   sbatch cluster/05_validation_gpu.slurm
#   sbatch --dependency=afterok:12345:12346 cluster/05_validation_gpu.slurm
#
# WAIC is skipped by default (--skip-waic) since it requires InferenceData .nc
# files which are not saved by default. Remove --skip-waic if .nc files exist.
# =============================================================================

#SBATCH --job-name=prl_valid
#SBATCH --output=cluster/logs/validation_%j.out
#SBATCH --error=cluster/logs/validation_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Parameter Recovery Validation - SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "============================================================"

# Activate ds_env (shared env across projects)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# JAX compilation cache
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Verify GPU
echo "Verifying GPU..."
python -c "
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if gpu_devices:
    print(f'GPU available: {gpu_devices}')
else:
    print('WARNING: No GPU detected, using CPU')
"
echo ""

# =============================================================================
# Run Validation (skip WAIC by default)
# =============================================================================
echo "Running parameter recovery validation..."
echo ""

python scripts/05_run_validation.py --skip-waic
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "*** ERROR: Validation FAILED with exit code $EXIT_CODE ***"
    echo "*** Check stderr log: cluster/logs/validation_${SLURM_JOB_ID}.err ***"
fi

echo ""
echo "============================================================"
echo "Validation complete. Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
