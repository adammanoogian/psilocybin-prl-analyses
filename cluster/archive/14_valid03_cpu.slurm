#!/bin/bash
# =============================================================================
# SLURM Job: Phase 14 — VALID-03 CPU Baseline Fit
# =============================================================================
# Runs a small batched fit on CPU, saves posterior means to JSON.
# Submit this AND 14_valid03_gpu.slurm, then compare results.
#
# Usage:
#   sbatch cluster/14_valid03_cpu.slurm
#
# Debug solo run with auto-push (per-job result push on success/crash/timeout):
#   bash cluster/submit_one.sh <this-script>
# =============================================================================

#SBATCH --job-name=prl_v03cpu
#SBATCH --output=logs/valid03_cpu_%j.out
#SBATCH --error=logs/valid03_cpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cluster/logs models/power
export PYTHONUNBUFFERED=1

_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# Force CPU platform and expose 2 CPU devices for pmap chain parallelism
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=2"

echo "============================================================"
echo "VALID-03: CPU Baseline Fit"
echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     ${SLURMD_NODENAME:-$(hostname)}"
echo "Platform: CPU (forced via JAX_PLATFORM_NAME=cpu)"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:    $(date)"
echo "============================================================"
echo ""

python tests/scientific/test_valid03_cross_platform.py run \
    --output models/power/valid03_cpu.json \
    --n-per-group 3 \
    --seed 42 \
    --n-chains 2 \
    --n-draws 200 \
    --n-tune 200

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 && -f models/power/valid03_cpu.json ]]; then
    echo "CPU fit saved to models/power/valid03_cpu.json"
else
    echo "CPU fit FAILED (exit code: $EXIT_CODE)"
fi

echo "End: $(date)"
exit $EXIT_CODE
