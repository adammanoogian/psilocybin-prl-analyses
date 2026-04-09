#!/bin/bash
# =============================================================================
# SLURM Job: Power Analysis Sweep (HGF BFDA) — Chunk-Based
# =============================================================================
# Each array task processes a contiguous chunk of the power sweep grid.
# JAX compiles the HGF model once per chunk, then reuses the compiled
# trace for all iterations — no redundant JIT overhead.
#
# Default: 3 chunks (one per effect size), ~1400 iterations each.
# Override n_chunks in configs/prl_analysis.yaml to change.
#
# Usage:
#   # Full power sweep (3 chunks):
#   sbatch cluster/08_power_sweep.slurm
#
#   # Dry-run (writes placeholder parquet rows, no MCMC):
#   sbatch --export=DRY_RUN=1 cluster/08_power_sweep.slurm
#
#   # Override chunk count at submission (must match YAML n_chunks):
#   sbatch --array=0-4 cluster/08_power_sweep.slurm
# =============================================================================

#SBATCH --job-name=prl_power
#SBATCH --output=cluster/logs/power_%A_%a.out
#SBATCH --error=cluster/logs/power_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
# Default: 3 chunks (one per effect size). Match to YAML n_chunks.
#SBATCH --array=0-2

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

# Activate prl_gpu env (try by name, fallback to scratch path)
_PROJECT="${PROJECT:-fc37}"
conda activate prl_gpu 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/prl_gpu 2>/dev/null || {
    echo "ERROR: Failed to activate prl_gpu conda environment"
    echo "Tried: prl_gpu (by name), /scratch/${_PROJECT}/${USER}/conda/envs/prl_gpu"
    echo "Create with: bash cluster/00_setup_env_gpu.sh"
    exit 1
}

# =============================================================================
# JAX Compilation Cache
# =============================================================================
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# =============================================================================
# Verify GPU Access
# =============================================================================
echo "Verifying GPU..."
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected! JAX will use CPU.')
    print('Check: CUDA_VISIBLE_DEVICES, conda env, jax[cuda] installation')
"
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "============================================================"
echo "Power Sweep - SLURM Array Job"
echo "============================================================"
echo "Job ID:     ${SLURM_ARRAY_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Array size: ${SLURM_ARRAY_TASK_COUNT:-unknown}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:      $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Run Power Iteration
# =============================================================================
MODE_FLAG=""
if [[ "${BENCHMARK:-0}" == "1" ]]; then
    MODE_FLAG="--benchmark"
    echo "Mode: BENCHMARK (JIT compile + single fit timing)"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    MODE_FLAG="--dry-run"
    echo "Mode: DRY RUN (placeholder parquet, no MCMC)"
else
    echo "Mode: FULL (simulate + fit + BF)"
fi
echo ""

python scripts/08_run_power_iteration.py \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" \
    --job-id "${SLURM_ARRAY_JOB_ID}" \
    --fit-chains 2 \
    --fit-draws 500 \
    --fit-tune 500 \
    --sampler "${SAMPLER:-pymc}" \
    ${MODE_FLAG}

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "*** ERROR: Task ${SLURM_ARRAY_TASK_ID} FAILED with exit code $EXIT_CODE ***"
    echo "*** Check: cluster/logs/power_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err ***"
    echo ""
fi

echo "Completed: $(date) (exit code: $EXIT_CODE)"

# =============================================================================
# Job Resource Report
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE REPORT"
echo "============================================================"
echo "Job ID:     ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo ""

# GPU stats via nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r name total used free util temp; do
        echo "  Model:       $name"
        echo "  VRAM Total:  $total"
        echo "  VRAM Used:   $used"
        echo "  VRAM Free:   $free"
        echo "  Utilization: $util"
        echo "  Temperature: $temp"
    done
    echo ""
fi

# SLURM accounting
echo "SLURM Accounting:"
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo ""

echo "For detailed stats after completion:"
echo "  jobstats ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================================"

exit $EXIT_CODE
