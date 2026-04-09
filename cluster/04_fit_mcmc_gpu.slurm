#!/bin/bash
# =============================================================================
# SLURM Job: GPU-Accelerated MCMC Fitting (HGF Models)
# =============================================================================
# Runs NUTS MCMC parameter estimation for HGF models using GPU acceleration.
# This is the main compute job in the pipeline.
#
# Usage:
#   sbatch cluster/04_fit_mcmc_gpu.slurm                              # hgf_2level (default)
#   sbatch --export=MODEL=hgf_3level cluster/04_fit_mcmc_gpu.slurm    # hgf_3level
#   sbatch --export=MODEL="hgf_2level hgf_3level" cluster/04_fit_mcmc_gpu.slurm  # Both (dispatches parallel)
#   sbatch --export=MODEL=hgf_2level,INPUT=data/simulated/custom.csv cluster/04_fit_mcmc_gpu.slurm
#
# GPU Memory Guide:
#   ~60 participants (2 groups x 30):  ~16 GB VRAM -> A40 (48GB) comfortable
#   With 4 chains x 1000 draws:       ~24 GB VRAM -> A40 recommended
#
# Available GPUs on M3:
#   A100: 40GB VRAM, 32 nodes  -> --gres=gpu:A100:1
#   A40:  48GB VRAM, 12 nodes  -> --gres=gpu:A40:1
#   T4:   16GB VRAM, 8 nodes   -> --gres=gpu:T4:1 (small jobs only)
# =============================================================================

#SBATCH --job-name=prl_fit
#SBATCH --output=cluster/logs/fit_mcmc_%j.out
#SBATCH --error=cluster/logs/fit_mcmc_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration
# =============================================================================
MODEL="${MODEL:-hgf_2level}"
INPUT="${INPUT:-}"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "MCMC Fitting - SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model(s): $MODEL"
echo "Input: ${INPUT:-default}"
echo "Start: $(date)"
echo "============================================================"

# Activate ds_env (shared env across projects)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
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
# Multi-Model Dispatch: submit separate GPU jobs if multiple models
# =============================================================================
MODEL_COUNT=$(echo $MODEL | wc -w)

if [[ "$MODEL_COUNT" -gt 1 ]]; then
    echo "Dispatching $MODEL_COUNT models as parallel GPU jobs..."
    echo ""

    JOBIDS=()
    for model in $MODEL; do
        EXPORTS="MODEL=$model"
        [[ -n "$INPUT" ]] && EXPORTS="$EXPORTS,INPUT=$INPUT"

        JOBID=$(sbatch --parsable --export="$EXPORTS" --job-name="prl_fit_${model}" "$0")
        JOBIDS+=("$JOBID")
        echo "  $model -> Job $JOBID"
    done

    echo ""
    echo "Monitor: squeue -u $USER"
    echo ""
    for i in "${!JOBIDS[@]}"; do
        model=$(echo $MODEL | cut -d' ' -f$((i+1)))
        echo "  tail -f cluster/logs/fit_mcmc_${JOBIDS[$i]}.out   # $model"
    done
    exit 0
fi

# =============================================================================
# Run Single Model Fit
# =============================================================================
echo "============================================================"
echo "Fitting: $MODEL (GPU-accelerated MCMC)"
echo "Started: $(date)"
echo "============================================================"

CMD="python scripts/04_fit_participants.py --model $MODEL"
[[ -n "$INPUT" ]] && CMD="$CMD --input $INPUT"

echo "Command: $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "*** ERROR: Model $MODEL FAILED with exit code $EXIT_CODE ***"
    echo "*** Check stderr log: cluster/logs/fit_mcmc_${SLURM_JOB_ID}.err ***"
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
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
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
sacct -j "$SLURM_JOB_ID" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo ""

echo "For detailed stats after completion:"
echo "  jobstats ${SLURM_JOB_ID}"
echo "============================================================"

exit $EXIT_CODE
