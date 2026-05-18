#!/bin/bash
# =============================================================================
# Fix: Swap jax[cuda12] → jax[cuda13] in ds_env_v10
# =============================================================================
# Run on M3 login node (no GPU needed for pip):
#   bash cluster/fix_cuda13_swap.sh
#
# Then verify on a GPU node:
#   srun --partition=gpu --gres=gpu:1 --time=00:05:00 --account=fc37 \
#     bash -c 'module load miniforge3; conda activate ds_env_v10; python -c "import jax; print(jax.devices())"'
# =============================================================================
set -euo pipefail

module load miniforge3
conda activate ds_env_v10

echo "=== Before: JAX CUDA packages ==="
pip list 2>/dev/null | grep -iE "jax|nvidia|cuda" || true
echo ""

echo "=== Step 1: Uninstall cuda12 plugin + all nvidia-*-cu12 packages ==="
pip uninstall -y \
    jax-cuda12-plugin \
    jax-cuda12-pjrt \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvcc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nccl-cu12 \
    2>/dev/null || true
echo ""

echo "=== Step 2: Install jax[cuda13] ==="
pip install -U "jax[cuda13]>=0.9,<0.10"
echo ""

echo "=== Step 3: Install cluster GPU requirements ==="
pip install -r cluster/requirements-gpu.txt
echo ""

echo "=== After: JAX CUDA packages ==="
pip list 2>/dev/null | grep -iE "jax|nvidia|cuda" || true
echo ""

echo "=== Step 4: CPU-only verification (login node) ==="
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices (login node, expect CPU): {jax.devices()}')
# Just verify import works — GPU check needs a compute node
print('OK: JAX imports successfully with cuda13 plugin')
"

echo ""
echo "=== Done. Now verify GPU on a compute node: ==="
echo 'srun --partition=gpu --gres=gpu:1 --time=00:05:00 --account=fc37 \'
echo '  bash -c "module load miniforge3; conda activate ds_env_v10; python -c \"import jax; print(jax.devices())\""'
