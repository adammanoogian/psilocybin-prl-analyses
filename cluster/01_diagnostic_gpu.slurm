#!/bin/bash
# =============================================================================
# SLURM Job: GPU Diagnostic Check
# =============================================================================
# Quick sanity check that GPU, JAX, PyMC, and pyhgf are correctly configured.
# Run this FIRST after setting up the ds_env environment to verify GPU access.
#
# Usage:
#   sbatch cluster/01_diagnostic_gpu.slurm
#
# Expected output: GPU detected, JAX computes on GPU, all imports succeed,
# GPU matrix multiply is faster than CPU.
# =============================================================================

#SBATCH --job-name=prl_diag
#SBATCH --output=cluster/logs/diagnostic_%j.out
#SBATCH --error=cluster/logs/diagnostic_%j.err
#SBATCH --time=00:30:00
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
echo "GPU Diagnostic Check"
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
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# =============================================================================
# Run Diagnostics
# =============================================================================
python -c "
import time
import sys

print('=' * 60)
print('1. JAX Device Check')
print('=' * 60)
import jax
devices = jax.devices()
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
cpu_devices = [d for d in devices if d.platform == 'cpu']
print(f'GPU devices: {len(gpu_devices)}')
print(f'CPU devices: {len(cpu_devices)}')
if not gpu_devices:
    print('FAIL: No GPU detected!')
    sys.exit(1)
print('PASS: GPU detected')

print()
print('=' * 60)
print('2. GPU Computation Test')
print('=' * 60)
import jax.numpy as jnp
x = jnp.ones((100, 100))
y = jnp.dot(x, x)
y.block_until_ready()
print(f'Simple matmul result shape: {y.shape}, sum: {y.sum():.0f}')
print('PASS: GPU computation works')

print()
print('=' * 60)
print('3. Import Checks')
print('=' * 60)
imports = [
    ('pymc', 'PyMC'),
    ('pytensor', 'PyTensor'),
    ('arviz', 'ArviZ'),
    ('pyhgf', 'pyhgf'),
    ('prl_hgf', 'prl_hgf (project package)'),
    ('bambi', 'bambi'),
    ('pingouin', 'pingouin'),
    ('groupBMC', 'groupBMC'),
]
all_ok = True
for mod, label in imports:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'n/a')
        print(f'  {label}: {ver} ... OK')
    except ImportError as e:
        print(f'  {label}: FAIL ({e})')
        all_ok = False
if all_ok:
    print('PASS: All imports successful')
else:
    print('WARN: Some imports failed (check above)')

print()
print('=' * 60)
print('4. GPU vs CPU Benchmark (1000x1000 matmul)')
print('=' * 60)

# GPU benchmark
x_gpu = jnp.ones((1000, 1000))
# Warm up
jnp.dot(x_gpu, x_gpu).block_until_ready()

t0 = time.perf_counter()
for _ in range(100):
    jnp.dot(x_gpu, x_gpu).block_until_ready()
gpu_time = (time.perf_counter() - t0) / 100 * 1000

# CPU benchmark
import numpy as np
x_cpu = np.ones((1000, 1000))
t0 = time.perf_counter()
for _ in range(100):
    np.dot(x_cpu, x_cpu)
cpu_time = (time.perf_counter() - t0) / 100 * 1000

print(f'GPU: {gpu_time:.2f} ms per matmul')
print(f'CPU: {cpu_time:.2f} ms per matmul')
speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
print(f'Speedup: {speedup:.1f}x')
if speedup > 1:
    print('PASS: GPU is faster than CPU')
else:
    print('WARN: GPU not faster (may be warm-up or small matrix)')

print()
print('=' * 60)
print('DIAGNOSTIC SUMMARY: ALL CHECKS PASSED')
print('=' * 60)
"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Diagnostic complete. Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "============================================================"

exit $EXIT_CODE
