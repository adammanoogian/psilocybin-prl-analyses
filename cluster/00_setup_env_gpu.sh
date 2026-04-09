#!/bin/bash
# =============================================================================
# GPU Environment Setup Script for Monash M3 Cluster
# =============================================================================
# Creates or updates the prl_gpu conda environment with JAX CUDA support
# for GPU-accelerated MCMC fitting of HGF models.
#
# Usage:
#   bash cluster/00_setup_env_gpu.sh           # Fresh install or interactive update
#   bash cluster/00_setup_env_gpu.sh --update  # Force update existing env (no delete)
#   bash cluster/00_setup_env_gpu.sh --fresh   # Delete and recreate from scratch
#   bash cluster/00_setup_env_gpu.sh --pull    # Git pull before setup (non-interactive update)
#   bash cluster/00_setup_env_gpu.sh --help    # Show this help
#
# Quick start on M3 (first time):
#   cd /scratch/fc37/$USER
#   git clone git@github.com:adammanoogian/psilocybin-prl-analyses.git
#   cd psilocybin-prl-analyses
#   bash cluster/00_setup_env_gpu.sh
#
# Subsequent sessions:
#   bash cluster/00_setup_env_gpu.sh --pull
#
# Note: CUDA module is NOT required. JAX's pip packages bundle their own CUDA
# runtime libraries (cuSPARSE, cuBLAS, cuDNN, etc.). GPU access works via the
# cluster's NVIDIA kernel driver on GPU nodes.
#
# Environment is stored in /scratch/fc37/$USER/conda to avoid home quota.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODE="interactive"
GIT_PULL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --update|-u)
            MODE="update"
            shift
            ;;
        --fresh|-f)
            MODE="fresh"
            shift
            ;;
        --pull|-p)
            GIT_PULL=true
            # Default to non-interactive update when --pull is used alone
            if [[ "$MODE" == "interactive" ]]; then
                MODE="update"
            fi
            shift
            ;;
        --help|-h)
            echo "Usage: bash cluster/00_setup_env_gpu.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --update, -u    Update existing environment (no delete)"
            echo "  --fresh, -f     Delete and recreate environment"
            echo "  --pull, -p      Git pull latest before setup (implies --update)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Default: Interactive mode (prompts for action if env exists)"
            echo ""
            echo "Quick start:"
            echo "  bash cluster/00_setup_env_gpu.sh --pull    # pull + update env"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "PRL HGF GPU Environment Setup (M3 Cluster)"
echo "============================================================"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project: $PROJECT_ROOT"
echo "Mode: $MODE"

# Git pull if requested
if [[ "$GIT_PULL" == true ]]; then
    echo ""
    echo "Pulling latest from origin/main..."
    git fetch origin
    git pull --ff-only origin main
    echo "HEAD: $(git log --oneline -1)"
fi

# Load miniforge3 (DO NOT load cuda module - JAX bundles CUDA libraries)
echo ""
echo "Loading modules..."
module load miniforge3

# Store conda envs in /scratch to avoid home quota
_PROJECT="${PROJECT:-fc37}"
export CONDA_ENVS_DIRS="/scratch/${_PROJECT}/${USER}/conda/envs"
mkdir -p "$CONDA_ENVS_DIRS" 2>/dev/null || true
echo "Conda envs dir: $CONDA_ENVS_DIRS"

# Check if environment exists
ENV_EXISTS=false
if conda env list | grep -q "prl_gpu"; then
    ENV_EXISTS=true
    echo "Environment 'prl_gpu' already exists."
fi

# =============================================================================
# Handle different modes
# =============================================================================

if [[ "$MODE" == "update" ]]; then
    if [[ "$ENV_EXISTS" == false ]]; then
        echo "ERROR: Environment 'prl_gpu' does not exist. Use without --update for fresh install."
        exit 1
    fi

    echo ""
    echo "Updating existing environment..."
    conda activate prl_gpu

    # Remove conflicting conda CUDA packages if present
    echo "Removing conflicting CUDA packages (if any)..."
    conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true

    # Verify Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" != "3.10" ]]; then
        echo "Upgrading Python $PYTHON_VERSION -> 3.10..."
        conda install python=3.10 -y
    fi

    # Force reinstall JAX with CUDA support (pyhgf 0.2.8 requires <0.4.32)
    echo "Installing/upgrading JAX with CUDA 12 support..."
    pip install --upgrade --force-reinstall "jax[cuda12]>=0.4.26,<0.4.32"

    # Install project package in editable mode
    echo "Installing prl_hgf package..."
    pip install -e "$PROJECT_ROOT"

    echo ""
    echo "Update complete!"

elif [[ "$MODE" == "fresh" ]]; then
    if [[ "$ENV_EXISTS" == true ]]; then
        echo "Removing existing environment..."
        conda env remove -n prl_gpu -y || {
            echo "WARNING: Could not remove environment. Trying update instead..."
            MODE="update"
            conda activate prl_gpu
            conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
            pip install --upgrade --force-reinstall "jax[cuda12]>=0.4.26,<0.4.32"
        }
    fi

    if [[ "$MODE" == "fresh" ]]; then
        echo ""
        echo "Creating fresh prl_gpu environment..."
        if command -v mamba &> /dev/null; then
            echo "Using mamba for faster installation..."
            mamba env create -f environment_gpu.yml
        else
            echo "Using conda..."
            conda env create -f environment_gpu.yml
        fi
        conda activate prl_gpu

        # Install project package in editable mode
        echo "Installing prl_hgf package..."
        pip install -e "$PROJECT_ROOT"
    fi

else
    # Interactive mode (default)
    if [[ "$ENV_EXISTS" == true ]]; then
        echo ""
        echo "Options:"
        echo "  [u] Update existing environment (recommended)"
        echo "  [f] Delete and recreate fresh"
        echo "  [q] Quit"
        read -p "Choose action [u/f/q]: " response

        case $response in
            [Uu]*)
                echo ""
                echo "Updating existing environment..."
                conda activate prl_gpu
                conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
                PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
                if [[ "$PYTHON_VERSION" != "3.10" ]]; then
                    echo "Upgrading Python $PYTHON_VERSION -> 3.10..."
                    conda install python=3.10 -y
                fi
                pip install --upgrade --force-reinstall "jax[cuda12]>=0.4.26,<0.4.32"
                pip install -e "$PROJECT_ROOT"
                ;;
            [Ff]*)
                echo "Removing existing environment..."
                conda env remove -n prl_gpu -y || {
                    echo "WARNING: Could not remove. Falling back to update..."
                    conda activate prl_gpu
                    conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
                    pip install --upgrade --force-reinstall "jax[cuda12]>=0.4.26,<0.4.32"
                    pip install -e "$PROJECT_ROOT"
                }
                if conda env list | grep -q "prl_gpu"; then
                    : # Already handled by fallback
                else
                    echo "Creating fresh environment..."
                    if command -v mamba &> /dev/null; then
                        mamba env create -f environment_gpu.yml
                    else
                        conda env create -f environment_gpu.yml
                    fi
                    conda activate prl_gpu
                    pip install -e "$PROJECT_ROOT"
                fi
                ;;
            *)
                echo "Exiting."
                exit 0
                ;;
        esac
    else
        # Environment doesn't exist - create fresh
        echo ""
        echo "Creating prl_gpu environment..."
        if command -v mamba &> /dev/null; then
            echo "Using mamba for faster installation..."
            mamba env create -f environment_gpu.yml
        else
            echo "Using conda..."
            conda env create -f environment_gpu.yml
        fi
        conda activate prl_gpu

        # Install project package in editable mode
        echo "Installing prl_hgf package..."
        pip install -e "$PROJECT_ROOT"
    fi
fi

# =============================================================================
# Verify installation
# =============================================================================

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "JAX version:"
python -c "import jax; print(f'JAX {jax.__version__}')"

echo ""
echo "PyMC version:"
python -c "import pymc; print(f'PyMC {pymc.__version__}')"

echo ""
echo "pyhgf version:"
python -c "import pyhgf; print(f'pyhgf {pyhgf.__version__}')"

echo ""
echo "prl_hgf package:"
python -c "import prl_hgf; print('prl_hgf imported successfully')"

echo ""
echo "Testing JAX device detection..."
python -c "
import jax
devices = jax.devices()
print(f'Available devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'SUCCESS: GPU(s) detected: {gpu_devices}')
    import jax.numpy as jnp
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print('GPU computation test passed!')
else:
    print('INFO: No GPU detected (normal on login node).')
    print('Test on a GPU node: srun --partition=gpu --gres=gpu:1 --pty bash')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To use the GPU environment:"
echo "  conda activate prl_gpu"
echo ""
echo "To test on a GPU node:"
echo "  srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash"
echo "  module load miniforge3 && conda activate prl_gpu"
echo "  python -c \"import jax; print(jax.devices())\""
echo ""
echo "To run the full pipeline:"
echo "  bash cluster/submit_full_pipeline.sh"
echo ""
echo "============================================================"
