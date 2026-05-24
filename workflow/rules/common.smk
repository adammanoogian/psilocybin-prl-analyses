# =============================================================================
# workflow/rules/common.smk — Shared helpers for all HGF pipeline rules
# =============================================================================
# Provides:
#   conda_preamble()  — returns shell preamble string for module + env setup
#   wildcard_constraints — model name allowlist
#
# Consumed by: simulate.smk, fit.smk, validate.smk, analyze.smk, power.smk,
#   smoke.smk (all rules that execute on M3 via sbatch)
# =============================================================================


def conda_preamble():
    """Return the shell preamble that activates the conda environment on M3.

    Idempotent across every rule shell block. Handles both the shared
    scratch path and the per-project conda env name (overridable via
    CONDA_ENV env var).

    Returns
    -------
    str
        Multi-line bash preamble string for insertion into rule shell blocks.
    """
    return (
        "module load miniforge3\n"
        '_PROJECT="${{PROJECT:-fc37}}"\n'
        '_CONDA_ENV="${{CONDA_ENV:-ds_env}}"\n'
        "conda activate \"${{_CONDA_ENV}}\" 2>/dev/null || \\\n"
        "conda activate /scratch/${{_PROJECT}}/${{USER}}/conda/envs/\"${{_CONDA_ENV}}\" 2>/dev/null\n"
        "export PYTHONUNBUFFERED=1\n"
        'export JAX_COMPILATION_CACHE_DIR="/scratch/${{_PROJECT}}/${{USER}}/.jax_cache_gpu"\n'
        "mkdir -p \"$JAX_COMPILATION_CACHE_DIR\" 2>/dev/null || true\n"
        # Strip Windows CRLF from rule files on first use (idempotent)
        "sed -i 's/\\r$//' workflow/rules/*.smk 2>/dev/null || true\n"
    )


wildcard_constraints:
    model="hgf_2level|hgf_3level",
