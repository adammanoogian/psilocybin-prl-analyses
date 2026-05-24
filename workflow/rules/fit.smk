# =============================================================================
# workflow/rules/fit.smk — GPU-accelerated MCMC fitting (Wave 2)
# =============================================================================
# Runs NUTS MCMC parameter estimation for {model} on GPU. Wildcarded so
# hgf_2level and hgf_3level jobs run in parallel.
#
# Reference: cluster/04_fit_mcmc_gpu.slurm for GPU environment setup patterns
# and the nvidia-smi verification block used in debug solo runs.
#
# DAG position: simulate -> [fit(hgf_2level), fit(hgf_3level)] -> validate.smk
#
# GPU memory guide (from cluster/04_fit_mcmc_gpu.slurm):
#   ~60 participants (2 groups x 30):  ~16 GB VRAM -> A40 (48GB) comfortable
#   With 4 chains x 1000 draws:       ~24 GB VRAM -> A40 recommended
# =============================================================================


rule fit:
    """Fit {wildcards.model} via BlackJAX NUTS on A40 GPU."""
    input:
        rules.simulate.output,
    output:
        "models/bayesian/{model}/fit_complete.flag",
    resources:
        slurm_partition="gpu",
        mem_mb=65536,
        runtime=720,
        cpus_per_task=4,
        slurm_extra="'--gres=gpu:A40:1'",
    shell:
        conda_preamble() + (
            "python scripts/04_main_analysis/b_bayesian/01_fit_participants.py"
            " --model {wildcards.model}"
            " && touch {output}"
        )
