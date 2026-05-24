# =============================================================================
# workflow/rules/validate.smk — Parameter recovery validation (Wave 3)
# =============================================================================
# Runs parameter recovery and convergence checks after both models have been
# fitted. Corresponds to Wave 3 in cluster/submit_full_pipeline.sh.
#
# DAG position: fit(2level) + fit(3level) -> [validate] -> analyze.smk
# =============================================================================


rule validate:
    """Run parameter-recovery validation after both model fits complete."""
    input:
        expand(
            "models/bayesian/{model}/fit_complete.flag",
            model=["hgf_2level", "hgf_3level"],
        ),
    output:
        "models/bayesian/validation_complete.flag",
    resources:
        slurm_partition="gpu",
        mem_mb=32768,
        runtime=120,
        cpus_per_task=4,
        slurm_extra="'--gres=gpu:A40:1'",
    shell:
        conda_preamble() + (
            "python scripts/04_main_analysis/b_bayesian/02_validate_recovery.py"
            " && touch {output}"
        )
