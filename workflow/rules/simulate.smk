# =============================================================================
# workflow/rules/simulate.smk — Cohort simulation (Wave 1)
# =============================================================================
# Generates synthetic participant data for downstream fitting and power
# analysis. Corresponds to Wave 1 in cluster/submit_full_pipeline.sh.
#
# DAG position: [start] -> simulate -> fit.smk / power.smk
# =============================================================================


rule simulate:
    """Simulate synthetic participant cohort from config-driven parameters."""
    output:
        "data/simulated/prl_simulated.csv",
    resources:
        slurm_partition="comp",
        mem_mb=8192,
        runtime=10,
        cpus_per_task=2,
    shell:
        conda_preamble() + (
            "python scripts/03_pre_analysis/01_simulate_participants.py"
        )
