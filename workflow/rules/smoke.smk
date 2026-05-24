# =============================================================================
# workflow/rules/smoke.smk — Standalone Laplace quickcheck
# =============================================================================
# Runs the VB-Laplace demo script as a fast end-to-end smoke test. No
# dependencies on other rules — can be targeted independently:
#   snakemake --profile workflow/profiles/slurm/ smoke
#
# Uses T4 GPU (16GB) since the demo fits only a small synthetic cohort.
# Reference: cluster/37_smoke_laplace_demo.slurm for the sbatch equivalent.
#
# DAG position: [standalone] laplace_demo -> (top-level 'smoke' rule)
# =============================================================================


rule laplace_demo:
    """Run VB-Laplace quickstart demo (standalone smoke check, ~5 min on T4)."""
    output:
        "reports/smoke/laplace_demo_complete.flag",
    resources:
        slurm_partition="gpu",
        mem_mb=32768,
        runtime=30,
        cpus_per_task=4,
        slurm_extra="'--gres=gpu:T4:1'",
    shell:
        conda_preamble() + (
            "mkdir -p reports/smoke\n"
            "python scripts/demo_quickstart.py"
            " && touch {output}"
        )
