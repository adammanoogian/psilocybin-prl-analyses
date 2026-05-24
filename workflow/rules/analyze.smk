# =============================================================================
# workflow/rules/analyze.smk — Group-level statistical analysis (Wave 4)
# =============================================================================
# Runs group-level mixed-effects analysis and BMS model comparison after
# validation passes. Wildcarded by {model} so both analyses can run in
# parallel once validate completes. Corresponds to Wave 4 in
# cluster/submit_full_pipeline.sh.
#
# DAG position: validate -> [analyze(hgf_2level), analyze(hgf_3level)]
# =============================================================================


rule group_analysis:
    """Run group-level analysis for {wildcards.model} (BMS + mixed-effects)."""
    input:
        rules.validate.output,
    output:
        "reports/{model}/analysis_complete.flag",
    resources:
        slurm_partition="comp",
        mem_mb=32768,
        runtime=120,
        cpus_per_task=4,
    shell:
        conda_preamble() + (
            "python scripts/05_analysis/01_group_analysis.py"
            " --model {wildcards.model}"
            " && touch {output}"
        )
