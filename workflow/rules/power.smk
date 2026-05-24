# =============================================================================
# workflow/rules/power.smk — BFDA power analysis sweep (parallel chunks)
# =============================================================================
# Submits power sweep chunks in parallel on GPU (wildcarded by {chunk_id}),
# then collects results via power_postprocess. Corresponds to the power
# pipeline in cluster/submit_power_pipeline.sh.
#
# DAG position: simulate -> power_sweep(0..N) -> power_postprocess
# =============================================================================

N_POWER_CHUNKS = config.get("n_power_chunks", 8)


rule power_sweep:
    """Run BFDA power sweep chunk {wildcards.chunk_id} of {N_POWER_CHUNKS}."""
    input:
        rules.simulate.output,
    output:
        "models/power/chunk_{chunk_id}_complete.flag",
    params:
        n_chunks=N_POWER_CHUNKS,
    resources:
        slurm_partition="gpu",
        mem_mb=32768,
        runtime=240,
        cpus_per_task=4,
        slurm_extra="'--gres=gpu:T4:1'",
    shell:
        conda_preamble() + (
            "python scripts/03_pre_analysis/03_run_power_iteration.py"
            " --chunk-id {wildcards.chunk_id}"
            " --n-chunks {params.n_chunks}"
            " && touch {output}"
        )


rule power_postprocess:
    """Aggregate all power sweep chunks into summary curves."""
    input:
        expand(
            "models/power/chunk_{chunk_id}_complete.flag",
            chunk_id=range(N_POWER_CHUNKS),
        ),
    output:
        "reports/power/power_analysis_complete.flag",
    resources:
        slurm_partition="comp",
        mem_mb=16384,
        runtime=30,
        cpus_per_task=2,
    shell:
        conda_preamble() + (
            "python scripts/03_pre_analysis/04_power_postprocess.py"
            " && touch {output}"
        )
