#!/bin/bash
# =============================================================================
# Phase 21 P-scan chain submission wrapper (plan 21-05)
# =============================================================================
# Submits 6 jobs total: for P in {5, 20, 50}, a COLD job followed by a WARM
# job that depends on the cold via --dependency=afterok:<COLD_JOBID>. Each job
# invokes cluster/21_diagnose_pscan.slurm with --export=ALL,PROBE_P=N,
# PROBE_ROLE=cold|warm and has a 30-minute walltime (enforced by the SBATCH
# directive in that script — Phase 21 SC7).
#
# Wall-clock budget per P-pair: up to 2×30min = 1h serial within that pair.
# Across 3 P values the three chains can run in parallel (SLURM will schedule
# them as GPU allocation permits), so aggregate wall clock is bounded above by
# 1h + queue waiting time.
#
# Early-stop policy (CONTEXT.md): if the P=5 results settle the verdict, the
# user can scancel the pending P=20 and P=50 jobs manually. The job IDs for
# each pair are printed at the end of this script to make cancellation easy.
#
# Infeasibility semantics: if the P=50 cold job exceeds 30min walltime, SLURM
# kills it with a non-zero exit code; the afterok dependency on the P=50 warm
# job then fails and the warm job is never scheduled. The chain for P=5 and
# P=20 continues uninterrupted (each P-pair's afterok is independent). Mark
# the P=50 CSV row status=infeasible in 21-05 SUMMARY (Task 4) manually.
#
# Usage (on M3, from repo root):
#   bash cluster/21_submit_pscan_chain.sh
#
# Prerequisite: cluster/21_diagnose_pscan.slurm on origin/main, git pulled.
# =============================================================================
set -u

if [[ ! -f cluster/21_diagnose_pscan.slurm ]]; then
    echo "ERROR: cluster/21_diagnose_pscan.slurm not found."
    echo "  Run: git pull origin main"
    exit 1
fi

# Strip Windows CRLF before sbatch (idempotent; gotcha #1 from slurm-autopush skill).
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

declare -a P_VALUES=(5 20 50)

declare -A COLD_JOBIDS
declare -A WARM_JOBIDS

for P in "${P_VALUES[@]}"; do
    echo "=============================================="
    echo "Submitting chain for P=${P}"
    echo "=============================================="

    # -------------------------------------------------------------------------
    # Submit COLD job: --export passes PROBE_P + PROBE_ROLE to the SLURM script
    # per the --export=ALL,VAR=VAL convention (mirrors 14_benchmark_gpu.slurm
    # usage: `sbatch --export=ALL,SAMPLER=numpyro ...`). --parsable returns
    # just the JOBID on stdout for easy capture.
    # -------------------------------------------------------------------------
    COLD_JOBID=$(sbatch \
        --parsable \
        --export=ALL,PROBE_P="${P}",PROBE_ROLE=cold \
        --job-name="prl_pscan_P${P}_cold" \
        cluster/21_diagnose_pscan.slurm)
    if [[ ! "${COLD_JOBID}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: cold submit failed for P=${P}: got '${COLD_JOBID}'"
        exit 2
    fi
    COLD_JOBIDS["${P}"]="${COLD_JOBID}"
    echo "Cold job submitted: P=${P} JOBID=${COLD_JOBID}"

    # -------------------------------------------------------------------------
    # Submit WARM job with --dependency=afterok:<COLD_JOBID>. SLURM will hold
    # the warm job in the queue until the cold job completes with exit code 0;
    # if the cold job fails (non-zero exit, timeout, OOM), the warm job is
    # automatically cancelled and never runs.
    # -------------------------------------------------------------------------
    WARM_JOBID=$(sbatch \
        --parsable \
        --export=ALL,PROBE_P="${P}",PROBE_ROLE=warm \
        --job-name="prl_pscan_P${P}_warm" \
        --dependency=afterok:"${COLD_JOBID}" \
        cluster/21_diagnose_pscan.slurm)
    if [[ ! "${WARM_JOBID}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: warm submit failed for P=${P}: got '${WARM_JOBID}'"
        exit 3
    fi
    WARM_JOBIDS["${P}"]="${WARM_JOBID}"
    echo "Warm job submitted: P=${P} JOBID=${WARM_JOBID} depends_on=${COLD_JOBID}"
done

echo ""
echo "=============================================="
echo "All 6 jobs submitted. Summary:"
echo "=============================================="
printf "%-6s %-12s %-12s\n" "P" "COLD_JOBID" "WARM_JOBID"
for P in "${P_VALUES[@]}"; do
    printf "%-6s %-12s %-12s\n" \
        "${P}" "${COLD_JOBIDS[${P}]}" "${WARM_JOBIDS[${P}]}"
done
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --name=prl_pscan_P5_cold,prl_pscan_P5_warm,prl_pscan_P20_cold,prl_pscan_P20_warm,prl_pscan_P50_cold,prl_pscan_P50_warm"
echo "Or track auto-push commits:"
echo "  watch -n 30 'git fetch origin main && git log origin/main --oneline -5'"
echo ""
echo "Early-stop: if P=5 results settle the verdict, cancel pending with:"
for P in 20 50; do
    echo "  scancel ${COLD_JOBIDS[${P}]} ${WARM_JOBIDS[${P}]}  # P=${P}"
done
echo ""
echo "Resume locally: git fetch && git merge origin/results/<branch>"
