#!/bin/bash
# =============================================================================
# SLURM Job: Phase 21 — Parametrized P-scan Probe (BENCH-DIAG-04, plan 21-05)
# =============================================================================
# Runs ONE cold OR ONE warm --p-scan probe at a SINGLE P value, selected via
# --export=ALL,PROBE_P=N,PROBE_ROLE=cold|warm. Designed to be chained by
# cluster/21_submit_pscan_chain.sh for P in {5, 20, 50}.
#
# Roles:
#   - cold: invokes scripts/03_pre_analysis/03_run_power_iteration.py --p-scan ${PROBE_P}
#           --diagnostic-mode. _run_p_scan_probe (from 21-02) runs two
#           in-process calls and writes a CSV row with cold_jit_s +
#           same_proc_warm_s filled; next_proc_warm_s remains empty.
#   - warm: invokes the same --p-scan ${PROBE_P} --diagnostic-mode, then an
#           inline python merge parses the fresh cold wall-clock from the
#           probe stdout and writes it into the pre-existing row's
#           next_proc_warm_s column (the persistent-cache cross-process
#           measurement — the authoritative SC4 signal).
#
# Captures per (P, role):
#   - JAX_LOG_COMPILES=1 stdout+stderr → archived under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/pscan_P${P}_${ROLE}.log
#   - XLA text-IR HLO dumps under
#     .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/pscan_P${P}_${ROLE}/
#     (distinct directories so 21-06 can diff cold-vs-warm HLO for the same P)
#
# Cross-job cache sharing:
#   JAX_COMPILATION_CACHE_DIR path is IDENTICAL to
#   cluster/21_diagnose_laplace.slurm and cluster/21_diagnose_nuts.slurm so the
#   warm role benefits from the persistent cache written by the prior cold job
#   (which is the point of the cross-process measurement). Shared path
#   /scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared — Phase 21 VERDICT §c
#   patch; ${SLURMD_NODENAME:-local} suffix dropped for cross-node cache reuse.
#
# Walltime: 00:30:00 per job (Phase 21 SC7). If P=50 cold exceeds walltime,
# the job times out, no CSV row is written (or the row is marked infeasible by
# the probe), and the subsequent warm job does NOT start (afterok fails).
#
# Scope (per 21-05-PLAN.md):
#   - This script does NOT touch results/power/benchmark_batched.json.
#   - This script does NOT touch apply_decision_gate thresholds.
#   - DIAGNOSTIC ONLY — no fixes, no production code changes.
#
# Usage (via submission wrapper):
#   bash cluster/21_submit_pscan_chain.sh
# Usage (standalone, single P+role):
#   sbatch --export=ALL,PROBE_P=5,PROBE_ROLE=cold cluster/21_diagnose_pscan.slurm
# =============================================================================

#SBATCH --job-name=prl_diag21_pscan
#SBATCH --output=logs/diag21_pscan_%j.out
#SBATCH --error=logs/diag21_pscan_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Shell hygiene: -u catches unbound variables early; deliberately NO -e so
# that a probe failure still reaches the git auto-push block at the bottom.
# =============================================================================
set -u

# =============================================================================
# Parametrization via --export env vars (submission wrapper contract)
# =============================================================================
: "${PROBE_P:?PROBE_P env var required: integer number of participants (e.g., 5, 20, 50)}"
: "${PROBE_ROLE:?PROBE_ROLE env var required: 'cold' or 'warm'}"

if [[ "${PROBE_ROLE}" != "cold" && "${PROBE_ROLE}" != "warm" ]]; then
    echo "ERROR: PROBE_ROLE must be 'cold' or 'warm', got '${PROBE_ROLE}'"
    exit 2
fi

if ! [[ "${PROBE_P}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PROBE_P must be a non-negative integer, got '${PROBE_P}'"
    exit 2
fi

# =============================================================================
# Environment Setup (mirrors cluster/21_diagnose_nuts.slurm verbatim)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs
mkdir -p .planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps
export PYTHONUNBUFFERED=1

# Activate ds_env (identical block to 14_benchmark_gpu.slurm + 21_diagnose_*.slurm)
_PROJECT="${PROJECT:-fc37}"
conda activate ds_env 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/ds_env 2>/dev/null || {
    echo "ERROR: Failed to activate ds_env conda environment"
    exit 1
}

# =============================================================================
# Add pip-installed ptxas to PATH (identical to 14_benchmark_gpu.slurm)
# =============================================================================
PTXAS_BIN=$(python -c '
import os, site
sp = site.getsitepackages()[0]
for subdir in ("nvidia/cu13/bin", "nvidia/cuda_nvcc/bin"):
    d = os.path.join(sp, subdir)
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "ptxas")):
        print(d); break
' 2>/dev/null)
if [[ -n "$PTXAS_BIN" && -d "$PTXAS_BIN" ]]; then
    export PATH="$PTXAS_BIN:$PATH"
    echo "Added ptxas to PATH: $PTXAS_BIN"
fi

# =============================================================================
# CUDA / PTX Version Compatibility Check (identical to 14_benchmark_gpu.slurm)
# =============================================================================
check_cuda_compat() {
    DRIVER_CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [[ -z "$DRIVER_CUDA_VER" ]]; then
        echo "WARNING: nvidia-smi not available or no GPU detected"
        return 1
    fi

    PTXAS_VER=$(ptxas --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    if [[ -z "$PTXAS_VER" ]]; then
        echo "WARNING: ptxas not found in PATH"
        return 1
    fi

    echo "Driver CUDA version: $DRIVER_CUDA_VER"
    echo "ptxas CUDA version:  $PTXAS_VER"

    DRIVER_MAJOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f1)
    DRIVER_MINOR=$(echo "$DRIVER_CUDA_VER" | cut -d. -f2)
    PTXAS_MAJOR=$(echo "$PTXAS_VER" | cut -d. -f1)
    PTXAS_MINOR=$(echo "$PTXAS_VER" | cut -d. -f2)

    if (( PTXAS_MAJOR > DRIVER_MAJOR )) || \
       (( PTXAS_MAJOR == DRIVER_MAJOR && PTXAS_MINOR > DRIVER_MINOR )); then
        echo ""
        echo "ERROR: ptxas CUDA version ($PTXAS_VER) > driver CUDA version ($DRIVER_CUDA_VER)"
        echo "  XLA will disable parallel compilation, causing 2-5x slower JIT."
        echo "  FIX: pip install -r cluster/requirements-gpu.txt"
        return 1
    fi

    echo "OK: ptxas version <= driver version (XLA parallel compilation enabled)"
    return 0
}

echo ""
echo "=== CUDA Environment Check ==="
check_cuda_compat || echo "WARNING: CUDA compatibility issue detected (non-fatal, continuing)"
echo ""

# =============================================================================
# JAX cache + diagnostic env vars (Phase 21)
# IDENTICAL path to cluster/21_diagnose_laplace.slurm and
# cluster/21_diagnose_nuts.slurm so the warm role benefits from the persistent
# cache written by the prior cold job — this is the whole point of the
# cross-process measurement. Shared across nodes via Lustre scratch (Phase 21
# VERDICT §c patch — ${SLURMD_NODENAME:-local} suffix dropped).
# =============================================================================
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu/shared"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}

# Pre-warm ArviZ daily-warning file to defeat import-time race (14.1-07 follow-up)
# arviz/__init__.py _warn_once_per_day() skips _atomic_write_text ONLY IF the
# stamp file already contains today's ISO date. Writing today's date here makes
# the racey tmp.replace(path) code path unreachable under parallel SLURM launches.
mkdir -p "$HOME/.cache/arviz"
date +%Y-%m-%d > "$HOME/.cache/arviz/daily_warning"

# Phase 21 addition: emit JIT compile events to stdout/stderr for forensic parsing.
export JAX_LOG_COMPILES=1

# Show actual VRAM usage, not preallocated pool (mirrors 14_benchmark_gpu.slurm)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# Banner
# =============================================================================
echo "============================================================"
echo "Phase 21 P-scan Probe"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):       ${CUDA_VISIBLE_DEVICES:-none}"
echo "JAX cache:    ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo "JAX_LOG_COMPILES: ${JAX_LOG_COMPILES}"
echo "PROBE_P:      ${PROBE_P}"
echo "PROBE_ROLE:   ${PROBE_ROLE}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start:        $(date)"
echo "============================================================"
echo ""

python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if d.platform == 'gpu']
print(f'JAX devices: {devices}')
if not gpu:
    print('ERROR: No GPU detected — probe results will not be meaningful!')
    import sys; sys.exit(1)
print(f'GPU: {gpu[0]}')
"
echo ""

# =============================================================================
# Probe execution: single P, single role.
#
# Both cold and warm roles call the SAME probe command: _run_p_scan_probe
# (from 21-02) runs two in-process calls at N, writing a CSV row with
# cold_jit_s + same_proc_warm_s filled. The distinction is purely which row
# this job writes/updates:
#   - cold: appends a fresh row (first-process compile; may be slow).
#   - warm: the probe's cold wall-clock here IS the cross-process persistent-
#           cache measurement, so the warm role merges it into the PRE-EXISTING
#           row's next_proc_warm_s column (populated below in Step 4).
# =============================================================================
echo "========================================"
echo "P-scan probe: P=${PROBE_P} role=${PROBE_ROLE}"
echo "========================================"
echo "Start: $(date)"

HLO_DUMP_DIR=".planning/phases/21-benchmark-bottleneck-diagnosis/hlo_dumps/pscan_P${PROBE_P}_${PROBE_ROLE}"
PSCAN_LOG=".planning/phases/21-benchmark-bottleneck-diagnosis/pscan_P${PROBE_P}_${PROBE_ROLE}.log"
mkdir -p "${HLO_DUMP_DIR}"

# Per-(P, role) XLA dump isolation so 21-06 can diff cold-vs-warm HLO hashes.
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_DIR}"
echo "XLA_FLAGS=${XLA_FLAGS}"

# _run_p_scan_probe writes to stdout AND stderr. JAX_LOG_COMPILES messages
# go to stderr via Python's logging module. Capture both streams so the
# log file has everything for post-hoc parsing by 21-06 / 21-07.
python scripts/03_pre_analysis/03_run_power_iteration.py \
    --p-scan "${PROBE_P}" \
    --diagnostic-mode \
    --chunk-id 0 \
    --job-id "${SLURM_JOB_ID:-diag21_pscan}" \
    > "${PSCAN_LOG}" 2>&1
PROBE_EXIT=$?

echo "Probe exit: ${PROBE_EXIT}"
echo "--- pscan P=${PROBE_P} role=${PROBE_ROLE} log tail (80 lines) ---"
tail -n 80 "${PSCAN_LOG}" || echo "(log file missing)"
echo "--- end log ---"
echo "End: $(date)"
echo ""

# Clear XLA_FLAGS so downstream python (merge block + git) doesn't inherit.
unset XLA_FLAGS

# =============================================================================
# Warm-role post-processing: fill next_proc_warm_s column.
#
# For the cold role we are DONE — _run_p_scan_probe already wrote cold_jit_s +
# same_proc_warm_s to jit_scaling_sweep.csv. For the warm role, the probe
# appended ANOTHER row (or overwrote the existing row depending on 21-02's
# idempotency logic). Either way, what we want is to parse the cold wall-
# clock from THIS job's stdout (which is the post-persistent-cache cold
# timing) and merge it into the PRE-EXISTING cold-job row's next_proc_warm_s
# column.
#
# The PYEOF heredoc uses bash string interpolation for ${PROBE_P} and
# ${PSCAN_LOG} — intentional; bash substitutes at SLURM-script-execution time.
# =============================================================================
if [[ "${PROBE_ROLE}" == "warm" ]]; then
    echo "========================================"
    echo "Merging next_proc_warm_s into jit_scaling_sweep.csv for P=${PROBE_P}"
    echo "========================================"
    python <<PYEOF
import re
from pathlib import Path

csv_path = Path(".planning/phases/21-benchmark-bottleneck-diagnosis/jit_scaling_sweep.csv")
log_path = Path("${PSCAN_LOG}")

if not csv_path.exists():
    print(f"ERROR: {csv_path} does not exist — cold job did not run or did not write a row")
    raise SystemExit(1)
if not log_path.exists():
    print(f"ERROR: {log_path} does not exist — warm probe produced no log")
    raise SystemExit(1)

text = log_path.read_text(encoding="utf-8", errors="replace")
# Parse the "cold=Xs" field from the probe's stdout. Note: from the
# persistent-cache perspective this IS the "next-process warm" timing — the
# first in-process call after a fresh python interpreter, with the persistent
# cache primed by the previous cold SLURM job.
match = re.search(r"--p-scan\s+\d+\s+complete:.*?cold=([\d.]+)s", text, re.DOTALL)
if not match:
    print(f"WARNING: could not parse cold_jit_s from {log_path}; leaving next_proc_warm_s empty")
    raise SystemExit(0)
next_proc_warm_s = float(match.group(1))

# Update the CSV: find the row with P=${PROBE_P} and empty next_proc_warm_s.
# CSV schema: P,cold_jit_s,same_proc_warm_s,next_proc_warm_s,draws_per_s,vram_peak_mb,status
lines = csv_path.read_text(encoding="utf-8").splitlines()
if not lines:
    print(f"ERROR: {csv_path} is empty")
    raise SystemExit(1)
header = lines[0]
data_lines = lines[1:]

updated = False
new_lines = [header]
target_P = ${PROBE_P}
for row in data_lines:
    cols = row.split(",")
    if len(cols) != 7:
        new_lines.append(row)
        continue
    try:
        row_P = int(cols[0])
    except ValueError:
        new_lines.append(row)
        continue
    if row_P == target_P and not cols[3]:
        # next_proc_warm_s is the 4th column (index 3), currently empty.
        cols[3] = f"{next_proc_warm_s:.2f}"
        new_lines.append(",".join(cols))
        updated = True
    else:
        new_lines.append(row)

if not updated:
    print(f"WARNING: no row with P={target_P} and empty next_proc_warm_s; appending degenerate row")
    # Degenerate case: cold job didn't write a row (OOM/timeout). Append a
    # minimal 7-field row so every CSV line stays schema-valid (21-05 Task 4 I2).
    # Schema: P,cold_jit_s,same_proc_warm_s,next_proc_warm_s,draws_per_s,vram_peak_mb,status
    new_lines.append(
        f"{target_P},,,{next_proc_warm_s:.2f},,,completed"
    )

csv_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print(f"Updated next_proc_warm_s={next_proc_warm_s:.2f} for P={target_P}")
PYEOF
fi

# Results return via cluster/99_push_results.slurm (Pattern A) when this job
# is submitted via submit_one.sh / orchestrator. Inline per-job push removed
# 2026-05-06: Mutagen syncs code, autopush handles results, and parallel
# jobs racing on the working tree silently dropped files (rlwm 1c37d3f).

# Exit with probe's exit code. A non-zero exit on the cold job will cause the
# afterok dependency to fail and the warm job to be cancelled — this is the
# intended behavior per 21-05 SC7 (infeasibility is data).
exit ${PROBE_EXIT}
