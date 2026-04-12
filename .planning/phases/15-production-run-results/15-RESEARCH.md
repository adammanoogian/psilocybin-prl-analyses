# Phase 15: Production Run + Results - Research

**Researched:** 2026-04-12
**Domain:** SLURM job orchestration, power sweep execution, aggregation pipeline, results packaging
**Confidence:** HIGH — all code already exists and has been read directly

## Summary

Phase 15 is an **execution and fixup phase**, not a development phase. All the code needed to run the production power sweep already exists from Phases 8-14:

- `scripts/08_run_power_iteration.py` — entry point for each SLURM chunk
- `scripts/09_aggregate_power.py` — parquet → CSV aggregation
- `scripts/10_plot_power_curves.py` — 4-panel publication figure
- `scripts/11_write_recommendation.py` — recommendation.md generator
- `cluster/08_power_sweep.slurm` — array job (3 chunks)
- `cluster/09_power_postprocess.slurm` — post-processing (aggregation + figures + recommendation)
- `cluster/99_push_results.slurm` — auto-push to git
- `cluster/submit_power_pipeline.sh` — single entry point for the whole pipeline

The work in Phase 15 is: (1) ensure the SLURM scripts use the platform decided by the Phase 14 benchmark (`benchmark_batched.json`), (2) run the full sweep, (3) fix any issues that arise (missing cells, divergence rate over 5%, git push failures), and (4) verify all 5 success criteria pass.

The primary gap is that `99_push_results.slurm` does **not** currently stage `results/power/` files — it only stages `data/fitted/`, `results/validation/`, `results/group_analysis/`, and `figures/`. This must be fixed before the auto-push works for Phase 15 deliverables.

**Primary recommendation:** The plans should be: (1) platform selection + SLURM script update, (2) run sweep + triage failures, (3) verify success criteria + fix push script for `results/power/`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SLURM sbatch | cluster standard | Job submission and dependency chains | Existing cluster infrastructure |
| JAX | >=0.4.26,<0.4.32 | JIT compilation cache persistence via `JAX_COMPILATION_CACHE_DIR` | Locked project constraint |
| pyarrow | project constraint | Parquet read/write in aggregation | Used by `write_parquet_batch` |
| pandas | project constraint | CSV output in aggregation + recommendation | Used throughout power pipeline |
| matplotlib | project constraint | 4-panel figure generation | Used by `10_plot_power_curves.py` |
| git | cluster standard | Auto-push results branch | Used by `99_push_results.slurm` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `prl_hgf.power.curves.aggregate_parquets` | project | Concatenates per-chunk parquets | Called by `09_aggregate_power.py` |
| `prl_hgf.power.curves.compute_power_a/b` | project | Power curve computation | Called by aggregation and plotting scripts |
| `submit_power_pipeline.sh` | project | End-to-end SLURM submission | Single entry point for the sweep |

### Alternatives Considered
None — all tooling is already in place and locked.

**Installation:**
No new packages required. The `ds_env` conda environment and `pip install -e .` are already the setup pattern (handled by `submit_power_pipeline.sh --setup`).

## Architecture Patterns

### Recommended Project Structure
All files already exist:
```
cluster/
├── 08_power_sweep.slurm      # Wave 1: 3-chunk GPU/CPU array job
├── 09_power_postprocess.slurm # Wave 2: aggregation + figures + recommendation
├── 99_push_results.slurm      # Wave 3: git push (NEEDS FIX: add results/power/)
└── submit_power_pipeline.sh   # Entry point
scripts/
├── 08_run_power_iteration.py  # Chunk entry point (--legacy flag available)
├── 09_aggregate_power.py      # Aggregation
├── 10_plot_power_curves.py    # Figures
└── 11_write_recommendation.py # Recommendation
results/power/                 # Output directory (written by sweep)
```

### Pattern 1: Platform Selection from benchmark_batched.json
**What:** After Phase 14 benchmark completes, `results/power/benchmark_batched.json` contains `"decision": "gpu"` or `"decision": "cpu_comp"`. The SLURM script `08_power_sweep.slurm` currently uses `#SBATCH --partition=gpu` and `#SBATCH --gres=gpu:1`. If the decision is `cpu_comp`, these lines must be changed to `#SBATCH --partition=comp` with no GPU request. The `--time` may also need adjustment — CPU fits take longer per iteration.

**Decision gate formula:** `per_iter_seconds * 600 / 3600 > 50` → if > 50 GPU-hours per chunk, use CPU.

**When to use:** Read `benchmark_batched.json` before submitting the production sweep.

### Pattern 2: 3-Wave Pipeline Submission
**What:** The existing `submit_power_pipeline.sh` chains three waves:
- Wave 1: `sbatch cluster/08_power_sweep.slurm` (array 0-2, GPU or CPU)
- Wave 2: `sbatch --dependency=afterok:${SWEEP_JOBID} cluster/09_power_postprocess.slurm`
- Wave 3: `sbatch --dependency=afterany:${POSTPROC_JOBID} cluster/99_push_results.slurm`

**When to use:** Single invocation: `bash cluster/submit_power_pipeline.sh` after updating partition.

### Pattern 3: Expected Output Schema
Each `run_sbf_iteration` call (one per chunk task_id) returns `3 * len(n_per_group_grid) = 3 * 7 = 21` rows. The 3 is for the three sweep_types: `did_postdose`, `did_followup`, `linear_trend`. The 7 is for `n_per_group_grid = [10, 15, 20, 25, 30, 40, 50]`.

**Total rows in `power_master.csv`:**
- 600 iterations (3 effect sizes × 200 iterations per chunk)
- × 3 sweep_types
- × 7 N-levels
- = **12,600 rows**

The ROADMAP success criterion states "360,000 rows" — this is INCORRECT based on code inspection. The actual schema produces 21 rows per iteration × 600 iterations = 12,600 rows. The planner must use 12,600, not 360,000. Do not propagate the ROADMAP's erroneous number.

**Schema columns (13):** `sweep_type`, `effect_size`, `n_per_group`, `trial_count`, `iteration`, `parameter`, `bf_value`, `bf_exceeds`, `bms_xp`, `bms_correct`, `recovery_r`, `n_divergences`, `mean_rhat`.

Note: `parameter` is always `"omega_2"` in the current implementation. Each row represents one (sweep_type, N_level, iteration, effect_size) combination.

### Pattern 4: Missing-Cell Detection in aggregate_parquets
`aggregate_parquets()` in `curves.py` compares actual row counts against the expected count from `load_power_config()`. If a chunk job failed or wrote fewer rows, a `UserWarning` is raised per sweep_type. The verification plan should check the aggregation log for these warnings.

### Pattern 5: 99_push_results.slurm Staging Gap
The push script currently stages:
- `data/fitted/*.csv`
- `results/validation/*.csv` and `*.png`
- `results/group_analysis/*.csv`
- `figures/*.png`

It does **not** stage `results/power/` files (CSVs, PDFs, PNGs, `recommendation.md`). This must be fixed by adding:
```bash
stage_files "results/power/*.csv" "power sweep CSVs"
stage_files "results/power/*.md" "power recommendation"
stage_files "results/power/*.pdf" "power figures PDF"
stage_files "results/power/*.png" "power figures PNG"
```

### Anti-Patterns to Avoid
- **Submitting without reading benchmark_batched.json first:** If the benchmark hasn't run, the partition choice is unknown. Don't guess; run `--benchmark` first if Phase 14 benchmark results aren't in `results/power/`.
- **Assuming 360,000 rows in power_master.csv:** The correct expected row count is 12,600. The ROADMAP number is wrong.
- **Using `--dependency=afterok` for Wave 3 push:** The existing `submit_power_pipeline.sh` correctly uses `afterany` for the push wave — this means the push runs even if post-processing partially fails. Do not change this to `afterok`.
- **Re-running all 3 chunks if only one fails:** SLURM array tasks are independent. A failed chunk can be re-submitted with `--array=N` for only the failed chunk ID and its output merged with the surviving chunks.
- **Committing results directly to main from the cluster:** The `99_push_results.slurm` correctly creates a results branch (`results/slurm-YYYYMMDD-HHMMSS`) and pushes that. The user merges locally.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet merging | Custom concat script | `aggregate_parquets()` in `curves.py` | Already handles schema validation + missing-cell warnings |
| Chunk re-submission | Full re-run | `sbatch --array=N cluster/08_power_sweep.slurm` | Only re-runs the failed chunk |
| Row count validation | Custom counter | `aggregate_parquets()` UserWarning system | Warns per sweep_type automatically |
| Git push from cluster | Custom push script | `99_push_results.slurm` | Already handles branch creation, conflict avoidance, SSH verification |

**Key insight:** Phase 15 is operational, not developmental. The code is complete. The plans should focus on configuration, execution, monitoring, and triage — not writing new modules.

## Common Pitfalls

### Pitfall 1: SLURM Partition Not Updated After Benchmark
**What goes wrong:** Production sweep submitted to `gpu` partition when benchmark decided `cpu_comp`. GPU queue wait time wasted; or submitted to `cpu_comp` when GPU was greenlit.
**Why it happens:** `08_power_sweep.slurm` has a hardcoded `#SBATCH --partition=gpu`. If someone submits without reading `benchmark_batched.json`, they use the wrong partition.
**How to avoid:** The plan should require reading `benchmark_batched.json` and updating the SLURM script (or passing `--partition=` override at sbatch time) before Wave 1 submission.
**Warning signs:** `benchmark_batched.json` exists and says `"decision": "cpu_comp"` but sweep is submitted to gpu partition.

### Pitfall 2: SSH/Git Authentication Not Configured on Compute Nodes
**What goes wrong:** `99_push_results.slurm` fails because compute nodes can't reach GitHub via SSH.
**Why it happens:** Compute nodes may not have SSH agent forwarding or git credential helpers configured. The push script explicitly checks `git ls-remote --exit-code origin HEAD` and exits with an error if it fails.
**How to avoid:** Test git push from a compute node (or login node) before the production run. Configure SSH agent forwarding or deploy key on the cluster.
**Warning signs:** Push job exits immediately with "Cannot reach origin remote" error in `cluster/logs/push_results_*.err`.

### Pitfall 3: JAX Compilation Cache Stale Between Phases
**What goes wrong:** Chunk 1 and Chunk 2 JIT cold start even though they should benefit from Chunk 0's compiled kernel. Causes extra wall time on GPU (but not a correctness issue).
**Why it happens:** `JAX_COMPILATION_CACHE_DIR` in `08_power_sweep.slurm` uses `/scratch/${_PROJECT}/${USER}/.jax_cache_gpu`. If this directory was purged, or if the chunks run on nodes with different GPU models, cache misses occur.
**How to avoid:** Verify the cache directory persists across jobs. Run chunks sequentially first (submit Chunk 0, verify Chunk 1 warm JIT < 5s) if suspecting cache issues.
**Warning signs:** All 3 chunks show similar elapsed time (no warmup benefit).

### Pitfall 4: Chunk Failure Leaves Missing Grid Cells
**What goes wrong:** If one chunk fails (OOM, walltime exceeded, node failure), its rows are missing from `power_master.csv`. The aggregation will warn but still write a partial result.
**Why it happens:** SLURM jobs can fail for transient reasons (node preemption, OOM on a specific iteration).
**How to avoid:** Check SLURM exit codes after the sweep. Re-run only the failed chunk. The chunk filenames are `job_{ARRAY_JOB_ID}_chunk_{CHUNK_ID:04d}.parquet` so a re-run writes a fresh file that `aggregate_parquets()` will pick up.
**Warning signs:** `aggregate_parquets()` prints `UserWarning` about missing rows; `power_master.csv` has fewer rows than expected (12,600 = 600 × 21).

### Pitfall 5: 99_push_results.slurm Stages No Power Files
**What goes wrong:** Wave 3 completes with "No new or modified result files to commit. STATUS: NOTHING TO PUSH" because `results/power/` globs are not in the staging list.
**Why it happens:** The existing `99_push_results.slurm` was written for the v1.0 pipeline outputs (fitted data, validation, group analysis). Phase 15 produces `results/power/` outputs that are not staged.
**How to avoid:** Add `results/power/` staging lines to `99_push_results.slurm` before submitting the production pipeline.
**Warning signs:** Push job reports "NOTHING TO PUSH" despite successful aggregation.

### Pitfall 6: recommend.md Has Placeholder Date
**What goes wrong:** `11_write_recommendation.py` hardcodes `**Generated:** 2026-04-07` in `generate_recommendation()`. The production recommendation will have the wrong date.
**Why it happens:** The `generate_recommendation` function in `scripts/11_write_recommendation.py` contains a hardcoded date in the header template.
**How to avoid:** Fix the date to use `datetime.date.today()` before the production run, or accept the wrong date and note it in the verification.
**Warning signs:** `recommendation.md` says "Generated: 2026-04-07" even though the production run was in April/May 2026.

## Code Examples

### Reading benchmark decision
```bash
# From project root on cluster
python -c "
import json
with open('results/power/benchmark_batched.json') as f:
    b = json.load(f)
print('Decision:', b['decision'])
print('GPU-hours/chunk:', b['gpu_hours_per_chunk'])
print('Per-iter seconds:', b['per_iteration_s'])
"
```

### Submitting with CPU comp partition override
```bash
# If benchmark said cpu_comp, override at submission time:
sbatch --partition=comp --gres="" --time=48:00:00 \
    --export="ALL,DRY_RUN=0,BENCHMARK=0,SAMPLER=numpyro" \
    cluster/08_power_sweep.slurm
```

### Re-running a single failed chunk
```bash
# Re-run only chunk 1 (effect size d=0.5) — replace 12345 with actual ARRAY_JOB_ID
sbatch --array=1 --export="ALL,DRY_RUN=0,BENCHMARK=0,SAMPLER=numpyro" \
    cluster/08_power_sweep.slurm
```

### Verifying row count after aggregation
```python
import pandas as pd
df = pd.read_csv("results/power/power_master.csv")
expected = 600 * 3 * 7  # 600 iterations × 3 sweep_types × 7 N-levels
print(f"Rows: {len(df)} (expected: {expected})")
assert len(df) == expected, f"Missing {expected - len(df)} rows"
```

### Adding results/power/ to 99_push_results.slurm
The following lines should be added after the existing `stage_files` calls in `99_push_results.slurm`:
```bash
stage_files "results/power/*.csv" "power sweep CSVs"
stage_files "results/power/*.md" "power recommendation"
stage_files "results/power/*.pdf" "power figures PDF"
stage_files "results/power/*.png" "power figures PNG"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.1 per-participant sequential MCMC | v1.2 batched hierarchical (one NUTS call for all participants) | Phase 12 | Amortises PCIe dispatch; GPU-feasible |
| v1.1 NumPy simulate_batch | JAX vmap simulate_batch (simulate_cohort_jax internally) | Phase 13 | ~10x faster simulation |
| Placeholder power_master.csv | Real sweep results from GPU/CPU cluster | Phase 15 (this phase) | Produces the actual v1.1 deliverables |

**Deprecated/outdated:**
- The ROADMAP success criterion "360,000 rows" is wrong. The correct expected row count from `run_sbf_iteration` is **12,600 rows** (3 sweep_types × 7 N-levels × 200 iterations × 3 effect sizes). The planner should verify this against actual output and not use 360,000.

## Open Questions

1. **Hardcoded date in recommendation.md header**
   - What we know: `scripts/11_write_recommendation.py` line 147 has `**Generated:** 2026-04-07` hardcoded in `generate_recommendation()`.
   - What's unclear: Whether the planner wants to fix this in Phase 15 or accept it.
   - Recommendation: Fix it (1-line change: `from datetime import date` + `f"**Generated:** {date.today()}"`) before the production run.

2. **Platform decision unknown until Phase 14 benchmark runs on GPU hardware**
   - What we know: Phase 14 code is complete; `--benchmark` flag exists; `benchmark_batched.json` schema is defined. But Phase 14 benchmark has not actually run on real GPU hardware yet (it was coded locally).
   - What's unclear: Whether the decision will be gpu or cpu_comp.
   - Recommendation: Phase 15 Plan 1 should be: "Read benchmark_batched.json → update SLURM script → submit dry-run → submit full sweep." If benchmark hasn't run yet, run it first as part of Plan 1.

3. **Expected divergence rate on real data**
   - What we know: Success criterion requires < 5% divergent-chain exclusion rate per cell. On synthetic data in testing, VALID-02 passed. Real production fits with higher N (up to 50/group) may behave differently.
   - What's unclear: Whether the 2 chains × 500 draws × 500 tune settings are sufficient for all N levels.
   - Recommendation: Check exclusion rate in recommendation.md section 7 after aggregation. If > 5%, consider increasing n_tune or target_accept for a targeted re-run.

4. **SSH/git authentication on compute nodes**
   - What we know: `99_push_results.slurm` requires `git ls-remote` to succeed from a compute node.
   - What's unclear: Whether M3 cluster compute nodes have SSH agent forwarding or a deploy key configured.
   - Recommendation: Plan 1 should include a manual SSH test from a compute node before Wave 3 submission.

## Sources

### Primary (HIGH confidence)
- Codebase: `scripts/08_run_power_iteration.py` — full entry point, CLI flags, chunk logic (read directly)
- Codebase: `scripts/09_aggregate_power.py` — aggregation pipeline (read directly)
- Codebase: `scripts/10_plot_power_curves.py` — plotting pipeline (read directly)
- Codebase: `scripts/11_write_recommendation.py` — recommendation generator (read directly)
- Codebase: `cluster/08_power_sweep.slurm` — SLURM array job script (read directly)
- Codebase: `cluster/09_power_postprocess.slurm` — post-processing SLURM script (read directly)
- Codebase: `cluster/99_push_results.slurm` — push script, confirmed missing `results/power/` staging (read directly)
- Codebase: `cluster/submit_power_pipeline.sh` — 3-wave pipeline submission (read directly)
- Codebase: `src/prl_hgf/power/iteration.py` — `run_sbf_iteration` returns 3*7=21 rows (read directly)
- Codebase: `src/prl_hgf/power/schema.py` — POWER_SCHEMA 13 columns (read directly)
- Codebase: `src/prl_hgf/power/curves.py` — `aggregate_parquets` missing-cell warning logic (read directly)
- Codebase: `.planning/STATE.md` — Phase 14 complete; benchmark code ready; next action: Phase 15

### Secondary (MEDIUM confidence)
- `.planning/phases/14-integration-gpu-benchmark/14-02-SUMMARY.md` — confirms benchmark JSON schema + decision gate is functional

### Tertiary (LOW confidence)
- ROADMAP.md success criterion "360,000 rows" — this is INCORRECT per code inspection; flagged as open question

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all code read directly; no new dependencies
- Architecture (pipeline): HIGH — all three SLURM scripts and 4 Python scripts read directly
- Architecture (row count): HIGH — computed directly from `run_sbf_iteration` return spec (3 × 7 × 600 = 12,600)
- Pitfalls: HIGH — all identified from direct code reading (staging gap confirmed in 99_push_results.slurm; hardcoded date confirmed in 11_write_recommendation.py)
- Open questions: MEDIUM — platform decision and SSH auth depend on cluster runtime behavior

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable; all dependencies pinned; code is frozen)
