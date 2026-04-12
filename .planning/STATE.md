# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-07)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Milestone v1.2 Hierarchical GPU Fitting — starting Phase 12 (refactor fitting pipeline so GPU actually accelerates)

## Current Position

Phase: 12 of 15 (Batched Hierarchical JAX Logp)
Plan: 02 of 04
Status: In progress
Last activity: 2026-04-12 — Completed 12-02-PLAN.md (batched JAX logp Op factory)

[==========██==]     v1.1 code-complete (Phases 1-11); v1.2 plans 12-01, 12-02 complete, 12-03 next

## Performance Metrics

| Metric | v1.0 Value |
|--------|------------|
| Phases shipped | 7 |
| Plans completed | 18 |
| Files created/modified | 105 |
| Lines of Python | ~11,016 |
| Days elapsed | 4 (2026-04-04 to 2026-04-07) |

## Accumulated Context

### Key Decisions

See `.planning/milestones/v1.0-ROADMAP.md` for v1.0 decision log.

| Decision | Rationale | Phase |
|----------|-----------|-------|
| MCMC throughout (no MAP proxy) | Simplicity; leverage cluster parallelism for power loop | v1.1 planning |
| Psilocybin vs placebo groups (both post-concussion) | Corrected study design | v1.1 planning |
| pingouin.bayesfactor_ttest over rpy2+anovaBF | pingouin already installed; anovaBF misspecified for RM designs (van den Bergh 2023) | v1.1 roadmap |
| SLURM array %50 throttle | Prevents Lustre metadata import storm on M3 MASSIVE | v1.1 roadmap |
| omega_3 BFDA labeled "exploratory — upper bound" | Recovery r ~ 0.67 with binary data; naive BFDA inflates power 20-40pp | v1.1 roadmap |
| SeedSequence for parallel RNG | task-ID integer seeding gives correlated streams; SeedSequence guarantees independence | v1.1 roadmap |
| power/ package wraps existing pipeline; no existing module modified | Eliminates regression risk; all existing functions called unchanged | v1.1 roadmap |
| make_power_config shifts psilocybin omega_2_deltas only; placebo unchanged | Study hypothesis: psilocybin increases learning rate; placebo is inert control | 08-01 |
| load_power_config reads only power: YAML key; does not re-parse task/simulation/fitting | Clean separation of concerns; existing load_config unaffected | 08-01 |
| write_parquet_row rejects missing AND extra columns | Strict schema enforcement prevents silent drift as Phase 10 adds real pipeline results | 08-02 |
| --output-dir flag on entry point for test isolation | Integration tests write to tmp_path, not results/; keeps test suite clean | 08-02 |
| make_trial_config scales only PhaseConfig.n_trials; TransferConfig.n_trials untouched | Transfer trials are fixed study design; only acquisition/reversal phases vary for trial count studies | 09-01 |
| run_recovery_precheck filters to baseline only before fit_batch | Avoids 3x compute cost; post_dose and followup sessions not needed for recovery gate | 09-01 |
| omega_3 eligibility always "exploratory -- upper bound" regardless of r | Locked decision: binary PRL data r~0.67 in literature; BFDA inflation 20-40pp | 09-01 |
| n_flagged uses groupby("participant_id")["flagged"].any().sum() | fit_df has per-parameter rows; participant is flagged if ANY parameter row is flagged | 09-01 |
| run_trial_sweep passes min_n=0 to build_recovery_df | Small trial counts may lose many participants to convergence failures; downstream callers apply own filter | 09-02 |
| find_minimum_trial_count excludes omega_3 from all-must-pass by default | Consistent with locked exploratory decision; omega_3 never gates the trial count requirement | 09-02 |
| seed+idx seeding per grid point in run_trial_sweep | Fresh independent participants at each trial count; no between-condition correlation in recovery estimates | 09-02 |
| compute_power_b deduplicates on (n_per_group, iteration) before mean | Each SLURM task writes 3 parquet rows (one per sweep_type) with identical BMS values; dedup prevents triple-counting | 11-01 |
| bf_threshold parameter on compute_power_a is documentary; uses pre-computed bf_exceeds bool column directly | No re-thresholding at aggregation time — threshold is baked into schema at write time | 11-01 |
| plot_power_a recomputes from raw bf_value (not bf_exceeds) so --bf-threshold CLI arg is live | Enables users to replot at different thresholds without re-running the sweep | 11-02 |
| plot_combined_figure saves both PDF and PNG from same stem | Grant submission needs PDF; PNG for quick inspection | 11-02 |
| _draw_*_panel helpers separate subplot rendering from file I/O | Enables test isolation and reuse in combined figure without recursive file saves | 11-02 |
| generate_recommendation re-applies bf_threshold to bf_value for N selection | Allows any --bf-threshold CLI arg to affect recommended N, not just pre-baked bf_exceeds bool | 11-03 |
| power_a_df and power_b_df recomputed from master if summary CSVs missing | Makes recommendation script self-sufficient when only power_master.csv exists | 11-03 |
| Chunk-based SLURM: 3 jobs instead of 4200 | JAX compiles once per chunk; reuses compiled model for ~1400 iterations; one combined parquet per chunk | 11 |
| legacy/batch.py imports from legacy/single.py (not shim) | Ensures frozen code calls frozen code; no circular dependency through shims | 12-01 |
| Shims use noqa: F401 for re-exports | Ruff would flag unused imports in shim modules; F401 suppression is standard for re-export patterns | 12-01 |
| Data as runtime args for vmap (not closure-over-data) | Clean vmap signature; XLA sees full data flow; no closure recreation on data change | 12-02 |
| Separate named functions for 2-level/3-level logp | Avoids mypy error from conditional redefinition with different signatures | 12-02 |
| Level-2 mean key is attrs[i]["mean"] | Confirmed via runtime inspection of pyhgf attribute pytree structure | 12-02 |

### Pending Todos

- manuscript/references.bib: mason2024 volume/page details need verification before submission
- quarto-arxiv extension must be installed before first arxiv-pdf render
- Phase 10 kappa effect size parameterization: verify kappa entry point in GroupConfig vs SessionConfig during Phase 10 planning (kappa delta lives in SessionConfig.kappa_deltas — confirmed in 08-01 tests, but grid parameterization for kappa needs review before Phase 10 sweep)
- Phase 10: run 100-iteration MAP vs NUTS pilot before committing to full NUTS budget

### Blockers/Concerns

- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env
- omega_3 parameter recovery expected to be challenging (known issue in literature)
- **v1.1 per-participant sequential MCMC is GPU-pessimal** — L40S benchmark showed ~1.5s/NUTS-sample vs ~5ms on CPU due to PCIe dispatch overhead. v1.2 refactor is mandatory for GPU feasibility.
- **Decision gate at Phase 14:** if batched hierarchical GPU benchmark is still > 50 GPU-hours per chunk, fall back to CPU `comp` partition (new batched code still wins over v1.1 sequential on CPU).
- pyhgf has no built-in NaN clamping — **RESOLVED in 12-02**: Layer 2 clamping implemented in hierarchical.py using jnp.where + tree_map (|mu_2| < 14 bound).
- `_init_jitter` PyTensor read-only-array bug means we can't use `pm.sample(...)` directly even with `nuts_sampler="numpyro"`; must call `pmjax.sample_numpyro_nuts()` directly.

## Quick Tasks

| ID  | Name | Status | Summary |
|-----|------|--------|---------|
| 001 | Cluster GPU Setup & Smoke Test | Complete | M3 SLURM infrastructure + smoke test PASS |

## Session Continuity

Last session: 2026-04-12
Stopped at: Completed 12-02-PLAN.md (batched JAX logp Op factory)
Resume file: None
Next action: Execute 12-03-PLAN.md (hierarchical PyMC model wrapper)
