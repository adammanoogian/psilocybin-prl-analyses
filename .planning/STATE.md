# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-07)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Milestone v1.1 Power Analysis — Phase 8 in progress

## Current Position

Phase: 8 - Config + Infrastructure (in progress)
Plan: 2 of 3
Status: In progress
Last activity: 2026-04-07 — Completed 08-02-PLAN.md (seeds, schema, SLURM template, entry point)

[============>       ] Phase 8 Plan 2/3 complete (v1.0 shipped; v1.1 in progress)

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

### Pending Todos

- manuscript/references.bib: mason2024 volume/page details need verification before submission
- quarto-arxiv extension must be installed before first arxiv-pdf render
- Phase 9 kappa effect size parameterization: verify kappa entry point in GroupConfig vs SessionConfig during Plan 02 unit tests (kappa delta lives in SessionConfig.kappa_deltas — confirmed in 08-01 tests, but grid parameterization for kappa needs review)
- Phase 10: run 100-iteration MAP vs NUTS pilot before committing to full NUTS budget

### Blockers/Concerns

- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env
- omega_3 parameter recovery expected to be challenging (known issue in literature)
- MCMC-based power loop is compute-heavy (~200-300 cluster-hours estimated)
- SLURM %50 throttle tuned from general guidance — verify empirically in Phase 8 smoke test

## Quick Tasks

| ID  | Name | Status | Summary |
|-----|------|--------|---------|
| 001 | Cluster GPU Setup & Smoke Test | Complete | M3 SLURM infrastructure + smoke test PASS |

## Session Continuity

Last session: 2026-04-07T15:37:54Z
Stopped at: Completed 08-02-PLAN.md
Resume file: None
Next action: Execute 08-03-PLAN.md
