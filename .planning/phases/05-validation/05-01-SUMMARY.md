---
phase: 05-validation
plan: 01
subsystem: analysis
tags: [pandas, scipy, matplotlib, seaborn, pearson-r, parameter-recovery, hgf]

# Dependency graph
requires:
  - phase: 03-simulation
    provides: simulate_batch output with true_* columns (trial-level sim_df)
  - phase: 04-fitting
    provides: fit_batch output in long-form FIT-04 schema with flagged column
provides:
  - build_recovery_df: joins sim_df + fit_df into wide recovery DataFrame
  - compute_recovery_metrics: per-parameter r, p, bias, RMSE with threshold flag
  - compute_correlation_matrix: cross-parameter posterior-mean correlation
  - plot_recovery_scatter: multi-panel true vs recovered scatter figures
  - plot_correlation_matrix: seaborn heatmap of parameter correlation matrix
  - Unit tests validating all public functions
affects:
  - 05-02 (pipeline script calling these functions on batch fit output)
  - 05-03 (any BMS or group-level analysis consuming recovery results)
  - 07-gui (recovery plots displayed in interactive dashboard)

# Tech tracking
tech-stack:
  added: [seaborn (heatmap), matplotlib Agg backend]
  patterns:
    - groupby(...).first().reset_index() to reduce trial-level sim_df to one row per participant-session
    - pivot_table(index=[id cols], columns="parameter", values="mean") to widen long-form fit_df
    - Inner merge on participant_id/group/session as the join key between sim and fit outputs
    - min_n guard in build_recovery_df enforces REC-04 statistical power requirement
    - omega_3 receives caveat annotation in scatter plots (known poor recovery)

key-files:
  created:
    - src/prl_hgf/analysis/recovery.py
    - src/prl_hgf/analysis/plots.py
    - tests/test_recovery.py
  modified:
    - src/prl_hgf/analysis/__init__.py

key-decisions:
  - "zip(axes, params, strict=True) used in plot loop per ruff B905 requirement"
  - "Fixture offsets scaled to sd=1.0 (not 0.5) for kappa/beta/zeta to ensure SNR >> 1 for the near-perfect recovery test with only 9 participants"
  - "Module-scoped fixture for synthetic data (not session-scoped) — no JAX compilation needed, so module scope is sufficient and avoids cross-test state"

patterns-established:
  - "Recovery DataFrame has wide form: one row per participant-session, columns are true_* and fitted param names"
  - "compute_recovery_metrics skips parameters not present in recovery_df (2-level model omits omega_3, kappa)"
  - "All plot functions accept save_path=None and always return Figure (caller responsible for plt.close)"

# Metrics
duration: 6min
completed: 2026-04-06
---

# Phase 5 Plan 1: Recovery Analysis Module Summary

**Pearson r/bias/RMSE recovery metrics with true-vs-recovered scatter plots and cross-parameter correlation heatmap, enforcing REC-04 min_n=30 guard**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-06T13:12:26Z
- **Completed:** 2026-04-06T13:18:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `recovery.py` delivers the three core analysis functions (`build_recovery_df`,
  `compute_recovery_metrics`, `compute_correlation_matrix`) with full edge-case
  handling (empty merge, NaN masking, missing parameters, flagged exclusion, min_n guard)
- `plots.py` provides Agg-backend scatter and heatmap helpers with omega_3 caveat
  annotation and high-correlation concern notes
- 8-test suite in `test_recovery.py` covers all public API paths including the
  REC-04 min_n enforcement and plot smoke tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Recovery analysis module** - `843ded8` (feat)
2. **Task 2: Visualization helpers and unit tests** - `23c9e2e` (feat)

**Plan metadata:** (see below)

## Files Created/Modified

- `src/prl_hgf/analysis/recovery.py` — `build_recovery_df`, `compute_recovery_metrics`, `compute_correlation_matrix`
- `src/prl_hgf/analysis/plots.py` — `plot_recovery_scatter`, `plot_correlation_matrix` (Agg backend)
- `src/prl_hgf/analysis/__init__.py` — updated to export all three recovery functions
- `tests/test_recovery.py` — 8 unit tests for recovery analysis pipeline

## Decisions Made

- **zip strict=True in plot loop**: ruff B905 requires explicit `strict=` parameter; `strict=True` chosen because `axes` and `params` are always the same length by construction.
- **Fixture offset scale sd=1.0 for all params**: with only 9 unflagged participants in the test fixture, kappa (originally `offset * 0.1`) had insufficient inter-subject variance for reliable Pearson r estimation. Scaling all offsets to ±1.0 standard deviations gives SNR >> 1 relative to the 0.05 recovery noise.
- **module-scoped fixture**: `synthetic_data` is module-scoped (not function-scoped) to avoid rebuilding DataFrames per test, and session-scoped is unnecessary since no JAX compilation occurs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed kappa near-zero variance causing spurious Pearson r in test fixture**

- **Found during:** Task 2 (running `test_compute_recovery_metrics_perfect`)
- **Issue:** `kappa` offset was `offset * 0.1` with `sd=0.5` master offset, producing only ~0.05 variance across 9 participants. The 0.05 recovery noise was comparable to signal, giving r=0.13 instead of >0.95.
- **Fix:** Increased all parameter offset multipliers to 0.5 and master offset sd to 1.0, giving inter-subject spread ~10x the recovery noise.
- **Files modified:** `tests/test_recovery.py`
- **Verification:** All 8 tests pass, kappa r > 0.95 confirmed.
- **Committed in:** `23c9e2e` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test fixture)
**Impact on plan:** Test fixture bug fix only; no changes to production code. No scope creep.

## Issues Encountered

None beyond the test fixture variance issue documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All three recovery analysis functions are importable and tested.
- Pipeline script (05-02) can call `build_recovery_df` + `compute_recovery_metrics` + `plot_recovery_scatter` directly.
- omega_3 recovery caveat is documented and will appear in scatter plots automatically.
- No blockers; plan 05-02 can proceed immediately.

---
*Phase: 05-validation*
*Completed: 2026-04-06*
