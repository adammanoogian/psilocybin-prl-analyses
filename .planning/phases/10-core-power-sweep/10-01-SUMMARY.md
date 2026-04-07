---
phase: 10-core-power-sweep
plan: 01
subsystem: analysis
tags: [pingouin, jzs-bf, bayes-factor, scipy, contrast]

# Dependency graph
requires:
  - phase: 08-power-infra
    provides: power/ package skeleton, write_parquet_row, make_power_config
provides:
  - compute_jzs_bf wrapping pingouin.bayesfactor_ttest with Welch t and r=sqrt(2)/2
  - compute_did_contrast extracting per-participant DiD vectors from fit_df
  - compute_linear_trend_contrast with [-1, 0, +1] weights
  - compute_all_contrasts returning 3 contrast dicts with bf_value and bf_exceeds
affects: [10-core-power-sweep, iteration, power-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns: [JZS BF wrapper via pingouin, DiD pivot extraction, linear trend contrast]

key-files:
  created:
    - src/prl_hgf/power/contrasts.py
    - tests/test_power_contrasts.py
  modified:
    - src/prl_hgf/power/__init__.py

key-decisions:
  - "compute_jzs_bf takes pre-extracted DiD arrays (not fit_df) for composability with different contrast types"
  - "compute_did_contrast and compute_linear_trend_contrast return raw arrays, not BF values, enabling custom r or threshold logic downstream"
  - "Test helper _make_test_fit_df uses noise_sd parameter to control within-group variance; noise_sd=0 for exact deterministic DiD checks, noise_sd>0 for BF-compatible tests"

patterns-established:
  - "JZS BF wrapper: Welch t-test via scipy, then pg.bayesfactor_ttest with r, nx, ny"
  - "Contrast extraction: filter by parameter, pivot_table on session, compute difference per participant, split by group"
  - "Synthetic fit_df builder for tests: _make_test_fit_df(n_per_group, effect_size, noise_sd)"

# Metrics
duration: 9min
completed: 2026-04-07
---

# Plan 01: JZS BF and Contrast Functions Summary

**JZS Bayes Factor wrapper and 3 contrast types (DiD post-dose, DiD followup, linear trend) with 10 unit tests including JASP reference validation**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-07T19:58:31Z
- **Completed:** 2026-04-07T20:07:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `power/contrasts.py` with 4 exported functions wrapping pingouin and scipy for JZS BF computation
- Three contrast types: did_postdose, did_followup, linear_trend each producing (psi_did, plc_did) arrays
- 10 unit tests covering accuracy, edge cases, JASP reference match (<1% relative error), and structural correctness
- Updated `power/__init__.py` with re-exports for all 4 new functions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create power/contrasts.py with JZS BF and contrast functions** - `ad9c6a5` (feat)
2. **Task 2: Write unit tests for contrast functions** - `7e1a3be` (test)

**Plan metadata:** see below (docs: complete plan)

## Files Created/Modified
- `src/prl_hgf/power/contrasts.py` - JZS BF computation and contrast extraction (4 functions)
- `tests/test_power_contrasts.py` - 10 unit tests for all contrast functions
- `src/prl_hgf/power/__init__.py` - Re-exports for compute_jzs_bf, compute_did_contrast, compute_linear_trend_contrast, compute_all_contrasts

## Decisions Made
- `compute_jzs_bf` takes pre-extracted DiD arrays rather than full fit_df, enabling reuse across different contrast types without re-pivoting
- Test helper uses a `noise_sd` parameter: `0.0` for exact deterministic DiD assertions, `>0` for tests requiring valid within-group variance (BF computation)
- compute_all_contrasts uses `>` threshold comparison (strictly greater), consistent with standard BF interpretation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Zero-variance DiD arrays caused NaN BF in tests 9/10**
- **Found during:** Task 2 (unit test execution)
- **Issue:** Synthetic fit_df helper produced exact DiD constants (e.g. all 2.0) with zero within-group variance, causing scipy.stats.ttest_ind to produce NaN t-statistics
- **Fix:** Added `noise_sd` parameter to `_make_test_fit_df`; tests 9/10 use `noise_sd=0.1`/`0.15` while tests 6/8 keep `noise_sd=0` for exact value checks
- **Files modified:** tests/test_power_contrasts.py
- **Verification:** All 10 tests pass; ruff clean
- **Committed in:** 7e1a3be (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix was necessary for test correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- contrast functions ready for `run_power_iteration()` in plan 10-03 (or later iteration plan)
- All 4 functions importable from `prl_hgf.power` top-level
- Test patterns established for future power module tests

---
*Phase: 10-core-power-sweep*
*Completed: 2026-04-07*
