---
phase: 10-core-power-sweep
plan: 03
subsystem: power-analysis
tags: [power, bfda, mcmc, bms, waic, parquet, slurm]

# Dependency graph
requires:
  - phase: 10-core-power-sweep/01
    provides: compute_all_contrasts, compute_jzs_bf for BF path
  - phase: 10-core-power-sweep/02
    provides: decode_task_id, total_grid_size for SLURM grid decode
provides:
  - run_power_iteration() — full simulate->fit->BF->BMS pipeline for one grid cell
  - build_arrays_from_sim() — trial array construction for WAIC computation
  - Entry point wired to full pipeline with 3-parquet-per-task output
  - 5 unit tests covering iteration pipeline and dry-run compat
affects: [10-core-power-sweep/04, 10-core-power-sweep/05]

# Tech tracking
tech-stack:
  added: []
  patterns: [incremental-waic, mock-based-pipeline-tests]

key-files:
  created:
    - src/prl_hgf/power/iteration.py
    - tests/test_power_iteration.py
  modified:
    - src/prl_hgf/power/__init__.py
    - scripts/08_run_power_iteration.py

key-decisions:
  - "Incremental WAIC: process 3-level idata first, delete it, then process 2-level to limit peak memory to one model's posterior samples"
  - "build_arrays_from_sim duplicates _build_arrays logic rather than importing private function from fitting.batch"
  - "3 parquet files per task using sweep_type suffix in filename (e.g. _did_postdose.parquet)"
  - "n_divergences placeholder is always 0 since fit_batch does not currently produce this column"

patterns-established:
  - "Per-contrast parquet output: one file per sweep_type per SLURM task"
  - "Mock-based pipeline testing: monkeypatch all heavy dependencies to test orchestration logic"

# Metrics
duration: 15min
completed: 2026-04-07
---

# Plan 03: Core Power Iteration Pipeline Summary

**run_power_iteration orchestrates simulate->fit(3L)->BF->diagnostics->fit(2L)->BMS with incremental WAIC and 3-parquet output per task**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created power/iteration.py with full simulate->fit->BF->BMS pipeline returning 3 POWER_SCHEMA-conforming dicts
- Implemented incremental WAIC in _compute_bms_power that deletes idata_3level before processing 2-level, keeping peak memory to one model
- Updated entry point to replace stub with decode_task_id -> run_power_iteration -> write 3 parquet files
- All 22 power tests pass (5 iteration + 10 contrasts + 7 grid)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create power/iteration.py with run_power_iteration and helpers** - `cada724` (feat)
2. **Task 2: Update entry point script and write unit tests** - `9e42500` (feat)

## Files Created/Modified
- `src/prl_hgf/power/iteration.py` - Core power iteration pipeline: build_arrays_from_sim, _compute_bms_power, _extract_diagnostics, run_power_iteration
- `src/prl_hgf/power/__init__.py` - Added run_power_iteration and build_arrays_from_sim exports
- `scripts/08_run_power_iteration.py` - Replaced stub with full pipeline; dry-run path preserved with output_path moved inside block
- `tests/test_power_iteration.py` - 5 unit tests: array builder, partial feedback, return structure, schema conformance, dry-run compat

## Decisions Made
- Incremental WAIC pattern: 3-level first, delete, then 2-level — limits peak memory to one model's idata at a time
- Duplicated _build_arrays logic in build_arrays_from_sim rather than importing private function
- n_divergences is a placeholder (always 0) since fit_batch doesn't produce that column yet

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed pandas indexing type in test assertion**
- **Found during:** Task 2 (unit tests)
- **Issue:** `subset.iloc[t]["cue_chosen"]` returns a numpy scalar that cannot index numpy arrays directly
- **Fix:** Added explicit `int()` and `float()` casts in test_build_arrays_from_sim_matches_build_arrays
- **Files modified:** tests/test_power_iteration.py
- **Verification:** All 5 tests pass
- **Committed in:** 9e42500 (Task 2 commit)

**2. [Rule 3 - Blocking] Fixed import sorting in test file**
- **Found during:** Task 2 (unit tests)
- **Issue:** ruff I001 import block unsorted
- **Fix:** Ran `ruff check --fix` to auto-sort imports
- **Files modified:** tests/test_power_iteration.py
- **Verification:** `ruff check` passes
- **Committed in:** 9e42500 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Minor test fixes, no scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Core power pipeline is complete and tested via mocks
- Ready for Plan 10-04 (aggregation/results collection) or end-to-end integration testing
- All 22 power module tests pass (5 iteration + 10 contrasts + 7 grid)

---
*Phase: 10-core-power-sweep*
*Completed: 2026-04-07*
