---
phase: 11-aggregation-publication
plan: 01
subsystem: power-analysis
tags: [pandas, parquet, pyarrow, power-analysis, bfda, bms, pytest, monkeypatch]

# Dependency graph
requires:
  - phase: 10-power-sweep
    provides: per-job parquet files with 13-column POWER_SCHEMA, grid/config/schema modules
provides:
  - aggregate_parquets(): glob + concat + schema validation + missing-cell warnings
  - compute_power_a(): P(BF > threshold) grouped by (n_per_group, effect_size)
  - compute_power_b(): P(correct BMS) grouped by n_per_group with deduplication
  - scripts/09_aggregate_power.py: CLI writing power_master.csv + summary CSVs
  - 9 unit tests covering all three public functions
affects:
  - 11-02 (power figures — imports compute_power_a and compute_power_b)
  - 11-03 (N recommendation — reads power_a_summary.csv and power_b_summary.csv)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "monkeypatch load_power_config via unittest.mock.patch for grid-dimension tests"
    - "drop_duplicates(subset=['n_per_group','iteration']) to deduplicate BMS rows before group mean"
    - "groupby().agg(col='mean', n='count').reset_index() pattern for power curve frames"

key-files:
  created:
    - src/prl_hgf/power/curves.py
    - scripts/09_aggregate_power.py
    - tests/test_power_curves.py
  modified:
    - src/prl_hgf/power/__init__.py

key-decisions:
  - "Missing-cell warning threshold uses total_grid_size() from grid.py against per-sweep_type actual count"
  - "compute_power_b deduplicates on (n_per_group, iteration) before computing mean — avoids triple-counting the 3 sweep_type rows written per SLURM task"
  - "bf_threshold parameter documents intent but uses pre-computed bf_exceeds bool column from schema directly"
  - "SIM117 fix: merged nested with-patch/pytest.warns into single parenthesized with statement (Python 3.10+)"

patterns-established:
  - "Windows-path-safe FileNotFoundError match: use plain English substring, not str(path) which contains backslashes"
  - "pytest.approx on pandas Series: convert .tolist() first, not (series == approx()).all()"

# Metrics
duration: 7min
completed: 2026-04-07
---

# Phase 11 Plan 01: Power Aggregation and Curve Computation Summary

**Parquet aggregation module (aggregate_parquets + compute_power_a + compute_power_b) with missing-cell warnings, BMS deduplication, CLI pipeline script, and 9 passing unit tests**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-07T20:56:24Z
- **Completed:** 2026-04-07T21:03:25Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `src/prl_hgf/power/curves.py`: three public functions with NumPy docstrings and `__all__`
- `scripts/09_aggregate_power.py`: argparse CLI writing three CSVs (master + two summaries) with summary statistics printout
- `tests/test_power_curves.py`: 9 unit tests, all passing, 283 lines; covers empty-dir error, single/multi parquet concat, missing-cell `UserWarning`, P(BF>threshold) computation and grouping, P(BMS correct) with and without deduplication

## Task Commits

1. **Task 1: Create power/curves.py and update __init__.py** - `2b75caf` (feat)
2. **Task 2: Pipeline script and unit tests** - `2ed77c0` (test)

## Files Created/Modified

- `src/prl_hgf/power/curves.py` — aggregate_parquets, compute_power_a, compute_power_b
- `src/prl_hgf/power/__init__.py` — re-exports three new functions
- `scripts/09_aggregate_power.py` — CLI aggregation pipeline
- `tests/test_power_curves.py` — 9 unit tests for all public functions

## Decisions Made

- `compute_power_b` deduplicates on `(n_per_group, iteration)` before computing the mean of `bms_correct`. Each SLURM task writes 3 parquet rows (one per sweep_type), all with identical BMS values; deduplication prevents triple-counting.
- Missing-cell warning uses `total_grid_size()` as the expected count per `sweep_type` and warns per `sweep_type` independently.
- `bf_threshold` parameter on `compute_power_a` documents intent but the pre-computed `bf_exceeds` boolean column from POWER_SCHEMA is used directly — no re-thresholding at aggregation time.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Windows backslash in FileNotFoundError match regex**

- **Found during:** Task 2 (unit tests first run)
- **Issue:** `pytest.raises(FileNotFoundError, match=str(tmp_path))` failed on Windows because backslash `\U` in path string is an incomplete regex escape
- **Fix:** Changed match string to `"No .parquet files found"` — stable substring independent of OS path separator
- **Files modified:** tests/test_power_curves.py
- **Verification:** test_aggregate_empty_dir_raises passes on Windows
- **Committed in:** 2ed77c0 (Task 2 commit)

**2. [Rule 1 - Bug] pytest.approx incompatible with set and pandas Series .all()**

- **Found during:** Task 2 (unit tests first run)
- **Issue 1:** `pytest.approx({0.3, 0.5})` raises TypeError — approx does not support unordered sets
- **Issue 2:** `(series == pytest.approx(0.6)).all()` returns np.False_ because the comparison yields an ApproxScalar, not a boolean array
- **Fix:** Changed to `sorted(result["effect_size"].unique()) == pytest.approx([0.3, 0.5])` and `result["p_bf_exceeds"].tolist() == pytest.approx([0.6]*4)`
- **Files modified:** tests/test_power_curves.py
- **Verification:** test_compute_power_a_groups_correctly passes
- **Committed in:** 2ed77c0 (Task 2 commit)

**3. [Rule 2 - Missing Critical] SIM117 ruff lint: nested with statements**

- **Found during:** Task 2 (ruff check)
- **Issue:** Nested `with patch(...): with pytest.warns(...)` triggers SIM117
- **Fix:** Merged into single parenthesized `with` statement (Python 3.10+ syntax)
- **Files modified:** tests/test_power_curves.py
- **Verification:** ruff check passes; test still passes
- **Committed in:** 2ed77c0 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 test correctness bugs, 1 lint fix)
**Impact on plan:** All fixes were within Task 2 test file; no scope creep. curves.py and 09_aggregate_power.py required no corrections.

## Issues Encountered

None in the module or script code. All three deviations were confined to the test file (Windows regex, pytest.approx API constraints, ruff SIM117).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `aggregate_parquets`, `compute_power_a`, `compute_power_b` are ready for import by Phase 11 Plan 02 (power figures)
- `power_master.csv`, `power_a_summary.csv`, `power_b_summary.csv` will be available once `09_aggregate_power.py` is run against real sweep results
- No blockers for Phase 11 Plan 02

---
*Phase: 11-aggregation-publication*
*Completed: 2026-04-07*
