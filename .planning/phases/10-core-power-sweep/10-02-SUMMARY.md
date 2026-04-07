---
phase: 10-core-power-sweep
plan: 02
subsystem: infra
tags: [slurm, power-analysis, bfda, grid-decode, mcmc]

# Dependency graph
requires:
  - phase: 09-power-config-seeds-schema
    provides: PowerConfig dataclass, load_power_config, YAML power section
provides:
  - decode_task_id mapping flat SLURM task IDs to (N, d, iteration) tuples
  - total_grid_size utility for computing grid dimensions
  - Updated YAML with d=[0.3, 0.5, 0.7], K=200, 4200 total jobs
  - SLURM script with reduced MCMC flags (2 chains, 500 draws, 500 tune)
  - Entry point CLI flags for --fit-chains/--fit-draws/--fit-tune
affects: [10-core-power-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns: [grid-index-arithmetic, reduced-mcmc-for-power]

key-files:
  created:
    - src/prl_hgf/power/grid.py
    - tests/test_power_grid.py
  modified:
    - configs/prl_analysis.yaml
    - cluster/08_power_sweep.slurm
    - scripts/08_run_power_iteration.py
    - src/prl_hgf/power/__init__.py

key-decisions:
  - "Grid layout is row-major: n_per_group (outer) x effect_size (middle) x iteration (inner)"
  - "Reduced MCMC defaults baked into argparse (2 chains, 500 draws, 500 tune) per Research Pitfall 3"

patterns-established:
  - "Grid decode: integer division and modulo on flat task ID, no CSV lookup"
  - "MCMC reduction via CLI flags rather than config file override"

# Metrics
duration: 8min
completed: 2026-04-07
---

# Plan 10-02: YAML Grid Update, Grid Decode, SLURM MCMC Flags Summary

**Grid decode module mapping 4200 SLURM task IDs to (N, d, iteration) tuples with reduced MCMC CLI flags for 8h wall time compliance**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Updated YAML power section to d=[0.3, 0.5, 0.7], K=200, n_jobs=4200 (7 x 3 x 200)
- Created power/grid.py with decode_task_id() and total_grid_size() using integer division arithmetic
- Updated SLURM script with --array=0-4199%50 in comments and --fit-chains 2 --fit-draws 500 --fit-tune 500 in python invocation
- Added --fit-chains/--fit-draws/--fit-tune CLI flags to 08_run_power_iteration.py entry point
- Wrote 7 unit tests including roundtrip uniqueness check over all 4200 task IDs

## Task Commits

Each task was committed atomically:

1. **Task 1: Update YAML power section and create grid decode module** - `456bfe1` (feat)
2. **Task 2: Update SLURM script, add MCMC CLI flags, write grid tests** - `bd7bb02` (feat)

## Files Created/Modified
- `configs/prl_analysis.yaml` - Updated power section: effect_size_grid, n_iterations, n_jobs
- `src/prl_hgf/power/grid.py` - Grid decode utility with decode_task_id() and total_grid_size()
- `src/prl_hgf/power/__init__.py` - Re-exports decode_task_id and total_grid_size
- `cluster/08_power_sweep.slurm` - Full sweep array directive (0-4199%50) and MCMC CLI flags
- `scripts/08_run_power_iteration.py` - Added --fit-chains/--fit-draws/--fit-tune argparse flags
- `tests/test_power_grid.py` - 7 unit tests for grid decode including roundtrip uniqueness

## Decisions Made
- Grid layout uses row-major order (n_per_group slowest, iteration fastest) matching Research Pattern 1
- Reduced MCMC defaults (2 chains, 500 draws, 500 tune) are baked into argparse defaults per locked Research Open Question 2

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Grid decode and MCMC CLI flags ready for Plan 10-03 (full pipeline wiring)
- SLURM script ready for smoke test (default 10 jobs) or full sweep (override to 0-4199%50)

---
*Phase: 10-core-power-sweep*
*Completed: 2026-04-07*
