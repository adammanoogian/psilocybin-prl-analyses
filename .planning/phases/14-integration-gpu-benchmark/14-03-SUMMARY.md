---
phase: 14-integration-gpu-benchmark
plan: "03"
subsystem: validation
tags: [jax, cross-platform, cpu-gpu, posterior-consistency, mcmc, arviz, argparse]

# Dependency graph
requires:
  - phase: 12-batched-hierarchical-jax-logp
    provides: fit_batch_hierarchical returning az.InferenceData with participant dimension
  - phase: 13-jax-native-cohort-simulation
    provides: simulate_batch with JAX vmap internals
  - phase: 14-integration-gpu-benchmark/14-01
    provides: dual-path run_sbf_iteration (already created valid03_cross_platform.py stub)

provides:
  - VALID-03 cross-platform consistency validation script
  - compare_results function with 1% relative error threshold
  - Unit tests for comparison logic covering 4 edge cases

affects:
  - cluster GPU benchmark execution (BENCH-01..05)
  - phase 14 decision gate documentation

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-invocation cross-platform design: JAX platform cannot change within process; separate runs + JSON comparison"
    - "Relative error formula: abs(mean_a - mean_b) / (abs(mean_a) + 1e-8) handles near-zero values"
    - "Deferred heavy imports (jax, prl_hgf) inside functions for fast --help and compare subcommand"

key-files:
  created:
    - validation/valid03_cross_platform.py
    - tests/test_valid03.py
  modified: []

key-decisions:
  - "VALID-03 is two separate script invocations + comparison step (not a single pytest) because JAX platform is a one-time global setting per process"
  - "compare_results uses abs(mean_a) + 1e-8 denominator to prevent near-zero division false failures"
  - "All 4 unit tests use synthetic JSON data only; no MCMC in the test suite"

patterns-established:
  - "Cross-platform validation pattern: run_fit_and_save + compare_results with JSON intermediates"
  - "Validation scripts live in validation/ and are standalone (sys.path insertion for project root)"

# Metrics
duration: 8min
completed: 2026-04-12
---

# Phase 14 Plan 03: Cross-Platform Posterior Consistency (VALID-03) Summary

**Two-invocation cross-platform consistency check: fit hgf_3level on CPU and GPU separately, save posterior means to JSON, compare all parameters within 1% relative error using abs(mean_a - mean_b) / (abs(mean_a) + 1e-8)**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-12T15:59:37Z
- **Completed:** 2026-04-12T16:07:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `validation/valid03_cross_platform.py` with `run` (fit + save JSON) and `compare` (assert 1% rtol) subcommands
- 4 unit tests in `tests/test_valid03.py` verify comparison logic: identical pass, within-tolerance pass, exceeds-tolerance fail, near-zero values handled safely
- Script structured for cluster GPU execution without code changes: set `JAX_PLATFORM_NAME=cpu` on CPU node, run normally on GPU node, compare with `compare` subcommand

## Task Commits

Each task was committed atomically:

1. **Task 1: Create validation/valid03_cross_platform.py** - `bcd8e02` (feat — created in prior 14-01 session; new write was identical, no new commit needed)
2. **Task 2: Create tests/test_valid03.py** - `951c45d` (test)

## Files Created/Modified

- `validation/valid03_cross_platform.py` - VALID-03 script: `run_fit_and_save` calls `fit_batch_hierarchical`, saves posterior means per-participant per-parameter to JSON; `compare_results` enforces 1% rtol with safe near-zero denominator; CLI `run`/`compare` subcommands
- `tests/test_valid03.py` - Unit tests for `compare_results`: identical, within-tolerance, exceeds-tolerance, near-zero cases; no MCMC

## Decisions Made

- The two-invocation design (separate processes + JSON intermediates) is the only feasible approach because `JAX_PLATFORM_NAME` is set at import time and cannot be changed within a running process.
- Denominator `abs(mean_a) + 1e-8` prevents false failures when parameter means are near zero (e.g., zeta ~ 0.001).
- Heavy imports (`jax`, `prl_hgf`) are deferred inside functions so `--help` and `compare` subcommand are lightweight.

## Deviations from Plan

None - plan executed exactly as written.

**Note:** `validation/valid03_cross_platform.py` was found to already exist from plan 14-01 (same session, plan 14-01 scope had expanded to include it). The file written in this plan execution was byte-for-byte identical to what was already committed; no additional commit was generated for Task 1.

## Issues Encountered

- `ruff` flagged an unused `pytest` import and an import ordering issue (I001) in `tests/test_valid03.py`. Fixed via `ruff --fix` and manual removal of the unused import. Tests still pass 4/4 after fix.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VALID-03 script is ready for cluster deployment: copy `validation/valid03_cross_platform.py` to cluster, run once with `JAX_PLATFORM_NAME=cpu`, once on GPU node, compare JSON outputs
- Plans 14-01 (batched iteration wiring) and 14-03 (VALID-03) are complete
- Remaining: plan 14-02 (GPU benchmark + decision gate in `08_run_power_iteration.py`) if not yet done

---
*Phase: 14-integration-gpu-benchmark*
*Completed: 2026-04-12*
