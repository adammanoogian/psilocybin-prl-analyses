---
phase: 13-jax-native-cohort-simulation
plan: 03
subsystem: testing
tags: [jax, numpy, hgf, pyhgf, simulation, statistical-equivalence, ks-test, scipy]

# Dependency graph
requires:
  - phase: 13-01
    provides: simulate_session_jax (JAX lax.scan path) and simulate_agent (NumPy loop path)
provides:
  - VALID-04: 100-replicate KS-based statistical equivalence test between legacy and JAX simulation paths
  - Confirmed both paths produce statistically indistinguishable choice frequency distributions per cue per phase
affects: [13-02, 14, 15, group-analysis, simulation-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Controlled equivalence test design: single fixed trial sequence isolates agent RNG from environment RNG"
    - "KS test on per-cue per-phase choice frequency distributions over 100 replicates"

key-files:
  created:
    - tests/test_valid04_simulation_equivalence.py
  modified: []

key-decisions:
  - "Single fixed trial sequence for all 100 replicates isolates agent RNG difference from environment variability"
  - "KS test threshold p > 0.05 is appropriate: NumPy PCG64 and JAX ThreeFry are independent RNG streams so non-matching per-trial sequences are expected; only aggregate equivalence is testable"

patterns-established:
  - "VALID-04 pattern: aggregate statistical equivalence via KS test is the correct method when two simulation paths use incompatible RNG backends"

# Metrics
duration: 34min
completed: 2026-04-12
---

# Phase 13 Plan 03: VALID-04 Simulation Equivalence Summary

**VALID-04 confirms simulate_agent (NumPy PCG64) and simulate_session_jax (JAX ThreeFry) are statistically equivalent: KS p > 0.05 for all 6 (phase x cue) pairs over 100 replicates**

## Performance

- **Duration:** 34 min (dominated by 100-replicate test runtime: 33:56 for 200 sessions)
- **Started:** 2026-04-12T13:30:35Z
- **Completed:** 2026-04-12T14:04:31Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments

- Created VALID-04 with two tests: `test_valid04_choice_range` (sanity check, 1 replicate) and `test_valid04_simulation_equivalence` (100-replicate KS test)
- Verified KS p > 0.05 for all 6 (phase_label x cue) pairs — no distributional bias in JAX path
- Confirmed controlled-design rationale: holding trial sequence constant across replicates isolates agent RNG as the only source of variation
- Ruff linting: all checks passed

## Task Commits

Each task was committed atomically:

1. **Task 1: VALID-04 simulation equivalence test** - `eeaf76c` (test)

## Files Created/Modified

- `tests/test_valid04_simulation_equivalence.py` - VALID-04 statistical equivalence validation (2 tests, 220 lines)

## Decisions Made

- Single fixed trial sequence for all 100 replicates is the correct controlled-comparison design: it isolates agent RNG (choice and reward sampling) from environment RNG (trial sequence generation). Varying environment seeds would conflate environment variability with implementation differences.
- KS test threshold p > 0.05 rather than exact match: the two paths use incompatible RNG backends (NumPy PCG64 vs JAX ThreeFry), so per-trial exact match is impossible by design. Aggregate distributional equivalence is the only scientifically valid comparison.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Both tests passed on first run. Runtime was ~34 minutes total (27s for sanity check + 33:56 for 100-replicate main test).

## Next Phase Readiness

- JAX simulation path (Phase 13-01) is now scientifically validated against the proven NumPy legacy path
- Both paths confirmed to produce statistically equivalent choice distributions per cue per phase
- Phase 13-02 (simulate_cohort_jax vmap path) can proceed: the per-session JAX kernel is validated
- No blockers

---
*Phase: 13-jax-native-cohort-simulation*
*Completed: 2026-04-12*
