---
phase: 03-simulation
plan: "02"
subsystem: simulation
tags: [pyhgf, numpy, pandas, pytest, jax, batch-simulation, parameter-recovery]

# Dependency graph
requires:
  - phase: 03-01
    provides: simulate_agent, sample_participant_params, SimulationResult, PARAM_BOUNDS
  - phase: 02-01
    provides: build_3level_network, INPUT_NODES
  - phase: 01-02
    provides: generate_session, Trial, AnalysisConfig
provides:
  - simulate_batch() orchestrator in src/prl_hgf/simulation/batch.py
  - scripts/03_simulate_participants.py pipeline entry point
  - Tidy trial-level CSV with ground-truth true_* parameter columns
  - 6 unit tests in tests/test_batch.py covering shape, domains, group differences, session deltas, reproducibility, CSV output
affects:
  - phase 04 (fitting): batch CSV is direct input for single-subject MCMC
  - phase 05 (parameter recovery): true_* columns enable recovery correlation analysis

# Tech tracking
tech-stack:
  added: [pandas]
  patterns:
    - Master-seed-derived per-session seeds drawn upfront before any simulation
    - JAX JIT pre-warm call before batch loop to amortize compilation cost
    - Frozen dataclass override via constructor for small test configs
    - Flat index across groups x participants x sessions for seed array indexing

key-files:
  created:
    - src/prl_hgf/simulation/batch.py
    - scripts/03_simulate_participants.py
    - tests/test_batch.py
  modified:
    - src/prl_hgf/simulation/__init__.py

key-decisions:
  - "Seed array drawn upfront: master RNG draws all (env_seed, sim_seed) pairs before loop starts — ensures changing n_participants_per_group does not alter earlier participant seeds"
  - "JIT pre-warm in batch.py (not agent.py): keeps agent.py clean and single-responsibility; warmup is a batch-level concern"
  - "session_labels list prepends 'baseline' before session_cfg.session_labels to build ['baseline', 'post_dose', 'followup']"
  - "Same rng_sim instance used for both sample_participant_params and simulate_agent — consistent with plan spec; seeds are drawn from the upfront seed array"
  - "_small_config() constructs new frozen SimulationConfig with n_participants_per_group=2 — correct pattern for frozen dataclass override in tests"

patterns-established:
  - "Batch orchestration: derive all seeds upfront from master_seed, JIT prewarm, then iterate groups x participants x sessions"
  - "Test config override: construct new frozen dataclass with overridden field, wrap in new AnalysisConfig"
  - "Pipeline scripts: add _PROJECT_ROOT to sys.path for direct script invocation outside pytest"

# Metrics
duration: 13min
completed: 2026-04-05
---

# Phase 3 Plan 2: Batch Simulation Summary

**simulate_batch() generating 2 groups x N participants x 3 sessions into a tidy trial-level DataFrame with embedded true_* parameters, using upfront seed derivation from master_seed for reproducibility**

## Performance

- **Duration:** 13 min
- **Started:** 2026-04-05T13:57:26Z
- **Completed:** 2026-04-05T14:10:41Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `simulate_batch()` orchestrates full cohort generation with deterministic upfront seed derivation
- JAX JIT pre-warm before the main loop eliminates first-participant compilation latency
- Pipeline script `03_simulate_participants.py` handles config loading, directory creation, CSV output, and summary stats
- 6 unit tests all pass covering output shape, column domains, group parameter differences, session delta application, reproducibility, and CSV serialization
- No regressions in existing 55 tests across test_agent.py, test_models.py, test_env_simulator.py, test_response.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement simulate_batch and pipeline script** - `1badb6b` (feat)
2. **Task 2: Unit tests for batch simulation** - `6414633` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified
- `src/prl_hgf/simulation/batch.py` - simulate_batch() with seed derivation, JIT prewarm, group/participant/session loop, tidy DataFrame assembly
- `src/prl_hgf/simulation/__init__.py` - Added simulate_batch to public exports
- `scripts/03_simulate_participants.py` - Pipeline entry point with config load, output dir creation, summary stats
- `tests/test_batch.py` - 6 slow-marked unit tests for batch simulation

## Decisions Made
- **Seed array drawn upfront:** `rng_master.integers(0, 2**31, size=(n_total, 2))` before the loop starts. This ensures reproducibility is not sensitive to loop order and that adding more participants later does not alter seeds for earlier participants.
- **JIT pre-warm in batch.py:** Keeps agent.py single-responsibility (one trial, one agent). Batch-level orchestration concerns (JIT warmup, progress logging) stay in batch.py.
- **session_labels list construction:** `["baseline"] + list(session_cfg.session_labels)` builds `["baseline", "post_dose", "followup"]` from the YAML-defined non-baseline labels — avoids hardcoding the full list anywhere.
- **Same rng_sim for sampling and simulation:** The plan spec passes the same RNG to both `sample_participant_params` and `simulate_agent`. The simulation RNG advances after sampling, providing additional entropy for the trial choices.
- **_small_config() test helper:** Constructs a new `SimulationConfig(n_participants_per_group=2, ...)` and wraps it in a new `AnalysisConfig` — the only correct approach with frozen dataclasses.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Batch simulation output (tidy CSV with true_* columns) is ready for Phase 4 (fitting)
- Full-run produces 180 participant-sessions x 420 trials = 75,600 rows (~24 min runtime)
- Parameter recovery (Phase 5) can use true_* columns directly for recovery correlation analysis
- Concern: each batch test re-runs 12 participant-sessions; total test suite now ~6-7 min. Consider whether test_batch.py should be excluded from CI fast runs using `-k "not slow"`.

---
*Phase: 03-simulation*
*Completed: 2026-04-05*
