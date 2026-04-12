---
phase: 13-jax-native-cohort-simulation
plan: 01
subsystem: simulation
tags: [jax, lax.scan, pyhgf, hgf, vmap, prng, xla]

# Dependency graph
requires:
  - phase: 12-batched-hierarchical-jax-logp
    provides: "Layer 2 NaN clamping pattern (_clamped_step), parameter injection pattern, scan_fn factory pattern from build_logp_ops_batched"
  - phase: 03-hgf-models
    provides: "build_3level_network, INPUT_NODES, BELIEF_NODES constants"
  - phase: 02-task-environment
    provides: "generate_session, Trial dataclass, cue_probs per-trial"
provides:
  - "simulate_session_jax: JAX-native single-session simulator via lax.scan"
  - "_build_session_scanner: factory that builds pyhgf Network once outside JAX trace"
  - "_run_session: pure-JAX vmappable kernel for cohort-level vmap in Plan 02"
  - "5 unit tests covering shapes, determinism, seed sensitivity, stickiness sentinel, beta effect"
affects: [13-02, 13-03, validation/test_jax_session_valid.py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Factory pattern for pyhgf scan_fn: build network once outside vmap, pass scan_fn + base_attrs to pure-JAX function"
    - "PRNG key threading in lax.scan carry: (attrs, rng_key, prev_choice) with split-in-carry"
    - "Read-before-update semantics: beliefs from carry attrs BEFORE scan_fn call"
    - "Layer 2 NaN clamping in simulation: identical jnp.where + tree_map pattern from hierarchical.py"

key-files:
  created:
    - src/prl_hgf/simulation/jax_session.py
    - tests/test_jax_session.py
  modified: []

key-decisions:
  - "Factory pattern (_build_session_scanner + _run_session) prevents network rebuild per vmap call"
  - "prev_choice = jnp.int32(-1) sentinel for zero stickiness on trial 0 (all-False comparison with arange(3))"
  - "values_t elements use .reshape(1) to match pyhgf scan_fn shape contract (1D not scalar)"
  - "simulate_session_jax is thin wrapper for convenience; vmap users call _run_session directly"

patterns-established:
  - "Simulation scan carry: (attrs, rng_key, prev_choice) for PRNG key threading"
  - "Per-trial scan input construction inside step: (values_t, observed_t, time_step_t, None)"

# Metrics
duration: 8min
completed: 2026-04-12
---

# Phase 13 Plan 01: JAX-Native Session Simulator Summary

**lax.scan-based HGF session simulator with PRNG carry, Layer 2 NaN clamping, and factory pattern enabling Plan 02 vmap cohort path**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-12T13:17:25Z
- **Completed:** 2026-04-12T13:25:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `simulate_session_jax` runs 420 trials via `lax.scan` without Python loops, returning `(choices, rewards, diverged)` with shapes `(420,)`, `(420,)`, `()`
- Factory pattern separates network construction (`_build_session_scanner`) from the pure-JAX kernel (`_run_session`), making the kernel vmappable for Plan 02's cohort path
- Layer 2 NaN clamping (identical to `hierarchical.py _clamped_step`) reverts belief state on unstable trials via `jnp.where + tree_map`
- 5 unit tests pass confirming shapes, determinism, seed sensitivity, stickiness sentinel safety, and beta effect on exploitation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create simulate_session_jax with lax.scan + clamping** - `8fccab5` (feat)
2. **Task 2: Unit tests for simulate_session_jax** - `7622e8e` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/prl_hgf/simulation/jax_session.py` - JAX-native session simulator: `_build_session_scanner`, `_run_session`, `simulate_session_jax`
- `tests/test_jax_session.py` - 5 unit tests marked `@pytest.mark.slow` (XLA JIT compile)

## Decisions Made

- **Factory pattern for vmappability:** `_build_session_scanner` builds pyhgf `Network()` once outside JAX trace; `_run_session` is pure-JAX and vmappable. Vmapping `simulate_session_jax` directly would rebuild the network per call.
- **`jnp.int32(-1)` sentinel for trial 0:** `(prev_choice == jnp.arange(3))` evaluates all-False for -1, giving zero stickiness without special-casing.
- **`values_t` elements use `.reshape(1)`:** Matches pyhgf `scan_fn` shape contract — the logp path passes `input_data[:, 0:1]` (shape `(n_trials, 1)`), which `lax.scan` slices to `(1,)` per step. Simulation must match.
- **`simulate_session_jax` is a thin wrapper:** Convenient for one-off calls; Plan 02 vmap users should call `_build_session_scanner` once then vmap `_run_session`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `_run_session` is ready to be vmapped in Plan 02 (`simulate_cohort_jax`): call `_build_session_scanner()` once, then `jax.vmap(_run_session, in_axes=(None, None, 0, 0, 0, 0, 0, None, 0))(scan_fn, base_attrs, *params_batch, cue_probs_arr, rng_keys_batch)`
- All must-haves confirmed: 420-trial lax.scan, NaN clamping, PRNG key threading, read-before-update semantics, determinism

---
*Phase: 13-jax-native-cohort-simulation*
*Completed: 2026-04-12*
