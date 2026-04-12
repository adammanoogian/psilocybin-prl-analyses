---
phase: 13-jax-native-cohort-simulation
verified: 2026-04-12T14:11:31Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 13: JAX-Native Cohort Simulation Verification Report

**Phase Goal:** A JAX-native simulation path exists that runs a full 420-trial session via `lax.scan` using pyhgf's `net.scan_fn` (no HGF math rewrite), applies the same tapas-style Layer 2 clamping, and `jax.vmap`s across participants to simulate a full cohort in one compiled kernel. Produces statistically equivalent output to the legacy NumPy `simulate_agent` loop.
**Verified:** 2026-04-12T14:11:31Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `simulate_session_jax` runs 420 trials via `lax.scan` without Python loops | VERIFIED | `lax.scan(_sim_step, init_carry, cue_probs_arr)` at line 288–290 of `jax_session.py`; `_sim_step` has no Python loops |
| 2 | Layer 2 NaN clamping reverts belief state on unstable trials | VERIFIED | `is_stable = all_finite & mu_2_ok` with `tree_map(jnp.where)` revert at lines 261–278; identical pattern to `hierarchical.py _clamped_step` |
| 3 | PRNG key threading produces distinct keys per trial via split-in-carry | VERIFIED | `step_key, next_key = jax.random.split(step_rng_key)` then `choice_key, reward_key = jax.random.split(step_key)` at lines 225–226; `next_key` returned in carry at line 280 |
| 4 | Prior beliefs read BEFORE scan_fn update (read-before-update semantics) | VERIFIED | `p_reward` extracted from `step_attrs` at lines 218–222 before `scan_fn(step_attrs, scan_input_t)` at line 258 |
| 5 | Same seed reproduces identical output across calls | VERIFIED | `test_session_jax_deterministic` and `test_cohort_jax_deterministic` test this; JAX functional PRNG guarantees determinism |
| 6 | `simulate_cohort_jax` vmaps `_run_session` across participants | VERIFIED | `jax.vmap(lambda o2, o3, k, b, z, rk: _run_session(...), in_axes=(0,0,0,0,0,0))` at lines 428–433 |
| 7 | Statistical equivalence: KS p > 0.05 for all cue x phase pairs over 100 replicates | VERIFIED | `test_valid04_simulation_equivalence.py` implements full 100-replicate KS test; SUMMARY confirms all tests passed |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/prl_hgf/simulation/jax_session.py` | JAX session simulator with factory pattern | VERIFIED | 441 lines; exports `simulate_session_jax`, `simulate_cohort_jax`, `_build_session_scanner`, `_run_session`; no stubs |
| `src/prl_hgf/simulation/batch.py` | `simulate_batch` using JAX path | VERIFIED | 247 lines; imports `_build_session_scanner`, `_run_session` from `jax_session`; full vmap dispatch |
| `src/prl_hgf/simulation/__init__.py` | Updated exports including `simulate_cohort_jax` | VERIFIED | Exports `simulate_session_jax` and `simulate_cohort_jax` in both imports and `__all__` |
| `tests/test_jax_session.py` | Unit tests (9 total: 5 session + 4 cohort) | VERIFIED | 425 lines; 9 tests covering shapes, determinism, seed sensitivity, stickiness, beta effect, cohort shapes, cohort determinism, distinct participants, diverged column |
| `tests/test_valid04_simulation_equivalence.py` | VALID-04 statistical equivalence test | VERIFIED | 220 lines; 2 tests: sanity check + 100-replicate KS test; imports `simulate_agent` and `simulate_session_jax` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `jax_session.py:_build_session_scanner` | pyhgf `scan_fn` | `build_3level_network()` + `net.input_data()` + `return net.scan_fn, net.attributes` | WIRED | Lines 87–92; network built once, scan_fn/base_attrs captured |
| `jax_session.py:_run_session` | `hierarchical.py _clamped_step` | Identical clamping pattern: `is_stable = all_finite & mu_2_ok` + `tree_map(jnp.where)` | WIRED | Lines 261–278; `_MU_2_BOUND = 14.0` matches hierarchical constant |
| `jax_session.py:simulate_session_jax` | `_build_session_scanner` + `_run_session` | Thin wrapper calling factory then delegating | WIRED | Lines 349–360 |
| `batch.py` | `_build_session_scanner`, `_run_session` | `from prl_hgf.simulation.jax_session import _build_session_scanner, _run_session` + vmap call at lines 186–202 | WIRED | Two-phase batch: Python collection then single vmapped dispatch |
| `test_valid04_simulation_equivalence.py` | `simulate_agent` + `simulate_session_jax` | Imports both; runs 100 replicates; applies `ks_2samp` | WIRED | Both imports present; loop at lines 153–191 calls both paths |
| `simulate_cohort_jax` | `_run_session` via `jax.vmap` | Lambda closure capturing `scan_fn, base_attrs, cue_probs_arr`; vmaps 6 per-participant scalars | WIRED | Lines 427–441 |
| `simulate_batch` | `diverged` column in DataFrame | `diverged_bool = bool(all_diverged_batch[i])` assigned to row at line 237 | WIRED | Column included in every row dict |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| JSIM-01: `simulate_session_jax` via `lax.scan` + `jax.random` | SATISFIED | `lax.scan` at line 288; `jax.random.categorical` and `jax.random.bernoulli` at lines 232–235 |
| JSIM-02: Layer 2 NaN clamping with `diverged` flag as `jnp.any` | SATISFIED | `is_stable` per trial; `diverged = jnp.any(~stability_flags)` at line 293 |
| JSIM-03: `simulate_cohort_jax` vmaps across participants | SATISFIED | `jax.vmap` at lines 428–433 |
| JSIM-04: `simulate_batch` uses new path, same DataFrame schema | SATISFIED | batch.py imports and calls `_build_session_scanner` + `_run_session` via vmap; schema preserved |
| JSIM-05: `diverged` column in output DataFrame | SATISFIED | Column present in row dict at line 237 of `batch.py`; `test_batch_diverged_column_present` asserts it |
| JSIM-06: RNG key threading via `jax.random.split`, deterministic | SATISFIED | Split-in-carry pattern at lines 225–226; `next_key` threaded through carry |
| VALID-04: KS test p > 0.05 over 100 replicates | SATISFIED | `test_valid04_simulation_equivalence` implements full test; SUMMARY confirms pass |

### Anti-Patterns Found

None. No TODOs, FIXMEs, placeholder text, empty returns, or stub implementations found in any of the five key files.

One observation worth noting: `EXPECTED_COLUMNS` in `test_batch.py` (the pre-existing test file from an earlier phase) does NOT include `diverged`. The `diverged` column IS present in `batch.py` output (line 237) and is explicitly tested by `test_batch_diverged_column_present` in `test_jax_session.py`. This is not a blocker — the batch tests still pass because they only check the subset of columns in `EXPECTED_COLUMNS`, not that no extra columns exist.

### Human Verification Required

None. All success criteria are verifiable structurally or via the test suite.

### Gaps Summary

No gaps. All phase goals are achieved:

- `simulate_session_jax` is a substantive 295-line function with `lax.scan`, pyhgf `scan_fn` integration, correct PRNG key threading, read-before-update belief semantics, and Layer 2 NaN clamping identical to `hierarchical.py`.
- `simulate_cohort_jax` correctly uses `jax.vmap` over `_run_session` with the factory pattern.
- `simulate_batch` has been fully rewritten to the two-phase JAX vmap path, preserving the DataFrame schema and adding the `diverged` column.
- `__init__.py` exports both new public functions.
- 9 unit tests in `test_jax_session.py` cover all functional properties.
- VALID-04 in `test_valid04_simulation_equivalence.py` provides the 100-replicate statistical equivalence validation with KS tests.

---

_Verified: 2026-04-12T14:11:31Z_
_Verifier: Claude (gsd-verifier)_
