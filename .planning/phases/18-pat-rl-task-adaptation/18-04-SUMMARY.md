---
phase: 18-pat-rl-task-adaptation
plan: 04
subsystem: fitting
tags: [pyhgf, blackjax, jax, vmap, lax-scan, hierarchical-mcmc, patrl, model-a]

# Dependency graph
requires:
  - phase: 18-02
    provides: generate_session_patrl, PATRLConfig, PATRLTrial (PAT-RL trial sequence generator)
  - phase: 18-03
    provides: build_2level_network_patrl, build_3level_network_patrl, model_a_logp, expected_value (HGF builders + Model A response)
  - phase: 17-01
    provides: _run_blackjax_nuts, _samples_to_idata, _extract_nuts_stats (generic BlackJAX helpers in hierarchical.py)
provides:
  - build_logp_fn_batched_patrl — vmapped JAX logp factory for PAT-RL Model A
  - fit_batch_hierarchical_patrl — BlackJAX NUTS orchestrator for PAT-RL cohorts
  - _build_arrays_single_patrl — sim_df to (P, n_trials) array converter
  - _build_session_scanner_patrl — pyhgf Network factory (scan_fn capture)
  - 8 test cases covering logp shape, grad, reference match, missing columns, NotImplementedError, regression
affects:
  - 18-05 (trajectory export — consumes fit_batch_hierarchical_patrl InferenceData)
  - 18-06 (end-to-end validation smoke — drives fit_batch_hierarchical_patrl)
  - Phase 19 (Models B/C/D — extend build_logp_fn_batched_patrl signature)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Factory pattern: _build_session_scanner_patrl builds pyhgf Network ONCE outside vmap/jit; scan_fn captured in closure"
    - "Parallel-stack invariant: PAT-RL fitting imports from hierarchical.py; does NOT modify it"
    - "Tapas Layer-2 clamping implemented inline (_MU_2_BOUND=14.0) to avoid coupling the parallel stacks"
    - "importorskip guard on smoke tests: blackjax-dependent tests skip cleanly when blackjax absent"
    - "float64 dtype for scan inputs: matches pyhgf attribute dtype so jax.lax.cond branches are consistent"

key-files:
  created:
    - src/prl_hgf/fitting/hierarchical_patrl.py
    - tests/test_hierarchical_patrl.py
  modified: []

key-decisions:
  - "float64 for scan inputs (state, time_steps): pyhgf primes attributes with float64 numpy arrays; jax.lax.cond inside continuous_node_posterior_update requires dtype consistency between branches"
  - "Closure-based logdensity_fn (not traced-arg sample loop): _build_sample_loop is pick_best_cue-specific and incompatible with PAT-RL param dict; PAT-RL uses legacy NUTS path (warmup+vmap chains). Traced-arg extension deferred to Phase 19"
  - "log_beta parameterisation: beta sampled in log-space for NUTS; prior on log_beta uses delta-method approximation N(log(beta_mean), beta_sd/beta_mean)"
  - "kappa injected via attrs[2]['volatility_coupling_children'] = jnp.asarray([kappa_i]): confirmed at runtime that kappa coupling strength lives in attributes dict (not just edges), enabling dynamic injection per participant"
  - "importorskip not @pytest.mark.skip: allows cluster operator to validate smoke tests simply by installing blackjax without touching test file"

patterns-established:
  - "PAT-RL parallel stack: src/prl_hgf/fitting/hierarchical_patrl.py imports from but never modifies hierarchical.py"
  - "scan input format for single-channel pyhgf: ((val_i,), (obs_i,), ts_i, None) with 1-tuples for values and observed"
  - "(P, n_trials) array contract: state, choice, reward_mag, shock_mag, delta_hr, trial_mask — delta_hr for CSV export only in Phase 18"

# Metrics
duration: 45min
completed: 2026-04-18
---

# Phase 18 Plan 04: PAT-RL Batched Logp Factory + BlackJAX Orchestrator Summary

**Pure-JAX vmapped logp factory + BlackJAX NUTS orchestrator for PAT-RL Model A, reusing hierarchical.py helpers without modification.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-04-18T00:00:00Z
- **Completed:** 2026-04-18T00:45:00Z
- **Tasks:** 3/3
- **Files modified:** 2 created

## Accomplishments

- Created `hierarchical_patrl.py` with 4 public exports: `build_logp_fn_batched_patrl`, `fit_batch_hierarchical_patrl`, `_build_arrays_single_patrl`, `_build_session_scanner_patrl`
- Reused `_run_blackjax_nuts` and `_samples_to_idata` from `hierarchical.py` without any modification (parallel-stack invariant confirmed via `git diff`)
- 8 test cases: 6 pass without blackjax, 2 smoke tests skip cleanly via `importorskip`; all 83 pick_best_cue regression tests pass (1 pre-existing blackjax failure unchanged)

## Task Commits

1. **Task 1: batched logp factory + array builder** - `f775cb6` (feat)
2. **Task 2+3: orchestrator + tests** - `266dc66` (feat)

## Files Created/Modified

- `src/prl_hgf/fitting/hierarchical_patrl.py` — PAT-RL fitting module (parallel stack)
- `tests/test_hierarchical_patrl.py` — 8 test cases (6 unit + 2 importorskip'd smoke)

## Decisions Made

**pyhgf dtype contract (float64 scan inputs):** pyhgf Network attributes are initialised with float64 NumPy arrays. When `jax.lax.scan` runs inside `jax.vmap`, the `jax.lax.cond` call inside `continuous_node_posterior_update` requires both branches to have identical dtypes. Using float32 for scan inputs caused `DIFFERENT ShapedArray(float64[]) vs. ShapedArray(float32[])` errors when P > 1. Fixed by using `jnp.float64` for values and time_steps throughout `_single_logp`.

**Closure-based logdensity_fn (not traced-arg sample loop):** The pick_best_cue `_build_sample_loop` factory hardcodes the pick_best_cue logp signature `(omega_2, beta, zeta, input_data, observed, choices, trial_mask)`. Re-using it for PAT-RL would require modifying hierarchical.py (violating the parallel-stack invariant) or duplicating large amounts of code. Instead, PAT-RL uses the legacy closure-based warmup+vmap chain path in `_run_blackjax_nuts` (passed `batched_logp_fn=None`). The XLA traced-arg cache optimisation is deferred to Phase 19+.

**kappa injection:** kappa coupling strength is stored in `attrs[VOLATILITY_NODE]["volatility_coupling_children"]` as a 1-element JAX array. This allows dynamic per-participant kappa injection inside `lax.scan` by patching the attributes dict, enabling principled 3-level fitting without freezing kappa at build time.

**log_beta parameterisation:** beta is sampled in log-space so NUTS can freely explore without hitting the positivity boundary. Prior: `log_beta ~ N(log(beta_mean), beta_sd/beta_mean)` (delta-method approximation centred near prior mean).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] float32/float64 dtype mismatch in pyhgf cond**

- **Found during:** Task 3 (test_build_logp_shape with P=3, T=192 and vmap)
- **Issue:** Plan sketch used `jnp.float32` for values and time_steps. pyhgf primes Network attrs with float64 numpy arrays. When `jax.vmap` traces `_clamped_step`, `jax.lax.cond` inside `continuous_node_posterior_update` demands dtype consistency; float32 inputs against float64 attrs caused `TypeError: true_fun and false_fun output must have identical types`
- **Fix:** Changed values and time_steps to `jnp.float64` throughout `_make_single_logp_fn`
- **Files modified:** `src/prl_hgf/fitting/hierarchical_patrl.py`
- **Commit:** `266dc66`

**2. [Rule 3 - Blocking] _build_sample_loop incompatible with PAT-RL logp signature**

- **Found during:** Task 2 design (read hierarchical.py `_build_sample_loop`)
- **Issue:** `_build_sample_loop` hardcodes pick_best_cue 7-arg logp signature; cannot be called with PAT-RL 2/5-arg params dict without modifying hierarchical.py
- **Fix:** Passed `batched_logp_fn=None` to `_run_blackjax_nuts` to use the legacy closure-based chain path. This is correct and valid — the traced-arg optimisation is not required for Phase 18 smoke runs
- **Files modified:** `src/prl_hgf/fitting/hierarchical_patrl.py`
- **Commit:** `266dc66`

## Authentication Gates

None.

## BlackJAX Gap

blackjax is not installed in `ds_env`. The 5-participant smoke tests (tests 5 and 6) are skipped via `pytest.importorskip("blackjax")`.

**Action required for cluster operator:**

```bash
# On M3 cluster (ds_env or equivalent):
pip install blackjax
pytest tests/test_hierarchical_patrl.py -v  # smoke tests will run
```

Expected smoke test budget: `n_chains=2, n_tune=200, n_draws=200` per model variant, total <5 min on CPU.

## Next Phase Readiness

- **18-05** (trajectory export): `fit_batch_hierarchical_patrl` returns `az.InferenceData` with `participant` coord — ready for trajectory extraction
- **18-06** (end-to-end validation): `fit_batch_hierarchical_patrl` callable from scripts; `_build_arrays_single_patrl` handles DataFrame → arrays conversion
- **Phase 19** (Models B/C/D): extend `build_logp_fn_batched_patrl` signature to include `delta_hr` (already assembled in `_build_arrays_single_patrl`); add `_build_sample_loop_patrl` with traced-arg optimisation
