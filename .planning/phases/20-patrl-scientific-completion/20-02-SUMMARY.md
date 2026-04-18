---
phase: 20-patrl-scientific-completion
plan: 02
subsystem: fitting
tags: [jax, hgf, patrl, vb-laplace, response-models, delta-hr, blackjax]

# Dependency graph
requires:
  - phase: 20-01
    provides: pat_rl.yaml priors.b/gamma/alpha fields + loader consumer spec
  - phase: 19-03
    provides: fit_vb_laplace_patrl.py + laplace_idata.py Hessian/idata path
  - phase: 18-04
    provides: hierarchical_patrl.py batched logp factory + closure-based log-posterior
provides:
  - model_a_logp with backward-compatible bias b kwarg (default 0.0)
  - model_b_logp: P(approach) = sigma(beta*EV + b + gamma*dHR)
  - model_c_logp: P(approach) = sigma((beta + alpha*dHR)*EV + b + gamma*dHR)
  - build_logp_fn_batched_patrl accepts delta_hr_arr and response_model dispatch
  - fit_batch_hierarchical_patrl + fit_vb_laplace_patrl dispatch model_a/b/c without NotImplementedError
  - 4 new _PARAM_ORDER tuples in laplace_idata.py for B/C (2-level and 3-level)
  - scripts/12 --response-model / --all-models flags
  - 14 new tests across 4 test files
affects: [20-03, 20-04, 20-05, 20-06, 20-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Python-level response_model dispatch at factory-build time (not inside JAX trace) — factory creates typed closure variant at construction; JAX trace never branches on string
    - jax.scipy.stats.norm.logpdf for closure-based priors (no numpyro at closure-build time)
    - b always sampled regardless of response_model — avoids conditional param presence across model comparison
    - pytest.mark.slow + RUN_SMOKE_TESTS=1 gate for Laplace B/C end-to-end smoke (105s)

key-files:
  created: []
  modified:
    - src/prl_hgf/models/response_patrl.py
    - src/prl_hgf/fitting/hierarchical_patrl.py
    - src/prl_hgf/fitting/fit_vb_laplace_patrl.py
    - src/prl_hgf/fitting/laplace_idata.py
    - tests/test_models_patrl.py
    - tests/test_hierarchical_patrl.py
    - tests/test_fit_vb_laplace_patrl.py
    - tests/test_smoke_patrl_foundation.py
    - scripts/12_smoke_patrl_foundation.py

key-decisions:
  - "Model A+b bias always sampled (b in every model): consistent param presence simplifies model comparison in downstream BMS"
  - "Python-level response_model dispatch (not JAX if/elif inside trace): resolves at factory-build time; JAX trace stays clean and vmappable"
  - "jax.scipy.stats.norm.logpdf for closure priors — no numpyro at closure-build time (Decision 119 preserved)"
  - "Stochastic avoid contingency intentionally NOT in logp (Models A-C): EV=0 by avoid-is-always-safe assumption; stochastic avoid wired in config/sim only"
  - "blackjax NUTS smoke for Models B/C deferred to cluster SLURM (Plan 20-07): only Laplace path exercised locally"
  - "4 _PARAM_ORDER tuples added for B/C (2-level and 3-level): ravel_pytree flat order must match insertion order for Hessian alignment"

patterns-established:
  - "Response model dispatch: Python-level factory returns typed vmapped closure; avoids JAX string-conditional branching"
  - "b always present in param pytree regardless of response_model for intra-group BMS compatibility"
  - "Slow Laplace smoke gated by RUN_SMOKE_TESTS=1 + pytest.mark.slow; fast tests always run"

# Metrics
duration: 120min (multi-session)
completed: 2026-04-18
---

# Phase 20 Plan 02: Model A+b, B, C Response Functions Summary

**Response bias b added to Model A (backward-compat); Models B (gamma*dHR additive) and C (alpha*dHR x EV interaction) fully wired through logp, hierarchical factory, Laplace fit, and 5-agent smoke — all three pass Laplace smoke in 105s**

## Performance

- **Duration:** ~120 min (multi-session, resumed from context)
- **Started:** 2026-04-18
- **Completed:** 2026-04-18
- **Tasks:** 5/5
- **Files modified:** 9

## Accomplishments

- `model_a_logp` extended with backward-compatible `b` kwarg (default 0.0); all Phase 18/19 callers unaffected
- `model_b_logp` and `model_c_logp` implemented with correct EV direction and ΔHR sign convention (bradycardia < 0)
- `build_logp_fn_batched_patrl` accepts `delta_hr_arr` and `response_model`; Python-level dispatch creates correct vmapped closure at factory-build time
- `fit_vb_laplace_patrl` dispatches model_a/b/c; 4 new `_PARAM_ORDER_*` tuples in `laplace_idata.py`; `fit_batch_hierarchical_patrl` no longer raises NotImplementedError for B/C
- Laplace 5-agent smoke passed for both Model B and Model C (105s, 2 tests in test_fit_vb_laplace_patrl.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend model_a_logp + add model_b/c** - `b81fdf9` (feat)
2. **Task 2: Dispatch in hierarchical_patrl + thread delta_hr_arr** - `bf7a321` (feat)
3. **Task 3: Extend fit_vb_laplace_patrl for Models A+b, B, C** - `14ca759` (feat)
4. **Task 4: Unit + factory tests for Models A+b, B, C + dispatch** - `90bd26c` (test)
5. **Task 5: Smoke script flags + test_smoke_patrl_foundation dispatch** - `b68c9b7` (feat)

## Files Created/Modified

- `src/prl_hgf/models/response_patrl.py` — model_a_logp + b; new model_b_logp, model_c_logp; updated __all__
- `src/prl_hgf/fitting/hierarchical_patrl.py` — delta_hr_arr kwarg; Python-level response_model dispatch; jax.scipy.stats.norm.logpdf priors for b/gamma/alpha
- `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` — response_model dispatch; param_order 6-way dict; init_arrays adds b/gamma/alpha conditionally
- `src/prl_hgf/fitting/laplace_idata.py` — _PARAM_ORDER_2LEVEL_B/C, _PARAM_ORDER_3LEVEL_B/C added; _VALID_PARAM_ORDERS updated
- `tests/test_models_patrl.py` — TestPhase20ResponseModels class: 7 new tests (b default, b shift, B shape/finite, gamma=0 identity, C shape/finite, alpha=0 identity, alpha effect sign)
- `tests/test_hierarchical_patrl.py` — replaced NotImplementedError-for-model_b with unknown-model guard test; 3 new factory tests (delta_hr_arr, model_b, model_c)
- `tests/test_fit_vb_laplace_patrl.py` — replaced NotImplementedError-for-model_b; added model_b_smoke and model_c_smoke (@pytest.mark.slow, 5 agents, 105s)
- `tests/test_smoke_patrl_foundation.py` — flag acceptance tests; TestPhase20SmokeModelDispatch with parametrized slow test (RUN_SMOKE_TESTS=1 gate)
- `scripts/12_smoke_patrl_foundation.py` — --response-model (a/b/c), --all-models flags; _EXPECTED_VARS posterior assertion dict

## Decisions Made

**Decision: Model A+b bias always sampled (b in every model)**
b is included in the param pytree and prior for all three response models (A, B, C). This means every model's posterior includes a `b` coordinate, which simplifies downstream BMS model comparison (no missing variable conditionals).

**Decision: Python-level response_model dispatch at factory-build time**
`_make_single_logp_fn` uses Python if/elif on `response_model` to produce a typed `_call_single_*` closure that is then vmapped. The dispatch is fully resolved before any JAX trace, keeping the HLO clean and the vmap axes correct. Avoids `jax.lax.cond` on string inputs.

**Decision: jax.scipy.stats.norm.logpdf for closure priors (Decision 119 preserved)**
`_build_patrl_log_posterior` in `hierarchical_patrl.py` uses `jax.scipy.stats.norm.logpdf` directly in the closure, not numpyro distribution objects at closure-build time. Preserves the closure-based logdensity_fn pattern from Phase 18-04.

**Decision: Stochastic avoid contingency intentionally NOT in logp**
Models A, B, C all compute EV assuming avoid is safe (P(reward|avoid) = P(shock|avoid) = 0). The stochastic avoid contingency (P(reward|avoid)=0.10, P(shock|avoid)=0.10) is wired in config and simulator only. Adding stochastic avoid to the logp would require scan-body surgery — deferred per plan to 20-03 (Model D). Documented in response_patrl.py module docstring.

**Decision: blackjax NUTS smoke for Models B/C deferred to cluster SLURM**
Only Laplace path exercised locally. NUTS B/C smoke requires cluster (Plan 20-07 verification gate). Documented in test_fit_vb_laplace_patrl.py and SUMMARY.

**M6 blackjax-guard decision (documented per plan output spec):**
`grep -n "^import blackjax|^from blackjax" src/prl_hgf/fitting/hierarchical_patrl.py` returned ZERO matches — blackjax is NOT imported at module level (only lazily inside fit_batch_hierarchical_patrl at NUTS-call time). Therefore `test_build_logp_fn_batched_patrl_*` tests do NOT need `pytest.importorskip('blackjax')`. Guards added only to the full NUTS pipeline smoke tests, consistent with Phase 18-06 decision.

## Deviations from Plan

**1. [Rule 1 - Bug] test_smoke_patrl_foundation.py existed from Phase 19 (plan assumed MISSING)**
- **Found during:** Task 5
- **Issue:** Plan said "create or extend"; file already existed with 9 tests from Phase 19-01/19-05 work
- **Fix:** Read file first, then used Edit to append new tests rather than Write overwrite
- **Files modified:** tests/test_smoke_patrl_foundation.py
- **Committed in:** b68c9b7 (Task 5 commit)

**2. [Rule 3 - Blocking] conda run doesn't support multiline -c arguments**
- **Found during:** Tasks 1, 2, 3 (verification steps)
- **Issue:** Verification scripts with newlines in `conda run -n ds_env python -c "..."` failed silently
- **Fix:** Wrote verification logic to temp .py files (_verify_task1.py, _verify_task2.py, _verify_task3.py) and ran those
- **Files modified:** Temporary files (not committed)

**3. [Rule 3 - Blocking] pytest --timeout flag not available (pytest-timeout not installed)**
- **Found during:** Task 4 slow test execution
- **Issue:** `pytest --timeout=300` was rejected — plugin not installed
- **Fix:** Removed --timeout flag; used `--override-ini="addopts=" -m slow` to run slow tests

---

**Total deviations:** 3 auto-fixed (1 Rule 1 bug, 2 Rule 3 blocking)
**Impact on plan:** All auto-fixes necessary for correct execution. No scope creep.

## Issues Encountered

- `addopts = "-m 'not slow'"` in pyproject.toml deselects @pytest.mark.slow tests by default. Slow Laplace B/C smoke requires `--override-ini="addopts=" -m slow` to run. This is working as intended per Phase 18-06 convention.

## Output Spec Compliance (per plan `<output>` section)

**Final function signatures:**
- `model_a_logp(mu2, choices, reward_mag, shock_mag, beta, b=0.0) -> jnp.ndarray`
- `model_b_logp(mu2, choices, reward_mag, shock_mag, beta, b, gamma, delta_hr) -> jnp.ndarray`
- `model_c_logp(mu2, choices, reward_mag, shock_mag, beta, b, alpha, gamma, delta_hr) -> jnp.ndarray`

**Prior log-density terms added to _build_patrl_log_posterior (both hierarchical and Laplace paths):**
- `b`: always added — `jax.scipy.stats.norm.logpdf(b_i, loc=cfg_priors.b.mean, scale=cfg_priors.b.sd)`
- `gamma`: added for model_b and model_c — `jax.scipy.stats.norm.logpdf(gamma_i, ...)`
- `alpha`: added for model_c only — `jax.scipy.stats.norm.logpdf(alpha_i, ...)`

**_PARAM_ORDER tuples added to laplace_idata.py:**
```python
_PARAM_ORDER_2LEVEL_B = ("omega_2", "log_beta", "b", "gamma")
_PARAM_ORDER_2LEVEL_C = ("omega_2", "log_beta", "b", "gamma", "alpha")
_PARAM_ORDER_3LEVEL_B = ("omega_2", "log_beta", "omega_3", "kappa", "mu3_0", "b", "gamma")
_PARAM_ORDER_3LEVEL_C = ("omega_2", "log_beta", "omega_3", "kappa", "mu3_0", "b", "gamma", "alpha")
```
All added to `_VALID_PARAM_ORDERS`.

**Tests added/updated:**
- `tests/test_models_patrl.py`: 7 new tests in TestPhase20ResponseModels
- `tests/test_hierarchical_patrl.py`: 1 updated + 3 new (4 total changes)
- `tests/test_fit_vb_laplace_patrl.py`: 1 updated + 2 new slow smoke tests; B+C PASSED (105s)
- `tests/test_smoke_patrl_foundation.py`: 3 new (flag acceptance + TestPhase20SmokeModelDispatch parametrized slow)

**blackjax NUTS for B/C deferred:** Cluster-only — to be captured in Plan 20-07 verification gate (cluster/SLURM smoke). Documented in test_fit_vb_laplace_patrl.py comments and in this SUMMARY.

**M6 blackjax module-level import grep:** ZERO matches in hierarchical_patrl.py — `pytest.importorskip('blackjax')` NOT needed on factory-level tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 20-03 (Model D: trial-varying omega scan body surgery) is the natural next step
- All Decisions 114-121 preserved; parallel-stack invariant maintained
- blackjax NUTS for B/C awaits cluster smoke (Plan 20-07)
- Stochastic avoid logp wiring still deferred (will be addressed in 20-03 scan body refactor or 20-07 verification)

---
*Phase: 20-patrl-scientific-completion*
*Completed: 2026-04-18*
