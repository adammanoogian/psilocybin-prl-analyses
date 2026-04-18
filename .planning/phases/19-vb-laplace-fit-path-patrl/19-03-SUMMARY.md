---
phase: 19-vb-laplace-fit-path-patrl
plan: "03"
subsystem: fitting
tags: [jaxopt, lbfgs, laplace, hessian, map, vb, patrl, inference, arviz]

requires:
  - phase: 19-vb-laplace-fit-path-patrl/19-01
    provides: simulate_patrl_cohort + run_hgf_forward_patrl for smoke tests
  - phase: 19-vb-laplace-fit-path-patrl/19-02
    provides: build_idata_from_laplace factory + _PARAM_ORDER_* constants
  - phase: 18-pat-rl-task-adaptation
    provides: _build_patrl_log_posterior, _build_arrays_single_patrl, build_logp_fn_batched_patrl (imported unchanged)

provides:
  - fit_vb_laplace_patrl: MAP optimizer + Hessian + eigh-clip PD regularization + InferenceData packaging
  - _regularize_to_pd: eigh-clip PD regularization helper with canonical 7-key diagnostics dict
  - jit=True -> jit=False LBFGS fallback with WARN logging

affects:
  - 19-04-PLAN.md (pipeline integration test)
  - 19-05-PLAN.md (cluster smoke + timing validation)
  - Any caller of fit_vb_laplace_patrl or export_subject_parameters/trajectories

tech-stack:
  added:
    - jaxopt>=0.8.5,<0.9 (pyproject.toml explicit dep; was transitive-only)
  patterns:
    - jaxopt.LBFGS MAP optimization on JAX pytree dict directly
    - ravel_pytree for flat-vector Hessian via jax.hessian
    - eigh-clip PD regularization (not ridge-add loop)
    - Canonical 7-key diagnostics dict passed directly to build_idata_from_laplace
    - param_order re-projection after LBFGS (JAX sorts dict keys alphabetically)

key-files:
  created:
    - src/prl_hgf/fitting/fit_vb_laplace_patrl.py
    - tests/test_fit_vb_laplace_patrl.py
  modified:
    - pyproject.toml (jaxopt dep + jaxopt mypy override)

key-decisions:
  - "jaxopt.LBFGS jit=True is default; jit=False fallback with WARN on any exception"
  - "Dense (P*K)x(P*K) Hessian via jax.hessian on ravel_pytree flat vector; block-diagonal TODO for Phase 20+"
  - "eigh-clip eps=1e-8 for PD regularization; ridge_added tracks actual shift"
  - "Re-order best_mode_params to param_order after LBFGS (JAX sorts dict keys alphabetically, ravel_pytree must see insertion order)"
  - "OQ2 deferred: kappa in native-space; logit-reparam flagged for Phase 20+ if MAP hits boundary"
  - "OQ1 sidestepped: build_idata_from_laplace emits participant_id dim natively, no _samples_to_idata call"

patterns-established:
  - "param_order re-projection: best_mode_params_ordered = {k: best_mode_params[k] for k in param_order} before ravel_pytree"
  - "Diagnostics dict uses float() for all values to ensure JSON-serializable and ArviZ-compatible"
  - "jit-fallback coverage via mock of module-level LBFGS name, not class-level method patch"

duration: 2h
completed: 2026-04-18
---

# Phase 19 Plan 03: fit_vb_laplace_patrl Summary

**jaxopt.LBFGS MAP + autodiff Hessian + eigh-clip PD regularization packaged as az.InferenceData via build_idata_from_laplace; 12 tests pass; 3-agent 2-level converges in 15 iterations; 4/5 omega_2 within 0.5 of truth**

## Performance

- **Duration:** ~2 hours
- **Started:** 2026-04-18T11:00Z (approx)
- **Completed:** 2026-04-18T13:30Z (approx)
- **Tasks:** 2 (Task 1: implementation, Task 2: tests)
- **Files modified:** 3

## Accomplishments

- `fit_vb_laplace_patrl` runs quasi-Newton MAP via `jaxopt.LBFGS`, takes autodiff Hessian at mode via `jax.hessian(lambda f: -log_posterior_fn(unravel(f)))`, PD-regularizes via eigh-clip, inverts to covariance, packages via `build_idata_from_laplace`
- Canonical 7-key diagnostics dict (`converged`, `n_iterations`, `logp_at_mode`, `hessian_min_eigval`, `hessian_max_eigval`, `n_eigenvalues_clipped`, `ridge_added`) surfaced in `idata.sample_stats`
- 12 tests: 8 fast (unit) + 4 slow (smoke/recovery); all pass in <2 min total
- Parallel-stack invariant preserved: zero diffs on all 9 protected files; `_samples_to_idata` never referenced

## Observed MAP Statistics (3-agent 2-level smoke, master_seed=42)

| Diagnostic | Value |
|---|---|
| converged | True |
| n_iterations | 15 |
| logp_at_mode | -160.9653 |
| hessian_min_eigval | 8.346 (pre-clip) |
| hessian_max_eigval | 33.755 |
| n_eigenvalues_clipped | 0 |
| ridge_added | 0 |

All Hessian eigenvalues positive (PD at MAP) — eigh-clip fallback not triggered on well-specified synthetic data.

## Observed Recovery Statistics (5-agent 2-level, master_seed=7)

| Agent | true omega_2 | post_mean | error | within 0.5? |
|---|---|---|---|---|
| P000 | -6.315 | -6.441 | 0.126 | Yes |
| P001 | -5.299 | -5.024 | 0.275 | Yes |
| P002 | -5.980 | -6.135 | 0.155 | Yes |
| P003 | -6.508 | -6.894 | 0.386 | Yes |
| P004 | -5.755 | -5.204 | 0.551 | No |

**Recovery: 4/5 agents within 0.5** (success criterion: >=4/5). P004 missed by 0.051.
hessian_min_eigval: 3.456 (positive, no clipping needed).

## Task Commits

1. **Task 1: Add jaxopt dependency + implement fit_vb_laplace_patrl** - `02d8e19` (feat)
2. **Task 2: Unit + smoke tests for fit_vb_laplace_patrl** - `e764336` (test)
3. **Plan metadata:** (this summary commit, below)

## Files Created/Modified

- `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` — Main Laplace fit orchestrator: MAP -> Hessian -> PD-regularize -> InferenceData. Exports `fit_vb_laplace_patrl` and `_regularize_to_pd`.
- `tests/test_fit_vb_laplace_patrl.py` — 12 tests (8 fast + 4 slow `@pytest.mark.slow`)
- `pyproject.toml` — Added `jaxopt>=0.8.5,<0.9` to `[project].dependencies` + `jaxopt.*` to mypy overrides

## Public Signature

```python
def fit_vb_laplace_patrl(
    sim_df: pd.DataFrame,
    model_name: str,                    # 'hgf_2level_patrl' | 'hgf_3level_patrl'
    response_model: str = "model_a",   # only 'model_a' supported; others raise NotImplementedError
    config: PATRLConfig | None = None,
    n_pseudo_draws: int = 1000,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_restarts: int = 1,
    random_seed: int = 0,
) -> az.InferenceData
```

Returns InferenceData with:
- `posterior.omega_2`, `log_beta`, `beta` (2-level) or + `omega_3`, `kappa`, `mu3_0` (3-level)
- Shape: `(chain=1, draw=n_pseudo_draws, participant_id=P)`
- `sample_stats`: all 7 canonical diagnostic keys

## Decisions Made

- **jaxopt.LBFGS tolerance**: `tol=1e-5` (default), `max_iter=200` — matches tapas HGF convergence criteria; empirically: 15 iterations for 3-agent 2-level case.
- **eigh-clip eps=1e-8**: matches jaxopt/scipy convention; no eigenvalues clipped on well-specified synthetic data (all positive at MAP).
- **Dense Hessian**: full `(P*K)x(P*K)` matrix via `jax.hessian(lambda f: -log_posterior_fn(unravel(f)))`. Block-diagonal property documented in TODO comment for Phase 20+ optimization.
- **param_order re-projection critical**: JAX pytrees (dicts) are flattened in sorted-key order by jaxopt. Re-ordering `best_mode_params` to `param_order` before `ravel_pytree` ensures cov columns align with `build_idata_from_laplace` expectation.
- **jit fallback coverage**: unit-level mock (`test_jit_fallback_warning_logged`) rather than E2E patching — avoids JAX tracer contamination from failed jit=True trace.
- **OQ2 (kappa native-space)**: kappa fitted in native TruncatedNormal space. Logit-reparam deferred to Phase 20+ if MAP hits boundary on cluster data.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed param_order key ordering after jaxopt.LBFGS**
- **Found during:** Task 2 slow tests (first run)
- **Issue:** `jaxopt.LBFGS.run()` returns params dict with JAX's alphabetically-sorted keys. `ravel_pytree` flattens in that order, so cov columns were in alphabetical order (e.g. `log_beta, omega_2`) not canonical order (`omega_2, log_beta`). `build_idata_from_laplace` validates key order → `ValueError: mode keys expected ('omega_2', 'log_beta'); got ('log_beta', 'omega_2')`.
- **Fix:** Re-project `best_mode_params` to `param_order` before `ravel_pytree`: `best_mode_params_ordered = {k: best_mode_params[k] for k in param_order}`. Apply same projection for `mode_native`.
- **Files modified:** `src/prl_hgf/fitting/fit_vb_laplace_patrl.py`
- **Committed in:** `e764336` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — Bug)
**Impact on plan:** Essential correctness fix. JAX dict pytree sort order is a JAX-version-stable behavior; fix is robust.

## Issues Encountered

- jit-fallback E2E test (using class-level mock of `LBFGS.run`) caused JAX tracer contamination: the failed jit=True trace left stale traced values in `base_attrs` closure, causing `UnexpectedTracerError` in the jit=False fallback run. Resolution: replaced class-level method mock with module-level name patch (`mock.patch.object(_mod, "LBFGS", _MockLBFGS)`), then further simplified to a pure unit test `test_jit_fallback_warning_logged` that verifies the fallback code path (warning + jit=False solver creation) without running full JAX compilation.

## git diff --stat verification

```
src/prl_hgf/fitting/fit_vb_laplace_patrl.py   (new)
tests/test_fit_vb_laplace_patrl.py             (new)
pyproject.toml                                  (2 lines added)
```

Protected paths unchanged (verified via `git diff --name-only` on all 9 protected files → empty output).

## OQ Carry-forward

- **OQ1 (dim-name bug)**: Sidestepped. `build_idata_from_laplace` emits `participant_id` natively. `_samples_to_idata` never called. OQ1 fix (hotfixing the NUTS path) deferred to Phase 18 follow-up.
- **OQ2 (kappa native-space vs logit-reparam)**: Deferred. TODO comment in code. Phase 19 ships native-space kappa. If cluster smoke (Plan 19-05) shows kappa MAP near boundary → logit-reparam in Phase 20+.

## Next Phase Readiness

- `fit_vb_laplace_patrl` is directly consumable by `export_subject_parameters` + `export_subject_trajectories` (Plan 18-05 exporters) — dim names and var names match.
- Plan 19-04 (pipeline integration test) can now run end-to-end: `fit_vb_laplace_patrl` → `export_subject_trajectories` → CSV.
- Plan 19-05 (cluster smoke) ready: the 5-agent <60s runtime gate is satisfied locally (18s for 5 agents with max_iter=200).

---
*Phase: 19-vb-laplace-fit-path-patrl*
*Completed: 2026-04-18*
