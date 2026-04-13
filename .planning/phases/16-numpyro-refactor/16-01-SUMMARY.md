---
phase: 16-numpyro-refactor
plan: 01
subsystem: fitting
tags: [numpyro, jax, mcmc, nuts, arviz, hierarchical]

# Dependency graph
requires:
  - phase: 12-batched-ops
    provides: batched JAX logp infrastructure (build_logp_ops_batched, _single_logp_3level, _clamped_scan)
  - phase: 13-jax-simulation
    provides: pure JAX simulation path used alongside fitting
provides:
  - build_logp_fn_batched: pure JAX logp factory with data as arguments (no closure)
  - _numpyro_model_3level / _numpyro_model_2level: numpyro model functions with exact prior matches
  - fit_batch_hierarchical rewritten to use numpyro MCMC + NUTS directly (no PyMC bridge)
  - InferenceData output with participant dim and group/session coords via az.from_numpyro
affects:
  - 16-02 (CUDA fix + environment diagnostics)
  - 14-01 (iteration.py calls fit_batch_hierarchical -- now uses numpyro path)
  - VALID-01/02 (backward compat preserved via deprecated build_logp_ops_batched)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "numpyro model function with numpyro.factor() for custom logp"
    - "Data as MCMC.run() kwargs for JIT cache reuse across iterations"
    - "chain_method=vectorized for single-GPU 4-chain execution"
    - "az.from_numpyro(mcmc, dims=, coords=) for InferenceData conversion"

key-files:
  created: []
  modified:
    - src/prl_hgf/fitting/hierarchical.py
    - src/prl_hgf/fitting/__init__.py

key-decisions:
  - "Additive refactor: all existing PyTensor Op code kept for VALID-01/02 backward compat"
  - "build_pymc_model_batched marked deprecated with DeprecationWarning, not deleted"
  - "chain_method=vectorized chosen over sequential+jit_model_args for single-GPU throughput"
  - "numpyro.sample() + numpyro.factor() pattern over raw potential_fn for ArviZ integration"
  - "Data passed as kwargs to mcmc.run(), not captured in closures, enabling XLA JIT cache reuse"

patterns-established:
  - "numpyro model function pattern: priors via numpyro.sample() + custom logp via numpyro.factor()"
  - "build_logp_fn_batched returns (callable, n_params) tuple with data as explicit args"

# Metrics
duration: 9min
completed: 2026-04-13
---

# Phase 16 Plan 01: NumPyro Direct Model Functions Summary

**Replaced PyMC bridge with direct numpyro MCMC sampling in fit_batch_hierarchical, enabling JIT cache reuse via data-as-arguments pattern**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-13T22:14:25Z
- **Completed:** 2026-04-13T22:22:55Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- Added `build_logp_fn_batched()` factory that returns a pure JAX callable taking data as explicit arguments (no closure capture)
- Added `_numpyro_model_3level` and `_numpyro_model_2level` with priors exactly matching the existing PyMC model
- Rewrote `fit_batch_hierarchical()` to use `numpyro.infer.MCMC` + `NUTS` directly with `chain_method="vectorized"`
- Data arrays passed as kwargs to `mcmc.run()` so XLA trace is shape-dependent but value-independent
- `az.from_numpyro()` produces InferenceData with `participant` dim and `participant_group`/`participant_session` coords
- Full backward compatibility: `build_logp_ops_batched`, `build_pymc_model_batched` still importable

## Task Commits

Each task was committed atomically:

1. **Task 1: Create build_logp_fn_batched and numpyro model functions** - `f6d5715` (feat)
2. **Task 2: Update __init__.py exports** - `03ad7d9` (chore)

## Files Created/Modified
- `src/prl_hgf/fitting/hierarchical.py` - Added build_logp_fn_batched, _numpyro_model_3level, _numpyro_model_2level; rewrote fit_batch_hierarchical; deprecated build_pymc_model_batched
- `src/prl_hgf/fitting/__init__.py` - Added build_logp_fn_batched to imports and __all__; updated module docstring

## Decisions Made
- Used additive refactor approach: kept all existing PyTensor Op infrastructure for backward compat with VALID-01/02 tests
- Chose `chain_method="vectorized"` over sequential with `jit_model_args=True` because vectorized compiles a single kernel for all 4 chains on one GPU
- Used `numpyro.sample()` + `numpyro.factor()` pattern rather than raw `potential_fn` to get proper named parameters in ArviZ output
- `sampler` parameter kept on `fit_batch_hierarchical` for API compatibility but `sampler="pymc"` now raises DeprecationWarning
- Deferred numpyro imports inside model functions and fit function to match existing deferred-import pattern

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `fit_batch_hierarchical` now uses numpyro-direct path; ready for Plan 16-02 (CUDA fix + environment diagnostics)
- Existing callers of `fit_batch_hierarchical` (e.g., `iteration.py`) need no changes -- same signature, same InferenceData output structure
- VALID-01/02 tests should still pass via the deprecated `build_logp_ops_batched` path
- End-to-end integration test (VALID-02 equivalent for numpyro path) deferred to Plan 16-02

---
*Phase: 16-numpyro-refactor*
*Completed: 2026-04-13*
