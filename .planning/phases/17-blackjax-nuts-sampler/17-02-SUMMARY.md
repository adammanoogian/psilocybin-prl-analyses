---
phase: 17-blackjax-nuts-sampler
plan: 02
subsystem: testing
tags: [blackjax, jax, nuts, validation, arviz, slurm, gpu, pmap, vmap]

# Dependency graph
requires:
  - phase: 17-blackjax-nuts-sampler (plan 01)
    provides: _build_log_posterior, _run_blackjax_nuts, _samples_to_idata, fit_batch_hierarchical with blackjax default
provides:
  - BlackJAX smoke tests: log-posterior finite scalar, gradient finite, _samples_to_idata correct dims
  - VALID-02 BlackJAX convergence test (slow, 5-participant 2-level fit)
  - NumPyro fallback path explicitly tested with sampler="numpyro"
  - SLURM scripts updated for BlackJAX default and pmap documentation
affects:
  - cluster deployment (SLURM scripts reference BlackJAX)
  - power sweep pipeline (default sampler changed to blackjax in 08_power_sweep.slurm)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BlackJAX logp smoke test: build_logp_fn_batched + _build_log_posterior + eval at prior modes"
    - "Gradient smoke test: jax.grad(logdensity_fn) validates differentiability through lax.scan"
    - "ArviZ conversion unit test: dummy positions/stats to _samples_to_idata with dim assertions"

key-files:
  created: []
  modified:
    - tests/test_hierarchical_logp.py
    - cluster/16_smoke_test_gpu.slurm
    - cluster/08_power_sweep.slurm

key-decisions:
  - "Separate 2-level and 3-level log-posterior smoke tests (not parameterized) for clearer failure diagnostics"
  - "SLURM power sweep default sampler changed from numpyro to blackjax via SAMPLER env var"
  - "Smoke test JIT gate thresholds kept at NumPyro levels; annotated for future BlackJAX-specific tuning"

patterns-established:
  - "BlackJAX test pattern: build_logp_fn_batched -> _build_log_posterior -> eval/grad at prior modes"
  - "SLURM sampler selection via SAMPLER env var (default: blackjax)"

# Metrics
duration: 9min
completed: 2026-04-15
---

# Phase 17 Plan 02: BlackJAX Smoke Tests and SLURM Updates Summary

**BlackJAX log-posterior and gradient smoke tests, ArviZ conversion validation, VALID-02 convergence test, and SLURM scripts updated for BlackJAX default sampler with pmap documentation**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-15T11:19:05Z
- **Completed:** 2026-04-15T11:27:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added 4 new BlackJAX-specific fast tests: 3-level logp smoke, 2-level logp smoke, gradient smoke, and _samples_to_idata conversion test
- Added VALID-02 BlackJAX convergence test (slow) exercising the default sampler path
- Updated existing VALID-02 NumPyro test to explicitly pass sampler="numpyro" for fallback path coverage
- Updated both SLURM scripts with BlackJAX documentation, pmap instructions, and JIT threshold review notes
- All 10 fast tests pass (6 existing VALID-01/clamping + 4 new BlackJAX)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update validation tests for BlackJAX path** - `5061845` (test)
2. **Task 2: Update SLURM scripts for BlackJAX + multi-GPU** - `34fbae0` (docs)

## Files Created/Modified
- `tests/test_hierarchical_logp.py` - Added BlackJAX smoke tests (logp, gradient, idata), VALID-02 blackjax convergence test, updated numpyro test with explicit sampler kwarg
- `cluster/16_smoke_test_gpu.slurm` - Updated header to Phase 16/17 BlackJAX, added pmap note, JIT gate threshold review comment
- `cluster/08_power_sweep.slurm` - Updated header with BlackJAX docs and pmap instructions, changed default SAMPLER from numpyro to blackjax

## Decisions Made
- **Separate 2-level/3-level smoke tests:** Used individual test functions instead of pytest.mark.parametrize for clearer failure diagnostics when one model variant fails
- **SLURM default sampler change:** Changed SAMPLER env var default in 08_power_sweep.slurm from "numpyro" to "blackjax" to match the fit_batch_hierarchical default
- **JIT gate thresholds preserved:** Kept existing cold JIT < 600s, cache speedup > 3x, warm JIT < 120s thresholds with annotation that they were set for NumPyro and may be tightened after BlackJAX benchmarking

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 17 complete: BlackJAX is the default sampler with full test coverage
- Existing VALID-01 (bit-exact) tests unchanged and passing
- SLURM scripts ready for cluster deployment with BlackJAX
- Multi-GPU pmap instructions documented in 08_power_sweep.slurm for L40S clusters
- Next: Run slow tests on cluster to validate VALID-02 with actual MCMC

---
*Phase: 17-blackjax-nuts-sampler*
*Completed: 2026-04-15*
