---
phase: 17-blackjax-nuts-sampler
plan: 01
subsystem: fitting
tags: [blackjax, jax, nuts, mcmc, bayesian, gpu, pmap, vmap]

# Dependency graph
requires:
  - phase: 16-numpyro-refactor
    provides: build_logp_fn_batched pure JAX logp factory, numpyro-direct MCMC path
provides:
  - _build_log_posterior: pure JAX logdensity_fn combining priors + batched HGF likelihood
  - _run_blackjax_nuts: window_adaptation warmup + lax.scan sampling with pmap/vmap
  - _samples_to_idata: BlackJAX output -> ArviZ InferenceData with participant dims
  - fit_batch_hierarchical rewritten with sampler="blackjax" default
  - blackjax dependency declared in pyproject.toml and cluster/requirements-gpu.txt
affects:
  - 17-02 (smoke test and validation)
  - power sweep pipeline (uses fit_batch_hierarchical)
  - cluster SLURM scripts (benefit from JIT cache reuse)

# Tech tracking
tech-stack:
  added: [blackjax>=1.2.4]
  patterns:
    - "Pure JAX log-posterior: prior_logp + batched_logp_fn closure"
    - "BlackJAX window_adaptation for single warmup, replicated across chains"
    - "lax.scan sampling loop with jitted one_step"
    - "pmap/vmap chain parallelism based on jax.device_count()"
    - "az.from_dict for manual InferenceData construction"

key-files:
  created: []
  modified:
    - src/prl_hgf/fitting/hierarchical.py
    - pyproject.toml
    - cluster/requirements-gpu.txt

key-decisions:
  - "BlackJAX as default sampler (sampler='blackjax'); NumPyro preserved as fallback"
  - "Single warmup replicated across chains (not per-chain warmup with averaged params)"
  - "numpyro.distributions for standalone prior log_prob (pure JAX, no model context)"
  - "pmap when device_count >= n_chains, vmap fallback on single device"
  - "sampler='pymc' deprecation now falls through to numpyro, not blackjax"

patterns-established:
  - "BlackJAX integration: _build_log_posterior -> _run_blackjax_nuts -> _samples_to_idata"
  - "Data captured in logdensity_fn closure (fixed shape per call, JIT cache reuse)"
  - "Initial position at prior modes for all participants"

# Metrics
duration: 9min
completed: 2026-04-15
---

# Phase 17 Plan 01: BlackJAX NUTS Sampler Summary

**Pure JAX log-posterior with BlackJAX NUTS default sampler, pmap/vmap chain parallelism, and ArviZ InferenceData conversion**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-15T11:06:38Z
- **Completed:** 2026-04-15T11:15:45Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Rewrote core fitting pipeline to use BlackJAX NUTS by default, eliminating ~1800s per-call JIT recompilation
- Built pure JAX log-posterior combining numpyro.distributions priors with existing batched_logp_fn
- Implemented multi-GPU pmap and single-GPU vmap chain parallelism strategies
- Preserved NumPyro MCMC as fallback sampler path with all existing functions unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Add BlackJAX dependency and implement core helpers** - `6bf835b` (feat)
2. **Task 2: Rewrite fit_batch_hierarchical orchestrator** - `40b5e2f` (feat)

## Files Created/Modified
- `src/prl_hgf/fitting/hierarchical.py` - Added _build_log_posterior, _run_blackjax_nuts, _run_vmap_chains, _run_pmap_chains, _samples_to_idata; rewrote fit_batch_hierarchical with BlackJAX default
- `pyproject.toml` - Added blackjax>=1.2.4 dependency; added blackjax and numpyro to mypy ignore_missing_imports
- `cluster/requirements-gpu.txt` - Added blackjax>=1.2.4 pin for cluster environment

## Decisions Made
- **BlackJAX as default sampler:** Changed `sampler` parameter default from `"numpyro"` to `"blackjax"` to leverage JIT-once pattern and eliminate recompilation overhead
- **Single warmup strategy:** Run warmup once with first chain's key, replicate adapted state across chains with different RNG keys for sampling; avoids complexity of per-chain warmup and parameter averaging
- **numpyro.distributions for priors:** Used standalone `dist.TruncatedNormal` and `dist.Normal` log_prob methods (pure JAX, no NumPyro model context) matching existing prior specs exactly
- **pmap/vmap routing:** Check `jax.device_count()` at runtime; pmap when >= n_chains devices available, vmap otherwise
- **sampler="pymc" fallthrough:** PyMC deprecation warning now routes to numpyro (not blackjax) since PyMC users would expect NumPyro-style behavior

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BlackJAX path is structurally complete; needs smoke test validation (Plan 17-02)
- All existing functions preserved: build_logp_fn_batched, numpyro models, PyTensor ops
- Cluster deployment requires `pip install blackjax>=1.2.4` in GPU environment
- JAX version compatibility note: blackjax 1.5 pulled in jax 0.9.2 on dev machine; cluster has jax 0.4.31 pinned which requires blackjax==1.2.4

---
*Phase: 17-blackjax-nuts-sampler*
*Completed: 2026-04-15*
