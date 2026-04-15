---
phase: quick-003
plan: 01
subsystem: fitting
tags: [blackjax, jit-cache, xla, power-sweep, performance]
dependency_graph:
  requires: [phase-17]
  provides: ["_build_sample_loop factory", "traced-arg BlackJAX sampling"]
  affects: [power-sweep-cluster-runs]
tech_stack:
  added: []
  patterns: ["traced-arg JIT factory for XLA cache reuse"]
key_files:
  created: []
  modified:
    - src/prl_hgf/fitting/hierarchical.py
decisions:
  - id: Q003-D1
    decision: "vmap path uses @jax.jit; pmap path lets pmap handle compilation"
    rationale: "pmap inside a JIT boundary is problematic; factory returns different function variants based on use_pmap"
  - id: Q003-D2
    decision: "Legacy fallback preserved when batched_logp_fn is None"
    rationale: "Backward compatibility for callers not providing data args; existing tests still pass"
  - id: Q003-D3
    decision: "Prior distributions constructed once in factory, captured by closure"
    rationale: "numpyro.distributions objects are parameterless JAX objects (no data); safe to capture without affecting HLO hash"
metrics:
  duration: "~8 minutes"
  completed: "2026-04-15"
---

# Quick Task 003: JIT Cache -- Data as Traced Args Summary

**One-liner:** BlackJAX sampling loop restructured so data arrays flow as traced JIT arguments, enabling persistent XLA compilation cache hits across power-sweep iterations with same-shape data.

## What Changed

### _build_sample_loop factory (NEW)

Factory function that builds a JIT'd sampling function where data arrays (input_data, observed, choices, trial_mask) are explicit function arguments instead of closure-captured constants. This is the core change that solves the XLA persistent cache problem.

- **vmap path** (single GPU): Returns `@jax.jit`-decorated function; data arrays become shape-dependent placeholders in HLO
- **pmap path** (multi-GPU): Returns a function using `jax.pmap` internally (pmap handles its own JIT compilation)
- Prior distributions constructed once in factory scope (parameterless JAX objects, safe to capture)
- logdensity_fn reconstructed inside JIT boundary using traced data args

### _run_blackjax_nuts (MODIFIED)

- New optional parameters: `batched_logp_fn`, `input_data`, `observed`, `choices`, `trial_mask`, `model_name`
- When all new parameters are provided: uses `_build_sample_loop` for sampling phase
- When new parameters are None: falls back to legacy closure-based `_run_vmap_chains`/`_run_pmap_chains`
- Warmup phase unchanged (closure-based `logdensity_fn`, runs once)
- Post-processing (JAX arrays to numpy, transpose) moved into `_run_blackjax_nuts`

### fit_batch_hierarchical (MODIFIED)

- BlackJAX call site now passes `batched_logp_fn`, all 4 data arrays, and `model_name` to `_run_blackjax_nuts`

## Why This Matters

The XLA persistent compilation cache keys on HLO hash. When data arrays are closure-captured, they become HLO constants -- different data values produce different HLO hashes, causing full recompilation (~1600s) on every power-sweep iteration even when shapes are identical. By making data arrays explicit JIT function arguments, XLA traces them as shape-dependent placeholders. Same shapes produce the same HLO hash, enabling cache hits.

## Verification

1. `from prl_hgf.fitting.hierarchical import fit_batch_hierarchical, _build_sample_loop` -- PASS
2. `ruff check` -- PASS (no lint errors)
3. `ruff format --check` -- PASS
4. BlackJAX smoke tests (4/4 pass locally): log_posterior_smoke_3level, log_posterior_smoke_2level, gradient_smoke, samples_to_idata_smoke
5. `test_valid_02_batched_blackjax_convergence` -- SKIP (blackjax not installed locally; cluster-only GPU dependency, pre-existing)

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 5e6d9f0 | feat | restructure BlackJAX sampling loop for JIT cache reuse |

## Next Steps

- Run full VALID-02 convergence test on cluster with blackjax installed
- Benchmark persistent cache hit rate across power-sweep iterations (expect ~1600s saved per iteration after first)
- Tighten JIT gate thresholds after cluster benchmarking
