---
phase: 30-laplace-warmup-fp64-multigpu-flags
plan: "03"
subsystem: fitting
tags: [jax, shard_map, pmap, multi-gpu, nuts, blackjax, chain-dispatch, MODEA-06]

# Dependency graph
requires:
  - phase: 30-02
    provides: multi-start Laplace warmup + basin diagnostic (same file)
  - phase: 29-m1-dense-lowrank-mass-matrix-wiring
    provides: FitConfig + MitigationConfig with use_shard_map field
provides:
  - jax.shard_map over 1-D Mesh("chains") replacing deprecated jax.pmap
  - _run_shard_map_chains replacing _run_pmap_chains (legacy fallback path)
  - sample_loop_shard_map replacing sample_loop_pmap in _build_sample_loop
  - CI test (P7) asserting 4 chains across 2 devices produce distinct samples
  - check_rep=False pattern documented for NUTS+shard_map compatibility
affects:
  - 30-04
  - 31-sampling-benchmarks
  - future phases using _run_shard_map_chains or _build_sample_loop

# Tech tracking
tech-stack:
  added: [jax.experimental.shard_map, jax.sharding.Mesh, jax.sharding.PartitionSpec]
  patterns:
    - "shard_map + check_rep=False for NUTS (lax.while_loop has no replication rule)"
    - "vmap inside shard_map body for local chain batching (n_chains > n_devices)"
    - "P7 RNG uniqueness: jax.random.split before shard_map, unique key per chain"

key-files:
  created:
    - tests/integration/test_shard_map_chains.py
  modified:
    - src/prl_hgf/fitting/hierarchical.py

key-decisions:
  - "check_rep=False required for NUTS inside shard_map (lax.while_loop tree expansion has no replication rule)"
  - "vmap inside shard_map body handles local_n chains per device (n_chains / n_devices)"
  - "use_pmap variable renamed to use_multi_device throughout to reflect shard_map semantics"
  - "MitigationConfig.use_shard_map field reserved for Phase 31 forced-sharding (not consumed here)"

patterns-established:
  - "shard_map NUTS pattern: check_rep=False + vmap inside shard body + unique keys via jax.random.split"
  - "P7 test pattern: subprocess with XLA_FLAGS mock devices + byte-for-byte chain comparison"

# Metrics
duration: 9min
completed: 2026-05-17
---

# Phase 30 Plan 03: shard_map Chain Dispatch Summary

**jax.pmap fully replaced by jax.shard_map over a 1-D Mesh("chains") in both chain dispatch paths, with check_rep=False for NUTS compatibility and a CI test asserting P7 RNG uniqueness across 4 chains / 2 mock CPU devices**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-17T17:46:54Z
- **Completed:** 2026-05-17T17:55:51Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced deprecated `jax.pmap` in both chain dispatch paths (`_run_pmap_chains` / `sample_loop_pmap`) with `jax.experimental.shard_map` over a named Mesh, per MODEA-06
- Added `check_rep=False` to both shard_map call sites so NUTS's `lax.while_loop` tree expansion is compatible
- Renamed `use_pmap` → `use_multi_device` throughout (variable, parameter, log message)
- Created integration test that mocks 2 CPU devices via `XLA_FLAGS`, runs 4 chains (2 per device via vmap inside shard body), and asserts all 6 pairwise chain samples differ byte-for-byte (P7)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace pmap with shard_map in chain dispatch paths** - `513a20f` (feat)
2. **Task 2: CI test + check_rep=False** - `a596ac7` (feat)

## Files Created/Modified

- `src/prl_hgf/fitting/hierarchical.py` - `_run_pmap_chains` deleted, `_run_shard_map_chains` added; `sample_loop_pmap` → `sample_loop_shard_map`; `use_pmap` → `use_multi_device`; `check_rep=False` added; MitigationConfig.use_shard_map reserved-for-Phase-31 comment added
- `tests/integration/test_shard_map_chains.py` - P7 integration test using subprocess XLA_FLAGS mock devices

## Decisions Made

- **check_rep=False mandatory for NUTS+shard_map**: `lax.while_loop` (NUTS tree expansion) has no replication rule in shard_map. `check_rep=False` disables the check; output sharding is still correctly declared via `out_specs=P("chains")`.
- **vmap inside shard body for n_chains > n_devices**: When `n_chains/n_devices > 1`, each shard receives `(local_n, ...)` tensors. `jax.vmap` inside the shard body handles the local batch independently. Test uses 4 chains / 2 devices to exercise this pattern.
- **use_pmap → use_multi_device**: Reflects shard_map semantics; no behavioral change (logic remains `n_devices >= n_chains`).
- **MitigationConfig.use_shard_map reserved for Phase 31**: Current decision stays automatic. Phase 31 will consume the flag for forced-sharding even when n_devices < n_chains.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added check_rep=False to shard_map call sites**

- **Found during:** Task 2 (test execution)
- **Issue:** NUTS uses `lax.while_loop` for tree expansion, which has no replication rule in shard_map. Both production call sites in hierarchical.py and the test script failed with `NotImplementedError: No replication rule for while`.
- **Fix:** Added `check_rep=False` to all three `shard_map(...)` calls (2 in hierarchical.py, 1 in test script).
- **Files modified:** `src/prl_hgf/fitting/hierarchical.py`, `tests/integration/test_shard_map_chains.py`
- **Verification:** Test passes; import check passes; ruff clean.
- **Committed in:** `a596ac7` (Task 2 commit)

**2. [Rule 1 - Bug] Added vmap inside shard body for correct 4-chain / 2-device layout**

- **Found during:** Task 2 (test execution, first attempt)
- **Issue:** With `n_chains=4, n_devices=2`, `P("chains")` gives each device a shard of shape `(2, ...)` for keys and states. `jax.random.split(rng_key, n_draws)` failed with "split accepts a single key, but was given a key array of shape (2, 2)". Direct function call without vmap doesn't handle the local batch.
- **Fix:** Wrapped per-chain function in `jax.vmap` inside the shard body (`_shard_body`), which maps over the local `(local_n, ...)` leading dimension. Updated test comment to document this pattern.
- **Files modified:** `tests/integration/test_shard_map_chains.py`
- **Verification:** Test passes cleanly (10s wall clock).
- **Committed in:** `a596ac7` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs discovered during test execution)
**Impact on plan:** Both fixes necessary for shard_map + NUTS compatibility. No scope creep. The `check_rep=False` pattern and `vmap-inside-shard` pattern are now documented for future phases.

## Issues Encountered

- Legacy JAX uint32 key format (shape `(2,)` per key, `(n, 2)` after split) requires `vmap` inside shard body when `n_chains > n_devices`. Modern typed-key JAX would have scalar `()` keys making direct calls feasible. Workaround is correct and documented.

## Next Phase Readiness

- `jax.pmap` fully removed from hierarchical.py chain dispatch paths; shard_map is the only multi-device path
- `check_rep=False` pattern established for all future NUTS+shard_map usage
- P7 RNG uniqueness test in place for CI regression prevention
- Phase 31 can consume `MitigationConfig.use_shard_map` to enable forced-sharding with vmap fallback

---
*Phase: 30-laplace-warmup-fp64-multigpu-flags*
*Completed: 2026-05-17*
