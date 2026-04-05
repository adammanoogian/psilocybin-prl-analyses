---
phase: 02-models
plan: "01"
subsystem: models
tags: [pyhgf, jax, hgf, binary-hgf, network-api, partial-feedback, belief-extraction]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: package skeleton, config system, task environment simulator
provides:
  - build_2level_network(): 3-branch 2-level binary HGF (6 nodes)
  - build_3level_network(): 3-branch 3-level binary HGF with shared volatility parent (7 nodes)
  - prepare_input_data(): partial-feedback array builder for pyhgf input_data()
  - extract_beliefs() / extract_beliefs_3level(): named belief trajectory extraction
  - Node index constants: INPUT_NODES, BELIEF_NODES, VOLATILITY_NODE, N_CUES
affects:
  - 03-simulation (builds synthetic agents that forward-pass these networks)
  - 04-fitting (PyMC fitting wraps these networks; HGFDistribution incompatibility documented)
  - 05-recovery (parameter recovery uses these exact builder functions)
  - 06-analysis (group-level analysis reads belief trajectories)
  - 07-gui (visualization of belief trajectories from these networks)

# Tech tracking
tech-stack:
  added:
    - pyhgf 0.2.8 (JAX-backed HGF, Network API) — first JAX code executed in project
  patterns:
    - Factory function pattern for network builders (build_2level_network, build_3level_network)
    - Partial feedback via observed mask (np.ndarray int dtype, shape (n_trials, 3))
    - Three-layer naming: omega_2/omega_3/kappa at API boundary, tonic_volatility internally
    - extract_beliefs() uses "mean" for continuous nodes (log-odds), "expected_mean" for binary nodes (probability)

key-files:
  created:
    - src/prl_hgf/models/hgf_2level.py
    - src/prl_hgf/models/hgf_3level.py
    - tests/test_models.py
  modified:
    - src/prl_hgf/models/__init__.py

key-decisions:
  - "net.edges is a tuple (not dict): node N is accessed as net.edges[N], not net.edges[N] with 'in' operator"
  - "extract_beliefs uses 'mean' field for continuous nodes (log-odds posterior) and 'expected_mean' for binary nodes (sigmoid probability in [0,1])"
  - "observed mask must use int dtype (not bool) per pyhgf JAX tracing requirements"

patterns-established:
  - "Pattern: pyhgf Network.edges is a tuple of AdjacencyLists; index by position not by 'in' membership test"
  - "Pattern: 3-level volatility parent uses volatility_children=([1,3,5],[kappa,kappa,kappa]) tuple form"
  - "Pattern: session-scoped fixtures for model tests (config, trials, simple_input) to avoid repeated JAX compilation"

# Metrics
duration: 7min
completed: 2026-04-05
---

# Phase 2 Plan 1: HGF Model Builders Summary

**3-branch 2-level and 3-level binary HGF networks via pyhgf 0.2.8 Network API with partial-feedback observed mask and named belief trajectory extraction**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-05T11:26:51Z
- **Completed:** 2026-04-05T11:33:44Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `build_2level_network()`: 6-node 3-branch binary HGF with INPUT_NODES=(0,2,4), BELIEF_NODES=(1,3,5), tonic_volatility mapped to omega_2
- `build_3level_network()`: 7-node extension with shared volatility parent (node 6) using `volatility_children=([1,3,5],[kappa,kappa,kappa])` coupling
- `prepare_input_data()`: builds partial-feedback `(input_data, observed)` arrays with int dtype for pyhgf compliance
- `extract_beliefs()` / `extract_beliefs_3level()`: extracts mu1 from continuous node `"mean"` field (log-odds) and p_reward from binary node `"expected_mean"` field (sigmoid probability)
- 19 unit tests all green, JAX + pyhgf integration confirmed working in ds_env

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement 2-level and 3-level HGF model builders** - `b17ab61` (feat)
2. **Task 2: Unit tests for model construction, forward pass, and belief extraction** - `059ecff` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `src/prl_hgf/models/hgf_2level.py` - Factory `build_2level_network()`, `prepare_input_data()`, `extract_beliefs()`; node index constants
- `src/prl_hgf/models/hgf_3level.py` - Factory `build_3level_network()`, `extract_beliefs_3level()`; VOLATILITY_NODE constant
- `src/prl_hgf/models/__init__.py` - Public API exports for all symbols
- `tests/test_models.py` - 19 unit tests (construction, forward pass, partial feedback, belief correctness, 3-level volatility, prepare_input_data)

## Decisions Made

- `net.edges` is a tuple (not a dict): node index N maps to `net.edges[N]` by position, not by dict-key lookup. Test `test_3level_volatility_parent_exists` fixed to use positional indexing.
- `extract_beliefs` uses `"mean"` field on continuous-state nodes (log-odds posterior mu1) and `"expected_mean"` on binary-state nodes (sigmoid-transformed reward probability). Using the wrong field would give values outside [0,1] for p_reward.
- `observed` mask uses `dtype=int` as required by pyhgf JAX tracing (bool dtype can cause issues under JAX `jit`).
- Session-scoped pytest fixtures used for `simple_input` to avoid repeated JAX compilation overhead (~1s per forward pass).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_3level_volatility_parent_exists using wrong membership test**

- **Found during:** Task 2 (unit test creation and first run)
- **Issue:** Test used `VOLATILITY_NODE in net.edges` which checks tuple value membership, not index existence. `net.edges` is a tuple of `AdjacencyLists` namedtuples — `6 in (adj0, adj1, ..., adj6)` is False because 6 is not an element of the tuple, only an index position.
- **Fix:** Changed test to access `net.edges[VOLATILITY_NODE]` (positional) and check `vol_edge.volatility_children == (1, 3, 5)` for positive confirmation of wiring.
- **Files modified:** `tests/test_models.py`
- **Verification:** `conda run -n ds_env python -m pytest tests/test_models.py -v` — 19/19 passed
- **Committed in:** `059ecff` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Single test fix required to correctly introspect pyhgf's tuple-based edges structure. No scope creep; all originally planned tests implemented.

## Issues Encountered

- JAX / pyhgf first execution took ~1-2 seconds per forward pass (JIT compilation overhead on first call); subsequent calls in same session are fast. Tests run in ~13s total due to JAX compilation per test (no shared compiled state across parametrized tests). Mitigated by session-scoped fixtures where possible.
- `conda run -n ds_env python -c "..."` does not support newline characters in `-c` arguments on Windows. Workaround: write verification code to a temporary `.py` file. The temp file `_tmp_verify_task1.py` was removed before the Task 1 commit.

## Next Phase Readiness

- Both HGF model variants are fully functional and tested; ready for Phase 3 (simulation of synthetic participants)
- `prepare_input_data()` API is designed for Phase 3: accepts trial list + choice/reward arrays, returns pyhgf-compatible arrays
- Phase 4 (fitting) caveat: `HGFDistribution` (pyhgf's PyMC Op) cannot be used with custom `Network`; a custom PyMC Op wrapping the multi-branch network's logp will be needed
- Open question from research (volatility PE gating for unobserved branches) was empirically confirmed: unobserved cue level-1 nodes have constant `mu1` (test `test_unobserved_cue_beliefs_constant` green)

---
*Phase: 02-models*
*Completed: 2026-04-05*
