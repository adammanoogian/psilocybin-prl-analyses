---
phase: 18-pat-rl-task-adaptation
plan: 03
subsystem: models
tags: [pyhgf, hgf, binary-hgf, jax, softmax, expected-value, patrl, bayesian-learning]

# Dependency graph
requires:
  - phase: 18-01
    provides: PATRLConfig dataclass tree + pat_rl.yaml config surface
provides:
  - build_2level_network_patrl: single binary-input 2-level HGF builder (INPUT_NODE=0, BELIEF_NODE=1)
  - build_3level_network_patrl: single binary-input 3-level HGF builder (adds VOLATILITY_NODE=2 with kappa coupling)
  - extract_beliefs_patrl: belief extractor returning mu2, sigma2, p_state, expected_precision
  - extract_beliefs_patrl_3level: extends extract_beliefs_patrl with mu3, sigma3, epsilon3
  - model_a_logp: per-trial binary choice log-likelihood via softmax([0, beta*EV_approach])
  - expected_value: EV_approach = (1-P_danger)*V_rew - P_danger*V_shk
  - 9-test unit suite in tests/test_models_patrl.py
affects:
  - 18-04 (PAT-RL batched logp for fitting — consumes builders)
  - 18-05 (trajectory export — re-runs networks at posterior means)
  - 18-06 (validation — runs full pipeline end-to-end)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Parallel-stack isolation: all PAT-RL files in src/prl_hgf/models/ with _patrl suffix; zero imports from pick_best_cue equivalents; __init__.py untouched
    - pyhgf 0.2.8 scalar topology: binary-state INPUT_NODE=0, continuous-state BELIEF_NODE=1 (vs pick_best_cue INPUT_NODES tuple)
    - kappa coupling via volatility_children=([child_idx], [kappa]) tuple-of-lists (not node_parameters)
    - time_steps must be 1D np.ones(n_trials) in pyhgf 0.2.8 scan contract
    - node_trajectories includes key -1 (time tracker); count nodes with k>=0

key-files:
  created:
    - src/prl_hgf/models/hgf_2level_patrl.py
    - src/prl_hgf/models/hgf_3level_patrl.py
    - src/prl_hgf/models/response_patrl.py
    - tests/test_models_patrl.py
  modified: []

key-decisions:
  - "EV direction: sigmoid(mu2) = P(state=1=dangerous); EV_approach = (1-P_danger)*V_rew - P_danger*V_shk; at mu2=0 with V_shk>>V_rew, P(approach)<<0.01"
  - "kappa coupling via volatility_children tuple not node_parameters (pyhgf 0.2.8 API)"
  - "time_steps must be 1D (not 2D matrix) in pyhgf 0.2.8 scan contract"
  - "MU2_CLIP=30 for sigmoid stability (outer envelope; HGF-level clamping at |mu2|<14 in hierarchical.py is inner envelope)"
  - "type: ignore[return-value] on expected_value return due to jax.nn.sigmoid Any-return stubs limitation"

patterns-established:
  - "PAT-RL builders: single input_idxs=(0,) scalar; pick_best_cue uses input_idxs=(0,2,4) tuple"
  - "extract_beliefs_patrl(_3level) uses node_trajectories[BELIEF_NODE]['temp']['effective_precision'] with fallback to zeros"
  - "model_a_logp: logits = [0, beta*EV]; log_softmax axis=-1; take_along_axis for choice indexing"

# Metrics
duration: 49min
completed: 2026-04-17
---

# Phase 18 Plan 03: PAT-RL HGF Builders + Model A Response Summary

**Single-input binary HGF topology (INPUT_NODE=0) and Model A softmax EV response for PAT-RL approach/avoid task**

## Performance

- **Duration:** 49 min
- **Started:** 2026-04-17T20:30:00Z
- **Completed:** 2026-04-17T21:19:06Z
- **Tasks:** 3/3
- **Files modified:** 0 (4 created)

## Accomplishments

- Implemented `hgf_2level_patrl.py` and `hgf_3level_patrl.py` as scalar single-input variants of the pick_best_cue 3-branch builders; verified forward pass on 192-trial binary sequences produces finite belief trajectories
- Implemented `response_patrl.py` with `model_a_logp` (differentiable under `jax.grad`) and `expected_value` using EV = (1-P_danger)*V_rew - P_danger*V_shk convention
- All 9 pytest cases pass; pick_best_cue regression suite shows 64/65 passed (1 pre-existing blackjax failure), zero new failures

## Task Commits

Each task was committed atomically:

1. **Task 1: 2-level and 3-level binary HGF builders** - `1ccf5fa` (feat)
2. **Task 2: Model A binary-choice response log-likelihood** - `9eb2ee0` (feat)
3. **Task 3: Unit tests for PAT-RL HGF builders + Model A response** - `8863e77` (test)

## Files Created/Modified

- `src/prl_hgf/models/hgf_2level_patrl.py` — 2-level PAT-RL builder: build_2level_network_patrl + extract_beliefs_patrl
- `src/prl_hgf/models/hgf_3level_patrl.py` — 3-level PAT-RL builder: build_3level_network_patrl + extract_beliefs_patrl_3level
- `src/prl_hgf/models/response_patrl.py` — Model A: model_a_logp + expected_value + MU2_CLIP constant
- `tests/test_models_patrl.py` — 9 unit tests (topology, forward pass, determinism, logp shape, EV semantics, differentiability, regression guard)

## Decisions Made

**EV direction convention:** `sigmoid(mu2) = P(state=1=dangerous)`. Therefore:
- EV_approach = (1 - P_danger) * V_rew - P_danger * V_shk
- When mu2 large positive (P_danger ≈ 1): EV_approach ≈ -V_shk (costly)
- When mu2 large negative (P_danger ≈ 0): EV_approach ≈ +V_rew (safe)
- Test 7 validates: at mu2=0 (P_danger=0.5), V_rew=1, V_shk=10, P(approach) < 0.01

**pyhgf 0.2.8 API corrections:** The plan sketch used `node_parameters={"volatility_coupling_children": (kappa,)}` which is not the 0.2.8 API. The correct API is `volatility_children=([BELIEF_NODE], [kappa])` as a tuple-of-lists. Verified via runtime inspection before implementation.

**time_steps shape:** The plan sketch used `time_steps=np.ones((n_trials, 1))` (2D). pyhgf 0.2.8 scan contract requires `time_steps=np.ones(n_trials)` (1D). The 2D shape causes a JAX carry-type mismatch.

**MU2_CLIP=30:** Conservative outer envelope for sigmoid stability in response computation. The HGF-level clamping at |mu_2| < 14 (in hierarchical.py) is the inner envelope that handles fitting; MU2_CLIP=30 handles extreme extrapolation at export time.

**node_trajectories key -1:** pyhgf 0.2.8 `node_trajectories` includes key -1 (global time_step tracker) in addition to node indices 0, 1, 2, etc. Tests count nodes with `k >= 0` to get the correct node count.

**type: ignore annotations:** `jax.nn.sigmoid` returns `Any` per JAX mypy stubs. Added `# type: ignore[assignment]` and `# type: ignore[return-value]` in `expected_value` to suppress the false-positive. This is a JAX stubs limitation, not a real type error.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pyhgf API correction: kappa coupling via volatility_children tuple, not node_parameters**

- **Found during:** Task 1 (3-level builder implementation)
- **Issue:** Plan sketch specified `node_parameters={"volatility_coupling_children": (kappa,)}` which is not valid in pyhgf 0.2.8. The node_parameters dict does not accept coupling strength keys.
- **Fix:** Used `volatility_children=([BELIEF_NODE], [kappa])` as documented in pyhgf 0.2.8 API. Verified with runtime inspection.
- **Files modified:** src/prl_hgf/models/hgf_3level_patrl.py
- **Committed in:** 1ccf5fa (Task 1 commit)

**2. [Rule 1 - Bug] time_steps must be 1D (not 2D) in pyhgf 0.2.8 scan**

- **Found during:** Task 1 verification (forward pass test)
- **Issue:** Plan sketch showed `time_steps=np.ones((192, 1))` (2D). pyhgf 0.2.8 scan_fn expects 1D time_steps; 2D causes JAX carry-type shape mismatch ("the input carry component attributes[-1]['time_step'] has type float32[] but the corresponding output carry component has type float32[1]").
- **Fix:** Changed to `time_steps=np.ones(n_trials)` (1D) in verification scripts and test file.
- **Files modified:** tests/test_models_patrl.py
- **Committed in:** 8863e77 (Task 3 commit)

**3. [Rule 1 - Bug] jax import missing from plan sketch + type: ignore for sigmoid stubs**

- **Found during:** Task 2 (response_patrl.py implementation)
- **Issue:** Plan sketch used `jax.nn.sigmoid` without `import jax` (noted in plan constraints). Also, mypy errors from jax.nn.sigmoid returning Any in stubs.
- **Fix:** Added explicit `import jax` at top of response_patrl.py. Added `# type: ignore` comments on affected lines.
- **Files modified:** src/prl_hgf/models/response_patrl.py
- **Committed in:** 9eb2ee0 (Task 2 commit)

**4. [Rule 1 - Bug] ruff unused import cleanup in test file**

- **Found during:** Task 3 verification (ruff check)
- **Issue:** Unused imports `pytest`, `BELIEF_NODE`, `VOLATILITY_NODE`, `expected_value` in test file; also import ordering.
- **Fix:** Removed unused imports, sorted import block with `ruff --fix`.
- **Files modified:** tests/test_models_patrl.py
- **Committed in:** 8863e77 (Task 3 commit)

---

**Total deviations:** 4 auto-fixed (4 Rule 1 - Bug)
**Impact on plan:** All fixes necessary for correct pyhgf API usage and code hygiene. No scope changes.

## Issues Encountered

- The `test_valid_02_batched_numpyro_convergence` test (regression suite) takes approximately 30 minutes on CPU. Ran in background to avoid blocking; confirmed pass in final output (1641s call time). Pre-existing `test_valid_02_batched_blackjax_convergence` failure persists due to blackjax not installed in ds_env.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 18-04 (batched logp for fitting) can import `build_2level_network_patrl`, `build_3level_network_patrl`, `model_a_logp` directly; no __init__.py changes needed
- 18-05 (trajectory export) can use `extract_beliefs_patrl(_3level)` after posterior MCMC
- The scalar topology (INPUT_NODE=0, BELIEF_NODE=1, VOLATILITY_NODE=2) is verified and documented; downstream phases should use these constants directly

---
*Phase: 18-pat-rl-task-adaptation*
*Completed: 2026-04-17*
