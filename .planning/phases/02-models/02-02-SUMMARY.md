---
phase: 02-models
plan: "02"
subsystem: models
tags: [jax, pyhgf, softmax, stickiness, response-function, likelihood, hgf, pytest]

# Dependency graph
requires:
  - phase: 02-01
    provides: HGF network builders (build_2level_network, build_3level_network), INPUT_NODES constant, net.surprise() API
provides:
  - softmax_stickiness_surprise() response function compatible with pyhgf Network.surprise() API
  - Comprehensive unit + integration tests for response function (12 tests)
  - Updated models __init__.py exporting response function
affects:
  - "03-simulation: synthetic participants will call net.surprise() with softmax_stickiness_surprise"
  - "04-fitting: PyMC Op wraps net.surprise() using this response function as the likelihood"
  - "05-recovery: parameter recovery validation uses this function to evaluate fits"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pyhgf response function protocol: fn(hgf, response_function_inputs, response_function_parameters) -> float"
    - "JAX-native array ops throughout response function for MCMC compatibility"
    - "Sentinel prev_choice=-1 for first trial to zero stickiness term"
    - "jax.nn.log_softmax for numerically stable log-probability computation"
    - "jnp.where NaN guard returns inf for degenerate parameter combos"

key-files:
  created:
    - src/prl_hgf/models/response.py
    - tests/test_response.py
  modified:
    - src/prl_hgf/models/__init__.py

key-decisions:
  - "Uses expected_mean from binary INPUT_NODES (0,2,4) as mu1_k — sigmoid-transformed P(reward|cue k) in [0,1], not continuous-node log-odds"
  - "First trial stickiness term is zero via sentinel prev_choice=-1 in jnp.concatenate"
  - "jax.nn.log_softmax used (not manual formula) since JAX 0.4+ provides it"
  - "response function must work with JAX arrays throughout — choices cast via jnp.asarray inside function"

patterns-established:
  - "Response function pattern: extract expected_mean from INPUT_NODES, stack to (n_trials,3), apply softmax"
  - "Test fixtures use _make_forward_result helper functions (not session-scoped fixtures) since JAX network state is mutable"
  - "Lint: ruff --fix corrects import sort; unused vars must be removed manually"

# Metrics
duration: ~8min
completed: 2026-04-05
---

# Phase 2 Plan 02: Response Function Summary

**Softmax + stickiness response function bridging HGF beliefs to choice likelihoods via pyhgf net.surprise() API, with 12 tests covering all behavioral properties**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-05T11:37:01Z
- **Completed:** 2026-04-05T11:45:11Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented `softmax_stickiness_surprise()` with formula `logit_k = beta * mu1_k + zeta * I[prev_choice==k]`, matching pyhgf `Network.surprise()` API exactly
- First trial correctly has zero stickiness term via sentinel `prev_choice=-1`
- NaN guard returns `inf` for degenerate parameter combinations
- 12 tests covering all behavioral properties: finite surprise, stickiness direction, beta concentration, NaN guard, full end-to-end pipeline, and JAX array compatibility for Phase 4

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement softmax + stickiness response function** - `4fd9058` (feat)
2. **Task 2: Unit and integration tests for response function** - `1c92df7` (feat)

**Plan metadata:** (committed next)

## Files Created/Modified

- `src/prl_hgf/models/response.py` - Response function module with `softmax_stickiness_surprise()`
- `tests/test_response.py` - 12 unit and integration tests
- `src/prl_hgf/models/__init__.py` - Added `softmax_stickiness_surprise` export

## Decisions Made

- **Uses `expected_mean` from binary INPUT_NODES (0,2,4) as mu1_k**: These are sigmoid-transformed P(reward|cue k) in [0,1], the correct quantity for the softmax formula per RSP-02. Continuous-state belief node log-odds (BELIEF_NODES) are not used.
- **`jax.nn.log_softmax` used instead of manual formula**: Available in JAX 0.4+ and cleaner; manual fallback is documented in plan but not needed.
- **Test fixtures use helper functions, not session-scoped pytest fixtures**: JAX network state is mutable after `input_data()`; each test gets a fresh network to prevent state leakage.
- **Response function casts `choices` to `jnp.asarray` internally**: Callers (both NumPy arrays from simulator and JAX traced arrays from PyMC) work without external conversion.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ruff import sort violation in test file**

- **Found during:** Task 2 (test implementation)
- **Issue:** ruff reported `I001` (unsorted import block) in `tests/test_response.py` — `import jax.numpy as jnp` was placed after `import numpy as np` and `import pytest` violating isort order
- **Fix:** Used `ruff check --fix` to auto-sort imports; also removed unused `choices_b` variable (F841)
- **Files modified:** tests/test_response.py
- **Verification:** `ruff check tests/test_response.py` passes with no errors
- **Committed in:** `1c92df7` (Task 2 commit includes corrected file)

---

**Total deviations:** 1 auto-fixed (lint/style)
**Impact on plan:** Trivial style fix, no logic impact.

## Issues Encountered

- `conda run -n ds_env python -c "..."` fails with multi-line `-c` scripts on Windows (conda 25.7.0 assertion error on newlines in arguments). Workaround: write script to a temp file and run via `python script.py`. Temp file was removed after verification.

## Next Phase Readiness

- Response function complete and tested. Full model pipeline (config -> session -> forward pass -> surprise) verified end-to-end on both model variants.
- Phase 3 (simulation) can now generate synthetic participants: provide known parameters to `softmax_stickiness_surprise` to simulate choice sequences.
- Phase 4 (fitting) will wrap `net.surprise()` with this response function inside a custom PyMC Op. The JAX-native implementation ensures autodiff compatibility.
- No blockers for Phase 3.

---
*Phase: 02-models*
*Completed: 2026-04-05*
