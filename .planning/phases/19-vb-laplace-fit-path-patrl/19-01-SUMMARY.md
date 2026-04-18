---
phase: 19-vb-laplace-fit-path-patrl
plan: "01"
subsystem: env/simulation
tags:
  - pat-rl
  - simulator
  - refactor
  - extraction
  - seed-determinism

dependency-graph:
  requires:
    - "18-01: PATRLConfig + load_pat_rl_config"
    - "18-02: generate_session_patrl + PATRLTrial"
    - "18-03: build_2level_network_patrl + build_3level_network_patrl"
  provides:
    - "prl_hgf.env.pat_rl_simulator: simulate_patrl_cohort, run_hgf_forward_patrl"
    - "tests/test_pat_rl_simulator.py: 6 unit tests"
  affects:
    - "19-05: Laplace smoke imports simulate_patrl_cohort for --fit-method laplace"
    - "19-06: VBL-06 Laplace-vs-NUTS comparison harness imports simulate_patrl_cohort"

tech-stack:
  added: []
  patterns:
    - "Extraction refactor: script-private helpers promoted to importable module"
    - "Seed-deterministic cohort simulation via SeedSequence"

key-files:
  created:
    - src/prl_hgf/env/pat_rl_simulator.py
    - tests/test_pat_rl_simulator.py
  modified:
    - scripts/12_smoke_patrl_foundation.py

decisions:
  - id: no-expected-value-jax-import
    decision: "simulate_patrl_cohort uses pure NumPy for EV/choice computation; does not import JAX expected_value from response_patrl"
    rationale: "The original _simulate_cohort used inline NumPy math (not JAX). Importing expected_value would introduce a JAX call inside the cohort simulation loop, changing behavior and breaking numpy-only simulation path. The plan listed the import as documentation of the relationship, not a code requirement."
  - id: run-hgf-forward-patrl-not-imported-in-script
    decision: "scripts/12 imports only simulate_patrl_cohort (not run_hgf_forward_patrl); ruff F401 would have flagged an unused import"
    rationale: "run_hgf_forward_patrl is called inside simulate_patrl_cohort; the script has no separate call site for it. The public API is complete; downstream Laplace smoke and VBL-06 can import run_hgf_forward_patrl directly from prl_hgf.env.pat_rl_simulator."

metrics:
  duration: "~43 minutes"
  completed: "2026-04-18"
  tasks-completed: 3/3
  tests-added: 6
  tests-passing: 6/6
  regressions: 0 (1 pre-existing blackjax failure tolerated)

commits:
  - hash: e59150c
    message: "feat(19-01): create pat_rl_simulator module with extracted helpers"
    task: 1
  - hash: c481f91
    message: "feat(19-01): refactor scripts/12 to import from pat_rl_simulator"
    task: 2
  - hash: 6362824
    message: "test(19-01): add 6 unit tests for pat_rl_simulator + regression guards"
    task: 3
---

# Phase 19 Plan 01: PAT-RL Simulator Extraction Summary

**One-liner:** Extracted `simulate_patrl_cohort` + `run_hgf_forward_patrl` from `scripts/12` into `prl_hgf.env.pat_rl_simulator` for seed-deterministic cohort reuse across NUTS and Laplace smoke paths.

## Public API

### `run_hgf_forward_patrl`

```python
def run_hgf_forward_patrl(
    trials: list[PATRLTrial],
    omega_2: float,
    level: int,
    omega_3: float = -6.0,
    kappa: float = 1.0,
    mu3_0: float = 1.0,
) -> np.ndarray:
    """Returns shape (n_trials,) float64 mu2 trajectory."""
```

### `simulate_patrl_cohort`

```python
def simulate_patrl_cohort(
    n_participants: int,
    level: int,
    master_seed: int,
    config: PATRLConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, list[PATRLTrial]]]:
    """Returns (sim_df, true_params_by_participant, trials_by_participant)."""
```

## Line-count delta on scripts/12

- Before refactor: 700 lines
- After refactor: 503 lines
- Delta: **-197 lines** (plan estimated -140; actual savings larger due to removed
  `generate_session_patrl`, `build_2level_network_patrl`, `build_3level_network_patrl`
  imports that were also no longer needed in the script)

## Parallel-stack invariant

```
git diff --name-only src/prl_hgf/fitting/hierarchical.py \
  src/prl_hgf/fitting/hierarchical_patrl.py \
  src/prl_hgf/env/task_config.py \
  src/prl_hgf/env/simulator.py \
  src/prl_hgf/models/hgf_2level.py \
  src/prl_hgf/models/hgf_3level.py \
  src/prl_hgf/models/hgf_2level_patrl.py \
  src/prl_hgf/models/hgf_3level_patrl.py \
  src/prl_hgf/models/response.py \
  src/prl_hgf/models/response_patrl.py \
  configs/prl_analysis.yaml configs/pat_rl.yaml
```
Output: empty (no protected files touched).

## Test results

| Test | Result |
|------|--------|
| test_simulate_patrl_cohort_shape_contract | PASSED |
| test_simulate_patrl_cohort_seed_determinism | PASSED |
| test_simulate_patrl_cohort_different_seeds_differ | PASSED |
| test_run_hgf_forward_patrl_2level_shape | PASSED |
| test_run_hgf_forward_patrl_3level_shape | PASSED |
| test_pick_best_cue_regression_unchanged | PASSED |
| test_smoke_patrl_foundation.py (5 structural) | 7 PASSED (7 including parametrized) |
| pick_best_cue suite (88 tests) | 88 PASSED, 1 FAILED (pre-existing blackjax) |
| PAT-RL suite (26 tests) | 24 PASSED, 2 SKIPPED (blackjax-dependent) |

All 6 new tests run in ~19.6 seconds on a laptop CPU (well under the 60s budget).

## Behavioral differences before vs after refactor

None. The extraction is bit-for-bit identical:
- `simulate_patrl_cohort` seed determinism test passes: same `master_seed` produces
  identical `sim_df`, `true_params`, and `trials_by_participant` across two calls.
- RNG-consumption order is preserved exactly: `SeedSequence`, `spawn`, `default_rng`
  sequence unchanged from original `_simulate_cohort`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Did not import `expected_value` from response_patrl**

- **Found during:** Task 1
- **Issue:** The plan's imports section listed `from prl_hgf.models.response_patrl import expected_value`, but the actual `_simulate_cohort` body uses pure NumPy for EV computation (not JAX). Importing `expected_value` would have been unused and caused an F401 ruff error.
- **Fix:** Omitted the `expected_value` import; kept pure-NumPy EV calculation as in original.
- **Impact:** None — behavior identical to original.

**2. [Rule 1 - Bug] Removed `run_hgf_forward_patrl` from scripts/12 import**

- **Found during:** Task 2
- **Issue:** After extraction, `scripts/12` calls only `simulate_patrl_cohort` directly; `run_hgf_forward_patrl` is called internally by `simulate_patrl_cohort`. Importing it in the script would trigger ruff F401 (unused import).
- **Fix:** Imported only `simulate_patrl_cohort` in the script.
- **Impact:** None — `run_hgf_forward_patrl` is still public and importable by Laplace/VBL-06 consumers.

**3. [Rule 1 - Bug] Ruff I001 (import sort) due to mid-block comment**

- **Found during:** Task 2
- **Issue:** Comment placed inside the import block broke isort ordering.
- **Fix:** Ran `ruff check --fix`; ruff moved comment above the import line.
- **Impact:** None — code behavior unchanged.

## Next Phase Readiness

Phase 19 Plan 02 (Laplace InferenceData factory) is confirmed complete by concurrent
wave 1 execution (commit `a3eaa5a`). Plan 19-01 and 19-02 together enable Plan 19-03
(Laplace optimizer) and Plan 19-05 (Laplace smoke via `scripts/12 --fit-method laplace`).

`simulate_patrl_cohort` is now importable by any downstream plan without script-level
`sys.path` manipulation — just `from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort`.
