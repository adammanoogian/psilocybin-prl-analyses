---
phase: 33-ts1-m2-w4-fused
plan: 04
subsystem: fitting-preflight
tags: [collinearity, preflight, mode-b, covariate]
dependency_graph:
  requires: ["33-03"]
  provides: ["collinearity-gate", "P8-prevention"]
  affects: ["33-05", "33-06"]
tech_stack:
  added: []
  patterns: ["pre-flight validation gate", "O(P) cheap check before O(P*T) work"]
key_files:
  created:
    - tests/unit/test_mode_b_preflight.py
  modified:
    - src/prl_hgf/fitting/preflight.py
    - src/prl_hgf/fitting/hierarchical.py
decisions:
  - id: "33-04-01"
    title: "Threshold 0.7 for collinearity refusal"
    rationale: "Standard VIF-based cutoff; |r|=0.7 implies R^2=0.49 shared variance"
metrics:
  duration: "~5 min"
  completed: "2026-05-18"
---

# Phase 33 Plan 04: Pre-flight Collinearity Check Summary

**One-liner:** O(P) Pearson correlation gate refuses Mode B fit when |cor(x_covariate, group_idx)| > 0.7 with group-mean-centering remediation advice.

## What Was Done

### Task 1: Implement collinearity check in preflight.py and integrate in hierarchical.py

Added `check_covariate_collinearity()` to `src/prl_hgf/fitting/preflight.py`:
- Computes |Pearson r| between covariate and group index
- Raises `ValueError` when |r| > threshold (default 0.7)
- Error message includes: actual |r|, threshold, explanation of unidentifiability, and remediation (group-mean-centering)
- O(P) computation — runs before any expensive O(P*T) array stacking

Integrated in `fit_batch_hierarchical` (hierarchical.py):
- Called after x_covariate shape validation
- Called before mean-centering and np.stack array assembly
- Only fires when x_covariate is not None (Mode B with covariate)

**Commit:** `8fa1e65`

### Task 2: Unit tests for collinearity check

Created `tests/unit/test_mode_b_preflight.py` with 7 tests:
1. `test_passes_orthogonal_covariate` — no raise on random x vs balanced groups
2. `test_raises_collinear_covariate` — raises on near-perfect correlation
3. `test_error_message_includes_actual_r` — verifies |r| value in message
4. `test_error_message_includes_remediation` — verifies "group-mean-centered" text
5. `test_custom_threshold` — lower threshold catches moderate correlation
6. `test_none_covariate_not_accepted` — function requires arrays
7. `test_expected_vs_actual_in_message` — verifies "Expected |r| < 0.7, got X.XXX"

All tests pass locally in <1s (pure numpy, no JAX/cluster needed).

**Commit:** `db21391`

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 33-04-01 | Threshold 0.7 | Standard collinearity cutoff; R^2=0.49 means nearly half the variance is shared, making beta_p and mu_g practically unidentifiable |
| 33-04-02 | numpy at module level in preflight.py | Already an implicit dependency (via other callers); explicit import is cleaner than local import inside function |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

1. `check_covariate_collinearity` raises ValueError with |r| > 0.7 -- PASS
2. Error message includes actual correlation value, threshold, and remediation -- PASS
3. Unit tests pass locally (pure numpy, <1s) -- PASS (7/7)
4. Integration point in hierarchical.py fires BEFORE array assembly -- PASS
5. ruff passes on all modified files -- PASS

## Next Phase Readiness

Plan 33-05 (simulation-based recovery validation) can proceed. The collinearity check is now in place to prevent confounded fits during recovery testing.
