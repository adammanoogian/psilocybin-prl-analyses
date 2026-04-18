---
phase: 19
plan: 02
subsystem: fitting
tags: [laplace, inference-data, arviz, participant-id, dim-name, packaging]

dependency-graph:
  requires: []
  provides:
    - src/prl_hgf/fitting/laplace_idata.py (build_idata_from_laplace)
    - tests/test_laplace_idata.py (11 tests)
  affects:
    - 19-03: fit_vb_laplace_patrl consumes build_idata_from_laplace
    - 19-04: SLURM wrapper consumes idata from 19-03
    - 19-05: Tapas comparison consumes idata from 19-03

tech-stack:
  added: []
  patterns:
    - Laplace pseudo-draw packaging via numpy multivariate_normal
    - cast(az.InferenceData, ...) pattern for mypy with arviz.from_dict
    - ravel_pytree flat layout: columns [i*P:(i+1)*P] = param_names[i]

key-files:
  created:
    - src/prl_hgf/fitting/laplace_idata.py
    - tests/test_laplace_idata.py
  modified: []

decisions:
  - id: participant_id-native
    choice: Emit dim 'participant_id' natively in build_idata_from_laplace
    rationale: >
      The NUTS path _samples_to_idata emits 'participant' (hierarchical.py line 1606)
      while export_subject_trajectories reads 'participant_id' (export_trajectories.py
      line 181). Phase 19 sidesteps this OQ1 bug by emitting the correct consumer-facing
      dim name without touching hierarchical.py (parallel-stack invariant).
  - id: cast-for-mypy
    choice: cast(az.InferenceData, az.from_dict(...)) to satisfy mypy
    rationale: az.from_dict stub returns Any in arviz typeshed; cast is zero-overhead

metrics:
  duration: 32 minutes
  completed: 2026-04-18
---

# Phase 19 Plan 02: Laplace InferenceData Factory Summary

**One-liner:** `build_idata_from_laplace(mode, cov, param_names, participant_ids)` packages
Laplace `(mode, Σ)` into `az.InferenceData` with dim `participant_id` and NUTS-parity
posterior schema, enabling direct consumption by `export_subject_parameters`.

## What Was Built

### `src/prl_hgf/fitting/laplace_idata.py`

Public function signature:

```python
def build_idata_from_laplace(
    mode: dict[str, np.ndarray],          # var -> shape (P,) native-space mode
    cov: np.ndarray,                      # (P*K, P*K) covariance in ravel_pytree order
    param_names: tuple[str, ...],         # _PARAM_ORDER_2LEVEL or _PARAM_ORDER_3LEVEL
    participant_ids: list[str],           # len P
    n_pseudo_draws: int = 1000,           # m6 resolution: default 1000
    rng_key: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> az.InferenceData:
```

Module-level constants:

```python
_PARAM_ORDER_2LEVEL: tuple[str, ...] = ("omega_2", "log_beta")
_PARAM_ORDER_3LEVEL: tuple[str, ...] = ("omega_2", "log_beta", "omega_3", "kappa", "mu3_0")
```

**Example posterior shape for 2-level P=5, n_pseudo_draws=1000:**

| Variable   | Shape        | Notes                          |
|------------|--------------|--------------------------------|
| omega_2    | (1, 1000, 5) | chain=1, draw=1000, P=5        |
| log_beta   | (1, 1000, 5) | optimizer-space                |
| beta       | (1, 1000, 5) | deterministic: exp(log_beta)   |

**For 3-level:** additionally `omega_3`, `kappa`, `mu3_0` each `(1, 1000, P)`.

### `tests/test_laplace_idata.py`

11 tests, all passing in < 15 seconds (pure numpy + ArviZ; no MCMC):

| # | Test | Checks |
|---|------|--------|
| 1 | `test_build_idata_2level_shape_contract` | (1,500,4) per var; participant_id coord |
| 2 | `test_build_idata_3level_shape_contract` | 6 vars present; (1,500,3) shapes |
| 3 | `test_deterministic_beta_is_exp_log_beta` | np.allclose to float32 tol |
| 4 | `test_dim_name_is_participant_id_not_participant` | OQ1 guard |
| 5 | `test_az_hdi_works_on_single_chain` | ArviZ 0.22+ lower/higher coords |
| 6 | `test_consumer_compatibility_export_subject_parameters` | Integration: valid CSV |
| 7 | `test_validates_param_names_mismatch` | ValueError with expected vs actual |
| 8 | `test_validates_cov_shape_mismatch` | ValueError with expected vs actual |
| 9 | `test_sample_stats_group_present` | All 7 canonical keys round-trip |
| 10 | `test_pick_best_cue_regression_unchanged` | bms.compute_subject_waic importable |
| 11 | `test_validates_mode_shape_mismatch` | ndim=2 AND shape[0]!=P cases |

## Key Design Choices

### Flat covariance layout (ravel_pytree parity)

The flat layout for a dict `{v1: (P,), v2: (P,), ...}` is:
```
[v1_0, v1_1, ..., v1_{P-1}, v2_0, v2_1, ..., v2_{P-1}, ...]
```
So column slice `[i*P : (i+1)*P]` = `param_names[i]` across all participants.
This matches `jax.flatten_util.ravel_pytree` for sorted-key dicts with `(P,)` values,
enabling Plan 19-03's optimizer to reuse the same unravel function.

### OQ1 dim-name divergence (participant_id vs participant)

The NUTS path `hierarchical.py::_samples_to_idata` emits dim name `"participant"` (line 1606).
The consumer `export_subject_trajectories` reads `"participant_id"` (line 181, 345).
This is a latent producer/consumer mismatch — the NUTS path currently only works because
`export_trajectories.py` tests use manually constructed idata with `"participant_id"`.

**Phase 19 Laplace path**: emits `"participant_id"` natively (correct), sidestepping the bug
without modifying `hierarchical.py` (parallel-stack invariant).

**Follow-up required**: The NUTS path `_samples_to_idata` dim name should be hotfixed in a
future phase. Tracked in STATE.md pending todos.

### export_subject_parameters consumer contract confirmed

Test 6 (`test_consumer_compatibility_export_subject_parameters`) runs the actual
`export_subject_parameters` function on Laplace idata and verifies:
- CSV has correct columns: `[participant_id, parameter, posterior_mean, hdi_low, hdi_high]`
- 2 params × 4 participants = 8 rows, all non-NaN
- `az.hdi(post[params], hdi_prob=0.94)` works with ArviZ 0.22 `lower`/`higher` coords

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] mypy error: az.from_dict returns Any**

- **Found during:** Task 1 ruff/mypy verification
- **Issue:** `az.from_dict` return type is `Any` in arviz typeshed; mypy flags
  `error: Returning Any from function declared to return "InferenceData"`
- **Fix:** Added `cast(az.InferenceData, az.from_dict(...))` and imported `cast` from `typing`
- **Files modified:** `src/prl_hgf/fitting/laplace_idata.py`
- **Commit:** a3eaa5a

**2. [Rule 1 - Bug] ruff F841: unused variable K in test_validates_cov_shape_mismatch**

- **Found during:** Task 2 ruff check
- **Issue:** `K = len(_PARAM_ORDER_2LEVEL)` assigned but not used in the test body
  (cov dimension was computed inline as `np.eye(7)` for the wrong-shape case)
- **Fix:** Removed the unused `K` assignment
- **Files modified:** `tests/test_laplace_idata.py`
- **Commit:** 473c720

## Verification Results

- `pytest tests/test_laplace_idata.py -v`: **11/11 passed** (12.7s)
- `pytest tests/test_export_trajectories.py -v`: **10/10 passed** (27.8s)
- `pytest tests/test_bms.py tests/test_env_simulator.py tests/test_response.py -v`:
  **31/31 passed** (86s)
- `ruff check src/prl_hgf/fitting/laplace_idata.py tests/test_laplace_idata.py`: **clean**
- `mypy src/prl_hgf/fitting/laplace_idata.py`: **clean**
- `git diff` protected paths: **empty** (parallel-stack invariant holds)
- `fitting/__init__.py`: **unchanged**

## Next Phase Readiness

Plan 19-03 (`fit_vb_laplace_patrl`) can immediately use `build_idata_from_laplace` as its
packaging step. The input contract is:
- `mode: dict[str, np.ndarray]` — keys matching `_PARAM_ORDER_2LEVEL` or `_PARAM_ORDER_3LEVEL`
- `cov: np.ndarray` — shape `(P*K, P*K)` in ravel_pytree column order
- `participant_ids: list[str]` — matching the fitting cohort
- `diagnostics: dict` — 7 canonical keys (all tested in Test 9)

The resulting idata is directly consumable by `export_subject_trajectories` and
`export_subject_parameters` without rename shims.
