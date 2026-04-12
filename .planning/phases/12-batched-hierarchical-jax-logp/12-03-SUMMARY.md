# Phase 12 Plan 03: Hierarchical PyMC Orchestrator Summary

**One-liner:** PyMC model factory with shape=(P,) IID priors and single-call cohort orchestrator via pmjax.sample_numpyro_nuts, with participant metadata coords on InferenceData.

## What Was Done

### Task 1: Add build_pymc_model_batched (07341ad)

Added `build_pymc_model_batched` to `src/prl_hgf/fitting/hierarchical.py`:

- Signature: `(input_data_arr, observed_arr, choices_arr, model_name, trial_mask) -> (model, var_names, n_participants)`
- Validates `input_data_arr.ndim == 3` and `model_name` membership
- Calls `build_logp_ops_batched` to get the vmapped logp Op
- Declares `shape=n_participants` IID priors matching v1.1 `models.py` byte-for-byte:
  - 2-level: `omega_2` (TruncatedNormal mu=-3, sigma=2, upper=0), `log_beta` (Normal mu=0, sigma=1.5), `beta` (Deterministic exp(log_beta)), `zeta` (Normal mu=0, sigma=2)
  - 3-level: adds `omega_3` (TruncatedNormal mu=-6, sigma=2, upper=0) and `kappa` (TruncatedNormal mu=1, sigma=0.5, lower=0.01, upper=2)
- Hooks Op via `pm.Potential("loglike", logp_op(...))`
- No hyperpriors, no partial pooling -- mathematically equivalent to P independent models

### Task 2: Add fit_batch_hierarchical and update exports (6e22442)

Added `fit_batch_hierarchical` orchestrator to `hierarchical.py`:

- Groups `sim_df` by `(participant_id, group, session)` preserving order
- Builds per-participant arrays via private `_build_arrays_single` helper (mirrors `legacy/batch.py::_build_arrays`)
- Stacks into `(P, n_trials, 3)` shape with trial-count equality guard
- Calls `build_pymc_model_batched` to construct the joint model
- Runs ONE `pmjax.sample_numpyro_nuts` call for the full cohort (NOT `pm.sample(nuts_sampler='numpyro')` which hits the `_init_jitter` PyTensor read-only-array bug)
- Falls back to `pm.sample` when `sampler="pymc"` for CPU-only environments
- Post-hoc renames the PyMC-generated dimension to `"participant"` and assigns `participant_id`, `participant_group`, `participant_session` coords

Updated `src/prl_hgf/fitting/__init__.py`:

- Added `from prl_hgf.fitting.hierarchical import (build_logp_ops_batched, build_pymc_model_batched, fit_batch_hierarchical)`
- Extended `__all__` with three new v1.2 exports
- Legacy import surface unchanged -- all existing imports resolve

## Design Decisions

### Participant dim labeling mechanism

PyMC assigns a default dimension name `"{var_name}_dim_0"` when `shape=` is used without `dims=`. The orchestrator detects this post-hoc by inspecting the `idata.posterior` dimensions, filtering out `"chain"` and `"draw"`, and renames the remaining dimension to `"participant"`. Additional coords (`participant_group`, `participant_session`) are attached along the same dimension for downstream metadata lookup. This approach was chosen over `pm.Model(coords=...)` + `dims=` because the `dims` parameter interacts unpredictably with `pm.Potential` in some PyMC versions.

### pmjax-specific quirks

The `pmjax.sample_numpyro_nuts()` call must be made directly (not through `pm.sample(nuts_sampler='numpyro')`) because the latter path hits a `_init_jitter` PyTensor read-only-array bug that corrupts the initial point. This is documented in STATE.md as a known blocker.

### Prior specification byte-identity confirmation

Prior specs in `build_pymc_model_batched` match `src/prl_hgf/fitting/models.py` exactly:
- `omega_2`: TruncatedNormal(mu=-3.0, sigma=2.0, upper=0.0) -- identical
- `omega_3`: TruncatedNormal(mu=-6.0, sigma=2.0, upper=0.0) -- identical
- `kappa`: TruncatedNormal(mu=1.0, sigma=0.5, lower=0.01, upper=2.0) -- identical
- `log_beta`: Normal(mu=0.0, sigma=1.5) -- identical
- `beta`: Deterministic(exp(log_beta)) -- identical
- `zeta`: Normal(mu=0.0, sigma=2.0) -- identical

The only difference is `shape=n_participants` (v1.1 uses scalar priors).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] TYPE_CHECKING import for pd/az type annotations**

- **Found during:** Task 2 ruff check
- **Issue:** ruff flagged `"pd.DataFrame"` and `"az.InferenceData"` as UP037 (unnecessary quoted annotations) and F821 (undefined names) since `pd` and `az` were not importable at the module level
- **Fix:** Added `TYPE_CHECKING` guard with `import arviz as az; import pandas as pd` for type-checking only; removed runtime `import arviz as az; import pandas as pd` from function body since neither is used directly at runtime
- **Files modified:** `src/prl_hgf/fitting/hierarchical.py`

## Pre-existing Test Failure

`test_omega2_positive_returns_neginf` in `tests/test_fitting.py` fails with `-49.53` instead of `-inf`. This is a pre-existing issue (confirmed by running the test on the commit before this plan's changes). The Layer 2 clamping from Plan 12-02 changed behavior such that omega_2 >= 0 no longer always produces NaN -- the clamping can keep things stable in some edge cases. This test exercises the legacy single-participant ops, not the new batched code. Not a regression from this plan.

## Verification Results

- `from prl_hgf.fitting import build_logp_ops_batched, build_pymc_model_batched, fit_batch_hierarchical` -- PASS
- `from prl_hgf.fitting import fit_batch, fit_participant, build_logp_ops_3level` -- PASS (legacy regression)
- `pytest tests/test_fitting.py -v -k "not slow"` -- 6/7 pass, 1 pre-existing failure
- `ruff check` -- clean on both modified files
- `grep pmjax.sample_numpyro_nuts` -- confirmed direct call at line 905

## Files Modified

| File | Action | Key Changes |
|------|--------|-------------|
| `src/prl_hgf/fitting/hierarchical.py` | Modified | Added `build_pymc_model_batched`, `fit_batch_hierarchical`, `_build_arrays_single`; TYPE_CHECKING imports |
| `src/prl_hgf/fitting/__init__.py` | Modified | Re-export 3 new symbols; extended `__all__` |

## Metrics

- **Duration:** ~10 minutes
- **Completed:** 2026-04-12
- **Tasks:** 2/2
