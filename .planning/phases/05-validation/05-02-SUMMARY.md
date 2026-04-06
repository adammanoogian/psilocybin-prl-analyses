---
phase: 05-validation
plan: 02
subsystem: analysis
tags: [bayesian-model-comparison, waic, groupBMC, arviz, exceedance-probability, pm.Potential]

# Dependency graph
requires:
  - phase: 04-fitting
    provides: InferenceData objects with posterior samples from PyMC NUTS fitting
  - phase: 03-simulation
    provides: trial-level sim_df with reward_c0/c1/c2, observed_c0/c1/c2, choice columns
  - 05-01
    provides: analysis package structure (recovery.py, plots.py, __init__.py)
provides:
  - compute_subject_waic: post-hoc WAIC via JAX logp re-evaluation over posterior samples
  - compute_batch_waic: batch WAIC computation for all participants x models
  - run_group_bms: random-effects BMS via groupBMC (Rigoux et al. 2014)
  - run_stratified_bms: full-sample + per-group BMS
  - plot_exceedance_probabilities: multi-panel EP + protected EP bar chart
  - Unit tests validating all public functions
affects:
  - 05-03 (pipeline script calling these functions on batch fit output)
  - 07-gui (EP plots in interactive dashboard)

# Tech tracking
tech-stack:
  added: [groupBMC==1.0 (Rigoux 2014 VB Dirichlet-Multinomial algorithm)]
  patterns:
    - Post-hoc WAIC via idata.add_groups({"log_likelihood": ...}) after re-evaluating JAX logp
    - GroupBMC(L) with L shape (n_models, n_subjects) — transposed from internal (n_subjects, n_models)
    - bor extracted from bmc.F1()-bmc.F0() free energy difference (not directly on GroupBMCResult)
    - GroupBMCResult exposes: exceedance_probability, protected_exceedance_probability, frequency_mean
    - Dataset.sizes (not .dims) for xarray compatibility with future versions

key-files:
  created:
    - src/prl_hgf/analysis/bms.py
    - tests/test_bms.py
  modified:
    - src/prl_hgf/analysis/__init__.py
    - pyproject.toml

# Decisions made
decisions:
  - decision: groupBMC 1.0 package installed (not from-scratch VB fallback)
    rationale: "pip install groupBMC succeeded; package implements Rigoux et al. 2014 VB algorithm exactly"
    phase: 05-02
  - decision: L matrix transposed before GroupBMC call
    rationale: "groupBMC expects (n_models, n_subjects); internal representation uses (n_subjects, n_models)"
    phase: 05-02
  - decision: bor computed from bmc.F1()-bmc.F0() not GroupBMCResult attribute
    rationale: "GroupBMCResult does not expose bor directly; GroupBMC.get_result() computes it internally but does not store it on the result object"
    phase: 05-02
  - decision: Dataset.sizes used instead of Dataset.dims
    rationale: "xarray FutureWarning: dims will return a set in future; sizes always returns a mapping"
    phase: 05-02
  - decision: WAIC loglike_dim_0 is a single scalar per sample (not per-trial)
    rationale: "pm.Potential logp computes the full trial-sum; ArviZ warns about this (expected behavior for this architecture)"
    phase: 05-02

# Metrics
metrics:
  duration: 10m
  completed: "2026-04-06"
  tasks_total: 2
  tasks_completed: 2
---

# Phase 5 Plan 02: Bayesian Model Comparison Module Summary

**One-liner:** groupBMC random-effects BMS with post-hoc WAIC via JAX logp re-evaluation over InferenceData posterior samples, bypassing pm.Potential log_likelihood limitation.

## What Was Built

### `src/prl_hgf/analysis/bms.py`

Five public functions implementing the full BMS pipeline:

1. **`compute_subject_waic`** — The core workaround for pm.Potential WAIC. Re-evaluates the JAX JIT-compiled logp Op over every (chain, draw) posterior sample, builds an `xr.DataArray` of shape `(chains, draws, 1)`, injects it into `idata.log_likelihood`, and calls `az.waic`. Returns `elpd_waic` (scalar float; higher = better model fit).

2. **`compute_batch_waic`** — Iterates over all `(participant_id, group, session)` combinations and models. Reconstructs `input_data_arr`, `observed_arr`, `choices_arr` from `sim_df` trial-level columns, then calls `compute_subject_waic`. Returns a tidy DataFrame with columns `[participant_id, group, session, model, elpd_waic]`.

3. **`run_group_bms`** — Wraps `groupBMC.GroupBMC` with a standardized interface. Transposes the `(n_subjects, n_models)` elpd matrix to `(n_models, n_subjects)` before passing to `GroupBMC(L)`. Extracts `bor` from `bmc.F1()-bmc.F0()` (not on `GroupBMCResult`). Returns a dict with keys: `alpha`, `exp_r`, `xp`, `pxp`, `bor`, `model_names`, `group_label`, `n_subjects`.

4. **`run_stratified_bms`** — Builds per-group ELPD matrices by averaging over sessions, then runs `run_group_bms` for the full sample plus each unique group. Warns when N < 20 (limited statistical power).

5. **`plot_exceedance_probabilities`** — Multi-panel grouped bar chart: EP (blue) and protected EP (orange) for each model, per group label. Horizontal dashed chance line, BOR annotated below bars.

### `tests/test_bms.py`

Five tests, all passing:

| Test | What it checks |
|------|----------------|
| `test_groupbmc_import` | `groupBMC.GroupBMC` is importable |
| `test_run_group_bms_synthetic` | Model 1 wins when log-evidence is -100 vs -200; EP[0] > 0.5; exp_r sums to ~1 |
| `test_run_group_bms_shape` | Output arrays all have length n_models (tested with n_models=3) |
| `test_compute_subject_waic_smoke` | 2-level model, 5 trials, 2 chains x 10 draws; returns finite float |
| `test_plot_exceedance_probabilities_runs` | Returns `plt.Figure` with correct number of subplots |

## Key Technical Decisions

### groupBMC Package vs From-Scratch VB

**Path taken:** `pip install groupBMC` succeeded (version 1.0). The package implements the Rigoux et al. 2014 VB Dirichlet-Multinomial algorithm. No from-scratch fallback needed.

**Important API quirks discovered:**
- `GroupBMC(L)` requires `L` shape `(n_models, n_subjects)` — the TRANSPOSE of the standard (n_subjects, n_models) representation
- `GroupBMCResult` does NOT expose `bor` as an attribute; it is computed internally in `get_result()` from `F1()-F0()`. We replicate the same formula
- Import path is `from groupBMC.groupBMC import GroupBMC` (nested module)

### pm.Potential WAIC Workaround

The plan correctly identified that `pm.Potential` does not populate `idata.log_likelihood` during sampling. The solution:

1. Build the logp Op fresh with trial-level arrays (same as done in fitting)
2. Iterate over all (chain, draw) posterior samples, calling `logp_op(...).eval()` for each
3. Wrap results in `xr.DataArray(shape=(chains, draws, 1))` with correct coordinate labels
4. Call `idata.add_groups({"log_likelihood": {"loglike": ll_da}})`
5. Call `az.waic(idata, var_name="loglike")`

Note: ArviZ warns that "point-wise WAIC is the same with sum WAIC" — this is expected because our logp function computes the trial-level sum (a single scalar per sample). The WAIC value is still mathematically valid as a model evidence quantity.

## Deviations from Plan

### Auto-fixed Issues

**[Rule 1 - Bug] Use `Dataset.sizes` instead of `Dataset.dims` for chain/draw counts**

- **Found during:** Task 1 (FutureWarning during test run)
- **Issue:** `posterior.dims["chain"]` returns the dimension size now but will change to return a set in future xarray; `posterior.sizes["chain"]` is the correct way to get the integer size
- **Fix:** Changed both `.dims["chain"]` and `.dims["draw"]` to `.sizes[...]` in `compute_subject_waic`
- **Files modified:** `src/prl_hgf/analysis/bms.py`
- **Commit:** 76ebf69

None other — plan executed as written.

## Authentication Gates

None.

## Next Phase Readiness

05-03 (pipeline script) can proceed immediately.

**Inputs 05-03 needs:**
- `compute_batch_waic(sim_df, idata_dict)` — idata_dict must be populated by storing idata in the fitting pipeline (fit_participant currently returns a DataFrame; 05-03 will need to modify the pipeline to also save .nc files)
- `run_stratified_bms(waic_df, model_names)` — straightforward once waic_df is available
- `plot_exceedance_probabilities(bms_results)` — ready to call

**Constraint documented in bms.py docstring:** `compute_batch_waic` requires that the pipeline script stores InferenceData objects. If idata was not saved during fitting, the pipeline script must re-run fitting with idata storage. The `.nc` files do not store `input_data_arr`, `observed_arr`, `choices_arr` — these must be reconstructed from `sim_df`.
