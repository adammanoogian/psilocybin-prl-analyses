---
phase: 05-validation
plan: 03
subsystem: pipeline
tags: [pipeline, recovery, bms, waic, argparse, integration-test, matplotlib, pandas]

# Dependency graph
requires:
  - phase: 05-validation/05-01
    provides: build_recovery_df, compute_recovery_metrics, compute_correlation_matrix, plot_recovery_scatter, plot_correlation_matrix
  - phase: 05-validation/05-02
    provides: compute_batch_waic, run_stratified_bms, plot_exceedance_probabilities
  - phase: 04-fitting
    provides: fit_batch (batch MCMC results in FIT-04 schema)
provides:
  - scripts/05_run_validation.py: end-to-end validation pipeline script
  - fit_batch(return_idata=True): InferenceData collection for WAIC post-hoc
  - RESULTS_DIR / VALIDATION_DIR: path constants in config.py
  - tests/test_validation_integration.py: 6 integration tests
affects:
  - 06-analysis (group analysis consumes validated parameters)
  - 07-gui (validation outputs viewable in dashboard)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - return_idata=False default preserves backward compatibility; True triggers tuple return
    - Pipeline script uses sys.exit(1) on missing data with clear instructions
    - Integration tests use tmp_path pytest fixture for PNG output assertion

# File tracking
key-files:
  created:
    - scripts/05_run_validation.py
    - tests/test_validation_integration.py
  modified:
    - config.py (RESULTS_DIR, VALIDATION_DIR added)
    - src/prl_hgf/fitting/batch.py (return_idata parameter)

# Decisions
decisions:
  - "return_idata=False default preserves backward compatibility with existing 04_fit_participants.py script"
  - "Pipeline --skip-waic flag avoids slow WAIC path during recovery-only diagnostics"
  - "Integration test fixture uses 5 participants (4 unflagged) to satisfy Pearson r min n>=3 guard"
  - "compute_batch_waic called via idata_dict (not direct .nc loading) per BMS module design"
  - "bms_summary.csv has columns: group, model, exp_r, xp, pxp, bor (group added for stratified output)"

# Metrics
metrics:
  duration: "13 minutes"
  completed: "2026-04-06"
---

# Phase 5 Plan 3: Validation Pipeline Summary

End-to-end pipeline script with fit_batch InferenceData support, argparse CLI, and 6 integration tests for recovery + BMS on synthetic data.

## What Was Built

### scripts/05_run_validation.py

Phase 5 pipeline entry point that:
1. Loads simulated data (with multi-path fallback) and fit CSVs for both model variants
2. Runs recovery analysis per model: `build_recovery_df` (min_n=30) + `compute_recovery_metrics` + `compute_correlation_matrix`
3. Saves recovery CSVs (`recovery_metrics_{model}.csv`) and plots (`recovery_scatter_{model}.png`, `correlation_matrix_{model}.png`) to `results/validation/`
4. Computes WAIC via `compute_batch_waic` (routes through `idata_dict`, reconstructs trial arrays from sim_df — does NOT call `compute_subject_waic` directly on .nc files)
5. Falls back to re-fitting via `fit_batch(return_idata=True)` when .nc files are absent
6. Runs `run_stratified_bms` + `plot_exceedance_probabilities`, saves `bms_summary.csv` and `bms_exceedance.png`
7. Prints final summary (passing/failing params by r>=0.7, winning model per group, high-correlation concerns)

CLI flags: `--skip-waic`, `--sim-path`, `--fit-path-2level`, `--fit-path-3level`, `--idata-dir`

### src/prl_hgf/fitting/batch.py — return_idata

`fit_batch` gains an optional `return_idata: bool = False` parameter. When `True`:
- Returns `(DataFrame, dict[tuple, az.InferenceData])` where the dict is keyed by `(participant_id, group, session)`
- Failed fits store `None` in the dict (not silently dropped)
- Default `False` returns only the DataFrame as before (backward compatible)

### config.py

Added `RESULTS_DIR = PROJECT_ROOT / "results"` and `VALIDATION_DIR = RESULTS_DIR / "validation"`.

### tests/test_validation_integration.py

6 integration tests marked `@pytest.mark.integration`:
1. `test_recovery_pipeline_integration` — build_recovery_df (exclude_flagged=True) → 4 rows, metrics shape, corr square
2. `test_recovery_pipeline_with_flagged_included` — all 5 participants present
3. `test_bms_pipeline_integration` — run_group_bms result keys, xp sums to 1, n_subjects correct
4. `test_plot_generation_integration` — scatter + correlation matrix PNGs exist and non-empty
5. `test_plot_exceedance_integration` — EP bar plot PNG exists and non-empty
6. `test_csv_outputs_written` — bms_summary.csv has (group, model, exp_r, xp, pxp, bor); recovery_metrics CSV has (parameter, r, p, bias, rmse, n, passes_threshold) and is non-empty

## Verification Results

```
pytest tests/test_recovery.py tests/test_bms.py tests/test_validation_integration.py -v
→ 19 passed in 32s

ruff check src/prl_hgf/analysis/ scripts/05_run_validation.py
    tests/test_recovery.py tests/test_bms.py tests/test_validation_integration.py
→ All checks passed

python scripts/05_run_validation.py --help
→ Usage printed correctly
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Integration test fixture had too few participants for Pearson r**

- **Found during:** Task 2 first test run
- **Issue:** With 3 participants and 1 flagged, only 2 valid observations remained. `compute_recovery_metrics` requires n >= 3 per parameter for Pearson r. `plot_recovery_scatter` received an empty metrics_df, causing matplotlib to create 0 columns.
- **Fix:** Extended fixture from 3 → 5 participants (1 flagged = P005), giving 4 valid unflagged participants. Updated all count assertions accordingly.
- **Files modified:** `tests/test_validation_integration.py`
- **Commit:** cbca583

**2. [Rule 1 - Bug] Unused `numpy` import in pipeline script**

- **Found during:** Task 1 ruff check
- **Issue:** `import numpy as np` was included but not used directly (numpy used only inside function-local scopes).
- **Fix:** Removed top-level numpy import.
- **Files modified:** `scripts/05_run_validation.py`
- **Commit:** a8b5782

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| `return_idata=False` default | Backward compatible with 04_fit_participants.py which uses positional return | 
| `--skip-waic` flag | WAIC is slow (30-60 min); recovery-only diagnostics are much faster | 
| Integration test 5 participants | Pearson r requires n >= 3; 5 total with 1 flagged gives 4 valid observations | 
| `bms_summary.csv` includes `group` column | Stratified BMS produces per-group rows; group column needed to distinguish them | 
| compute_batch_waic routing | Correct API: pass sim_df + idata_dict, not direct per-participant .nc loading |

## Next Phase Readiness

Phase 5 plan 03 completes the validation phase (05-01 recovery + 05-02 BMS + 05-03 pipeline).  All three plans delivered.

Phase 6 (Group Analysis) can proceed once real MCMC fitting is complete:
- Run `04_fit_participants.py` on real data → produce fit CSVs
- Run `05_run_validation.py` on simulated data to verify recovery before real analysis
- Phase 6 consumes validated model + fit results for mixed-effects group × session analysis

Remaining concern: WAIC computation requires InferenceData objects (either .nc files saved during fitting, or re-fitting with `return_idata=True`). The pipeline handles both paths but re-fitting is slow (3-5 hours). Saving .nc files during Phase 4 fitting is recommended.
