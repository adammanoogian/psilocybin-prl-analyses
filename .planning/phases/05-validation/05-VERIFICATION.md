---
phase: 05-validation
verified: 2026-04-06T13:57:09Z
status: passed
score: 8/8 must-haves verified
---

# Phase 5: Validation & Comparison Verification Report

**Phase Goal:** Parameter recovery is verified (or limitations documented), and formal model comparison identifies whether the 3-level model is justified by the data.
**Verified:** 2026-04-06T13:57:09Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Recovery metrics (r, p, bias, RMSE) are computed for each parameter | VERIFIED | compute_recovery_metrics returns DataFrame with all 7 required columns; R_THRESHOLD=0.7 applied |
| 2 | Scatter plots show true vs recovered values with identity line and r annotation | VERIFIED | plot_recovery_scatter draws identity line, r/p annotation, threshold-based title color |
| 3 | Correlation matrix reveals inter-parameter confounds | VERIFIED | compute_correlation_matrix returns Pearson corr of posterior means; abs(r)>0.8 flagged |
| 4 | Flagged fits are excluded from primary recovery; count is reported | VERIFIED | build_recovery_df excludes flagged==True rows and logs count at INFO level |
| 5 | omega_3 recovery quality is explicitly documented with warning annotation | VERIFIED | plots.py line 26: _OMEGA3_CAVEAT constant; italic annotation if r < threshold; pipeline prints NOTE |
| 6 | Recovery raises ValueError if fewer than min_n (default 30) remain | VERIFIED | recovery.py lines 151-156: ValueError with message naming min_n and actual count |
| 7 | BMS produces exceedance probabilities for 2-level vs 3-level, full and per-group | VERIFIED | run_stratified_bms loops over full sample and each unique group; EP bar plot renders EP + PEP |
| 8 | Model comparison summary table and exceedance probability bar plot are generated | VERIFIED | Pipeline saves bms_summary.csv (columns: group, model, exp_r, xp, pxp, bor) and bms_exceedance.png |

**Score:** 8/8 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/prl_hgf/analysis/recovery.py | build_recovery_df, compute_recovery_metrics, compute_correlation_matrix | VERIFIED | 279 lines; all 3 functions in __all__; NaN masking, flagging, min_n guard implemented |
| src/prl_hgf/analysis/plots.py | plot_recovery_scatter, plot_correlation_matrix | VERIFIED | 207 lines; Agg backend set at module top; both functions return Figure; omega_3 caveat at line 115 |
| src/prl_hgf/analysis/bms.py | compute_subject_waic, compute_batch_waic, run_group_bms, run_stratified_bms, plot_exceedance_probabilities | VERIFIED | 556 lines; all 5 in __all__; groupBMC wired; WAIC post-hoc workaround for pm.Potential |
| src/prl_hgf/analysis/__init__.py | Exports recovery and BMS functions | PARTIAL | Exports 8 recovery/BMS functions; plot_recovery_scatter and plot_correlation_matrix NOT re-exported here (non-blocking: all callers import from prl_hgf.analysis.plots directly) |
| scripts/05_run_validation.py | End-to-end pipeline script | VERIFIED | 638 lines; CLI args; full orchestration; all output paths under VALIDATION_DIR |
| src/prl_hgf/fitting/batch.py | return_idata parameter support | VERIFIED | return_idata: bool = False at line 164; idata_dict returned at line 349 |
| tests/test_recovery.py | Unit tests for recovery analysis | VERIFIED | 261 lines; 8 named tests: shape, flagging, min_n guard, columns, corr matrix, plot smoke tests |
| tests/test_bms.py | Unit tests for BMS | VERIFIED | 205 lines; 5 named tests: import, synthetic win, shape, WAIC smoke (@slow), EP plot smoke |
| tests/test_validation_integration.py | Integration tests | VERIFIED | 405 lines; 6 @pytest.mark.integration tests including CSV output validation for CMP-04 |
| config.py | RESULTS_DIR and VALIDATION_DIR path constants | VERIFIED | Lines 20-21: both defined; pipeline imports both at line 59 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| recovery.py | sim_df true_* columns | groupby(...).first().reset_index() | WIRED | Lines 111-113 |
| recovery.py | fit_df parameter column | pivot_table(index=_ID_COLS, columns=parameter, values=mean) | WIRED | Lines 132-138 |
| plots.py | recovery.py outputs | Consumes recovery_df and metrics_df | WIRED | plot_recovery_scatter signature; _TRUE_MAP dict |
| bms.py | prl_hgf.fitting.ops | build_logp_ops_2level / build_logp_ops_3level | WIRED | Line 103: lazy import inside compute_subject_waic |
| bms.py | groupBMC | GroupBMC(L).get_result() | WIRED | Line 42: import; line 342: GroupBMC(L) with L transposed to (n_models, n_subjects) |
| bms.py | arviz | az.waic(idata, var_name=loglike) | WIRED | Line 166: call after injecting xr.DataArray into idata log_likelihood group |
| scripts/05_run_validation.py | recovery.py | build_recovery_df + compute_recovery_metrics | WIRED | Lines 319, 334 in _run_recovery() |
| scripts/05_run_validation.py | bms.py | compute_batch_waic + run_stratified_bms | WIRED | Lines 569, 413 in main() |
| scripts/05_run_validation.py | plots.py | plot_recovery_scatter + plot_correlation_matrix | WIRED | Lines 388, 394 in _save_recovery_outputs() |
| scripts/05_run_validation.py | results/validation/ | VALIDATION_DIR paths for all output files | WIRED | Lines 382-396, 444, 469, 574 |
| fit_batch | idata return | return (results_df, idata_dict) when return_idata=True | WIRED | Lines 296-297, 316-317, 349-351 |

---

### Requirements Coverage

| Requirement | Description | Status | Supporting Evidence |
|-------------|-------------|--------|---------------------|
| REC-01 | Scatter plots of true vs recovered with Pearson r and bias metrics | SATISFIED | plot_recovery_scatter + compute_recovery_metrics |
| REC-02 | r > 0.7 threshold for omega_2, beta, zeta; flag omega_3 and kappa explicitly | SATISFIED | passes_threshold column; omega_3 caveat annotation in plots.py and pipeline |
| REC-03 | Confusion/correlation matrix between parameters | SATISFIED | compute_correlation_matrix + plot_correlation_matrix with abs(r)>0.8 concern flag |
| REC-04 | N >= 30 participants guard | SATISFIED | min_n=30 default; ValueError raised with message naming expected and actual count |
| CMP-01 | Per-subject WAIC via ArviZ | SATISFIED | compute_subject_waic re-evaluates JAX logp post-hoc and calls az.waic |
| CMP-02 | Random-effects BMS (Rigoux 2014): exceedance probability | SATISFIED | run_group_bms wraps groupBMC 1.0 |
| CMP-03 | BMS stratified by group; protected exceedance probability | SATISFIED | run_stratified_bms per-group BMS; pxp from result.protected_exceedance_probability |
| CMP-04 | Model comparison summary table and visualization | SATISFIED | bms_summary.csv with group/model/exp_r/xp/pxp/bor; bms_exceedance.png |

---

### Anti-Patterns Found

No stub patterns, TODO/FIXME comments, placeholder content, empty returns, or console-log-only handlers found in any production files.

---

### Minor Gaps (Non-Blocking)

1. **plots.py not re-exported from __init__.py**: Plan 05-01 specified updating __init__.py to import plot functions. They are absent from __init__.py __all__. All callers import directly from prl_hgf.analysis.plots - no functionality broken. Discoverability gap only.

2. **05-03-SUMMARY.md absent**: Plan 03 was executed (all three files it specifies exist and are substantive) but the summary file was not written. Does not affect goal achievement.

---

### Human Verification Required

| Test | Expected | Why human |
|------|----------|-----------|
| Run python scripts/05_run_validation.py --skip-waic against real batch sim + fit CSVs | recovery_metrics_hgf_2level.csv shows passes_threshold=True for omega_2, beta, zeta | Requires Phase 3 and Phase 4 CSV outputs not in repo |
| Run full pipeline without --skip-waic against 3-level simulated data | bms_summary.csv shows xp for hgf_3level > 0.5 in the all group | Requires InferenceData objects from full MCMC run |
| Inspect results/validation/bms_exceedance.png after a full run | Grouped bars with EP (blue) and PEP (orange), chance line, BOR annotation | Visual rendering cannot be verified structurally |

---

## Verification Summary

All 8 observable truths verified structurally. All 11 key links wired. All 8 requirements (REC-01 through REC-04, CMP-01 through CMP-04) satisfied. No blocker anti-patterns. Two non-blocking gaps (plots.py not re-exported from __init__.__all__, missing 05-03-SUMMARY.md) do not affect goal achievement. Three human verification items are expected follow-on steps requiring Phase 3/4 run outputs - all structural pre-conditions are in place.

**Phase 5 goal is structurally achieved.**

---

_Verified: 2026-04-06T13:57:09Z_
_Verifier: Claude (gsd-verifier)_
