---
phase: 14-integration-gpu-benchmark
plan: 01
subsystem: fitting
tags: [arviz, xarray, InferenceData, fit_batch_hierarchical, JAX, power, MCMC, integration]

# Dependency graph
requires:
  - phase: 12-batched-hierarchical-jax-logp
    provides: fit_batch_hierarchical returning az.InferenceData with participant dim
  - phase: 13-jax-native-cohort-simulation
    provides: simulate_batch using JAX vmap internally
provides:
  - run_sbf_iteration with dual-path: batched hierarchical (default) or legacy
  - _idata_to_fit_df: batched InferenceData to legacy 12-column fit_df schema
  - _split_idata: joint InferenceData to single-participant InferenceData
  - _build_idata_dict: joint InferenceData to per-(pid,grp,sess) dict for WAIC
  - --legacy flag on 08_run_power_iteration.py CLI
  - 4 new tests (VALID-05 suite)
affects:
  - 14-02 (GPU benchmark timing uses run_sbf_iteration)
  - 14-03 (VALID-03 cross-platform consistency uses same pipeline)
  - 15 (group analysis ingests power sweep results from batched path)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-path function: use_legacy=False (default) routes to new code; use_legacy=True preserves old code verbatim"
    - "Deferred import inside else block: `from prl_hgf.fitting.hierarchical import fit_batch_hierarchical` avoids heavy JAX imports on legacy path"
    - "az.rhat(da)[param].values for scalar extraction from Dataset returned by az.rhat"
    - "az.hdi(da, hdi_prob=0.94)[param].values[0/1] for HDI lower/upper from Dataset"

key-files:
  created: []
  modified:
    - src/prl_hgf/power/iteration.py
    - scripts/08_run_power_iteration.py
    - tests/test_power_iteration.py

key-decisions:
  - "Deferred import of fit_batch_hierarchical inside else block keeps heavy JAX imports out of legacy path"
  - "fit_df_2 not constructed in batched path — only fit_df_3 used in SBF subsampling loop for BF contrasts and diagnostics"
  - "az.rhat/az.ess return Dataset not scalar — extract via [param].values pattern"
  - "strict=True on all zip() calls in new helpers — catches participant metadata misalignment early"

patterns-established:
  - "idata_to_fit_df pattern: extract participant coords as ground truth, iterate param data_vars, use az.rhat/ess/hdi with Dataset[param].values extraction"
  - "split_idata pattern: posterior.isel(participant=i) then az.InferenceData(posterior=slice)"
  - "build_idata_dict pattern: enumerate zip(pids,groups,sessions,strict=True) to build (pid,grp,sess)->idata dict"

# Metrics
duration: 15min
completed: 2026-04-12
---

# Phase 14 Plan 01: Integration + GPU Benchmark Summary

**Dual-path `run_sbf_iteration` wiring Phase 12 batched hierarchical MCMC as default fitting backend, with `_idata_to_fit_df` / `_split_idata` / `_build_idata_dict` helpers bridging `az.InferenceData` to the legacy fit-df schema and WAIC dict contract**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-12T16:09:46Z
- **Completed:** 2026-04-12T16:24:40Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `run_sbf_iteration` now routes to `fit_batch_hierarchical` by default (`use_legacy=False`), enabling the Phase 12 batched JAX MCMC path to be used in the power sweep pipeline for the first time
- Three private helpers (`_idata_to_fit_df`, `_split_idata`, `_build_idata_dict`) bridge the structural gap between `az.InferenceData` with a `participant` dimension and the `(fit_df, idata_dict)` contract expected by downstream WAIC/BMS/recovery code
- `--legacy` flag on `08_run_power_iteration.py` preserves the v1.1 sequential path for reproducibility and debugging (VALID-05)
- 4 new tests pass: schema test for `_idata_to_fit_df`, correctness test for `_split_idata`, and wiring smoke tests for both legacy and batched paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _idata_to_fit_df, _split_idata, dual-path run_sbf_iteration** - `bcd8e02` (feat)
2. **Task 2: Add --legacy flag to CLI and write tests (BENCH-03, VALID-05)** - `f68bcd2` (feat)

## Files Created/Modified

- `src/prl_hgf/power/iteration.py` - Added `_idata_to_fit_df`, `_split_idata`, `_build_idata_dict` helpers; modified `run_sbf_iteration` to add `use_legacy` parameter with dual-path logic
- `scripts/08_run_power_iteration.py` - Added `--legacy` flag to `parse_args()`; pass `use_legacy=args.legacy` to `run_sbf_iteration`; fixed 3 pre-existing F541 f-string warnings
- `tests/test_power_iteration.py` - Added 4 tests: `test_idata_to_fit_df_schema`, `test_split_idata_produces_single_participant`, `test_run_sbf_iteration_legacy_flag_calls_fit_batch`, `test_run_sbf_iteration_default_calls_hierarchical`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Deferred import of `fit_batch_hierarchical` inside else block | Keeps heavy JAX/PyMC/pyhgf imports out of import-time when using legacy path; consistent with existing pattern in hierarchical.py |
| `fit_df_2` not constructed in batched path | SBF subsampling loop only uses `fit_df_3` for BF contrasts and diagnostics; `fit_df_2` is unused in legacy path downstream code too — no structural reason to create it |
| `az.rhat(da)[param].values` for scalar extraction | `az.rhat(DataArray)` returns a Dataset with one variable named after the input param; `.values` extracts the 0-d NumPy scalar correctly |
| `strict=True` on all `zip()` calls in new helpers | Catches participant metadata length mismatches at helper boundaries rather than silently misaligning participant-to-posterior mappings |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect az.rhat/az.ess extraction**

- **Found during:** Task 1 (implementation + integration test)
- **Issue:** Plan specified `az.rhat(da).item()` but `az.rhat(DataArray)` returns a `Dataset` (not a scalar DataArray), so `.item()` fails with `AttributeError: 'Dataset' has no attribute 'item'`
- **Fix:** Changed to `az.rhat(da)[param].values` which indexes the Dataset by variable name then extracts the 0-d NumPy array value; same pattern for `az.ess`
- **Files modified:** `src/prl_hgf/power/iteration.py`
- **Verification:** Quick integration test with mock InferenceData (2 participants, 2 params) confirmed correct scalar values returned
- **Committed in:** `bcd8e02` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed HDI extraction from az.hdi Dataset**

- **Found during:** Task 1 (same integration test)
- **Issue:** Plan used conditional `hdi_result[param].values[0] if hasattr(hdi_result, "data_vars")` — `az.hdi(DataArray)` always returns a Dataset, making the condition always True but the simpler form `hdi_result[param].values[0/1]` is sufficient
- **Fix:** Simplified to direct `hdi_result[param].values[0]` and `hdi_result[param].values[1]`
- **Files modified:** `src/prl_hgf/power/iteration.py`
- **Verification:** Same integration test confirmed correct lower/upper bounds
- **Committed in:** `bcd8e02` (Task 1 commit)

**3. [Rule 1 - Bug] Fixed pre-existing F541 f-string warnings in 08_run_power_iteration.py**

- **Found during:** Task 2 (ruff check on modified file)
- **Issue:** Three `print(f"...")` calls with no format placeholders in `_run_benchmark`
- **Fix:** Removed `f` prefix from the three print statements
- **Files modified:** `scripts/08_run_power_iteration.py`
- **Committed in:** `f68bcd2` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs — arviz API mismatches from plan; 1 Rule 1 pre-existing lint)
**Impact on plan:** All auto-fixes essential for correctness. No scope creep. The arviz API differences were not discoverable without running against a real InferenceData object.

## Issues Encountered

None beyond the arviz API issues documented above, which were fixed inline.

## Next Phase Readiness

- `run_sbf_iteration` is now wired to `fit_batch_hierarchical` by default — Plan 14-02 (GPU benchmark timing) can call `run_sbf_iteration` without `use_legacy=True` to benchmark the actual batched path
- Legacy path preserved and tested (VALID-05) — can be activated via `--legacy` for reproducibility comparison
- `_idata_to_fit_df` schema matches the 12-column legacy schema exactly — all downstream consumers (`compute_all_contrasts`, `_extract_diagnostics`, `build_recovery_df`) will receive the expected format
- Blocker for Plan 14-02: benchmark requires actual GPU hardware (or emulation); `_run_benchmark` in `08_run_power_iteration.py` still measures single-participant time — Plan 14-02 updates this to full-iteration timing

---
*Phase: 14-integration-gpu-benchmark*
*Completed: 2026-04-12*
