---
phase: 19-vb-laplace-fit-path-patrl
plan: "05"
subsystem: fitting
tags: [laplace, vb-laplace, patrl, smoke, cli, recovery, fit-method, argparse]

# Dependency graph
requires:
  - phase: 19-vb-laplace-fit-path-patrl
    provides: fit_vb_laplace_patrl (19-03) + export_trajectories (18-05) + pat_rl_simulator (19-01)
provides:
  - --fit-method {blackjax,laplace,both} CLI flag on scripts/12_smoke_patrl_foundation.py
  - VB-Laplace end-to-end smoke: simulate -> Laplace fit -> export (trajectory CSVs + parameter_summary.csv)
  - true_params.csv written alongside parameter_summary.csv for laplace/both modes (recovery comparison without re-simulation)
  - Explicit _sanity_check(idata, method) dispatch (no bare try/except around sample_stats)
  - 4 new fast tests + 2 RUN_SMOKE_TESTS=1-gated tests in tests/test_smoke_patrl_foundation.py
  - Phase 19 Success Criterion #3 (exporters consume Laplace idata unchanged) demonstrated end-to-end
  - Phase 19 Success Criterion #4 (5-agent <60s; 4/5 omega_2 recovery) demonstrated end-to-end
affects:
  - Phase 20 cluster validation runs (uses --fit-method laplace/both)
  - VBL-06 Laplace-vs-NUTS comparison harness (validation/vbl06_laplace_vs_nuts.py)
  - OQ7 closure memo (deferred until cluster NUTS numbers land)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Method-explicit sanity_check dispatch: if/elif/else on method str, no bare try/except"
    - "Lazy import pattern for optional comparison harness (vbl06_laplace_vs_nuts)"
    - "true_params.csv alongside parameter_summary.csv for recovery gate (no log parsing)"
    - "Primary/secondary idata pattern: for both mode, primary=laplace feeds export, secondary=blackjax is baseline"

key-files:
  created: []
  modified:
    - scripts/12_smoke_patrl_foundation.py
    - tests/test_smoke_patrl_foundation.py

key-decisions:
  - "--fit-method blackjax default preserves Phase 18 behavior bit-for-bit (true_params.csv not written for blackjax path)"
  - "primary = laplace, secondary = blackjax when method='both': reflects Phase 19 Laplace as the new path under test"
  - "_sanity_check raises ValueError if method='both' passed directly; caller must call leaf functions separately"
  - "both-mode VBL-06 comparison is lazy import: WARN if absent, not hard failure"
  - "_log_recovery_table extracted from old _sanity_check body; method-agnostic (works for Laplace and NUTS idata)"
  - "OQ7 closure memo deferred via TODO comment; written only after cluster NUTS numbers land"

patterns-established:
  - "Smoke script method dispatch: _fit_blackjax / _fit_laplace / _fit(method) returning (primary, secondary|None)"
  - "Test gate pattern: @pytest.mark.skipif(os.getenv('RUN_SMOKE_TESTS') != '1', ...) for slow subprocess tests"

# Metrics
duration: 15min
completed: 2026-04-18
---

# Phase 19 Plan 05: Smoke Script --fit-method Flag + Laplace End-to-End Summary

**`--fit-method {blackjax,laplace,both}` wired into 12_smoke_patrl_foundation.py; Laplace 5-agent smoke runs in 30.7s with 4/5 omega_2 recovery within 0.5 (Success Criteria #3 + #4 demonstrated end-to-end)**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-18T12:19:52Z
- **Completed:** 2026-04-18T12:35:42Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `--fit-method {blackjax,laplace,both}` argparse flag (default `blackjax`) preserving Phase 18 behavior
- Rewired `_fit()` as a dispatch function: `_fit_blackjax` / `_fit_laplace` / combined returning `(primary, secondary|None)`
- Implemented `_sanity_check(idata, method)` with explicit `if/elif/else` dispatch — no bare `try/except` around `sample_stats` access
- Wrote `_write_true_params_csv` for recovery comparison (Columns: `participant_id`, `parameter`, `true_value`)
- Wired lazy VBL-06 comparison import in `both` mode with informative WARN if module absent
- Extended test suite with 4 fast + 2 slow (env-gated) tests; all 11 fast and 2 slow tests pass

## 5-Agent Laplace Recovery Table

Observed at `--seed 42 --level 2 --n-participants 5` (wall time: **30.7 s total**):

| Participant | True omega_2 | Posterior Mean | abs_diff | PASS? |
|-------------|--------------|----------------|----------|-------|
| P000        | -5.791       | -5.331         | 0.460    | YES   |
| P001        | -5.373       | -5.090         | 0.283    | YES   |
| P002        | -6.825       | -6.928         | 0.104    | YES   |
| P003        | -6.050       | -6.268         | 0.218    | YES   |
| P004        | -6.172       | -6.873         | 0.702    | NO    |

**4/5 agents within |diff| < 0.5 — Success Criterion #4 MET.** Wall time 30.7s << 60s gate.

## Task Commits

1. **Task 1: Add --fit-method flag + dispatch** - `afb69c3` (feat)
2. **Task 2: Extend tests with --fit-method coverage** - `bf232b0` (test)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `scripts/12_smoke_patrl_foundation.py` — Added `--fit-method` flag; `_fit_blackjax` / `_fit_laplace` / `_fit()` dispatch; `_sanity_check(idata, method)` explicit dispatch; `_write_true_params_csv`; `_log_recovery_table`; lazy VBL-06 import hook; OQ7 TODO comment; ruff clean
- `tests/test_smoke_patrl_foundation.py` — 4 new fast tests (`test_script_accepts_fit_method_flag`, `test_script_rejects_invalid_fit_method`, `test_script_lazy_imports_vbl06_for_both_mode`, `test_pick_best_cue_regression_still_passes`) + 2 slow env-gated tests (`test_smoke_end_to_end_laplace_2level`, `test_smoke_laplace_recovery_sanity_4_of_5`)

## Phase 19 Open Questions — Final Disposition

| OQ  | Description | Disposition |
|-----|-------------|-------------|
| OQ1 | `_samples_to_idata` emits `participant` coord (not `participant_id`) | **Sidestepped**: Laplace path (`build_idata_from_laplace`) emits `participant_id` natively; NUTS dim-name mismatch tracked in STATE.md for follow-up |
| OQ2 | kappa logit reparametrization | **Deferred**: native-space kappa shipped; cluster VBL-06 will reveal if MAP hits boundary (see STATE.md blocker) |
| OQ3 | `n_pseudo_draws` configurability | **Done**: default 1000, configurable via kwarg in `fit_vb_laplace_patrl` (19-03) |
| OQ4 | Models B/C/D support | **Done**: `NotImplementedError` wired in `fit_vb_laplace_patrl` for non-model_a response_model |
| OQ5 | MAP restart mechanism | **Done**: `n_restarts` kwarg with perturbed initialization in 19-03 |
| OQ6 | Diagnostics in `sample_stats` | **Done**: `build_idata_from_laplace` stashes `converged`, `n_iterations`, `logp_at_mode`, `hessian_min_eigval`, `ridge_added` in `sample_stats` |
| OQ7 | Phase 19 closure memo | **Deferred**: TODO comment added near module top; memo written only after cluster NUTS numbers land |

## Phase 19 Success Criteria — Final Check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `fit_vb_laplace_patrl` passes 12-test unit suite | DONE (19-03) |
| 2 | `build_idata_from_laplace` NUTS shape-parity + ArviZ compatibility | DONE (19-02) |
| 3 | Plan 18-05 exporters consume Laplace idata unchanged | DONE (19-05 end-to-end) |
| 4 | 5-agent CPU smoke <60s + 4/5 omega_2 recovery | DONE (30.7s; 4/5 pass) |
| 5 | Parallel-stack invariant preserved | DONE (git diff empty for protected paths) |
| 6 | No NUTS path modified | DONE (parallel-stack invariant confirmed) |

**All 6 Phase 19 Success Criteria satisfied.**

## Parallel-Stack Invariant Check

```
git diff --name-only src/prl_hgf/fitting/hierarchical.py \
    src/prl_hgf/fitting/hierarchical_patrl.py \
    src/prl_hgf/env/task_config.py \
    src/prl_hgf/env/simulator.py \
    src/prl_hgf/models/hgf_2level.py \
    src/prl_hgf/models/hgf_3level.py \
    src/prl_hgf/models/response.py \
    configs/prl_analysis.yaml configs/pat_rl.yaml
```

**Output: EMPTY** — Invariant preserved.

## Decisions Made

- `--fit-method blackjax` default preserves Phase 18 behavior bit-for-bit (`true_params.csv` not written for blackjax path)
- `primary = laplace, secondary = blackjax` when `method='both'`: Laplace is the "new path under test"; BlackJAX is the baseline
- `_sanity_check` raises `ValueError` if `method='both'` is passed directly; callers must dispatch to leaf calls separately (prevents accidental silent no-op)
- VBL-06 comparison in `both` mode is a lazy import — `WARN` if absent, not hard failure; keeps default runs clean
- `_log_recovery_table` extracted from old `_sanity_check` body to be method-agnostic (works for Laplace and NUTS idata)
- OQ7 closure memo deferred via TODO comment; written only after cluster NUTS numbers land

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed redundant `import arviz as az` inside `both` branch**

- **Found during:** Task 1 (ruff check)
- **Issue:** `import arviz as az` was added inside the `both` block but `az` was never used (`.to_netcdf` called on idata directly); ruff raised F401
- **Fix:** Removed the unused import; `.to_netcdf` called via duck-typing (InferenceData always has this method)
- **Files modified:** `scripts/12_smoke_patrl_foundation.py`
- **Verification:** `ruff check scripts/12_smoke_patrl_foundation.py` clean
- **Committed in:** afb69c3 (Task 1 commit)

**2. [Rule 1 - Bug] Reorganized import block to fix I001 (isort violation)**

- **Found during:** Task 1 (ruff check)
- **Issue:** Import block had `fit_vb_laplace_patrl` import between `pat_rl_config` and `pat_rl_simulator`; ruff I001 required alphabetical grouping
- **Fix:** Reordered to `pat_rl_config` → `pat_rl_simulator` → `fit_vb_laplace_patrl` (alphabetical within package)
- **Files modified:** `scripts/12_smoke_patrl_foundation.py`
- **Verification:** `ruff check scripts/12_smoke_patrl_foundation.py` clean
- **Committed in:** afb69c3 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — ruff-detected bugs)
**Impact on plan:** Minor import ordering; no behavioral change.

## Issues Encountered

None — plan executed exactly as specified except minor ruff import ordering fixes applied immediately.

## Next Phase Readiness

- Phase 19 is **complete** (all 5 plans done; all 6 success criteria satisfied)
- VB-Laplace path ready for cluster validation via `sbatch cluster/18_smoke_patrl_cpu.slurm`
- OQ7 closure memo pending cluster NUTS results; write `.planning/phases/19-vb-laplace-fit-path-patrl/19-CLOSURE-MEMO.md` after numbers land
- OQ1 follow-up pending: `_samples_to_idata` NUTS coord is `participant` (not `participant_id`); tracked in STATE.md

---
*Phase: 19-vb-laplace-fit-path-patrl*
*Completed: 2026-04-18*
