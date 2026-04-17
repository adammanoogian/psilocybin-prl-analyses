---
phase: 18-pat-rl-task-adaptation
plan: "05"
subsystem: analysis
tags: [pyhgf, arviz, csv-export, dcm-pytorch, hgf, trajectory, integration-surface]

requires:
  - phase: 18-pat-rl-task-adaptation
    provides: "18-04 PAT-RL batched logp factory + fit_batch_hierarchical_patrl with InferenceData output"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-03 hgf_2level_patrl + hgf_3level_patrl Network builders + extract_beliefs helpers"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-02 generate_session_patrl + PATRLTrial dataclass with outcome_time_s"

provides:
  - "export_subject_trajectories(): post-hoc forward pass at posterior means → per-trial CSV"
  - "export_subject_parameters(): posterior mean + 94% HDI per-subject parameter summary CSV"
  - "_safe_temp() helper: pyhgf version-safe temp-dict extraction with NaN fallback"
  - "18-05-dcm-interface-notes.md: audited dcm_pytorch consumer contract with file:line citations"
  - "10 pytest cases in test_export_trajectories.py including pyhgf temp-key canary"

affects:
  - "18-06 validation (uses exported CSVs for scientific validation)"
  - "dcm_pytorch HEART2ADAPT integration (v0.4+): consumes _trajectories.csv as stimulus input"
  - "future PEB-lite export (deferred): will extend export_trajectories.py with WAIC/ΔWAIC columns"

tech-stack:
  added: []
  patterns:
    - "Post-hoc forward pass at posterior means: build Network at mean params, call input_data(), read node_trajectories"
    - "TYPE_CHECKING guard for arviz import: deferred runtime import + type: ignore[attr-defined] for InferenceData.posterior"
    - "_safe_temp() NaN fallback pattern: protects against pyhgf temp-dict key renaming across versions"
    - "Frozen column schema (_TRAJECTORY_COLUMNS list): single source of truth for CSV column order"

key-files:
  created:
    - src/prl_hgf/analysis/export_trajectories.py
    - tests/test_export_trajectories.py
    - .planning/phases/18-pat-rl-task-adaptation/18-05-dcm-interface-notes.md
  modified: []

key-decisions:
  - "bilinear B-matrix in dcm_pytorch v0.3.0 is LIVE (not deferred to v0.4+): parameterize_B + compute_effective_A in neural_state.py"
  - "Modulator channel values: raw float64, no bounding or normalization (dcm_pytorch prior N(0,1) does regularization)"
  - "outcome_time_s feeds stimulus['times'] directly: both are absolute seconds from session start"
  - "az.hdi returns 'lower'/'higher' coordinate labels (not 'low'/'high') in ArviZ 0.22+"
  - "pyhgf 0.2.8 temp keys confirmed: value_prediction_error, effective_precision, volatility_prediction_error all present"
  - "input_data() call: (n_trials, 1) input_data + 1D np.ones(n_trials) time_steps confirmed correct for pyhgf 0.2.8"
  - "__init__.py untouched: callers import from prl_hgf.analysis.export_trajectories directly"

patterns-established:
  - "PRL.4 producer side: export_subject_trajectories() is the single function to call for dcm_pytorch ingestion"
  - "3-level-only columns present in 2-level CSV as NaN: schema consistency for downstream concat/join"

duration: 32min
completed: 2026-04-17
---

# Phase 18 Plan 05: DCM Integration Surface (Trajectory Export) Summary

**Per-trial HGF belief trajectory CSV exporter with dcm_pytorch bilinear-DCM consumer contract verified; pyhgf 0.2.8 temp-key extraction confirmed; 10 pytest cases green including temp-key canary.**

## Performance

- **Duration:** 32 min
- **Started:** 2026-04-17T22:23:41Z
- **Completed:** 2026-04-17T22:55:41Z
- **Tasks:** 3
- **Files created:** 3 (export_trajectories.py, test_export_trajectories.py, 18-05-dcm-interface-notes.md)

## Accomplishments

- Audited dcm_pytorch v0.3.0 consumer interfaces (task_simulator.py + neural_state.py) — confirmed bilinear B-matrix path is live in v0.3.0, not deferred. Plan sketch was incorrect on this point.
- Implemented `export_subject_trajectories()`: post-hoc forward pass at posterior means for 2-level and 3-level PAT-RL HGF, produces 19-column trajectory CSV per subject.
- Implemented `export_subject_parameters()`: per-subject posterior mean + 94% HDI in long format (participant_id, parameter, posterior_mean, hdi_low, hdi_high).
- Added `_safe_temp()` helper with NaN fallback to guard against future pyhgf temp-dict key changes.
- 10 pytest cases all green; pyhgf temp-key canary confirms value_prediction_error, effective_precision, volatility_prediction_error in pyhgf 0.2.8 node_trajectories.
- `src/prl_hgf/analysis/__init__.py` unchanged (parallel-stack policy maintained).

## Task Commits

1. **Task 1: DCM consumer interface audit + notes file** — `9ab460c` (docs)
2. **Task 2: export_subject_trajectories + export_subject_parameters** — `57d69b2` (feat)
3. **Task 3: 10 pytest cases** — `21f71c6` (test)

## Files Created/Modified

- `src/prl_hgf/analysis/export_trajectories.py` — module with `export_subject_trajectories`, `export_subject_parameters`, `_safe_temp`
- `tests/test_export_trajectories.py` — 10 tests: schema, dtypes, 2-level NaN, 3-level populated, row count, outcome_time_s monotone, psi2>0, param summary columns/rows, pyhgf canary, pick_best_cue regression
- `.planning/phases/18-pat-rl-task-adaptation/18-05-dcm-interface-notes.md` — consumer-interface audit with file:line citations

## Final CSV Schema

Per-trial trajectory CSV (19 columns, ordered):

| column | dtype | notes |
|---|---|---|
| participant_id | str | subject identifier |
| trial_idx | int32 | 0..n_trials-1 |
| run_idx | int32 | 0..n_runs-1 |
| trial_in_run | int32 | 0..trials_per_run-1 |
| regime | str | "stable" or "volatile" |
| outcome_time_s | float64 | absolute seconds from session start → feeds stimulus["times"] |
| state | int32 | 0=safe, 1=dangerous |
| choice | int32 | 0=avoid, 1=approach |
| reward_mag | float64 | reward level |
| shock_mag | float64 | shock level |
| delta_hr | float64 | anticipatory Delta-HR (bpm) |
| mu2 | float64 | HGF level-2 posterior mean (log-odds) |
| sigma2 | float64 | 1/precision at level 2 |
| mu3 | float64 | 3-level only; NaN for 2-level |
| sigma3 | float64 | 3-level only; NaN for 2-level |
| delta1 | float64 | INPUT_NODE value_prediction_error |
| epsilon2 | float64 | BELIEF_NODE value_prediction_error |
| epsilon3 | float64 | 3-level only; NaN for 2-level |
| psi2 | float64 | BELIEF_NODE effective_precision |

Parameter summary CSV (5 columns): participant_id, parameter, posterior_mean, hdi_low, hdi_high

**Estimated CSV size for 32-subject cohort (2-level):**
- Per subject: 192 rows × 19 columns ≈ 4 KB per CSV
- 32 subjects: ~128 KB total trajectory CSVs
- Parameter summary (2-level): 32 × 2 = 64 rows ≈ 3 KB
- dcm_pytorch ingestion budget: well within typical file I/O limits

## pyhgf 0.2.8 Temp Keys Confirmed

Runtime verification (pyhgf 0.2.8, BELIEF_NODE = node 1):

- `temp["value_prediction_error"]` ✓ present, all finite
- `temp["effective_precision"]` ✓ present, all > 0
- `temp["volatility_prediction_error"]` ✓ present (level-2→level-3 PE)
- INPUT_NODE temp: `temp["value_prediction_error"]` ✓ present

`_safe_temp()` fallback returns NaN array if any key disappears in future pyhgf versions.

## Decisions Made

| Decision | Rationale |
|---|---|
| bilinear B-matrix confirmed LIVE in v0.3.0 | Read neural_state.py:4-15 + coupled_system.py:20-22; plan sketch was wrong about "deferred" status |
| Modulator values: raw float64, no normalization | neural_state.py:104-113: off-diagonal pass through via pure mask; N(0,1) prior does regularization |
| az.hdi uses "lower"/"higher" labels | Verified at runtime: ArviZ 0.22+ returns hdi coord with values ["lower", "higher"] |
| TYPE_CHECKING guard for arviz import | Matches existing pattern in analysis/group.py + bms.py; avoids heavy import at module load |
| 3-level-only cols present as NaN in 2-level | Schema consistency: downstream concat/join across model variants works without column-presence check |
| input_data() call: 1D time_steps | Matches 18-03 confirmed API: np.ones(n_trials) not np.ones((n_trials,1)) |

## Deviations from Plan

### Schema Correction (Not a Code Deviation)

**[Rule 1 - Bug in Plan Spec] Bilinear DCM status in plan sketch incorrect**

- **Found during:** Task 1 (dcm_pytorch consumer interface audit)
- **Issue:** Plan sketch stated "v0.3.0 form is linear dx/dt = Ax + Cu (no B matrix yet)" and "bilinear DCM deferred to v0.4+". Both claims are incorrect.
- **Reality:** `parameterize_B`, `compute_effective_A`, and `NeuralStateEquation.derivatives(B=, u_mod=)` are all present and functional in v0.3.0 (neural_state.py lines 62-224, coupled_system.py lines 20-73).
- **Impact:** No code change needed — the CSV schema (raw float64 values, no normalization) is actually more compatible with the live bilinear API. The dcm_pytorch caller decides transforms.
- **Documented in:** `18-05-dcm-interface-notes.md` Section 3.

No code deviations — plan executed as specified after Task 1 schema correction.

## Issues Encountered

- `az.hdi` returns `"lower"`/`"higher"` coordinate labels (not `"low"`/`"high"` as the plan assumed). Fixed in implementation; documented above.
- Ruff flagged 3 minor issues in test file (import order, unused `_safe_temp` import, spurious f-string prefix) — all auto-fixed with `ruff --fix`.

## Next Phase Readiness

- Phase 18-06 (validation): trajectory CSVs producible for any fitted subject; parameter summary CSVs ready for BFDA input.
- dcm_pytorch HEART2ADAPT integration: `outcome_time_s` → `stimulus["times"]`; select any subset of HGF belief columns for `stimulus["values"][:, j]`. Bilinear B-matrix path confirmed live in v0.3.0.
- Deferred: PEB covariate export (ΔWAIC/ΔF), stratified BMS, actual feeding into dcm_pytorch (deferred per plan scope).
- Blocker: none — all upstream PAT-RL tests green; pick_best_cue regressions clean.

---
*Phase: 18-pat-rl-task-adaptation*
*Completed: 2026-04-17*
