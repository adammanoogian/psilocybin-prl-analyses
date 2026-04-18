---
phase: 18-pat-rl-task-adaptation
plan: "06"
subsystem: testing
tags: [pat-rl, smoke-test, end-to-end, cluster, slurm, blackjax, mcmc, cpu, integration]

requires:
  - phase: 18-pat-rl-task-adaptation
    provides: "18-01 load_pat_rl_config + PATRLConfig dataclass (parallel config loader)"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-02 generate_session_patrl + PATRLTrial dataclass"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-03 hgf_2level_patrl + hgf_3level_patrl Network builders + Model A response"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-04 fit_batch_hierarchical_patrl BlackJAX orchestrator + _build_patrl_log_posterior"
  - phase: 18-pat-rl-task-adaptation
    provides: "18-05 export_subject_trajectories + export_subject_parameters CSV writers"

provides:
  - "scripts/12_smoke_patrl_foundation.py: numbered pipeline script running simulate -> fit -> export -> sanity end-to-end for N synthetic PAT-RL agents on CPU"
  - "tests/test_smoke_patrl_foundation.py: 7 structural/regression tests (py_compile, argparse-gate, parallel-stack invariant, lazy-blackjax, pick_best_cue canaries)"
  - "cluster/18_smoke_patrl_cpu.slurm: SLURM wrapper running the smoke on M3 MASSIVE comp partition (CPU)"
  - "Demonstrated end-to-end PAT-RL foundation wiring: config -> trials -> HGF -> Model A logp -> BlackJAX NUTS -> trajectory CSVs -> parameter summary CSV"

affects:
  - "Phase 19 VB-Laplace: shares simulate_patrl_cohort helper (extracted into pat_rl_simulator module); same CSV exporters; comparison smoke reuses this script as NUTS reference"
  - "Phase 19+ PRL-V1 recovery gate (deferred): will reuse this harness with r>=0.7 Pearson correlation on omega_2"
  - "dcm_pytorch the consumer study integration: consumes the exported trajectory CSVs as stimulus input"

tech-stack:
  added: []
  patterns:
    - "Lazy blackjax import inside _fit() so --dry-run works without blackjax installed"
    - "Exit code map: 0 success / 1 runtime error / 2 blackjax missing (actionable CI behaviour)"
    - "Parallel-stack invariant enforced by static test: forbidden-substring scan of pick_best_cue imports"

key-files:
  created:
    - scripts/12_smoke_patrl_foundation.py
    - tests/test_smoke_patrl_foundation.py
    - cluster/18_smoke_patrl_cpu.slurm
  modified:
    - src/prl_hgf/fitting/hierarchical.py
    - src/prl_hgf/fitting/hierarchical_patrl.py

key-decisions:
  - "_samples_to_idata() gains coord_name kwarg (default 'participant' for PRL back-compat); hierarchical_patrl passes 'participant_id' to match exporter contract"
  - "blackjax imported lazily inside _fit() (not module top-level) so --dry-run path runs in blackjax-free environments"
  - "Smoke script exit codes: 0 success / 1 runtime error / 2 blackjax missing (distinct from generic failure for CI / docs diagnosis)"
  - "Structural tests run unconditionally; end-to-end slow test gated by env var (deferred to cluster-side execution; no wall-clock cost locally)"
  - "Sanity gate is divergence<20% + finite posteriors only; directional check is log-only; full PRL-V1 r>=0.7 recovery gate deferred to Phase 19+"
  - "SLURM wrapper runs on CPU `comp` partition (not GPU) because smoke budget is ~15s end-to-end; no reason to consume GPU hours"

patterns-established:
  - "Numbered pipeline scripts (scripts/NN_verb_noun.py) acquire CLI --dry-run for module-wiring validation separate from full-cost execution"
  - "Parallel-stack hygiene enforced at two levels: (1) production code imports from _patrl variants only, (2) static test fails CI if pick_best_cue import creeps in"
  - "Two-job SLURM debugging pattern: first job surfaces integration bug, targeted fix, second job validates end-to-end"

duration: writeup-only (underlying code completed in quick-004 + this-session coord fix)
completed: 2026-04-18
---

# Phase 18 Plan 06: PAT-RL Foundation End-to-End Smoke Summary

**5-agent PAT-RL foundation smoke (simulate -> fit -> export -> sanity) passed on M3 MASSIVE CPU in 13.9s wall time; validates the complete parallel PAT-RL stack from config loader to CSV export.**

## Performance

- **Duration (writeup):** this agent only wrote the summary + STATE update; code was authored in quick-004 + a targeted coord fix today (commit `d5c0b72`)
- **Cluster wall time (job 54894259):** 13.9s total (0.7s simulate + 11.8s fit + ~1s export + ~0.4s sanity) on M3 comp partition, single CPU
- **Completed:** 2026-04-18
- **Tasks:** 2 tasks in the plan (smoke script + pytest); both shipped
- **Files created:** 3 (script, test, SLURM wrapper)
- **Files modified today for wiring:** 2 (hierarchical.py, hierarchical_patrl.py)

## Accomplishments

- End-to-end PAT-RL foundation stack executes cleanly on CPU: config -> trial generator -> 2-level HGF Network -> Model A logp -> BlackJAX NUTS -> per-subject trajectory CSV + parameter summary CSV -> divergence/finiteness sanity gate
- Cluster smoke `patrl18smoke_54894259` **PASSED** with **0 divergences / 1000 samples** at `n_tune=500 n_draws=500 n_participants=5`
- All 5/5 participants pass the directional omega_2 recovery check (posterior mean moves toward true relative to prior mean); per-participant posterior vs true diffs logged below
- 7 structural tests pass locally on Windows without blackjax installed (18.94s including argparse subprocess latency)
- Parallel-stack invariant enforced: smoke script contains zero imports from pick_best_cue modules (task_config, hgf_2level, response) — verified both by static test and by grep

## Task Commits

Each task was committed atomically during quick-004, with a single follow-up fix landed today when the export step was exercised end-to-end for the first time:

1. **Task 1 + Task 2 (combined): smoke script + structural tests + cluster SLURM wrapper** - `468278d` (feat/quick-004)
   - `scripts/12_smoke_patrl_foundation.py` created
   - `tests/test_smoke_patrl_foundation.py` created (7 tests)
   - `cluster/18_smoke_patrl_cpu.slurm` created
2. **Export coordinate fix (today)** - `d5c0b72` (fix/18)
   - `_samples_to_idata()` in `hierarchical.py` gains `coord_name` kwarg (default `"participant"` for PRL back-compat)
   - `hierarchical_patrl.py` passes `coord_name="participant_id"` to match the contract expected by `export_subject_trajectories`
3. **Scripts/12 refactored to import from `pat_rl_simulator` module** - `c481f91` (feat/19-01; belongs to Phase 19 but affects this file)

**Plan metadata (this commit):** `docs(18-06): complete end-to-end PAT-RL foundation smoke`

## Files Created/Modified

**Created (quick-004):**
- `scripts/12_smoke_patrl_foundation.py` - 5-agent smoke driver; argparse with `--level {2,3}`, `--output-dir`, `--n-tune`, `--n-draws`, `--seed`, `--n-participants`, `--dry-run`; lazy blackjax import; exit-code map (0/1/2)
- `tests/test_smoke_patrl_foundation.py` - 5 declared tests (expanded to 7 via parametrize); py_compile, argparse gate, forbidden-import scan, lazy-blackjax scan, pick_best_cue regression canary
- `cluster/18_smoke_patrl_cpu.slurm` - SLURM submission wrapper for M3 MASSIVE `comp` partition

**Modified (today, 2026-04-18, to unblock export step):**
- `src/prl_hgf/fitting/hierarchical.py` - `_samples_to_idata()` now accepts `coord_name` kwarg; all internal dims/coords/attrs keyed off that name
- `src/prl_hgf/fitting/hierarchical_patrl.py` - passes `coord_name="participant_id"` at the single call site

**Pick_best_cue modules unchanged by Phase 18 (confirmed):**
- `src/prl_hgf/fitting/hierarchical.py` *does* gain a new kwarg but the default preserves the exact PRL-side behaviour; all PRL call sites exercise the default
- `src/prl_hgf/models/hgf_2level.py`, `src/prl_hgf/models/hgf_3level.py`, `src/prl_hgf/models/response.py`, `src/prl_hgf/env/task_config.py`, `src/prl_hgf/env/simulator.py`, `configs/prl_analysis.yaml` — untouched

## Cluster Validation (Primary Evidence)

Two SLURM jobs on M3 MASSIVE `comp` partition (CPU-only, JAX-CPU):

**Job 54893705 (FAILED at export step):**
- Start: Sat Apr 18 19:30:08 AEST 2026  End: 19:31:14 AEST (1m06s total; the fit itself took ~15.3s)
- Simulation completed (5 participants, approach rates 0.30-0.62)
- Error at export: `ValueError: export_subject_trajectories: participant_id='P000' not found in idata.posterior participant_id coordinate.`
- Root cause: `_samples_to_idata()` hardcoded the coord name to `"participant"` (PRL convention); `export_subject_trajectories` was written against the PAT-RL exporter spec which documents `"participant_id"`

**Targeted fix (commit d5c0b72):** added `coord_name` kwarg; default preserves PRL behaviour; PAT-RL path passes `"participant_id"`.

**Job 54894259 (PASSED):**
- Start: Sat Apr 18 20:06:52 AEST 2026  End: 20:08:17 AEST (1m25s including environment load; actual script ran 13.9s)
- Params: level=2, n_tune=500, n_draws=500, n_participants=5, seed=42
- **Divergence rate: 0.000 (0 / 1000)** — well under the 20% smoke gate
- Fit wall-clock: 11.8s  /  Simulation: 0.7s  /  Total script: 13.9s
- CSV output: 6 files, 137.3 KB total, at `output/patrl_smoke_54894259/`
- Directional recovery check: **5/5 participants** passed (posterior omega_2 moved toward true relative to prior)

**Per-participant posterior vs true (job 54894259):**

| PID  | omega_2 true | omega_2 post | diff  | beta true | beta post | diff  |
|------|--------------|--------------|-------|-----------|-----------|-------|
| P000 | -5.791       | -5.345       | +0.445| 2.303     | 2.828     | +0.525|
| P001 | -5.373       | -5.143       | +0.230| 2.303     | 2.953     | +0.650|
| P002 | -6.825       | -6.957       | -0.132| 2.703     | 2.535     | -0.168|
| P003 | -6.050       | -6.290       | -0.240| 2.730     | 2.965     | +0.235|
| P004 | -6.172       | -7.221       | -1.049| 1.170     | 1.398     | +0.228|

Magnitudes of omega_2 errors (0.13-1.05) are within the envelope expected for N=192 trials with a weakly-informative prior; no hard recovery gate applied (deferred to Phase 19+ PRL-V1).

## Local Pytest (this-session verification)

```
pytest tests/test_smoke_patrl_foundation.py -v --tb=short
...
tests/test_smoke_patrl_foundation.py::test_smoke_script_py_compiles PASSED
tests/test_smoke_patrl_foundation.py::test_smoke_script_rejects_invalid_level PASSED
tests/test_smoke_patrl_foundation.py::test_smoke_script_has_no_pick_best_cue_imports PASSED
tests/test_smoke_patrl_foundation.py::test_smoke_script_lazy_imports_blackjax PASSED
tests/test_smoke_patrl_foundation.py::test_pick_best_cue_modules_still_compile[src/prl_hgf/env/task_config.py] PASSED
tests/test_smoke_patrl_foundation.py::test_pick_best_cue_modules_still_compile[src/prl_hgf/models/hgf_2level.py] PASSED
tests/test_smoke_patrl_foundation.py::test_pick_best_cue_modules_still_compile[src/prl_hgf/models/response.py] PASSED

============================= 7 passed in 18.94s ==============================
```

All 7 tests pass post coord_name fix. Argparse invalid-level test dominates the runtime (~18.76s) because it spawns a Python subprocess that imports the full JAX/pyhgf stack to argparse-reject before import side-effects are triggered; the static-scan tests are sub-millisecond.

## Decisions Made

Locked for future phases (these supplement, not duplicate, the 18-05 decisions already in STATE.md):

1. **`_samples_to_idata()` gains `coord_name` kwarg, default `"participant"` for PRL back-compat.** Rationale: PRL pipeline treats `participant` as the canonical coord label; PAT-RL Plan 18-05 `export_subject_trajectories` documents `participant_id`; back-compat preserved by keeping the old default and making PAT-RL the explicit opt-in.
2. **Blackjax imported lazily inside `_fit()`.** Rationale: allows `--dry-run` to validate simulate + forward-pass on any machine without blackjax; lets Windows/dev workflow test the script end-to-end without installing the GPU stack.
3. **Smoke exit codes are a 3-state map: 0 success / 1 runtime error / 2 blackjax missing.** Rationale: CI can distinguish "real bug" from "environment not provisioned" without parsing stderr.
4. **SLURM wrapper runs `comp` partition (CPU), not GPU.** Rationale: fit wall-clock is 11.8s on single-thread CPU; GPU dispatch overhead dominates at this batch size; no justification for consuming GPU hours for the smoke.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `_samples_to_idata()` coord name mismatch with exporter**
- **Found during:** First cluster run of Task 1 (job 54893705)
- **Issue:** Plan 18-04 produced idata with coord `"participant"`; Plan 18-05 `export_subject_trajectories` selects on `"participant_id"`. The mismatch is silent until export is actually called end-to-end — both unit test suites pass because each stub was coord-agnostic.
- **Fix:** Added `coord_name` kwarg to `_samples_to_idata()` defaulting to `"participant"` (PRL back-compat); PAT-RL path passes `"participant_id"`. Ten-line diff, fully back-compatible.
- **Files modified:** `src/prl_hgf/fitting/hierarchical.py`, `src/prl_hgf/fitting/hierarchical_patrl.py`
- **Verification:** Job 54894259 passed the full simulate -> fit -> export -> sanity path with 0 divergences.
- **Committed in:** `d5c0b72`

**2. [Rule 3 - Blocking] Script refactored out to `pat_rl_simulator` for Phase 19 reuse**
- **Found during:** Phase 19-01 work (parallel-stream Phase 19 window)
- **Issue:** Phase 19 Laplace smoke needs the same synthetic cohort; inlining the simulator inside `scripts/12` would force duplication.
- **Fix:** Extracted `simulate_patrl_cohort` into `src/prl_hgf/env/pat_rl_simulator.py`; `scripts/12` now imports it.
- **Files modified:** `scripts/12_smoke_patrl_foundation.py`, `src/prl_hgf/env/pat_rl_simulator.py` (Phase 19 territory)
- **Committed in:** `c481f91` (tagged `feat(19-01)`; noted here for lineage completeness, not owned by Phase 18)

---

**Total deviations:** 1 Phase-18-owned fix (`d5c0b72`); 1 Phase-19-owned refactor that touches this script (`c481f91`) noted for lineage.
**Impact on plan:** Both changes strengthen the surface without expanding scope. The coord fix was essential for the plan's "end-to-end" success criterion.

## Authentication Gates

None. All cluster execution is SSH-key-authenticated and runs under the user's M3 account; no interactive auth was needed during this plan.

## Issues Encountered

- **Coord name drift between 18-04 fit output and 18-05 export input** was the only live issue; resolved in `d5c0b72`. Root cause is a documentation debt — both plans were written correctly against their own specs; the integration surface was never exercised end-to-end until this plan ran.
- **Local Windows pytest cannot exercise the MCMC path** because blackjax is not installed in the dev environment. This was anticipated and the smoke test suite was designed to be useful without it: 7 structural tests run locally; the end-to-end fit path is exercised by the cluster SLURM job. This matches the quick-004 VB-Laplace feasibility memo's Option C reasoning.

## Deferred (Not in Scope — Phase 19+)

Consistent with Phase 18 Option A scope limits, the following are explicitly deferred:

- **Models B / C / D** (trial-varying omega; multi-modulator response) — Phase 19+
- **PRL-V1 r >= 0.7 recovery gate** — this plan's sanity check is finite+direction+divergence only; full correlation gate deferred
- **PRL-V2 phenotype identifiability grid** — requires multi-phenotype cohort, separate plan
- **Stratified BMS across phenotypes** — requires Models B/C/D first
- **PEB covariate export** — requires WAIC/∆WAIC columns, deferred to dedicated export plan

## Next Phase Readiness

- Foundation smoke PASSED on cluster; all wiring validated.
- `simulate_patrl_cohort` extracted into `pat_rl_simulator` for Phase 19 reuse (commit `c481f91`).
- Phase 19 VB-Laplace has a known-good BlackJAX reference fit from job 54894259 to compare against.
- No blockers owned by Phase 18 remain.

**Note:** Phase 19 VB-Laplace work is proceeding in a parallel Claude Code window. That work owns the `fit_vb_laplace_patrl.py` path and associated test files; this summary deliberately does not cover that surface.

---
*Phase: 18-pat-rl-task-adaptation*
*Completed: 2026-04-18*
