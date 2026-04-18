---
phase: 18-pat-rl-task-adaptation
verified: 2026-04-18T00:00:00Z
status: passed
score: 8/8 must-haves verified
scope: Option A Minimum Viable (user-confirmed 2026-04-17; ROADMAP.md line 235)
deferred_to_phase_19:
  - Models B/C/D (only Model A in scope)
  - Full PRL-V1 r>=0.7 recovery gate
  - PRL-V2 phenotype identifiability
  - PRL.5 stratified BMS / PEB covariate export
  - VB-Laplace fit path
---

# Phase 18: PAT-RL Task Adaptation (HEART2ADAPT) Verification Report

**Phase Goal (Option A scope):** Ship config + trial generator + binary-state
HGF builders (2-level & 3-level) + Model A response + BlackJAX fit + trajectory
export + 5-agent CPU smoke — as a **parallel** task stack alongside the existing
`pick_best_cue` pipeline.

**Verified:** 2026-04-18
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Option A)

| #   | Truth                                                                            | Status     | Evidence                                                                |
| --- | -------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------- |
| 1   | PAT-RL YAML config is a distinct source of truth from `prl_analysis.yaml`        | VERIFIED   | `configs/pat_rl.yaml` (114 lines); only docstring reference to task_config — no import |
| 2   | 192-trial PAT-RL session with hazard-rate state, 2x2 magnitudes, Delta-HR        | VERIFIED   | `generate_session_patrl` at `pat_rl_sequence.py:266`; 4x48 runs, SeedSequence(seed).spawn(4) |
| 3   | Binary-state HGFs (single input node) distinct from 3-cue pick_best_cue Network  | VERIFIED   | `hgf_2level_patrl.py` 152L, `hgf_3level_patrl.py` 204L; both use single input node 0 |
| 4   | Model A response logp works over (mu2, choices, reward_mag, shock_mag, beta)     | VERIFIED   | `response_patrl.py:75 model_a_logp`; stacked softmax on approach-EV     |
| 5   | PAT-RL hierarchical fit is a parallel stack using BlackJAX helpers               | VERIFIED   | `hierarchical_patrl.py:660 fit_batch_hierarchical_patrl` reuses `_run_blackjax_nuts` |
| 6   | Trajectory + parameter CSV exporters emit the frozen DCM-interface schema        | VERIFIED   | `export_trajectories.py:123 export_subject_trajectories`, `:292 export_subject_parameters`; 19-column schema matches `18-05-dcm-interface-notes.md` |
| 7   | End-to-end smoke (simulate→fit→export) runnable + structurally tested            | VERIFIED   | `scripts/12_smoke_patrl_foundation.py` (503L); `tests/test_smoke_patrl_foundation.py` (5 funcs + 3 parametrized = 7 nodeids) |
| 8   | Cluster evidence: 5 agents pass end-to-end on CPU partition                      | VERIFIED   | `cluster/logs/patrl18smoke_54894259.out` — "PAT-RL SMOKE PASSED (level=2, tune=500, draws=500)" in 13.9s wall |

**Score:** 8/8 truths verified

### Required Artifacts (Level 1/2/3)

| Artifact                                            | Exists | Substantive | Wired | Status   |
| --------------------------------------------------- | :----: | :---------: | :---: | -------- |
| `configs/pat_rl.yaml`                               | Y      | Y (114L)    | Y     | VERIFIED |
| `src/prl_hgf/env/pat_rl_config.py`                  | Y      | Y (733L)    | Y     | VERIFIED |
| `src/prl_hgf/env/pat_rl_sequence.py`                | Y      | Y (368L)    | Y     | VERIFIED |
| `src/prl_hgf/models/hgf_2level_patrl.py`            | Y      | Y (152L)    | Y     | VERIFIED |
| `src/prl_hgf/models/hgf_3level_patrl.py`            | Y      | Y (204L)    | Y     | VERIFIED |
| `src/prl_hgf/models/response_patrl.py`              | Y      | Y (141L)    | Y     | VERIFIED |
| `src/prl_hgf/fitting/hierarchical_patrl.py`         | Y      | Y (868L)    | Y     | VERIFIED |
| `src/prl_hgf/analysis/export_trajectories.py`       | Y      | Y (379L)    | Y     | VERIFIED |
| `scripts/12_smoke_patrl_foundation.py`              | Y      | Y (503L)    | Y     | VERIFIED |
| `tests/test_smoke_patrl_foundation.py`              | Y      | Y (115L)    | Y     | VERIFIED |
| `cluster/logs/patrl18smoke_54894259.out`            | Y      | Y (PASSED)  | —     | VERIFIED |

### Key Link Verification

| From                            | To                                         | Status  | Evidence                                                                               |
| ------------------------------- | ------------------------------------------ | ------- | -------------------------------------------------------------------------------------- |
| `hierarchical_patrl.py`         | `hgf_2level_patrl` + `hgf_3level_patrl`    | WIRED   | imports at L63-64; dispatch at L146-149                                                |
| `hierarchical_patrl.py`         | `pat_rl_config.load_pat_rl_config`         | WIRED   | import L54; default at L742                                                            |
| `hierarchical_patrl.py`         | BlackJAX NUTS helper                       | WIRED   | reuses `_run_blackjax_nuts` at L811 (imported from `hierarchical.py`)                  |
| `export_trajectories.py`        | `hgf_{2,3}level_patrl` builders            | WIRED   | imports at L41-49                                                                      |
| `export_trajectories.py`        | idata `participant_id` coord               | WIRED   | enabled by `coord_name="participant_id"` kwarg (commit d5c0b72)                        |
| `scripts/12_smoke_patrl_foundation.py` | `fit_batch_hierarchical_patrl`      | WIRED   | lazy-imported inside `_fit` at L162 (keeps dry-run independent of blackjax)            |
| `scripts/12_smoke_patrl_foundation.py` | `export_subject_{trajectories,parameters}` | WIRED | imports L38-40; called in `_export` at L227/238                                        |

### Parallel-Stack Invariant

`git log --oneline` on pick_best_cue files (`fitting/hierarchical.py`,
`env/task_config.py`, `env/simulator.py`, `models/hgf_{2,3}level.py`,
`models/response.py`, `configs/prl_analysis.yaml`) shows only one Phase-18
touch: **commit `d5c0b72`** on `hierarchical.py`.

That commit adds a `coord_name` kwarg to `_samples_to_idata` with default
`"participant"` (PRL back-compat preserved); `hierarchical_patrl.py` passes
`coord_name="participant_id"`. Backward-compatible kwarg addition — invariant
is intact per the verification prompt's explicit allowance.

No PAT-RL module imports `prl_hgf.env.task_config`, `prl_hgf.env.simulator`,
or the pick_best_cue HGF/response modules (grep-verified across all 7 PAT-RL
source files: only one docstring reference, zero runtime imports).

### Requirements Coverage

Phase 18 must-haves 1–8 (as enumerated in the verification prompt) map 1:1 to
the truths table above — all SATISFIED.

### Anti-Patterns Found

None blocking. The `_reserved` 4th RNG stream in `generate_session_patrl` is
intentional future-proofing (documented at `pat_rl_sequence.py:307`), not a
stub.

### Human Verification Required

None. All Option-A must-haves are structurally verifiable from the codebase
and cluster logs. Scientific validation (parameter recovery r>=0.7,
phenotype identifiability, BMS/PEB) is explicitly deferred to Phase 19+ per
ROADMAP line 235.

### Gaps Summary

No gaps. The PAT-RL task adaptation ships a complete, parallel Option-A stack:
config → sequence → binary-state HGFs → Model A logp → BlackJAX hierarchical
fit → DCM-interface CSV export, end-to-end validated by a 5-agent CPU cluster
smoke (job 54894259, 13.9s wall, passed) and a 7-test local regression suite.

---

*Verified: 2026-04-18*
*Verifier: Claude (gsd-verifier)*
