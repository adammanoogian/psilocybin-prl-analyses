---
phase: 03-simulation
verified: 2026-04-05T14:30:00Z
status: passed
score: 10/10 must-haves verified
gaps: []
---

# Phase 3: Simulation Verification Report

**Phase Goal:** Synthetic participants with known parameters produce realistic choice data, and batch simulation generates a complete group x session dataset ready for fitting.
**Verified:** 2026-04-05T14:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | High-beta agent (beta=5) with healthy_control omega_2 chooses best cue >=80% in late stable phases | VERIFIED | `test_simulate_agent_high_beta_accuracy` passes (seed=3, 83.33%); confirmed by inspecting the test at tests/test_agent.py:249-292 |
| 2 | Agent shows transient accuracy drops after reversals (volatile < stable late-phase) | VERIFIED | `test_simulate_agent_reversal_accuracy_drop` passes; volatile accuracy asserted strictly less than stable late accuracy |
| 3 | Batch generates 30/group x 2 groups x 3 sessions = 180 participant-sessions | VERIFIED | `simulate_batch` sets `n_total = n_groups * n_per_group * n_sessions` (batch.py:121); config has n_participants_per_group=30, 2 groups, 3 sessions; `test_batch_output_shape` validates shape |
| 4 | Each participant-session row has all required trial-level columns and true_* parameters | VERIFIED | `test_batch_column_values` checks all 19 columns from EXPECTED_COLUMNS set; `test_batch_output_shape` verifies no missing columns |
| 5 | Post-concussion group has lower kappa than healthy_control at baseline | VERIFIED | Config: PC kappa mean=0.8, HC kappa mean=1.0; `test_batch_group_param_differences` passes (HC > PC at N=2 with fixed seed) |
| 6 | Session 2 parameters shift from session 1 by configured deltas, with larger shift for post_concussion | VERIFIED | PC omega_2_deltas=[1.5, 0.8] vs HC=[0.5, 0.2]; `test_batch_session_deltas_visible` passes per-participant comparison; `test_sample_params_session_deltas_applied` verifies exact delta arithmetic |
| 7 | Same master_seed produces identical batch output | VERIFIED | `test_batch_reproducible` uses pd.testing.assert_frame_equal with check_exact=True |
| 8 | First trial has no stickiness effect (sentinel prev_choice=-1) | VERIFIED | `test_simulate_agent_first_trial_no_stickiness` passes: zeta=0 vs zeta=10 produce identical choice on trial 0 |
| 9 | Parameter sampling clips to model bounds after adding deltas | VERIFIED | `test_sample_params_within_bounds` (100 draws, both groups); `test_sample_params_clip_after_delta` checks clip-after-delta ordering explicitly |
| 10 | Pipeline script runs end-to-end and saves output CSV | VERIFIED | `test_batch_csv_output` verifies file creation and row count; script at scripts/03_simulate_participants.py is substantive (92 lines), has `if __name__ == "__main__":` guard |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Lines | Substantive | Wired | Status |
|----------|-------|-------------|-------|--------|
| `src/prl_hgf/simulation/__init__.py` | 37 | YES — exports 5 symbols | YES — imports from agent and batch | VERIFIED |
| `src/prl_hgf/simulation/agent.py` | 273 | YES — full implementation with PARAM_BOUNDS, SimulationResult, two public functions | YES — imported by batch.py and tests | VERIFIED |
| `src/prl_hgf/simulation/batch.py` | 214 | YES — full implementation with seed derivation, JIT prewarm, group loop, DataFrame assembly | YES — imported by __init__.py and pipeline script | VERIFIED |
| `scripts/03_simulate_participants.py` | 92 | YES — config load, output dir creation, summary stats, main() guard | YES — imports simulate_batch, called as entrypoint | VERIFIED |
| `tests/test_agent.py` | 370 | YES — 10 tests in two classes | YES — imports all agent symbols and runs via pytest | VERIFIED |
| `tests/test_batch.py` | 248 | YES — 6 slow-marked tests | YES — imports simulate_batch and runs via pytest | VERIFIED |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `agent.py` | `prl_hgf.models.hgf_2level.INPUT_NODES` | import at line 29 | WIRED | Used at lines 237-241 to read `expected_mean` for each cue node |
| `agent.py` | `prl_hgf.env.simulator.generate_reward` | import at line 27 | WIRED | Called at line 255 per trial |
| `agent.py` | `net.attributes[node]["expected_mean"]` | read before input_data | WIRED | Lines 235-241 read prior beliefs before update |
| `agent.py` | `net.attributes = net.last_attributes` | attribute carry pattern | WIRED | Line 269 — critical state threading pattern present |
| `batch.py` | `prl_hgf.simulation.agent.simulate_agent` | import at line 37 | WIRED | Called at line 167 per participant-session |
| `batch.py` | `prl_hgf.simulation.agent.sample_participant_params` | import at line 37 | WIRED | Called at line 152 per participant-session |
| `batch.py` | `prl_hgf.env.simulator.generate_session` | import at line 34 | WIRED | Called at line 164 per participant-session |
| `batch.py` | `prl_hgf.models.hgf_3level.build_3level_network` | import at line 36 | WIRED | Called at lines 54 (prewarm) and 157 (per participant) |
| `scripts/03_simulate_participants.py` | `prl_hgf.simulation.simulate_batch` | import at line 40 | WIRED | Called at line 69 inside main() |

---

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| Single agent with beta=5 achieves >80% in late stable phases | SATISFIED | test_simulate_agent_high_beta_accuracy: 83.33% with seed=3 |
| Batch generates 180 synthetic datasets with trial-level data and ground-truth params | SATISFIED | Production config: 2 groups x 30 participants x 3 sessions; test validates at N=2 |
| Two groups show visibly different parameter distributions | SATISFIED | Config encodes PC kappa_mean=0.8 vs HC kappa_mean=1.0; test_batch_group_param_differences verifies |
| Session 2 parameters shift by configured deltas, magnitude differs by group | SATISFIED | PC omega_2_delta=1.5 vs HC=0.5; test_batch_session_deltas_visible verifies per-participant |

---

### Anti-Patterns Found

None. No TODO, FIXME, placeholder, return null, or empty handler patterns found in any simulation files.

---

### Human Verification Required

None. All success criteria are verifiable programmatically via the test suite.

---

### Test Suite Results

All 61 tests pass in 357s total:
- 5 parameter sampling tests (fast): PASSED
- 5 agent simulation tests (slow, JAX): PASSED
- 6 batch simulation tests (slow, JAX): PASSED
- 15 env simulator tests: PASSED (no regressions)
- 18 model tests: PASSED (no regressions)
- 12 response model tests: PASSED (no regressions)

---

### Summary

Phase 3 goal is fully achieved. The simulation subsystem delivers:

1. A trial-by-trial HGF agent simulator using the attribute carry pattern (`net.attributes = net.last_attributes`) verified to produce realistic behavior: >80% accuracy in late stable acquisition phases and measurable accuracy drops after reversals.

2. A batch orchestrator that generates 30 participants/group x 2 groups x 3 sessions = 180 participant-sessions (420 trials each = 75,600 rows total) with ground-truth `true_*` parameter columns embedded. The upfront seed derivation from `master_seed` guarantees reproducibility.

3. Correctly encoded group x session design: post_concussion kappa_mean=0.8 vs healthy_control kappa_mean=1.0 at baseline; post_concussion omega_2 shifts +1.5 at post_dose vs +0.5 for healthy_controls.

4. A runnable pipeline script (`scripts/03_simulate_participants.py`) that produces a tidy CSV at `data/simulated/simulated_participants.csv`.

All code is lint-clean (ruff), type-hinted (Python 3.10+ syntax), and follows project conventions.

---

_Verified: 2026-04-05T14:30:00Z_
_Verifier: Claude (gsd-verifier)_
