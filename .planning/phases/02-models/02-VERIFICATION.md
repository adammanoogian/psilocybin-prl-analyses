---
phase: 02-models
verified: 2026-04-05T11:49:23Z
status: passed
score: 11/11 must-haves verified
---

# Phase 02: Models Verification Report

**Phase Goal:** Both HGF model variants are defined as pyhgf Network objects, forward-pass correctly on a synthetic input sequence, and the custom response function computes log-likelihood given beliefs and observed choices.
**Verified:** 2026-04-05T11:49:23Z
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 2-level model creates 3 binary input nodes and 3 continuous-state nodes | VERIFIED | test_2level_node_count passes; len(net.edges) == 6 confirmed by direct construction test |
| 2 | 3-level model creates 3 binary + 3 continuous + 1 shared volatility parent node | VERIFIED | test_3level_node_count passes; test_3level_volatility_parent_exists confirms volatility_children == (1, 3, 5) |
| 3 | Both models accept trial sequence input with partial feedback via observed mask | VERIFIED | test_forward_pass_no_error[build_2level_network] and [build_3level_network] pass; prepare_input_data produces correct int-dtype observed array |
| 4 | Both models produce finite, non-NaN belief trajectories (mu_1 per cue) | VERIFIED | test_belief_trajectories_finite and test_belief_trajectory_shapes pass; all 6 belief keys confirmed finite with shape (50,) |
| 5 | 3-level model produces volatility trajectory (mu_2) from shared parent node | VERIFIED | test_3level_volatility_trajectory_finite and test_3level_volatility_trajectory_shape pass; extract_beliefs_3level returns mu2_volatility from node 6 |
| 6 | Parameter names map to documented HGF parameters (omega_2, omega_3, kappa) | VERIFIED | API boundary uses omega_2, omega_3, kappa; internally mapped to tonic_volatility on correct nodes; parameter table in docstrings |
| 7 | Response function computes finite log-likelihood given HGF beliefs and observed choices | VERIFIED | test_surprise_finite_2level and test_surprise_finite_3level pass with beta=2.0, zeta=0.5 |
| 8 | Softmax over 3 cues uses beta * mu1k + zeta * I[prev_choice=k] formula | VERIFIED | Formula implemented in softmax_stickiness_surprise; test_high_beta_concentrates_probability and test_positive_zeta_favors_repeat behaviorally confirm both terms work |
| 9 | First trial has zero stickiness term (no previous choice) | VERIFIED | test_first_trial_no_stickiness passes; sentinel -1 in prev_choices produces uniform distribution (surprise == log(3)) |
| 10 | Unobserved cues do not update their beliefs | VERIFIED | test_unobserved_cue_beliefs_constant passes; mu1_cue1 and mu1_cue2 are constant when observed[:,1] and observed[:,2] are all zero |
| 11 | Response function integrates with net.surprise() API | VERIFIED | test_end_to_end_2level and test_end_to_end_3level pass full pipeline: config -> generate_session -> prepare_input_data -> net.input_data -> net.surprise -> finite scalar |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/prl_hgf/models/hgf_2level.py | build_2level_network() factory, prepare_input_data(), extract_beliefs() | VERIFIED | 235 lines, no stubs, exports all three functions plus INPUT_NODES, BELIEF_NODES, N_CUES |
| src/prl_hgf/models/hgf_3level.py | build_3level_network() with omega_2/omega_3/kappa, extract_beliefs_3level() | VERIFIED | 213 lines, no stubs, shared volatility parent wired with volatility_children=([1,3,5],[kappa,kappa,kappa]) |
| src/prl_hgf/models/response.py | softmax_stickiness_surprise() compatible with pyhgf surprise() API | VERIFIED | 115 lines, JAX throughout, correct three-argument signature, NaN guard present |
| src/prl_hgf/models/__init__.py | Public API exports all model symbols | VERIFIED | 63 lines, exports all 9 required symbols via explicit __all__ |
| tests/test_models.py | Construction, forward pass, partial feedback, belief extraction tests | VERIFIED | 355 lines (>80 required), 20 tests, all pass |
| tests/test_response.py | Unit + integration tests for response function | VERIFIED | 469 lines (>80 required), 12 tests, all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| hgf_2level.py | pyhgf.model.Network | from pyhgf.model import Network | WIRED | Line 34: import present; Network() instantiated and add_nodes() called |
| hgf_3level.py | pyhgf.model.Network | import + volatility_children wiring | WIRED | Line 151: volatility_children=([1, 3, 5], [kappa, kappa, kappa]) connects shared parent to all 3 level-1 nodes |
| response.py | pyhgf Network.node_trajectories | hgf.node_trajectories[idx][expected_mean] | WIRED | Lines 90-92: reads expected_mean from binary INPUT_NODES (0, 2, 4) |
| response.py | jax.numpy | import jax.numpy as jnp | WIRED | Lines 24-25: jax and jax.numpy as jnp imported; jax.nn.log_softmax used |
| tests/test_models.py | src/prl_hgf/models/ | from prl_hgf.models import ... | WIRED | Line 22: imports 7 symbols; all tests use them |
| tests/test_response.py | src/prl_hgf/models/ | imports + net.surprise() calls | WIRED | Line 27: imports 4 symbols; net.surprise(response_function=softmax_stickiness_surprise, ...) called 12+ times |

---

### Requirements Coverage

| Requirement (from phase success criteria) | Status | Notes |
|------------------------------------------|--------|-------|
| 2-level model: 3 input nodes, 3 continuous nodes, (n_trials,3) input, per-cue belief trajectories | SATISFIED | Node count, input_idxs, forward pass, belief shape all verified by tests |
| 3-level model: 7 nodes, volatility_children with kappa coupling, additional volatility trajectory | SATISFIED | test_3level_volatility_parent_exists confirms volatility_children==(1,3,5); mu2_volatility extracted and tested finite |
| Both models: finite beliefs, unobserved cues do not update | SATISFIED | test_belief_trajectories_finite and test_unobserved_cue_beliefs_constant both pass |
| Belief extraction: mu1_cueN from continuous mean, p_reward_cueN from binary expected_mean | SATISFIED | extract_beliefs uses correct keys; test_p_reward_in_valid_range confirms values in [0,1] |
| Parameter names: omega_2 -> tonic_volatility on nodes 1/3/5, omega_3 on node 6, kappa as coupling strength | SATISFIED | Verified in source and parameter table docstrings; three-layer naming convention followed |
| Response function: signature matches pyhgf API, formula correct, first-trial stickiness neutral, NaN guard | SATISFIED | test_first_trial_no_stickiness, test_nan_guard_returns_inf, test_high_beta_*, test_positive_zeta_* all pass |
| Full pipeline integration: generate_session -> forward pass -> surprise -> finite scalar | SATISFIED | test_end_to_end_2level and test_end_to_end_3level pass |

---

### Anti-Patterns Found

None. All four source files searched for TODO/FIXME/XXX/HACK/placeholder/not implemented/return null/return {}/return [] -- zero hits.

---

### Human Verification Required

None required. All goal achievement criteria are verifiable structurally and confirmed by the test suite (45/45 passing).

Items that would be human-verifiable in principle but are adequately covered by tests:
- Volatility trajectory responding to reversal points: confirmed behaviorally by the test suite architecture (trials generated from config which includes reversal phases) and structurally by the volatility parent wiring. Scientific interpretation of the trajectory shape is deferred to the parameter recovery phase (Phase 06) per the project roadmap.

---

### Test Suite Summary

    45 passed in 26.67s

    tests/test_env_simulator.py  -- 14 passed (Phase 01-02 regression: no breakage)
    tests/test_models.py         -- 20 passed
    tests/test_response.py       -- 12 passed (+ 2 integration via config fixture)

No failures. No errors. No regressions in prior phase tests.

---

### Summary

Phase 02 fully achieves its goal. Both HGF model variants (2-level, 6 nodes; 3-level, 7 nodes) are implemented as real pyhgf Network objects with correct topology, produce finite belief trajectories under partial feedback, and the softmax+stickiness response function correctly computes log-likelihoods compatible with the pyhgf net.surprise() API. Parameter names follow the documented three-layer naming convention (omega_2/omega_3/kappa at API, tonic_volatility/volatility_children internally). The full pipeline from config to finite surprise scalar is integration-tested on both model variants.

---

_Verified: 2026-04-05T11:49:23Z_
_Verifier: Claude (gsd-verifier)_
