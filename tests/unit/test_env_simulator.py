"""Unit tests for the task environment simulator (src/prl_hgf/env/simulator.py).

Covers trial count, phase structure, phase labels, cue probabilities,
best-cue tracking across reversals, seed reproducibility (ENV-05),
and stochastic reward generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from prl_hgf.env.simulator import generate_reward, generate_session
from prl_hgf.env.task_config import load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def trials(config):
    return generate_session(config, seed=42)


# ---------------------------------------------------------------------------
# Trial count tests
# ---------------------------------------------------------------------------


def test_total_trial_count(config, trials):
    expected = config.task.n_trials_total  # 3 * 140 = 420
    assert len(trials) == expected, (
        f"Expected {expected} trials, got {len(trials)}"
    )


def test_trials_per_set(config, trials):
    expected_per_set = config.task.n_trials_per_set  # 140
    for set_idx in range(config.task.n_sets):
        set_trials = [t for t in trials if t.set_idx == set_idx]
        assert len(set_trials) == expected_per_set, (
            f"Set {set_idx}: expected {expected_per_set} trials, "
            f"got {len(set_trials)}"
        )


def test_transfer_phase_present(config, trials):
    transfer_trials = [t for t in trials if t.phase_name == "transfer"]
    expected = config.task.n_sets * config.task.transfer.n_trials
    assert len(transfer_trials) == expected, (
        f"Expected {expected} transfer trials, got {len(transfer_trials)}"
    )


# ---------------------------------------------------------------------------
# Phase label tests
# ---------------------------------------------------------------------------


def test_phase_labels_present(trials):
    labels = {t.phase_label for t in trials}
    assert labels == {"stable", "volatile"}, (
        f"Expected labels {{'stable', 'volatile'}}, got {labels}"
    )


def test_acquisition_is_stable(trials):
    acquisition_trials = [
        t for t in trials if t.phase_name.startswith("acquisition")
    ]
    assert len(acquisition_trials) > 0, "No acquisition trials found"
    bad = [t for t in acquisition_trials if t.phase_label != "stable"]
    assert not bad, (
        f"Found {len(bad)} acquisition trials with label != 'stable': "
        f"{[t.phase_name for t in bad[:3]]}"
    )


def test_reversal_is_volatile(trials):
    reversal_trials = [
        t for t in trials if t.phase_name.startswith("reversal")
    ]
    assert len(reversal_trials) > 0, "No reversal trials found"
    bad = [t for t in reversal_trials if t.phase_label != "volatile"]
    assert not bad, (
        f"Found {len(bad)} reversal trials with label != 'volatile': "
        f"{[t.phase_name for t in bad[:3]]}"
    )


# ---------------------------------------------------------------------------
# Cue probability tests
# ---------------------------------------------------------------------------


def test_cue_probs_match_config(config, trials):
    phase_map = {p.name: p.cue_probs for p in config.task.phases}
    phase_map["transfer"] = config.task.transfer.cue_probs

    for trial in trials:
        expected_probs = tuple(phase_map[trial.phase_name])
        assert trial.cue_probs == expected_probs, (
            f"Trial {trial.trial_idx} phase '{trial.phase_name}': "
            f"expected cue_probs {expected_probs}, got {trial.cue_probs}"
        )


def test_cue_probs_are_valid(config, trials):
    n_cues = config.task.n_cues
    for trial in trials:
        assert len(trial.cue_probs) == n_cues, (
            f"Trial {trial.trial_idx}: expected {n_cues} cue_probs, "
            f"got {len(trial.cue_probs)}"
        )
        for i, p in enumerate(trial.cue_probs):
            assert 0.0 <= p <= 1.0, (
                f"Trial {trial.trial_idx}: cue_probs[{i}] = {p} "
                f"not in [0.0, 1.0]"
            )


# ---------------------------------------------------------------------------
# Best-cue tests
# ---------------------------------------------------------------------------


def test_best_cue_changes_across_phases(config, trials):
    phase_best: dict[str, set[int]] = {}
    for trial in trials:
        phase_best.setdefault(trial.phase_name, set()).add(trial.best_cue)

    # acquisition_1: cue_A dominant → index 0
    assert phase_best["acquisition_1"] == {0}, (
        f"acquisition_1 best_cue: expected {{0}}, got {phase_best['acquisition_1']}"
    )
    # reversal_1: cue_B dominant → index 1
    assert phase_best["reversal_1"] == {1}, (
        f"reversal_1 best_cue: expected {{1}}, got {phase_best['reversal_1']}"
    )
    # acquisition_2: cue_A dominant → index 0
    assert phase_best["acquisition_2"] == {0}, (
        f"acquisition_2 best_cue: expected {{0}}, got {phase_best['acquisition_2']}"
    )
    # reversal_2: cue_C dominant → index 2
    assert phase_best["reversal_2"] == {2}, (
        f"reversal_2 best_cue: expected {{2}}, got {phase_best['reversal_2']}"
    )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed(config):
    trials_a = generate_session(config, seed=42)
    trials_b = generate_session(config, seed=42)
    assert trials_a == trials_b, (
        "Same seed=42 produced different trial sequences (ENV-05 failure)"
    )


def test_different_seeds_same_structure(config):
    trials_42 = generate_session(config, seed=42)
    trials_99 = generate_session(config, seed=99)
    assert len(trials_42) == len(trials_99), (
        f"Different seeds produced different trial counts: "
        f"seed=42 → {len(trials_42)}, seed=99 → {len(trials_99)}"
    )
    # Phase structure is deterministic — only rewards differ between seeds
    for i, (ta, tb) in enumerate(zip(trials_42, trials_99, strict=True)):
        assert ta.phase_name == tb.phase_name, (
            f"Trial {i}: phase_name differs between seeds "
            f"({ta.phase_name!r} vs {tb.phase_name!r})"
        )
        assert ta.cue_probs == tb.cue_probs, (
            f"Trial {i}: cue_probs differ between seeds"
        )


# ---------------------------------------------------------------------------
# Sequential index test
# ---------------------------------------------------------------------------


def test_trial_idx_sequential(trials):
    for expected_idx, trial in enumerate(trials):
        assert trial.trial_idx == expected_idx, (
            f"Trial at position {expected_idx} has trial_idx={trial.trial_idx} "
            f"(expected {expected_idx})"
        )


# ---------------------------------------------------------------------------
# generate_reward tests
# ---------------------------------------------------------------------------


def test_generate_reward_deterministic_with_seed():
    probs = (0.8, 0.2, 0.5)
    rewards_a = []
    rng_a = np.random.default_rng(7)
    for cue in range(3):
        rewards_a.append(generate_reward(cue, probs, rng_a))

    rewards_b = []
    rng_b = np.random.default_rng(7)
    for cue in range(3):
        rewards_b.append(generate_reward(cue, probs, rng_b))

    assert rewards_a == rewards_b, (
        f"Same seed produced different rewards: {rewards_a} vs {rewards_b}"
    )


def test_generate_reward_respects_probability():
    rng = np.random.default_rng(0)
    probs: tuple[float, ...] = (1.0, 0.0, 0.0)

    # cue 0 has p=1.0 → always rewarded
    for _ in range(20):
        reward = generate_reward(0, probs, rng)
        assert reward == 1, (
            f"cue 0 with p=1.0: expected reward=1, got {reward}"
        )

    # cue 1 has p=0.0 → never rewarded
    for _ in range(20):
        reward = generate_reward(1, probs, rng)
        assert reward == 0, (
            f"cue 1 with p=0.0: expected reward=0, got {reward}"
        )
