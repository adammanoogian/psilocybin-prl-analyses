"""Tests for the PAT-RL binary-state trial sequence generator.

Covers: session shape, run structure, hazard-rate statistics, 2x2 magnitude
coverage, state-conditioned Delta-HR, determinism, override, monotone outcome
times, and pick_best_cue import regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_sequence import generate_session_patrl

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load default PAT-RL config once for the module."""
    return load_pat_rl_config()


@pytest.fixture(scope="module")
def session_42(cfg):
    """Generate a single 192-trial session with seed=42."""
    return generate_session_patrl(cfg, seed=42)


# ---------------------------------------------------------------------------
# Test 1: Session shape and basic invariants
# ---------------------------------------------------------------------------


def test_session_shape(cfg, session_42):
    """Session has exactly n_trials trials with valid state values."""
    trials = session_42
    assert len(trials) == cfg.task.n_trials, (
        f"Expected {cfg.task.n_trials} trials, got {len(trials)}"
    )
    # Trial indices are sequential
    for i, t in enumerate(trials):
        assert t.trial_idx == i, (
            f"Expected trial_idx={i}, got {t.trial_idx}"
        )
    # All states are binary
    states = {t.state for t in trials}
    assert states <= {0, 1}, f"Expected states in {{0, 1}}, got {states}"


# ---------------------------------------------------------------------------
# Test 2: Run structure integrity
# ---------------------------------------------------------------------------


def test_run_structure(cfg, session_42):
    """Run index, trial-in-run, and regime labels are self-consistent."""
    trials_per_run = cfg.task.trials_per_run
    run_order = cfg.task.run_order
    for t in session_42:
        expected_run = t.trial_idx // trials_per_run
        expected_in_run = t.trial_idx % trials_per_run
        assert t.run_idx == expected_run, (
            f"trial {t.trial_idx}: expected run_idx={expected_run}, "
            f"got {t.run_idx}"
        )
        assert t.trial_in_run == expected_in_run, (
            f"trial {t.trial_idx}: expected trial_in_run={expected_in_run}, "
            f"got {t.trial_in_run}"
        )
        assert 0 <= t.trial_in_run < trials_per_run, (
            f"trial_in_run {t.trial_in_run} out of range [0, {trials_per_run})"
        )
        expected_regime = run_order[expected_run]
        assert t.regime == expected_regime, (
            f"trial {t.trial_idx}: expected regime={expected_regime!r}, "
            f"got {t.regime!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: Hazard rate produces correct flip statistics
# ---------------------------------------------------------------------------


def test_hazard_rate_approximate(cfg):
    """Stable runs flip less than volatile; both within ±3 SD of expectation.

    Over 100 seeded sessions: collect mean flip counts per regime type.
    Stable hazard = 0.03 -> expected flips = 0.03 * 47 ≈ 1.41
    Volatile hazard = 0.10 -> expected flips = 0.10 * 47 ≈ 4.70
    """
    n_sessions = 100
    trials_per_run = cfg.task.trials_per_run
    n_per_run = trials_per_run  # alias
    run_order = cfg.task.run_order
    stable_h = cfg.task.hazards.stable
    volatile_h = cfg.task.hazards.volatile

    stable_flip_counts: list[float] = []
    volatile_flip_counts: list[float] = []

    rng_seed = np.random.default_rng(999)

    for _ in range(n_sessions):
        seed = int(rng_seed.integers(0, 2**31))
        trials = generate_session_patrl(cfg, seed=seed)
        for run_idx in range(cfg.task.n_runs):
            start = run_idx * n_per_run
            run_trials = trials[start : start + n_per_run]
            states = [t.state for t in run_trials]
            flips = sum(
                1
                for a, b in zip(states[:-1], states[1:], strict=True)
                if a != b
            )
            if run_order[run_idx] == "stable":
                stable_flip_counts.append(flips)
            else:
                volatile_flip_counts.append(flips)

    mean_stable = np.mean(stable_flip_counts)
    mean_volatile = np.mean(volatile_flip_counts)

    assert mean_stable < mean_volatile, (
        f"Expected mean_stable ({mean_stable:.3f}) < mean_volatile "
        f"({mean_volatile:.3f})"
    )

    # Within ±3 SD of Bernoulli expectation (n = 47 flip opportunities)
    n_opps = trials_per_run - 1

    for h, mean_val, label in (
        (stable_h, mean_stable, "stable"),
        (volatile_h, mean_volatile, "volatile"),
    ):
        expected = h * n_opps
        # Variance of sum of n_opps Bernoulli(h) = n_opps * h * (1-h)
        # Variance of sample mean over n_sessions = n_opps * h * (1-h) / n_sessions
        sd_of_mean = np.sqrt(n_opps * h * (1 - h) / n_sessions)
        assert abs(mean_val - expected) <= 3 * sd_of_mean, (
            f"{label}: mean flip count {mean_val:.3f} is more than 3 SD "
            f"from expected {expected:.3f} (SD={sd_of_mean:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 4: All 4 magnitude combinations covered
# ---------------------------------------------------------------------------


def test_magnitudes_cover_2x2(cfg, session_42):
    """All four (reward, shock) magnitude combinations appear >= 10 times."""
    reward_levels = cfg.task.magnitudes.reward_levels
    shock_levels = cfg.task.magnitudes.shock_levels
    expected_combos = {
        (r, s)
        for r in reward_levels
        for s in shock_levels
    }
    from collections import Counter
    combo_counts: Counter[tuple[float, float]] = Counter(
        (t.reward_mag, t.shock_mag) for t in session_42
    )
    for combo in expected_combos:
        count = combo_counts[combo]
        assert count >= 10, (
            f"Combo {combo} appeared only {count} times; expected >= 10"
        )
    assert set(combo_counts.keys()) == expected_combos, (
        f"Unexpected combos: {set(combo_counts.keys()) - expected_combos}"
    )


# ---------------------------------------------------------------------------
# Test 5: Delta-HR is state-conditioned with correct means and bounds
# ---------------------------------------------------------------------------


def test_delta_hr_state_conditioned(cfg):
    """Delta-HR has state-conditioned means and is clipped to bounds.

    Over 500 sessions:
    - mean(delta_hr | state=1) < 0  (bradycardia on dangerous)
    - mean(delta_hr | state=0) is within 0.3 bpm of zero
    - all values in [bounds[0], bounds[1]]
    """
    n_sessions = 500
    lo, hi = cfg.task.delta_hr_stub.bounds
    rng_seed = np.random.default_rng(7777)

    dangerous_vals: list[float] = []
    safe_vals: list[float] = []

    for _ in range(n_sessions):
        seed = int(rng_seed.integers(0, 2**31))
        trials = generate_session_patrl(cfg, seed=seed)
        for t in trials:
            # Bounds check
            assert lo <= t.delta_hr <= hi, (
                f"delta_hr {t.delta_hr:.3f} outside bounds [{lo}, {hi}]"
            )
            if t.state == 1:
                dangerous_vals.append(t.delta_hr)
            else:
                safe_vals.append(t.delta_hr)

    mean_dangerous = np.mean(dangerous_vals)
    mean_safe = np.mean(safe_vals)

    assert mean_dangerous < 0, (
        f"Expected mean delta_hr on dangerous state < 0, got {mean_dangerous:.4f}"
    )
    assert abs(mean_safe) < 0.3, (
        f"Expected mean delta_hr on safe state within 0.3 of 0, "
        f"got {mean_safe:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Determinism under fixed seed
# ---------------------------------------------------------------------------


def test_determinism(cfg):
    """Two calls with the same seed produce identical PATRLTrial lists."""
    trials_a = generate_session_patrl(cfg, seed=123)
    trials_b = generate_session_patrl(cfg, seed=123)
    assert trials_a == trials_b, (
        "Two calls with seed=123 produced different trial lists"
    )


# ---------------------------------------------------------------------------
# Test 7: Delta-HR override is applied and clipped
# ---------------------------------------------------------------------------


def test_delta_hr_override_applied(cfg):
    """A supplied delta_hr_override of 2.5 is applied to every trial."""
    override = np.full(cfg.task.n_trials, 2.5)
    trials = generate_session_patrl(cfg, seed=0, delta_hr_override=override)
    for t in trials:
        assert t.delta_hr == pytest.approx(2.5), (
            f"Expected delta_hr=2.5 from override, got {t.delta_hr}"
        )


# ---------------------------------------------------------------------------
# Test 8: Outcome times are monotone increasing with correct spacing
# ---------------------------------------------------------------------------


def test_outcome_time_s_monotone_increasing(cfg, session_42):
    """Outcome times are strictly increasing; spacing matches timing config."""
    trials = session_42
    trial_dur = cfg.task.timing.trial_duration_s  # 11.0 s
    run_gap_s = 15.0

    prev_time = trials[0].outcome_time_s
    for i in range(1, len(trials)):
        t = trials[i]
        assert t.outcome_time_s > prev_time, (
            f"Outcome times not strictly increasing at trial {i}: "
            f"{prev_time:.3f} -> {t.outcome_time_s:.3f}"
        )
        # Within-run spacing should equal trial_duration_s
        if t.trial_in_run > 0:
            spacing = t.outcome_time_s - trials[i - 1].outcome_time_s
            assert spacing == pytest.approx(trial_dur, rel=1e-9), (
                f"Within-run spacing at trial {i}: expected {trial_dur}, "
                f"got {spacing:.6f}"
            )
        # At run boundary: gap >= trial_duration_s + run_gap_s
        elif t.trial_in_run == 0 and t.run_idx > 0:
            gap = t.outcome_time_s - trials[i - 1].outcome_time_s
            min_gap = trial_dur + run_gap_s
            assert gap >= min_gap - 1e-9, (
                f"Run boundary gap at trial {i}: expected >= {min_gap}, "
                f"got {gap:.6f}"
            )
        prev_time = t.outcome_time_s


# ---------------------------------------------------------------------------
# Test 9: pick_best_cue regression — simulator still importable
# ---------------------------------------------------------------------------


def test_pick_best_cue_still_loadable():
    """Importing prl_hgf.env.simulator does not fail after adding pat_rl_sequence."""
    from prl_hgf.env.simulator import generate_session  # noqa: F401
