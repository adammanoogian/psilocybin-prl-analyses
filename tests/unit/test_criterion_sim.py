"""Unit tests for criterion-based (closed-loop) task simulation.

Covers the ``task.reversal`` schema extension in
:mod:`prl_hgf.env.task_config` (valid configs plus error cases), the
criterion mechanics in :mod:`prl_hgf.simulation.criterion_sim` (jittered
consecutive-correct thresholds, window criterion, max-trials cap), the
``random_nonbest`` reversal target rule, seed reproducibility, and the
output CSV schema of the closed-loop batch path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prl_hgf.env.task_config import (
    CriterionConfig,
    CueProbPair,
    ReversalConfig,
    TaskConfig,
    TransferConfig,
    load_config,
)
from prl_hgf.simulation.batch import simulate_batch
from prl_hgf.simulation.criterion_sim import (
    ConsecutiveCorrectTracker,
    CriterionEnvironment,
    WindowTracker,
    draw_new_best_cue,
    simulate_criterion_session,
)

# ---------------------------------------------------------------------------
# Config helpers and fixtures
# ---------------------------------------------------------------------------

CRITERION_YAML = """\
task:
  name: pick_best_cue_criterion
  n_cues: 3
  n_sets: 2
  initial_best_per_set: [0, 1]
  cue_probs: {best: 0.80, other: 0.20}
  reversal:
    criterion:
      type: consecutive_correct
      n_correct_min: 2
      n_correct_max: 3
      window: null
    max_trials_per_phase: 6
    n_reversals_per_set: 2
    target_rule: random_nonbest
  transfer: {phase_type: stable, n_trials: 3, cue_probs: [0.33, 0.33, 0.33]}

simulation:
  n_participants_per_group: 1
  master_seed: 7
  groups:
    placebo:
      omega_2: {mean: -3.0, sd: 0.5}
      omega_3: {mean: -6.0, sd: 0.5}
      kappa: {mean: 1.0, sd: 0.2}
      beta: {mean: 6.0, sd: 0.5}
      zeta: {mean: 0.0, sd: 0.1}
  session_deltas:
    placebo:
      session_labels: []
      omega_2_deltas: []
      kappa_deltas: []
      beta_deltas: []
      zeta_deltas: []

fitting:
  n_chains: 1
  n_draws: 10
  n_tune: 10
  target_accept: 0.9
  random_seed: 0
"""


def _load_yaml_variant(tmp_path, yaml_text):
    """Write a YAML string to a temp file and load it via load_config."""
    path = tmp_path / "criterion.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return load_config(path)


def make_criterion_task(
    n_cues=3,
    n_sets=1,
    initial_best_per_set=(0,),
    n_correct_min=3,
    n_correct_max=3,
    criterion_type="consecutive_correct",
    window_size=None,
    window_n_correct=None,
    max_trials_per_phase=6,
    n_reversals_per_set=2,
    target_rule="random_nonbest",
    transfer_n_trials=3,
):
    """Construct a criterion-based TaskConfig directly from dataclasses."""
    criterion = CriterionConfig(
        criterion_type=criterion_type,
        n_correct_min=n_correct_min,
        n_correct_max=n_correct_max,
        window_size=window_size,
        window_n_correct=window_n_correct,
    )
    return TaskConfig(
        name="pick_best_cue_criterion",
        description="",
        n_cues=n_cues,
        cue_labels=[f"cue_{i}" for i in range(n_cues)],
        n_sets=n_sets,
        phases=[],
        transfer=TransferConfig(
            phase_type="stable",
            n_trials=transfer_n_trials,
            cue_probs=[round(1.0 / n_cues, 2)] * n_cues,
        ),
        partial_feedback=True,
        task_seed=0,
        reversal=ReversalConfig(
            criterion=criterion,
            max_trials_per_phase=max_trials_per_phase,
            n_reversals_per_set=n_reversals_per_set,
            target_rule=target_rule,
        ),
        initial_best_per_set=list(initial_best_per_set),
        cue_prob_pair=CueProbPair(best=0.8, other=0.2),
    )


@pytest.fixture(scope="module")
def crit_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("crit_cfg") / "criterion.yaml"
    path.write_text(CRITERION_YAML, encoding="utf-8")
    return load_config(path)


@pytest.fixture(scope="module")
def batch_df(crit_config):
    return simulate_batch(crit_config)


# ---------------------------------------------------------------------------
# Schema: valid criterion config
# ---------------------------------------------------------------------------


def test_criterion_config_loads(crit_config):
    task = crit_config.task
    assert crit_config.is_criterion_based is True
    assert task.is_criterion_based is True
    assert task.phases == []
    assert task.reversal is not None
    assert task.reversal.criterion.criterion_type == "consecutive_correct"
    assert task.reversal.criterion.n_correct_min == 2
    assert task.reversal.criterion.n_correct_max == 3
    assert task.reversal.max_trials_per_phase == 6
    assert task.reversal.n_reversals_per_set == 2
    assert task.reversal.target_rule == "random_nonbest"
    assert task.cue_prob_pair == CueProbPair(best=0.8, other=0.2)
    assert task.initial_best_per_set == [0, 1]


def test_criterion_config_trial_count_properties(crit_config):
    task = crit_config.task
    # (2 reversals + 1 acquisition) * 6 max trials + 3 transfer = 21
    assert task.max_trials_per_set == 21
    with pytest.raises(ValueError, match="n_trials_per_set is undefined"):
        _ = task.n_trials_per_set
    with pytest.raises(ValueError, match="n_trials_per_set is undefined"):
        _ = task.n_trials_total


def test_default_fixed_phase_config_unaffected():
    config = load_config()
    assert config.is_criterion_based is False
    assert config.task.is_criterion_based is False
    assert config.task.n_trials_total == 420
    assert config.task.max_trials_per_set == config.task.n_trials_per_set


# ---------------------------------------------------------------------------
# Schema: error cases
# ---------------------------------------------------------------------------


def test_invalid_criterion_type_rejected(tmp_path):
    bad = CRITERION_YAML.replace("type: consecutive_correct", "type: bogus")
    with pytest.raises(ValueError, match="criterion type.*bogus"):
        _load_yaml_variant(tmp_path, bad)


def test_min_greater_than_max_rejected(tmp_path):
    bad = CRITERION_YAML.replace("n_correct_min: 2", "n_correct_min: 7")
    with pytest.raises(ValueError, match="n_correct_max must be >= n_correct_min"):
        _load_yaml_variant(tmp_path, bad)


def test_cue_prob_out_of_range_rejected(tmp_path):
    bad = CRITERION_YAML.replace("best: 0.80", "best: 1.50")
    with pytest.raises(ValueError, match=r"'best' must be in \[0.0, 1.0\]"):
        _load_yaml_variant(tmp_path, bad)


def test_bad_target_rule_rejected(tmp_path):
    bad = CRITERION_YAML.replace(
        "target_rule: random_nonbest", "target_rule: sticky_best"
    )
    with pytest.raises(ValueError, match="target_rule.*sticky_best"):
        _load_yaml_variant(tmp_path, bad)


def test_initial_best_wrong_length_rejected(tmp_path):
    bad = CRITERION_YAML.replace(
        "initial_best_per_set: [0, 1]", "initial_best_per_set: [0]"
    )
    with pytest.raises(ValueError, match="initial_best_per_set length"):
        _load_yaml_variant(tmp_path, bad)


def test_phases_and_reversal_mutually_exclusive(tmp_path):
    bad = CRITERION_YAML.replace(
        "  reversal:",
        "  phases:\n"
        "    - {name: acquisition_1, phase_type: stable, n_trials: 30,"
        " cue_probs: [0.8, 0.2, 0.2]}\n"
        "  reversal:",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load_yaml_variant(tmp_path, bad)


def test_window_type_requires_window_block(tmp_path):
    bad = CRITERION_YAML.replace("type: consecutive_correct", "type: window")
    with pytest.raises(ValueError, match="requires window.size"):
        _load_yaml_variant(tmp_path, bad)


def test_window_type_accepts_window_block(tmp_path):
    good = CRITERION_YAML.replace("type: consecutive_correct", "type: window").replace(
        "window: null", "window: {size: 10, n_correct: 8}"
    )
    config = _load_yaml_variant(tmp_path, good)
    criterion = config.task.reversal.criterion
    assert criterion.window_size == 10
    assert criterion.window_n_correct == 8


def test_window_n_correct_exceeding_size_rejected():
    with pytest.raises(ValueError, match=r"window.n_correct must be in \[1"):
        CriterionConfig(criterion_type="window", window_size=5, window_n_correct=6)


# ---------------------------------------------------------------------------
# Criterion trackers
# ---------------------------------------------------------------------------


def test_consecutive_tracker_fires_at_threshold():
    tracker = ConsecutiveCorrectTracker(threshold=3)
    assert tracker.update(True) is False
    assert tracker.update(True) is False
    assert tracker.update(True) is True


def test_consecutive_tracker_resets_on_nonbest_choice():
    tracker = ConsecutiveCorrectTracker(threshold=3)
    assert tracker.update(True) is False
    assert tracker.update(True) is False
    assert tracker.update(False) is False  # reset
    assert tracker.update(True) is False
    assert tracker.update(True) is False
    assert tracker.update(True) is True


def test_window_tracker_counts_within_sliding_window():
    tracker = WindowTracker(window_size=4, n_correct_required=3)
    # W B B -> 2/3 in window, not met; B -> window [W,B,B,B] = 3, met.
    assert tracker.update(False) is False
    assert tracker.update(True) is False
    assert tracker.update(True) is False
    assert tracker.update(True) is True


def test_window_tracker_evicts_old_trials():
    tracker = WindowTracker(window_size=2, n_correct_required=2)
    tracker.update(True)
    tracker.update(False)
    assert tracker.update(True) is False  # window [F, T]
    assert tracker.update(True) is True  # window [T, T]


# ---------------------------------------------------------------------------
# Environment mechanics (scripted choices, no HGF agent)
# ---------------------------------------------------------------------------


def _run_scripted(task_cfg, choose, seed=0):
    """Run the environment with a scripted choice policy; return trial rows."""
    env = CriterionEnvironment(task_cfg, np.random.default_rng(seed))
    records = []
    while not env.done:
        trial = env.current_trial()
        choice = choose(trial)
        env.step(choice)
        records.append((trial, choice))
    return records


def test_reversal_fires_after_fixed_threshold():
    # min == max == 3: an always-correct agent ends each learning phase in
    # exactly 3 trials.
    cfg = make_criterion_task(n_correct_min=3, n_correct_max=3)
    records = _run_scripted(cfg, lambda t: t.best_cue)
    phase_names = [t.phase_name for t, _ in records]
    assert phase_names == (
        ["acquisition_1"] * 3
        + ["reversal_1"] * 3
        + ["reversal_2"] * 3
        + ["transfer"] * 3
    ), f"Expected 3-trial phases, got {phase_names}"


def test_jittered_threshold_stays_in_bounds_and_varies():
    cfg = make_criterion_task(n_correct_min=2, n_correct_max=6, n_sets=1)
    lengths = []
    for seed in range(30):
        records = _run_scripted(cfg, lambda t: t.best_cue, seed=seed)
        phase_names = [t.phase_name for t, _ in records]
        for name in ("acquisition_1", "reversal_1", "reversal_2"):
            lengths.append(phase_names.count(name))
    assert min(lengths) >= 2, f"Expected lengths >= 2, got min {min(lengths)}"
    assert max(lengths) <= 6, f"Expected lengths <= 6, got max {max(lengths)}"
    assert len(set(lengths)) >= 3, (
        f"Expected jittered thresholds to vary, got only {sorted(set(lengths))}"
    )


def test_max_trials_cap_forces_reversal():
    # An always-wrong agent never meets the criterion: every learning phase
    # must end at max_trials_per_phase.
    cfg = make_criterion_task(max_trials_per_phase=5)
    records = _run_scripted(cfg, lambda t: (t.best_cue + 1) % 3)
    phase_names = [t.phase_name for t, _ in records]
    for name in ("acquisition_1", "reversal_1", "reversal_2"):
        assert phase_names.count(name) == 5, (
            f"Phase {name}: expected 5 trials (max cap), got {phase_names.count(name)}"
        )


def test_window_criterion_in_environment():
    cfg = make_criterion_task(
        criterion_type="window",
        n_correct_min=None,
        n_correct_max=None,
        window_size=4,
        window_n_correct=3,
        n_reversals_per_set=0,
        transfer_n_trials=1,
    )
    # Wrong on trial 0, correct afterwards: window fills to 3 correct on
    # trial 3 (0-based), so acquisition lasts exactly 4 trials.
    records = _run_scripted(
        cfg, lambda t: t.best_cue if t.trial_idx > 0 else (t.best_cue + 1) % 3
    )
    phase_names = [t.phase_name for t, _ in records]
    assert phase_names == ["acquisition_1"] * 4 + ["transfer"], (
        f"Expected 4 acquisition trials then transfer, got {phase_names}"
    )


def test_environment_cue_probs_and_sets():
    cfg = make_criterion_task(
        n_sets=2, initial_best_per_set=(0, 1), n_correct_min=2, n_correct_max=2
    )
    records = _run_scripted(cfg, lambda t: t.best_cue)
    for trial, _ in records:
        if trial.phase_name == "transfer":
            assert trial.cue_probs == (0.33, 0.33, 0.33)
        else:
            assert trial.cue_probs[trial.best_cue] == 0.8
            assert all(
                p == 0.2 for i, p in enumerate(trial.cue_probs) if i != trial.best_cue
            )
    # Trial indices are continuous 0..n-1 across the session.
    assert [t.trial_idx for t, _ in records] == list(range(len(records)))
    # Each set's acquisition starts with its configured best cue.
    first_acq = {
        t.set_idx: t.best_cue
        for t, _ in reversed(records)
        if t.phase_name == "acquisition_1"
    }
    assert first_acq == {0: 0, 1: 1}


def test_reversal_changes_best_cue():
    cfg = make_criterion_task(n_correct_min=2, n_correct_max=2)
    records = _run_scripted(cfg, lambda t: t.best_cue)
    by_phase = {}
    for trial, _ in records:
        if trial.phase_name != "transfer":
            by_phase.setdefault(trial.phase_name, trial.best_cue)
    assert by_phase["acquisition_1"] != by_phase["reversal_1"]
    assert by_phase["reversal_1"] != by_phase["reversal_2"]


# ---------------------------------------------------------------------------
# Target rule: random_nonbest
# ---------------------------------------------------------------------------


def test_random_nonbest_never_repeats_current_best():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        new = draw_new_best_cue(1, 3, "random_nonbest", rng)
        assert new in {0, 2}, f"Expected non-best cue, got {new}"


def test_random_nonbest_covers_all_nonbest_cues():
    draws = {
        draw_new_best_cue(0, 3, "random_nonbest", np.random.default_rng(seed))
        for seed in range(50)
    }
    assert draws == {1, 2}, f"Expected both non-best cues across seeds, got {draws}"


def test_random_nonbest_two_cues_is_deterministic():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert draw_new_best_cue(0, 2, "random_nonbest", rng) == 1
        assert draw_new_best_cue(1, 2, "random_nonbest", rng) == 0


def test_unknown_target_rule_raises():
    with pytest.raises(ValueError, match="target_rule must be 'random_nonbest'"):
        draw_new_best_cue(0, 3, "always_next", np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Full closed-loop simulation (HGF agent in the loop)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "participant_id",
    "group",
    "session",
    "session_idx",
    "trial",
    "cue_chosen",
    "reward",
    "cue_0_prob",
    "cue_1_prob",
    "cue_2_prob",
    "phase_label",
    "phase_name",
    "best_cue",
    "true_omega_2",
    "true_omega_3",
    "true_kappa",
    "true_beta",
    "true_zeta",
    "model",
    "diverged",
]


def test_batch_output_schema(batch_df):
    assert list(batch_df.columns) == EXPECTED_COLUMNS, (
        f"Expected columns {EXPECTED_COLUMNS}, got {list(batch_df.columns)}"
    )
    assert batch_df["model"].unique().tolist() == ["hgf_3level"]
    assert set(batch_df["phase_name"]) <= {
        "acquisition_1",
        "reversal_1",
        "reversal_2",
        "transfer",
    }
    assert set(batch_df["phase_label"]) <= {"stable", "volatile"}
    volatile = batch_df[batch_df["phase_label"] == "volatile"]
    assert set(volatile["phase_name"]) <= {"reversal_1", "reversal_2"}


def test_batch_trials_continuous_within_session(batch_df):
    for (_, _), grp in batch_df.groupby(["participant_id", "session"]):
        assert grp["trial"].tolist() == list(range(len(grp))), (
            "Expected 0-indexed continuous trial numbering within session"
        )


def test_batch_phase_lengths_respect_cap(batch_df, crit_config):
    max_trials = crit_config.task.reversal.max_trials_per_phase
    n_transfer = crit_config.task.transfer.n_trials
    for (_, _), grp in batch_df.groupby(["participant_id", "session"]):
        # Run-length encode phase_name to recover per-phase trial counts.
        run_ids = (grp["phase_name"] != grp["phase_name"].shift()).cumsum()
        for _, phase_grp in grp.groupby(run_ids):
            name = phase_grp["phase_name"].iloc[0]
            if name == "transfer":
                assert len(phase_grp) == n_transfer
            else:
                assert 1 <= len(phase_grp) <= max_trials, (
                    f"Phase {name}: expected <= {max_trials} trials, "
                    f"got {len(phase_grp)}"
                )


def test_batch_reproducibility(batch_df, crit_config):
    df_again = simulate_batch(crit_config)
    pd.testing.assert_frame_equal(batch_df.reset_index(drop=True), df_again)


def test_transfer_trials_do_not_update_beliefs(crit_config):
    params = {
        "omega_2": -3.0,
        "omega_3": -6.0,
        "kappa": 1.0,
        "beta": 6.0,
        "zeta": 0.0,
    }
    result = simulate_criterion_session(
        crit_config.task,
        params,
        env_rng=np.random.default_rng(11),
        choice_rng=np.random.default_rng(12),
    )
    # Any trial following a transfer trial must carry identical beliefs:
    # no feedback means no belief update.
    checked = 0
    for i in range(len(result.trials) - 1):
        if result.trials[i].phase_name == "transfer":
            assert result.beliefs[i + 1] == result.beliefs[i], (
                f"Beliefs changed across no-feedback transfer trial {i}: "
                f"expected {result.beliefs[i]}, got {result.beliefs[i + 1]}"
            )
            checked += 1
    assert checked > 0, "Expected at least one transfer trial pair to check"


def test_session_reproducibility(crit_config):
    params = {
        "omega_2": -3.0,
        "omega_3": -6.0,
        "kappa": 1.0,
        "beta": 6.0,
        "zeta": 0.0,
    }
    results = [
        simulate_criterion_session(
            crit_config.task,
            params,
            env_rng=np.random.default_rng(5),
            choice_rng=np.random.default_rng(6),
        )
        for _ in range(2)
    ]
    assert results[0].choices == results[1].choices
    assert results[0].rewards == results[1].rewards
    assert results[0].trials == results[1].trials
    assert results[0].beliefs == results[1].beliefs
