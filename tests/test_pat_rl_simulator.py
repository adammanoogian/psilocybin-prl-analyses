"""Unit tests for prl_hgf.env.pat_rl_simulator.

Tests cover:
1. Output shape contract for simulate_patrl_cohort
2. Seed determinism: same master_seed -> identical cohort
3. Different seeds produce different cohorts
4. run_hgf_forward_patrl 2-level shape + finiteness
5. run_hgf_forward_patrl 3-level shape + finiteness
6. Pick_best_cue regression guard (canary imports)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_sequence import PATRLTrial, generate_session_patrl
from prl_hgf.env.pat_rl_simulator import (
    run_hgf_forward_patrl,
    simulate_patrl_cohort,
)

_N_TRIALS = 192  # from configs/pat_rl.yaml: n_trials: 192


def test_simulate_patrl_cohort_shape_contract() -> None:
    """simulate_patrl_cohort returns (DataFrame, dict, dict) with correct shapes.

    n_participants=3, level=2, 192 trials each -> 576 rows total.
    true_params has 3 keys; trials_by_participant has 3 keys of 192 PATRLTrial.
    """
    config = load_pat_rl_config()
    sim_df, true_params, trials_by_participant = simulate_patrl_cohort(
        n_participants=3,
        level=2,
        master_seed=42,
        config=config,
    )

    assert isinstance(sim_df, pd.DataFrame), (
        f"Expected pd.DataFrame; got {type(sim_df)}"
    )
    assert isinstance(true_params, dict), (
        f"Expected dict; got {type(true_params)}"
    )
    assert isinstance(trials_by_participant, dict), (
        f"Expected dict; got {type(trials_by_participant)}"
    )

    expected_rows = 3 * _N_TRIALS
    assert len(sim_df) == expected_rows, (
        f"Expected {expected_rows} rows in sim_df; got {len(sim_df)}"
    )

    assert len(true_params) == 3, (
        f"Expected 3 participant keys in true_params; got {len(true_params)}"
    )
    assert len(trials_by_participant) == 3, (
        f"Expected 3 participant keys in trials_by_participant; "
        f"got {len(trials_by_participant)}"
    )

    for pid, trials in trials_by_participant.items():
        assert len(trials) == _N_TRIALS, (
            f"Participant {pid}: expected {_N_TRIALS} trials; got {len(trials)}"
        )
        assert all(isinstance(t, PATRLTrial) for t in trials), (
            f"Participant {pid}: not all elements are PATRLTrial"
        )


def test_simulate_patrl_cohort_seed_determinism() -> None:
    """Same master_seed produces bit-for-bit identical cohort on two calls."""
    config = load_pat_rl_config()
    df1, params1, trials1 = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=42, config=config
    )
    df2, params2, trials2 = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=42, config=config
    )

    pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    assert params1 == params2, (
        f"true_params differ between calls with same seed.  "
        f"First call: {params1}  Second call: {params2}"
    )

    for pid in trials1:
        t_pairs = zip(trials1[pid], trials2[pid], strict=True)
        for idx, (t_a, t_b) in enumerate(t_pairs):
            a_key = (t_a.state, t_a.reward_mag, t_a.shock_mag, t_a.delta_hr)
            b_key = (t_b.state, t_b.reward_mag, t_b.shock_mag, t_b.delta_hr)
            assert a_key == b_key, (
                f"Participant {pid} trial {idx}: trial fields differ between "
                f"calls with same seed.  Got {a_key} vs {b_key}."
            )


def test_simulate_patrl_cohort_different_seeds_differ() -> None:
    """Different master_seeds produce different cohorts (choice column differs)."""
    config = load_pat_rl_config()
    df_42, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=42, config=config
    )
    df_43, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=43, config=config
    )

    choices_42 = df_42["choice"].to_numpy()
    choices_43 = df_43["choice"].to_numpy()
    assert not np.array_equal(choices_42, choices_43), (
        "choice column is identical for master_seed=42 and master_seed=43; "
        "expected different cohorts when seed changes."
    )


def test_run_hgf_forward_patrl_2level_shape() -> None:
    """run_hgf_forward_patrl (level=2) returns shape (n_trials,) float64, all finite."""
    config = load_pat_rl_config()
    trials = generate_session_patrl(config, seed=0)
    mu2 = run_hgf_forward_patrl(trials, omega_2=-3.0, level=2)

    assert mu2.shape == (_N_TRIALS,), (
        f"Expected shape ({_N_TRIALS},); got {mu2.shape}"
    )
    assert mu2.dtype == np.float64, (
        f"Expected float64; got {mu2.dtype}"
    )
    assert np.all(np.isfinite(mu2)), (
        f"Non-finite values in mu2 trajectory (level=2).  "
        f"n_nonfinite={int(np.sum(~np.isfinite(mu2)))}"
    )


def test_run_hgf_forward_patrl_3level_shape() -> None:
    """run_hgf_forward_patrl (level=3) returns shape (n_trials,) float64, all finite."""
    config = load_pat_rl_config()
    trials = generate_session_patrl(config, seed=0)
    mu2 = run_hgf_forward_patrl(
        trials,
        omega_2=-3.0,
        level=3,
        omega_3=-6.0,
        kappa=1.0,
        mu3_0=1.0,
    )

    assert mu2.shape == (_N_TRIALS,), (
        f"Expected shape ({_N_TRIALS},); got {mu2.shape}"
    )
    assert mu2.dtype == np.float64, (
        f"Expected float64; got {mu2.dtype}"
    )
    assert np.all(np.isfinite(mu2)), (
        f"Non-finite values in mu2 trajectory (level=3).  "
        f"n_nonfinite={int(np.sum(~np.isfinite(mu2)))}"
    )


def test_pick_best_cue_regression_unchanged() -> None:
    """Pick_best_cue modules are importable after Phase 19-01 refactor (canary).

    Verifies that the parallel-stack invariant holds: the Phase 19-01 extraction
    into pat_rl_simulator.py did not accidentally modify pick_best_cue modules.
    """
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: F401
    from prl_hgf.env.simulator import generate_session  # noqa: F401
