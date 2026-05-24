"""Unit tests for prl_hgf.env.pat_rl_simulator.

Tests cover:
1. Output shape contract for simulate_patrl_cohort
2. Seed determinism: same master_seed -> identical cohort
3. Different seeds produce different cohorts
4. run_hgf_forward_patrl 2-level shape + finiteness
5. run_hgf_forward_patrl 3-level shape + finiteness
6. Pick_best_cue regression guard (canary imports)
7. run_hgf_forward_patrl returns epsilon2 tuple when return_epsilon2=True
8. run_hgf_forward_patrl default return unchanged (scalar ndarray) when return_epsilon2=False
9. simulate_patrl_cohort has phenotype column
10. simulate_patrl_cohort delta_hr deterministic across same seed
11. Different phenotypes produce different mean delta_hr (phenotype effect survives)
12. simulate_patrl_cohort delta_hr within config bounds
13. epsilon2 coupling contributes non-zero variance vs zero coupling
14. simulate_patrl_cohort model_d uses lam_true (choices differ at lam=0 vs lam=0.5)
15. [Plan 20-05] simulate_patrl_cohort multi-phenotype default all four
16. [Plan 20-05] simulate_patrl_cohort determinism across phenotype subsets
17. [Plan 20-05] simulate_patrl_cohort repeat call identical choices (L10)
18. [Plan 20-05] simulate_patrl_cohort global PID naming P000..P{N-1}
19. [Plan 20-05] simulate_patrl_cohort legacy phenotype_name kwarg deprecated
20. [Plan 20-05] simulate_patrl_cohort full 160-agent scale (slow)
21. [Plan 20-05] simulate_patrl_cohort unknown phenotype raises ValueError
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_sequence import PATRLTrial, generate_session_patrl
from prl_hgf.env.pat_rl_simulator import (
    run_hgf_forward_patrl,
    simulate_patrl_cohort,
)

_N_TRIALS = 192  # from configs/pat_rl.yaml: n_trials: 192


def test_simulate_patrl_cohort_shape_contract() -> None:
    """simulate_patrl_cohort returns (DataFrame, dict, dict) with correct shapes.

    Single-phenotype call (phenotypes=['healthy']): n_participants=3, level=2,
    192 trials each -> 576 rows total.  true_params has 3 keys;
    trials_by_participant has 3 keys of 192 PATRLTrial.
    """
    config = load_pat_rl_config()
    sim_df, true_params, trials_by_participant = simulate_patrl_cohort(
        n_participants=3,
        level=2,
        master_seed=42,
        config=config,
        phenotypes=["healthy"],
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
        n_participants=3, level=2, master_seed=42, config=config,
        phenotypes=["healthy"],
    )
    df2, params2, trials2 = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=42, config=config,
        phenotypes=["healthy"],
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
        n_participants=3, level=2, master_seed=42, config=config,
        phenotypes=["healthy"],
    )
    df_43, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=43, config=config,
        phenotypes=["healthy"],
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


# ---------------------------------------------------------------------------
# Phase 20-04 new tests: epsilon2 return, epsilon2-coupled ΔHR, phenotype col
# ---------------------------------------------------------------------------


def test_run_hgf_forward_patrl_returns_epsilon2() -> None:
    """return_epsilon2=True returns (mu2, eps2) tuple; shapes match; all finite.

    Decision 126: pyhgf 0.2.8 exposes
    ``node_trajectories[1]['temp']['value_prediction_error']`` as epsilon2.
    """
    config = load_pat_rl_config()
    trials = generate_session_patrl(config, seed=7)
    result = run_hgf_forward_patrl(trials, omega_2=-3.0, level=2, return_epsilon2=True)

    assert isinstance(result, tuple), (
        f"Expected tuple when return_epsilon2=True; got {type(result)}"
    )
    mu2, eps2 = result
    assert mu2.shape == (_N_TRIALS,), (
        f"mu2 shape: expected ({_N_TRIALS},); got {mu2.shape}"
    )
    assert eps2.shape == mu2.shape, (
        f"epsilon2 shape {eps2.shape} != mu2 shape {mu2.shape}"
    )
    assert np.all(np.isfinite(eps2)), (
        f"epsilon2 contains non-finite values; n_nonfinite={int(np.sum(~np.isfinite(eps2)))}"
    )
    assert np.all(np.isfinite(mu2)), (
        f"mu2 contains non-finite values; n_nonfinite={int(np.sum(~np.isfinite(mu2)))}"
    )


def test_run_hgf_forward_patrl_default_scalar_return_unchanged() -> None:
    """Default (return_epsilon2=False) returns plain np.ndarray, not tuple.

    Backward-compatibility anchor: Phase 18/19 callers expect a bare ndarray.
    """
    config = load_pat_rl_config()
    trials = generate_session_patrl(config, seed=0)
    result = run_hgf_forward_patrl(trials, omega_2=-3.0, level=2)

    assert isinstance(result, np.ndarray), (
        f"Expected np.ndarray when return_epsilon2=False; got {type(result)}"
    )
    assert result.ndim == 1, (
        f"Expected 1-D array; got ndim={result.ndim}"
    )


def test_simulate_patrl_cohort_has_phenotype_column() -> None:
    """sim_df contains a 'phenotype' column with non-empty strings for all rows.

    Single-phenotype call (phenotypes=['healthy']): asserts exactly one phenotype
    present.  Default call (phenotypes=None) returns all 4 — tested separately
    by test_simulate_patrl_cohort_multi_phenotype_default_all_four.
    """
    config = load_pat_rl_config()
    df, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=42, config=config,
        phenotypes=["healthy"],
    )

    assert "phenotype" in df.columns, (
        f"Expected 'phenotype' column; got columns: {list(df.columns)}"
    )
    assert (df["phenotype"].str.len() > 0).all(), (
        "Expected all phenotype values to be non-empty strings; "
        f"got: {df['phenotype'].unique()}"
    )
    assert set(df["phenotype"].unique()) == {"healthy"}, (
        f"Expected only 'healthy' phenotype when phenotypes=['healthy']; "
        f"got {df['phenotype'].unique()}"
    )


def test_simulate_patrl_cohort_epsilon2_coupled_dhr_deterministic() -> None:
    """Same master_seed produces bit-for-bit identical delta_hr trajectory.

    Verifies that spawning dhr_rng from the per-participant rng preserves
    the seed-determinism guarantee after Phase 20 SC4 changes.
    """
    config = load_pat_rl_config()
    df1, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=99, config=config,
        phenotypes=["healthy"],
    )
    df2, _, _ = simulate_patrl_cohort(
        n_participants=3, level=2, master_seed=99, config=config,
        phenotypes=["healthy"],
    )

    assert df1["delta_hr"].equals(df2["delta_hr"]), (
        "delta_hr differs between two calls with same master_seed=99. "
        "Seed-determinism broken by Phase 20 SC4 changes."
    )


def test_simulate_patrl_cohort_dhr_differs_from_pure_phenotype() -> None:
    """Healthy (dhr_mean=-2) and high_anxiety (dhr_mean=-0.5) phenotypes produce different mean ΔHR.

    Validates that the phenotype-specific Gaussian parameters actually
    propagate into the ΔHR column. After epsilon2 coupling and clipping,
    the healthy cohort should still have lower (more bradycardic) mean ΔHR
    than the high_anxiety cohort.

    Phenotype values from configs/pat_rl.yaml:
      healthy: dhr_mean=-2.0, dhr_sd=0.5
      high_anxiety: dhr_mean=-0.5, dhr_sd=0.8
    """
    config = load_pat_rl_config()
    df_healthy, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config,
        phenotype_name="healthy",
    )
    df_high_anxiety, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config,
        phenotype_name="high_anxiety",
    )

    mean_dhr_healthy = float(df_healthy["delta_hr"].mean())
    mean_dhr_high_anxiety = float(df_high_anxiety["delta_hr"].mean())

    assert mean_dhr_healthy < mean_dhr_high_anxiety, (
        f"Expected healthy mean ΔHR ({mean_dhr_healthy:.3f}) < high_anxiety mean "
        f"ΔHR ({mean_dhr_high_anxiety:.3f}). Phenotype effect not surviving."
    )


def test_simulate_patrl_cohort_dhr_within_bounds() -> None:
    """All delta_hr values must lie within config.task.delta_hr_stub.bounds after clipping."""
    config = load_pat_rl_config()
    df, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=7, config=config,
        phenotypes=["healthy"],
    )
    lo, hi = config.task.delta_hr_stub.bounds
    assert (df["delta_hr"] >= lo).all(), (
        f"delta_hr below lower bound {lo}: min={df['delta_hr'].min():.3f}"
    )
    assert (df["delta_hr"] <= hi).all(), (
        f"delta_hr above upper bound {hi}: max={df['delta_hr'].max():.3f}"
    )


def test_simulate_patrl_cohort_epsilon2_contributes_nonzero_variance() -> None:
    """epsilon2_coupling_coef=0 vs 0.3 produces different ΔHR std.

    Confirms the ε₂ coupling term actually fires. Patches the healthy
    phenotype in-memory to set coupling to zero and compares ΔHR std
    against the default (0.3).
    """
    import dataclasses

    config = load_pat_rl_config()

    # Build a config copy with epsilon2_coupling_coef=0.0 for healthy.
    healthy_zero = dataclasses.replace(
        config.simulation.phenotypes["healthy"],
        epsilon2_coupling_coef=0.0,
    )
    phenotypes_zero = {
        k: (healthy_zero if k == "healthy" else v)
        for k, v in config.simulation.phenotypes.items()
    }
    sim_zero = dataclasses.replace(
        config.simulation,
        phenotypes=phenotypes_zero,
    )
    config_zero = dataclasses.replace(config, simulation=sim_zero)

    df_coupled, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config
    )
    df_zero, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config_zero
    )

    std_coupled = float(df_coupled["delta_hr"].std())
    std_zero = float(df_zero["delta_hr"].std())

    assert abs(std_coupled - std_zero) > 1e-6, (
        f"ΔHR std with coupling ({std_coupled:.4f}) == std without coupling "
        f"({std_zero:.4f}). epsilon2 coupling appears to have no effect."
    )


def test_simulate_patrl_cohort_model_d_uses_lam_true() -> None:
    """Model D lam_true affects choices and raises ValueError when misused.

    Three sub-checks:
    1. lam_true=0.0 vs lam_true=0.5 produce different choice arrays.
    2. response_model='model_d' + lam_true=None raises ValueError.
    3. response_model='model_a' + lam_true=0.1 raises ValueError.
    """
    config = load_pat_rl_config()

    # Sub-check 1: lam_true=0.0 vs lam_true=0.5 should differ.
    df_lam0, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config,
        response_model="model_d", lam_true=0.0,
    )
    df_lam05, _, _ = simulate_patrl_cohort(
        n_participants=5, level=2, master_seed=42, config=config,
        response_model="model_d", lam_true=0.5,
    )
    choices_0 = df_lam0["choice"].to_numpy()
    choices_05 = df_lam05["choice"].to_numpy()
    assert not np.array_equal(choices_0, choices_05), (
        "choices are identical for lam_true=0.0 and lam_true=0.5; "
        "expected non-zero lambda to bend belief trajectory and change choices."
    )

    # Sub-check 2: model_d + lam_true=None must raise ValueError.
    try:
        simulate_patrl_cohort(
            n_participants=2, level=2, master_seed=1, config=config,
            response_model="model_d", lam_true=None,
        )
        raise AssertionError("Expected ValueError for model_d + lam_true=None")
    except ValueError as exc:
        assert "model_d" in str(exc).lower() or "lam_true" in str(exc).lower(), (
            f"ValueError message should mention model_d/lam_true; got: {exc}"
        )

    # Sub-check 3: model_a + lam_true=0.1 must raise ValueError (defensive).
    try:
        simulate_patrl_cohort(
            n_participants=2, level=2, master_seed=1, config=config,
            response_model="model_a", lam_true=0.1,
        )
        raise AssertionError("Expected ValueError for model_a + lam_true=0.1")
    except ValueError as exc:
        assert "lam_true" in str(exc).lower() or "model_a" in str(exc).lower(), (
            f"ValueError message should mention lam_true/model_a; got: {exc}"
        )


# ---------------------------------------------------------------------------
# Phase 20-05 new tests: multi-phenotype determinism (SC5 + L10)
# ---------------------------------------------------------------------------


def test_simulate_patrl_cohort_multi_phenotype_default_all_four() -> None:
    """phenotypes=None returns all 4 phenotypes × n_participants agents.

    SC5: 3 agents × 4 phenotypes = 12 unique participants, 12 × 192 = 2304 rows.
    All 4 phenotype names must be present.
    """
    config = load_pat_rl_config()
    df, tp, tbp = simulate_patrl_cohort(
        n_participants=3, phenotypes=None, master_seed=42, config=config
    )

    n_unique = len(df["participant_id"].unique())
    assert n_unique == 12, (
        f"Expected 12 unique participants (3 × 4); got {n_unique}"
    )

    expected_phenotypes = {"healthy", "high_anxiety", "reward_susceptible", "anxious_reward"}
    actual_phenotypes = set(df["phenotype"].unique())
    assert actual_phenotypes == expected_phenotypes, (
        f"Expected all 4 phenotypes {expected_phenotypes}; got {actual_phenotypes}"
    )

    assert len(tp) == 12, (
        f"Expected 12 keys in true_params; got {len(tp)}"
    )
    assert len(tbp) == 12, (
        f"Expected 12 keys in trials_by_participant; got {len(tbp)}"
    )

    expected_rows = 12 * _N_TRIALS
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} rows (12 agents × {_N_TRIALS} trials); got {len(df)}"
    )

    # Verify b_true is present in true_params (Plan 20-02 SC1).
    for pid, params in tp.items():
        assert "b" in params, (
            f"Participant {pid}: 'b' missing from true_params; got keys {list(params.keys())}"
        )


def test_simulate_patrl_cohort_determinism_across_phenotype_subsets() -> None:
    """Subset determinism: healthy rows in full-cohort run match healthy-only run.

    This is the L10 user-observable property: the per-phenotype SeedSequence
    spawn guarantees that ``phenotypes=['healthy']`` produces IDENTICAL agent
    data as the 'healthy' slice of ``phenotypes=None``, regardless of whether
    other phenotypes are also generated in the same call.

    Reference: RESEARCH.md §5, Decision 20-05 (per-phenotype SeedSequence spawn).
    """
    config = load_pat_rl_config()

    # Run 1: all 4 phenotypes.
    df_all, _, _ = simulate_patrl_cohort(
        n_participants=3, phenotypes=None, master_seed=7, config=config
    )
    # Run 2: healthy only.
    df_h, _, _ = simulate_patrl_cohort(
        n_participants=3, phenotypes=["healthy"], master_seed=7, config=config
    )

    healthy_all = df_all[df_all["phenotype"] == "healthy"].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        healthy_all[["trial_idx", "state", "choice", "delta_hr"]],
        df_h[["trial_idx", "state", "choice", "delta_hr"]],
        check_exact=True,
        obj="healthy subset vs healthy-only run",
    )


def test_simulate_patrl_cohort_repeat_call_identical_choices() -> None:
    """Identical args → identical choice array (L10 user-observable determinism).

    Calling simulate_patrl_cohort with the same master_seed twice must
    produce bit-for-bit identical 'choice' columns.  This anchors the
    determinism guarantee for downstream callers that need reproducible
    simulation results.
    """
    config = load_pat_rl_config()

    df_a, _, _ = simulate_patrl_cohort(
        n_participants=3, phenotypes=["healthy"], master_seed=42, config=config
    )
    df_b, _, _ = simulate_patrl_cohort(
        n_participants=3, phenotypes=["healthy"], master_seed=42, config=config
    )

    np.testing.assert_array_equal(
        df_a["choice"].to_numpy(),
        df_b["choice"].to_numpy(),
        err_msg=(
            "choice arrays differ between two calls with identical args. "
            "Seed determinism broken (L10 violation)."
        ),
    )


def test_simulate_patrl_cohort_pid_naming() -> None:
    """Global PID counter produces P000..P011 for 3 agents × 4 phenotypes."""
    config = load_pat_rl_config()
    df, _, _ = simulate_patrl_cohort(
        n_participants=3, phenotypes=None, master_seed=1, config=config
    )

    unique_pids = sorted(df["participant_id"].unique())
    expected_pids = [f"P{i:03d}" for i in range(12)]

    assert unique_pids == expected_pids, (
        f"PID list mismatch.\n"
        f"Expected: {expected_pids}\n"
        f"Got:      {unique_pids}"
    )

    # Verify P000..P002 are healthy, P003..P005 are reward_susceptible, etc.
    # (order from YAML: healthy, reward_susceptible, high_anxiety, anxious_reward)
    config_phenotypes = list(config.simulation.phenotypes.keys())
    for ph_idx, ph_name in enumerate(config_phenotypes):
        for local_i in range(3):
            global_i = ph_idx * 3 + local_i
            pid = f"P{global_i:03d}"
            pid_rows = df[df["participant_id"] == pid]
            actual_ph = pid_rows["phenotype"].unique()
            assert list(actual_ph) == [ph_name], (
                f"PID {pid} has phenotype {actual_ph}; expected [{ph_name}] "
                f"(phenotype index {ph_idx}, local agent {local_i})"
            )


def test_simulate_patrl_cohort_legacy_phenotype_name_kwarg() -> None:
    """Legacy phenotype_name kwarg emits DeprecationWarning and produces correct result.

    phenotype_name='healthy' must:
    1. Emit a DeprecationWarning mentioning 'deprecated'.
    2. Produce the same sim_df as phenotypes=['healthy'] with the same master_seed.
    """
    config = load_pat_rl_config()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_legacy, _, _ = simulate_patrl_cohort(
            n_participants=2,
            phenotype_name="healthy",
            master_seed=42,
            config=config,
        )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, (
            f"Expected at least one DeprecationWarning; got {[str(x.message) for x in w]}"
        )
        msg_lower = str(dep_warnings[0].message).lower()
        assert "deprecated" in msg_lower, (
            f"DeprecationWarning message should contain 'deprecated'; got: {dep_warnings[0].message}"
        )

    df_new, _, _ = simulate_patrl_cohort(
        n_participants=2, phenotypes=["healthy"], master_seed=42, config=config
    )
    pd.testing.assert_frame_equal(df_legacy, df_new, check_exact=True)


@pytest.mark.scientific
def test_simulate_patrl_cohort_full_160_scale() -> None:
    """Full 160-agent cohort (40 per phenotype × 4 phenotypes) runs in <5 min on CPU.

    This is the SC5 production-scale test.  Marked @pytest.mark.scientific; run with
    ``pytest --run-slow`` or ``pytest -m slow``.

    Asserts:
    - 160 unique participants
    - 40 per phenotype
    - 160 × 192 = 30720 rows total
    - All required columns present
    """
    config = load_pat_rl_config()
    df, tp, tbp = simulate_patrl_cohort(
        n_participants=40, phenotypes=None, master_seed=42, config=config
    )

    n_unique = len(df["participant_id"].unique())
    assert n_unique == 160, (
        f"Expected 160 unique participants (40 × 4); got {n_unique}"
    )

    expected_phenotypes = {"healthy", "high_anxiety", "reward_susceptible", "anxious_reward"}
    actual_phenotypes = set(df["phenotype"].unique())
    assert actual_phenotypes == expected_phenotypes, (
        f"Expected all 4 phenotypes; got {actual_phenotypes}"
    )

    phenotype_counts = df.groupby("phenotype")["participant_id"].nunique()
    for ph_name in expected_phenotypes:
        count = phenotype_counts.get(ph_name, 0)
        assert count == 40, (
            f"Phenotype '{ph_name}': expected 40 agents; got {count}"
        )

    expected_rows = 160 * _N_TRIALS
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} rows (160 × {_N_TRIALS}); got {len(df)}"
    )

    required_cols = {"participant_id", "phenotype", "trial_idx", "state", "choice",
                     "reward_mag", "shock_mag", "delta_hr", "outcome_time_s"}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, (
        f"sim_df missing required columns: {missing_cols}"
    )


def test_simulate_patrl_cohort_unknown_phenotype_raises() -> None:
    """Passing an unknown phenotype name raises ValueError with available names listed."""
    config = load_pat_rl_config()

    with pytest.raises(ValueError, match="nonexistent"):
        simulate_patrl_cohort(
            n_participants=2, phenotypes=["nonexistent"], master_seed=1, config=config
        )
