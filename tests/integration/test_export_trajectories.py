"""Tests for src/prl_hgf/analysis/export_trajectories.py.

Verifies:
- Per-trial CSV schema, column order, dtypes
- 2-level NaN filling for 3-level-only columns
- 3-level columns populated (non-NaN)
- Row count equals n_trials
- Monotone, seconds-scaled outcome_time_s
- psi2 > 0; epsilon2 finite
- Parameter summary CSV columns + rows
- 3-level parameter summary contains omega_3, kappa
- pyhgf temp-key extraction canary (fires if keys renamed)
- pick_best_cue analysis regression (bms.compute_subject_waic importable)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest

from prl_hgf.analysis.export_trajectories import (
    export_subject_parameters,
    export_subject_trajectories,
)
from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_sequence import generate_session_patrl
from prl_hgf.models.hgf_2level_patrl import (
    BELIEF_NODE,
    INPUT_NODE,
    build_2level_network_patrl,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N_PARTICIPANTS = 3
_PARTICIPANT_IDS = ["P0", "P1", "P2"]
_SEED = 42


def _make_2level_idata(rng: np.random.Generator) -> az.InferenceData:
    """Build a minimal 2-level InferenceData with fake posteriors."""
    return az.from_dict(
        posterior={
            "omega_2": rng.standard_normal((2, 10, _N_PARTICIPANTS)) - 4.0,
            "beta": np.abs(rng.standard_normal((2, 10, _N_PARTICIPANTS))) + 2.0,
        },
        coords={
            "chain": range(2),
            "draw": range(10),
            "participant_id": _PARTICIPANT_IDS,
        },
        dims={"omega_2": ["participant_id"], "beta": ["participant_id"]},
    )


def _make_3level_idata(rng: np.random.Generator) -> az.InferenceData:
    """Build a minimal 3-level InferenceData with fake posteriors."""
    return az.from_dict(
        posterior={
            "omega_2": rng.standard_normal((2, 10, _N_PARTICIPANTS)) - 4.0,
            "omega_3": rng.standard_normal((2, 10, _N_PARTICIPANTS)) - 6.0,
            "kappa": np.abs(rng.standard_normal((2, 10, _N_PARTICIPANTS))) + 0.5,
            "beta": np.abs(rng.standard_normal((2, 10, _N_PARTICIPANTS))) + 2.0,
            "mu3_0": rng.standard_normal((2, 10, _N_PARTICIPANTS)) + 1.0,
        },
        coords={
            "chain": range(2),
            "draw": range(10),
            "participant_id": _PARTICIPANT_IDS,
        },
        dims={k: ["participant_id"] for k in ["omega_2", "omega_3", "kappa", "beta", "mu3_0"]},
    )


@pytest.fixture(scope="module")
def session_trials() -> list:
    """Generate a default PAT-RL session (192 trials)."""
    cfg = load_pat_rl_config()
    return generate_session_patrl(cfg, seed=_SEED)


@pytest.fixture(scope="module")
def choices_arr(session_trials: list) -> np.ndarray:
    """Random binary choice array matching the session length."""
    rng = np.random.default_rng(_SEED)
    return rng.integers(0, 2, size=len(session_trials)).astype(np.int32)


# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = [
    "participant_id",
    "trial_idx",
    "run_idx",
    "trial_in_run",
    "regime",
    "outcome_time_s",
    "state",
    "choice",
    "reward_mag",
    "shock_mag",
    "delta_hr",
    "mu2",
    "sigma2",
    "mu3",
    "sigma3",
    "delta1",
    "epsilon2",
    "epsilon3",
    "psi2",
]

_INT_COLUMNS = {"trial_idx", "run_idx", "trial_in_run", "state", "choice"}
_FLOAT_COLUMNS = {
    "outcome_time_s",
    "reward_mag",
    "shock_mag",
    "delta_hr",
    "mu2",
    "sigma2",
    "mu3",
    "sigma3",
    "delta1",
    "epsilon2",
    "epsilon3",
    "psi2",
}


# ---------------------------------------------------------------------------
# Test 1: 2-level export has required columns in correct order
# ---------------------------------------------------------------------------


def test_export_2level_has_required_columns(session_trials: list, choices_arr: np.ndarray) -> None:
    """2-level CSV has all required columns in the correct order with correct dtypes."""
    rng = np.random.default_rng(0)
    idata = _make_2level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_2level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    assert list(df.columns) == _REQUIRED_COLUMNS, (
        f"Column order mismatch.\n  Expected: {_REQUIRED_COLUMNS}\n  Got: {list(df.columns)}"
    )
    for col in _INT_COLUMNS:
        assert pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_int64_dtype(df[col]), (
            f"Column {col!r}: expected integer dtype, got {df[col].dtype}"
        )
    for col in _FLOAT_COLUMNS:
        assert pd.api.types.is_float_dtype(df[col]), (
            f"Column {col!r}: expected float dtype, got {df[col].dtype}"
        )


# ---------------------------------------------------------------------------
# Test 2: 3-level export has all columns, mu3/sigma3/epsilon3 non-NaN
# ---------------------------------------------------------------------------


def test_export_3level_has_all_columns(session_trials: list, choices_arr: np.ndarray) -> None:
    """3-level CSV has all schema columns; mu3, sigma3, epsilon3 are non-NaN."""
    rng = np.random.default_rng(1)
    idata = _make_3level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_3level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    assert list(df.columns) == _REQUIRED_COLUMNS
    for col in ("mu3", "sigma3", "epsilon3"):
        assert not df[col].isna().all(), (
            f"Column {col!r}: expected non-NaN values in 3-level export, "
            f"but all values are NaN."
        )


# ---------------------------------------------------------------------------
# Test 3: 2-level NaN in 3-level-only columns
# ---------------------------------------------------------------------------


def test_export_2level_nans_3level_only_columns(
    session_trials: list, choices_arr: np.ndarray
) -> None:
    """2-level export: mu3, sigma3, epsilon3 present but all NaN (schema consistency)."""
    rng = np.random.default_rng(2)
    idata = _make_2level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_2level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    for col in ("mu3", "sigma3", "epsilon3"):
        assert df[col].isna().all(), (
            f"Column {col!r}: expected all NaN in 2-level export, "
            f"got {df[col].notna().sum()} non-NaN values."
        )


# ---------------------------------------------------------------------------
# Test 4: row count equals n_trials
# ---------------------------------------------------------------------------


def test_n_rows_equals_n_trials(session_trials: list, choices_arr: np.ndarray) -> None:
    """CSV has exactly n_trials rows (192 by default)."""
    rng = np.random.default_rng(3)
    idata = _make_2level_idata(rng)
    n_trials = len(session_trials)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_2level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    assert len(df) == n_trials, (
        f"Row count mismatch: expected {n_trials}, got {len(df)}."
    )


# ---------------------------------------------------------------------------
# Test 5: outcome_time_s is monotone and seconds-scaled
# ---------------------------------------------------------------------------


def test_outcome_time_s_is_monotone_and_seconds_scaled(
    session_trials: list, choices_arr: np.ndarray
) -> None:
    """outcome_time_s strictly increases; last value matches expected session end."""
    rng = np.random.default_rng(4)
    idata = _make_2level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_2level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    times = df["outcome_time_s"].values
    diffs = np.diff(times)
    assert (diffs > 0).all(), (
        f"outcome_time_s is not strictly monotone increasing. "
        f"Min diff: {diffs.min():.4f}"
    )
    # Expected last value: run 3, trial 47 in run
    # trial_duration_s = 11 s; outcome_offset_within_trial = 7 s (cue 1.5 + anticipation 5.5)
    # run_duration_s = 48 * 11 = 528 s; run_gap_s = 15 s
    # last_outcome = 3*(528+15) + 47*11 + 7 = 1629 + 517 + 7 = 2153 s
    expected_last = 2153.0
    assert abs(times[-1] - expected_last) <= 1.0, (
        f"Last outcome_time_s: expected ~{expected_last:.1f} s, got {times[-1]:.3f} s."
    )


# ---------------------------------------------------------------------------
# Test 6: psi2 > 0; epsilon2 finite
# ---------------------------------------------------------------------------


def test_psi2_positive_epsilon2_finite(session_trials: list, choices_arr: np.ndarray) -> None:
    """psi2 is strictly positive everywhere; epsilon2 is finite throughout."""
    rng = np.random.default_rng(5)
    idata = _make_2level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_trajectories(
            "P0", idata, session_trials, choices_arr, "hgf_2level_patrl", Path(tmp)
        )
        df = pd.read_csv(path)

    psi2 = df["psi2"].values
    assert (psi2 > 0).all(), (
        f"psi2 must be strictly positive; got min={psi2.min():.6f}."
    )
    epsilon2 = df["epsilon2"].values
    assert np.isfinite(epsilon2).all(), (
        f"epsilon2 must be finite everywhere; got {np.sum(~np.isfinite(epsilon2))} non-finite."
    )


# ---------------------------------------------------------------------------
# Test 7: parameter summary columns
# ---------------------------------------------------------------------------


def test_parameter_summary_columns(session_trials: list) -> None:
    """export_subject_parameters CSV has correct columns."""
    rng = np.random.default_rng(6)
    idata = _make_2level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_parameters(idata, "hgf_2level_patrl", Path(tmp))
        sdf = pd.read_csv(path)

    expected_cols = ["participant_id", "parameter", "posterior_mean", "hdi_low", "hdi_high"]
    assert list(sdf.columns) == expected_cols, (
        f"Parameter summary columns mismatch.\n"
        f"  Expected: {expected_cols}\n  Got: {list(sdf.columns)}"
    )
    # 3 participants * 2 parameters = 6 rows
    expected_rows = _N_PARTICIPANTS * 2  # omega_2, beta
    assert len(sdf) == expected_rows, (
        f"Expected {expected_rows} rows for 2-level, got {len(sdf)}."
    )
    assert set(sdf["participant_id"].unique()) == set(_PARTICIPANT_IDS), (
        "participant_id values mismatch."
    )


# ---------------------------------------------------------------------------
# Test 8: 3-level parameter summary has omega_3, kappa, mu3_0
# ---------------------------------------------------------------------------


def test_parameter_summary_3level_has_omega3_kappa(session_trials: list) -> None:
    """3-level export: each participant has rows for omega_2, omega_3, kappa, beta, mu3_0."""
    rng = np.random.default_rng(7)
    idata = _make_3level_idata(rng)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_subject_parameters(idata, "hgf_3level_patrl", Path(tmp))
        sdf = pd.read_csv(path)

    expected_params = {"omega_2", "omega_3", "kappa", "beta", "mu3_0"}
    actual_params = set(sdf["parameter"].unique())
    assert expected_params == actual_params, (
        f"3-level parameter set mismatch.\n"
        f"  Expected: {sorted(expected_params)}\n  Got: {sorted(actual_params)}"
    )
    # 3 participants * 5 parameters = 15 rows
    assert len(sdf) == _N_PARTICIPANTS * 5, (
        f"Expected {_N_PARTICIPANTS * 5} rows for 3-level, got {len(sdf)}."
    )


# ---------------------------------------------------------------------------
# Test 9: pyhgf temp-key extraction canary
# ---------------------------------------------------------------------------


def test_pyhgf_temp_keys_extracted() -> None:
    """Canary: verify pyhgf 0.2.x temp-dict keys exist after a forward pass.

    If pyhgf renames or removes 'value_prediction_error' or
    'effective_precision', this test fires immediately.
    """
    net = build_2level_network_patrl(omega_2=-4.0)
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, size=20).astype(np.float64)
    net.input_data(input_data=u[:, None], time_steps=np.ones(20, dtype=np.float64))

    traj = net.node_trajectories[BELIEF_NODE]
    temp = traj.get("temp", {})
    assert "value_prediction_error" in temp, (
        f"pyhgf temp-dict missing 'value_prediction_error'. "
        f"Present keys: {sorted(temp.keys())}. "
        f"pyhgf may have renamed this key in the installed version."
    )
    assert "effective_precision" in temp, (
        f"pyhgf temp-dict missing 'effective_precision'. "
        f"Present keys: {sorted(temp.keys())}."
    )
    vpe = np.asarray(temp["value_prediction_error"], dtype=np.float64)
    ep = np.asarray(temp["effective_precision"], dtype=np.float64)
    assert np.isfinite(vpe).all(), (
        f"value_prediction_error: expected all finite, got {np.sum(~np.isfinite(vpe))} non-finite."
    )
    assert (ep > 0).all(), (
        f"effective_precision: expected all > 0, got min={ep.min():.6f}."
    )
    # Also check INPUT_NODE has value_prediction_error
    traj0 = net.node_trajectories[INPUT_NODE]
    temp0 = traj0.get("temp", {})
    assert "value_prediction_error" in temp0, (
        f"INPUT_NODE temp-dict missing 'value_prediction_error'. "
        f"Present keys: {sorted(temp0.keys())}."
    )


# ---------------------------------------------------------------------------
# Test 10: pick_best_cue analysis regression (no import error)
# ---------------------------------------------------------------------------


def test_pick_best_cue_analysis_unchanged() -> None:
    """Regression: pick_best_cue analysis modules still import cleanly.

    Checks that the parallel-stack policy was honoured — adding
    export_trajectories.py did not break the existing analysis package.
    """
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: F401, PLC0415
