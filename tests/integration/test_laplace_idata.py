"""Tests for src/prl_hgf/fitting/laplace_idata.py.

Verifies:
- 2-level and 3-level posterior shape contract
- Deterministic beta = exp(log_beta)
- Dim name is 'participant_id' (not 'participant') — OQ1 consumer guard
- az.hdi works on single-chain output with ArviZ 0.22+ coord structure
- export_subject_parameters consumes Laplace idata unchanged (integration test)
- param_names mismatch raises ValueError
- cov shape mismatch raises ValueError
- mode shape mismatch raises ValueError (m4 coverage)
- sample_stats group present with all 7 canonical diagnostic keys
- pick_best_cue regression: bms.compute_subject_waic importable
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from prl_hgf.analysis.export_trajectories import export_subject_parameters
from prl_hgf.fitting.laplace_idata import (
    _PARAM_ORDER_2LEVEL,
    _PARAM_ORDER_3LEVEL,
    build_idata_from_laplace,
)

# ---------------------------------------------------------------------------
# Shared helper: build minimal valid Laplace idata
# ---------------------------------------------------------------------------

_P2 = 4  # participant count for 2-level tests
_P3 = 3  # participant count for 3-level tests
_IDS2 = ["P0", "P1", "P2", "P3"]
_IDS3 = ["Q0", "Q1", "Q2"]
_N_DRAWS = 500


def _make_2level_idata(
    P: int = _P2,
    pids: list[str] | None = None,
    n_draws: int = _N_DRAWS,
    rng_key: int = 0,
    diagnostics: dict | None = None,
) -> az.InferenceData:
    """Build a minimal valid 2-level Laplace InferenceData."""
    if pids is None:
        pids = [f"P{i}" for i in range(P)]
    K = len(_PARAM_ORDER_2LEVEL)
    mode = {
        "omega_2": np.full(P, -3.0),
        "log_beta": np.full(P, 1.0),
    }
    cov = np.eye(P * K) * 0.05
    return build_idata_from_laplace(
        mode,
        cov,
        _PARAM_ORDER_2LEVEL,
        pids,
        n_pseudo_draws=n_draws,
        rng_key=rng_key,
        diagnostics=diagnostics,
    )


def _make_3level_idata(
    P: int = _P3,
    pids: list[str] | None = None,
    n_draws: int = _N_DRAWS,
    rng_key: int = 0,
) -> az.InferenceData:
    """Build a minimal valid 3-level Laplace InferenceData."""
    if pids is None:
        pids = [f"Q{i}" for i in range(P)]
    K = len(_PARAM_ORDER_3LEVEL)
    mode = {
        "omega_2": np.full(P, -3.0),
        "log_beta": np.full(P, 1.0),
        "omega_3": np.full(P, -6.0),
        "kappa": np.full(P, 1.0),
        "mu3_0": np.full(P, 0.0),
    }
    cov = np.eye(P * K) * 0.05
    return build_idata_from_laplace(
        mode,
        cov,
        _PARAM_ORDER_3LEVEL,
        pids,
        n_pseudo_draws=n_draws,
        rng_key=rng_key,
    )


# ---------------------------------------------------------------------------
# Test 1: 2-level shape contract
# ---------------------------------------------------------------------------


def test_build_idata_2level_shape_contract() -> None:
    """2-level idata has correct shapes and participant_id coordinate."""
    idata = _make_2level_idata(P=4, pids=_IDS2, n_draws=500)
    post = idata.posterior

    assert post.omega_2.shape == (1, 500, 4), (
        f"omega_2 shape expected (1, 500, 4); got {post.omega_2.shape}"
    )
    assert post.log_beta.shape == (1, 500, 4), (
        f"log_beta shape expected (1, 500, 4); got {post.log_beta.shape}"
    )
    assert post.beta.shape == (1, 500, 4), (
        f"beta shape expected (1, 500, 4); got {post.beta.shape}"
    )

    pid_coord = list(post.coords["participant_id"].values)
    assert pid_coord == _IDS2, (
        f"participant_id coordinate expected {_IDS2!r}; got {pid_coord!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: 3-level shape contract
# ---------------------------------------------------------------------------


def test_build_idata_3level_shape_contract() -> None:
    """3-level idata has correct shapes for all 5 sampled vars + beta."""
    idata = _make_3level_idata(P=3, pids=_IDS3, n_draws=500)
    post = idata.posterior

    expected_vars = {"omega_2", "log_beta", "omega_3", "kappa", "mu3_0", "beta"}
    actual_vars = set(post.data_vars)
    assert actual_vars == expected_vars, (
        f"posterior vars expected {expected_vars!r}; got {actual_vars!r}"
    )

    for var in expected_vars:
        assert post[var].shape == (1, 500, 3), (
            f"{var} shape expected (1, 500, 3); got {post[var].shape}"
        )

    assert "participant_id" in post.dims, (
        "participant_id not in posterior dims for 3-level idata"
    )


# ---------------------------------------------------------------------------
# Test 3: deterministic beta is exp(log_beta)
# ---------------------------------------------------------------------------


def test_deterministic_beta_is_exp_log_beta() -> None:
    """beta posterior equals exp(log_beta) to float32 tolerance."""
    idata = _make_2level_idata()
    post = idata.posterior
    assert np.allclose(
        post.beta.values,
        np.exp(post.log_beta.values),
        rtol=1e-6,
        atol=1e-6,
    ), "beta != exp(log_beta)"


# ---------------------------------------------------------------------------
# Test 4: dim name is participant_id (OQ1 parity guard)
# ---------------------------------------------------------------------------


def test_dim_name_is_participant_id_not_participant() -> None:
    """Posterior uses dim 'participant_id', NOT 'participant'.

    This is the OQ1 consumer contract guard.  The NUTS path's
    ``_samples_to_idata`` emits ``'participant'`` while
    ``export_subject_trajectories`` reads ``'participant_id'``.
    The Laplace path emits the correct consumer-facing name natively,
    sidestepping the bug without modifying ``hierarchical.py``.
    """
    idata = _make_2level_idata()
    post = idata.posterior

    assert "participant_id" in post.omega_2.dims, (
        f"Expected 'participant_id' in omega_2.dims; got {post.omega_2.dims}"
    )
    assert "participant" not in post.omega_2.dims, (
        f"'participant' must NOT be in omega_2.dims; got {post.omega_2.dims}"
    )


# ---------------------------------------------------------------------------
# Test 5: az.hdi works on single-chain output
# ---------------------------------------------------------------------------


def test_az_hdi_works_on_single_chain() -> None:
    """az.hdi(posterior, hdi_prob=0.94) returns Dataset with lower/higher coords."""
    P = 4
    idata = _make_2level_idata(P=P)
    post = idata.posterior

    hdi = az.hdi(post, hdi_prob=0.94)

    assert isinstance(hdi, xr.Dataset), (
        f"az.hdi should return xr.Dataset; got {type(hdi)}"
    )

    # ArviZ 0.22.0 convention: hdi coord values are 'lower' and 'higher'
    hdi_coord_values = list(hdi.coords["hdi"].values)
    assert "lower" in hdi_coord_values, (
        f"'lower' not in hdi coord values; got {hdi_coord_values}"
    )
    assert "higher" in hdi_coord_values, (
        f"'higher' not in hdi coord values; got {hdi_coord_values}"
    )

    # omega_2 HDI should be (P, 2): one lower+higher per participant
    assert hdi.omega_2.shape == (P, 2), (
        f"hdi.omega_2.shape expected ({P}, 2); got {hdi.omega_2.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: consumer compatibility — export_subject_parameters
# ---------------------------------------------------------------------------


def test_consumer_compatibility_export_subject_parameters() -> None:
    """export_subject_parameters runs on Laplace idata and yields valid CSV.

    This is the critical integration test: if this passes, Laplace output
    is NUTS-compatible with the Plan 18-05 exporter.
    """
    P = 4
    pids = _IDS2
    idata = _make_2level_idata(P=P, pids=pids, n_draws=200)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_subject_parameters(
            idata,
            model_name="hgf_2level_patrl",
            output_dir=Path(tmp),
        )
        assert out_path.exists(), f"CSV not written to {out_path}"

        df = pd.read_csv(out_path)

    expected_cols = ["participant_id", "parameter", "posterior_mean", "hdi_low", "hdi_high"]
    assert list(df.columns) == expected_cols, (
        f"CSV columns expected {expected_cols!r}; got {list(df.columns)!r}"
    )

    # 2-level params: omega_2, beta — 2 params × P participants = 2*P rows
    assert len(df) == 2 * P, (
        f"Expected {2 * P} rows (2 params × {P} participants); got {len(df)}"
    )

    # All values must be finite (not NaN)
    numeric_cols = ["posterior_mean", "hdi_low", "hdi_high"]
    for col in numeric_cols:
        assert df[col].notna().all(), f"Column '{col}' has NaN values"

    # participant_id column must contain all expected IDs
    assert set(df["participant_id"].unique()) == set(pids), (
        f"participant_id values expected {set(pids)!r}; "
        f"got {set(df['participant_id'].unique())!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: param_names mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_validates_param_names_mismatch() -> None:
    """Passing unknown param_names raises ValueError with expected vs actual."""
    mode = {"foo": np.zeros(3)}
    cov = np.eye(3)
    with pytest.raises(ValueError, match="param_names expected"):
        build_idata_from_laplace(
            mode,
            cov,
            ("foo",),
            ["P0", "P1", "P2"],
        )


# ---------------------------------------------------------------------------
# Test 8: cov shape mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_validates_cov_shape_mismatch() -> None:
    """Wrong cov shape raises ValueError with expected vs actual shapes."""
    P = 4
    mode = {
        "omega_2": np.zeros(P),
        "log_beta": np.zeros(P),
    }
    # P*K = 8; pass eye(7) instead — shape mismatch
    cov = np.eye(7)
    with pytest.raises(ValueError, match="cov.shape expected"):
        build_idata_from_laplace(
            mode,
            cov,
            _PARAM_ORDER_2LEVEL,
            [f"P{i}" for i in range(P)],
        )


# ---------------------------------------------------------------------------
# Test 9: sample_stats group with all 7 canonical diagnostic keys
# ---------------------------------------------------------------------------


def test_sample_stats_group_present() -> None:
    """sample_stats group has all 7 canonical keys, round-tripping values."""
    _CANONICAL_DIAGNOSTICS = {
        "converged": 1.0,
        "n_iterations": 87.0,
        "logp_at_mode": -123.4,
        "hessian_min_eigval": 0.05,
        "hessian_max_eigval": 12.3,
        "n_eigenvalues_clipped": 0.0,
        "ridge_added": 0.0,
    }
    n_draws = 200
    idata = _make_2level_idata(n_draws=n_draws, diagnostics=_CANONICAL_DIAGNOSTICS)

    assert "sample_stats" in idata.groups(), (
        "'sample_stats' group missing from InferenceData"
    )

    ss = idata.sample_stats
    for key, expected_val in _CANONICAL_DIAGNOSTICS.items():
        assert key in ss.data_vars, (
            f"Canonical diagnostic key '{key}' missing from sample_stats"
        )
        arr = ss[key].values
        assert arr.shape == (1, n_draws), (
            f"sample_stats['{key}'].shape expected (1, {n_draws}); got {arr.shape}"
        )
        # Round-trip: all entries must equal the input scalar (no key remap)
        assert float(arr[0, 0]) == pytest.approx(expected_val), (
            f"sample_stats['{key}'][0,0] expected {expected_val}; "
            f"got {float(arr[0, 0])}"
        )


# ---------------------------------------------------------------------------
# Test 10: pick_best_cue regression guard
# ---------------------------------------------------------------------------


def test_pick_best_cue_regression_unchanged() -> None:
    """bms.compute_subject_waic is still importable (pick_best_cue regression guard)."""
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: F401


# ---------------------------------------------------------------------------
# Test 11: mode shape mismatch raises ValueError (m4 coverage)
# ---------------------------------------------------------------------------


def test_validates_mode_shape_mismatch() -> None:
    """Wrong mode array ndim or shape raises ValueError naming the variable."""
    P = 3
    K = len(_PARAM_ORDER_2LEVEL)
    pids = [f"P{i}" for i in range(P)]
    cov = np.eye(P * K) * 0.05

    # Case A: ndim=2 (e.g. caller passes (1, P) instead of (P,))
    bad_mode_2d = {
        "omega_2": np.zeros((1, P)),  # wrong ndim
        "log_beta": np.zeros(P),
    }
    with pytest.raises(ValueError, match="omega_2"):
        build_idata_from_laplace(bad_mode_2d, cov, _PARAM_ORDER_2LEVEL, pids)

    # Case B: shape[0] != P (e.g. caller passes P+1 instead of P values)
    bad_mode_wrong_p = {
        "omega_2": np.zeros(P + 1),  # wrong shape[0]
        "log_beta": np.zeros(P),
    }
    with pytest.raises(ValueError, match="omega_2"):
        build_idata_from_laplace(bad_mode_wrong_p, cov, _PARAM_ORDER_2LEVEL, pids)
