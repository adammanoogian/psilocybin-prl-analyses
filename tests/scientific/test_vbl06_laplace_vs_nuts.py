"""Unit tests for VBL-06: Laplace-vs-NUTS posterior comparison harness.

Structure mirrors ``tests/test_valid03.py`` (6+ tests covering identical /
within-tolerance / exceeds-tolerance / near-zero / missing-coord /
cli-smoke).

Tests 1-7 and 9 are fast (<5 s total, pure numpy + ArviZ).
Test 8 is slow (real Laplace fit) and is env-gated by ``RUN_SMOKE_TESTS=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import arviz as az
import numpy as np
import pandas as pd
import pytest

# Make the project root importable when running pytest from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402
from prl_hgf.fitting.laplace_idata import (  # noqa: E402
    _PARAM_ORDER_2LEVEL,
    build_idata_from_laplace,
)
from tests.scientific.test_vbl06_laplace_vs_nuts import (  # noqa: E402
    _apply_hard_gates,
    compare_posteriors,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_laplace_idata(
    omega2_means: list[float],
    omega2_sd: float = 0.1,
    log_beta_means: list[float] | None = None,
    log_beta_sd: float = 0.05,
    pids: list[str] | None = None,
    n_pseudo_draws: int = 400,
    rng_key: int = 0,
) -> az.InferenceData:
    """Build a minimal Laplace-style InferenceData via build_idata_from_laplace.

    Parameters
    ----------
    omega2_means : list[float]
        Per-participant omega_2 MAP mode values.
    omega2_sd : float
        Standard deviation for omega_2 (diagonal covariance entry).
    log_beta_means : list[float] | None
        Per-participant log_beta MAP mode values.  Defaults to 0.0 per participant.
    log_beta_sd : float
        Standard deviation for log_beta (diagonal covariance entry).
    pids : list[str] | None
        Participant IDs.  Defaults to ['P0', 'P1', ...].
    n_pseudo_draws : int
        Number of pseudo-samples.
    rng_key : int
        RNG seed.

    Returns
    -------
    az.InferenceData
        Laplace InferenceData with ``participant_id`` coord.
    """
    P = len(omega2_means)
    if pids is None:
        pids = [f"P{i}" for i in range(P)]
    if log_beta_means is None:
        log_beta_means = [0.0] * P

    mode = {
        "omega_2": np.array(omega2_means, dtype=np.float64),
        "log_beta": np.array(log_beta_means, dtype=np.float64),
    }
    # Diagonal covariance: [omega_2_var, ..., log_beta_var, ...]
    var_vec = np.concatenate(
        [np.full(P, omega2_sd**2), np.full(P, log_beta_sd**2)]
    )
    cov = np.diag(var_vec)

    return cast(
        az.InferenceData,
        build_idata_from_laplace(
            mode=mode,
            cov=cov,
            param_names=_PARAM_ORDER_2LEVEL,
            participant_ids=pids,
            n_pseudo_draws=n_pseudo_draws,
            rng_key=rng_key,
        ),
    )


def _make_nuts_like_idata(
    omega2_means: list[float],
    omega2_sd: float = 0.1,
    beta_means: list[float] | None = None,
    beta_sd: float = 0.05,
    pids: list[str] | None = None,
    n_pseudo_draws: int = 400,
    coord_name: str = "participant_id",
    rng_key: int = 42,
) -> az.InferenceData:
    """Build a minimal NUTS-like InferenceData from numpy arrays.

    Parameters
    ----------
    omega2_means : list[float]
        Per-participant omega_2 posterior means.
    omega2_sd : float
        Per-participant omega_2 posterior SD.
    beta_means : list[float] | None
        Per-participant beta posterior means.  Defaults to 1.0 per participant.
    beta_sd : float
        Per-participant beta posterior SD.
    pids : list[str] | None
        Participant IDs.  Defaults to ['P0', 'P1', ...].
    n_pseudo_draws : int
        Posterior draws.
    coord_name : str
        Coordinate name for the participant dimension.  Use ``'participant'``
        to simulate the pre-hotfix NUTS idata (OQ1 follow-up).
    rng_key : int
        RNG seed for normal samples.

    Returns
    -------
    az.InferenceData
    """
    P = len(omega2_means)
    if pids is None:
        pids = [f"P{i}" for i in range(P)]
    if beta_means is None:
        beta_means = [1.0] * P

    rng = np.random.default_rng(rng_key)
    # shape: (chain=1, draw, participant)
    omega2_samples = rng.normal(
        loc=omega2_means, scale=omega2_sd, size=(1, n_pseudo_draws, P)
    )
    beta_samples = rng.normal(
        loc=beta_means, scale=beta_sd, size=(1, n_pseudo_draws, P)
    )
    posterior = {"omega_2": omega2_samples, "beta": beta_samples}
    coords: dict = {
        "chain": [0],
        "draw": np.arange(n_pseudo_draws),
        coord_name: pids,
    }
    dims = {"omega_2": [coord_name], "beta": [coord_name]}
    return cast(
        az.InferenceData,
        az.from_dict(posterior=posterior, coords=coords, dims=dims),
    )


# ---------------------------------------------------------------------------
# Test 1: identical idatas → all within gate
# ---------------------------------------------------------------------------


def test_compare_posteriors_identical_idatas_all_within_gate() -> None:
    """Identical Laplace idatas: all diffs are ~0 and within_gate is True."""
    idata = _make_laplace_idata([-3.5, -4.0, -3.8])
    df = compare_posteriors(idata, idata, ("omega_2", "beta"))

    assert isinstance(df, pd.DataFrame), "Expected pd.DataFrame output"
    expected_cols = {
        "participant_id", "parameter", "mean_laplace", "mean_nuts",
        "sd_laplace", "sd_nuts", "abs_diff_mean", "abs_diff_log_sd",
        "sd_ratio", "within_gate",
    }
    assert expected_cols.issubset(df.columns), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )

    assert (df["abs_diff_mean"] < 1e-10).all(), (
        f"Expected abs_diff_mean < 1e-10, got max={df['abs_diff_mean'].max()}"
    )
    assert (df["abs_diff_log_sd"] < 1e-10).all(), (
        f"Expected abs_diff_log_sd < 1e-10, got max={df['abs_diff_log_sd'].max()}"
    )

    omega2_rows = df[df["parameter"] == "omega_2"]
    assert omega2_rows["within_gate"].all(), (
        "Expected within_gate=True for all omega_2 rows when idatas are identical"
    )

    beta_rows = df[df["parameter"] == "beta"]
    assert beta_rows["within_gate"].isna().all(), (
        "Expected within_gate=NA for all beta rows (informational only)"
    )

    assert df["within_gate"].dtype == pd.BooleanDtype(), (
        f"Expected BooleanDtype, got {df['within_gate'].dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2: within-tolerance → passes gate
# ---------------------------------------------------------------------------


def test_compare_posteriors_within_tolerance_passes() -> None:
    """omega_2 diff < 0.3 mean and < 0.5 log_sd → within_gate=True."""
    pids = ["P0", "P1", "P2"]
    idata_lap = _make_laplace_idata([-3.5, -4.0, -3.8], omega2_sd=0.2, pids=pids)
    # NUTS: means shifted by 0.2 (< 0.3 gate) and sd similar (sd_ratio ≈ 1)
    idata_nuts = _make_nuts_like_idata(
        [-3.3, -3.8, -3.6], omega2_sd=0.2, pids=pids, rng_key=7
    )
    df = compare_posteriors(idata_lap, idata_nuts, ("omega_2",))

    omega2 = df[df["parameter"] == "omega_2"]
    assert (omega2["abs_diff_mean"] < 0.3).all(), (
        f"Expected all abs_diff_mean < 0.3; got {omega2['abs_diff_mean'].tolist()}"
    )
    assert omega2["within_gate"].all(), (
        f"Expected within_gate=True; got {omega2['within_gate'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: exceeds tolerance → gate fails
# ---------------------------------------------------------------------------


def test_compare_posteriors_exceeds_tolerance_fails_gate() -> None:
    """omega_2 mean diff 0.5 (> 0.3 gate) → all_pass=False, HARD FAIL message."""
    pids = ["P0", "P1", "P2"]
    idata_lap = _make_laplace_idata([-3.5, -4.0, -3.8], omega2_sd=0.2, pids=pids)
    # NUTS: means shifted by 0.5 in all participants (> 0.3 gate)
    idata_nuts = _make_nuts_like_idata(
        [-3.0, -3.5, -3.3], omega2_sd=0.2, pids=pids, rng_key=99
    )
    df = compare_posteriors(idata_lap, idata_nuts, ("omega_2",))

    all_pass, messages = _apply_hard_gates(df)
    assert not all_pass, "Expected all_pass=False when abs_diff_mean > 0.3"
    hard_fail_msgs = [m for m in messages if m.startswith("HARD FAIL")]
    assert len(hard_fail_msgs) > 0, (
        f"Expected at least one HARD FAIL message; got messages={messages}"
    )
    assert "omega_2" in hard_fail_msgs[0], (
        f"Expected 'omega_2' in HARD FAIL message; got {hard_fail_msgs[0]}"
    )


# ---------------------------------------------------------------------------
# Test 4: near-zero sd → handled without crash
# ---------------------------------------------------------------------------


def test_compare_posteriors_near_zero_sd_handled() -> None:
    """Near-zero sd_nuts: no crash; abs_diff_log_sd=inf and sd_ratio=inf returned."""
    pids = ["P0", "P1"]
    idata_lap = _make_laplace_idata([-3.5, -4.0], omega2_sd=0.1, pids=pids)

    # Build NUTS idata where sd ≈ 0 (all draws identical for P0 omega_2)
    rng = np.random.default_rng(123)
    omega2_samples = np.zeros((1, 200, 2))
    omega2_samples[:, :, 0] = -3.5  # P0: all draws identical → sd ≈ 0
    omega2_samples[:, :, 1] = rng.normal(-4.0, 0.1, size=(1, 200))
    posterior = {"omega_2": omega2_samples, "beta": np.ones((1, 200, 2))}
    idata_nuts = az.from_dict(
        posterior=posterior,
        coords={"chain": [0], "draw": np.arange(200), "participant_id": pids},
        dims={"omega_2": ["participant_id"], "beta": ["participant_id"]},
    )

    # Should not raise; P0 omega_2 should have abs_diff_log_sd=inf
    df = compare_posteriors(idata_lap, idata_nuts, ("omega_2",))
    p0_row = df[(df["participant_id"] == "P0") & (df["parameter"] == "omega_2")]
    assert len(p0_row) == 1, "Expected exactly one P0 omega_2 row"
    assert np.isinf(p0_row["abs_diff_log_sd"].iloc[0]) or np.isinf(
        p0_row["sd_ratio"].iloc[0]
    ), (
        "Expected inf for abs_diff_log_sd or sd_ratio when sd_nuts≈0; "
        f"got abs_diff_log_sd={p0_row['abs_diff_log_sd'].iloc[0]}, "
        f"sd_ratio={p0_row['sd_ratio'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# Test 5: missing coord → rename gracefully
# ---------------------------------------------------------------------------


def test_compare_posteriors_missing_coord_renames_gracefully() -> None:
    """NUTS idata with 'participant' coord is renamed on a copy; caller unmutated."""
    pids = ["P0", "P1", "P2"]
    idata_lap = _make_laplace_idata([-3.5, -4.0, -3.8], pids=pids)
    # NUTS idata uses old 'participant' coord (pre-hotfix OQ1 style)
    idata_nuts = _make_nuts_like_idata(
        [-3.5, -4.0, -3.8], pids=pids, coord_name="participant", rng_key=5
    )

    # Record original coord names to verify caller idata is not mutated
    original_nuts_coords = list(idata_nuts.posterior.coords)  # type: ignore[attr-defined]

    # Should rename on copy without error
    df = compare_posteriors(idata_lap, idata_nuts, ("omega_2",))

    # Caller idata is unchanged
    post_call_coords = list(idata_nuts.posterior.coords)  # type: ignore[attr-defined]
    assert original_nuts_coords == post_call_coords, (
        f"Caller idata_nuts was mutated! "
        f"Before: {original_nuts_coords}, After: {post_call_coords}"
    )

    # Comparison completed successfully
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3 participants × 1 param
    assert set(df["participant_id"].unique()) == {"P0", "P1", "P2"}


# ---------------------------------------------------------------------------
# Test 6: mismatched participant sets → RuntimeError
# ---------------------------------------------------------------------------


def test_compare_posteriors_mismatched_participant_sets_raises() -> None:
    """Different participant IDs → RuntimeError naming both sets."""
    idata_lap = _make_laplace_idata([-3.5, -4.0, -3.8], pids=["P0", "P1", "P2"])
    idata_nuts = _make_nuts_like_idata(
        [-3.5, -4.0, -3.8], pids=["P3", "P4", "P5"], rng_key=11
    )

    with pytest.raises(RuntimeError, match=r"(?i)(participant|mismatch)"):
        compare_posteriors(idata_lap, idata_nuts, ("omega_2",))


# ---------------------------------------------------------------------------
# Test 7: _apply_hard_gates — omega_2 only
# ---------------------------------------------------------------------------


def test_apply_hard_gates_omega_2_only() -> None:
    """beta with large diff is informational; only omega_2 gates the hard pass.

    Case A: omega_2 fine, beta diff=1.0 → all_pass=True; beta within_gate=NA.
    Case B: omega_2 diff=0.5 (> 0.3 gate) → all_pass=False; beta within_gate=NA.
    """
    # Build a diff DataFrame directly (do not run compare_posteriors end-to-end).
    def _make_diff_df(
        omega2_abs_diff_mean: float,
        omega2_abs_diff_log_sd: float = 0.1,
        beta_abs_diff_mean: float = 1.0,
    ) -> pd.DataFrame:
        rows = [
            # omega_2 rows
            {
                "participant_id": "P0",
                "parameter": "omega_2",
                "mean_laplace": -3.5,
                "mean_nuts": -3.5 + omega2_abs_diff_mean,
                "sd_laplace": 0.2,
                "sd_nuts": 0.2 * np.exp(omega2_abs_diff_log_sd),
                "abs_diff_mean": omega2_abs_diff_mean,
                "abs_diff_log_sd": omega2_abs_diff_log_sd,
                "sd_ratio": 1.0,
            },
            {
                "participant_id": "P1",
                "parameter": "omega_2",
                "mean_laplace": -4.0,
                "mean_nuts": -4.0 + omega2_abs_diff_mean,
                "sd_laplace": 0.2,
                "sd_nuts": 0.2 * np.exp(omega2_abs_diff_log_sd),
                "abs_diff_mean": omega2_abs_diff_mean,
                "abs_diff_log_sd": omega2_abs_diff_log_sd,
                "sd_ratio": 1.0,
            },
            # beta rows (informational only)
            {
                "participant_id": "P0",
                "parameter": "beta",
                "mean_laplace": 1.2,
                "mean_nuts": 1.2 + beta_abs_diff_mean,
                "sd_laplace": 0.1,
                "sd_nuts": 0.1,
                "abs_diff_mean": beta_abs_diff_mean,
                "abs_diff_log_sd": 0.0,
                "sd_ratio": 1.0,
            },
            {
                "participant_id": "P1",
                "parameter": "beta",
                "mean_laplace": 0.9,
                "mean_nuts": 0.9 + beta_abs_diff_mean,
                "sd_laplace": 0.1,
                "sd_nuts": 0.1,
                "abs_diff_mean": beta_abs_diff_mean,
                "abs_diff_log_sd": 0.0,
                "sd_ratio": 1.0,
            },
        ]
        df = pd.DataFrame(rows)
        # Set within_gate with BooleanDtype
        df["within_gate"] = pd.array([pd.NA] * len(df), dtype=pd.BooleanDtype())
        omega2_mask = df["parameter"] == "omega_2"
        gate_vals = (df.loc[omega2_mask, "abs_diff_mean"] < 0.3) & (
            df.loc[omega2_mask, "abs_diff_log_sd"] < 0.5
        )
        df.loc[omega2_mask, "within_gate"] = pd.array(
            gate_vals.tolist(), dtype=pd.BooleanDtype()
        )
        return df

    # Case A: omega_2 fine (diff=0.1), beta diff=1.0
    df_a = _make_diff_df(omega2_abs_diff_mean=0.1, beta_abs_diff_mean=1.0)
    all_pass_a, msgs_a = _apply_hard_gates(df_a)

    assert all_pass_a, (
        f"Case A: Expected all_pass=True when omega_2 diff<0.3 and beta is "
        f"informational. Messages: {msgs_a}"
    )

    # beta rows still have NA
    beta_rows_a = df_a[df_a["parameter"] == "beta"]
    assert beta_rows_a["within_gate"].isna().all(), (
        "Case A: beta within_gate should remain pd.NA (no silent coercion)"
    )

    # omega_2 rows have True
    omega2_rows_a = df_a[df_a["parameter"] == "omega_2"]
    assert omega2_rows_a["within_gate"].all(), (
        "Case A: omega_2 within_gate should be True"
    )

    # Case B: omega_2 diff=0.5 (above gate), beta diff=1.0
    df_b = _make_diff_df(omega2_abs_diff_mean=0.5, beta_abs_diff_mean=1.0)
    all_pass_b, msgs_b = _apply_hard_gates(df_b)

    assert not all_pass_b, (
        "Case B: Expected all_pass=False when omega_2 abs_diff_mean=0.5 > 0.3"
    )
    hard_fails_b = [m for m in msgs_b if m.startswith("HARD FAIL")]
    assert len(hard_fails_b) > 0, (
        f"Case B: Expected at least one HARD FAIL; got messages={msgs_b}"
    )

    # beta rows still NA in case B
    beta_rows_b = df_b[df_b["parameter"] == "beta"]
    assert beta_rows_b["within_gate"].isna().all(), (
        "Case B: beta within_gate should remain pd.NA even when omega_2 fails gate"
    )


# ---------------------------------------------------------------------------
# Test 8: CLI smoke — run with --skip-nuts-comparison (slow / env-gated)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_cli_compare_skip_flag(tmp_path: Path) -> None:
    """Run CLI 'run --skip-nuts-comparison' on a tiny fake sim_df; exits 0.

    This test is env-gated by ``RUN_SMOKE_TESTS=1`` (requires a Laplace fit,
    which needs JAX and jaxopt in the environment).
    """
    if not os.environ.get("RUN_SMOKE_TESTS"):
        pytest.skip("Set RUN_SMOKE_TESTS=1 to run CLI smoke test")

    # Build a minimal sim_df matching _build_arrays_single_patrl input contract.
    from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort

    n_trials = 40
    sim_df = simulate_patrl_cohort(
        n_participants=3,
        n_trials=n_trials,
        model_name="hgf_2level_patrl",
        seed=7,
    )
    sim_df_path = tmp_path / "sim_df.parquet"
    sim_df.to_parquet(str(sim_df_path))

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "validation" / "vbl06_laplace_vs_nuts.py"),
            "run",
            "--sim-df",
            str(sim_df_path),
            "--model",
            "hgf_2level_patrl",
            "--output-dir",
            str(tmp_path),
            "--skip-nuts-comparison",
        ],
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "NUTS comparison skipped" in combined_output, (
        f"Expected 'NUTS comparison skipped' in output; got:\n{combined_output}"
    )
    laplace_nc = tmp_path / "laplace.nc"
    assert laplace_nc.exists(), (
        f"Expected laplace.nc to be written to {tmp_path}"
    )


# ---------------------------------------------------------------------------
# Test 9: pick_best_cue regression guard
# ---------------------------------------------------------------------------


def test_pick_best_cue_regression_unchanged() -> None:
    """Pick-best-cue pipeline imports are unaffected by VBL-06 additions."""
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: F401
    from prl_hgf.env.simulator import generate_session  # noqa: F401
