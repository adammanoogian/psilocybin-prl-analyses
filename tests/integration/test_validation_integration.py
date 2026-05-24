"""Integration tests for the Phase 5 validation pipeline.

Tests cover the full recovery + BMS pipeline on tiny synthetic datasets,
without running actual MCMC fitting.  All tests are marked
``@pytest.mark.integration`` for selective execution::

    pytest tests/test_validation_integration.py -v
    pytest tests/ -v -m integration

The CSV output tests validate CMP-04 (summary table requirement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_TRIALS = 10
_PIDS = ["P001", "P002", "P003", "P004", "P005"]
_FLAGGED_PID = "P005"
_GROUP = "test_group"
_SESSION = "baseline"
_MODEL_NAMES = ["hgf_2level", "hgf_3level"]

# True parameter values used in synthetic fixtures
_TRUE_PARAMS_2LEVEL = {
    "omega_2": -3.0,
    "beta": 2.0,
    "zeta": 0.3,
}
_TRUE_PARAMS_3LEVEL = {
    "omega_2": -3.0,
    "omega_3": -6.0,
    "kappa": 1.0,
    "beta": 2.0,
    "zeta": 0.3,
}


@pytest.fixture(scope="module")
def synthetic_sim_df() -> pd.DataFrame:
    """Tiny trial-level sim_df with 3 participants, 10 trials each.

    Includes true_* columns with per-participant offsets to give
    inter-subject variability for recovery.
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    for _i, pid in enumerate(_PIDS):
        offset = float(rng.normal(0, 1.0))
        for t in range(_N_TRIALS):
            rows.append(
                {
                    "participant_id": pid,
                    "group": _GROUP,
                    "session": _SESSION,
                    "trial": t,
                    "cue_chosen": int(rng.integers(0, 3)),
                    "reward": int(rng.integers(0, 2)),
                    # Trial-level columns required by compute_batch_waic
                    "reward_c0": float(rng.uniform(0, 1)),
                    "reward_c1": float(rng.uniform(0, 1)),
                    "reward_c2": float(rng.uniform(0, 1)),
                    "observed_c0": 1,
                    "observed_c1": 0,
                    "observed_c2": 0,
                    "choice": 0,
                    # True parameters (constant across trials for one participant)
                    "true_omega_2": _TRUE_PARAMS_3LEVEL["omega_2"] + offset,
                    "true_omega_3": _TRUE_PARAMS_3LEVEL["omega_3"] + offset * 0.5,
                    "true_kappa": _TRUE_PARAMS_3LEVEL["kappa"] + offset * 0.5,
                    "true_beta": _TRUE_PARAMS_3LEVEL["beta"] + abs(offset) * 0.5,
                    "true_zeta": _TRUE_PARAMS_3LEVEL["zeta"] + offset * 0.5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_fit_df(synthetic_sim_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form fit_df with 4 unflagged and 1 flagged participant.

    Posterior means are close to true values (tiny noise).  P005 is flagged.
    """
    rng = np.random.default_rng(seed=99)
    true_wide = (
        synthetic_sim_df.groupby(["participant_id", "group", "session"])[
            ["true_omega_2", "true_omega_3", "true_kappa", "true_beta", "true_zeta"]
        ]
        .first()
        .reset_index()
    )

    rows = []
    for _, row in true_wide.iterrows():
        pid = str(row["participant_id"])
        is_flagged = pid == _FLAGGED_PID
        for param, true_col in [
            ("omega_2", "true_omega_2"),
            ("omega_3", "true_omega_3"),
            ("kappa", "true_kappa"),
            ("beta", "true_beta"),
            ("zeta", "true_zeta"),
        ]:
            true_val = float(row[true_col])
            noise = float(rng.normal(0, 0.05))
            rows.append(
                {
                    "participant_id": pid,
                    "group": row["group"],
                    "session": row["session"],
                    "model": "hgf_3level",
                    "parameter": param,
                    "mean": true_val + noise,
                    "sd": 0.1,
                    "hdi_3%": true_val + noise - 0.2,
                    "hdi_97%": true_val + noise + 0.2,
                    "r_hat": 1.01,
                    "ess": 800,
                    "flagged": is_flagged,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_waic_df() -> pd.DataFrame:
    """WAIC DataFrame for 4 participants (P001-P004) × 2 models."""
    rng = np.random.default_rng(seed=7)
    rows = []
    for pid in ["P001", "P002", "P003", "P004"]:
        for model in _MODEL_NAMES:
            # hgf_2level gets slightly better elpd
            base = -120.0 if model == "hgf_2level" else -140.0
            rows.append(
                {
                    "participant_id": pid,
                    "group": _GROUP,
                    "session": _SESSION,
                    "model": model,
                    "elpd_waic": float(rng.normal(base, 5.0)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Full recovery pipeline (exclude_flagged=True, min_n=0)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_recovery_pipeline_integration(
    synthetic_sim_df: pd.DataFrame,
    synthetic_fit_df: pd.DataFrame,
) -> None:
    """Recovery pipeline with flagged exclusion should yield 2 participants."""
    from prl_hgf.analysis.recovery import (
        build_recovery_df,
        compute_correlation_matrix,
        compute_recovery_metrics,
    )

    recovery_df = build_recovery_df(
        synthetic_sim_df,
        synthetic_fit_df,
        exclude_flagged=True,
        min_n=0,
    )

    # P005 is flagged, so only P001-P004 remain
    assert len(recovery_df) == 4, (
        f"Expected 4 rows (P001-P004), got {len(recovery_df)}"
    )

    metrics_df = compute_recovery_metrics(recovery_df)
    assert "parameter" in metrics_df.columns, "metrics_df missing 'parameter' column"
    assert "r" in metrics_df.columns, "metrics_df missing 'r' column"
    assert "passes_threshold" in metrics_df.columns

    corr_df = compute_correlation_matrix(recovery_df)
    assert corr_df.shape[0] == corr_df.shape[1], (
        f"Correlation matrix is not square: {corr_df.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Recovery with flagged participants included
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_recovery_pipeline_with_flagged_included(
    synthetic_sim_df: pd.DataFrame,
    synthetic_fit_df: pd.DataFrame,
) -> None:
    """With exclude_flagged=False all 3 participants should appear."""
    from prl_hgf.analysis.recovery import build_recovery_df

    recovery_df = build_recovery_df(
        synthetic_sim_df,
        synthetic_fit_df,
        exclude_flagged=False,
        min_n=0,
    )

    assert len(recovery_df) == 5, (
        f"Expected 5 rows (all participants), got {len(recovery_df)}"
    )
    assert set(recovery_df["participant_id"].tolist()) == set(_PIDS)


# ---------------------------------------------------------------------------
# Test 3: BMS pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bms_pipeline_integration(synthetic_waic_df: pd.DataFrame) -> None:
    """BMS on synthetic WAIC data should return expected structure."""
    from prl_hgf.analysis.bms import run_group_bms

    # Build elpd matrix (2 subjects x 2 models)
    pivot = synthetic_waic_df.pivot(
        index="participant_id",
        columns="model",
        values="elpd_waic",
    )
    pivot = pivot[_MODEL_NAMES]
    elpd_matrix = pivot.to_numpy(dtype=float)

    result = run_group_bms(elpd_matrix, _MODEL_NAMES, group_label="all")

    # Expected keys
    for key in ("alpha", "exp_r", "xp", "pxp", "bor", "model_names", "n_subjects"):
        assert key in result, f"Missing key '{key}' in BMS result"

    # xp should sum to ~1.0
    xp_sum = float(sum(result["xp"]))
    assert abs(xp_sum - 1.0) < 0.05, (
        f"Expected xp sum ~1.0, got {xp_sum:.4f}"
    )

    # n_subjects should match elpd_matrix rows (4 unflagged participants)
    assert result["n_subjects"] == elpd_matrix.shape[0]


# ---------------------------------------------------------------------------
# Test 4: Plot generation (recovery scatter + correlation matrix)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_plot_generation_integration(
    tmp_path: Path,
    synthetic_sim_df: pd.DataFrame,
    synthetic_fit_df: pd.DataFrame,
) -> None:
    """Recovery plots should produce non-empty PNG files in tmpdir."""
    import matplotlib.pyplot as plt

    from prl_hgf.analysis.plots import plot_correlation_matrix, plot_recovery_scatter
    from prl_hgf.analysis.recovery import (
        build_recovery_df,
        compute_correlation_matrix,
        compute_recovery_metrics,
    )

    recovery_df = build_recovery_df(
        synthetic_sim_df, synthetic_fit_df, exclude_flagged=True, min_n=0
    )
    metrics_df = compute_recovery_metrics(recovery_df)
    corr_df = compute_correlation_matrix(recovery_df)

    scatter_path = tmp_path / "test_scatter.png"
    fig = plot_recovery_scatter(recovery_df, metrics_df, save_path=scatter_path)
    plt.close(fig)

    assert scatter_path.exists(), f"Scatter PNG not created at {scatter_path}"
    assert scatter_path.stat().st_size > 0, "Scatter PNG is empty"

    corr_path = tmp_path / "test_corr.png"
    fig = plot_correlation_matrix(corr_df, save_path=corr_path)
    plt.close(fig)

    assert corr_path.exists(), f"Correlation matrix PNG not created at {corr_path}"
    assert corr_path.stat().st_size > 0, "Correlation matrix PNG is empty"


# ---------------------------------------------------------------------------
# Test 5: EP plot generation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_plot_exceedance_integration(tmp_path: Path) -> None:
    """plot_exceedance_probabilities should produce a non-empty PNG."""
    import matplotlib.pyplot as plt
    import numpy as np

    from prl_hgf.analysis.bms import plot_exceedance_probabilities

    bms_results = {
        "all": {
            "xp": np.array([0.85, 0.15]),
            "pxp": np.array([0.80, 0.20]),
            "model_names": _MODEL_NAMES,
            "n_subjects": 2,
            "bor": 0.08,
            "group_label": "all",
        }
    }

    ep_path = tmp_path / "test_ep.png"
    fig = plot_exceedance_probabilities(bms_results, save_path=ep_path)
    plt.close(fig)

    assert ep_path.exists(), f"EP PNG not created at {ep_path}"
    assert ep_path.stat().st_size > 0, "EP PNG is empty"


# ---------------------------------------------------------------------------
# Test 6: CSV output validation (CMP-04)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_csv_outputs_written(
    tmp_path: Path,
    synthetic_sim_df: pd.DataFrame,
    synthetic_fit_df: pd.DataFrame,
) -> None:
    """BMS and recovery metrics CSVs are written with correct columns (CMP-04)."""
    import numpy as np

    from prl_hgf.analysis.bms import run_group_bms
    from prl_hgf.analysis.recovery import (
        build_recovery_df,
        compute_recovery_metrics,
    )

    # --- BMS summary CSV ---
    # Build a synthetic BMS result matching the run_group_bms output schema
    rng = np.random.default_rng(seed=5)
    elpd = np.column_stack(
        [rng.normal(-100, 5, 4), rng.normal(-120, 5, 4)]
    )
    bms_result = run_group_bms(elpd, _MODEL_NAMES, group_label="all")

    flat_rows = []
    for m_idx, m_name in enumerate(bms_result["model_names"]):
        flat_rows.append(
            {
                "group": bms_result["group_label"],
                "model": m_name,
                "exp_r": float(bms_result["exp_r"][m_idx]),
                "xp": float(bms_result["xp"][m_idx]),
                "pxp": float(bms_result["pxp"][m_idx]),
                "bor": float(bms_result["bor"]),
            }
        )
    bms_df = pd.DataFrame(flat_rows, columns=["group", "model", "exp_r", "xp", "pxp", "bor"])
    bms_csv = tmp_path / "bms_summary.csv"
    bms_df.to_csv(bms_csv, index=False)

    assert bms_csv.exists(), "bms_summary.csv was not created"

    bms_loaded = pd.read_csv(bms_csv)
    assert set(bms_loaded.columns) >= {"model", "exp_r", "xp", "pxp", "bor"}, (
        f"bms_summary.csv missing required columns. Got: {list(bms_loaded.columns)}"
    )
    assert len(bms_loaded) == len(_MODEL_NAMES), (
        f"Expected {len(_MODEL_NAMES)} rows, got {len(bms_loaded)}"
    )

    # --- Recovery metrics CSV ---
    recovery_df = build_recovery_df(
        synthetic_sim_df, synthetic_fit_df, exclude_flagged=True, min_n=0
    )
    metrics_df = compute_recovery_metrics(recovery_df)

    metrics_csv = tmp_path / "recovery_metrics_hgf_3level.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    assert metrics_csv.exists(), "recovery_metrics CSV was not created"

    metrics_loaded = pd.read_csv(metrics_csv)
    expected_cols = {"parameter", "r", "p", "bias", "rmse", "n", "passes_threshold"}
    assert expected_cols.issubset(set(metrics_loaded.columns)), (
        f"recovery_metrics CSV missing required columns. "
        f"Got: {list(metrics_loaded.columns)}"
    )
    assert len(metrics_loaded) > 0, "recovery_metrics CSV is empty"
