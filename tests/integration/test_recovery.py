"""Unit tests for the parameter recovery analysis module.

Tests cover:
- DataFrame joining (sim_df + fit_df → recovery_df)
- Flagged-participant exclusion logic
- min_n guard enforcement (REC-04)
- Per-parameter metric computation (r, bias, RMSE)
- Output column schema
- Correlation matrix shape
- Plot function smoke tests (return Figure, no display required)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

_PARAMS = ["omega_2", "omega_3", "kappa", "beta", "zeta"]
_TRUE_VALS = {
    "omega_2": -3.0,
    "omega_3": -6.0,
    "kappa": 1.0,
    "beta": 2.0,
    "zeta": 0.3,
}
_N_TOTAL = 10  # 9 unflagged + 1 flagged
_N_TRIALS = 5  # small; only used to build trial-level sim_df


def _make_sim_df() -> pd.DataFrame:
    """Build a minimal trial-level sim_df with true_* columns.

    Each of the 10 participants has ``_N_TRIALS`` rows.  True parameters
    are deterministic (centre values + per-participant offset so each
    participant has a clearly different value), repeated across all trials.

    Offsets are scaled to give adequate inter-subject spread relative to the
    recovery noise (0.05) so that Pearson r is robustly high in
    ``test_compute_recovery_metrics_perfect``.
    """
    rng = np.random.default_rng(seed=0)
    rows = []
    for i in range(_N_TOTAL):
        # Larger offsets ensure SNR >> 1 for the near-perfect recovery test
        offset = rng.normal(0, 1.0)
        for t in range(_N_TRIALS):
            rows.append(
                {
                    "participant_id": f"sub_{i:02d}",
                    "group": "psilocybin" if i < 5 else "placebo",
                    "session": "baseline",
                    "session_idx": 0,
                    "trial": t,
                    "cue_chosen": 0,
                    "reward": 1,
                    "true_omega_2": _TRUE_VALS["omega_2"] + offset,
                    "true_omega_3": _TRUE_VALS["omega_3"] + offset * 0.5,
                    "true_kappa": _TRUE_VALS["kappa"] + offset * 0.5,
                    "true_beta": _TRUE_VALS["beta"] + abs(offset) * 0.5,
                    "true_zeta": _TRUE_VALS["zeta"] + offset * 0.5,
                }
            )
    return pd.DataFrame(rows)


def _make_fit_df(sim_df: pd.DataFrame, flagged_id: str = "sub_09") -> pd.DataFrame:
    """Build a long-form fit_df matching the FIT-04 schema.

    Recovered values are the true values plus tiny noise (good recovery).
    Participant ``flagged_id`` has ``flagged=True`` on all its rows.
    """
    rng = np.random.default_rng(seed=1)
    # Extract one row per participant to get true values
    true_wide = sim_df.groupby(["participant_id", "group", "session"])[
        [f"true_{p}" for p in _PARAMS]
    ].first().reset_index()

    rows = []
    for _, row in true_wide.iterrows():
        pid = row["participant_id"]
        is_flagged = pid == flagged_id
        for param in _PARAMS:
            true_val = float(row[f"true_{param}"])
            noise = rng.normal(0, 0.05)
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
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sim_df, fit_df) for the 10-participant synthetic scenario."""
    sim_df = _make_sim_df()
    fit_df = _make_fit_df(sim_df)
    return sim_df, fit_df


# ---------------------------------------------------------------------------
# Tests: build_recovery_df
# ---------------------------------------------------------------------------


def test_build_recovery_df_shape(synthetic_data):
    """Unflagged merge should yield 9 rows with all expected columns."""
    from prl_hgf.analysis.recovery import build_recovery_df

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)

    assert len(recovery_df) == 9, f"Expected 9 rows, got {len(recovery_df)}"

    # Check true_* columns present
    for param in _PARAMS:
        assert f"true_{param}" in recovery_df.columns, f"Missing true_{param}"

    # Check fitted columns present
    for param in _PARAMS:
        assert param in recovery_df.columns, f"Missing fitted column {param}"


def test_build_recovery_df_include_flagged(synthetic_data):
    """With exclude_flagged=False all 10 participants should appear."""
    from prl_hgf.analysis.recovery import build_recovery_df

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=False, min_n=0)

    assert len(recovery_df) == 10, f"Expected 10 rows, got {len(recovery_df)}"


def test_build_recovery_df_min_n_guard(synthetic_data):
    """min_n guard raises ValueError when threshold not met, passes when it is."""
    from prl_hgf.analysis.recovery import build_recovery_df

    sim_df, fit_df = synthetic_data

    # Default min_n=30 should raise because only 9 unflagged participants
    with pytest.raises(ValueError, match="at least 30 participants"):
        build_recovery_df(sim_df, fit_df, exclude_flagged=True)

    # min_n=5 should succeed because 9 > 5
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=5)
    assert len(recovery_df) == 9


# ---------------------------------------------------------------------------
# Tests: compute_recovery_metrics
# ---------------------------------------------------------------------------


def test_compute_recovery_metrics_perfect(synthetic_data):
    """With tiny noise, all parameters should have r > 0.95."""
    from prl_hgf.analysis.recovery import build_recovery_df, compute_recovery_metrics

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    for _, row in metrics_df.iterrows():
        assert row["r"] > 0.95, (
            f"Parameter {row['parameter']}: r={row['r']:.3f} < 0.95 "
            "(expected near-perfect recovery with tiny noise fixture)"
        )


def test_compute_recovery_metrics_columns(synthetic_data):
    """Output DataFrame must have the seven required columns."""
    from prl_hgf.analysis.recovery import build_recovery_df, compute_recovery_metrics

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    expected_cols = {"parameter", "r", "p", "bias", "rmse", "n", "passes_threshold"}
    assert set(metrics_df.columns) == expected_cols, (
        f"Column mismatch. Got: {set(metrics_df.columns)}"
    )


# ---------------------------------------------------------------------------
# Tests: compute_correlation_matrix
# ---------------------------------------------------------------------------


def test_compute_correlation_matrix_shape(synthetic_data):
    """Correlation matrix should be square with param names as index/columns."""
    from prl_hgf.analysis.recovery import build_recovery_df, compute_correlation_matrix

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)
    corr_df = compute_correlation_matrix(recovery_df)

    n = len(corr_df)
    assert corr_df.shape == (n, n), f"Expected square matrix, got {corr_df.shape}"

    # All param cols should appear (all 5 present in fixture)
    for param in _PARAMS:
        assert param in corr_df.columns, f"Missing column {param} in corr matrix"
        assert param in corr_df.index, f"Missing index {param} in corr matrix"

    # Diagonal should be 1.0
    for param in _PARAMS:
        assert abs(corr_df.loc[param, param] - 1.0) < 1e-10, (
            f"Diagonal element for {param} != 1.0"
        )


# ---------------------------------------------------------------------------
# Tests: plot smoke tests
# ---------------------------------------------------------------------------


def test_plot_recovery_scatter_runs(synthetic_data):
    """plot_recovery_scatter should return a matplotlib Figure without error."""
    import matplotlib.pyplot as plt

    from prl_hgf.analysis.plots import plot_recovery_scatter
    from prl_hgf.analysis.recovery import build_recovery_df, compute_recovery_metrics

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    fig = plot_recovery_scatter(recovery_df, metrics_df)
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure"
    plt.close(fig)


def test_plot_correlation_matrix_runs(synthetic_data):
    """plot_correlation_matrix should return a matplotlib Figure without error."""
    import matplotlib.pyplot as plt

    from prl_hgf.analysis.plots import plot_correlation_matrix
    from prl_hgf.analysis.recovery import build_recovery_df, compute_correlation_matrix

    sim_df, fit_df = synthetic_data
    recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=0)
    corr_df = compute_correlation_matrix(recovery_df)

    fig = plot_correlation_matrix(corr_df)
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure"
    plt.close(fig)
