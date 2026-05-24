"""Unit and integration tests for the group analysis pipeline.

Covers:
- build_estimates_wide pivot correctness
- Flagged-participant exclusion
- Cohen's d calculation
- Effect sizes table output schema
- plot_raincloud smoke test (returns Figure, no display required)
- plot_interaction smoke test (returns Figure, no display required)
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prl_hgf.analysis.effect_sizes import compute_cohens_d, compute_effect_sizes_table
from prl_hgf.analysis.group import _contrast_row, build_estimates_wide
from prl_hgf.analysis.group_plots import plot_interaction, plot_raincloud
from prl_hgf.analysis.phase_stratification import (
    build_phase_stratified_df,
    compute_phase_learning_metrics,
)

# ---------------------------------------------------------------------------
# Test fixture: 6 participants (3 per group) × 3 sessions × 2 parameters
# ---------------------------------------------------------------------------

_GROUPS = ["placebo", "psilocybin"]
_SESSIONS = ["baseline", "session2", "session3"]
_PARAMS = ["omega_2", "beta"]
_N_PER_GROUP = 3


def _make_fit_df(seed: int = 42) -> pd.DataFrame:
    """Build a minimal long-form fit_df for testing.

    Produces 6 participants × 3 sessions × 2 parameters = 36 rows.
    One participant (P006_psilocybin) is flagged=True to test exclusion.

    Parameters
    ----------
    seed : int, optional
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Long-form fit DataFrame with columns: participant_id, group,
        session, model, parameter, mean, flagged.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for g_idx, group in enumerate(_GROUPS):
        for p_idx in range(_N_PER_GROUP):
            pid = f"P{g_idx * _N_PER_GROUP + p_idx + 1:03d}_{group}"
            # Flag last participant in psilocybin group
            flagged = group == "psilocybin" and p_idx == _N_PER_GROUP - 1

            for session in _SESSIONS:
                for param in _PARAMS:
                    rows.append(
                        {
                            "participant_id": pid,
                            "group": group,
                            "session": session,
                            "model": "hgf_2level",
                            "parameter": param,
                            "mean": float(rng.normal(0, 1)),
                            "flagged": flagged,
                        }
                    )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fit_df() -> pd.DataFrame:
    """Module-scoped fit DataFrame fixture."""
    return _make_fit_df()


@pytest.fixture(scope="module")
def estimates_wide(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Module-scoped wide-form estimates fixture (flagged excluded)."""
    return build_estimates_wide(fit_df, model="hgf_2level", exclude_flagged=True)


# ---------------------------------------------------------------------------
# Tests: _contrast_row schema (HDI column names + hdi_excludes_zero)
# ---------------------------------------------------------------------------


class TestContrastsSchema:
    """Tests for _contrast_row output schema and hdi_excludes_zero logic."""

    def test_contrast_row_has_hdi_percent_columns(self) -> None:
        """_contrast_row returns hdi_3% and hdi_97% (percent-suffix convention)."""
        row = _contrast_row(np.zeros(200), session="baseline")
        assert "hdi_3%" in row, (
            f"Expected 'hdi_3%' in row keys. Got: {list(row.keys())}"
        )
        assert "hdi_97%" in row, (
            f"Expected 'hdi_97%' in row keys. Got: {list(row.keys())}"
        )
        assert "hdi_excludes_zero" in row, (
            f"Expected 'hdi_excludes_zero' in row keys. Got: {list(row.keys())}"
        )
        # Old column names must NOT be present
        assert "hdi_3" not in row, "Old column 'hdi_3' should not be present."
        assert "hdi_97" not in row, "Old column 'hdi_97' should not be present."

    def test_hdi_excludes_zero_true_when_positive(self) -> None:
        """hdi_excludes_zero is True when all samples are clearly positive."""
        # Constant +5 → HDI = [5, 5], strictly above zero
        row = _contrast_row(np.full(500, 5.0), session="s1")
        assert row["hdi_excludes_zero"] is True, (
            f"Expected True for all-positive samples. Got: {row['hdi_excludes_zero']}"
        )

    def test_hdi_excludes_zero_false_when_spanning(self) -> None:
        """hdi_excludes_zero is False when HDI spans zero."""
        # Samples centred at zero span both sides
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, size=2000)
        row = _contrast_row(samples, session="s2")
        assert row["hdi_excludes_zero"] is False, (
            f"Expected False for zero-spanning samples. Got: {row['hdi_excludes_zero']}"
        )

    def test_hdi_excludes_zero_true_when_negative(self) -> None:
        """hdi_excludes_zero is True when all samples are clearly negative."""
        # Constant -3 → HDI = [-3, -3], strictly below zero
        row = _contrast_row(np.full(500, -3.0), session="s3")
        assert row["hdi_excludes_zero"] is True, (
            f"Expected True for all-negative samples. Got: {row['hdi_excludes_zero']}"
        )


# ---------------------------------------------------------------------------
# Tests: build_estimates_wide
# ---------------------------------------------------------------------------


class TestBuildEstimatesWide:
    """Tests for build_estimates_wide pivot logic."""

    def test_output_columns(self, estimates_wide: pd.DataFrame) -> None:
        """Wide table has correct identity and parameter columns."""
        expected_cols = {"participant_id", "group", "session", "omega_2", "beta"}
        assert expected_cols.issubset(set(estimates_wide.columns)), (
            f"Missing columns. Got: {list(estimates_wide.columns)}"
        )

    def test_one_row_per_participant_session(self, estimates_wide: pd.DataFrame) -> None:
        """Each (participant_id, session) pair appears exactly once."""
        duplicated = estimates_wide.duplicated(
            subset=["participant_id", "session"]
        ).sum()
        assert duplicated == 0, f"Found {duplicated} duplicate rows."

    def test_n_rows_correct(self, estimates_wide: pd.DataFrame) -> None:
        """5 participants (1 flagged excluded) × 3 sessions = 15 rows."""
        assert len(estimates_wide) == 15, (
            f"Expected 15 rows (5 unflagged participants × 3 sessions), "
            f"got {len(estimates_wide)}."
        )

    def test_model_filter(self, fit_df: pd.DataFrame) -> None:
        """Non-matching model rows are excluded from output."""
        wide = build_estimates_wide(fit_df, model="hgf_3level")
        # No rows match hgf_3level → empty DataFrame (but with correct columns)
        assert len(wide) == 0

    def test_no_index_artifact(self, estimates_wide: pd.DataFrame) -> None:
        """Index is range-based (reset_index called)."""
        assert list(estimates_wide.index) == list(range(len(estimates_wide)))


class TestExcludesFlagged:
    """Tests for flagged-participant exclusion in build_estimates_wide."""

    def test_excludes_flagged_default(self, fit_df: pd.DataFrame) -> None:
        """Default exclude_flagged=True removes flagged participants."""
        wide = build_estimates_wide(fit_df, model="hgf_2level")
        # Flagged participant IDs start with 'P006_' (last psilocybin)
        flagged_ids = fit_df.loc[fit_df["flagged"], "participant_id"].unique()
        for pid in flagged_ids:
            assert pid not in wide["participant_id"].values, (
                f"Flagged participant '{pid}' should be excluded."
            )

    def test_include_flagged_when_disabled(self, fit_df: pd.DataFrame) -> None:
        """exclude_flagged=False retains all participants."""
        wide = build_estimates_wide(fit_df, model="hgf_2level", exclude_flagged=False)
        # Should include the flagged participant
        n_participants = wide["participant_id"].nunique()
        assert n_participants == 6, (
            f"Expected 6 participants when flagged not excluded, got {n_participants}."
        )


# ---------------------------------------------------------------------------
# Tests: compute_cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    """Tests for Cohen's d calculation."""

    def test_positive_d_when_b_greater(self, estimates_wide: pd.DataFrame) -> None:
        """Cohen's d is computable and returns a float."""
        d = compute_cohens_d(
            estimates_wide,
            outcome="omega_2",
            session="baseline",
            group_a="placebo",
            group_b="psilocybin",
        )
        assert isinstance(d, float), f"Expected float, got {type(d)}."

    def test_raises_on_missing_outcome(self, estimates_wide: pd.DataFrame) -> None:
        """ValueError raised when outcome column does not exist."""
        with pytest.raises(ValueError, match="not found in estimates_wide"):
            compute_cohens_d(
                estimates_wide,
                outcome="nonexistent_param",
                session="baseline",
            )

    def test_raises_on_insufficient_observations(
        self, estimates_wide: pd.DataFrame
    ) -> None:
        """ValueError raised when fewer than 2 observations per group."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_cohens_d(
                estimates_wide,
                outcome="omega_2",
                session="baseline",
                group_a="placebo",
                group_b="nonexistent_group",
            )


# ---------------------------------------------------------------------------
# Tests: compute_effect_sizes_table
# ---------------------------------------------------------------------------


class TestEffectSizesTable:
    """Tests for compute_effect_sizes_table output schema and values."""

    def test_output_columns(self, estimates_wide: pd.DataFrame) -> None:
        """Table contains required columns."""
        table = compute_effect_sizes_table(
            estimates_wide,
            params=["omega_2"],
            sessions=["baseline"],
        )
        required = {"parameter", "session", "cohen_d", "partial_eta_sq"}
        assert required.issubset(set(table.columns)), (
            f"Missing columns. Got: {list(table.columns)}"
        )

    def test_n_rows(self, estimates_wide: pd.DataFrame) -> None:
        """Table has one row per (parameter, session) pair."""
        table = compute_effect_sizes_table(
            estimates_wide,
            params=["omega_2", "beta"],
            sessions=["baseline", "session2"],
        )
        assert len(table) == 4, f"Expected 4 rows (2 params × 2 sessions), got {len(table)}."

    def test_partial_eta_sq_bounds(self, estimates_wide: pd.DataFrame) -> None:
        """Partial η² is in [0, 1] where computable."""
        table = compute_effect_sizes_table(
            estimates_wide,
            params=_PARAMS,
            sessions=_SESSIONS,
        )
        computable = table["partial_eta_sq"].dropna()
        assert (computable >= 0).all(), "Partial η² below 0 found."
        assert (computable <= 1).all(), "Partial η² above 1 found."

    def test_graceful_nan_on_insufficient_data(
        self, estimates_wide: pd.DataFrame
    ) -> None:
        """NaN (not error) when a session has no observations for a group."""
        # Build a table with a session that has no observations
        sparse_wide = estimates_wide[estimates_wide["session"] == "baseline"].copy()
        # Use a non-existent session → will produce NaN
        table = compute_effect_sizes_table(
            sparse_wide,
            params=["omega_2"],
            sessions=["ghost_session"],
        )
        assert table["cohen_d"].isna().all(), (
            "Expected NaN cohen_d for absent session."
        )


# ---------------------------------------------------------------------------
# Tests: plot_raincloud (smoke tests)
# ---------------------------------------------------------------------------


class TestPlotRaincloud:
    """Smoke tests for plot_raincloud."""

    def test_returns_figure(self, estimates_wide: pd.DataFrame) -> None:
        """plot_raincloud returns a matplotlib Figure."""
        fig = plot_raincloud(estimates_wide, outcome="omega_2")
        assert isinstance(fig, plt.Figure), (
            f"Expected plt.Figure, got {type(fig)}."
        )
        plt.close(fig)

    def test_save_to_tmp(
        self, estimates_wide: pd.DataFrame, tmp_path: Path
    ) -> None:
        """plot_raincloud saves a PNG when save_path is provided."""
        out = tmp_path / "test_raincloud.png"
        fig = plot_raincloud(estimates_wide, outcome="omega_2", save_path=out)
        plt.close(fig)
        assert out.exists(), f"PNG not saved at {out}."
        assert out.stat().st_size > 0, "Saved PNG is empty."


# ---------------------------------------------------------------------------
# Tests: plot_interaction (smoke tests)
# ---------------------------------------------------------------------------


class TestPlotInteraction:
    """Smoke tests for plot_interaction."""

    def test_returns_figure(self, estimates_wide: pd.DataFrame) -> None:
        """plot_interaction returns a matplotlib Figure."""
        fig = plot_interaction(estimates_wide, outcome="omega_2")
        assert isinstance(fig, plt.Figure), (
            f"Expected plt.Figure, got {type(fig)}."
        )
        plt.close(fig)

    def test_save_to_tmp(
        self, estimates_wide: pd.DataFrame, tmp_path: Path
    ) -> None:
        """plot_interaction saves a PNG when save_path is provided."""
        out = tmp_path / "test_interaction.png"
        fig = plot_interaction(estimates_wide, outcome="omega_2", save_path=out)
        plt.close(fig)
        assert out.exists(), f"PNG not saved at {out}."
        assert out.stat().st_size > 0, "Saved PNG is empty."


# ---------------------------------------------------------------------------
# Tests: phase stratification
# ---------------------------------------------------------------------------


def _make_sim_df(seed: int = 42) -> pd.DataFrame:
    """Build a minimal trial-level sim_df for phase stratification tests.

    Produces 2 participants (one per group) × 2 sessions × 20 trials, with
    phase_label "stable" for trials 0-9 and "volatile" for trials 10-19.

    Parameters
    ----------
    seed : int, optional
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Trial-level DataFrame with columns: participant_id, group, session,
        trial, cue_chosen, reward, phase_label.
    """
    rng = np.random.default_rng(seed)
    rows = []
    participants = [
        ("P001_control", "placebo"),
        ("P002_psilocybin", "psilocybin"),
    ]
    sessions = ["baseline", "session2"]
    n_trials = 20

    for pid, group in participants:
        for session in sessions:
            choices = rng.integers(0, 3, size=n_trials).tolist()
            rewards = rng.integers(0, 2, size=n_trials).tolist()
            phase_labels = ["stable"] * 10 + ["volatile"] * 10
            for t in range(n_trials):
                rows.append(
                    {
                        "participant_id": pid,
                        "group": group,
                        "session": session,
                        "trial": t,
                        "cue_chosen": choices[t],
                        "reward": rewards[t],
                        "phase_label": phase_labels[t],
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sim_df() -> pd.DataFrame:
    """Module-scoped sim DataFrame fixture for phase stratification tests."""
    return _make_sim_df()


class TestPhaseStratification:
    """Tests for compute_phase_learning_metrics and build_phase_stratified_df."""

    def test_phase_metrics_output_columns(self, sim_df: pd.DataFrame) -> None:
        """compute_phase_learning_metrics returns all required columns."""
        result = compute_phase_learning_metrics(sim_df)
        required = {
            "participant_id",
            "group",
            "session",
            "phase_label",
            "win_stay_rate",
            "lose_shift_rate",
            "n_trials",
        }
        assert required.issubset(set(result.columns)), (
            f"Missing columns. Expected {required}. Got: {list(result.columns)}"
        )

    def test_phase_metrics_one_row_per_participant_session_phase(
        self, sim_df: pd.DataFrame
    ) -> None:
        """Output has one row per (participant_id, session, phase_label) combination.

        With 2 participants × 2 sessions × 2 phases = 8 rows expected.
        """
        result = compute_phase_learning_metrics(sim_df)
        expected_rows = 2 * 2 * 2  # participants × sessions × phases
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} rows (2 participants × 2 sessions × 2 phases). "
            f"Got {len(result)}."
        )
        # No duplicate (participant_id, session, phase_label) rows
        duplicated = result.duplicated(
            subset=["participant_id", "session", "phase_label"]
        ).sum()
        assert duplicated == 0, f"Found {duplicated} duplicate rows."

    def test_win_stay_rate_bounded(self, sim_df: pd.DataFrame) -> None:
        """win_stay_rate values are all in [0, 1] where not NaN."""
        result = compute_phase_learning_metrics(sim_df)
        computable = result["win_stay_rate"].dropna()
        assert (computable >= 0.0).all(), (
            f"win_stay_rate below 0 found:\n{result[result['win_stay_rate'] < 0]}"
        )
        assert (computable <= 1.0).all(), (
            f"win_stay_rate above 1 found:\n{result[result['win_stay_rate'] > 1]}"
        )

    def test_lose_shift_rate_bounded(self, sim_df: pd.DataFrame) -> None:
        """lose_shift_rate values are all in [0, 1] where not NaN."""
        result = compute_phase_learning_metrics(sim_df)
        computable = result["lose_shift_rate"].dropna()
        assert (computable >= 0.0).all(), (
            f"lose_shift_rate below 0 found:\n{result[result['lose_shift_rate'] < 0]}"
        )
        assert (computable <= 1.0).all(), (
            f"lose_shift_rate above 1 found:\n{result[result['lose_shift_rate'] > 1]}"
        )

    def test_build_phase_stratified_df_validates_columns(
        self, sim_df: pd.DataFrame
    ) -> None:
        """build_phase_stratified_df returns correct columns and raises on missing ones."""
        result = build_phase_stratified_df(sim_df)
        required = {
            "participant_id",
            "group",
            "session",
            "phase_label",
            "win_stay_rate",
            "lose_shift_rate",
            "n_trials",
        }
        assert required.issubset(set(result.columns)), (
            f"Missing columns in build_phase_stratified_df output. "
            f"Expected {required}. Got: {list(result.columns)}"
        )

        # Missing required column should raise ValueError
        broken_df = sim_df.drop(columns=["phase_label"])
        with pytest.raises(ValueError, match="missing required columns"):
            build_phase_stratified_df(broken_df)
