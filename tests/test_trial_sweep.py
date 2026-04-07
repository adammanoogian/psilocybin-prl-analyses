"""Unit tests for trial count sweep functions in power/precheck.py.

Tests cover SweepPoint dataclass, find_minimum_trial_count (basic, none,
omega_3 exclusion, custom eligible), and plot_trial_sweep (figure creation
and reference line).

No calls to simulate_batch or fit_batch — all tests use synthetic DataFrames
and SweepPoint instances.

Run with::

    conda run -n ds_env python -m pytest tests/test_trial_sweep.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from prl_hgf.power.precheck import (  # noqa: E402
    SweepPoint,
    find_minimum_trial_count,
    plot_trial_sweep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(params: list[str], r_vals: list[float]) -> pd.DataFrame:
    """Build a minimal metrics_df compatible with find_minimum_trial_count.

    Parameters
    ----------
    params : list[str]
        Parameter names.
    r_vals : list[float]
        Pearson r for each parameter (same order as params).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``parameter``, ``r``, ``p``, ``bias``,
        ``rmse``, ``n``, ``passes_threshold``.
    """
    return pd.DataFrame(
        {
            "parameter": params,
            "r": r_vals,
            "p": [0.01] * len(params),
            "bias": [0.0] * len(params),
            "rmse": [0.1] * len(params),
            "n": [25] * len(params),
            "passes_threshold": [r >= 0.7 for r in r_vals],
        }
    )


def _make_sweep_point(trial_count: int, params: list[str], r_vals: list[float]) -> SweepPoint:
    """Return a SweepPoint with synthetic metrics.

    Parameters
    ----------
    trial_count : int
        Total trial count for this grid point.
    params : list[str]
        Parameter names.
    r_vals : list[float]
        Pearson r values.

    Returns
    -------
    SweepPoint
    """
    return SweepPoint(
        trial_count=trial_count,
        metrics_df=_make_metrics(params, r_vals),
        n_flagged=2,
        n_total=30,
    )


# ---------------------------------------------------------------------------
# test_sweep_point_dataclass
# ---------------------------------------------------------------------------


def test_sweep_point_dataclass() -> None:
    """SweepPoint is frozen and has the expected fields.

    Verifies field accessibility and that mutation raises FrozenInstanceError.
    """
    pt = SweepPoint(
        trial_count=200,
        metrics_df=pd.DataFrame({"parameter": ["omega_2"], "r": [0.75]}),
        n_flagged=3,
        n_total=30,
    )

    assert pt.trial_count == 200
    assert pt.n_flagged == 3
    assert pt.n_total == 30
    assert isinstance(pt.metrics_df, pd.DataFrame)

    # Frozen — mutation must raise
    with pytest.raises(FrozenInstanceError):
        pt.trial_count = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# test_find_minimum_trial_count_basic
# ---------------------------------------------------------------------------


def test_find_minimum_trial_count_basic() -> None:
    """find_minimum_trial_count returns the first point where all params pass.

    Setup:
    - trial_count=150: omega_2 r=0.60, beta r=0.65 (both fail)
    - trial_count=250: omega_2 r=0.75, beta r=0.72 (both pass)
    - trial_count=420: omega_2 r=0.85, beta r=0.80 (both pass)

    Expected: 250 (first where both pass).
    """
    sweep = [
        _make_sweep_point(150, ["omega_2", "beta"], [0.60, 0.65]),
        _make_sweep_point(250, ["omega_2", "beta"], [0.75, 0.72]),
        _make_sweep_point(420, ["omega_2", "beta"], [0.85, 0.80]),
    ]
    result = find_minimum_trial_count(sweep)
    assert result == 250, f"Expected 250, got {result}"


# ---------------------------------------------------------------------------
# test_find_minimum_trial_count_none
# ---------------------------------------------------------------------------


def test_find_minimum_trial_count_none() -> None:
    """find_minimum_trial_count returns None when no grid point satisfies threshold.

    Setup: beta never reaches r=0.7 across all grid points.
    """
    sweep = [
        _make_sweep_point(150, ["omega_2", "beta"], [0.75, 0.50]),
        _make_sweep_point(250, ["omega_2", "beta"], [0.80, 0.60]),
        _make_sweep_point(420, ["omega_2", "beta"], [0.85, 0.68]),
    ]
    result = find_minimum_trial_count(sweep)
    assert result is None, f"Expected None, got {result}"


# ---------------------------------------------------------------------------
# test_find_minimum_trial_count_excludes_omega3
# ---------------------------------------------------------------------------


def test_find_minimum_trial_count_excludes_omega3() -> None:
    """omega_3 is excluded from the all-must-pass requirement by default.

    Setup: omega_3 never reaches 0.7; all other params pass at trial_count=200.
    Expected: 200 (omega_3 is excluded from the check).
    """
    sweep = [
        _make_sweep_point(
            150,
            ["omega_2", "omega_3", "beta"],
            [0.60, 0.50, 0.65],
        ),
        _make_sweep_point(
            200,
            ["omega_2", "omega_3", "beta"],
            [0.75, 0.55, 0.72],
        ),
        _make_sweep_point(
            420,
            ["omega_2", "omega_3", "beta"],
            [0.85, 0.62, 0.80],
        ),
    ]
    result = find_minimum_trial_count(sweep)
    assert result == 200, (
        f"Expected 200 (omega_3 excluded from requirement), got {result}"
    )


# ---------------------------------------------------------------------------
# test_find_minimum_trial_count_custom_eligible
# ---------------------------------------------------------------------------


def test_find_minimum_trial_count_custom_eligible() -> None:
    """Custom eligible_params restricts the check to the specified parameters.

    Setup: omega_2 passes at 200; beta fails at 200 but passes at 300.
    With eligible_params=["omega_2"], should return 200 (ignoring beta).
    """
    sweep = [
        _make_sweep_point(150, ["omega_2", "beta"], [0.60, 0.55]),
        _make_sweep_point(200, ["omega_2", "beta"], [0.75, 0.65]),
        _make_sweep_point(300, ["omega_2", "beta"], [0.80, 0.72]),
    ]
    result = find_minimum_trial_count(sweep, eligible_params=["omega_2"])
    assert result == 200, (
        f"Expected 200 with eligible_params=['omega_2'], got {result}"
    )


# ---------------------------------------------------------------------------
# test_plot_trial_sweep_creates_figure
# ---------------------------------------------------------------------------


def test_plot_trial_sweep_creates_figure() -> None:
    """plot_trial_sweep returns a matplotlib Figure with the correct number of lines.

    A figure with 2 parameters + 1 reference line should have 3 Line2D objects.
    """
    sweep = [
        _make_sweep_point(150, ["omega_2", "beta"], [0.60, 0.65]),
        _make_sweep_point(250, ["omega_2", "beta"], [0.75, 0.72]),
        _make_sweep_point(420, ["omega_2", "beta"], [0.85, 0.80]),
    ]
    fig = plot_trial_sweep(sweep)
    try:
        assert isinstance(fig, plt.Figure), (
            f"Expected plt.Figure, got {type(fig)}"
        )
        ax = fig.axes[0]
        # 2 parameter lines + 1 reference line = 3 lines
        n_lines = len(ax.get_lines())
        assert n_lines == 3, (
            f"Expected 3 lines (2 params + 1 reference), got {n_lines}"
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# test_plot_trial_sweep_reference_line
# ---------------------------------------------------------------------------


def test_plot_trial_sweep_reference_line() -> None:
    """plot_trial_sweep includes a horizontal reference line at y=0.7.

    Checks ax.lines for a line with constant y-data equal to 0.7.
    """
    sweep = [
        _make_sweep_point(150, ["omega_2"], [0.60]),
        _make_sweep_point(420, ["omega_2"], [0.85]),
    ]
    fig = plot_trial_sweep(sweep, r_threshold=0.7)
    try:
        ax = fig.axes[0]
        ref_lines = [
            line
            for line in ax.get_lines()
            if len(set(line.get_ydata())) == 1
            and abs(float(line.get_ydata()[0]) - 0.7) < 1e-9
        ]
        assert len(ref_lines) >= 1, (
            "Expected at least one horizontal line at y=0.7 (reference line)"
        )
    finally:
        plt.close(fig)
