"""Unit tests for scripts/10_plot_power_curves.py plot functions.

All tests run without MCMC, JAX, or real CSV files.  Synthetic DataFrames
are constructed inline.  Uses ``importlib.util`` to load the script as a
module so the ``sys.path.insert`` and ``matplotlib.use("Agg")`` preamble
runs correctly.

Run with::

    pytest tests/test_power_plots.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Load plotting module via importlib
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "03_pre_analysis" / "06_plot_power_curves.py"
)

spec = importlib.util.spec_from_file_location("plot_power", str(_SCRIPT_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]

plot_precheck_recovery = mod.plot_precheck_recovery
plot_power_a = mod.plot_power_a
plot_power_b = mod.plot_power_b
plot_sensitivity_heatmap = mod.plot_sensitivity_heatmap
plot_combined_figure = mod.plot_combined_figure


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_master_df(
    n_levels: list[int] | None = None,
    effect_sizes: list[float] | None = None,
    n_iter: int = 4,
    bf_value_fn=None,
    bms_correct: bool = True,
) -> pd.DataFrame:
    """Build a minimal master_df with required columns.

    Parameters
    ----------
    n_levels : list[int] or None
        N per group levels. Default [20, 30].
    effect_sizes : list[float] or None
        Effect size levels. Default [0.3, 0.5].
    n_iter : int
        Iterations per cell. Default 4.
    bf_value_fn : callable or None
        Function (n, d, i) -> float for bf_value. Default constant 10.0.
    bms_correct : bool
        Constant value for bms_correct column. Default True.

    Returns
    -------
    pd.DataFrame
        Minimal valid master DataFrame.
    """
    if n_levels is None:
        n_levels = [20, 30]
    if effect_sizes is None:
        effect_sizes = [0.3, 0.5]
    if bf_value_fn is None:
        def bf_value_fn(n, d, i):  # noqa: E306
            return 10.0

    rows = []
    for n in n_levels:
        for d in effect_sizes:
            for it in range(n_iter):
                bfv = bf_value_fn(n, d, it)
                rows.append(
                    {
                        "sweep_type": "did_postdose",
                        "n_per_group": n,
                        "effect_size": d,
                        "iteration": it,
                        "bf_value": float(bfv),
                        "bf_exceeds": bfv >= 6.0,
                        "bms_correct": bms_correct,
                    }
                )
    return pd.DataFrame(rows)


def _make_power_a_df(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute power_a_df from master_df for use in tests.

    Parameters
    ----------
    master_df : pd.DataFrame
        Master sweep results.

    Returns
    -------
    pd.DataFrame
        Power A summary DataFrame.
    """
    from prl_hgf.power.curves import compute_power_a

    return compute_power_a(master_df)


def _make_power_b_df(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute power_b_df from master_df for use in tests.

    Parameters
    ----------
    master_df : pd.DataFrame
        Master sweep results.

    Returns
    -------
    pd.DataFrame
        Power B summary DataFrame.
    """
    from prl_hgf.power.curves import compute_power_b

    return compute_power_b(master_df)


def _make_precheck_df(
    trial_counts: list[int] | None = None,
    params: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal trial sweep CSV DataFrame.

    Parameters
    ----------
    trial_counts : list[int] or None
        Trial count grid. Default [150, 200, 250].
    params : list[str] or None
        Parameter names. Default ["omega_2", "kappa"].

    Returns
    -------
    pd.DataFrame
        Long-form trial sweep DataFrame with columns ``trial_count``,
        ``parameter``, ``r``.
    """
    if trial_counts is None:
        trial_counts = [150, 200, 250]
    if params is None:
        params = ["omega_2", "kappa"]

    rows = []
    for t in trial_counts:
        for p in params:
            rows.append({"trial_count": t, "parameter": p, "r": 0.75})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotPowerA:
    """Tests for plot_power_a."""

    def test_plot_power_a_returns_figure(self) -> None:
        """plot_power_a returns a matplotlib Figure object.

        Verifies the function returns a Figure, not None or another type.
        """
        master_df = _make_master_df()
        power_a_df = _make_power_a_df(master_df)

        fig = plot_power_a(power_a_df, master_df, bf_threshold=6.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_power_a_saves_png(self, tmp_path: Path) -> None:
        """plot_power_a saves a PNG file at the specified path.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        master_df = _make_master_df()
        power_a_df = _make_power_a_df(master_df)
        save_path = tmp_path / "power_a.png"

        fig = plot_power_a(power_a_df, master_df, bf_threshold=6.0, save_path=save_path)
        try:
            assert save_path.exists(), f"PNG not created at {save_path}"
            assert save_path.stat().st_size > 0
        finally:
            plt.close(fig)

    def test_plot_power_a_annotates_crossing(self, tmp_path: Path) -> None:
        """plot_power_a adds an annotation when d=0.5 crosses 80% power.

        Constructs a master_df where power increases from 0 to 1 across N
        levels so the 80% crossing is guaranteed.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        # Create bf_values that give p=0 at N=10 and p=1 at N=30
        def bf_fn(n, d, i):
            if n <= 10:
                return 1.0  # BF < 6 -> not exceeds
            return 20.0  # BF > 6 -> exceeds

        master_df = _make_master_df(
            n_levels=[10, 20, 30],
            effect_sizes=[0.5],
            n_iter=5,
            bf_value_fn=bf_fn,
        )
        power_a_df = _make_power_a_df(master_df)

        fig = plot_power_a(power_a_df, master_df, bf_threshold=6.0)
        try:
            # Annotation presence: there should be at least one annotation text
            # on the axes (the crossing annotation).
            ax = fig.axes[0]
            texts = ax.texts
            assert len(texts) >= 1, "Expected at least one annotation on Power A figure"
        finally:
            plt.close(fig)


class TestPlotPowerB:
    """Tests for plot_power_b."""

    def test_plot_power_b_returns_figure(self) -> None:
        """plot_power_b returns a matplotlib Figure object."""
        master_df = _make_master_df()
        power_b_df = _make_power_b_df(master_df)

        fig = plot_power_b(power_b_df)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_power_b_saves_png(self, tmp_path: Path) -> None:
        """plot_power_b saves a PNG file at the specified path.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        master_df = _make_master_df()
        power_b_df = _make_power_b_df(master_df)
        save_path = tmp_path / "power_b.png"

        fig = plot_power_b(power_b_df, save_path=save_path)
        try:
            assert save_path.exists(), f"PNG not created at {save_path}"
            assert save_path.stat().st_size > 0
        finally:
            plt.close(fig)


class TestPlotSensitivityHeatmap:
    """Tests for plot_sensitivity_heatmap."""

    def test_plot_sensitivity_heatmap_returns_figure(self) -> None:
        """plot_sensitivity_heatmap returns a matplotlib Figure object."""
        master_df = _make_master_df()

        fig = plot_sensitivity_heatmap(master_df, bf_threshold=6.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_sensitivity_heatmap_saves_png(self, tmp_path: Path) -> None:
        """plot_sensitivity_heatmap saves a PNG file at the specified path.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        master_df = _make_master_df()
        save_path = tmp_path / "heatmap.png"

        fig = plot_sensitivity_heatmap(master_df, bf_threshold=6.0, save_path=save_path)
        try:
            assert save_path.exists(), f"PNG not created at {save_path}"
        finally:
            plt.close(fig)

    def test_heatmap_cell_values_in_range(self) -> None:
        """Heatmap cell text values are percentages between 0 and 100.

        Verifies that all text annotations on the heatmap axes contain valid
        percentage strings (integers ending in '%').
        """
        master_df = _make_master_df(n_iter=10)

        fig = plot_sensitivity_heatmap(master_df, bf_threshold=6.0)
        try:
            ax = fig.axes[0]
            for txt in ax.texts:
                label = txt.get_text()
                # Expect format like "75%"
                assert label.endswith("%"), f"Unexpected cell label: {label!r}"
                pct_val = int(label.rstrip("%"))
                assert 0 <= pct_val <= 100, f"Cell percentage out of range: {pct_val}"
        finally:
            plt.close(fig)


class TestPlotCombinedFigure:
    """Tests for plot_combined_figure."""

    def test_plot_combined_saves_pdf_and_png(self, tmp_path: Path) -> None:
        """plot_combined_figure saves both PDF and PNG at save_path stem.

        The function receives a .pdf path and must produce both .pdf and .png.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        master_df = _make_master_df()
        power_a_df = _make_power_a_df(master_df)
        power_b_df = _make_power_b_df(master_df)
        precheck_df = _make_precheck_df()

        save_path = tmp_path / "fig_combined.pdf"

        fig = plot_combined_figure(
            master_df=master_df,
            power_a_df=power_a_df,
            power_b_df=power_b_df,
            precheck_sweep_df=precheck_df,
            bf_threshold=6.0,
            save_path=save_path,
        )
        try:
            pdf_path = tmp_path / "fig_combined.pdf"
            png_path = tmp_path / "fig_combined.png"
            assert pdf_path.exists(), "PDF not saved"
            assert pdf_path.stat().st_size > 0, "PDF is empty"
            assert png_path.exists(), "PNG not saved"
            assert png_path.stat().st_size > 0, "PNG is empty"
        finally:
            plt.close(fig)

    def test_plot_combined_returns_figure(self) -> None:
        """plot_combined_figure returns a matplotlib Figure object."""
        master_df = _make_master_df()
        power_a_df = _make_power_a_df(master_df)
        power_b_df = _make_power_b_df(master_df)
        precheck_df = _make_precheck_df()

        fig = plot_combined_figure(
            master_df=master_df,
            power_a_df=power_a_df,
            power_b_df=power_b_df,
            precheck_sweep_df=precheck_df,
            bf_threshold=6.0,
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_combined_has_four_axes(self) -> None:
        """plot_combined_figure produces a figure with exactly 4 axes (2x2 panel).

        The combined figure must have one axes per panel (A, B, C, D).
        """
        master_df = _make_master_df()
        power_a_df = _make_power_a_df(master_df)
        power_b_df = _make_power_b_df(master_df)
        precheck_df = _make_precheck_df()

        fig = plot_combined_figure(
            master_df=master_df,
            power_a_df=power_a_df,
            power_b_df=power_b_df,
            precheck_sweep_df=precheck_df,
            bf_threshold=6.0,
        )
        try:
            # 2x2 layout = 4 axes; colorbar may add extra axes, so check >= 4
            assert len(fig.axes) >= 4, (
                f"Expected at least 4 axes in combined figure, got {len(fig.axes)}"
            )
        finally:
            plt.close(fig)


class TestPlotPrecheckRecovery:
    """Tests for plot_precheck_recovery."""

    def test_plot_precheck_recovery_returns_figure(self) -> None:
        """plot_precheck_recovery returns a matplotlib Figure object."""
        precheck_df = _make_precheck_df()

        fig = plot_precheck_recovery(precheck_df)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_precheck_recovery_empty_df(self) -> None:
        """plot_precheck_recovery handles empty DataFrame without error."""
        empty_df = pd.DataFrame(columns=["trial_count", "parameter", "r"])

        fig = plot_precheck_recovery(empty_df)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_precheck_saves_png(self, tmp_path: Path) -> None:
        """plot_precheck_recovery saves a PNG file at the specified path.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory.
        """
        precheck_df = _make_precheck_df()
        save_path = tmp_path / "precheck.png"

        fig = plot_precheck_recovery(precheck_df, save_path=save_path)
        try:
            assert save_path.exists(), f"PNG not created at {save_path}"
            assert save_path.stat().st_size > 0
        finally:
            plt.close(fig)
