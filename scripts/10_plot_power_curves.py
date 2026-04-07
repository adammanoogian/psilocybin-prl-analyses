#!/usr/bin/env python
"""Generate publication-quality power analysis figures from aggregated results.

Produces four figures:

- VIZ-01 (precheck): Parameter recovery r vs trial count from Phase 9 sweep.
- VIZ-02 (Power A):  P(BF > threshold) vs N per group, one curve per effect
  size, with 80% and 90% reference lines and an annotation marking N where
  d=0.5 crosses 80%.
- VIZ-03 (Power B):  P(correct BMS) vs N per group with a 75% reference line.
- VIZ-04 (combined): 2x2 panel combining all three plots plus a sensitivity
  heatmap.  Saved as both PDF and PNG.

Run after ``09_aggregate_power.py`` has produced the summary CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work on cluster
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as _cfg  # noqa: E402
from prl_hgf.power.curves import (  # noqa: E402
    compute_power_a,
    compute_power_b,
)

# ---------------------------------------------------------------------------
# Parameter-specific styles (replicated from precheck.py — private constants)
# ---------------------------------------------------------------------------

_PARAM_COLORS: dict[str, str] = {
    "omega_2": "#1f77b4",
    "omega_3": "#ff7f0e",
    "kappa": "#2ca02c",
    "beta": "#9467bd",
    "zeta": "#d62728",
}
_PARAM_MARKERS: dict[str, str] = {
    "omega_2": "o",
    "omega_3": "s",
    "kappa": "^",
    "beta": "D",
    "zeta": "v",
}

# Effect-size line styles for Power A (distinguishable without color alone)
_EFFECT_LINESTYLES: list[str] = ["-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# VIZ-01: Precheck recovery figure
# ---------------------------------------------------------------------------


def plot_precheck_recovery(
    precheck_sweep_df: pd.DataFrame,
    r_threshold: float = 0.7,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot parameter recovery r vs total trial count from Phase 9 sweep.

    One line per parameter, omega_3 rendered dashed and labelled
    "(exploratory)".  A horizontal dashed red reference line marks
    ``r_threshold``.

    Parameters
    ----------
    precheck_sweep_df : pd.DataFrame
        Long-form CSV produced by :func:`~prl_hgf.power.precheck.run_trial_sweep`
        with columns ``trial_count``, ``parameter``, ``r``.
    r_threshold : float, optional
        Reference line y-position. Default 0.7.
    save_path : Path or None, optional
        If provided, saves the figure as PNG at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    df = precheck_sweep_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Parameter recovery vs trial count (no data)")
        return fig

    params = df["parameter"].unique().tolist()
    trial_counts = sorted(df["trial_count"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for param in params:
        sub = df[df["parameter"] == param].set_index("trial_count")
        r_vals = [sub.loc[t, "r"] if t in sub.index else float("nan") for t in trial_counts]

        color = _PARAM_COLORS.get(param)
        marker = _PARAM_MARKERS.get(param, "o")
        linestyle = "--" if param == "omega_3" else "-"
        label = f"{param} (exploratory)" if param == "omega_3" else param

        ax.plot(
            trial_counts,
            r_vals,
            linestyle=linestyle,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(
        y=r_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"r = {r_threshold} (threshold)",
    )

    ax.set_xlabel("Total trials per session")
    ax.set_ylabel("Pearson r (true vs recovered)")
    ax.set_title("PRE-04/05: Parameter recovery vs trial count")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# VIZ-02: Power A figure
# ---------------------------------------------------------------------------


def plot_power_a(
    power_a_df: pd.DataFrame,
    master_df: pd.DataFrame,
    bf_threshold: float = 6.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot P(BF > bf_threshold) vs N per group for Power Analysis A (VIZ-02).

    Recomputes empirical power from ``master_df`` (raw ``bf_value`` column)
    for ``sweep_type == "did_postdose"``, producing one curve per effect size.
    Reference lines are drawn at y=0.80 and y=0.90.  The N value where
    d=0.5 crosses 80% power is annotated (interpolated if no exact crossing).

    Parameters
    ----------
    power_a_df : pd.DataFrame
        Precomputed summary from ``power_a_summary.csv`` (used for N levels
        and effect size levels; actual power is recomputed from master_df).
    master_df : pd.DataFrame
        Full concatenated sweep results with columns ``sweep_type``,
        ``n_per_group``, ``effect_size``, ``bf_value``.
    bf_threshold : float, optional
        Bayes factor threshold. Default 6.0.
    save_path : Path or None, optional
        If provided, saves the figure as PNG at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    # Recompute P(BF > threshold) from raw bf_value for did_postdose
    subset = master_df[master_df["sweep_type"] == "did_postdose"].copy()
    subset = subset.assign(bf_exceeds_thresh=subset["bf_value"] >= bf_threshold)

    grouped = (
        subset.groupby(["n_per_group", "effect_size"])["bf_exceeds_thresh"]
        .mean()
        .reset_index()
        .rename(columns={"bf_exceeds_thresh": "p_bf_exceeds"})
    )

    effect_sizes = sorted(grouped["effect_size"].unique())
    n_levels = sorted(grouped["n_per_group"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, d in enumerate(effect_sizes):
        sub = grouped[grouped["effect_size"] == d].sort_values("n_per_group")
        ls = _EFFECT_LINESTYLES[i % len(_EFFECT_LINESTYLES)]
        ax.plot(
            sub["n_per_group"],
            sub["p_bf_exceeds"],
            linestyle=ls,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=f"d = {d:.2f}",
        )

    # Reference lines
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1.2, label="80% power")
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1.2, label="90% power")

    # Annotate N where d=0.5 crosses 80%
    _annotate_crossing(ax, grouped, effect_size=0.5, power_level=0.80, n_levels=n_levels)

    ax.set_xlabel("N per group")
    ax.set_ylabel(f"P(BF > {bf_threshold:.0f})")
    ax.set_title(f"Power A: P(BF > {bf_threshold:.0f}) vs N per group")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _annotate_crossing(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    effect_size: float,
    power_level: float,
    n_levels: list[int | float],
) -> None:
    """Annotate the N where a given effect size crosses a power level.

    Uses linear interpolation between adjacent grid points when no exact
    crossing is available.  If the curve never reaches ``power_level`` no
    annotation is drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    grouped : pd.DataFrame
        Grouped power data with columns ``effect_size``, ``n_per_group``,
        ``p_bf_exceeds``.
    effect_size : float
        Effect size to annotate (e.g. 0.5).
    power_level : float
        Power threshold to annotate (e.g. 0.80).
    n_levels : list[int | float]
        Sorted list of N per group levels.
    """
    sub = grouped[grouped["effect_size"] == effect_size].sort_values("n_per_group")
    if sub.empty:
        return

    ns = sub["n_per_group"].tolist()
    ps = sub["p_bf_exceeds"].tolist()

    # Find interpolated crossing
    n_cross: float | None = None
    for j in range(len(ps) - 1):
        if ps[j] <= power_level <= ps[j + 1]:
            # Linear interpolation
            denom = ps[j + 1] - ps[j]
            if abs(denom) < 1e-9:
                n_cross = float(ns[j])
            else:
                frac = (power_level - ps[j]) / denom
                n_cross = ns[j] + frac * (ns[j + 1] - ns[j])
            break
        elif ps[j] >= power_level:
            # Already above threshold at first point
            n_cross = float(ns[j])
            break

    if n_cross is None:
        return

    n_cross_int = int(round(n_cross))

    ax.axvline(
        x=n_cross,
        color="steelblue",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.annotate(
        f"N={n_cross_int}\n(d={effect_size:.1f}, {int(power_level * 100)}%)",
        xy=(n_cross, power_level),
        xytext=(n_cross + (max(n_levels) - min(n_levels)) * 0.04, power_level - 0.08),
        fontsize=8,
        color="steelblue",
        arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 1.0},
    )


# ---------------------------------------------------------------------------
# VIZ-03: Power B figure
# ---------------------------------------------------------------------------


def plot_power_b(
    power_b_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot P(correct BMS) vs N per group for Power Analysis B (VIZ-03).

    Single line with markers.  A horizontal dashed reference line marks
    75% discriminability.

    Parameters
    ----------
    power_b_df : pd.DataFrame
        Summary DataFrame from ``power_b_summary.csv`` with columns
        ``n_per_group`` and ``p_bms_correct``.
    save_path : Path or None, optional
        If provided, saves the figure as PNG at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    df = power_b_df.sort_values("n_per_group")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df["n_per_group"],
        df["p_bms_correct"],
        linestyle="-",
        marker="s",
        color="#2ca02c",
        linewidth=1.8,
        markersize=6,
        label="P(correct BMS)",
    )

    ax.axhline(
        0.75,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="75% discriminability",
    )

    ax.set_xlabel("N per group")
    ax.set_ylabel("P(correct BMS)")
    ax.set_title("Power B: BMS discriminability vs N per group")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------


def plot_sensitivity_heatmap(
    master_df: pd.DataFrame,
    bf_threshold: float = 6.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a sensitivity heatmap of P(BF > threshold) across N x d grid.

    Recomputes power from raw ``bf_value`` for ``sweep_type == "did_postdose"``,
    pivots to a matrix of N (rows) by effect size (columns), and renders as an
    imshow with "RdYlGn" colormap.  Cell percentages are annotated.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full concatenated sweep results with columns ``sweep_type``,
        ``n_per_group``, ``effect_size``, ``bf_value``.
    bf_threshold : float, optional
        Bayes factor threshold for P(BF > threshold). Default 6.0.
    save_path : Path or None, optional
        If provided, saves the figure as PNG at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    subset = master_df[master_df["sweep_type"] == "did_postdose"].copy()
    subset = subset.assign(bf_exceeds_thresh=subset["bf_value"] >= bf_threshold)

    pivot = (
        subset.groupby(["n_per_group", "effect_size"])["bf_exceeds_thresh"]
        .mean()
        .unstack("effect_size")
    )

    n_rows, n_cols = pivot.shape
    fig_width = max(6, 1.5 * n_cols + 2)
    fig_height = max(4, 0.7 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    mat = pivot.values
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.2 < val < 0.8 else "white"
                ax.text(
                    j,
                    i,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{d:.2f}" for d in pivot.columns], fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(n) for n in pivot.index], fontsize=8)

    ax.set_xlabel("Effect size (d)")
    ax.set_ylabel("N per group")
    ax.set_title(f"Sensitivity: P(BF > {bf_threshold:.0f}) by N and effect size")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"P(BF > {bf_threshold:.0f})", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# VIZ-04: 4-panel combined figure
# ---------------------------------------------------------------------------


def plot_combined_figure(
    master_df: pd.DataFrame,
    power_a_df: pd.DataFrame,
    power_b_df: pd.DataFrame,
    precheck_sweep_df: pd.DataFrame,
    bf_threshold: float = 6.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Produce the 4-panel publication figure combining all power analyses (VIZ-04).

    Panels:

    - A (top-left):  Precheck recovery r vs trial count.
    - B (top-right): Power A — P(BF > threshold) vs N.
    - C (bottom-left): Power B — P(correct BMS) vs N.
    - D (bottom-right): Sensitivity heatmap.

    Saved as both PDF and PNG using ``save_path`` stem when ``save_path`` is
    provided.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full concatenated sweep results.
    power_a_df : pd.DataFrame
        Precomputed Power A summary (used to obtain N/effect-size grid levels).
    power_b_df : pd.DataFrame
        Precomputed Power B summary.
    precheck_sweep_df : pd.DataFrame
        Trial sweep results from Phase 9 (``trial_sweep_results.csv``).
    bf_threshold : float, optional
        Bayes factor threshold. Default 6.0.
    save_path : Path or None, optional
        Base path (with any extension).  Figure is saved as both
        ``{stem}.pdf`` and ``{stem}.png`` at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed 4-panel figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Panel A: Precheck recovery ---
    _draw_precheck_panel(ax_a, precheck_sweep_df, r_threshold=0.7)

    # --- Panel B: Power A ---
    _draw_power_a_panel(ax_b, master_df, power_a_df, bf_threshold)

    # --- Panel C: Power B ---
    _draw_power_b_panel(ax_c, power_b_df)

    # --- Panel D: Sensitivity heatmap ---
    _draw_heatmap_panel(ax_d, fig, master_df, bf_threshold)

    # Panel labels
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"], strict=True):
        ax.text(
            -0.12,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout()

    if save_path is not None:
        stem = Path(save_path).with_suffix("")
        pdf_path = stem.with_suffix(".pdf")
        png_path = stem.with_suffix(".png")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Panel drawing helpers (draw into existing Axes — no file I/O)
# ---------------------------------------------------------------------------


def _draw_precheck_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    r_threshold: float = 0.7,
) -> None:
    """Draw the precheck recovery panel into an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pd.DataFrame
        Long-form trial sweep CSV with columns ``trial_count``, ``parameter``,
        ``r``.
    r_threshold : float, optional
        Reference line y-position. Default 0.7.
    """
    if df.empty:
        ax.set_title("Recovery vs trial count (no data)")
        return

    params = df["parameter"].unique().tolist()
    trial_counts = sorted(df["trial_count"].unique())

    for param in params:
        sub = df[df["parameter"] == param].set_index("trial_count")
        r_vals = [sub.loc[t, "r"] if t in sub.index else float("nan") for t in trial_counts]

        color = _PARAM_COLORS.get(param)
        marker = _PARAM_MARKERS.get(param, "o")
        linestyle = "--" if param == "omega_3" else "-"
        label = f"{param} (exp.)" if param == "omega_3" else param

        ax.plot(
            trial_counts,
            r_vals,
            linestyle=linestyle,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=5,
        )

    ax.axhline(
        y=r_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"r={r_threshold}",
    )

    ax.set_xlabel("Total trials per session", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_title("Recovery vs trial count", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))


def _draw_power_a_panel(
    ax: plt.Axes,
    master_df: pd.DataFrame,
    power_a_df: pd.DataFrame,
    bf_threshold: float,
) -> None:
    """Draw the Power A panel into an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    master_df : pd.DataFrame
        Full concatenated sweep results.
    power_a_df : pd.DataFrame
        Precomputed Power A summary (for grid levels).
    bf_threshold : float
        Bayes factor threshold.
    """
    subset = master_df[master_df["sweep_type"] == "did_postdose"].copy()
    subset = subset.assign(bf_exceeds_thresh=subset["bf_value"] >= bf_threshold)

    grouped = (
        subset.groupby(["n_per_group", "effect_size"])["bf_exceeds_thresh"]
        .mean()
        .reset_index()
        .rename(columns={"bf_exceeds_thresh": "p_bf_exceeds"})
    )

    effect_sizes = sorted(grouped["effect_size"].unique())
    n_levels = sorted(grouped["n_per_group"].unique())

    for i, d in enumerate(effect_sizes):
        sub = grouped[grouped["effect_size"] == d].sort_values("n_per_group")
        ls = _EFFECT_LINESTYLES[i % len(_EFFECT_LINESTYLES)]
        ax.plot(
            sub["n_per_group"],
            sub["p_bf_exceeds"],
            linestyle=ls,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"d={d:.2f}",
        )

    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1.0, label="80%")
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1.0, label="90%")
    _annotate_crossing(ax, grouped, effect_size=0.5, power_level=0.80, n_levels=n_levels)

    ax.set_xlabel("N per group", fontsize=9)
    ax.set_ylabel(f"P(BF > {bf_threshold:.0f})", fontsize=9)
    ax.set_title("Power A: BFDA power curves", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7)


def _draw_power_b_panel(ax: plt.Axes, power_b_df: pd.DataFrame) -> None:
    """Draw the Power B panel into an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    power_b_df : pd.DataFrame
        Summary DataFrame with columns ``n_per_group`` and ``p_bms_correct``.
    """
    df = power_b_df.sort_values("n_per_group")
    ax.plot(
        df["n_per_group"],
        df["p_bms_correct"],
        linestyle="-",
        marker="s",
        color="#2ca02c",
        linewidth=1.5,
        markersize=5,
        label="P(correct BMS)",
    )
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1.0, label="75%")

    ax.set_xlabel("N per group", fontsize=9)
    ax.set_ylabel("P(correct BMS)", fontsize=9)
    ax.set_title("Power B: BMS discriminability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7)


def _draw_heatmap_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    master_df: pd.DataFrame,
    bf_threshold: float,
) -> None:
    """Draw the sensitivity heatmap panel into an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    fig : matplotlib.figure.Figure
        Parent figure (needed for colorbar placement).
    master_df : pd.DataFrame
        Full concatenated sweep results.
    bf_threshold : float
        Bayes factor threshold.
    """
    subset = master_df[master_df["sweep_type"] == "did_postdose"].copy()
    subset = subset.assign(bf_exceeds_thresh=subset["bf_value"] >= bf_threshold)

    pivot = (
        subset.groupby(["n_per_group", "effect_size"])["bf_exceeds_thresh"]
        .mean()
        .unstack("effect_size")
    )

    n_rows, n_cols = pivot.shape
    mat = pivot.values
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.2 < val < 0.8 else "white"
                ax.text(
                    j,
                    i,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{d:.2f}" for d in pivot.columns], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(n) for n in pivot.index], fontsize=7)

    ax.set_xlabel("Effect size (d)", fontsize=9)
    ax.set_ylabel("N per group", fontsize=9)
    ax.set_title("Sensitivity heatmap", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"P(BF>{bf_threshold:.0f})", fontsize=8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: input_dir, precheck_dir, output_dir, bf_threshold.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality power analysis figures. "
            "Reads summary CSVs from --input-dir and precheck CSV from "
            "--precheck-dir. Writes figures to --output-dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_cfg.RESULTS_DIR / "power",
        help="Directory containing power_master.csv, power_a_summary.csv, power_b_summary.csv.",
    )
    parser.add_argument(
        "--precheck-dir",
        type=Path,
        default=_cfg.RESULTS_DIR / "power",
        help="Directory containing trial_sweep_results.csv from Phase 9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_cfg.RESULTS_DIR / "power",
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--bf-threshold",
        type=float,
        default=6.0,
        help="Bayes factor threshold for power curves.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all four power analysis figures.

    Reads ``power_master.csv``, ``power_a_summary.csv``,
    ``power_b_summary.csv`` from ``--input-dir`` and
    ``trial_sweep_results.csv`` from ``--precheck-dir``.  Writes individual
    PNG figures and a combined PDF + PNG publication figure to ``--output-dir``.

    Raises
    ------
    SystemExit
        On argument parse error (via argparse).
    """
    args = parse_args()

    input_dir: Path = args.input_dir
    precheck_dir: Path = args.precheck_dir
    output_dir: Path = args.output_dir
    bf_threshold: float = args.bf_threshold

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {input_dir}")
    master_df = pd.read_csv(input_dir / "power_master.csv")

    # Use compute_power_a / compute_power_b to derive summaries from master data
    # so figures reflect the same threshold used by this script, not a baked-in value.
    power_a_df = compute_power_a(master_df)
    power_b_df = compute_power_b(master_df)

    # Fall back to pre-computed CSVs for grid-level metadata if needed.
    if not (input_dir / "power_a_summary.csv").exists():
        print("power_a_summary.csv not found — using freshly computed summary.")
    else:
        _ = pd.read_csv(input_dir / "power_a_summary.csv")  # validate readable

    if not (input_dir / "power_b_summary.csv").exists():
        print("power_b_summary.csv not found — using freshly computed summary.")
    else:
        _ = pd.read_csv(input_dir / "power_b_summary.csv")  # validate readable

    precheck_csv = precheck_dir / "trial_sweep_results.csv"
    if precheck_csv.exists():
        precheck_sweep_df = pd.read_csv(precheck_csv)
        print(f"Loaded precheck sweep: {len(precheck_sweep_df)} rows")
    else:
        print(f"Warning: {precheck_csv} not found — Panel A will be empty.")
        precheck_sweep_df = pd.DataFrame(
            columns=["trial_count", "parameter", "r"]
        )

    print(f"BF threshold: {bf_threshold}")

    # VIZ-01: Precheck recovery
    precheck_path = output_dir / "fig_precheck_recovery.png"
    fig_precheck = plot_precheck_recovery(
        precheck_sweep_df, save_path=precheck_path
    )
    plt.close(fig_precheck)
    print(f"Saved: {precheck_path}")

    # VIZ-02: Power A
    power_a_path = output_dir / "fig_power_a.png"
    fig_a = plot_power_a(power_a_df, master_df, bf_threshold=bf_threshold, save_path=power_a_path)
    plt.close(fig_a)
    print(f"Saved: {power_a_path}")

    # VIZ-03: Power B
    power_b_path = output_dir / "fig_power_b.png"
    fig_b = plot_power_b(power_b_df, save_path=power_b_path)
    plt.close(fig_b)
    print(f"Saved: {power_b_path}")

    # VIZ-04 (individual): Sensitivity heatmap
    heatmap_path = output_dir / "fig_sensitivity_heatmap.png"
    fig_hmap = plot_sensitivity_heatmap(master_df, bf_threshold=bf_threshold, save_path=heatmap_path)
    plt.close(fig_hmap)
    print(f"Saved: {heatmap_path}")

    # VIZ-04: Combined 4-panel publication figure (saves PDF + PNG)
    combined_stem = output_dir / "fig_combined_power"
    fig_combined = plot_combined_figure(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=precheck_sweep_df,
        bf_threshold=bf_threshold,
        save_path=combined_stem.with_suffix(".pdf"),
    )
    plt.close(fig_combined)
    print(f"Saved: {combined_stem}.pdf and {combined_stem}.png")


if __name__ == "__main__":
    main()
