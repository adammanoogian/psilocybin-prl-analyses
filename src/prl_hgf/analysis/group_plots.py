"""Visualization helpers for group-level HGF parameter analysis.

Provides raincloud plots (distribution + box + scatter overlay) and
interaction plots (group × session trajectory) for HGF parameters.

Matplotlib uses the non-interactive Agg backend so plots can be generated
in headless environments (CI, HPC clusters, remote servers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import ptitprince  # noqa: E402
import seaborn as sns  # noqa: E402

__all__ = ["plot_raincloud", "plot_interaction", "plot_all_rainclouds"]


def plot_raincloud(
    estimates_wide: pd.DataFrame,
    outcome: str,
    palette: str = "colorblind",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a raincloud showing parameter distribution by group × session.

    A raincloud combines a kernel density estimate (cloud), individual data
    points (rain), and a boxplot (umbrella) to give a full distributional
    picture of each group × session combination.

    Parameters
    ----------
    estimates_wide : pd.DataFrame
        Wide-form estimates from
        :func:`~prl_hgf.analysis.group.build_estimates_wide`.  Must contain
        columns ``session``, ``group``, and *outcome*.
    outcome : str
        Column name of the HGF parameter to plot (e.g. ``"omega_2"``).
    palette : str, optional
        Seaborn palette name for group hue colouring. Default
        ``"colorblind"``.
    save_path : Path or None, optional
        If provided, the figure is saved as a PNG at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ptitprince.RainCloud(
        x="session",
        y=outcome,
        hue="group",
        data=estimates_wide,
        bw=0.2,
        width_viol=0.6,
        orient="h",
        alpha=0.65,
        dodge=True,
        pointplot=False,
        move=0.2,
        palette=palette,
        ax=ax,
    )

    ax.set_title(f"Parameter distribution by group × session: {outcome}", fontsize=11)
    ax.set_xlabel(outcome)
    ax.set_ylabel("Session")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_interaction(
    estimates_wide: pd.DataFrame,
    outcome: str,
    palette: str = "colorblind",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot group × session interaction trajectories for one parameter.

    Uses a seaborn point plot to show the mean trajectory of each group
    across sessions, allowing visual inspection of group × session
    interactions.

    Parameters
    ----------
    estimates_wide : pd.DataFrame
        Wide-form estimates from
        :func:`~prl_hgf.analysis.group.build_estimates_wide`.  Must contain
        columns ``session``, ``group``, and *outcome*.
    outcome : str
        Column name of the HGF parameter to plot (e.g. ``"omega_2"``).
    palette : str, optional
        Seaborn palette name for group hue colouring. Default
        ``"colorblind"``.
    save_path : Path or None, optional
        If provided, the figure is saved as a PNG at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.pointplot(
        data=estimates_wide,
        x="session",
        y=outcome,
        hue="group",
        dodge=0.1,
        markers=["o", "s"],
        linestyles=["-", "--"],
        palette=palette,
        ax=ax,
    )

    ax.set_title(f"Group × session interaction: {outcome}", fontsize=11)
    ax.set_xlabel("Session")
    ax.set_ylabel(outcome)
    ax.legend(title="Group", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_all_rainclouds(
    estimates_wide: pd.DataFrame,
    params: list[str],
    output_dir: Path,
    palette: str = "colorblind",
) -> list[plt.Figure]:
    """Generate and save raincloud plots for a list of parameters.

    Each parameter gets its own figure, saved as
    ``{output_dir}/raincloud_{param}.png`` and then closed to free memory.

    Parameters
    ----------
    estimates_wide : pd.DataFrame
        Wide-form estimates from
        :func:`~prl_hgf.analysis.group.build_estimates_wide`.
    params : list[str]
        HGF parameter column names to plot (e.g.
        ``["omega_2", "kappa", "beta", "zeta"]``).
    output_dir : Path
        Directory in which to save PNG files.  Created if it does not exist.
    palette : str, optional
        Seaborn palette name for group hue colouring. Default
        ``"colorblind"``.

    Returns
    -------
    list[matplotlib.figure.Figure]
        List of figure objects (already saved and closed; kept for testing
        convenience).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[plt.Figure] = []
    for param in params:
        if param not in estimates_wide.columns:
            continue
        save_path = output_dir / f"raincloud_{param}.png"
        fig = plot_raincloud(
            estimates_wide,
            outcome=param,
            palette=palette,
            save_path=save_path,
        )
        figures.append(fig)
        plt.close(fig)

    return figures
