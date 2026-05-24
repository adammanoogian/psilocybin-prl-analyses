"""Matplotlib-based HGF model schematic for Jupyter notebooks.

Renders an annotated graphical model showing the hierarchical structure,
parameter locations, and coupling types.  Color palette matches the JSX
model explorer in ``reports/figures/hgf_model_explorer.jsx``.

Designed to sit above the ParamExplorer's behavioral panels so users
can see *where* a parameter lives before adjusting *what* it does.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Color palette (from JSX model explorer)
_C = {
    "l3_fill": "#FAECE7",
    "l3_stroke": "#993C1D",
    "l2_fill": "#E1F5EE",
    "l2_stroke": "#0F6E56",
    "l1_fill": "#E6F1FB",
    "l1_stroke": "#185FA5",
    "resp_fill": "#EEEDFE",
    "resp_stroke": "#534AB7",
    "prm_fill": "#FFFBEF",
    "prm_stroke": "#9A7820",
    "e_val": "#185FA5",
    "e_vol": "#993C1D",
}


def plot_hgf_schematic(
    model_level: int = 3,
    task: str = "pick_best_cue",
    highlight_param: str | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    """Draw an annotated HGF model schematic.

    Parameters
    ----------
    model_level : int, default 3
        ``2`` for 2-level HGF, ``3`` for 3-level.
    task : str, default "pick_best_cue"
        ``"pick_best_cue"`` (3-branch) or ``"pat_rl"`` (1-branch).
    highlight_param : str or None
        If set, highlight the node/edge associated with this parameter
        (e.g. ``"omega_2"``, ``"kappa"``, ``"beta"``).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    hl = highlight_param
    n_branches = 3 if task == "pick_best_cue" else 1
    cue_labels = ["A", "B", "C"] if n_branches == 3 else [""]
    branch_xs = [2.5, 5.0, 7.5] if n_branches == 3 else [5.0]

    # ── Level 3: volatility node ─────────────────────────────────
    if model_level == 3:
        vol_x, vol_y = 5.0, 7.0
        _draw_node(
            ax,
            vol_x,
            vol_y,
            "x₃\nvolatility",
            _C["l3_fill"],
            _C["l3_stroke"],
            shape="hexagon",
            highlight=hl in ("omega_3", "kappa"),
        )

        # omega_3 label
        _draw_param_box(ax, vol_x + 2.2, vol_y, "ω₃", highlight=hl == "omega_3")

        # Volatility coupling arrows (x3 → each belief node)
        for bx in branch_xs:
            _draw_arrow(
                ax,
                vol_x,
                vol_y - 0.6,
                bx,
                4.6,
                color=_C["e_vol"],
                style="dashed",
                highlight=hl == "kappa",
            )

        # kappa label (on middle arrow)
        mid_x = 5.0 if n_branches == 3 else 5.0
        ax.text(
            mid_x + 0.3,
            5.8,
            "κ",
            fontsize=14,
            fontweight="bold" if hl == "kappa" else "normal",
            color=_C["l3_stroke"],
            ha="left",
        )

    # ── Level 2: belief nodes ────────────────────────────────────
    for i, (bx, label) in enumerate(zip(branch_xs, cue_labels, strict=False)):
        belief_y = 4.0
        node_label = f"x₂{label}\nbelief" if label else "x₂\nbelief"
        _draw_node(
            ax,
            bx,
            belief_y,
            node_label,
            _C["l2_fill"],
            _C["l2_stroke"],
            shape="circle",
            highlight=hl == "omega_2",
        )

        # omega_2 label (on first branch only to avoid clutter)
        if i == 0:
            _draw_param_box(
                ax, bx - 2.0, belief_y, "ω₂", highlight=hl == "omega_2"
            )

    # ── Level 1: input nodes ─────────────────────────────────────
    for bx, label in zip(branch_xs, cue_labels, strict=False):
        input_y = 1.0
        node_label = f"u{label}\ninput" if label else "u\ninput"
        _draw_node(
            ax,
            bx,
            input_y,
            node_label,
            _C["l1_fill"],
            _C["l1_stroke"],
            shape="square",
        )

        # Value coupling arrow (belief ↔ input)
        _draw_arrow(
            ax,
            bx,
            4.0 - 0.6,
            bx,
            1.6,
            color=_C["e_val"],
            style="solid",
        )

    # ── Response model ───────────────────────────────────────────
    resp_y = -0.5
    resp_box = FancyBboxPatch(
        (1.5, resp_y - 0.4),
        7.0,
        0.8,
        boxstyle="round,pad=0.15",
        facecolor=_C["resp_fill"],
        edgecolor=_C["resp_stroke"],
        linewidth=2 if hl in ("beta", "zeta") else 1,
        linestyle="--",
    )
    ax.add_patch(resp_box)
    ax.text(
        5.0,
        resp_y,
        "P(k) = softmax( β · μ₁ₖ + ζ · 𝟙[prev=k] )",
        fontsize=11,
        ha="center",
        va="center",
        color=_C["resp_stroke"],
        fontweight="bold" if hl in ("beta", "zeta") else "normal",
    )

    # beta and zeta labels
    _draw_param_box(ax, 9.0, resp_y + 0.5, "β", highlight=hl == "beta")
    _draw_param_box(ax, 9.0, resp_y - 0.5, "ζ", highlight=hl == "zeta")

    # Belief → response arrows
    for bx in branch_xs:
        _draw_arrow(
            ax,
            bx,
            1.0 - 0.6,
            bx,
            resp_y + 0.4,
            color=_C["resp_stroke"],
            style="dotted",
            highlight=hl in ("beta", "zeta"),
        )

    # ── Title and legend ─────────────────────────────────────────
    level_str = f"{model_level}-level"
    task_str = "3-cue PRL" if task == "pick_best_cue" else "PAT-RL"
    ax.set_title(
        f"{level_str} binary HGF — {task_str}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    legend_elements = [
        mpatches.Patch(facecolor=_C["l1_fill"], edgecolor=_C["l1_stroke"], label="Level 1 (input)"),
        mpatches.Patch(facecolor=_C["l2_fill"], edgecolor=_C["l2_stroke"], label="Level 2 (belief)"),
    ]
    if model_level == 3:
        legend_elements.append(
            mpatches.Patch(facecolor=_C["l3_fill"], edgecolor=_C["l3_stroke"], label="Level 3 (volatility)")
        )
    legend_elements.append(
        mpatches.Patch(facecolor=_C["resp_fill"], edgecolor=_C["resp_stroke"], label="Response model"),
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return fig


def _draw_node(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    fill: str,
    stroke: str,
    shape: str = "circle",
    highlight: bool = False,
) -> None:
    """Draw a single node (circle, hexagon, or square)."""
    lw = 3 if highlight else 1.5
    r = 0.55

    if shape == "hexagon":
        angles = np.linspace(0, 2 * np.pi, 7)
        xs = x + r * 1.2 * np.cos(angles)
        ys = y + r * 1.2 * np.sin(angles)
        ax.fill(xs, ys, color=fill)
        ax.plot(xs, ys, color=stroke, linewidth=lw)
    elif shape == "square":
        rect = FancyBboxPatch(
            (x - r, y - r),
            2 * r,
            2 * r,
            boxstyle="round,pad=0.05",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
        )
        ax.add_patch(rect)
    else:
        circle = plt.Circle((x, y), r, facecolor=fill, edgecolor=stroke, linewidth=lw)
        ax.add_patch(circle)

    ax.text(x, y, label, fontsize=9, ha="center", va="center", fontweight="bold")


def _draw_param_box(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    highlight: bool = False,
) -> None:
    """Draw a small parameter label box."""
    fill = "#FFF3CC" if highlight else _C["prm_fill"]
    stroke = "#CC8800" if highlight else _C["prm_stroke"]
    lw = 2.5 if highlight else 1

    rect = FancyBboxPatch(
        (x - 0.35, y - 0.25),
        0.7,
        0.5,
        boxstyle="round,pad=0.08",
        facecolor=fill,
        edgecolor=stroke,
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, fontsize=12, ha="center", va="center", color=stroke, fontweight="bold")


def _draw_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "black",
    style: str = "solid",
    highlight: bool = False,
) -> None:
    """Draw a coupling arrow between nodes."""
    ls = {"solid": "-", "dashed": "--", "dotted": ":"}[style]
    lw = 2.5 if highlight else 1.5
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        color=color,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(arrow)
