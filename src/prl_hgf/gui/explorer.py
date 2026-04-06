"""Interactive HGF parameter explorer for the PRL pick_best_cue pipeline.

Provides :class:`ParamExplorer`, a self-contained ipywidgets + matplotlib
widget that lets users manipulate HGF model parameters via sliders and
immediately see updated belief trajectories, volatility estimates, choice
probabilities, and effective learning rates.

Architecture
------------
* **Batch forward pass** (``net.input_data`` with all 420 trials at once)
  instead of the trial-by-trial ``simulate_agent`` loop — ~1.1 s (2-level)
  or ~1.7 s (3-level) vs ~4.2 s trial-by-trial.
* **Belief caching**: HGF parameters (omega_2, omega_3, kappa, mu_1_0,
  mu_3_0) trigger a full forward pass; response parameters (beta, zeta)
  only recompute the softmax from cached beliefs (<1 ms).
* **``continuous_update=False``** on all sliders: callbacks fire only on
  release, preventing intermediate updates during dragging.
* **Unobserve/observe pattern** for preset loading: bulk-sets all slider
  values without firing N intermediate callbacks.

Notes
-----
The observation mask (choices + rewards) is fixed from a one-time warm-up
simulation using ``GUI_SEED``. The belief trajectories shown represent one
specific simulated agent session, not a real participant. This is
intentional: the GUI is a parameter exploration tool, not a data viewer.
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import load_config
from prl_hgf.models.hgf_2level import (
    BELIEF_NODES,
    INPUT_NODES,
    build_2level_network,
    extract_beliefs,
    prepare_input_data,
)
from prl_hgf.models.hgf_3level import (
    VOLATILITY_NODE,
    build_3level_network,
    extract_beliefs_3level,
)
from prl_hgf.simulation.agent import PARAM_BOUNDS, simulate_agent

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Fixed seed for warm-up simulation reproducibility.
GUI_SEED: int = 0

#: Parameters that require a full HGF forward pass when changed.
_HGF_PARAMS: frozenset[str] = frozenset(
    {"omega_2", "omega_3", "kappa", "mu_1_0", "mu_3_0"}
)

#: Parameters that only require softmax recompute (no forward pass).
_RESPONSE_PARAMS: frozenset[str] = frozenset({"beta", "zeta"})

#: Preset parameter profiles for common clinical scenarios.
PROFILES: dict[str, dict[str, float]] = {
    "healthy baseline": {
        "omega_2": -3.0,
        "omega_3": -6.0,
        "kappa": 1.0,
        "beta": 2.5,
        "zeta": 0.2,
        "mu_1_0": 0.0,
        "mu_3_0": 0.0,
    },
    "post-concussion": {
        "omega_2": -4.0,
        "omega_3": -7.0,
        "kappa": 0.8,
        "beta": 2.0,
        "zeta": 0.3,
        "mu_1_0": 0.0,
        "mu_3_0": 0.0,
    },
    "post-psilocybin": {
        "omega_2": -2.5,
        "omega_3": -7.0,
        "kappa": 1.1,
        "beta": 2.0,
        "zeta": 0.3,
        "mu_1_0": 0.0,
        "mu_3_0": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_phase_boundaries(trials: list) -> list[dict]:
    """Extract stable/volatile phase boundary segments from trial list.

    Parameters
    ----------
    trials : list
        Trial objects from :func:`~prl_hgf.env.simulator.generate_session`.

    Returns
    -------
    list[dict]
        List of dicts with keys ``"start"``, ``"end"``, ``"label"``
        (``"stable"`` or ``"volatile"``).
    """
    boundaries: list[dict] = []
    current_key: tuple | None = None
    start: int = 0

    for t in trials:
        key = (t.set_idx, t.phase_name, t.phase_label)
        if key != current_key:
            if current_key is not None:
                boundaries.append(
                    {
                        "start": start,
                        "end": t.trial_idx,
                        "label": current_key[2],
                    }
                )
            current_key = key
            start = t.trial_idx

    if current_key is not None:
        last_trial_idx = trials[-1].trial_idx
        boundaries.append(
            {
                "start": start,
                "end": last_trial_idx + 1,
                "label": current_key[2],
            }
        )

    return boundaries


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ParamExplorer:
    """Interactive HGF parameter explorer widget.

    Instantiate once in a Jupyter notebook cell. The constructor performs
    a one-time JAX warm-up (~5-10 s) and then creates the slider controls
    and 4-panel figure. Call :meth:`display` to render the widget.

    Parameters
    ----------
    config : object or None, optional
        :class:`~prl_hgf.env.task_config.AnalysisConfig` object.  If
        ``None``, loads the default config via
        :func:`~prl_hgf.env.task_config.load_config`.

    Attributes
    ----------
    _sliders : dict[str, ipywidgets.FloatSlider]
        Keyed by parameter name.
    _cached_beliefs_ : dict[str, numpy.ndarray] or None
        Cached belief trajectories from the last forward pass.
    _cached_choice_probs_ : numpy.ndarray or None
        Cached choice probability matrix, shape ``(n_trials, 3)``.
    _cached_lr_ : numpy.ndarray or None
        Cached effective learning rates, shape ``(n_trials, 3)``.
    _fig : matplotlib.figure.Figure
        The 4-panel figure.
    _axes : list[matplotlib.axes.Axes]
        Four axes: beliefs, volatility, choice probs, effective LR.

    Examples
    --------
    In a Jupyter notebook::

        %matplotlib widget
        from prl_hgf.gui import ParamExplorer
        explorer = ParamExplorer()
        display(explorer.display())
    """

    def __init__(self, config=None) -> None:  # noqa: ANN001
        """Initialise the explorer: warm up JAX, create sliders and figure."""
        # ------------------------------------------------------------------ #
        # 1. Load config and generate fixed trial sequence
        # ------------------------------------------------------------------ #
        if config is None:
            config = load_config()
        self._config = config

        trials = generate_session(config, seed=GUI_SEED)
        self._trials = trials
        n_trials = len(trials)

        # Ground-truth reward probabilities — shape (n_trials, 3)
        self._gt_probs: np.ndarray = np.array(
            [t.cue_probs for t in trials], dtype=float
        )

        # Phase boundaries for background shading
        self._phase_boundaries = _extract_phase_boundaries(trials)

        # ------------------------------------------------------------------ #
        # 2. Warm up JAX (one-time JIT compilation)
        # ------------------------------------------------------------------ #
        print("Warming up JAX (one-time ~5-10s)...")
        net_wu = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
        rng_wu = np.random.default_rng(GUI_SEED)
        result_wu = simulate_agent(
            net_wu, trials, beta=2.5, zeta=0.2, rng=rng_wu
        )
        self._warmup_choices_: list[int] = result_wu.choices

        # Build fixed observation mask from warm-up simulation
        self._inp_data_, self._obs_mask_ = prepare_input_data(
            trials, result_wu.choices, result_wu.rewards
        )

        # Also warm up the batch forward pass path for both model levels
        _net2 = build_2level_network(omega_2=-3.0)
        _net2.input_data(
            input_data=self._inp_data_, observed=self._obs_mask_
        )
        _net3 = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
        _net3.input_data(
            input_data=self._inp_data_, observed=self._obs_mask_
        )
        print("JAX warm-up complete.")

        # ------------------------------------------------------------------ #
        # 3. Create sliders
        # ------------------------------------------------------------------ #
        slider_layout = widgets.Layout(width="400px")
        slider_style = {"description_width": "80px"}

        self._sliders: dict[str, widgets.FloatSlider] = {
            "omega_2": widgets.FloatSlider(
                value=-3.0,
                min=PARAM_BOUNDS["omega_2"][0],
                max=PARAM_BOUNDS["omega_2"][1],
                step=0.1,
                description="omega_2",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "omega_3": widgets.FloatSlider(
                value=-6.0,
                min=PARAM_BOUNDS["omega_3"][0],
                max=PARAM_BOUNDS["omega_3"][1],
                step=0.1,
                description="omega_3",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "kappa": widgets.FloatSlider(
                value=1.0,
                min=PARAM_BOUNDS["kappa"][0],
                max=PARAM_BOUNDS["kappa"][1],
                step=0.05,
                description="kappa",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "beta": widgets.FloatSlider(
                value=2.5,
                min=PARAM_BOUNDS["beta"][0],
                max=PARAM_BOUNDS["beta"][1],
                step=0.1,
                description="beta",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "zeta": widgets.FloatSlider(
                value=0.2,
                min=PARAM_BOUNDS["zeta"][0],
                max=PARAM_BOUNDS["zeta"][1],
                step=0.1,
                description="zeta",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "mu_1_0": widgets.FloatSlider(
                value=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                description="mu_1_0",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
            "mu_3_0": widgets.FloatSlider(
                value=0.0,
                min=-4.0,
                max=4.0,
                step=0.1,
                description="mu_3_0",
                continuous_update=False,
                style=slider_style,
                layout=slider_layout,
            ),
        }

        # ------------------------------------------------------------------ #
        # 4. Create model toggle and preset buttons
        # ------------------------------------------------------------------ #
        self._model_toggle = widgets.ToggleButtons(
            options=["2-level", "3-level"],
            value="3-level",
            description="Model:",
            style={"description_width": "50px"},
        )
        self._preset_toggle = widgets.ToggleButtons(
            options=list(PROFILES.keys()),
            value=None,
            description="Preset:",
            style={"description_width": "50px"},
            allow_none=True,
        )

        # ------------------------------------------------------------------ #
        # 5. Create 4-panel figure
        # ------------------------------------------------------------------ #
        # Use non-interactive Agg if no display backend is set yet, otherwise
        # allow the caller (notebook) to set %matplotlib widget beforehand.
        fig, axes = plt.subplots(
            4, 1, figsize=(12, 10), sharex=True
        )
        self._fig = fig
        self._axes: list[matplotlib.axes.Axes] = list(axes)

        trial_x = np.arange(n_trials)
        cue_colors = ["tab:blue", "tab:orange", "tab:green"]

        # Panel 0: Belief trajectories
        ax0 = self._axes[0]
        ax0.set_ylabel("P(reward | cue)")
        ax0.set_ylim(0, 1)
        ax0.set_title("Belief trajectories P(reward)")
        self._belief_lines = [
            ax0.plot(trial_x, np.full(n_trials, 0.5), color=cue_colors[i],
                     label=f"Cue {i}")[0]
            for i in range(3)
        ]
        self._gt_lines = [
            ax0.plot(trial_x, self._gt_probs[:, i], "--",
                     color=cue_colors[i], alpha=0.4,
                     label=f"GT cue {i}")[0]
            for i in range(3)
        ]
        ax0.legend(loc="upper right", fontsize=7, ncol=2)

        # Panel 1: Volatility (3-level only)
        ax1 = self._axes[1]
        ax1.set_ylabel("mu2 (volatility)")
        ax1.set_title("Shared volatility mu2")
        self._mu2_line = ax1.plot(
            trial_x, np.zeros(n_trials), color="tab:red"
        )[0]

        # Panel 2: Choice probabilities
        ax2 = self._axes[2]
        ax2.set_ylabel("P(choose cue)")
        ax2.set_ylim(0, 1)
        ax2.set_title("Choice probabilities P(choose cue)")
        self._prob_lines = [
            ax2.plot(trial_x, np.full(n_trials, 1.0 / 3.0),
                     color=cue_colors[i], label=f"Cue {i}")[0]
            for i in range(3)
        ]
        ax2.legend(loc="upper right", fontsize=7)

        # Panel 3: Effective learning rate
        ax3 = self._axes[3]
        ax3.set_ylabel("Effective LR")
        ax3.set_ylim(0, 1)
        ax3.set_title("Effective learning rate per cue")
        ax3.set_xlabel("Trial")
        self._lr_lines = [
            ax3.plot(trial_x, np.ones(n_trials),
                     color=cue_colors[i], label=f"Cue {i}")[0]
            for i in range(3)
        ]
        ax3.legend(loc="upper right", fontsize=7)

        # Phase boundary shading on all panels
        shade_colors = {"stable": "lightblue", "volatile": "lightyellow"}
        for boundary in self._phase_boundaries:
            for ax in self._axes:
                ax.axvspan(
                    boundary["start"],
                    boundary["end"],
                    alpha=0.05,
                    color=shade_colors.get(boundary["label"], "lightgray"),
                )

        fig.tight_layout()

        # ------------------------------------------------------------------ #
        # 6. Cache init
        # ------------------------------------------------------------------ #
        self._cached_beliefs_: dict[str, np.ndarray] | None = None
        self._cached_choice_probs_: np.ndarray | None = None
        self._cached_lr_: np.ndarray | None = None

        # ------------------------------------------------------------------ #
        # 7. Wire up observers
        # ------------------------------------------------------------------ #
        self._observe_all()
        self._model_toggle.observe(self._on_model_toggle, names="value")
        self._preset_toggle.observe(self._load_preset, names="value")

        # ------------------------------------------------------------------ #
        # 8. Initial forward pass so figure is populated on creation
        # ------------------------------------------------------------------ #
        self._run_forward_pass()
        self._update_choice_probs()
        self._update_plot()

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def _run_forward_pass(self) -> None:
        """Run batch forward pass and cache belief trajectories.

        Reads current slider values, builds the appropriate network, runs
        ``net.input_data`` over all trials, and stores results in
        ``_cached_beliefs_`` and ``_cached_lr_``.
        """
        model = self._model_toggle.value
        omega_2 = self._sliders["omega_2"].value
        mu_1_0 = self._sliders["mu_1_0"].value

        if model == "2-level":
            net = build_2level_network(omega_2=omega_2)
        else:
            omega_3 = self._sliders["omega_3"].value
            kappa = self._sliders["kappa"].value
            net = build_3level_network(
                omega_2=omega_2, omega_3=omega_3, kappa=kappa
            )

        # Set initial beliefs on binary INPUT nodes
        for idx in INPUT_NODES:
            net.attributes[idx]["expected_mean"] = float(mu_1_0)

        # For 3-level: also set initial volatility node prior mean
        if model == "3-level":
            mu_3_0 = self._sliders["mu_3_0"].value
            net.attributes[VOLATILITY_NODE]["mean"] = float(mu_3_0)

        # Batch forward pass over all trials
        net.input_data(
            input_data=self._inp_data_, observed=self._obs_mask_
        )

        # Extract beliefs
        if model == "2-level":
            self._cached_beliefs_ = extract_beliefs(net)
        else:
            self._cached_beliefs_ = extract_beliefs_3level(net)

        # Extract effective learning rate: expected_precision / precision
        # for each BELIEF_NODE (continuous-state level-1 nodes 1, 3, 5)
        traj = net.node_trajectories
        lr_cols = []
        for node_idx in BELIEF_NODES:
            ep = np.asarray(traj[node_idx]["expected_precision"])
            p = np.asarray(traj[node_idx]["precision"])
            # Clip denominator to avoid divide-by-zero
            lr_cols.append(ep / np.where(p == 0, 1e-10, p))
        self._cached_lr_ = np.column_stack(lr_cols)

    # ---------------------------------------------------------------------- #
    # Choice probability update
    # ---------------------------------------------------------------------- #

    def _update_choice_probs(self) -> None:
        """Recompute choice probabilities from cached beliefs and slider values.

        Uses ``scipy.special.softmax`` with stickiness based on warm-up
        choices.  Does NOT re-run the HGF forward pass.
        """
        if self._cached_beliefs_ is None:
            return

        beta = self._sliders["beta"].value
        zeta = self._sliders["zeta"].value

        mu1 = np.column_stack(
            [
                self._cached_beliefs_["p_reward_cue0"],
                self._cached_beliefs_["p_reward_cue1"],
                self._cached_beliefs_["p_reward_cue2"],
            ]
        )  # shape (n_trials, 3)

        n = mu1.shape[0]
        stick = np.zeros((n, 3))
        choices_arr = np.array(self._warmup_choices_)
        prev = np.concatenate([[-1], choices_arr[:-1]])
        for k in range(3):
            stick[prev == k, k] = 1.0

        logits = beta * mu1 + zeta * stick
        self._cached_choice_probs_ = softmax(logits, axis=1)

    # ---------------------------------------------------------------------- #
    # Plot update
    # ---------------------------------------------------------------------- #

    def _update_plot(self) -> None:
        """Update all line artists from cached arrays and request redraw.

        Applies ``relim``/``autoscale_view`` for dynamic-range axes (mu2),
        keeping fixed-range axes (beliefs, choice probs, LR) at [0, 1].
        """
        if self._cached_beliefs_ is None:
            return

        beliefs = self._cached_beliefs_

        # Panel 0: belief trajectories
        for i in range(3):
            self._belief_lines[i].set_ydata(beliefs[f"p_reward_cue{i}"])
            self._gt_lines[i].set_ydata(self._gt_probs[:, i])

        # Panel 1: volatility (only meaningful for 3-level)
        if "mu2_volatility" in beliefs:
            self._mu2_line.set_ydata(beliefs["mu2_volatility"])
            self._axes[1].relim()
            self._axes[1].autoscale_view()

        # Panel 2: choice probabilities
        if self._cached_choice_probs_ is not None:
            for i in range(3):
                self._prob_lines[i].set_ydata(
                    self._cached_choice_probs_[:, i]
                )

        # Panel 3: effective learning rates
        if self._cached_lr_ is not None:
            for i in range(3):
                self._lr_lines[i].set_ydata(self._cached_lr_[:, i])

        self._fig.canvas.draw_idle()

    # ---------------------------------------------------------------------- #
    # Observer callbacks
    # ---------------------------------------------------------------------- #

    def _on_slider_change(self, change: dict) -> None:  # type: ignore[override]
        """Respond to a slider value change.

        If the changed parameter is an HGF parameter (requires forward
        pass), runs ``_run_forward_pass``.  Always recomputes choice
        probabilities and updates the plot.

        Parameters
        ----------
        change : dict
            ipywidgets change dict.  ``change["owner"].description`` gives
            the slider's parameter name.
        """
        param = change["owner"].description
        if param in _HGF_PARAMS:
            self._run_forward_pass()
        self._update_choice_probs()
        self._update_plot()

    def _on_model_toggle(self, change: dict) -> None:  # type: ignore[override]
        """Respond to model type toggle (2-level / 3-level).

        Shows/hides the volatility panel and 3-level-only sliders, then
        reruns the forward pass for the selected model.

        Parameters
        ----------
        change : dict
            ipywidgets change dict.  ``change["new"]`` is ``"2-level"``
            or ``"3-level"``.
        """
        is_3level = change["new"] == "3-level"
        self._axes[1].set_visible(is_3level)

        # Show/hide 3-level-only sliders
        vis = "" if is_3level else "none"
        for name in ("omega_3", "kappa", "mu_3_0"):
            self._sliders[name].layout.display = vis

        self._fig.tight_layout()
        self._run_forward_pass()
        self._update_choice_probs()
        self._update_plot()

    def _load_preset(self, change: dict) -> None:  # type: ignore[override]
        """Load a preset parameter profile.

        Suppresses intermediate slider callbacks while bulk-setting values,
        then triggers exactly one forward pass.

        Parameters
        ----------
        change : dict
            ipywidgets change dict.  ``change["new"]`` is the preset name.
        """
        preset_name = change["new"]
        if preset_name is None or preset_name not in PROFILES:
            return

        profile = PROFILES[preset_name]
        self._unobserve_all()
        for name, slider in self._sliders.items():
            if name in profile:
                slider.value = profile[name]
        self._observe_all()

        self._run_forward_pass()
        self._update_choice_probs()
        self._update_plot()

    # ---------------------------------------------------------------------- #
    # Observer management
    # ---------------------------------------------------------------------- #

    def _unobserve_all(self) -> None:
        """Disconnect all slider observers to suppress intermediate callbacks."""
        for slider in self._sliders.values():
            slider.unobserve(self._on_slider_change, names="value")

    def _observe_all(self) -> None:
        """Reconnect all slider observers."""
        for slider in self._sliders.values():
            slider.observe(self._on_slider_change, names="value")

    # ---------------------------------------------------------------------- #
    # Display
    # ---------------------------------------------------------------------- #

    def display(self) -> widgets.Widget:
        """Assemble and return the full widget layout.

        The caller (notebook cell) is responsible for calling
        ``display(explorer.display())`` or simply placing the return value
        as the last expression in a cell.  Do NOT call ``plt.show()`` or
        ``display()`` inside this method.

        Returns
        -------
        ipywidgets.Widget
            A :class:`~ipywidgets.VBox` containing the control panel and
            figure canvas arranged in rows.
        """
        top_row = widgets.HBox([self._model_toggle, self._preset_toggle])
        slider_col = widgets.VBox(list(self._sliders.values()))

        try:
            canvas = self._fig.canvas
            main_row = widgets.HBox([slider_col, canvas])
        except AttributeError:
            # Agg backend (tests): canvas is not a widget; return sliders only
            main_row = widgets.HBox([slider_col])

        return widgets.VBox([top_row, main_row])
