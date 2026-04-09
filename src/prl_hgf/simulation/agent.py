"""Single-agent HGF simulator for the PRL pick_best_cue pipeline.

Implements trial-by-trial simulation using the pyhgf Network API with the
attribute carry pattern: after each 1-trial ``input_data`` call, the final
state is threaded forward via ``net.attributes = net.last_attributes``.

This module provides two public functions:

* :func:`simulate_agent` — runs one session's worth of trials for a single
  participant given a pre-built network and response parameters.
* :func:`sample_participant_params` — draws individual parameters from group
  distributions with optional session deltas and clips to model bounds.

Notes
-----
All simulation math uses NumPy (not JAX) — no gradients are needed in the
simulation path.  The network itself still uses JAX internally via pyhgf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyhgf.model import Network

from prl_hgf.env.simulator import Trial, generate_reward
from prl_hgf.env.task_config import GroupConfig, SessionConfig
from prl_hgf.models.hgf_2level import INPUT_NODES

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Parameter bounds (inclusive on both sides) from prl_analysis.yaml.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "omega_2": (-8.0, 2.0),
    "omega_3": (-12.0, 0.0),
    "kappa": (0.01, 2.0),
    "beta": (0.01, 20.0),
    "zeta": (-5.0, 5.0),
}

__all__ = [
    "simulate_agent",
    "sample_participant_params",
    "SimulationResult",
    "PARAM_BOUNDS",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Immutable output of one simulated agent session.

    Parameters
    ----------
    choices : list[int]
        Chosen cue index (0, 1, or 2) for each trial.
    rewards : list[int]
        Binary reward outcome (0 or 1) for each trial.
    beliefs : list[tuple[float, float, float]]
        Prior reward probability for each cue at each trial, read from
        ``net.attributes[node]["expected_mean"]`` *before* the trial's
        belief update.  Useful for verifying that the agent correctly
        tracks the reward-generating process.
    diverged : bool
        ``True`` if the HGF state produced NaN beliefs during the session.
        Remaining trials after divergence use uniform random choice.
    """

    choices: list[int]
    rewards: list[int]
    beliefs: list[tuple[float, float, float]]
    diverged: bool = False


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------


def sample_participant_params(
    group_cfg: GroupConfig,
    session_cfg: SessionConfig,
    session_idx: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample individual participant parameters from group distributions.

    Draws each parameter from ``Normal(group_mean, group_sd)``, adds session
    deltas for non-baseline sessions, then clips to model bounds.  The
    clip-after-delta ordering ensures that a delta applied to a clipped
    baseline value cannot push the result out of bounds.

    Parameters
    ----------
    group_cfg : GroupConfig
        Parameter distributions for the participant's group.
    session_cfg : SessionConfig
        Session-level deltas for the participant's group.  Used when
        ``session_idx > 0``.
    session_idx : int
        Session index: 0 = baseline (no delta), 1 = post_dose, 2 = followup.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    dict[str, float]
        Parameter dictionary with keys ``"omega_2"``, ``"omega_3"``,
        ``"kappa"``, ``"beta"``, ``"zeta"``.  All values are within
        :data:`PARAM_BOUNDS`.

    Examples
    --------
    >>> import numpy as np
    >>> from prl_hgf.env.task_config import load_config
    >>> config = load_config()
    >>> group_cfg = config.simulation.groups["placebo"]
    >>> session_cfg = config.simulation.session_deltas["placebo"]
    >>> rng = np.random.default_rng(0)
    >>> params = sample_participant_params(group_cfg, session_cfg, 0, rng)
    >>> set(params.keys()) == {"omega_2", "omega_3", "kappa", "beta", "zeta"}
    True
    """
    omega_2 = rng.normal(group_cfg.omega_2.mean, group_cfg.omega_2.sd)
    omega_3 = rng.normal(group_cfg.omega_3.mean, group_cfg.omega_3.sd)
    kappa = rng.normal(group_cfg.kappa.mean, group_cfg.kappa.sd)
    beta = rng.normal(group_cfg.beta.mean, group_cfg.beta.sd)
    zeta = rng.normal(group_cfg.zeta.mean, group_cfg.zeta.sd)

    # Apply session deltas for non-baseline sessions (additive on natural scale)
    if session_idx > 0:
        delta_idx = session_idx - 1  # 0 = post_dose, 1 = followup
        omega_2 += session_cfg.omega_2_deltas[delta_idx]
        kappa += session_cfg.kappa_deltas[delta_idx]
        beta += session_cfg.beta_deltas[delta_idx]
        zeta += session_cfg.zeta_deltas[delta_idx]

    # Clip ALL parameters to model bounds (after adding deltas)
    omega_2 = float(np.clip(omega_2, *PARAM_BOUNDS["omega_2"]))
    omega_3 = float(np.clip(omega_3, *PARAM_BOUNDS["omega_3"]))
    kappa = float(np.clip(kappa, *PARAM_BOUNDS["kappa"]))
    beta = float(np.clip(beta, *PARAM_BOUNDS["beta"]))
    zeta = float(np.clip(zeta, *PARAM_BOUNDS["zeta"]))

    return {
        "omega_2": omega_2,
        "omega_3": omega_3,
        "kappa": kappa,
        "beta": beta,
        "zeta": zeta,
    }


# ---------------------------------------------------------------------------
# Agent simulation loop
# ---------------------------------------------------------------------------


def simulate_agent(
    net: Network,
    trials: list[Trial],
    beta: float,
    zeta: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate one agent's trial-by-trial choices through a session.

    Uses the attribute carry pattern: after each 1-trial ``input_data`` call,
    the posterior state is copied back into ``net.attributes`` so that it
    serves as the prior for the next trial.  Prior beliefs for choice
    generation are read *before* calling ``input_data`` on each trial.

    The softmax response model is::

        logits = beta * p_reward + zeta * stick
        probs = softmax(logits)  # numpy, not JAX

    where ``stick[k] = 1`` if ``k == prev_choice`` (0 on trial 0 via
    sentinel ``prev_choice = -1``).

    Parameters
    ----------
    net : pyhgf.model.Network
        Freshly built network configured with the participant's parameters.
        The network's ``attributes`` field must hold the initial priors (i.e.,
        ``input_data`` must NOT have been called yet, or the state must have
        been reset).
    trials : list[Trial]
        Ordered list of trials for the session, from
        :func:`~prl_hgf.env.simulator.generate_session`.
    beta : float
        Inverse temperature (exploitation weight, must be > 0).
    zeta : float
        Stickiness / choice perseveration weight.
    rng : numpy.random.Generator
        Seeded random number generator for stochastic choice and reward
        sampling.

    Returns
    -------
    SimulationResult
        Immutable result with ``choices``, ``rewards``, and ``beliefs``
        (prior ``p_reward`` tuple per trial, shape ``(n_trials, 3)``).

    Notes
    -----
    Each call mutates ``net.attributes`` in-place (attribute carry pattern).
    Do NOT reuse ``net`` after calling this function without rebuilding it.

    Examples
    --------
    >>> import numpy as np
    >>> from prl_hgf.models.hgf_3level import build_3level_network
    >>> from prl_hgf.env.task_config import load_config
    >>> from prl_hgf.env.simulator import generate_session
    >>> config = load_config()
    >>> trials = generate_session(config, seed=0)
    >>> net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    >>> rng = np.random.default_rng(42)
    >>> result = simulate_agent(net, trials, beta=2.0, zeta=0.0, rng=rng)
    >>> len(result.choices)
    420
    """
    choices: list[int] = []
    rewards: list[int] = []
    beliefs: list[tuple[float, float, float]] = []
    prev_choice: int = -1  # sentinel: no stickiness on the first trial
    diverged = False

    for trial in trials:
        if diverged:
            # HGF state is broken — use uniform random for remaining trials
            choice = int(rng.choice(3))
            reward = generate_reward(choice, trial.cue_probs, rng)
            choices.append(choice)
            rewards.append(reward)
            beliefs.append((float("nan"), float("nan"), float("nan")))
            prev_choice = choice
            continue

        # --- Step 1: Read PRIOR beliefs before this trial's update ---
        p_reward = np.array(
            [
                float(net.attributes[INPUT_NODES[0]]["expected_mean"]),  # cue 0
                float(net.attributes[INPUT_NODES[1]]["expected_mean"]),  # cue 1
                float(net.attributes[INPUT_NODES[2]]["expected_mean"]),  # cue 2
            ]
        )

        # --- Step 1b: Check for NaN beliefs (HGF diverged) ---
        if np.any(np.isnan(p_reward)):
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "HGF beliefs diverged to NaN on trial %d — "
                "using uniform random choice for remaining trials",
                len(choices),
            )
            diverged = True
            choice = int(rng.choice(3))
            reward = generate_reward(choice, trial.cue_probs, rng)
            choices.append(choice)
            rewards.append(reward)
            beliefs.append((float("nan"), float("nan"), float("nan")))
            prev_choice = choice
            continue

        # --- Step 2: Softmax with stickiness (numpy, no JAX) ---
        stick = np.zeros(3)
        if prev_choice >= 0:
            stick[prev_choice] = 1.0
        logits = beta * p_reward + zeta * stick
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()

        # --- Step 3: Sample choice ---
        choice = int(rng.choice(3, p=probs))

        # --- Step 4: Generate reward from trial environment ---
        reward = generate_reward(choice, trial.cue_probs, rng)

        choices.append(choice)
        rewards.append(reward)
        beliefs.append((float(p_reward[0]), float(p_reward[1]), float(p_reward[2])))

        # --- Step 5: Update network with this 1-trial observation ---
        inp_t = np.zeros((1, 3), dtype=float)
        obs_t = np.zeros((1, 3), dtype=int)
        inp_t[0, choice] = float(reward)
        obs_t[0, choice] = 1
        net.input_data(input_data=inp_t, observed=obs_t)

        # --- Step 6: CRITICAL — carry posterior forward as next trial's prior ---
        net.attributes = net.last_attributes

        prev_choice = choice

    return SimulationResult(
        choices=choices, rewards=rewards, beliefs=beliefs, diverged=diverged,
    )
