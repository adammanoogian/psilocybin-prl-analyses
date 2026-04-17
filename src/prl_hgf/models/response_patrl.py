"""PAT-RL Model A response log-likelihood (binary approach/avoid softmax).

Model A computes per-trial expected value of approach:

    EV_approach = sigmoid(mu_2) * V_rew - (1 - sigmoid(mu_2)) * V_shk

where ``sigmoid(mu_2) = P(state=1) = P(dangerous)``.  The approach action
yields reward ``V_rew`` in the safe state (state=0) and shock ``V_shk`` in
the dangerous state (state=1).  Therefore:

    EV_approach = (1 - P_danger) * V_rew - P_danger * V_shk
                = sigmoid(-mu_2) * V_rew - sigmoid(mu_2) * V_shk

**Convention**: ``state=1`` means dangerous.  ``sigmoid(mu_2) = P(dangerous)``.

    - When ``mu_2`` is large positive: P(dangerous) ≈ 1, EV_approach ≈ -V_shk
      (approach is costly).
    - When ``mu_2`` is large negative: P(dangerous) ≈ 0, EV_approach ≈ V_rew
      (approach is safe and rewarding).

The avoid action has zero EV by construction (no outcome is delivered for
avoidance).  Binary choice:

    P(approach) = softmax([0, beta * EV_approach])[1]

Numerical safety
----------------
``mu_2`` is clipped to ``[-30, 30]`` before sigmoid to keep tails finite
under aggressive fits (matches the |mu_2| < 14 HGF-level clamping in the
pick_best_cue hierarchical fitter, with a generous outer envelope here
because this is pure response computation without HGF internals).

Models B (Delta-HR bias), C (Delta-HR x value), and D (trial-varying omega)
will land in Phase 19+; this module is Model A only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.nn import log_softmax

#: Clip bound applied to mu_2 before sigmoid to prevent numerical overflow.
MU2_CLIP: float = 30.0


def expected_value(
    mu2: jnp.ndarray,
    reward_mag: jnp.ndarray,
    shock_mag: jnp.ndarray,
) -> jnp.ndarray:
    """Compute expected value of the approach action.

    Parameters
    ----------
    mu2 : jnp.ndarray
        Shape ``(n_trials,)``.  Continuous value posterior mean (log-odds of
        state=1=dangerous).
    reward_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Reward magnitude for approach in safe state.
    shock_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Shock magnitude for approach in dangerous state.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial expected value of approach.
        Positive → approach is preferable; negative → avoid is preferable.
    """
    mu2_safe = jnp.clip(mu2, -MU2_CLIP, MU2_CLIP)
    p_danger: jnp.ndarray = jax.nn.sigmoid(mu2_safe)  # type: ignore[assignment]
    return (1.0 - p_danger) * reward_mag - p_danger * shock_mag  # type: ignore[return-value]


def model_a_logp(
    mu2: jnp.ndarray,
    choices: jnp.ndarray,
    reward_mag: jnp.ndarray,
    shock_mag: jnp.ndarray,
    beta: float | jnp.ndarray,
) -> jnp.ndarray:
    """Per-trial binary choice log-probability under Model A.

    Computes:

    .. math::

        \\text{EV}_{\\text{approach}} = (1 - \\sigma(\\mu_2)) V_{\\text{rew}}
                                       - \\sigma(\\mu_2) V_{\\text{shk}}

    Then evaluates:

    .. math::

        \\log P(\\text{choice}) = \\log\\text{softmax}([0,\\;
                                  \\beta \\cdot \\text{EV}_{\\text{approach}}])
                                  [\\text{choice}]

    Parameters
    ----------
    mu2 : jnp.ndarray
        Shape ``(n_trials,)``.  Continuous value posterior mean (log-odds of
        state=1=dangerous) from the HGF belief node.
    choices : jnp.ndarray
        Shape ``(n_trials,)``, dtype ``int32``.  Observed binary choices:
        ``1`` = approach, ``0`` = avoid.
    reward_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Reward magnitude per trial.
    shock_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Shock magnitude per trial.
    beta : float or jnp.ndarray
        Inverse temperature (decision noise).  Scalar.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial log-likelihood of the observed
        choice.  All values are finite for finite inputs.

    Notes
    -----
    The avoid action (choice=0) always has logit 0, which anchors the
    softmax.  The approach action (choice=1) has logit ``beta * EV_approach``.
    When ``EV_approach < 0`` the avoid logit is higher, so P(approach) < 0.5.
    """
    ev_approach = expected_value(mu2, reward_mag, shock_mag)
    # logits: [avoid_logit=0, approach_logit=beta*EV]  shape (n_trials, 2)
    logits = jnp.stack(
        [jnp.zeros_like(ev_approach), beta * ev_approach],
        axis=-1,
    )
    logp_all = log_softmax(logits, axis=-1)  # (n_trials, 2)
    # Index per-trial choice (0 or 1) from the last axis
    return jnp.take_along_axis(
        logp_all,
        choices[:, None].astype(jnp.int32),
        axis=-1,
    ).squeeze(-1)


__all__ = ["model_a_logp", "expected_value", "MU2_CLIP"]
