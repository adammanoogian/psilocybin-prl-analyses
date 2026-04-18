"""PAT-RL response log-likelihood functions: Models A, B, and C.

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

    Model A:  P(approach) = softmax([0, beta * EV + b])[1]
    Model B:  P(approach) = softmax([0, beta * EV + b + gamma * ΔHR(t)])[1]
    Model C:  P(approach) = softmax([0, (beta + alpha*ΔHR(t)) * EV + b
                                       + gamma * ΔHR(t)])[1]

Models B and C extend Model A's logit by adding ΔHR-modulated terms.
Model D (trial-varying omega, ω_eff(t) = ω + λ·ΔHR(t)) modifies the HGF
scan body and is deferred to Plan 20-03; its code lives in
``hierarchical_patrl.py::_clamped_step_model_d``.

Avoid EV = 0 by construction
------------------------------
The response-model logits are ``[0, beta*EV + b + ...]`` where the ``0``
anchors the avoid logit.  The participant receives a stochastic outcome
from avoid (see ``configs/pat_rl.yaml contingencies.avoid``), but the
**participant's EV calculation** only depends on the approach option — they
experience the avoid outcome regardless of their belief about the state.
Therefore ``EV(avoid) = 0`` (no expected-value signal from avoidance) is
correct for the decision model.  The stochastic avoid contingency is wired
into the simulator (Plans 20-02/20-03) but does NOT enter the logp.  See
dependency_context in 20-02-PLAN.md for the full rationale.

Numerical safety
----------------
``mu_2`` is clipped to ``[-30, 30]`` before sigmoid to keep tails finite
under aggressive fits (matches the |mu_2| < 14 HGF-level clamping in the
pick_best_cue hierarchical fitter, with a generous outer envelope here
because this is pure response computation without HGF internals).
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
    b: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Per-trial binary choice log-probability under Model A.

    Computes:

    .. math::

        \\text{EV}_{\\text{approach}} = (1 - \\sigma(\\mu_2)) V_{\\text{rew}}
                                       - \\sigma(\\mu_2) V_{\\text{shk}}

    Then evaluates:

    .. math::

        \\log P(\\text{choice}) = \\log\\text{softmax}([0,\\;
                                  \\beta \\cdot \\text{EV}_{\\text{approach}}
                                  + b])
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
    b : float or jnp.ndarray, default 0.0
        Response bias on the approach logit.  Positive = approach bias;
        negative = avoid bias.  Default 0.0 preserves Phase 18 behavior.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial log-likelihood of the observed
        choice.  All values are finite for finite inputs.

    Notes
    -----
    The avoid action (choice=0) always has logit 0, which anchors the
    softmax.  The approach action (choice=1) has logit
    ``beta * EV_approach + b``.
    When ``EV_approach < 0`` and ``b = 0`` the avoid logit is higher, so
    P(approach) < 0.5.  A positive ``b`` shifts the approach logit upward,
    introducing a systematic approach preference.
    """
    ev_approach = expected_value(mu2, reward_mag, shock_mag)
    # logits: [avoid_logit=0, approach_logit=beta*EV+b]  shape (n_trials, 2)
    logits = jnp.stack(
        [jnp.zeros_like(ev_approach), beta * ev_approach + b],
        axis=-1,
    )
    logp_all = log_softmax(logits, axis=-1)  # (n_trials, 2)
    # Index per-trial choice (0 or 1) from the last axis
    return jnp.take_along_axis(
        logp_all,
        choices[:, None].astype(jnp.int32),
        axis=-1,
    ).squeeze(-1)


def model_b_logp(
    mu2: jnp.ndarray,
    choices: jnp.ndarray,
    reward_mag: jnp.ndarray,
    shock_mag: jnp.ndarray,
    beta: float | jnp.ndarray,
    b: float | jnp.ndarray,
    gamma: float | jnp.ndarray,
    delta_hr: jnp.ndarray,
) -> jnp.ndarray:
    """Per-trial log-probability under Model B: ΔHR as additive bias on approach.

    Computes:

    .. math::

        \\log P(\\text{choice}) = \\log\\text{softmax}([0,\\;
                                  \\beta \\cdot \\text{EV} + b
                                  + \\gamma \\cdot \\Delta HR(t)])
                                  [\\text{choice}]

    Sign convention
    ---------------
    ΔHR < 0 indicates anticipatory bradycardia (freezing-associated).
    Positive γ = more avoidance when more bradycardia (because γ * ΔHR
    becomes more negative, lowering the approach logit).  Reference:
    Klaassen et al. 2024, Communications Biology.

    Parameters
    ----------
    mu2 : jnp.ndarray
        Shape ``(n_trials,)``.  HGF level-2 belief mean.
    choices : jnp.ndarray
        Shape ``(n_trials,)``, dtype ``int32``.  Observed binary choices.
    reward_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Reward magnitude per trial.
    shock_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Shock magnitude per trial.
    beta : float or jnp.ndarray
        Inverse temperature (decision noise).  Scalar.
    b : float or jnp.ndarray
        Response bias on approach logit (constant offset).
    gamma : float or jnp.ndarray
        ΔHR additive weight.  Scalar.
    delta_hr : jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial anticipatory ΔHR in bpm.
        Negative = bradycardia.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial log-likelihood of the observed
        choice.  All values are finite for finite inputs.

    Notes
    -----
    Avoid EV is 0 by construction — see module docstring.
    gamma=0 makes Model B identical to Model A (with the same b).
    """
    ev_approach = expected_value(mu2, reward_mag, shock_mag)
    approach_logit = beta * ev_approach + b + gamma * delta_hr
    logits = jnp.stack(
        [jnp.zeros_like(ev_approach), approach_logit],
        axis=-1,
    )
    logp_all = log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        logp_all,
        choices[:, None].astype(jnp.int32),
        axis=-1,
    ).squeeze(-1)


def model_c_logp(
    mu2: jnp.ndarray,
    choices: jnp.ndarray,
    reward_mag: jnp.ndarray,
    shock_mag: jnp.ndarray,
    beta: float | jnp.ndarray,
    b: float | jnp.ndarray,
    alpha: float | jnp.ndarray,
    gamma: float | jnp.ndarray,
    delta_hr: jnp.ndarray,
) -> jnp.ndarray:
    """Per-trial log-probability under Model C: ΔHR modulates value sensitivity.

    Computes:

    .. math::

        \\beta_{\\text{eff}}(t) = \\beta + \\alpha \\cdot \\Delta HR(t)

    .. math::

        \\log P(\\text{choice}) = \\log\\text{softmax}([0,\\;
                                  \\beta_{\\text{eff}}(t)
                                  \\cdot \\text{EV} + b
                                  + \\gamma \\cdot \\Delta HR(t)])
                                  [\\text{choice}]

    Model C nests Model B: set alpha=0 to recover Model B.

    Sign convention
    ---------------
    ΔHR < 0 (bradycardia) with alpha > 0 lowers beta_eff below beta,
    reducing value sensitivity during freeze-associated trials.  Reference:
    Klaassen et al. 2024, Communications Biology.

    Parameters
    ----------
    mu2 : jnp.ndarray
        Shape ``(n_trials,)``.  HGF level-2 belief mean.
    choices : jnp.ndarray
        Shape ``(n_trials,)``, dtype ``int32``.  Observed binary choices.
    reward_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Reward magnitude per trial.
    shock_mag : jnp.ndarray
        Shape ``(n_trials,)``.  Shock magnitude per trial.
    beta : float or jnp.ndarray
        Baseline inverse temperature (decision noise).  Scalar.
    b : float or jnp.ndarray
        Response bias on approach logit (constant offset).
    alpha : float or jnp.ndarray
        ΔHR × EV interaction weight: modulates effective beta per trial.
    gamma : float or jnp.ndarray
        ΔHR additive weight.  Scalar.
    delta_hr : jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial anticipatory ΔHR in bpm.
        Negative = bradycardia.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_trials,)``.  Per-trial log-likelihood of the observed
        choice.  All values are finite for finite inputs.

    Notes
    -----
    Avoid EV is 0 by construction — see module docstring.
    alpha=0 reduces Model C to Model B (with the same b and gamma).
    """
    ev_approach = expected_value(mu2, reward_mag, shock_mag)
    effective_beta = beta + alpha * delta_hr
    approach_logit = effective_beta * ev_approach + b + gamma * delta_hr
    logits = jnp.stack(
        [jnp.zeros_like(ev_approach), approach_logit],
        axis=-1,
    )
    logp_all = log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        logp_all,
        choices[:, None].astype(jnp.int32),
        axis=-1,
    ).squeeze(-1)


__all__ = [
    "model_a_logp",
    "model_b_logp",
    "model_c_logp",
    "expected_value",
    "MU2_CLIP",
]
