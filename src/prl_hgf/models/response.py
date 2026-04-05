"""Softmax + stickiness response function for the PRL pick_best_cue pipeline.

Implements the choice likelihood function used in Bayesian fitting (Phase 4).
The function bridges HGF beliefs to observed choices via a softmax over three
cues where the logit for cue *k* is:

    logit_k = beta * mu1_k + zeta * I[prev_choice == k]

where ``mu1_k`` is the sigmoid-transformed reward probability P(reward | cue k)
from the binary-state input node (``expected_mean``), ``beta`` is the inverse
temperature (decision noise), and ``zeta`` is the stickiness parameter
controlling choice perseveration.

The function signature matches the pyhgf ``Network.surprise()`` API exactly.

Notes
-----
All array operations use JAX to ensure compatibility with JAX-traced gradients
required for MCMC sampling in Phase 4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pyhgf.model import Network

from prl_hgf.models.hgf_2level import INPUT_NODES


def softmax_stickiness_surprise(
    hgf: Network,
    response_function_inputs: jnp.ndarray,
    response_function_parameters: jnp.ndarray,
) -> float:
    """Compute negative log-likelihood (surprise) for an observed choice sequence.

    This function is compatible with the pyhgf ``Network.surprise()`` API and
    serves as the likelihood function linking HGF beliefs to observed choices.

    The choice probability for cue *k* on trial *t* is:

    .. math::

        P(choice_t = k) = \\frac{\\exp(\\beta \\cdot \\mu^{(1)}_{k,t}
                                        + \\zeta \\cdot \\mathbb{1}[c_{t-1} = k])}
                                {\\sum_{j} \\exp(\\beta \\cdot \\mu^{(1)}_{j,t}
                                                  + \\zeta \\cdot \\mathbb{1}[c_{t-1} = j])}

    where :math:`\\mu^{(1)}_{k,t}` is the expected reward probability for cue *k*
    at trial *t* (``expected_mean`` from the binary-state input node), and
    :math:`c_{t-1}` is the choice on the previous trial.  For the first trial
    there is no previous choice, so the stickiness term is zero for all cues.

    Parameters
    ----------
    hgf : pyhgf.model.Network
        A fitted HGF network (after calling ``input_data``).
        Belief trajectories are read from ``hgf.node_trajectories``.
    response_function_inputs : array-like, shape (n_trials,)
        Observed choice indices, values in ``{0, 1, 2}`` (one per cue).
        May be a NumPy array; cast to JAX internally.
    response_function_parameters : jax.numpy.ndarray, shape (2,)
        Parameter vector ``[beta, zeta]``.  ``beta = params[0]`` is the inverse
        temperature; ``zeta = params[1]`` is the stickiness weight.

    Returns
    -------
    float
        Total surprise (negative sum of log-likelihoods) across all trials.
        Returns ``jnp.inf`` if any intermediate value is NaN (guards against
        degenerate parameter combinations).

    Notes
    -----
    The function uses ``expected_mean`` from the binary-state INPUT_NODES
    ``(0, 2, 4)`` as ``mu1_k``, which represents the sigmoid-transformed
    reward probability P(reward | cue k) in [0, 1].  This is the correct
    quantity for the softmax formula (RSP-02).  Continuous-state belief nodes
    are NOT used here.
    """
    choices = jnp.asarray(response_function_inputs)
    n_trials = choices.shape[0]

    beta = response_function_parameters[0]
    zeta = response_function_parameters[1]

    # Extract expected reward probabilities from binary input nodes
    # INPUT_NODES = (0, 2, 4) -- each is a binary-state node with expected_mean in [0,1]
    mu1_0 = hgf.node_trajectories[INPUT_NODES[0]]["expected_mean"]  # cue 0
    mu1_1 = hgf.node_trajectories[INPUT_NODES[1]]["expected_mean"]  # cue 1
    mu1_2 = hgf.node_trajectories[INPUT_NODES[2]]["expected_mean"]  # cue 2

    # Stack to shape (n_trials, 3)
    mu1 = jnp.stack([mu1_0, mu1_1, mu1_2], axis=1)

    # Build stickiness indicator matrix of shape (n_trials, 3)
    # First trial uses sentinel -1 so stickiness term is 0 for all cues (RSP-04)
    prev_choices = jnp.concatenate([jnp.array([-1]), choices[:-1]])
    cue_indices = jnp.arange(3)
    # stick[t, k] = 1.0 if prev_choices[t] == k else 0.0
    stick = (prev_choices[:, None] == cue_indices[None, :]).astype(jnp.float32)

    # Compute logits and numerically-stable log-softmax
    logits = beta * mu1 + zeta * stick
    log_probs = jax.nn.log_softmax(logits, axis=1)

    # Index chosen cue log-probability for each trial
    trial_loglik = log_probs[jnp.arange(n_trials), choices.astype(jnp.int32)]

    # Total surprise = negative sum of log-likelihoods
    surprise = -jnp.sum(trial_loglik)

    # NaN guard: return inf if any numerical issues occurred
    return jnp.where(jnp.isnan(surprise), jnp.inf, surprise)
