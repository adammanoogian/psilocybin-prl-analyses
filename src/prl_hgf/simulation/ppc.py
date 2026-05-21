"""Posterior predictive check (PPC) replay module.

Consumes ArviZ ``InferenceData`` posterior draws and replays each
draw through the HGF forward model to produce per-trial
``P(choice_observed)`` for every participant.  This enables posterior
predictive checks: if the model is well-calibrated, the predicted
choice probabilities should be concentrated near 1 for trials where
the model is confident and at chance for ambiguous trials.

The replay reuses the exact same HGF ``scan_fn``, parameter-injection
pattern, and softmax-stickiness computation from
:mod:`prl_hgf.fitting.hierarchical`, so the logp computation is
bit-identical to the fitting path.

Usage
-----
::

    from prl_hgf.simulation.ppc import posterior_predictive_replay

    p_choice = posterior_predictive_replay(
        idata, input_data_arr, observed_arr, choices_arr,
        model_name="hgf_3level", n_draws=200,
    )
    # p_choice shape: (n_draws_used, P, n_trials)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import arviz as az

__all__ = ["posterior_predictive_replay"]

# ---------------------------------------------------------------------------
# Constants (must match hierarchical.py / jax_session.py)
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 means.
_MU_2_BOUND: float = 14.0

#: Node indices for continuous-state level-1 belief nodes.
_BELIEF_NODES: tuple[int, ...] = (1, 3, 5)

#: Supported model names.
_MODEL_NAMES: tuple[str, ...] = ("hgf_2level", "hgf_3level")


# ---------------------------------------------------------------------------
# Internal helpers (mirrors hierarchical.py exactly)
# ---------------------------------------------------------------------------


def _build_scan_inputs(
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    n_trials: int,
) -> tuple:
    """Build the scan-input tuple expected by pyhgf's ``scan_fn``.

    Parameters
    ----------
    input_data : jnp.ndarray, shape (n_trials, 3)
        Float reward-value array for a single participant.
    observed : jnp.ndarray, shape (n_trials, 3)
        Binary observed mask for a single participant.
    n_trials : int
        Number of trials.

    Returns
    -------
    scan_inputs : tuple
        ``(values, observed_cols, time_steps, None)`` matching the pyhgf
        scan-input contract.
    """
    values = (
        input_data[:, 0:1],
        input_data[:, 1:2],
        input_data[:, 2:3],
    )
    observed_cols = (
        observed[:, 0],
        observed[:, 1],
        observed[:, 2],
    )
    time_steps = jnp.ones(n_trials)
    return (values, observed_cols, time_steps, None)


def _clamped_scan(
    scan_fn: object,
    attrs: dict,
    scan_inputs: tuple,
) -> tuple[dict, tuple[dict, jnp.ndarray]]:
    """Run ``lax.scan`` with Layer 2 NaN-clamping wrapper.

    Identical to :func:`prl_hgf.fitting.hierarchical._clamped_scan`.

    Parameters
    ----------
    scan_fn : callable
        The pyhgf ``Network.scan_fn`` function.
    attrs : dict
        Initial (parameter-injected) attributes pytree.
    scan_inputs : tuple
        ``(values, observed_cols, time_steps, None)``.

    Returns
    -------
    final_attrs : dict
        Final attributes after the clamped scan.
    node_traj : dict
        Per-trial node trajectory.
    stability_mask : jnp.ndarray, shape (n_trials,)
        Boolean mask: ``True`` for stable trials, ``False`` for reverted.
    """
    from jax import lax

    def _clamped_step(
        carry: dict,
        x: tuple,
    ) -> tuple[dict, tuple[dict, jnp.ndarray]]:
        prev_attrs = carry
        new_attrs, new_node = scan_fn(prev_attrs, x)

        leaves = jax.tree_util.tree_leaves(new_attrs)
        all_finite = jnp.all(
            jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        )

        mu_2_vals = jnp.array(
            [
                new_attrs[1]["mean"],
                new_attrs[3]["mean"],
                new_attrs[5]["mean"],
            ]
        )
        mu_2_ok = jnp.all(jnp.abs(mu_2_vals) < _MU_2_BOUND)
        is_stable = all_finite & mu_2_ok

        safe_attrs = jax.tree_util.tree_map(
            lambda n, o: jnp.where(is_stable, n, o),
            new_attrs,
            prev_attrs,
        )

        return safe_attrs, (new_node, is_stable)

    final_attrs, (node_traj, stability_mask) = lax.scan(
        _clamped_step, attrs, scan_inputs
    )

    return final_attrs, (node_traj, stability_mask)


def _compute_p_choice(
    node_traj: dict,
    choices_jax: jnp.ndarray,
    n_trials: int,
    beta: jnp.ndarray,
    zeta: jnp.ndarray,
    stability_mask: jnp.ndarray,
    trial_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute P(choice_observed) per trial from node trajectories.

    Uses the exact same softmax-stickiness computation as
    :func:`prl_hgf.fitting.hierarchical._compute_logp`, but returns
    per-trial probabilities instead of summed log-probabilities.

    Parameters
    ----------
    node_traj : dict
        Per-trial node trajectory from ``lax.scan``.
    choices_jax : jnp.ndarray, shape (n_trials,)
        Chosen cue indices (0, 1, or 2) as int32.
    n_trials : int
        Number of trials.
    beta : jnp.ndarray
        Inverse temperature (scalar).
    zeta : jnp.ndarray
        Stickiness parameter (scalar).
    stability_mask : jnp.ndarray, shape (n_trials,)
        Boolean mask from Layer 2 clamping.
    trial_mask : jnp.ndarray, shape (n_trials,)
        External trial mask (for variable-length cohorts).

    Returns
    -------
    p_choice : jnp.ndarray, shape (n_trials,)
        P(choice_observed) per trial.  Unstable or masked trials get
        ``NaN`` to distinguish them from legitimate low probabilities.
    """
    # expected_mean from binary INPUT_NODES (0, 2, 4)
    mu1 = jnp.stack(
        [
            node_traj[0]["expected_mean"],
            node_traj[2]["expected_mean"],
            node_traj[4]["expected_mean"],
        ],
        axis=1,
    )

    # Softmax-stickiness (identical to _compute_logp in hierarchical.py)
    prev = jnp.concatenate([jnp.array([-1]), choices_jax[:-1]])
    stick = (prev[:, None] == jnp.arange(3)[None, :]).astype(jnp.float32)
    logits = beta * mu1 + zeta * stick
    probs = jax.nn.softmax(logits, axis=1)
    per_trial_p = probs[jnp.arange(n_trials), choices_jax]

    # Mark unstable or masked trials with NaN
    valid = stability_mask.astype(jnp.float32) * trial_mask.astype(
        jnp.float32
    )
    per_trial_p = jnp.where(valid > 0.5, per_trial_p, jnp.nan)

    return per_trial_p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def posterior_predictive_replay(
    idata: az.InferenceData,
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    model_name: str = "hgf_3level",
    n_draws: int | None = None,
    trial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Replay posterior draws through the HGF to compute P(choice_observed).

    For each posterior draw, injects the draw's parameter values into the
    HGF network, runs a clamped forward scan over the observed input
    sequence (no re-simulation of choices/rewards), and computes the
    softmax-stickiness probability of the actually-observed choice on
    each trial.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted inference data with a ``posterior`` group containing
        parameter arrays with dimensions ``(chain, draw, ...)``.
    input_data_arr : numpy.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays for all participants.
    observed_arr : numpy.ndarray, shape (P, n_trials, 3)
        Binary observed masks for all participants.
    choices_arr : numpy.ndarray, shape (P, n_trials)
        Chosen cue indices for all participants.
    model_name : str, optional
        Model variant: ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    n_draws : int or None, optional
        Number of posterior draws to use.  If ``None``, use all draws
        (flattened across chains).  If given, take the first ``n_draws``
        from the flattened chain x draw array.
    trial_mask : numpy.ndarray or None, shape (P, n_trials)
        Binary mask for variable-length cohorts.  ``1`` for real trials,
        ``0`` for padding.  Defaults to all-ones.

    Returns
    -------
    p_choice : numpy.ndarray, shape (n_draws_used, P, n_trials)
        P(choice_observed) for each draw, participant, and trial.
        Unstable or masked trials contain ``NaN``.

    Raises
    ------
    ValueError
        If ``model_name`` is not supported or array shapes are
        inconsistent.
    KeyError
        If required parameter variables are missing from ``idata.posterior``.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if model_name not in _MODEL_NAMES:
        msg = (
            f"model_name must be one of {_MODEL_NAMES}, "
            f"got {model_name!r}"
        )
        raise ValueError(msg)

    n_participants = input_data_arr.shape[0]
    n_trials = input_data_arr.shape[1]

    if observed_arr.shape[0] != n_participants:
        msg = (
            f"observed_arr leading dimension ({observed_arr.shape[0]}) "
            f"does not match input_data_arr ({n_participants})"
        )
        raise ValueError(msg)
    if choices_arr.shape[0] != n_participants:
        msg = (
            f"choices_arr leading dimension ({choices_arr.shape[0]}) "
            f"does not match input_data_arr ({n_participants})"
        )
        raise ValueError(msg)

    if trial_mask is None:
        trial_mask = np.ones((n_participants, n_trials), dtype=int)

    is_3level = model_name == "hgf_3level"

    # ------------------------------------------------------------------
    # Extract and flatten posterior draws
    # ------------------------------------------------------------------
    posterior = idata.posterior

    # Required parameters for all models
    omega_2_draws = posterior["omega_2"].values  # (chain, draw, P)
    beta_draws = posterior["beta"].values  # (chain, draw, P)
    zeta_draws = posterior["zeta"].values  # (chain, draw, P)

    # Flatten chains x draws -> (n_total_draws, P)
    n_chains, n_draws_per_chain = omega_2_draws.shape[:2]
    n_total = n_chains * n_draws_per_chain

    omega_2_flat = omega_2_draws.reshape(n_total, n_participants)
    beta_flat = beta_draws.reshape(n_total, n_participants)
    zeta_flat = zeta_draws.reshape(n_total, n_participants)

    if is_3level:
        omega_3_draws = posterior["omega_3"].values
        kappa_draws = posterior["kappa"].values
        omega_3_flat = omega_3_draws.reshape(n_total, n_participants)
        kappa_flat = kappa_draws.reshape(n_total, n_participants)

    # Subset draws if requested
    n_draws_used = n_total if n_draws is None else min(n_draws, n_total)
    omega_2_flat = omega_2_flat[:n_draws_used]
    beta_flat = beta_flat[:n_draws_used]
    zeta_flat = zeta_flat[:n_draws_used]

    if is_3level:
        omega_3_flat = omega_3_flat[:n_draws_used]
        kappa_flat = kappa_flat[:n_draws_used]

    # ------------------------------------------------------------------
    # Build network once to capture base_attrs and scan_fn
    # ------------------------------------------------------------------
    if is_3level:
        from prl_hgf.models.hgf_3level import build_3level_network

        net = build_3level_network()
    else:
        from prl_hgf.models.hgf_2level import build_2level_network

        net = build_2level_network()

    net.input_data(input_data=input_data_arr[0], observed=observed_arr[0])
    base_attrs = net.attributes
    scan_fn = net.scan_fn

    # Convert data to JAX arrays
    jax_input_data = jnp.array(input_data_arr, dtype=jnp.float32)
    jax_observed = jnp.array(observed_arr, dtype=jnp.int32)
    jax_choices = jnp.array(choices_arr, dtype=jnp.int32)
    jax_trial_mask = jnp.array(trial_mask, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Define per-participant PPC kernel
    # ------------------------------------------------------------------

    def _ppc_single_3level(
        omega_2: jnp.ndarray,
        omega_3: jnp.ndarray,
        kappa: jnp.ndarray,
        beta: jnp.ndarray,
        zeta: jnp.ndarray,
        input_data: jnp.ndarray,
        observed: jnp.ndarray,
        choices: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """PPC for a single participant with 3-level model."""
        scan_inputs = _build_scan_inputs(input_data, observed, n_trials)

        # Inject parameters (shallow-copy pattern from hierarchical.py)
        attrs = dict(base_attrs)

        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        node6 = dict(attrs[6])
        node6["tonic_volatility"] = omega_3
        node6["volatility_coupling_children"] = jnp.array(
            [kappa, kappa, kappa]
        )
        attrs[6] = node6

        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["volatility_coupling_parents"] = jnp.array([kappa])
            attrs[idx] = node

        _, (node_traj, stability_mask) = _clamped_scan(
            scan_fn, attrs, scan_inputs
        )

        return _compute_p_choice(
            node_traj,
            choices.astype(jnp.int32),
            n_trials,
            beta,
            zeta,
            stability_mask,
            mask,
        )

    def _ppc_single_2level(
        omega_2: jnp.ndarray,
        beta: jnp.ndarray,
        zeta: jnp.ndarray,
        input_data: jnp.ndarray,
        observed: jnp.ndarray,
        choices: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """PPC for a single participant with 2-level model."""
        scan_inputs = _build_scan_inputs(input_data, observed, n_trials)

        attrs = dict(base_attrs)
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        _, (node_traj, stability_mask) = _clamped_scan(
            scan_fn, attrs, scan_inputs
        )

        return _compute_p_choice(
            node_traj,
            choices.astype(jnp.int32),
            n_trials,
            beta,
            zeta,
            stability_mask,
            mask,
        )

    # ------------------------------------------------------------------
    # Build vmapped kernel: inner over participants, outer over draws
    # ------------------------------------------------------------------

    if is_3level:
        # vmap over participants (axis 0 of data arrays, scalar params)
        _vmap_participants = jax.vmap(
            _ppc_single_3level,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

        # Wrapper that takes full draw-level param vectors
        def _ppc_one_draw(
            o2: jnp.ndarray,
            o3: jnp.ndarray,
            kp: jnp.ndarray,
            bt: jnp.ndarray,
            zt: jnp.ndarray,
        ) -> jnp.ndarray:
            return _vmap_participants(
                o2, o3, kp, bt, zt,
                jax_input_data, jax_observed, jax_choices, jax_trial_mask,
            )

        # vmap over draws
        _vmap_draws = jax.vmap(
            _ppc_one_draw,
            in_axes=(0, 0, 0, 0, 0),
        )

        # Run all draws
        jax_omega_2 = jnp.array(omega_2_flat, dtype=jnp.float32)
        jax_omega_3 = jnp.array(omega_3_flat, dtype=jnp.float32)
        jax_kappa = jnp.array(kappa_flat, dtype=jnp.float32)
        jax_beta = jnp.array(beta_flat, dtype=jnp.float32)
        jax_zeta = jnp.array(zeta_flat, dtype=jnp.float32)

        result = _vmap_draws(
            jax_omega_2, jax_omega_3, jax_kappa, jax_beta, jax_zeta
        )

    else:
        _vmap_participants = jax.vmap(
            _ppc_single_2level,
            in_axes=(0, 0, 0, 0, 0, 0, 0),
        )

        def _ppc_one_draw(
            o2: jnp.ndarray,
            bt: jnp.ndarray,
            zt: jnp.ndarray,
        ) -> jnp.ndarray:
            return _vmap_participants(
                o2, bt, zt,
                jax_input_data, jax_observed, jax_choices, jax_trial_mask,
            )

        _vmap_draws = jax.vmap(
            _ppc_one_draw,
            in_axes=(0, 0, 0),
        )

        jax_omega_2 = jnp.array(omega_2_flat, dtype=jnp.float32)
        jax_beta = jnp.array(beta_flat, dtype=jnp.float32)
        jax_zeta = jnp.array(zeta_flat, dtype=jnp.float32)

        result = _vmap_draws(jax_omega_2, jax_beta, jax_zeta)

    # result shape: (n_draws_used, P, n_trials)
    return np.asarray(result)
