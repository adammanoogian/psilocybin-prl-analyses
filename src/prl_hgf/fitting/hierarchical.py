"""Batched hierarchical JAX logp Op for vmap'd cohort-level MCMC.

This module is the v1.2 replacement for the per-participant
:mod:`prl_hgf.fitting.ops` module.  It generalises the single-participant
logp into a ``jax.vmap``'d kernel that evaluates the entire cohort in a
single JAX call, amortising PCIe dispatch cost across all participants.

The per-participant math is **identical** to ``ops.py`` so that VALID-01
(Plan 12-04) can assert bit-exact agreement at ``P=1``.  The only additions
are:

1. **Layer 2 NaN clamping** — a tapas-style per-trial stability check
   inside the ``lax.scan`` step.  If any leaf in the updated attributes
   pytree is non-finite, or if any level-2 mean exceeds a magnitude bound
   (``|mu_2| < 14``, following ``tapas_ehgf_binary.m``), the belief state
   is reverted to the previous trial's values and the trial contributes
   ``0`` to the log-likelihood via a stability mask.

2. **trial_mask plumbing** — an optional ``(P, n_trials)`` binary array
   that zeros out logp contributions for padded trials, enabling future
   variable-length cohorts to reuse the compiled XLA kernel without
   recompilation.

3. **vmap reduction** — ``jax.vmap`` maps the per-participant logp across
   the participant dimension; the Op forward pass returns
   ``jnp.sum(per_participant_logps)`` as a scalar.

The two-Op split (``_BatchedLogpOp`` + ``_BatchedGradOp``) mirrors
``ops.py`` so that PyMC's gradient machinery works unchanged.  A
``@jax_funcify.register`` dispatch lets ``pmjax.sample_numpyro_nuts``
JAX-trace through the Op.

All HGF updates flow through ``pyhgf.Network.scan_fn`` — no HGF math is
reimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import lax
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

# Suppress PyTensor g++ compilation warning — not needed when Op.perform
# delegates entirely to JAX JIT.
pytensor.config.cxx = ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 means (``tapas_ehgf_binary.m``).
_MU_2_BOUND: float = 14.0

#: Supported model names.
_MODEL_NAMES: tuple[str, ...] = ("hgf_2level", "hgf_3level")

#: Node indices for the continuous-state level-1 belief nodes.
_BELIEF_NODES: tuple[int, ...] = (1, 3, 5)

#: Node indices for the binary-state input nodes.
_INPUT_NODES: tuple[int, ...] = (0, 2, 4)


# ---------------------------------------------------------------------------
# Per-participant logp builders (private)
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
    scan_fn,  # noqa: ANN001
    attrs: dict,
    scan_inputs: tuple,
) -> tuple[dict, tuple[dict, jnp.ndarray]]:
    """Run ``lax.scan`` with Layer 2 NaN-clamping wrapper.

    Wraps each step of the pyhgf ``scan_fn`` with a stability check.  If
    the updated attributes contain any non-finite values, or if any
    level-2 mean (``attrs[i]['mean']`` for ``i in {1, 3, 5}``) exceeds the
    magnitude bound ``_MU_2_BOUND``, the belief state is reverted to the
    previous trial's values.  The per-step stability flag is collected into
    a ``(n_trials,)`` boolean mask so that unstable trials contribute 0 to
    the log-likelihood downstream.

    All branching uses ``jnp.where`` / ``jax.tree_util.tree_map`` — no
    Python ``if`` on traced values — so the function stays compatible with
    ``jax.jit`` and ``jax.vmap``.

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
        Per-trial node trajectory (from the ``scan_fn`` second output).
    stability_mask : jnp.ndarray, shape (n_trials,)
        Boolean mask: ``True`` for stable trials, ``False`` for reverted
        trials.
    """

    def _clamped_step(
        carry: dict,
        x: tuple,
    ) -> tuple[dict, tuple[dict, jnp.ndarray]]:
        prev_attrs = carry
        new_attrs, new_node = scan_fn(prev_attrs, x)

        # Finiteness check across the entire pytree
        leaves = jax.tree_util.tree_leaves(new_attrs)
        all_finite = jnp.all(
            jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        )

        # Hard magnitude bound on level-2 means (tapas convention)
        mu_2_vals = jnp.array([
            new_attrs[1]["mean"],
            new_attrs[3]["mean"],
            new_attrs[5]["mean"],
        ])
        mu_2_ok = jnp.all(jnp.abs(mu_2_vals) < _MU_2_BOUND)

        is_stable = all_finite & mu_2_ok

        # Revert belief state on instability
        safe_attrs = jax.tree_util.tree_map(
            lambda n, o: jnp.where(is_stable, n, o),
            new_attrs,
            prev_attrs,
        )

        # Pass through node trajectory unchanged; the stability mask will
        # zero out the logp contribution of unstable trials downstream.
        return safe_attrs, (new_node, is_stable)

    final_attrs, (node_traj, stability_mask) = lax.scan(
        _clamped_step, attrs, scan_inputs
    )

    return final_attrs, (node_traj, stability_mask)


def _compute_logp(
    node_traj: dict,
    choices_jax: jnp.ndarray,
    n_trials: int,
    beta: jnp.ndarray,
    zeta: jnp.ndarray,
    stability_mask: jnp.ndarray,
    trial_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute softmax-stickiness log-likelihood from node trajectories.

    Replicates the logp computation from ``ops.py`` exactly, with the
    addition of Layer 2 stability masking and trial masking.

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
    logp : jnp.ndarray
        Scalar log-likelihood with ``-jnp.inf`` sentinel for NaN results.
    """
    # expected_mean from binary INPUT_NODES (0, 2, 4) — sigmoid P in [0,1]
    mu1 = jnp.stack(
        [
            node_traj[0]["expected_mean"],
            node_traj[2]["expected_mean"],
            node_traj[4]["expected_mean"],
        ],
        axis=1,
    )

    # Softmax-stickiness log-likelihood (identical to ops.py)
    prev = jnp.concatenate([jnp.array([-1]), choices_jax[:-1]])
    stick = (prev[:, None] == jnp.arange(3)[None, :]).astype(jnp.float32)
    logits = beta * mu1 + zeta * stick
    lp = jax.nn.log_softmax(logits, axis=1)
    per_trial_logp = lp[jnp.arange(n_trials), choices_jax]

    # Layer 2 mask: unstable trials contribute 0
    per_trial_logp = per_trial_logp * stability_mask.astype(
        per_trial_logp.dtype
    )

    # External trial mask: padded trials contribute 0
    per_trial_logp = per_trial_logp * trial_mask.astype(per_trial_logp.dtype)

    result = jnp.sum(per_trial_logp)

    # Layer 3 sentinel: NaN → -inf (same as ops.py)
    return jnp.where(jnp.isnan(result), -jnp.inf, result)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_logp_ops_batched(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    model_name: str = "hgf_3level",
    trial_mask: np.ndarray | None = None,
) -> tuple[Op, int, int]:
    """Build a batched JAX logp Op for cohort-level hierarchical fitting.

    Constructs a PyTensor Op whose forward pass ``jax.vmap``'s a
    per-participant logp across the participant dimension and reduces to a
    scalar via ``jnp.sum``.  The per-participant logp reuses pyhgf's
    ``Network.scan_fn`` with tapas-style Layer 2 NaN clamping.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays for all participants.
    observed_arr : numpy.ndarray, shape (P, n_trials, 3)
        Binary observed masks for all participants.
    choices_arr : numpy.ndarray, shape (P, n_trials)
        Chosen cue indices for all participants.
    model_name : str, optional
        Model variant: ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    trial_mask : numpy.ndarray or None, shape (P, n_trials)
        Binary mask for variable-length cohorts.  ``1`` for real trials,
        ``0`` for padding.  Defaults to all-ones.

    Returns
    -------
    logp_op : Op
        PyTensor Op accepting K parameter vectors of shape ``(P,)`` and
        returning a scalar log-likelihood.
    n_participants : int
        Number of participants ``P``.
    n_trials : int
        Number of trials per participant.

    Raises
    ------
    ValueError
        If ``model_name`` is not in ``_MODEL_NAMES`` or if the leading
        dimensions of ``input_data_arr``, ``observed_arr``, and
        ``choices_arr`` do not match.

    Notes
    -----
    The Op signature depends on ``model_name``:

    * ``"hgf_2level"``: ``op(omega_2, beta, zeta) -> scalar``
      where each argument has shape ``(P,)``.
    * ``"hgf_3level"``: ``op(omega_2, omega_3, kappa, beta, zeta) -> scalar``
      where each argument has shape ``(P,)``.

    The two-Op split (``_BatchedLogpOp`` + ``_BatchedGradOp``) mirrors the
    pattern in :mod:`prl_hgf.fitting.ops` so that PyMC's gradient machinery
    works unchanged.  The ``@jax_funcify.register`` dispatch lets
    ``pmjax.sample_numpyro_nuts`` JAX-trace through the Op.
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

    n_trials = input_data_arr.shape[1]

    if trial_mask is None:
        trial_mask = np.ones((n_participants, n_trials), dtype=int)

    # ------------------------------------------------------------------
    # Build network once to capture base_attrs and scan_fn
    # ------------------------------------------------------------------
    is_3level = model_name == "hgf_3level"

    if is_3level:
        from prl_hgf.models.hgf_3level import build_3level_network

        net = build_3level_network()
    else:
        from prl_hgf.models.hgf_2level import build_2level_network

        net = build_2level_network()

    # Seed with first participant's data to create scan_fn
    net.input_data(
        input_data=input_data_arr[0], observed=observed_arr[0]
    )
    base_attrs = net.attributes
    scan_fn = net.scan_fn

    # Convert data to JAX arrays
    jax_input_data = jnp.array(input_data_arr, dtype=jnp.float32)
    jax_observed = jnp.array(observed_arr, dtype=jnp.int32)
    jax_choices = jnp.array(choices_arr, dtype=jnp.int32)
    jax_trial_mask = jnp.array(trial_mask, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Per-participant logp function (data as runtime arguments)
    # ------------------------------------------------------------------
    # Define model-specific single-participant logp closures.  They share
    # base_attrs and scan_fn via closure, receive per-participant data and
    # parameters at call time.

    def _single_logp_3level(
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
        scan_inputs = _build_scan_inputs(input_data, observed, n_trials)

        # Inject parameters (shallow-copy pattern from ops.py)
        attrs = dict(base_attrs)

        # omega_2 into level-1 belief nodes (1, 3, 5)
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        # omega_3 and kappa children-side into volatility node 6
        node6 = dict(attrs[6])
        node6["tonic_volatility"] = omega_3
        node6["volatility_coupling_children"] = jnp.array(
            [kappa, kappa, kappa]
        )
        attrs[6] = node6

        # kappa parents-side into nodes 1, 3, 5
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["volatility_coupling_parents"] = jnp.array([kappa])
            attrs[idx] = node

        # Clamped scan
        _, (node_traj, stability_mask) = _clamped_scan(
            scan_fn, attrs, scan_inputs
        )

        return _compute_logp(
            node_traj,
            choices.astype(jnp.int32),
            n_trials,
            beta,
            zeta,
            stability_mask,
            mask,
        )

    def _single_logp_2level(
        omega_2: jnp.ndarray,
        beta: jnp.ndarray,
        zeta: jnp.ndarray,
        input_data: jnp.ndarray,
        observed: jnp.ndarray,
        choices: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        scan_inputs = _build_scan_inputs(input_data, observed, n_trials)

        # Inject parameters (shallow-copy pattern from ops.py)
        attrs = dict(base_attrs)
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        # Clamped scan
        _, (node_traj, stability_mask) = _clamped_scan(
            scan_fn, attrs, scan_inputs
        )

        return _compute_logp(
            node_traj,
            choices.astype(jnp.int32),
            n_trials,
            beta,
            zeta,
            stability_mask,
            mask,
        )

    if is_3level:
        n_params = 5  # omega_2, omega_3, kappa, beta, zeta
        _single_participant_logp = _single_logp_3level  # type: ignore[assignment]
    else:
        n_params = 3  # omega_2, beta, zeta
        _single_participant_logp = _single_logp_2level  # type: ignore[assignment]
    param_argnums = tuple(range(n_params))

    # ------------------------------------------------------------------
    # vmap'd batched logp
    # ------------------------------------------------------------------
    # in_axes: all arguments vmapped over axis 0
    _batched_logp = jax.vmap(
        _single_participant_logp,
        in_axes=tuple(
            [0] * (n_params + 4)  # params + input_data, observed, choices, mask
        ),
    )

    def _jax_logp_batched(*param_arrays: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the batched logp: sum across participants.

        Parameters
        ----------
        *param_arrays : jnp.ndarray
            K parameter arrays, each of shape ``(P,)``.

        Returns
        -------
        jnp.ndarray
            Scalar total log-likelihood.
        """
        per_participant = _batched_logp(  # type: ignore[call-arg]
            *param_arrays,
            jax_input_data,
            jax_observed,
            jax_choices,
            jax_trial_mask,
        )
        return jnp.sum(per_participant)

    # ------------------------------------------------------------------
    # Two-Op split (mirrors ops.py)
    # ------------------------------------------------------------------
    _jit_val_grad = jax.jit(
        jax.value_and_grad(_jax_logp_batched, argnums=param_argnums)
    )
    _jit_logp = jax.jit(_jax_logp_batched)

    class _BatchedGradOp(Op):
        """Return gradients of batched logp w.r.t. parameter vectors."""

        def make_node(self, *inputs):  # noqa: ANN002
            tensor_inputs = [
                pt.as_tensor_variable(x) for x in inputs
            ]
            return Apply(
                self,
                tensor_inputs,
                [inp.type() for inp in tensor_inputs],
            )

        def perform(self, node, inputs, outputs):  # noqa: ANN001
            (_, grads) = _jit_val_grad(
                *[np.asarray(x, dtype=np.float64) for x in inputs]
            )
            for i, g in enumerate(grads):
                outputs[i][0] = np.asarray(
                    g, dtype=node.outputs[i].dtype
                )

    _grad_op = _BatchedGradOp()

    class _BatchedLogpOp(Op):
        """Forward batched logp Op; delegates gradients to _BatchedGradOp."""

        def make_node(self, *inputs):  # noqa: ANN002
            tensor_inputs = [
                pt.as_tensor_variable(x) for x in inputs
            ]
            return Apply(
                self,
                tensor_inputs,
                [pt.scalar(dtype="float64")],
            )

        def perform(self, node, inputs, outputs):  # noqa: ANN001
            outputs[0][0] = np.asarray(
                _jit_logp(
                    *[np.asarray(x, dtype=np.float64) for x in inputs]
                ),
                dtype=np.float64,
            )

        def grad(self, inputs, output_gradients):  # noqa: ANN001
            grads = _grad_op(*inputs)
            og = output_gradients[0]
            return [og * g for g in grads]  # type: ignore[union-attr]

    # Register JAX dispatch so sample_numpyro_nuts can convert this Op
    @jax_funcify.register(_BatchedLogpOp)
    def _logp_op_jax(op, **kwargs):  # noqa: ANN001, ANN003, ARG001
        fn = _jax_logp_batched

        def impl(*args):  # noqa: ANN002
            return fn(*args)

        return impl

    return _BatchedLogpOp(), n_participants, n_trials



# ---------------------------------------------------------------------------
# Hierarchical PyMC model factory
# ---------------------------------------------------------------------------


def build_pymc_model_batched(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    model_name: str = "hgf_3level",
    trial_mask: np.ndarray | None = None,
) -> tuple:
    """Build a hierarchical PyMC model with shape=(P,) IID priors.

    Constructs a PyMC model where every free parameter has
    ``shape=n_participants`` — one independent prior per participant,
    with **no hyperpriors and no partial pooling**.  This gives identical
    statistical semantics to v1.1's per-participant loop but packs
    everything into a single model graph so that one
    ``pmjax.sample_numpyro_nuts`` call fits the entire cohort.

    The ``shape=(P,)`` trick exploits the fact that PyMC's IID priors
    with no plate-level coupling are mathematically equivalent to P
    independent models.  The only difference is that NUTS explores all
    P posteriors in one joint step, amortising launch overhead.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays for all participants.
    observed_arr : numpy.ndarray, shape (P, n_trials, 3)
        Binary observed masks for all participants.
    choices_arr : numpy.ndarray, shape (P, n_trials)
        Chosen cue indices for all participants.
    model_name : str, optional
        Model variant: ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    trial_mask : numpy.ndarray or None, shape (P, n_trials)
        Binary mask for variable-length cohorts.  Defaults to all-ones.

    Returns
    -------
    model : pymc.Model
        Compiled PyMC model with IID priors and ``pm.Potential`` hook.
    var_names : list[str]
        Names of the free parameters for ``az.summary``.
    n_participants : int
        Number of participants ``P``.

    Raises
    ------
    ValueError
        If ``input_data_arr`` is not 3-dimensional or ``model_name`` is
        not recognised.
    """
    import pymc as pm

    if input_data_arr.ndim != 3:
        msg = (
            f"input_data_arr must be 3-dimensional (P, n_trials, 3), "
            f"got ndim={input_data_arr.ndim}"
        )
        raise ValueError(msg)

    if model_name not in _MODEL_NAMES:
        msg = (
            f"model_name must be one of {_MODEL_NAMES}, "
            f"got {model_name!r}"
        )
        raise ValueError(msg)

    n_participants = input_data_arr.shape[0]

    logp_op, _P, _T = build_logp_ops_batched(
        input_data_arr,
        observed_arr,
        choices_arr,
        model_name=model_name,
        trial_mask=trial_mask,
    )

    with pm.Model() as model:
        if model_name == "hgf_2level":
            # Perceptual parameter: tonic volatility (must be < 0)
            omega_2 = pm.TruncatedNormal(
                "omega_2", mu=-3.0, sigma=2.0, upper=0.0,
                shape=n_participants,
            )

            # Response parameters
            log_beta = pm.Normal(
                "log_beta", mu=0.0, sigma=1.5,
                shape=n_participants,
            )
            beta = pm.Deterministic(
                "beta", pm.math.exp(log_beta),
            )
            zeta = pm.Normal(
                "zeta", mu=0.0, sigma=2.0,
                shape=n_participants,
            )

            pm.Potential("loglike", logp_op(omega_2, beta, zeta))
            var_names = ["omega_2", "beta", "zeta"]

        else:
            # Perceptual parameters
            omega_2 = pm.TruncatedNormal(
                "omega_2", mu=-3.0, sigma=2.0, upper=0.0,
                shape=n_participants,
            )
            omega_3 = pm.TruncatedNormal(
                "omega_3", mu=-6.0, sigma=2.0, upper=0.0,
                shape=n_participants,
            )
            kappa = pm.TruncatedNormal(
                "kappa", mu=1.0, sigma=0.5, lower=0.01, upper=2.0,
                shape=n_participants,
            )

            # Response parameters
            log_beta = pm.Normal(
                "log_beta", mu=0.0, sigma=1.5,
                shape=n_participants,
            )
            beta = pm.Deterministic(
                "beta", pm.math.exp(log_beta),
            )
            zeta = pm.Normal(
                "zeta", mu=0.0, sigma=2.0,
                shape=n_participants,
            )

            pm.Potential(
                "loglike",
                logp_op(omega_2, omega_3, kappa, beta, zeta),
            )
            var_names = ["omega_2", "omega_3", "kappa", "beta", "zeta"]

    return model, var_names, n_participants


__all__ = ["build_logp_ops_batched", "build_pymc_model_batched"]
