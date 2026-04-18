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

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import lax
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd

# Suppress PyTensor g++ compilation warning — not needed when Op.perform
# delegates entirely to JAX JIT.
pytensor.config.cxx = ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 means (``tapas_ehgf_binary.m``).
_MU_2_BOUND: float = 14.0

#: Frozen volatility-coupling strength for 3-level HGF fitting.
#:
#: The ω₃ × κ product is multiplicatively confounded in the HGF likelihood
#: (high ω₃ + low κ ≈ low ω₃ + high κ for observable learning-rate
#: dynamics), producing a curved ridge that diagonal-mass-matrix NUTS
#: cannot navigate without saturating ``max_tree_depth``.  Freezing κ at
#: 1.0 collapses the ridge to a line; matches the TAPAS convention where
#: κ is fixed in most PRL/volatility applications (Mathys 2011).  The
#: batched logp function still accepts κ as an argument for legacy
#: simulator compatibility — callers in the fitting path always pass 1.0.
_KAPPA_FIXED: float = 1.0

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
        mu_2_vals = jnp.array(
            [
                new_attrs[1]["mean"],
                new_attrs[3]["mean"],
                new_attrs[5]["mean"],
            ]
        )
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
    per_trial_logp = per_trial_logp * stability_mask.astype(per_trial_logp.dtype)

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
        msg = f"model_name must be one of {_MODEL_NAMES}, got {model_name!r}"
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
    net.input_data(input_data=input_data_arr[0], observed=observed_arr[0])
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
        node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
        attrs[6] = node6

        # kappa parents-side into nodes 1, 3, 5
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["volatility_coupling_parents"] = jnp.array([kappa])
            attrs[idx] = node

        # Clamped scan
        _, (node_traj, stability_mask) = _clamped_scan(scan_fn, attrs, scan_inputs)

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
        _, (node_traj, stability_mask) = _clamped_scan(scan_fn, attrs, scan_inputs)

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
            tensor_inputs = [pt.as_tensor_variable(x) for x in inputs]
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
                outputs[i][0] = np.asarray(g, dtype=node.outputs[i].dtype)

    _grad_op = _BatchedGradOp()

    class _BatchedLogpOp(Op):
        """Forward batched logp Op; delegates gradients to _BatchedGradOp."""

        def make_node(self, *inputs):  # noqa: ANN002
            tensor_inputs = [pt.as_tensor_variable(x) for x in inputs]
            return Apply(
                self,
                tensor_inputs,
                [pt.scalar(dtype="float64")],
            )

        def perform(self, node, inputs, outputs):  # noqa: ANN001
            outputs[0][0] = np.asarray(
                _jit_logp(*[np.asarray(x, dtype=np.float64) for x in inputs]),
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
# Pure JAX logp factory (numpyro-direct path)
# ---------------------------------------------------------------------------


def build_logp_fn_batched(
    model_name: str = "hgf_3level",
    n_trials: int = 100,
) -> tuple:
    """Build a pure JAX batched logp function with data as arguments.

    Unlike :func:`build_logp_ops_batched`, the returned callable does **not**
    capture data in a closure.  Data arrays are explicit arguments, making the
    XLA trace shape-dependent but value-independent.  This enables JIT cache
    reuse across power-sweep iterations with different data.

    The only values captured via closure are the *static* model structure:
    ``base_attrs``, ``scan_fn``, and ``n_trials``.

    Parameters
    ----------
    model_name : str, optional
        Model variant: ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    n_trials : int, optional
        Number of trials per participant.  Used to build the pyhgf
        ``Network`` once and to size the scan inputs.

    Returns
    -------
    batched_logp_fn : callable
        For 3-level: ``(omega_2, omega_3, kappa, beta, zeta,
        input_data, observed, choices, trial_mask) -> scalar``.
        For 2-level: ``(omega_2, beta, zeta,
        input_data, observed, choices, trial_mask) -> scalar``.
    n_params : int
        Number of model parameters (5 for 3-level, 3 for 2-level).

    Raises
    ------
    ValueError
        If ``model_name`` is not in ``_MODEL_NAMES``.
    """
    if model_name not in _MODEL_NAMES:
        msg = f"model_name must be one of {_MODEL_NAMES}, got {model_name!r}"
        raise ValueError(msg)

    is_3level = model_name == "hgf_3level"

    # Build network once to capture base_attrs and scan_fn (static)
    if is_3level:
        from prl_hgf.models.hgf_3level import build_3level_network

        net = build_3level_network()
    else:
        from prl_hgf.models.hgf_2level import build_2level_network

        net = build_2level_network()

    dummy_input = np.zeros((n_trials, 3), dtype=float)
    dummy_obs = np.zeros((n_trials, 3), dtype=int)
    net.input_data(input_data=dummy_input, observed=dummy_obs)
    base_attrs = net.attributes
    scan_fn = net.scan_fn

    # Per-participant logp: same math as build_logp_ops_batched closures
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
        attrs = dict(base_attrs)
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node
        node6 = dict(attrs[6])
        node6["tonic_volatility"] = omega_3
        node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
        attrs[6] = node6
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["volatility_coupling_parents"] = jnp.array([kappa])
            attrs[idx] = node
        _, (node_traj, stability_mask) = _clamped_scan(scan_fn, attrs, scan_inputs)
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
        attrs = dict(base_attrs)
        for idx in _BELIEF_NODES:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node
        _, (node_traj, stability_mask) = _clamped_scan(scan_fn, attrs, scan_inputs)
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
        n_params = 5
        _single_participant_logp = _single_logp_3level  # type: ignore[assignment]
    else:
        n_params = 3
        _single_participant_logp = _single_logp_2level  # type: ignore[assignment]

    _batched_logp = jax.vmap(
        _single_participant_logp,
        in_axes=tuple([0] * (n_params + 4)),
    )

    def batched_logp_fn(
        *args: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate batched logp: sum across participants.

        Parameters
        ----------
        *args : jnp.ndarray
            K parameter arrays of shape ``(P,)`` followed by
            ``input_data (P, T, 3)``, ``observed (P, T, 3)``,
            ``choices (P, T)``, ``trial_mask (P, T)``.

        Returns
        -------
        jnp.ndarray
            Scalar total log-likelihood.
        """
        per_participant = _batched_logp(*args)
        return jnp.sum(per_participant)

    return batched_logp_fn, n_params


# ---------------------------------------------------------------------------
# BlackJAX log-posterior and sampling helpers
# ---------------------------------------------------------------------------


def _build_log_posterior(
    batched_logp_fn,  # noqa: ANN001
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    model_name: str = "hgf_3level",
) -> callable:
    """Build a pure JAX log-posterior function for BlackJAX.

    Combines independent priors (via ``numpyro.distributions``) with the
    batched HGF log-likelihood from :func:`build_logp_fn_batched` into a
    single ``logdensity_fn(params_dict) -> scalar`` callable suitable for
    BlackJAX NUTS.

    Data arrays are captured in the closure (fixed shape per call),
    enabling JIT cache reuse across MCMC steps.

    Parameters
    ----------
    batched_logp_fn : callable
        Pure JAX logp from :func:`build_logp_fn_batched`.
    input_data : jnp.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays.
    observed : jnp.ndarray, shape (P, n_trials, 3)
        Binary observed masks.
    choices : jnp.ndarray, shape (P, n_trials)
        Chosen cue indices.
    trial_mask : jnp.ndarray, shape (P, n_trials)
        Binary trial mask for variable-length cohorts.
    n_participants : int
        Number of participants ``P``.
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).

    Returns
    -------
    logdensity_fn : callable
        ``dict[str, jnp.ndarray] -> scalar``.  Keys match the model
        parameter names; each value has shape ``(P,)``.
    """
    import numpyro.distributions as dist

    is_3level = model_name == "hgf_3level"

    # Define priors matching _numpyro_model_3level / _numpyro_model_2level
    prior_omega_2 = dist.TruncatedNormal(
        loc=-3.0,
        scale=2.0,
        high=0.0,
    )
    prior_log_beta = dist.Normal(0.0, 1.5)
    prior_zeta = dist.Normal(0.0, 2.0)
    if is_3level:
        prior_omega_3 = dist.TruncatedNormal(
            loc=-6.0,
            scale=2.0,
            high=0.0,
        )
        # κ is frozen at _KAPPA_FIXED (1.0) — see module docstring.  No prior
        # needed because κ is not sampled.

    def logdensity_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute log-posterior: prior + likelihood.

        Parameters
        ----------
        params : dict[str, jnp.ndarray]
            Parameter dict with keys matching the model.  Each value
            has shape ``(P,)``.

        Returns
        -------
        jnp.ndarray
            Scalar log-posterior.
        """
        omega_2 = params["omega_2"]
        log_beta = params["log_beta"]
        beta = jnp.exp(log_beta)
        zeta = params["zeta"]

        # Sum prior logp across participants
        prior_lp = jnp.sum(prior_omega_2.log_prob(omega_2))
        prior_lp = prior_lp + jnp.sum(
            prior_log_beta.log_prob(log_beta),
        )
        prior_lp = prior_lp + jnp.sum(prior_zeta.log_prob(zeta))

        if is_3level:
            omega_3 = params["omega_3"]
            prior_lp = prior_lp + jnp.sum(
                prior_omega_3.log_prob(omega_3),
            )
            # κ frozen at 1.0 — pass as broadcast constant to batched logp.
            kappa = jnp.full_like(omega_2, _KAPPA_FIXED)
            likelihood_lp = batched_logp_fn(
                omega_2,
                omega_3,
                kappa,
                beta,
                zeta,
                input_data,
                observed,
                choices,
                trial_mask,
            )
        else:
            likelihood_lp = batched_logp_fn(
                omega_2,
                beta,
                zeta,
                input_data,
                observed,
                choices,
                trial_mask,
            )

        return prior_lp + likelihood_lp

    return logdensity_fn


def _extract_nuts_stats(
    infos,  # blackjax NUTSInfo pytree
    transpose: bool,
) -> dict[str, np.ndarray]:
    """Convert a stacked NUTSInfo pytree into an ArviZ-friendly dict.

    Carries ``diverging``, ``acceptance_rate``, ``num_integration_steps``,
    ``num_trajectory_expansions``, and ``energy`` so downstream diagnostics
    can measure per-draw integrator work, not just acceptance.

    Parameters
    ----------
    infos : blackjax.mcmc.nuts.NUTSInfo
        Stacked info output from a sampling scan.
    transpose : bool
        True when leading axes are ``(n_draws, n_chains)`` (vmap scan
        layout) and need to be swapped to ``(n_chains, n_draws)`` for
        ArviZ.  False when already ``(n_chains, n_draws)`` (pmap layout).

    Returns
    -------
    dict[str, numpy.ndarray]
        Each value has shape ``(n_chains, n_draws)``.
    """

    def _to_np(x: jnp.ndarray) -> np.ndarray:
        if transpose:
            return np.asarray(jnp.transpose(x, (1, 0)))
        return np.asarray(x)

    return {
        "diverging": _to_np(infos.is_divergent),
        "acceptance_rate": _to_np(infos.acceptance_rate),
        "num_integration_steps": _to_np(infos.num_integration_steps),
        "num_trajectory_expansions": _to_np(infos.num_trajectory_expansions),
        "energy": _to_np(infos.energy),
    }


def _run_blackjax_nuts(
    logdensity_fn,  # noqa: ANN001
    initial_position: dict[str, jnp.ndarray],
    rng_key: jnp.ndarray,
    n_tune: int = 1000,
    n_draws: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    batched_logp_fn=None,  # noqa: ANN001
    input_data: jnp.ndarray | None = None,
    observed: jnp.ndarray | None = None,
    choices: jnp.ndarray | None = None,
    trial_mask: jnp.ndarray | None = None,
    model_name: str = "hgf_3level",
    warmup_params: dict | None = None,
    log_every: int = 0,
    phase_label: str = "sample",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, dict]:
    """Run BlackJAX NUTS with window_adaptation warmup and lax.scan sampling.

    The warmup phase uses the closure-based ``logdensity_fn`` from
    :func:`_build_log_posterior` (runs once, no cache benefit needed).
    The sampling phase uses :func:`_build_sample_loop` to pass data
    arrays as traced JIT arguments, enabling persistent XLA compilation
    cache hits across power-sweep iterations with same-shape data.

    When ``warmup_params`` is provided, the warmup phase is skipped
    entirely and the pre-adapted step size and mass matrix are used
    directly.  This enables reusing adapted parameters across
    power-sweep iterations at the same sample size, avoiding the
    ~1100s warmup compilation cost on iterations 2+.

    Parameters
    ----------
    logdensity_fn : callable
        ``dict -> scalar`` log-posterior from :func:`_build_log_posterior`.
        Used for warmup only (ignored when ``warmup_params`` is provided).
    initial_position : dict[str, jnp.ndarray]
        Starting values for each parameter.  Each value has shape ``(P,)``.
        Used as warmup starting point and as the base state for sampling
        when ``warmup_params`` is provided.
    rng_key : jnp.ndarray
        JAX PRNGKey.
    n_tune : int, optional
        Number of warmup steps.  Default ``1000``.  Ignored when
        ``warmup_params`` is provided.
    n_draws : int, optional
        Number of posterior draws per chain.  Default ``1000``.
    n_chains : int, optional
        Number of MCMC chains.  Default ``4``.
    target_accept : float, optional
        Target acceptance rate for NUTS.  Default ``0.95``.  Ignored
        when ``warmup_params`` is provided.
    batched_logp_fn : callable or None, optional
        Pure JAX batched logp from :func:`build_logp_fn_batched`.  Required
        for the traced-arg sampling loop.  If ``None``, falls back to the
        closure-based chain runners.
    input_data : jnp.ndarray or None, optional
        Float reward-value arrays, shape ``(P, n_trials, 3)``.
    observed : jnp.ndarray or None, optional
        Binary observed masks, shape ``(P, n_trials, 3)``.
    choices : jnp.ndarray or None, optional
        Chosen cue indices, shape ``(P, n_trials)``.
    trial_mask : jnp.ndarray or None, optional
        Binary trial mask, shape ``(P, n_trials)``.
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).  Controls prior
        structure inside the traced-arg sampling loop.
    warmup_params : dict or None, optional
        Pre-adapted NUTS parameters (``step_size`` and
        ``inverse_mass_matrix``) from a previous call.  When provided,
        warmup is skipped entirely — saves ~1100s per call in the
        power sweep.  Obtain from the 4th element of this function's
        return tuple.

    Returns
    -------
    positions_dict : dict[str, numpy.ndarray]
        Parameter samples shaped ``(n_chains, n_draws, P)``.
    sample_stats_dict : dict[str, numpy.ndarray]
        ``"diverging"`` bool and ``"acceptance_rate"`` float, each
        shaped ``(n_chains, n_draws)``.
    n_chains_actual : int
        Actual number of chains used (may differ from ``n_chains`` if
        pmap path adjusts it).
    adapted_params : dict
        Adapted NUTS parameters (``step_size`` and
        ``inverse_mass_matrix``).  Pass back as ``warmup_params`` on
        subsequent calls to skip warmup.
    """
    import blackjax

    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    warmup_state = None  # Set by window_adaptation if warmup runs

    # Phase 1: Window adaptation (skip if pre-adapted params provided)
    if warmup_params is None:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=target_accept,
            is_mass_matrix_diagonal=True,
        )
        (warmup_state, warmup_params), _warmup_info = warmup.run(
            warmup_key,
            initial_position,
            num_steps=n_tune,
        )

    # Phase 2: Determine chain strategy
    n_devices = jax.device_count()
    use_pmap = n_devices >= n_chains

    # Phase 3: Sampling with traced-arg sample loop (or legacy fallback)
    _has_traced_args = (
        batched_logp_fn is not None
        and input_data is not None
        and observed is not None
        and choices is not None
        and trial_mask is not None
    )

    if _has_traced_args:
        # Traced-arg path: data flows as JIT arguments for cache reuse
        sample_loop = _build_sample_loop(
            batched_logp_fn,
            model_name,
            n_chains,
            n_draws,
            use_pmap,
            log_every=log_every,
            phase_label=phase_label,
        )

        # init_position: use adapted position from warmup if available,
        # otherwise the prior-mode initial_position passed in.  Either
        # way, sample_loop calls nuts.init() inside JIT — value_and_grad
        # of the closure-free logdensity is compiled with traced data
        # and cached.
        init_pos = (
            warmup_state.position if warmup_state is not None
            else initial_position
        )

        all_states, all_infos = sample_loop(
            init_pos,
            warmup_params,
            sample_key,
            input_data,
            observed,
            choices,
            trial_mask,
        )

        # Post-process: convert JAX arrays to numpy
        if use_pmap:
            # pmap: (n_chains, n_draws, P) -- already correct layout
            positions_dict = {k: np.asarray(v) for k, v in all_states.position.items()}
            stats_dict = _extract_nuts_stats(all_infos, transpose=False)
        else:
            # vmap: (n_draws, n_chains, P) -> (n_chains, n_draws, P)
            positions_dict = {
                k: np.asarray(jnp.transpose(v, (1, 0, 2)))
                for k, v in all_states.position.items()
            }
            stats_dict = _extract_nuts_stats(all_infos, transpose=True)

        return positions_dict, stats_dict, n_chains, warmup_params

    # Legacy fallback: closure-based chain runners
    nuts = blackjax.nuts(logdensity_fn, **warmup_params)

    # Build warmup_state if it wasn't created by window_adaptation
    # (happens when warmup_params was provided to skip warmup)
    if warmup_state is None:
        warmup_state = nuts.init(initial_position)

    if use_pmap:
        positions, stats, n_actual = _run_pmap_chains(
            nuts,
            warmup_state,
            sample_key,
            n_draws,
            n_chains,
        )
    else:
        positions, stats, n_actual = _run_vmap_chains(
            nuts,
            warmup_state,
            sample_key,
            n_draws,
            n_chains,
        )

    return positions, stats, n_actual, warmup_params


def _run_vmap_chains(
    nuts,  # noqa: ANN001
    warmup_state,  # noqa: ANN001
    sample_key: jnp.ndarray,
    n_draws: int,
    n_chains: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Run multiple MCMC chains via vmap on a single device.

    Parameters
    ----------
    nuts : blackjax.mcmc.nuts.SamplingAlgorithm
        Configured NUTS kernel with adapted parameters.
    warmup_state : blackjax NUTSState
        Adapted state from warmup.
    sample_key : jnp.ndarray
        JAX PRNGKey for sampling.
    n_draws : int
        Number of posterior draws per chain.
    n_chains : int
        Number of chains to run.

    Returns
    -------
    positions_dict : dict[str, numpy.ndarray]
        Samples shaped ``(n_chains, n_draws, P)``.
    sample_stats_dict : dict[str, numpy.ndarray]
        Diagnostics shaped ``(n_chains, n_draws)``.
    n_chains_actual : int
        Number of chains (equals ``n_chains``).
    """
    chain_keys = jax.random.split(sample_key, n_chains)

    # Replicate warmup state across chains
    replicated_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
        warmup_state,
    )

    @jax.jit
    def _one_step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        new_states, infos = jax.vmap(nuts.step)(keys, states)
        return new_states, (new_states, infos)

    # Generate per-draw RNG keys
    draw_keys = jax.random.split(chain_keys[0], n_draws)

    _, (all_states, all_infos) = lax.scan(
        _one_step,
        replicated_state,
        draw_keys,
    )

    # all_states.position: dict of (n_draws, n_chains, P)
    # Transpose to (n_chains, n_draws, P) for ArviZ
    positions_dict = {
        k: np.asarray(jnp.transpose(v, (1, 0, 2)))
        for k, v in all_states.position.items()
    }

    # Diagnostics: (n_draws, n_chains) -> (n_chains, n_draws)
    stats_dict = _extract_nuts_stats(all_infos, transpose=True)

    return positions_dict, stats_dict, n_chains


def _run_pmap_chains(
    nuts,  # noqa: ANN001
    warmup_state,  # noqa: ANN001
    sample_key: jnp.ndarray,
    n_draws: int,
    n_chains: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Run multiple MCMC chains via pmap across devices.

    Parameters
    ----------
    nuts : blackjax.mcmc.nuts.SamplingAlgorithm
        Configured NUTS kernel with adapted parameters.
    warmup_state : blackjax NUTSState
        Adapted state from warmup.
    sample_key : jnp.ndarray
        JAX PRNGKey for sampling.
    n_draws : int
        Number of posterior draws per chain.
    n_chains : int
        Number of chains to run (one per device).

    Returns
    -------
    positions_dict : dict[str, numpy.ndarray]
        Samples shaped ``(n_chains, n_draws, P)``.
    sample_stats_dict : dict[str, numpy.ndarray]
        Diagnostics shaped ``(n_chains, n_draws)``.
    n_chains_actual : int
        Number of chains (equals ``n_chains``).
    """
    chain_keys = jax.random.split(sample_key, n_chains)

    # Replicate warmup state across devices
    replicated_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
        warmup_state,
    )

    def _sample_one_chain(
        rng_key: jnp.ndarray,
        state,  # noqa: ANN001
    ) -> tuple:
        @jax.jit
        def _one_step(s, k):  # noqa: ANN001
            new_s, info = nuts.step(k, s)
            return new_s, (new_s, info)

        keys = jax.random.split(rng_key, n_draws)
        _, (states, infos) = lax.scan(_one_step, state, keys)
        return states, infos

    # pmap: distribute chains across devices
    all_states, all_infos = jax.pmap(_sample_one_chain)(
        chain_keys,
        replicated_state,
    )

    # all_states.position: dict of (n_chains, n_draws, P) -- already correct
    positions_dict = {k: np.asarray(v) for k, v in all_states.position.items()}

    # Diagnostics: (n_chains, n_draws)
    stats_dict = _extract_nuts_stats(all_infos, transpose=False)

    return positions_dict, stats_dict, n_chains


def _build_progress_callback(
    log_every: int,
    n_chains: int,
    phase_label: str,
):  # noqa: ANN205
    """Build a host-side callback emitting NUTS progress every ``log_every`` draws.

    Called from inside ``jax.debug.callback`` so it runs on the host while
    the compiled scan continues on the accelerator.  Logs cumulative wall
    time, per-chain integration-step distribution, trajectory-expansion
    distribution, acceptance rate, and divergence count.

    Parameters
    ----------
    log_every : int
        Emit a line every ``log_every`` draws.  ``0`` disables.
    n_chains : int
        Number of chains, used for divergence denominator.
    phase_label : str
        Short tag included in each log line (e.g. ``"cold"``, ``"warm1"``).

    Returns
    -------
    callable or None
        Host callback ``(step_idx, int_steps, accept, div, expansions) -> None``.
        ``None`` if ``log_every <= 0``.
    """
    if log_every <= 0:
        return None

    import time

    start = [time.time()]
    last = [start[0]]

    def _cb(step_idx, int_steps, accept, div, expansions):  # noqa: ANN001
        step = int(step_idx)
        # Log draw 1, every log_every draws, and nothing else
        if step == 0:
            return
        if step % log_every != 0:
            return
        now = time.time()
        int_steps_np = np.asarray(int_steps)
        accept_np = np.asarray(accept)
        div_np = np.asarray(div)
        exp_np = np.asarray(expansions)
        elapsed = now - start[0]
        since_last = now - last[0]
        print(
            f"  [{phase_label}] draw {step:>5d} | "
            f"t={elapsed:7.1f}s Δ={since_last:5.1f}s | "
            f"int_steps mean={int_steps_np.mean():6.1f} "
            f"p50={int(np.percentile(int_steps_np, 50)):>4d} "
            f"max={int(int_steps_np.max()):>4d} | "
            f"exp mean={exp_np.mean():.1f} max={int(exp_np.max())} | "
            f"accept={accept_np.mean():.3f} "
            f"div={int(div_np.sum())}/{n_chains}",
            flush=True,
        )
        last[0] = now

    return _cb


def _build_sample_loop(
    batched_logp_fn,  # noqa: ANN001
    model_name: str,
    n_chains: int,
    n_draws: int,
    use_pmap: bool,
    log_every: int = 0,
    phase_label: str = "sample",
):  # noqa: ANN205
    """Build a JIT'd sampling function where data flows as traced arguments.

    This factory solves the XLA persistent compilation cache problem: when
    data arrays are captured in a closure, they become HLO constants.
    Different data values produce different HLO hashes, causing full
    recompilation (~1600s) on every power-sweep iteration even when shapes
    are identical.

    By making data arrays explicit function arguments, XLA traces them as
    shape-dependent placeholders.  Same shapes produce the same HLO hash,
    enabling persistent cache hits across iterations.

    The warmup phase (which runs once per call) still uses closure-based
    ``logdensity_fn`` from :func:`_build_log_posterior` -- no cache benefit
    is needed there.  Only the sampling phase (which we want to cache
    across power-sweep calls) uses the traced-arg pattern.

    Parameters
    ----------
    batched_logp_fn : callable
        Pure JAX batched logp from :func:`build_logp_fn_batched`.  Closes
        over ``base_attrs``, ``scan_fn``, ``n_trials`` (static per model
        shape -- safe to capture).
    model_name : str
        ``"hgf_2level"`` or ``"hgf_3level"``.  Controls prior structure.
    n_chains : int
        Number of MCMC chains.
    n_draws : int
        Number of posterior draws per chain.
    use_pmap : bool
        If ``True``, use ``jax.pmap`` for multi-GPU chain parallelism.
        If ``False``, use ``jax.vmap`` on a single device with
        ``@jax.jit``.

    Returns
    -------
    sample_loop : callable
        ``(init_position, warmup_params, sample_key, input_data, observed,
        choices, trial_mask) -> (all_states, all_infos)`` where data
        arrays and ``init_position`` are traced JIT arguments (not
        closure constants).  ``nuts.init()`` is called inside the JIT
        body so ``value_and_grad`` of the traced-data logdensity is
        compiled and cached together with the sampling loop.
    """
    import blackjax
    import numpyro.distributions as dist

    is_3level = model_name == "hgf_3level"

    # Prior distributions -- parameterless JAX objects, safe to capture
    prior_omega_2 = dist.TruncatedNormal(loc=-3.0, scale=2.0, high=0.0)
    prior_log_beta = dist.Normal(0.0, 1.5)
    prior_zeta = dist.Normal(0.0, 2.0)
    if is_3level:
        prior_omega_3 = dist.TruncatedNormal(
            loc=-6.0,
            scale=2.0,
            high=0.0,
        )
        # κ frozen at _KAPPA_FIXED (1.0); no sampled prior.

    if use_pmap:
        # pmap path: pmap handles its own JIT compilation
        def sample_loop_pmap(
            init_position: dict[str, jnp.ndarray],
            warmup_params: dict,
            sample_key: jnp.ndarray,
            input_data: jnp.ndarray,
            observed: jnp.ndarray,
            choices: jnp.ndarray,
            trial_mask: jnp.ndarray,
        ) -> tuple:
            # Reconstruct logdensity_fn with traced data args
            def logdensity_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
                omega_2 = params["omega_2"]
                log_beta = params["log_beta"]
                beta = jnp.exp(log_beta)
                zeta = params["zeta"]
                prior_lp = jnp.sum(prior_omega_2.log_prob(omega_2))
                prior_lp = prior_lp + jnp.sum(
                    prior_log_beta.log_prob(log_beta),
                )
                prior_lp = prior_lp + jnp.sum(
                    prior_zeta.log_prob(zeta),
                )
                if is_3level:
                    omega_3 = params["omega_3"]
                    prior_lp = prior_lp + jnp.sum(
                        prior_omega_3.log_prob(omega_3),
                    )
                    kappa = jnp.full_like(omega_2, _KAPPA_FIXED)
                    likelihood_lp = batched_logp_fn(
                        omega_2,
                        omega_3,
                        kappa,
                        beta,
                        zeta,
                        input_data,
                        observed,
                        choices,
                        trial_mask,
                    )
                else:
                    likelihood_lp = batched_logp_fn(
                        omega_2,
                        beta,
                        zeta,
                        input_data,
                        observed,
                        choices,
                        trial_mask,
                    )
                return prior_lp + likelihood_lp

            nuts = blackjax.nuts(logdensity_fn, **warmup_params)
            # Build initial state INSIDE JIT — value_and_grad uses traced data
            initial_state = nuts.init(init_position)
            chain_keys = jax.random.split(sample_key, n_chains)

            replicated_state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
                initial_state,
            )

            def _sample_one_chain(
                rng_key: jnp.ndarray,
                state,  # noqa: ANN001
            ) -> tuple:
                @jax.jit
                def _one_step(s, k):  # noqa: ANN001
                    new_s, info = nuts.step(k, s)
                    return new_s, (new_s, info)

                keys = jax.random.split(rng_key, n_draws)
                _, (states, infos) = lax.scan(_one_step, state, keys)
                return states, infos

            all_states, all_infos = jax.pmap(_sample_one_chain)(
                chain_keys,
                replicated_state,
            )
            return all_states, all_infos

        return sample_loop_pmap

    # vmap path: wrap with @jax.jit for persistent cache
    @jax.jit
    def sample_loop_vmap(
        init_position: dict[str, jnp.ndarray],
        warmup_params: dict,
        sample_key: jnp.ndarray,
        input_data: jnp.ndarray,
        observed: jnp.ndarray,
        choices: jnp.ndarray,
        trial_mask: jnp.ndarray,
    ) -> tuple:
        # Reconstruct logdensity_fn with traced data args
        def logdensity_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            omega_2 = params["omega_2"]
            log_beta = params["log_beta"]
            beta = jnp.exp(log_beta)
            zeta = params["zeta"]
            prior_lp = jnp.sum(prior_omega_2.log_prob(omega_2))
            prior_lp = prior_lp + jnp.sum(
                prior_log_beta.log_prob(log_beta),
            )
            prior_lp = prior_lp + jnp.sum(prior_zeta.log_prob(zeta))
            if is_3level:
                omega_3 = params["omega_3"]
                prior_lp = prior_lp + jnp.sum(
                    prior_omega_3.log_prob(omega_3),
                )
                kappa = jnp.full_like(omega_2, _KAPPA_FIXED)
                likelihood_lp = batched_logp_fn(
                    omega_2,
                    omega_3,
                    kappa,
                    beta,
                    zeta,
                    input_data,
                    observed,
                    choices,
                    trial_mask,
                )
            else:
                likelihood_lp = batched_logp_fn(
                    omega_2,
                    beta,
                    zeta,
                    input_data,
                    observed,
                    choices,
                    trial_mask,
                )
            return prior_lp + likelihood_lp

        nuts = blackjax.nuts(logdensity_fn, **warmup_params)
        # Build initial state INSIDE JIT — value_and_grad uses traced data
        initial_state = nuts.init(init_position)
        chain_keys = jax.random.split(sample_key, n_chains)

        replicated_state = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
            initial_state,
        )

        # Progress callback (closure-captured so different phase_label /
        # log_every produce distinct HLO — OK because smoke test recompiles
        # per phase anyway).  Host callback is async (ordered=False) so it
        # does not block the scan.
        progress_cb = _build_progress_callback(
            log_every, n_chains, phase_label,
        )

        def _one_step(states, scan_in):  # noqa: ANN001
            rng_key, step_idx = scan_in
            keys = jax.random.split(rng_key, n_chains)
            new_states, infos = jax.vmap(nuts.step)(keys, states)
            if progress_cb is not None:
                jax.debug.callback(
                    progress_cb,
                    step_idx + 1,
                    infos.num_integration_steps,
                    infos.acceptance_rate,
                    infos.is_divergent.astype(jnp.int32),
                    infos.num_trajectory_expansions,
                    ordered=False,
                )
            return new_states, (new_states, infos)

        draw_keys = jax.random.split(chain_keys[0], n_draws)
        step_idxs = jnp.arange(n_draws, dtype=jnp.int32)
        _, (all_states, all_infos) = lax.scan(
            _one_step,
            replicated_state,
            (draw_keys, step_idxs),
        )
        return all_states, all_infos

    return sample_loop_vmap


def _samples_to_idata(
    positions: dict[str, np.ndarray],
    sample_stats: dict[str, np.ndarray],
    var_names: list[str],
    participant_ids: list[str],
    participant_groups: list[str],
    participant_sessions: list[str],
    model_name: str = "hgf_3level",
    coord_name: str = "participant",
) -> az.InferenceData:
    """Convert BlackJAX sample arrays to ArviZ InferenceData.

    Adds the deterministic ``beta = exp(log_beta)`` transform and
    constructs an ``InferenceData`` with ``coord_name`` as a named
    dimension and ``participant_group`` / ``participant_session`` as
    additional coordinates.

    Parameters
    ----------
    positions : dict[str, numpy.ndarray]
        Posterior samples.  Each value has shape
        ``(n_chains, n_draws, P)``.
    sample_stats : dict[str, numpy.ndarray]
        ``"diverging"`` and ``"acceptance_rate"``, each shaped
        ``(n_chains, n_draws)``.
    var_names : list[str]
        Names of all variables (including ``"beta"``).
    participant_ids : list[str]
        Participant identifier strings.
    participant_groups : list[str]
        Group labels per participant.
    participant_sessions : list[str]
        Session labels per participant.
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    coord_name : str, optional
        Name for the participant dimension/coordinate.  Defaults to
        ``"participant"`` for the PRL pipeline; PAT-RL passes
        ``"participant_id"`` to match downstream exporter expectations.

    Returns
    -------
    arviz.InferenceData
        Posterior with the chosen participant coord and
        ``participant_group``/``participant_session`` metadata.
    """
    import arviz as az

    # Build posterior dict from sampled positions
    posterior_dict: dict[str, np.ndarray] = {}
    for var in var_names:
        if var == "beta":
            # Deterministic transform
            posterior_dict["beta"] = np.exp(positions["log_beta"])
        elif var in positions:
            posterior_dict[var] = positions[var]

    dims_dict = {var: [coord_name] for var in posterior_dict}
    coords_dict: dict[str, list[str]] = {
        coord_name: participant_ids,
    }

    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=sample_stats,
        dims=dims_dict,
        coords=coords_dict,
    )

    # Attach group and session metadata as additional coords
    idata.posterior = idata.posterior.assign_coords(
        participant_group=(coord_name, participant_groups),
        participant_session=(coord_name, participant_sessions),
    )

    return idata


# ---------------------------------------------------------------------------
# NumPyro model functions
# ---------------------------------------------------------------------------


def _numpyro_model_3level(
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    batched_logp_fn,  # noqa: ANN001
) -> None:
    """NumPyro model: 3-level HGF with IID priors per participant.

    Priors match :func:`build_pymc_model_batched` exactly.  Data is received
    as arguments (forwarded from ``MCMC.run`` kwargs) so that XLA sees them
    as dynamic traced values and can reuse the compiled kernel across
    iterations with different data of the same shape.

    Parameters
    ----------
    input_data : jnp.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays.
    observed : jnp.ndarray, shape (P, n_trials, 3)
        Binary observed masks.
    choices : jnp.ndarray, shape (P, n_trials)
        Chosen cue indices.
    trial_mask : jnp.ndarray, shape (P, n_trials)
        Binary trial mask for variable-length cohorts.
    n_participants : int
        Number of participants ``P``.
    batched_logp_fn : callable
        Pure JAX batched logp from :func:`build_logp_fn_batched`.
    """
    import numpyro
    import numpyro.distributions as dist

    # Perceptual parameters
    omega_2 = numpyro.sample(
        "omega_2",
        dist.TruncatedNormal(
            loc=-3.0,
            scale=2.0,
            high=0.0,
        ).expand([n_participants]),
    )
    omega_3 = numpyro.sample(
        "omega_3",
        dist.TruncatedNormal(
            loc=-6.0,
            scale=2.0,
            high=0.0,
        ).expand([n_participants]),
    )
    # κ frozen at _KAPPA_FIXED (1.0) — collapses ω₃×κ ridge that otherwise
    # saturates NUTS tree depth (see module constant docstring).
    kappa = jnp.full((n_participants,), _KAPPA_FIXED)

    # Response parameters
    log_beta = numpyro.sample(
        "log_beta",
        dist.Normal(0.0, 1.5).expand([n_participants]),
    )
    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    zeta = numpyro.sample(
        "zeta",
        dist.Normal(0.0, 2.0).expand([n_participants]),
    )

    # Custom HGF log-likelihood
    logp = batched_logp_fn(
        omega_2,
        omega_3,
        kappa,
        beta,
        zeta,
        input_data,
        observed,
        choices,
        trial_mask,
    )
    numpyro.factor("hgf_loglike", logp)


def _numpyro_model_2level(
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    batched_logp_fn,  # noqa: ANN001
) -> None:
    """NumPyro model: 2-level HGF with IID priors per participant.

    Priors match the 2-level branch of :func:`build_pymc_model_batched`
    exactly.  See :func:`_numpyro_model_3level` for argument descriptions.

    Parameters
    ----------
    input_data : jnp.ndarray, shape (P, n_trials, 3)
        Float reward-value arrays.
    observed : jnp.ndarray, shape (P, n_trials, 3)
        Binary observed masks.
    choices : jnp.ndarray, shape (P, n_trials)
        Chosen cue indices.
    trial_mask : jnp.ndarray, shape (P, n_trials)
        Binary trial mask for variable-length cohorts.
    n_participants : int
        Number of participants ``P``.
    batched_logp_fn : callable
        Pure JAX batched logp from :func:`build_logp_fn_batched`.
    """
    import numpyro
    import numpyro.distributions as dist

    # Perceptual parameter
    omega_2 = numpyro.sample(
        "omega_2",
        dist.TruncatedNormal(
            loc=-3.0,
            scale=2.0,
            high=0.0,
        ).expand([n_participants]),
    )

    # Response parameters
    log_beta = numpyro.sample(
        "log_beta",
        dist.Normal(0.0, 1.5).expand([n_participants]),
    )
    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    zeta = numpyro.sample(
        "zeta",
        dist.Normal(0.0, 2.0).expand([n_participants]),
    )

    # Custom HGF log-likelihood
    logp = batched_logp_fn(
        omega_2,
        beta,
        zeta,
        input_data,
        observed,
        choices,
        trial_mask,
    )
    numpyro.factor("hgf_loglike", logp)


# ---------------------------------------------------------------------------
# Hierarchical PyMC model factory (DEPRECATED)
# ---------------------------------------------------------------------------


def build_pymc_model_batched(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    model_name: str = "hgf_3level",
    trial_mask: np.ndarray | None = None,
) -> tuple:
    """Build a hierarchical PyMC model with shape=(P,) IID priors.

    .. deprecated::
        Use :func:`build_logp_fn_batched` with :func:`fit_batch_hierarchical`
        instead.  The PyMC bridge path is retained for backward compatibility
        with VALID-01/02 tests but will be removed in a future release.

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
    import warnings

    import pymc as pm

    warnings.warn(
        "build_pymc_model_batched is deprecated. Use build_logp_fn_batched "
        "with fit_batch_hierarchical (numpyro-direct path) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if input_data_arr.ndim != 3:
        msg = (
            f"input_data_arr must be 3-dimensional (P, n_trials, 3), "
            f"got ndim={input_data_arr.ndim}"
        )
        raise ValueError(msg)

    if model_name not in _MODEL_NAMES:
        msg = f"model_name must be one of {_MODEL_NAMES}, got {model_name!r}"
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
                "omega_2",
                mu=-3.0,
                sigma=2.0,
                upper=0.0,
                shape=n_participants,
            )

            # Response parameters
            log_beta = pm.Normal(
                "log_beta",
                mu=0.0,
                sigma=1.5,
                shape=n_participants,
            )
            beta = pm.Deterministic(
                "beta",
                pm.math.exp(log_beta),
            )
            zeta = pm.Normal(
                "zeta",
                mu=0.0,
                sigma=2.0,
                shape=n_participants,
            )

            pm.Potential("loglike", logp_op(omega_2, beta, zeta))
            var_names = ["omega_2", "beta", "zeta"]

        else:
            # Perceptual parameters
            omega_2 = pm.TruncatedNormal(
                "omega_2",
                mu=-3.0,
                sigma=2.0,
                upper=0.0,
                shape=n_participants,
            )
            omega_3 = pm.TruncatedNormal(
                "omega_3",
                mu=-6.0,
                sigma=2.0,
                upper=0.0,
                shape=n_participants,
            )
            kappa = pm.TruncatedNormal(
                "kappa",
                mu=1.0,
                sigma=0.5,
                lower=0.01,
                upper=2.0,
                shape=n_participants,
            )

            # Response parameters
            log_beta = pm.Normal(
                "log_beta",
                mu=0.0,
                sigma=1.5,
                shape=n_participants,
            )
            beta = pm.Deterministic(
                "beta",
                pm.math.exp(log_beta),
            )
            zeta = pm.Normal(
                "zeta",
                mu=0.0,
                sigma=2.0,
                shape=n_participants,
            )

            pm.Potential(
                "loglike",
                logp_op(omega_2, omega_3, kappa, beta, zeta),
            )
            var_names = ["omega_2", "omega_3", "kappa", "beta", "zeta"]

    return model, var_names, n_participants


# ---------------------------------------------------------------------------
# Cohort orchestrator
# ---------------------------------------------------------------------------


def _build_arrays_single(
    subset: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (input_data, observed, choices) arrays for one participant.

    Mirrors the partial-feedback logic from
    ``legacy/batch.py::_build_arrays`` — only the chosen cue receives a
    reward signal on each trial; unchosen cues have ``observed=0``.

    Parameters
    ----------
    subset : pandas.DataFrame
        Rows for one participant-session with columns ``cue_chosen`` and
        ``reward``.  Must be sorted by trial order.

    Returns
    -------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.
    """
    n_trials = len(subset)
    choices = subset["cue_chosen"].to_numpy(dtype=int)
    rewards = subset["reward"].to_numpy(dtype=float)

    input_data_arr = np.zeros((n_trials, 3), dtype=float)
    observed_arr = np.zeros((n_trials, 3), dtype=int)

    for t in range(n_trials):
        cue = choices[t]
        input_data_arr[t, cue] = rewards[t]
        observed_arr[t, cue] = 1

    return input_data_arr, observed_arr, choices


def fit_batch_hierarchical(
    sim_df: pd.DataFrame,
    model_name: str = "hgf_2level",
    n_chains: int = 4,
    n_draws: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int = 42,
    sampler: str = "blackjax",
    progressbar: bool = True,
    warmup_params: dict | None = None,
    log_every: int = 0,
) -> az.InferenceData | tuple[az.InferenceData, dict]:
    """Fit an entire cohort via BlackJAX NUTS (default) or NumPyro MCMC.

    Groups ``sim_df`` by ``(participant_id, group, session)``, builds the
    stacked ``(P, n_trials, 3)`` arrays, constructs a pure JAX logp via
    :func:`build_logp_fn_batched`, and runs NUTS for the full cohort.
    Returns an ``InferenceData`` with a ``participant`` dimension on every
    parameter so downstream analysis can map posterior slices back to
    individual participants.

    **BlackJAX path** (default): Builds a pure JAX log-posterior via
    :func:`_build_log_posterior`, runs NUTS with window adaptation via
    :func:`_run_blackjax_nuts`, and converts to ArviZ via
    :func:`_samples_to_idata`.  Compiles the NUTS step function once via
    ``jax.jit`` and reuses it across all MCMC steps — no per-call
    recompilation.

    **NumPyro path** (fallback): Uses numpyro MCMC with
    ``chain_method="vectorized"`` and ``jit_model_args=True``.  Data
    arrays are passed as kwargs to ``MCMC.run()`` for JIT cache reuse.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level DataFrame with columns ``participant_id``, ``group``,
        ``session``, ``cue_chosen``, ``reward``.
    model_name : str, optional
        ``"hgf_2level"`` (default — primary target; no ω₃×κ ridge) or
        ``"hgf_3level"`` (exploratory; κ frozen at 1.0 per
        :data:`_KAPPA_FIXED` to avoid ω₃×κ multiplicative ridge that
        otherwise saturates NUTS ``max_tree_depth``).
    n_chains : int, optional
        Number of MCMC chains.  Default ``4``.
    n_draws : int, optional
        Posterior draws per chain after tuning.  Default ``1000``.
    n_tune : int, optional
        Tuning steps per chain.  Default ``1000``.
    target_accept : float, optional
        NUTS target acceptance rate.  Default ``0.95``.
    random_seed : int, optional
        RNG seed for reproducibility.  Default ``42``.
    sampler : str, optional
        ``"blackjax"`` (default) uses BlackJAX NUTS with pmap/vmap chain
        parallelism.  ``"numpyro"`` uses NumPyro MCMC (vectorized vmap).
        ``"pymc"`` raises :class:`DeprecationWarning` and falls through
        to the numpyro path.
    progressbar : bool, optional
        Show MCMC progress bar (numpyro path only).  Default ``True``.
    warmup_params : dict or None, optional
        Pre-adapted NUTS parameters from a previous call.  When
        provided, warmup is skipped (~1100s savings per call).  Pass
        the second element of the return tuple from a prior call.
        Only used with the BlackJAX path.
    log_every : int, optional
        If ``> 0``, emit NUTS progress diagnostics every ``log_every``
        draws via ``jax.debug.callback`` fired from inside the JIT'd
        scan.  Reports wall time, integration-step distribution,
        trajectory-expansion distribution, acceptance rate, and
        divergence count.  ``0`` (default) disables — keeps the
        production HLO identical to pre-instrumentation runs so the
        XLA persistent cache keeps hitting.  Only used with the
        BlackJAX path, vmap variant (single-device).

    Returns
    -------
    arviz.InferenceData or tuple[arviz.InferenceData, dict]
        If ``warmup_params`` is ``None`` (first call): returns
        ``(idata, adapted_params)`` tuple so caller can cache the
        adapted parameters.  If ``warmup_params`` is provided
        (subsequent calls): returns just ``idata``.

    Raises
    ------
    ValueError
        If ``sim_df`` is missing required columns or participants have
        different trial counts.
    """
    import warnings

    # Deprecation gate for sampler="pymc"
    if sampler == "pymc":
        warnings.warn(
            "sampler='pymc' is deprecated and ignored. "
            "fit_batch_hierarchical falls through to the numpyro path. "
            "Use sampler='blackjax' (default) for best performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        sampler = "numpyro"

    # ------------------------------------------------------------------
    # Validate input DataFrame
    # ------------------------------------------------------------------
    required_cols = {
        "participant_id",
        "group",
        "session",
        "cue_chosen",
        "reward",
    }
    missing_cols = required_cols - set(sim_df.columns)
    if missing_cols:
        msg = (
            f"sim_df is missing required columns: {sorted(missing_cols)}. "
            f"Got columns: {sorted(sim_df.columns)}"
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Group by (participant_id, group, session)
    # ------------------------------------------------------------------
    group_keys = ["participant_id", "group", "session"]
    groups = list(sim_df.groupby(group_keys, sort=False))

    # ------------------------------------------------------------------
    # Build per-participant arrays and stack into (P, n_trials, 3)
    # ------------------------------------------------------------------
    input_data_list: list[np.ndarray] = []
    observed_list: list[np.ndarray] = []
    choices_list: list[np.ndarray] = []
    participant_ids: list[str] = []
    participant_groups: list[str] = []
    participant_sessions: list[str] = []

    for (pid, grp, sess), subset in groups:
        # Sort by trial index if column exists
        if "trial" in subset.columns:
            subset = subset.sort_values("trial")

        inp, obs, ch = _build_arrays_single(subset)
        input_data_list.append(inp)
        observed_list.append(obs)
        choices_list.append(ch)
        participant_ids.append(str(pid))
        participant_groups.append(str(grp))
        participant_sessions.append(str(sess))

    # Trial-count guard: all participants must have the same n_trials
    trial_counts = [arr.shape[0] for arr in input_data_list]
    if len(set(trial_counts)) != 1:
        msg = (
            f"All participants must have the same number of trials. "
            f"Got trial counts: {trial_counts}"
        )
        raise ValueError(msg)

    n_trials = trial_counts[0]
    n_participants = len(input_data_list)

    input_data_arr = np.stack(input_data_list, axis=0)
    observed_arr = np.stack(observed_list, axis=0)
    choices_arr = np.stack(choices_list, axis=0)

    # ------------------------------------------------------------------
    # Build the pure JAX logp function (no data closure)
    # ------------------------------------------------------------------
    logp_fn, _n_params = build_logp_fn_batched(model_name, n_trials)

    # ------------------------------------------------------------------
    # Convert data to JAX arrays
    # ------------------------------------------------------------------
    jax_input_data = jnp.array(input_data_arr, dtype=jnp.float32)
    jax_observed = jnp.array(observed_arr, dtype=jnp.int32)
    jax_choices = jnp.array(choices_arr, dtype=jnp.int32)
    jax_trial_mask = jnp.ones(
        (n_participants, n_trials),
        dtype=jnp.float32,
    )

    rng_key = jax.random.PRNGKey(random_seed)

    if sampler == "blackjax":
        # ==============================================================
        # BlackJAX path (default): pure JAX log-posterior + NUTS
        # ==============================================================

        # Build log-posterior (priors + batched HGF likelihood)
        logdensity_fn = _build_log_posterior(
            logp_fn,
            jax_input_data,
            jax_observed,
            jax_choices,
            jax_trial_mask,
            n_participants,
            model_name,
        )

        # Build initial position dict at prior modes.
        # κ is frozen at _KAPPA_FIXED (1.0) — not sampled, not in initial_position.
        if model_name == "hgf_3level":
            initial_position: dict[str, jnp.ndarray] = {
                "omega_2": jnp.full((n_participants,), -3.0),
                "omega_3": jnp.full((n_participants,), -6.0),
                "log_beta": jnp.full((n_participants,), 0.0),
                "zeta": jnp.full((n_participants,), 0.0),
            }
            var_names = [
                "omega_2",
                "omega_3",
                "log_beta",
                "beta",
                "zeta",
            ]
        else:
            initial_position = {
                "omega_2": jnp.full((n_participants,), -3.0),
                "log_beta": jnp.full((n_participants,), 0.0),
                "zeta": jnp.full((n_participants,), 0.0),
            }
            var_names = ["omega_2", "log_beta", "beta", "zeta"]

        # Run MCMC (data as traced args for JIT cache reuse)
        positions, sample_stats, n_chains_actual, adapted_params = (
            _run_blackjax_nuts(
                logdensity_fn,
                initial_position,
                rng_key,
                n_tune=n_tune,
                n_draws=n_draws,
                n_chains=n_chains,
                target_accept=target_accept,
                batched_logp_fn=logp_fn,
                input_data=jax_input_data,
                observed=jax_observed,
                choices=jax_choices,
                trial_mask=jax_trial_mask,
                model_name=model_name,
                warmup_params=warmup_params,
                log_every=log_every,
                phase_label=model_name.replace("hgf_", ""),
            )
        )

        # Convert to ArviZ InferenceData
        idata = _samples_to_idata(
            positions,
            sample_stats,
            var_names,
            participant_ids,
            participant_groups,
            participant_sessions,
            model_name,
        )

        # Return adapted params so caller can skip warmup next time
        if warmup_params is None:
            # First call: caller should cache adapted_params
            return idata, adapted_params
        return idata

    else:
        # ==============================================================
        # NumPyro path (fallback): numpyro MCMC with vectorized chains
        # ==============================================================
        import arviz as az
        from numpyro.infer import MCMC, NUTS

        if model_name == "hgf_3level":
            model_fn = _numpyro_model_3level
            # κ frozen — not in fitted var_names (see _KAPPA_FIXED docstring).
            var_names = [
                "omega_2",
                "omega_3",
                "log_beta",
                "beta",
                "zeta",
            ]
        else:
            model_fn = _numpyro_model_2level
            var_names = ["omega_2", "log_beta", "beta", "zeta"]

        # Always use "vectorized" (vmap): compiles a single fused kernel
        # for all chains, enables jit_model_args for trace-cache reuse
        # across calls with the same shapes.
        from functools import partial

        bound_model = partial(
            model_fn,
            n_participants=n_participants,
            batched_logp_fn=logp_fn,
        )
        kernel = NUTS(bound_model, target_accept_prob=target_accept)
        mcmc = MCMC(
            kernel,
            num_warmup=n_tune,
            num_samples=n_draws,
            num_chains=n_chains,
            chain_method="vectorized",
            jit_model_args=True,
            progress_bar=progressbar,
        )
        mcmc.run(
            rng_key,
            input_data=jax_input_data,
            observed=jax_observed,
            choices=jax_choices,
            trial_mask=jax_trial_mask,
        )

        # Convert to ArviZ InferenceData with participant coords
        dims_dict = {vn: ["participant"] for vn in var_names}
        coords_dict: dict[str, list[str]] = {
            "participant": participant_ids,
        }

        idata = az.from_numpyro(
            mcmc,
            dims=dims_dict,
            coords=coords_dict,
        )

        # Attach group and session metadata as additional coords
        idata.posterior = idata.posterior.assign_coords(
            participant_group=("participant", participant_groups),
            participant_session=("participant", participant_sessions),
        )

    return idata


__all__ = [
    "_build_sample_loop",
    "build_logp_fn_batched",
    "build_logp_ops_batched",
    "build_pymc_model_batched",
    "fit_batch_hierarchical",
]
