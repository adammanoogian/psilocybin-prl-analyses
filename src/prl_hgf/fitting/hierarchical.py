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

import time
import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import lax
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from prl_hgf.fitting.config import FitConfig
from prl_hgf.fitting.priors import HGFPriorSpec

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
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
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
    prior_spec : HGFPriorSpec or None, optional
        Prior specification.  If ``None``, uses the default for the
        given ``model_name``.

    Returns
    -------
    logdensity_fn : callable
        ``dict[str, jnp.ndarray] -> scalar``.  Keys match the model
        parameter names; each value has shape ``(P,)``.
    """
    from prl_hgf.fitting.priors import HGFPriorSpec

    is_3level = model_name == "hgf_3level"

    if prior_spec is None:
        prior_spec = (
            HGFPriorSpec.default_3level()
            if is_3level
            else HGFPriorSpec.default_2level()
        )

    # Build numpyro distribution objects from the prior spec
    prior_omega_2 = prior_spec.omega_2.to_numpyro_dist()
    prior_log_beta = prior_spec.log_beta.to_numpyro_dist()
    prior_zeta = prior_spec.zeta.to_numpyro_dist()
    if is_3level:
        prior_omega_3 = prior_spec.omega_3.to_numpyro_dist()
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


# ---------------------------------------------------------------------------
# BlackJAX Mode B (hierarchical) log-posterior
# ---------------------------------------------------------------------------

#: Parameters that participate in the hierarchical model for each level.
_HIERARCHICAL_PARAMS_2LEVEL: tuple[str, ...] = ("omega_2", "log_beta", "zeta")
_HIERARCHICAL_PARAMS_3LEVEL: tuple[str, ...] = (
    "omega_2",
    "log_beta",
    "zeta",
    "omega_3",
)


def _build_log_posterior_hierarchical(
    batched_logp_fn,  # noqa: ANN001
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    n_groups: int,
    group_idx: jnp.ndarray,
    model_name: str = "hgf_3level",
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
    non_centered: tuple[str, ...] = (),
    x_covariate: jnp.ndarray | None = None,
) -> callable:
    """Build a pure JAX log-posterior for Mode B (hierarchical) BlackJAX NUTS.

    Implements the Boehm 2018 formulation:

        theta_{k,p} ~ Normal(mu_{g(k),p} + beta_p * x_k, sigma_p)

    with shared sigma_p per parameter (not per-group).  Sigma parameters
    are stored in log-space (``log_sigma_*``) with exp-transform Jacobian
    correction to keep NUTS unconstrained.

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
    n_groups : int
        Number of experimental groups ``G``.
    group_idx : jnp.ndarray, shape (P,)
        Integer group index for each participant (0-based).
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    prior_spec : HGFPriorSpec or None, optional
        Prior specification with hyperprior fields.  If ``None``, uses the
        hierarchical default for the given ``model_name``.
    non_centered : tuple of str, optional
        Parameter names to apply non-centered reparameterization.
        E.g. ``("omega_2", "omega_3")``.  Non-centered parameters use
        ``*_nc`` keys in the params dict with N(0,1) prior.
    x_covariate : jnp.ndarray or None, shape (P,)
        Mean-centered continuous covariate.  If provided, adds per-param
        slope ``beta_*`` to the group mean.

    Returns
    -------
    logdensity_fn : callable
        ``dict[str, jnp.ndarray] -> scalar``.  Keys include hyperprior
        parameters (``mu_*``, ``log_sigma_*``), optional covariate slopes
        (``beta_*``), and participant-level parameters (``*`` or ``*_nc``).
    """
    from prl_hgf.fitting.priors import HGFPriorSpec

    is_3level = model_name == "hgf_3level"

    if prior_spec is None:
        prior_spec = (
            HGFPriorSpec.default_3level_hierarchical()
            if is_3level
            else HGFPriorSpec.default_2level_hierarchical()
        )

    # Determine which parameters are hierarchical for this model
    h_params = _HIERARCHICAL_PARAMS_3LEVEL if is_3level else _HIERARCHICAL_PARAMS_2LEVEL

    # Build hyperprior distribution objects (numpyro dists for log_prob)
    mu_hypers: dict = {}
    sigma_hypers: dict = {}
    for p_name in h_params:
        mu_hyper_field = getattr(prior_spec, f"{p_name}_mu_hyper", None)
        sigma_hyper_field = getattr(prior_spec, f"{p_name}_sigma_hyper", None)
        if mu_hyper_field is not None:
            mu_hypers[p_name] = mu_hyper_field.to_numpyro_dist()
        if sigma_hyper_field is not None:
            sigma_hypers[p_name] = sigma_hyper_field.to_numpyro_dist()

    def logdensity_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute hierarchical log-posterior: hyperprior + prior + lik.

        Parameters
        ----------
        params : dict[str, jnp.ndarray]
            Parameter dict with hyperprior and participant-level keys.

        Returns
        -------
        jnp.ndarray
            Scalar log-posterior.
        """
        prior_lp = jnp.array(0.0)

        # Resolve participant-level parameter values from hierarchical
        # structure
        resolved: dict[str, jnp.ndarray] = {}

        for p_name in h_params:
            # --- Hyperpriors ---
            mu_key = f"mu_{p_name}"
            log_sigma_key = f"log_sigma_{p_name}"

            mu_p = params[mu_key]  # shape (n_groups,)
            log_sigma_p = params[log_sigma_key]  # shape ()
            sigma_p = jnp.exp(log_sigma_p)

            # Hyperprior log-prob on mu_p
            if p_name in mu_hypers:
                prior_lp = prior_lp + jnp.sum(mu_hypers[p_name].log_prob(mu_p))

            # Hyperprior log-prob on sigma_p (HalfNormal on sigma_p)
            # with Jacobian correction for exp-transform:
            # log p(sigma) + log|d(sigma)/d(log_sigma)|
            # = log p(exp(log_sigma)) + log_sigma
            if p_name in sigma_hypers:
                prior_lp = prior_lp + jnp.sum(
                    sigma_hypers[p_name].log_prob(sigma_p)
                )
                # Jacobian: |d(exp(x))/dx| = exp(x) => log|J| = x
                prior_lp = prior_lp + log_sigma_p

            # --- Participant-level mean ---
            mean_p = mu_p[group_idx]  # shape (P,)

            # Covariate slope
            if x_covariate is not None:
                beta_key = f"beta_{p_name}"
                beta_p = params[beta_key]  # shape ()
                mean_p = mean_p + beta_p * x_covariate
                # Weakly informative prior on beta_p: N(0, 1)
                prior_lp = prior_lp + jnp.sum(
                    jax.scipy.stats.norm.logpdf(beta_p, 0.0, 1.0)
                )

            # --- Participant-level prior ---
            if p_name in non_centered:
                nc_key = f"{p_name}_nc"
                nc_vals = params[nc_key]  # shape (P,)
                # Standard normal prior on non-centered params
                prior_lp = prior_lp + jnp.sum(
                    jax.scipy.stats.norm.logpdf(nc_vals, 0.0, 1.0)
                )
                # Deterministic transform to centered space
                resolved[p_name] = mean_p + sigma_p * nc_vals
            else:
                centered_vals = params[p_name]  # shape (P,)
                # Normal prior centered on group mean
                prior_lp = prior_lp + jnp.sum(
                    jax.scipy.stats.norm.logpdf(
                        centered_vals, mean_p, sigma_p
                    )
                )
                resolved[p_name] = centered_vals

        # --- Compute likelihood ---
        omega_2 = resolved["omega_2"]
        log_beta = resolved["log_beta"]
        beta = jnp.exp(log_beta)
        zeta = resolved["zeta"]

        if is_3level:
            omega_3 = resolved["omega_3"]
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


def _build_initial_position_hierarchical(
    n_participants: int,
    n_groups: int,
    model_name: str = "hgf_3level",
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
    non_centered: tuple[str, ...] = (),
    x_covariate: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Build initial position dict for Mode B BlackJAX NUTS.

    Returns the initial parameter dict with correct keys matching
    the logdensity_fn produced by :func:`_build_log_posterior_hierarchical`.

    Parameters
    ----------
    n_participants : int
        Number of participants ``P``.
    n_groups : int
        Number of experimental groups ``G``.
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).
    prior_spec : HGFPriorSpec or None, optional
        Prior specification with hyperprior fields.
    non_centered : tuple of str, optional
        Parameter names using non-centered reparameterization.
    x_covariate : jnp.ndarray or None, shape (P,)
        If provided, adds ``beta_*`` keys to the initial position.

    Returns
    -------
    position : dict[str, jnp.ndarray]
        Initial position dict.  Keys:
        - ``mu_{param}``: shape ``(n_groups,)`` at prior mode
        - ``log_sigma_{param}``: shape ``()`` at 0.0 (sigma=1)
        - ``{param}`` or ``{param}_nc``: shape ``(P,)``
        - ``beta_{param}``: shape ``()`` if covariate provided
    """
    from prl_hgf.fitting.priors import HGFPriorSpec

    is_3level = model_name == "hgf_3level"

    if prior_spec is None:
        prior_spec = (
            HGFPriorSpec.default_3level_hierarchical()
            if is_3level
            else HGFPriorSpec.default_2level_hierarchical()
        )

    h_params = _HIERARCHICAL_PARAMS_3LEVEL if is_3level else _HIERARCHICAL_PARAMS_2LEVEL

    position: dict[str, jnp.ndarray] = {}

    for p_name in h_params:
        # mu initialized at hyperprior mode (loc)
        mu_hyper_field = getattr(prior_spec, f"{p_name}_mu_hyper", None)
        mu_init = mu_hyper_field.loc if mu_hyper_field is not None else 0.0
        position[f"mu_{p_name}"] = jnp.full((n_groups,), mu_init)

        # log_sigma initialized at 0.0 (sigma = 1.0)
        position[f"log_sigma_{p_name}"] = jnp.array(0.0)

        # Participant-level parameters
        if p_name in non_centered:
            # Non-centered: initialize at 0 (standard normal mode)
            position[f"{p_name}_nc"] = jnp.zeros(n_participants)
        else:
            # Centered: initialize at prior mode (mu_init)
            position[p_name] = jnp.full(n_participants, mu_init)

        # Covariate slope
        if x_covariate is not None:
            position[f"beta_{p_name}"] = jnp.array(0.0)

    return position


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


def _unwrap_nuts_info(info):  # noqa: ANN001
    """Pull the per-step NUTSInfo out of a possibly-wrapped adaptation info.

    BlackJAX's ``window_adaptation.run`` returns an ``AdaptationInfo`` that
    nests the per-step ``NUTSInfo`` under attribute ``.info``; sampling
    scans return the bare stacked ``NUTSInfo`` directly.  This helper
    handles both shapes plus a final defensive fallback so version drift
    in the BlackJAX layout cannot kill the diagnostic prints.
    """
    if hasattr(info, "is_divergent"):
        return info
    if hasattr(info, "info") and hasattr(info.info, "is_divergent"):
        return info.info
    return info


def _log_nuts_diagnostics(
    label: str,
    info,  # noqa: ANN001
    *,
    n_steps: int,
    n_chains: int,
    p_axis: int,
    elapsed_s: float | None = None,
    adapted_step_size=None,  # noqa: ANN001
    adapted_inverse_mass_matrix=None,  # noqa: ANN001
) -> None:
    """Print summary statistics from a stacked NUTSInfo pytree.

    Reads only fields that BlackJAX has already populated inside the
    JIT'd scan (``num_trajectory_expansions``, ``num_integration_steps``,
    ``acceptance_rate``, ``is_divergent``, ``energy``) so adding the call
    is HLO-invariant and persistent-cache-safe.  Surfaces the four
    quantities that distinguish a benign warmup from one stuck on bad
    posterior geometry: tree-depth saturation, leapfrog totals, divergent
    transitions, and energy drift.  When ``label == "warmup"`` and the
    Stan-default 500-step schedule is detected, also breaks the trace
    into init/slow1-4/term windows so the .out file shows when (in
    adaptation time) tree depth blows up.
    """
    nuts_info = _unwrap_nuts_info(info)

    def _safe_np(x):  # noqa: ANN001, ANN202
        if x is None:
            return None
        try:
            return np.asarray(jax.block_until_ready(x))
        except Exception:  # noqa: BLE001
            return None

    depth = _safe_np(getattr(nuts_info, "num_trajectory_expansions", None))
    n_lf = _safe_np(getattr(nuts_info, "num_integration_steps", None))
    accept = _safe_np(getattr(nuts_info, "acceptance_rate", None))
    div = _safe_np(getattr(nuts_info, "is_divergent", None))
    energy = _safe_np(getattr(nuts_info, "energy", None))

    print(
        f"[diag {label}] n_steps={n_steps} n_chains={n_chains} P={p_axis}"
        + (f" elapsed={elapsed_s:.1f}s" if elapsed_s is not None else ""),
        flush=True,
    )
    if depth is not None:
        print(
            f"[diag {label}] tree_depth: "
            f"mean={depth.mean():.2f} median={float(np.median(depth)):.0f} "
            f"p95={float(np.percentile(depth, 95)):.0f} max={int(depth.max())} "
            f"saturated_frac={float((depth >= 10).mean()):.3f}",
            flush=True,
        )
    if n_lf is not None:
        per_step_mean = float(n_lf.mean())
        print(
            f"[diag {label}] leapfrog_per_step: "
            f"mean={per_step_mean:.1f} median={float(np.median(n_lf)):.0f} "
            f"p95={float(np.percentile(n_lf, 95)):.0f} max={int(n_lf.max())} "
            f"total={int(n_lf.sum())}",
            flush=True,
        )
        if elapsed_s is not None and n_lf.sum() > 0:
            print(
                f"[diag {label}] derived: "
                f"{elapsed_s / float(n_lf.sum()) * 1000:.2f}ms per leapfrog "
                f"(across all chains)",
                flush=True,
            )
    if accept is not None:
        msg = f"[diag {label}] accept_rate: mean={accept.mean():.3f}"
        if accept.shape[0] >= 50:
            msg += f" final50={accept[-50:].mean():.3f}"
        print(msg, flush=True)
    if div is not None:
        print(
            f"[diag {label}] divergent: "
            f"count={int(div.sum())} rate={float(div.mean()):.4f}",
            flush=True,
        )
    if energy is not None:
        e_finite = energy[np.isfinite(energy)]
        if e_finite.size > 1:
            de = np.diff(e_finite.reshape(-1))
            print(
                f"[diag {label}] energy: "
                f"mean={float(e_finite.mean()):.2f} std={float(e_finite.std()):.2f} "
                f"|dE|_mean={float(np.abs(de).mean()):.3f} "
                f"|dE|_max={float(np.abs(de).max()):.2f}",
                flush=True,
            )

    # Window-partitioned summary for warmup at the Stan-default 500-step
    # schedule (init=75, slow=[25,50,100,200], term=50).  Tells you when in
    # adaptation tree depth saturates / divergences cluster.
    if label == "warmup" and depth is not None and depth.shape[0] == 500:
        windows = [
            (0, 75, "init"),
            (75, 100, "slow1"),
            (100, 150, "slow2"),
            (150, 250, "slow3"),
            (250, 450, "slow4"),
            (450, 500, "term"),
        ]
        for start, end, name in windows:
            seg_depth = depth[start:end]
            seg_div = div[start:end] if div is not None else None
            seg_acc = accept[start:end] if accept is not None else None
            div_count = int(seg_div.sum()) if seg_div is not None else -1
            acc_mean = float(seg_acc.mean()) if seg_acc is not None else float("nan")
            print(
                f"[diag warmup window={name} steps={start}-{end}] "
                f"depth_mean={seg_depth.mean():.2f} "
                f"depth_max={int(seg_depth.max())} "
                f"saturated={float((seg_depth >= 10).mean()):.3f} "
                f"accept={acc_mean:.3f} div={div_count}",
                flush=True,
            )

    # Adapted parameters (warmup only — these are the things passed forward
    # as warmup_params to the sampler).
    ss_np = _safe_np(adapted_step_size)
    if ss_np is not None:
        if ss_np.ndim == 0:
            print(f"[diag {label}] adapted step_size: {float(ss_np):.4g}", flush=True)
        else:
            print(
                f"[diag {label}] adapted step_size (per-chain): "
                f"min={float(ss_np.min()):.4g} max={float(ss_np.max()):.4g} "
                f"mean={float(ss_np.mean()):.4g}",
                flush=True,
            )
    imm_np = _safe_np(adapted_inverse_mass_matrix)
    if imm_np is not None:
        flat = imm_np.reshape(-1)
        finite_pos = flat[np.isfinite(flat) & (flat > 0)]
        if finite_pos.size > 0:
            cond = float(finite_pos.max() / finite_pos.min())
            print(
                f"[diag {label}] inverse_mass_matrix: "
                f"size={finite_pos.size} min={float(finite_pos.min()):.4g} "
                f"max={float(finite_pos.max()):.4g} cond={cond:.2g}",
                flush=True,
            )


def _laplace_warmup_params(
    logdensity_fn,  # noqa: ANN001
    initial_position: dict[str, jnp.ndarray],
    *,
    n_starts: int = 4,
    n_lbfgs_iter: int = 200,
    lbfgs_tol: float = 1e-5,
    ridge: float = 1e-4,
) -> dict | None:
    """Compute (step_size, inverse_mass_matrix) via multi-start Laplace approximation.

    Variant 2 of the Phase 14.2 comparison, upgraded to multi-start LBFGS with
    basin-comparison diagnostic (Phase 30, MODEA-04).  Runs ``jaxopt.LBFGS`` on
    ``-logdensity_fn`` from ``n_starts`` perturbed starting points to find an
    approximate MAP, selects the best MAP (lowest neg_logp), then computes the
    Hessian diagonal at the best MAP via Hessian-vector products
    (one ``jvp`` per parameter).  Regularizes positive, inverts to a
    per-parameter inverse mass matrix, picks an initial step_size from
    the median IMM.  Returns the dict in the shape that the existing
    ``warmup_params`` hook accepts — when fed back into
    :func:`_run_blackjax_nuts`, BlackJAX's ``window_adaptation`` is
    skipped entirely (see line ~1135 in this file).

    **Multi-start LBFGS (P5 prevention):** When ``n_starts > 1``, LBFGS runs
    from the original initial position plus ``n_starts - 1`` scale-adaptive
    perturbations.  After collecting all successful endpoints, a basin-comparison
    diagnostic checks whether any non-best endpoint disagrees with the best MAP
    by more than 2 SE (where SE is estimated from the Hessian diagonal at the
    best MAP).  If any endpoint disagrees, a ``MULTIMODAL WARNING`` is logged and
    ``None`` is returned, causing the caller to fall back to standard
    ``window_adaptation`` NUTS warmup.  This prevents Laplace warmup from silently
    locking onto one mode of a multimodal posterior (e.g., label-switching at high
    P), which would bias the subsequent NUTS chain.

    **Basin-comparison criterion:** ``|endpoint_i - best_map| / se_per_param > 2.0``
    on any parameter dimension triggers disagreement.  ``se_per_param =
    sqrt(1 / hess_diag_pd)`` is the Laplace posterior SD estimate.

    **Single-start backward compatibility:** When ``n_starts=1``, the basin check
    is skipped and the function behaves identically to the original single-start
    version.

    The Hessian diagonal is mathematically equivalent to the diagonal
    of a per-PS block-Hessian because the BlackJAX path uses IID priors
    per participant-session and a likelihood that factorizes across
    PS — the off-block entries of the joint Hessian are zero by
    construction.  See ``_build_log_posterior`` for that factorization.

    Cost is ~``P*K`` likelihood evaluations (one jvp each) plus
    ``n_starts`` × LBFGS iterations.  At the production shape (P=300, K=4)
    that's ~1200 evaluations for the diagonal — minutes, not hours.

    Returns ``None`` if any step (LBFGS, Hessian, regularization)
    produces NaNs or non-finite outputs, or if fewer than 2 starts converge,
    or if basin comparison detects multimodality; caller falls back to the
    standard window_adaptation warmup with a logged warning.

    Parameters
    ----------
    logdensity_fn : callable
        ``dict[str, jnp.ndarray] -> scalar`` log-posterior.
    initial_position : dict[str, jnp.ndarray]
        Starting point for LBFGS (typically prior means).  Each value
        has shape ``(P,)``.
    n_starts : int, default 4
        Number of LBFGS runs from perturbed starting points.  Start 0 is the
        original ``initial_position``.  Starts 1..n_starts-1 are
        ``flat_init + Normal(0, 0.5 * |flat_init| + 0.1)`` perturbations.
        When ``n_starts=1``, basin comparison is skipped (backward compat).
    n_lbfgs_iter : int, default 200
        ``jaxopt.LBFGS`` ``maxiter``.
    lbfgs_tol : float, default 1e-5
        ``jaxopt.LBFGS`` ``tol``.
    ridge : float, default 1e-4
        Diagonal floor applied to the Hessian before inversion.  Larger
        values produce a more conservative (smaller-magnitude) IMM.

    Returns
    -------
    dict or None
        ``{"step_size": float, "inverse_mass_matrix": jnp.ndarray}`` or
        None on failure or multimodality detection.
    """
    try:
        import jaxopt
        from jax.flatten_util import ravel_pytree
    except ImportError as exc:
        print(f"[laplace_warmup] import failed ({exc!r}); falling back", flush=True)
        return None

    flat_init, unravel = ravel_pytree(initial_position)
    n_flat = int(flat_init.shape[0])

    @jax.jit
    def neg_logp_flat(flat: jnp.ndarray) -> jnp.ndarray:
        return -logdensity_fn(unravel(flat))

    grad_neg_logp_flat = jax.jit(jax.grad(neg_logp_flat))

    # ------------------------------------------------------------------ #
    # Generate perturbed starting points                                   #
    # ------------------------------------------------------------------ #
    # Use a deterministic key derived from a hash of flat_init so results
    # are reproducible across calls with the same initial position.
    _key_seed = int(jnp.sum(jnp.abs(flat_init))) % (2**31)
    _rng = jax.random.PRNGKey(_key_seed)
    starts: list[jnp.ndarray] = [flat_init]
    for _s in range(1, n_starts):
        _rng, _subkey = jax.random.split(_rng)
        _noise_scale = 0.5 * jnp.abs(flat_init) + 0.1
        _noise = jax.random.normal(_subkey, shape=flat_init.shape) * _noise_scale
        starts.append(flat_init + _noise)

    print(
        f"[laplace_warmup] running multi-start LBFGS (n_starts={n_starts}, "
        f"n_flat={n_flat}, max_iter={n_lbfgs_iter}, tol={lbfgs_tol})...",
        flush=True,
    )

    solver = jaxopt.LBFGS(
        fun=neg_logp_flat,
        maxiter=n_lbfgs_iter,
        tol=lbfgs_tol,
    )

    # ------------------------------------------------------------------ #
    # Run LBFGS from each start                                            #
    # ------------------------------------------------------------------ #
    successful_maps: list[jnp.ndarray] = []
    successful_neg_logps: list[float] = []

    for i, start_i in enumerate(starts):
        t0 = time.perf_counter()
        try:
            res_i = solver.run(start_i)
            flat_map_i = res_i.params
            jax.block_until_ready(flat_map_i)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(
                f"[laplace_warmup] start {i + 1}/{n_starts}: LBFGS raised "
                f"{exc!r} ({elapsed:.1f}s) — skipping",
                flush=True,
            )
            continue
        elapsed = time.perf_counter() - t0
        neg_logp_i = float(neg_logp_flat(flat_map_i))
        if not np.isfinite(neg_logp_i):
            print(
                f"[laplace_warmup] start {i + 1}/{n_starts}: "
                f"neg_logp={neg_logp_i} (non-finite, {elapsed:.1f}s) — skipping",
                flush=True,
            )
            continue
        print(
            f"[laplace_warmup] start {i + 1}/{n_starts}: "
            f"neg_logp={neg_logp_i:.2f} ({elapsed:.1f}s)",
            flush=True,
        )
        successful_maps.append(flat_map_i)
        successful_neg_logps.append(neg_logp_i)

    n_success = len(successful_maps)
    if n_success == 0:
        print(
            "[laplace_warmup] all LBFGS starts failed; falling back",
            flush=True,
        )
        return None

    # ------------------------------------------------------------------ #
    # Select best MAP (lowest neg_logp)                                    #
    # ------------------------------------------------------------------ #
    best_idx = int(np.argmin(successful_neg_logps))
    flat_map = successful_maps[best_idx]
    map_neg_logp = successful_neg_logps[best_idx]
    print(
        f"[laplace_warmup] best MAP: start index={best_idx}, "
        f"neg_logp={map_neg_logp:.2f} ({n_success}/{n_starts} starts converged)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Hessian diagonal via Hessian-vector products: H[i,i] = e_i^T H e_i #
    # Computed as jvp(grad_neg_logp_flat, x, e_i) which is one extra      #
    # forward pass on top of the existing reverse-mode grad.  vmap'd       #
    # across the i dimension produces the full diagonal in one compiled    #
    # call.                                                                 #
    # ------------------------------------------------------------------ #
    print(
        f"[laplace_warmup] computing Hessian diagonal (n={n_flat} jvps)...",
        flush=True,
    )
    t0 = time.perf_counter()

    def hess_diag_at(i: jnp.ndarray) -> jnp.ndarray:
        e_i = jnp.zeros_like(flat_map).at[i].set(1.0)
        _, hv = jax.jvp(grad_neg_logp_flat, (flat_map,), (e_i,))
        return hv[i]

    try:
        hess_diag = jax.vmap(hess_diag_at)(jnp.arange(n_flat))
        jax.block_until_ready(hess_diag)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[laplace_warmup] Hessian-diagonal jvp raised {exc!r}; falling back",
            flush=True,
        )
        return None
    hess_diag_s = time.perf_counter() - t0
    hess_diag_np = np.asarray(hess_diag)
    n_nonpos = int((hess_diag_np <= 0).sum())
    n_nonfinite = int((~np.isfinite(hess_diag_np)).sum())
    print(
        f"[laplace_warmup] Hessian diag done in {hess_diag_s:.1f}s, "
        f"min={float(hess_diag_np.min()):.3e} "
        f"max={float(hess_diag_np.max()):.3e} "
        f"non-positive={n_nonpos} non-finite={n_nonfinite}",
        flush=True,
    )
    if n_nonfinite > 0:
        print(
            "[laplace_warmup] non-finite Hessian-diagonal entries; falling back",
            flush=True,
        )
        return None

    # Regularize: clip to >= ridge so reciprocal is bounded.  The negative
    # entries here mean the LBFGS solution wasn't a true minimum along
    # that direction — replacing with `ridge` gives a conservative
    # (large) inverse mass matrix value, which BlackJAX will refine
    # downward during sampling if too generous.
    hess_diag_pd = jnp.maximum(hess_diag, ridge)
    inverse_mass_matrix = 1.0 / hess_diag_pd
    inverse_mass_matrix_np = np.asarray(inverse_mass_matrix)

    # ------------------------------------------------------------------ #
    # Basin-comparison diagnostic (P5 prevention)                          #
    # ------------------------------------------------------------------ #
    # se_per_param = sqrt(1 / hess_diag_pd) is the Laplace posterior SD   #
    # estimate.  For each non-best successful endpoint, check whether any  #
    # parameter dimension differs from the best MAP by more than 2 SE.     #
    # If so, the posterior is likely multimodal and Laplace warmup would   #
    # lock onto one mode — return None to trigger window_adaptation.       #
    # Basin check is skipped when n_starts=1 (backward compatibility) or  #
    # when fewer than 2 starts converged.                                  #
    # ------------------------------------------------------------------ #
    se_per_param = np.sqrt(1.0 / np.asarray(hess_diag_pd))
    flat_map_np = np.asarray(flat_map)

    max_dev_overall = 0.0
    worst_dim = -1
    basin_status = "AGREE"

    if n_starts > 1 and n_success >= 2:
        for j, other_map in enumerate(successful_maps):
            if j == best_idx:
                continue
            other_np = np.asarray(other_map)
            deviations = np.abs(other_np - flat_map_np) / (se_per_param + 1e-30)
            max_dev_j = float(deviations.max())
            dim_j = int(deviations.argmax())
            if max_dev_j > max_dev_overall:
                max_dev_overall = max_dev_j
                worst_dim = dim_j
            if max_dev_j > 2.0:
                basin_status = "DISAGREE"

        if basin_status == "DISAGREE":
            print(
                f"[laplace_warmup] MULTIMODAL WARNING: endpoints disagree by "
                f"{max_dev_overall:.1f} SE on parameter dim {worst_dim} "
                f"— falling back to standard window_adaptation. "
                "This posterior may be multimodal; Laplace warmup would lock "
                "onto one mode.",
                flush=True,
            )
    elif n_starts > 1 and n_success < 2:
        print(
            f"[laplace_warmup] only {n_success}/{n_starts} starts converged; "
            "cannot perform basin comparison — falling back",
            flush=True,
        )
        print(
            f"[laplace_warmup] basin diagnostic: {n_success}/{n_starts} starts "
            f"converged, best_neg_logp={map_neg_logp:.2f}, "
            f"max_deviation=N/A (threshold=2.0, status=INSUFFICIENT)",
            flush=True,
        )
        return None

    print(
        f"[laplace_warmup] basin diagnostic: {n_success}/{n_starts} starts "
        f"converged, best_neg_logp={map_neg_logp:.2f}, "
        f"max_deviation={max_dev_overall:.2f} "
        f"(threshold=2.0, status={basin_status})",
        flush=True,
    )

    if basin_status == "DISAGREE":
        return None

    # ------------------------------------------------------------------ #
    # Build warmup_params from best MAP Hessian                            #
    # ------------------------------------------------------------------ #
    # Heuristic step size: sqrt of median IMM gives an O(1) leapfrog
    # displacement under the preconditioned Hamiltonian.  BlackJAX's
    # default initial step_size when window_adaptation runs is 1.0, so
    # this is starting closer to a workable scale.
    step_size = float(np.sqrt(np.median(inverse_mass_matrix_np)))
    cond = float(
        inverse_mass_matrix_np.max() / max(inverse_mass_matrix_np.min(), 1e-30)
    )
    print(
        f"[laplace_warmup] inverse_mass_matrix: size={n_flat} "
        f"min={float(inverse_mass_matrix_np.min()):.3e} "
        f"max={float(inverse_mass_matrix_np.max()):.3e} "
        f"cond={cond:.2g}",
        flush=True,
    )
    print(
        f"[laplace_warmup] step_size = sqrt(median(IMM)) = {step_size:.4g}",
        flush=True,
    )

    return {
        "step_size": jnp.asarray(step_size),
        "inverse_mass_matrix": jnp.asarray(inverse_mass_matrix_np),
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
    max_tree_depth: int = 10,
    use_laplace_warmup: bool = False,
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
    is_mass_matrix_diagonal: bool = True,
    use_shard_map: bool = False,
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
        shard_map path adjusts device count).
    adapted_params : dict
        Adapted NUTS parameters (``step_size`` and
        ``inverse_mass_matrix``).  Pass back as ``warmup_params`` on
        subsequent calls to skip warmup.
    """
    import blackjax

    _t_fn0 = time.perf_counter()
    _p_axis = jax.tree_util.tree_leaves(initial_position)[0].shape[0]
    print(
        f"[hierarchical] _run_blackjax_nuts entered: model={model_name} "
        f"P={_p_axis} n_chains={n_chains} n_tune={n_tune} n_draws={n_draws} "
        f"warmup_skipped={warmup_params is not None}",
        flush=True,
    )

    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    warmup_state = None  # Set by window_adaptation if warmup runs

    # Variant 2 (Phase 14.2): if use_laplace_warmup is set and no
    # warmup_params were passed in, compute (step_size, inverse_mass_matrix)
    # via Laplace approximation and feed them into the existing
    # warmup_params skip-window-adaptation hook.  Falls back to the
    # default warmup if Laplace fails.
    if warmup_params is None and use_laplace_warmup:
        _t_lp0 = time.perf_counter()
        print(
            f"[hierarchical t={_t_lp0 - _t_fn0:.1f}s] "
            "computing Laplace warmup_params (multi-start, n_starts=4)...",
            flush=True,
        )
        laplace_params = _laplace_warmup_params(logdensity_fn, initial_position)
        if laplace_params is not None:
            warmup_params = laplace_params
            print(
                f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
                f"Laplace warmup_params computed in "
                f"{time.perf_counter() - _t_lp0:.1f}s "
                "— window_adaptation will be skipped",
                flush=True,
            )
        else:
            print(
                f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
                "Laplace warmup failed; falling back to standard "
                "window_adaptation",
                flush=True,
            )

    # Phase 1: Window adaptation (skip if pre-adapted params provided)
    if warmup_params is None:
        _t_w0 = time.perf_counter()
        print(
            f"[hierarchical t={_t_w0 - _t_fn0:.1f}s] starting window_adaptation "
            f"(num_steps={n_tune}, target_accept={target_accept})",
            flush=True,
        )
        # Mitigation M1 (dense mass matrix): controlled by
        # is_mass_matrix_diagonal param threaded from FitConfig.
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=target_accept,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            max_num_doublings=max_tree_depth,
        )
        (warmup_state, warmup_params), warmup_info = warmup.run(
            warmup_key,
            initial_position,
            num_steps=n_tune,
        )
        # Block on the warmup outputs so the timing reflects compile+execute.
        jax.block_until_ready(warmup_state.position)
        _warmup_elapsed = time.perf_counter() - _t_w0
        print(
            f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
            f"window_adaptation complete in {_warmup_elapsed:.1f}s",
            flush=True,
        )
        try:
            _log_nuts_diagnostics(
                "warmup",
                warmup_info,
                n_steps=n_tune,
                n_chains=n_chains,
                p_axis=_p_axis,
                elapsed_s=_warmup_elapsed,
                adapted_step_size=warmup_params.get("step_size"),
                adapted_inverse_mass_matrix=warmup_params.get("inverse_mass_matrix"),
            )
        except Exception as exc:  # noqa: BLE001
            # Diagnostics must never break a fit — log and continue.
            print(f"[diag warmup] FAILED: {exc!r}", flush=True)
    else:
        print(
            f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
            "skipping window_adaptation (warmup_params provided)",
            flush=True,
        )

    # Phase 2: Determine chain strategy
    n_devices = jax.device_count()
    if use_shard_map:
        # Phase 31: forced shard_map dispatch. When n_devices < n_chains,
        # the shard body uses vmap for local_n = n_chains // n_devices chains.
        use_multi_device = True
    else:
        use_multi_device = n_devices >= n_chains
    _shard_source = "forced" if use_shard_map else "automatic"
    print(
        f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
        f"chain strategy: use_shard_map={use_multi_device} "
        f"({_shard_source}, n_devices={n_devices}, n_chains={n_chains})",
        flush=True,
    )

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
        _t_b0 = time.perf_counter()
        print(
            f"[hierarchical t={_t_b0 - _t_fn0:.1f}s] building sample loop "
            f"(traced-arg path, log_every={log_every})",
            flush=True,
        )
        sample_loop = _build_sample_loop(
            batched_logp_fn,
            model_name,
            n_chains,
            n_draws,
            use_multi_device,
            log_every=log_every,
            phase_label=phase_label,
            max_num_doublings=max_tree_depth,
            prior_spec=prior_spec,
        )
        print(
            f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
            f"sample loop built in {time.perf_counter() - _t_b0:.1f}s",
            flush=True,
        )

        # init_position: use adapted position from warmup if available,
        # otherwise the prior-mode initial_position passed in.  Either
        # way, sample_loop calls nuts.init() inside JIT — value_and_grad
        # of the closure-free logdensity is compiled with traced data
        # and cached.
        init_pos = (
            warmup_state.position if warmup_state is not None else initial_position
        )

        _t_s0 = time.perf_counter()
        print(
            f"[hierarchical t={_t_s0 - _t_fn0:.1f}s] sample_loop dispatch "
            "(compile+sample begins)",
            flush=True,
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
        # Block on positions to ensure the JIT'd scan completes before we
        # log "sample_loop complete" — without this, the elapsed time
        # would understate the true wall clock by the async dispatch lag.
        jax.tree_util.tree_map(jax.block_until_ready, all_states.position)
        _sample_elapsed = time.perf_counter() - _t_s0
        print(
            f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
            f"sample_loop complete in {_sample_elapsed:.1f}s",
            flush=True,
        )
        try:
            _log_nuts_diagnostics(
                "sample",
                all_infos,
                n_steps=n_draws,
                n_chains=n_chains,
                p_axis=_p_axis,
                elapsed_s=_sample_elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[diag sample] FAILED: {exc!r}", flush=True)

        # Post-process: convert JAX arrays to numpy
        _t_p0 = time.perf_counter()
        if use_multi_device:
            # shard_map: (n_chains, n_draws, P) -- already correct layout
            positions_dict = {k: np.asarray(v) for k, v in all_states.position.items()}
            stats_dict = _extract_nuts_stats(all_infos, transpose=False)
        else:
            # vmap: (n_draws, n_chains, P) -> (n_chains, n_draws, P)
            positions_dict = {
                k: np.asarray(jnp.transpose(v, (1, 0, 2)))
                for k, v in all_states.position.items()
            }
            stats_dict = _extract_nuts_stats(all_infos, transpose=True)
        print(
            f"[hierarchical t={time.perf_counter() - _t_fn0:.1f}s] "
            f"post-process complete in {time.perf_counter() - _t_p0:.1f}s "
            "(traced-arg path returning)",
            flush=True,
        )

        return positions_dict, stats_dict, n_chains, warmup_params

    # Legacy fallback: closure-based chain runners.
    # BlackJAX >=1.x returns ``max_num_doublings`` inside ``warmup_params``; merge
    # so our explicit value wins instead of triggering a duplicate-kwarg TypeError.
    nuts = blackjax.nuts(
        logdensity_fn,
        **{**warmup_params, "max_num_doublings": max_tree_depth},
    )

    # Build warmup_state if it wasn't created by window_adaptation
    # (happens when warmup_params was provided to skip warmup)
    if warmup_state is None:
        warmup_state = nuts.init(initial_position)

    if use_multi_device:
        positions, stats, n_actual = _run_shard_map_chains(
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


def _run_shard_map_chains(
    nuts,  # noqa: ANN001
    warmup_state,  # noqa: ANN001
    sample_key: jnp.ndarray,
    n_draws: int,
    n_chains: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Run multiple MCMC chains via shard_map across devices.

    Uses ``jax.experimental.shard_map.shard_map`` over a 1-D Mesh
    (axis name ``"chains"``) for chain-axis parallelism.  Each device
    receives a unique PRNG key ensuring independent samples (P7
    prevention: pmap/shard_map accidentally broadcasting same RNG).

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
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()[:n_chains]
    mesh = Mesh(np.array(devices), axis_names=("chains",))

    # Each chain gets a unique PRNG key -- P7 prevention
    chain_keys = jax.random.split(sample_key, n_chains)

    # Replicate warmup state across chain axis
    replicated_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
        warmup_state,
    )

    def _sample_one_chain(
        rng_key: jnp.ndarray,
        state,  # noqa: ANN001
    ) -> tuple:
        def _one_step(s, k):  # noqa: ANN001
            new_s, info = nuts.step(k, s)
            return new_s, (new_s, info)

        keys = jax.random.split(rng_key, n_draws)
        _, (states, infos) = lax.scan(_one_step, state, keys)
        return states, infos

    # shard_map: leading axis is "chains", one shard per device.
    # in_specs:  chain_keys (n_chains,) and replicated_state (n_chains, ...)
    #            are both sharded on the "chains" axis.
    # out_specs: outputs have the same leading chain axis.
    @jax.jit
    def _run_sharded(chain_keys_arr, rep_state):  # noqa: ANN001
        # check_rep=False: NUTS uses lax.while_loop (tree expansion) which
        # has no replication rule in shard_map.  Outputs are still correctly
        # sharded on the "chains" axis via out_specs.
        return shard_map(
            _sample_one_chain,
            mesh=mesh,
            in_specs=(P("chains"), P("chains")),
            out_specs=(P("chains"), P("chains")),
            check_rep=False,
        )(chain_keys_arr, rep_state)

    all_states, all_infos = _run_sharded(chain_keys, replicated_state)

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
    use_multi_device: bool,
    log_every: int = 0,
    phase_label: str = "sample",
    max_num_doublings: int = 10,
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
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
    use_multi_device : bool
        If ``True``, use ``jax.experimental.shard_map`` over a 1-D Mesh
        for multi-device chain parallelism (MODEA-06 migration from pmap).
        If ``False``, use ``jax.vmap`` on a single device with
        ``@jax.jit``.
    prior_spec : HGFPriorSpec or None, optional
        Prior specification.  If ``None``, uses the default for the
        given ``model_name``.

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

    from prl_hgf.fitting.priors import HGFPriorSpec

    is_3level = model_name == "hgf_3level"

    if prior_spec is None:
        prior_spec = (
            HGFPriorSpec.default_3level()
            if is_3level
            else HGFPriorSpec.default_2level()
        )

    # Prior distributions -- parameterless JAX objects, safe to capture
    prior_omega_2 = prior_spec.omega_2.to_numpyro_dist()
    prior_log_beta = prior_spec.log_beta.to_numpyro_dist()
    prior_zeta = prior_spec.zeta.to_numpyro_dist()
    if is_3level:
        prior_omega_3 = prior_spec.omega_3.to_numpyro_dist()
        # κ frozen at _KAPPA_FIXED (1.0); no sampled prior.

    if use_multi_device:
        # shard_map path: distribute chains across devices via Mesh
        # (MODEA-06: migrated from deprecated jax.pmap)
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        devices = jax.devices()[:n_chains]
        mesh = Mesh(np.array(devices), axis_names=("chains",))

        def sample_loop_shard_map(
            init_position: dict[str, jnp.ndarray],
            warmup_params: dict,
            sample_key: jnp.ndarray,
            input_data: jnp.ndarray,
            observed: jnp.ndarray,
            choices: jnp.ndarray,
            trial_mask: jnp.ndarray,
        ) -> tuple:
            # Reconstruct logdensity_fn with traced data args
            # (identical to vmap path -- only dispatch mechanism differs)
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

            # BlackJAX >=1.x duplicates ``max_num_doublings`` inside warmup_params;
            # merge so our explicit value wins (see legacy-fallback site for context).
            nuts = blackjax.nuts(
                logdensity_fn,
                **{**warmup_params, "max_num_doublings": max_num_doublings},
            )
            # Build initial state INSIDE JIT — value_and_grad uses traced data
            initial_state = nuts.init(init_position)
            # Each chain gets a unique PRNG key (P7 prevention)
            chain_keys = jax.random.split(sample_key, n_chains)

            replicated_state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
                initial_state,
            )

            def _sample_one_chain(
                rng_key: jnp.ndarray,
                state,  # noqa: ANN001
            ) -> tuple:
                def _one_step(s, k):  # noqa: ANN001
                    new_s, info = nuts.step(k, s)
                    return new_s, (new_s, info)

                keys = jax.random.split(rng_key, n_draws)
                _, (states, infos) = lax.scan(_one_step, state, keys)
                return states, infos

            # shard_map: leading axis is "chains", one shard per device.
            # check_rep=False: NUTS uses lax.while_loop (tree expansion)
            # which has no replication rule in shard_map.
            all_states, all_infos = shard_map(
                _sample_one_chain,
                mesh=mesh,
                in_specs=(P("chains"), P("chains")),
                out_specs=(P("chains"), P("chains")),
                check_rep=False,
            )(chain_keys, replicated_state)
            return all_states, all_infos

        return sample_loop_shard_map

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

        # BlackJAX >=1.x duplicates ``max_num_doublings`` inside warmup_params;
        # merge so our explicit value wins (see legacy-fallback site for context).
        nuts = blackjax.nuts(
            logdensity_fn,
            **{**warmup_params, "max_num_doublings": max_num_doublings},
        )
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
            log_every,
            n_chains,
            phase_label,
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


def _samples_to_idata_hierarchical(
    positions: dict[str, np.ndarray],
    sample_stats: dict[str, np.ndarray],
    var_names: list[str],
    participant_ids: list[str],
    participant_groups: list[str],
    participant_sessions: list[str],
    group_labels: list[str],
    model_name: str = "hgf_3level",
) -> az.InferenceData:
    """Convert hierarchical BlackJAX samples to ArviZ InferenceData.

    Extends :func:`_samples_to_idata` with group-level coordinate handling
    for Mode B hierarchical models.  Hyperparameters (``mu_*``) get a
    ``group`` dimension; participant-level parameters get a ``participant``
    dimension; scalars (``log_sigma_*``, ``beta_*``) get no extra dims.

    Parameters
    ----------
    positions : dict[str, numpy.ndarray]
        Posterior samples.  Values have shape ``(n_chains, n_draws, ...)``.
    sample_stats : dict[str, numpy.ndarray]
        NUTS diagnostics.
    var_names : list[str]
        All variable names to include in idata.
    participant_ids : list[str]
        Participant identifier strings.
    participant_groups : list[str]
        Group labels per participant.
    participant_sessions : list[str]
        Session labels per participant.
    group_labels : list[str]
        Sorted unique group labels.
    model_name : str, optional
        ``"hgf_2level"`` or ``"hgf_3level"`` (default).

    Returns
    -------
    arviz.InferenceData
        Posterior with participant and group coordinates.
    """
    import arviz as az

    n_participants = len(participant_ids)
    n_groups = len(group_labels)

    # Build posterior dict and dims from sampled positions
    posterior_dict: dict[str, np.ndarray] = {}
    dims_dict: dict[str, list[str]] = {}

    for var in var_names:
        if var == "beta":
            # Deterministic transform of log_beta
            if "log_beta" in positions:
                posterior_dict["beta"] = np.exp(positions["log_beta"])
                dims_dict["beta"] = ["participant"]
            elif "log_beta_nc" in positions:
                # Non-centered: log_beta stored elsewhere; skip beta here
                continue
            continue
        if var not in positions:
            continue

        arr = positions[var]
        posterior_dict[var] = arr

        # Determine dimension from array shape
        # positions shape: (n_chains, n_draws, ...) or (n_chains, n_draws)
        if arr.ndim == 3:
            last_dim = arr.shape[-1]
            if last_dim == n_participants:
                dims_dict[var] = ["participant"]
            elif last_dim == n_groups:
                dims_dict[var] = ["group"]
            else:
                # Unknown dimension — leave undimmed
                pass
        # ndim == 2 means scalar parameter — no extra dims

    coords_dict: dict[str, list[str]] = {
        "participant": participant_ids,
        "group": group_labels,
    }

    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=sample_stats,
        dims=dims_dict,
        coords=coords_dict,
    )

    # Attach group and session metadata as additional coords
    idata.posterior = idata.posterior.assign_coords(
        participant_group=("participant", participant_groups),
        participant_session=("participant", participant_sessions),
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
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
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
    prior_spec : HGFPriorSpec or None, optional
        Prior specification.  If ``None``, uses default 3-level priors.
    """
    import numpyro

    from prl_hgf.fitting.priors import HGFPriorSpec

    if prior_spec is None:
        prior_spec = HGFPriorSpec.default_3level()

    # Perceptual parameters
    omega_2 = numpyro.sample(
        "omega_2",
        prior_spec.omega_2.to_numpyro_dist().expand([n_participants]),
    )
    omega_3 = numpyro.sample(
        "omega_3",
        prior_spec.omega_3.to_numpyro_dist().expand([n_participants]),
    )
    # κ frozen at _KAPPA_FIXED (1.0) — collapses ω₃×κ ridge that otherwise
    # saturates NUTS tree depth (see module constant docstring).
    kappa = jnp.full((n_participants,), _KAPPA_FIXED)

    # Response parameters
    log_beta = numpyro.sample(
        "log_beta",
        prior_spec.log_beta.to_numpyro_dist().expand([n_participants]),
    )
    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    zeta = numpyro.sample(
        "zeta",
        prior_spec.zeta.to_numpyro_dist().expand([n_participants]),
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
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
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
    prior_spec : HGFPriorSpec or None, optional
        Prior specification.  If ``None``, uses default 2-level priors.
    """
    import numpyro

    from prl_hgf.fitting.priors import HGFPriorSpec

    if prior_spec is None:
        prior_spec = HGFPriorSpec.default_2level()

    # Perceptual parameter
    omega_2 = numpyro.sample(
        "omega_2",
        prior_spec.omega_2.to_numpyro_dist().expand([n_participants]),
    )

    # Response parameters
    log_beta = numpyro.sample(
        "log_beta",
        prior_spec.log_beta.to_numpyro_dist().expand([n_participants]),
    )
    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    zeta = numpyro.sample(
        "zeta",
        prior_spec.zeta.to_numpyro_dist().expand([n_participants]),
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
# NumPyro hierarchical (Mode B) model functions
# ---------------------------------------------------------------------------


def _numpyro_model_hierarchical_2level(
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    n_groups: int,
    batched_logp_fn,  # noqa: ANN001
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
    group_idx=None,  # noqa: ANN001  # jnp.ndarray | None
    x_covariate=None,  # noqa: ANN001  # jnp.ndarray | None
) -> None:
    """NumPyro hierarchical model: 2-level HGF with group hyperpriors.

    Implements Mode B hierarchical pooling per Boehm 2018.  Each cognitive
    parameter has a group-level mean (``mu_p``) per group and a shared
    population spread (``sigma_p``).  Participant-level parameters are
    drawn from ``Normal(mean_p, sigma_p)`` (required for LocScaleReparam).

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
    n_groups : int
        Number of experimental groups ``K``.
    batched_logp_fn : callable
        Pure JAX batched logp from :func:`build_logp_fn_batched`.
    prior_spec : HGFPriorSpec or None, optional
        Prior specification with hyperprior fields populated.
        If ``None``, uses default 2-level hierarchical priors.
    group_idx : jnp.ndarray or None, shape (P,)
        Integer group assignment per participant (0-indexed).
        Required for group-level hyperpriors.
    x_covariate : jnp.ndarray or None, shape (P,)
        Optional continuous covariate (mean-centered internally).
    """
    import numpyro
    import numpyro.distributions as dist

    from prl_hgf.fitting.priors import HGFPriorSpec

    if prior_spec is None:
        prior_spec = HGFPriorSpec.default_2level_hierarchical()

    has_covariate = x_covariate is not None
    if has_covariate:
        x_c = x_covariate - jnp.mean(x_covariate)

    # --- omega_2 hyperpriors ---
    mu_omega2 = numpyro.sample(
        "mu_omega2",
        prior_spec.omega_2_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_omega2 = numpyro.sample(
        "sigma_omega2",
        prior_spec.omega_2_sigma_hyper.to_numpyro_dist(),
    )
    mean_omega2_p = mu_omega2[group_idx]
    if has_covariate:
        beta_omega2 = numpyro.sample(
            "beta_omega2", dist.Normal(0.0, 1.0)
        )
        mean_omega2_p = mean_omega2_p + beta_omega2 * x_c

    # --- log_beta hyperpriors ---
    mu_log_beta = numpyro.sample(
        "mu_log_beta",
        prior_spec.log_beta_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_log_beta = numpyro.sample(
        "sigma_log_beta",
        prior_spec.log_beta_sigma_hyper.to_numpyro_dist(),
    )
    mean_log_beta_p = mu_log_beta[group_idx]
    if has_covariate:
        beta_log_beta = numpyro.sample(
            "beta_log_beta", dist.Normal(0.0, 1.0)
        )
        mean_log_beta_p = mean_log_beta_p + beta_log_beta * x_c

    # --- zeta hyperpriors ---
    mu_zeta = numpyro.sample(
        "mu_zeta",
        prior_spec.zeta_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_zeta = numpyro.sample(
        "sigma_zeta",
        prior_spec.zeta_sigma_hyper.to_numpyro_dist(),
    )
    mean_zeta_p = mu_zeta[group_idx]
    if has_covariate:
        beta_zeta = numpyro.sample(
            "beta_zeta", dist.Normal(0.0, 1.0)
        )
        mean_zeta_p = mean_zeta_p + beta_zeta * x_c

    # --- Per-participant parameters (Normal for LocScaleReparam compat) ---
    with numpyro.plate("participants", n_participants):
        omega_2 = numpyro.sample(
            "omega_2", dist.Normal(mean_omega2_p, sigma_omega2)
        )
        log_beta = numpyro.sample(
            "log_beta", dist.Normal(mean_log_beta_p, sigma_log_beta)
        )
        zeta = numpyro.sample(
            "zeta", dist.Normal(mean_zeta_p, sigma_zeta)
        )

    beta = numpyro.deterministic("beta", jnp.exp(log_beta))

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


def _numpyro_model_hierarchical_3level(
    input_data: jnp.ndarray,
    observed: jnp.ndarray,
    choices: jnp.ndarray,
    trial_mask: jnp.ndarray,
    n_participants: int,
    n_groups: int,
    batched_logp_fn,  # noqa: ANN001
    prior_spec=None,  # noqa: ANN001  # HGFPriorSpec | None
    group_idx=None,  # noqa: ANN001  # jnp.ndarray | None
    x_covariate=None,  # noqa: ANN001  # jnp.ndarray | None
) -> None:
    """NumPyro hierarchical model: 3-level HGF with group hyperpriors.

    Implements Mode B hierarchical pooling per Boehm 2018 for the 3-level
    HGF.  Extends the 2-level variant with omega_3 hyperpriors.  Kappa is
    frozen at ``_KAPPA_FIXED`` (1.0) to collapse the multiplicative ridge.

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
    n_groups : int
        Number of experimental groups ``K``.
    batched_logp_fn : callable
        Pure JAX batched logp from :func:`build_logp_fn_batched`.
    prior_spec : HGFPriorSpec or None, optional
        Prior specification with hyperprior fields populated.
        If ``None``, uses default 3-level hierarchical priors.
    group_idx : jnp.ndarray or None, shape (P,)
        Integer group assignment per participant (0-indexed).
        Required for group-level hyperpriors.
    x_covariate : jnp.ndarray or None, shape (P,)
        Optional continuous covariate (mean-centered internally).
    """
    import numpyro
    import numpyro.distributions as dist

    from prl_hgf.fitting.priors import HGFPriorSpec

    if prior_spec is None:
        prior_spec = HGFPriorSpec.default_3level_hierarchical()

    has_covariate = x_covariate is not None
    if has_covariate:
        x_c = x_covariate - jnp.mean(x_covariate)

    # --- omega_2 hyperpriors ---
    mu_omega2 = numpyro.sample(
        "mu_omega2",
        prior_spec.omega_2_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_omega2 = numpyro.sample(
        "sigma_omega2",
        prior_spec.omega_2_sigma_hyper.to_numpyro_dist(),
    )
    mean_omega2_p = mu_omega2[group_idx]
    if has_covariate:
        beta_omega2 = numpyro.sample(
            "beta_omega2", dist.Normal(0.0, 1.0)
        )
        mean_omega2_p = mean_omega2_p + beta_omega2 * x_c

    # --- log_beta hyperpriors ---
    mu_log_beta = numpyro.sample(
        "mu_log_beta",
        prior_spec.log_beta_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_log_beta = numpyro.sample(
        "sigma_log_beta",
        prior_spec.log_beta_sigma_hyper.to_numpyro_dist(),
    )
    mean_log_beta_p = mu_log_beta[group_idx]
    if has_covariate:
        beta_log_beta = numpyro.sample(
            "beta_log_beta", dist.Normal(0.0, 1.0)
        )
        mean_log_beta_p = mean_log_beta_p + beta_log_beta * x_c

    # --- zeta hyperpriors ---
    mu_zeta = numpyro.sample(
        "mu_zeta",
        prior_spec.zeta_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_zeta = numpyro.sample(
        "sigma_zeta",
        prior_spec.zeta_sigma_hyper.to_numpyro_dist(),
    )
    mean_zeta_p = mu_zeta[group_idx]
    if has_covariate:
        beta_zeta = numpyro.sample(
            "beta_zeta", dist.Normal(0.0, 1.0)
        )
        mean_zeta_p = mean_zeta_p + beta_zeta * x_c

    # --- omega_3 hyperpriors ---
    mu_omega3 = numpyro.sample(
        "mu_omega3",
        prior_spec.omega_3_mu_hyper.to_numpyro_dist().expand([n_groups]),
    )
    sigma_omega3 = numpyro.sample(
        "sigma_omega3",
        prior_spec.omega_3_sigma_hyper.to_numpyro_dist(),
    )
    mean_omega3_p = mu_omega3[group_idx]
    if has_covariate:
        beta_omega3 = numpyro.sample(
            "beta_omega3", dist.Normal(0.0, 1.0)
        )
        mean_omega3_p = mean_omega3_p + beta_omega3 * x_c

    # --- Per-participant parameters (Normal for LocScaleReparam compat) ---
    with numpyro.plate("participants", n_participants):
        omega_2 = numpyro.sample(
            "omega_2", dist.Normal(mean_omega2_p, sigma_omega2)
        )
        log_beta = numpyro.sample(
            "log_beta", dist.Normal(mean_log_beta_p, sigma_log_beta)
        )
        zeta = numpyro.sample(
            "zeta", dist.Normal(mean_zeta_p, sigma_zeta)
        )
        omega_3 = numpyro.sample(
            "omega_3", dist.Normal(mean_omega3_p, sigma_omega3)
        )

    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    # kappa frozen at _KAPPA_FIXED (collapses omega_3 x kappa ridge)
    kappa = jnp.full((n_participants,), _KAPPA_FIXED)

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


# ---------------------------------------------------------------------------
# LocScaleReparam application helper
# ---------------------------------------------------------------------------


def _apply_reparam(model_fn, non_centered: tuple[str, ...]):
    """Wrap a NumPyro model with LocScaleReparam for specified sites.

    Applies fully non-centered reparameterization (``centered=0``) to
    each site named in ``non_centered``.  Sites must have Normal
    distribution (unconstrained support) for LocScaleReparam to work.

    When ``non_centered`` is empty, returns the model unchanged (no-op).

    Parameters
    ----------
    model_fn : callable
        NumPyro model function.
    non_centered : tuple[str, ...]
        Parameter names to apply non-centered reparameterization.
        Must be sites with Normal distribution (unconstrained support).

    Returns
    -------
    callable
        Wrapped model with LocScaleReparam applied to specified sites.
    """
    if not non_centered:
        return model_fn
    from numpyro import handlers
    from numpyro.infer.reparam import LocScaleReparam

    reparam_config = {
        name: LocScaleReparam(centered=0) for name in non_centered
    }
    return handlers.reparam(config=reparam_config)(model_fn)


def _get_numpyro_model_hierarchical(
    model_name: str, non_centered: tuple[str, ...]
):
    """Route to the correct hierarchical NumPyro model and apply reparam.

    Selects between 2-level and 3-level hierarchical model functions
    based on ``model_name``, then wraps with LocScaleReparam for the
    sites listed in ``non_centered``.

    Parameters
    ----------
    model_name : str
        Model identifier: ``"hgf_2level"`` or ``"hgf_3level"``.
    non_centered : tuple[str, ...]
        Parameter names for non-centered reparameterization.

    Returns
    -------
    callable
        NumPyro model function (possibly reparam-wrapped).

    Raises
    ------
    ValueError
        If ``model_name`` is not recognized.
    """
    if model_name == "hgf_2level":
        model_fn = _numpyro_model_hierarchical_2level
    elif model_name == "hgf_3level":
        model_fn = _numpyro_model_hierarchical_3level
    else:
        msg = (
            f"Unknown model_name for hierarchical NumPyro model: "
            f"{model_name!r}. Expected one of {_MODEL_NAMES}."
        )
        raise ValueError(msg)
    return _apply_reparam(model_fn, non_centered)


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
    fit_config: FitConfig,
    prior_spec: HGFPriorSpec | None = None,
    warmup_params: dict | None = None,
    x_covariate: np.ndarray | None = None,
) -> az.InferenceData | tuple[az.InferenceData, dict]:
    """Fit an entire cohort via BlackJAX NUTS (default) or NumPyro MCMC.

    Groups ``sim_df`` by ``(participant_id, group, session)``, builds the
    stacked ``(P, n_trials, 3)`` arrays, constructs a pure JAX logp via
    :func:`build_logp_fn_batched`, and runs NUTS for the full cohort.
    Returns an ``InferenceData`` with a ``participant`` dimension on every
    parameter so downstream analysis can map posterior slices back to
    individual participants.

    **Mode A** (``fit_config.covariate.pooling == "none"``): Independent
    priors per participant — no hyperpriors, no partial pooling.

    **Mode B** (``fit_config.covariate.pooling == "hierarchical"``): Group-
    level hyperpriors per Boehm 2018.  Routes to hierarchical model
    builders that implement partial pooling with optional covariates.

    **BlackJAX path** (default): Builds a pure JAX log-posterior via
    :func:`_build_log_posterior` (Mode A) or
    :func:`_build_log_posterior_hierarchical` (Mode B), runs NUTS with
    window adaptation via :func:`_run_blackjax_nuts`, and converts to
    ArviZ via :func:`_samples_to_idata`.  Compiles the NUTS step function
    once via ``jax.jit`` and reuses it across all MCMC steps — no per-call
    recompilation.

    **NumPyro path** (fallback): Uses numpyro MCMC with
    ``chain_method="vectorized"`` and ``jit_model_args=True``.  Data
    arrays are passed as kwargs to ``MCMC.run()`` for JIT cache reuse.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level DataFrame with columns ``participant_id``, ``group``,
        ``session``, ``cue_chosen``, ``reward``.
    fit_config : FitConfig
        Complete fitting configuration — sampler backend, chain count,
        draw count, warmup steps, target acceptance, tree depth, mitigation
        flags, and logging options.  Single source of truth for all
        sampling settings.
    prior_spec : HGFPriorSpec or None, optional
        Prior distributions for model parameters.  If ``None`` (default),
        derived from ``fit_config.model_name`` and pooling mode: Mode A
        uses :meth:`HGFPriorSpec.default_2level` / ``default_3level``;
        Mode B uses ``default_2level_hierarchical`` /
        ``default_3level_hierarchical``.
    warmup_params : dict or None, optional
        Pre-adapted NUTS parameters from a previous call.  When
        provided, warmup is skipped (~1100s savings per call).  Pass
        the second element of the return tuple from a prior call.
        Only used with the BlackJAX path.
    x_covariate : numpy.ndarray or None, optional
        Continuous covariate array of shape ``(P,)`` matching participant
        order from the groupby.  Mean-centered internally before passing
        to model functions.  Only used when
        ``fit_config.covariate.pooling == "hierarchical"``.

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
    from prl_hgf.fitting.preflight import validate_fit_config

    # ------------------------------------------------------------------
    # Extract settings from FitConfig
    # ------------------------------------------------------------------
    model_name = fit_config.model_name
    n_chains = fit_config.sampler.n_chains
    n_draws = fit_config.sampler.n_draws
    n_tune = fit_config.sampler.n_warmup
    target_accept = fit_config.sampler.target_accept
    random_seed = fit_config.sampler.random_seed
    sampler = fit_config.sampler.backend
    progressbar = fit_config.progressbar
    log_every = fit_config.log_every
    max_tree_depth = fit_config.sampler.max_tree_depth
    use_laplace_warmup = fit_config.mitigation.use_laplace_warmup
    mass_matrix_kind = fit_config.mitigation.mass_matrix_kind

    if mass_matrix_kind == "low_rank":
        warnings.warn(
            "mass_matrix_kind='low_rank' has no BlackJAX 1.5 backend support "
            "(window_adaptation only exposes is_mass_matrix_diagonal: bool). "
            "Falling through to 'dense' (is_mass_matrix_diagonal=False). "
            "Use 'dense' explicitly to suppress this warning, or wait for a "
            "future BlackJAX release with low-rank API.",
            UserWarning,
            stacklevel=2,
        )
        mass_matrix_kind = "dense"

    is_mass_matrix_diagonal = mass_matrix_kind == "diagonal"

    # ------------------------------------------------------------------
    # Derive prior_spec if not provided
    # ------------------------------------------------------------------
    is_hierarchical = fit_config.covariate.pooling == "hierarchical"
    if prior_spec is None:
        if is_hierarchical:
            if model_name == "hgf_3level":
                prior_spec = HGFPriorSpec.default_3level_hierarchical()
            else:
                prior_spec = HGFPriorSpec.default_2level_hierarchical()
        else:
            if model_name == "hgf_3level":
                prior_spec = HGFPriorSpec.default_3level()
            else:
                prior_spec = HGFPriorSpec.default_2level()

    # Pre-flight validation (n_participants not yet known; deferred below)

    _t_fb0 = time.perf_counter()
    print(
        f"[fit_batch_hierarchical] entered: model={model_name} "
        f"sampler={sampler} n_chains={n_chains} n_tune={n_tune} "
        f"n_draws={n_draws} target_accept={target_accept} "
        f"warmup_skipped={warmup_params is not None} "
        f"sim_df_rows={len(sim_df)}",
        flush=True,
    )

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

    # ------------------------------------------------------------------
    # Extract group_idx and mean-center covariate (Mode B only)
    # ------------------------------------------------------------------
    if is_hierarchical:
        group_labels = sorted(set(participant_groups))
        group_idx = np.array(
            [group_labels.index(g) for g in participant_groups]
        )
        n_groups = len(group_labels)
        # Mean-center covariate before passing to model functions
        if x_covariate is not None:
            if x_covariate.shape != (n_participants,):
                msg = (
                    f"x_covariate shape mismatch: expected ({n_participants},), "
                    f"got {x_covariate.shape}"
                )
                raise ValueError(msg)
            # P8 prevention: refuse collinear covariate before expensive work
            from prl_hgf.fitting.preflight import (
                check_covariate_collinearity,
            )

            check_covariate_collinearity(x_covariate, group_idx)
            x_covariate_centered = x_covariate - np.mean(x_covariate)
        else:
            x_covariate_centered = None

    # Pre-flight validation (memory guard for dense mass matrix)
    validate_fit_config(fit_config, prior_spec, n_participants)

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
        print(
            f"[fit_batch_hierarchical t={time.perf_counter() - _t_fb0:.1f}s] "
            f"cohort assembled: P={n_participants} n_trials={n_trials} "
            f"(BlackJAX path, {'Mode B' if is_hierarchical else 'Mode A'})",
            flush=True,
        )

        # Build log-posterior (priors + batched HGF likelihood)
        _t_lp0 = time.perf_counter()
        print(
            f"[fit_batch_hierarchical t={_t_lp0 - _t_fb0:.1f}s] "
            "building closure-based logdensity (warmup-only, no JIT yet)",
            flush=True,
        )

        if is_hierarchical:
            # ----------------------------------------------------------
            # Mode B: hierarchical log-posterior with hyperpriors
            # ----------------------------------------------------------
            jax_x_covariate = (
                jnp.array(x_covariate_centered, dtype=jnp.float32)
                if x_covariate_centered is not None
                else None
            )
            logdensity_fn = _build_log_posterior_hierarchical(
                logp_fn,
                jax_input_data,
                jax_observed,
                jax_choices,
                jax_trial_mask,
                n_participants,
                n_groups,
                jnp.array(group_idx, dtype=jnp.int32),
                model_name,
                prior_spec=prior_spec,
                non_centered=fit_config.mitigation.non_centered,
                x_covariate=jax_x_covariate,
            )
        else:
            # ----------------------------------------------------------
            # Mode A: independent priors per participant (no hyperpriors)
            # ----------------------------------------------------------
            logdensity_fn = _build_log_posterior(
                logp_fn,
                jax_input_data,
                jax_observed,
                jax_choices,
                jax_trial_mask,
                n_participants,
                model_name,
                prior_spec=prior_spec,
            )

        print(
            f"[fit_batch_hierarchical t={time.perf_counter() - _t_fb0:.1f}s] "
            f"logdensity built in {time.perf_counter() - _t_lp0:.1f}s",
            flush=True,
        )

        # Build initial position dict at prior modes.
        if is_hierarchical:
            # Mode B initial position: hyperpriors + participant-level
            jax_x_cov_init = (
                jnp.array(x_covariate_centered, dtype=jnp.float32)
                if x_covariate_centered is not None
                else None
            )
            initial_position = _build_initial_position_hierarchical(
                n_participants,
                n_groups,
                model_name,
                prior_spec=prior_spec,
                non_centered=fit_config.mitigation.non_centered,
                x_covariate=jax_x_cov_init,
            )
            # var_names for hierarchical idata include hyperprior keys
            h_params = (
                _HIERARCHICAL_PARAMS_3LEVEL
                if model_name == "hgf_3level"
                else _HIERARCHICAL_PARAMS_2LEVEL
            )
            var_names: list[str] = []
            for p_name in h_params:
                var_names.append(f"mu_{p_name}")
                var_names.append(f"log_sigma_{p_name}")
                if x_covariate_centered is not None:
                    var_names.append(f"beta_{p_name}")
                # Participant-level key
                if p_name in fit_config.mitigation.non_centered:
                    var_names.append(f"{p_name}_nc")
                else:
                    var_names.append(p_name)
            # Add deterministic beta (exp(log_beta))
            var_names.append("beta")
        else:
            # Mode A initial position: κ frozen — not sampled.
            if model_name == "hgf_3level":
                initial_position = {
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
        _t_nuts0 = time.perf_counter()
        print(
            f"[fit_batch_hierarchical t={_t_nuts0 - _t_fb0:.1f}s] "
            "dispatching to _run_blackjax_nuts",
            flush=True,
        )
        positions, sample_stats, n_chains_actual, adapted_params = _run_blackjax_nuts(
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
            max_tree_depth=max_tree_depth,
            use_laplace_warmup=use_laplace_warmup,
            prior_spec=prior_spec,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            use_shard_map=fit_config.mitigation.use_shard_map,
        )
        print(
            f"[fit_batch_hierarchical t={time.perf_counter() - _t_fb0:.1f}s] "
            f"_run_blackjax_nuts returned in "
            f"{time.perf_counter() - _t_nuts0:.1f}s",
            flush=True,
        )

        # Convert to ArviZ InferenceData
        _t_id0 = time.perf_counter()
        if is_hierarchical:
            # Mode B: hierarchical idata with group coords
            idata = _samples_to_idata_hierarchical(
                positions,
                sample_stats,
                var_names,
                participant_ids,
                participant_groups,
                participant_sessions,
                group_labels,
                model_name,
            )
        else:
            idata = _samples_to_idata(
                positions,
                sample_stats,
                var_names,
                participant_ids,
                participant_groups,
                participant_sessions,
                model_name,
            )
        print(
            f"[fit_batch_hierarchical t={time.perf_counter() - _t_fb0:.1f}s] "
            f"_samples_to_idata complete in {time.perf_counter() - _t_id0:.1f}s "
            f"(BlackJAX path returning, total wall "
            f"{time.perf_counter() - _t_fb0:.1f}s)",
            flush=True,
        )

        # Provenance: record full config in idata attrs
        idata.attrs["fit_config"] = fit_config.to_json()

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

        if is_hierarchical:
            # ----------------------------------------------------------
            # Mode B: hierarchical NumPyro model with reparam
            # ----------------------------------------------------------
            model_fn = _get_numpyro_model_hierarchical(
                model_name, fit_config.mitigation.non_centered
            )
            # var_names for hierarchical models
            h_params = (
                _HIERARCHICAL_PARAMS_3LEVEL
                if model_name == "hgf_3level"
                else _HIERARCHICAL_PARAMS_2LEVEL
            )
            var_names = []
            for p_name in h_params:
                var_names.extend([
                    f"mu_{p_name}",
                    f"sigma_{p_name}",
                    p_name,
                ])
                if x_covariate is not None:
                    var_names.append(f"beta_{p_name}")
            var_names.append("beta")  # deterministic exp(log_beta)
        else:
            # ----------------------------------------------------------
            # Mode A: independent priors per participant
            # ----------------------------------------------------------
            if model_name == "hgf_3level":
                model_fn = _numpyro_model_3level
                # κ frozen — not in fitted var_names.
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

        if is_hierarchical:
            bound_model = partial(
                model_fn,
                n_participants=n_participants,
                n_groups=n_groups,
                batched_logp_fn=logp_fn,
                prior_spec=prior_spec,
            )
        else:
            bound_model = partial(
                model_fn,
                n_participants=n_participants,
                batched_logp_fn=logp_fn,
                prior_spec=prior_spec,
            )
        kernel = NUTS(
            bound_model,
            target_accept_prob=target_accept,
            dense_mass=(mass_matrix_kind != "diagonal"),
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=n_tune,
            num_samples=n_draws,
            num_chains=n_chains,
            chain_method="vectorized",
            jit_model_args=True,
            progress_bar=progressbar,
        )

        if is_hierarchical:
            # Mode B: pass group_idx and covariate to NumPyro model
            jax_group_idx = jnp.array(group_idx, dtype=jnp.int32)
            jax_x_covariate = (
                jnp.array(x_covariate_centered, dtype=jnp.float32)
                if x_covariate_centered is not None
                else None
            )
            mcmc.run(
                rng_key,
                extra_fields=("num_steps", "mean_accept_prob"),
                input_data=jax_input_data,
                observed=jax_observed,
                choices=jax_choices,
                trial_mask=jax_trial_mask,
                group_idx=jax_group_idx,
                x_covariate=jax_x_covariate,
            )
        else:
            mcmc.run(
                rng_key,
                extra_fields=("num_steps", "mean_accept_prob"),
                input_data=jax_input_data,
                observed=jax_observed,
                choices=jax_choices,
                trial_mask=jax_trial_mask,
            )

        # Convert to ArviZ InferenceData with participant coords
        if is_hierarchical:
            # Mode B: add group dimension for hyperparameters
            dims_dict: dict[str, list[str]] = {}
            for vn in var_names:
                if vn.startswith("mu_"):
                    dims_dict[vn] = ["group"]
                elif vn in (
                    "omega_2", "omega_3", "log_beta", "zeta", "beta"
                ):
                    dims_dict[vn] = ["participant"]
                # sigma_* and beta_* are scalar — no dims needed
            coords_dict: dict[str, list[str]] = {
                "participant": participant_ids,
                "group": group_labels,
            }
        else:
            dims_dict = {vn: ["participant"] for vn in var_names}
            coords_dict = {
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

    # Provenance: record full config in idata attrs
    idata.attrs["fit_config"] = fit_config.to_json()

    return idata


__all__ = [
    "_build_sample_loop",
    "build_logp_fn_batched",
    "build_logp_ops_batched",
    "build_pymc_model_batched",
    "fit_batch_hierarchical",
]
