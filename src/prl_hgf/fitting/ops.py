"""Custom PyTensor Ops wrapping JAX-compiled HGF logp and gradients.

Implements the two-Op split pattern from pyhgf/distribution.py:

* ``_GradOp`` — calls ``jax.value_and_grad`` and returns the raw gradients.
* ``_LogpOp`` — calls ``jax.jit(_jax_logp)`` and delegates gradients to
  ``_GradOp``.

Two public factory functions are exported:

* :func:`build_logp_ops_2level` — wraps the 2-level binary HGF.
  Op signature: ``logp_op(omega_2, beta, zeta) -> scalar``.
* :func:`build_logp_ops_3level` — wraps the 3-level binary HGF.
  Op signature: ``logp_op(omega_2, omega_3, kappa, beta, zeta) -> scalar``.

Each factory builds a fresh Network, freezes its scan_fn and attributes
once, then returns a pre-compiled Op ready for use inside a PyMC model.

Notes
-----
Parameter injection uses the shallow-copy pattern: the outer attributes dict
is shallow-copied, then each modified node dict is also shallow-copied.
This preserves JAX traceability (deepcopy breaks it).

The NaN guard returns ``-jnp.inf`` (not ``+inf``) because the function
computes *logp*, not surprise.  Returning ``-inf`` tells NUTS to reject the
proposal cleanly.

PyTensor g++ warnings are suppressed at module import time because the
performance-critical path calls directly into JIT-compiled JAX.
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

import jax
import jax.numpy as jnp
from jax import lax

# Suppress PyTensor g++ compilation warning — not needed when Op.perform
# delegates entirely to JAX JIT.
pytensor.config.cxx = ""


# ---------------------------------------------------------------------------
# 2-level factory
# ---------------------------------------------------------------------------


def build_logp_ops_2level(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
) -> tuple[Op, int]:
    """Build JIT-compiled logp Op for the 2-level binary HGF.

    Builds a fresh 3-branch 2-level binary HGF network, freezes its
    ``scan_fn`` and ``attributes`` once, and returns a PyTensor Op that
    computes the softmax-stickiness log-likelihood given ``(omega_2, beta,
    zeta)``.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.  ``input_data_arr[t, k]`` is the reward
        on trial ``t`` if cue ``k`` was chosen, else ``0.0``.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary mask: ``1`` when cue ``k`` was chosen on trial ``t``.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index ``(0, 1, or 2)`` for each trial.

    Returns
    -------
    logp_op : Op
        PyTensor Op whose ``__call__(omega_2, beta, zeta)`` returns a scalar
        log-likelihood.
    n_trials : int
        Number of trials; useful for downstream diagnostics.

    Notes
    -----
    The scan is called with the exact tuple structure expected by pyhgf:
    ``(values, observed_cols, time_steps, None)``.
    """
    from prl_hgf.models.hgf_2level import build_2level_network

    n_trials = input_data_arr.shape[0]

    # Build network and trigger scan_fn creation
    net = build_2level_network()
    net.input_data(input_data=input_data_arr, observed=observed_arr)
    base_attrs = net.attributes
    scan_fn = net.scan_fn

    # Precompute static scan inputs
    values = tuple(np.split(input_data_arr, [1, 2], axis=1))
    observed_cols = tuple(observed_arr[:, i] for i in range(3))
    time_steps = np.ones(n_trials)
    scan_inputs = (values, observed_cols, time_steps, None)

    choices_jax = jnp.array(choices_arr, dtype=jnp.int32)

    def _jax_logp(omega_2: float, beta: float, zeta: float) -> float:
        # Shallow-copy outer dict then each modified node dict
        attrs = dict(base_attrs)
        for idx in [1, 3, 5]:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        _, node_traj = lax.scan(scan_fn, attrs, scan_inputs)

        # Use expected_mean from binary INPUT_NODES (0, 2, 4) — sigmoid P in [0,1]
        mu1 = jnp.stack(
            [
                node_traj[0]["expected_mean"],
                node_traj[2]["expected_mean"],
                node_traj[4]["expected_mean"],
            ],
            axis=1,
        )

        # Softmax-stickiness log-likelihood
        prev = jnp.concatenate([jnp.array([-1]), choices_jax[:-1]])
        stick = (prev[:, None] == jnp.arange(3)[None, :]).astype(jnp.float32)
        logits = beta * mu1 + zeta * stick
        lp = jax.nn.log_softmax(logits, axis=1)
        result = jnp.sum(lp[jnp.arange(n_trials), choices_jax])

        # NaN guard: -inf is the correct sentinel for logp
        return jnp.where(jnp.isnan(result), -jnp.inf, result)

    _jit_val_grad = jax.jit(jax.value_and_grad(_jax_logp, argnums=(0, 1, 2)))
    _jit_logp = jax.jit(_jax_logp)

    class _GradOp(Op):
        """Return gradients of logp w.r.t. (omega_2, beta, zeta)."""

        def make_node(self, o2, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, b, z]]
            return Apply(self, inputs, [inp.type() for inp in inputs])

        def perform(self, node, inputs, outputs):
            (_, grads) = _jit_val_grad(*[float(x) for x in inputs])
            for i, g in enumerate(grads):
                outputs[i][0] = np.asarray(g, dtype=node.outputs[i].dtype)

    _grad_op = _GradOp()

    class _LogpOp(Op):
        """Forward logp Op; delegates gradients to _GradOp."""

        def make_node(self, o2, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, b, z]]
            return Apply(self, inputs, [pt.scalar(dtype=float)])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(
                _jit_logp(*[float(x) for x in inputs]), dtype=float
            )

        def grad(self, inputs, output_gradients):
            grads = _grad_op(*inputs)
            og = output_gradients[0]
            return [og * g for g in grads]

    return _LogpOp(), n_trials


# ---------------------------------------------------------------------------
# 3-level factory
# ---------------------------------------------------------------------------


def build_logp_ops_3level(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
) -> tuple[Op, int]:
    """Build JIT-compiled logp Op for the 3-level binary HGF.

    Extends the 2-level factory by also injecting ``omega_3`` into node 6's
    ``tonic_volatility`` and ``kappa`` into both endpoints of each 6→(1,3,5)
    volatility edge (``volatility_coupling_children`` on node 6 and
    ``volatility_coupling_parents`` on nodes 1, 3, 5).

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.

    Returns
    -------
    logp_op : Op
        PyTensor Op whose ``__call__(omega_2, omega_3, kappa, beta, zeta)``
        returns a scalar log-likelihood.
    n_trials : int
        Number of trials.
    """
    from prl_hgf.models.hgf_3level import build_3level_network

    n_trials = input_data_arr.shape[0]

    net = build_3level_network()
    net.input_data(input_data=input_data_arr, observed=observed_arr)
    base_attrs = net.attributes
    scan_fn = net.scan_fn

    values = tuple(np.split(input_data_arr, [1, 2], axis=1))
    observed_cols = tuple(observed_arr[:, i] for i in range(3))
    time_steps = np.ones(n_trials)
    scan_inputs = (values, observed_cols, time_steps, None)

    choices_jax = jnp.array(choices_arr, dtype=jnp.int32)

    def _jax_logp(
        omega_2: float,
        omega_3: float,
        kappa: float,
        beta: float,
        zeta: float,
    ) -> float:
        attrs = dict(base_attrs)

        # omega_2 into level-1 belief nodes (1, 3, 5)
        for idx in [1, 3, 5]:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node

        # omega_3 and kappa children-side into volatility node 6
        node6 = dict(attrs[6])
        node6["tonic_volatility"] = omega_3
        node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
        attrs[6] = node6

        # kappa parents-side into nodes 1, 3, 5 (both endpoints required)
        for idx in [1, 3, 5]:
            node = dict(attrs[idx])
            node["volatility_coupling_parents"] = jnp.array([kappa])
            attrs[idx] = node

        _, node_traj = lax.scan(scan_fn, attrs, scan_inputs)

        mu1 = jnp.stack(
            [
                node_traj[0]["expected_mean"],
                node_traj[2]["expected_mean"],
                node_traj[4]["expected_mean"],
            ],
            axis=1,
        )

        prev = jnp.concatenate([jnp.array([-1]), choices_jax[:-1]])
        stick = (prev[:, None] == jnp.arange(3)[None, :]).astype(jnp.float32)
        logits = beta * mu1 + zeta * stick
        lp = jax.nn.log_softmax(logits, axis=1)
        result = jnp.sum(lp[jnp.arange(n_trials), choices_jax])

        return jnp.where(jnp.isnan(result), -jnp.inf, result)

    _jit_val_grad = jax.jit(
        jax.value_and_grad(_jax_logp, argnums=(0, 1, 2, 3, 4))
    )
    _jit_logp = jax.jit(_jax_logp)

    class _GradOp(Op):
        """Return gradients of logp w.r.t. (omega_2, omega_3, kappa, beta, zeta)."""

        def make_node(self, o2, o3, k, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, o3, k, b, z]]
            return Apply(self, inputs, [inp.type() for inp in inputs])

        def perform(self, node, inputs, outputs):
            (_, grads) = _jit_val_grad(*[float(x) for x in inputs])
            for i, g in enumerate(grads):
                outputs[i][0] = np.asarray(g, dtype=node.outputs[i].dtype)

    _grad_op = _GradOp()

    class _LogpOp(Op):
        """Forward logp Op; delegates gradients to _GradOp."""

        def make_node(self, o2, o3, k, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, o3, k, b, z]]
            return Apply(self, inputs, [pt.scalar(dtype=float)])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(
                _jit_logp(*[float(x) for x in inputs]), dtype=float
            )

        def grad(self, inputs, output_gradients):
            grads = _grad_op(*inputs)
            og = output_gradients[0]
            return [og * g for g in grads]

    return _LogpOp(), n_trials
