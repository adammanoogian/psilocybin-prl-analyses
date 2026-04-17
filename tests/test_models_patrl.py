"""Unit tests for PAT-RL HGF builders and Model A response module.

Covers:
- 2-level and 3-level PAT-RL model construction (node count, input_idxs)
- Forward pass shape and finite-value checks for both model variants
- Determinism: two networks with identical params produce identical trajectories
- Model A log-likelihood shape, numerical stability, and EV semantics
- Differentiability of model_a_logp under jax.grad
- Pick_best_cue regression: existing exports unchanged
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prl_hgf.models.hgf_2level_patrl import (
    INPUT_NODE,
    build_2level_network_patrl,
    extract_beliefs_patrl,
)
from prl_hgf.models.hgf_3level_patrl import (
    build_3level_network_patrl,
    extract_beliefs_patrl_3level,
)
from prl_hgf.models.response_patrl import model_a_logp

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

N_TRIALS = 192
_RNG = np.random.default_rng(42)
_BINARY_INPUT = _RNG.integers(0, 2, size=N_TRIALS).astype(float)
_TIME_STEPS = np.ones(N_TRIALS)


# ---------------------------------------------------------------------------
# Test 1: 2-level topology
# ---------------------------------------------------------------------------


def test_build_2level_topology() -> None:
    """2-level network has exactly 2 nodes and input_idxs == (0,)."""
    net = build_2level_network_patrl(omega_2=-4.0)
    assert net.input_idxs == (INPUT_NODE,), (
        f"Expected input_idxs=(0,) got {net.input_idxs}"
    )
    # Run a short forward pass so node_trajectories is populated.
    # node_trajectories includes key -1 (time tracker); count only non-negative keys.
    net.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)
    n_nodes = sum(1 for k in net.node_trajectories if k >= 0)
    assert n_nodes == 2, f"Expected 2 nodes, got {n_nodes}"


# ---------------------------------------------------------------------------
# Test 2: 3-level topology
# ---------------------------------------------------------------------------


def test_build_3level_topology() -> None:
    """3-level network has exactly 3 nodes and input_idxs == (0,)."""
    net = build_3level_network_patrl()
    assert net.input_idxs == (INPUT_NODE,), (
        f"Expected input_idxs=(0,) got {net.input_idxs}"
    )
    # node_trajectories includes key -1 (time tracker); count only non-negative keys.
    net.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)
    n_nodes = sum(1 for k in net.node_trajectories if k >= 0)
    assert n_nodes == 3, f"Expected 3 nodes, got {n_nodes}"


# ---------------------------------------------------------------------------
# Test 3: 2-level forward pass shape + finiteness
# ---------------------------------------------------------------------------


def test_forward_pass_2level_shape() -> None:
    """2-level forward pass produces belief arrays of shape (n_trials,) all-finite."""
    net = build_2level_network_patrl(omega_2=-4.0)
    net.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)
    beliefs = extract_beliefs_patrl(net)

    required_keys = {"mu2", "sigma2", "p_state", "expected_precision"}
    assert required_keys.issubset(beliefs.keys()), (
        f"Missing keys: {required_keys - beliefs.keys()}"
    )
    for key, arr in beliefs.items():
        assert arr.shape == (N_TRIALS,), (
            f"beliefs['{key}'] shape {arr.shape} != ({N_TRIALS},)"
        )
        assert np.all(np.isfinite(arr)), (
            f"beliefs['{key}'] contains non-finite values"
        )


# ---------------------------------------------------------------------------
# Test 4: 3-level forward pass shape + finiteness
# ---------------------------------------------------------------------------


def test_forward_pass_3level_shape() -> None:
    """3-level forward pass adds mu3, sigma3, epsilon3 all shape (n_trials,) finite."""
    net = build_3level_network_patrl()
    net.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)
    beliefs = extract_beliefs_patrl_3level(net)

    required_keys = {
        "mu2", "sigma2", "p_state", "expected_precision",
        "mu3", "sigma3", "epsilon3",
    }
    assert required_keys.issubset(beliefs.keys()), (
        f"Missing keys: {required_keys - beliefs.keys()}"
    )
    for key in ("mu3", "sigma3", "epsilon3"):
        arr = beliefs[key]
        assert arr.shape == (N_TRIALS,), (
            f"beliefs['{key}'] shape {arr.shape} != ({N_TRIALS},)"
        )
        assert np.all(np.isfinite(arr)), (
            f"beliefs['{key}'] contains non-finite values"
        )


# ---------------------------------------------------------------------------
# Test 5: Determinism
# ---------------------------------------------------------------------------


def test_forward_pass_is_deterministic() -> None:
    """Two networks with identical params and same input produce identical trajectories."""
    net_a = build_2level_network_patrl(omega_2=-4.0)
    net_b = build_2level_network_patrl(omega_2=-4.0)

    net_a.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)
    net_b.input_data(input_data=_BINARY_INPUT[:, None], time_steps=_TIME_STEPS)

    beliefs_a = extract_beliefs_patrl(net_a)
    beliefs_b = extract_beliefs_patrl(net_b)

    for key in beliefs_a:
        np.testing.assert_allclose(
            beliefs_a[key],
            beliefs_b[key],
            atol=1e-10,
            err_msg=f"beliefs['{key}'] differs between two identical networks",
        )


# ---------------------------------------------------------------------------
# Test 6: model_a_logp shape
# ---------------------------------------------------------------------------


def test_model_a_logp_shape() -> None:
    """model_a_logp returns (n_trials,) finite array."""
    rng = np.random.default_rng(7)
    mu2 = jnp.array(rng.standard_normal(N_TRIALS).astype(np.float32))
    choices = jnp.array(rng.integers(0, 2, size=N_TRIALS), dtype=jnp.int32)
    reward_mag = jnp.full(N_TRIALS, 5.0)
    shock_mag = jnp.full(N_TRIALS, 5.0)

    logp = model_a_logp(mu2, choices, reward_mag, shock_mag, beta=2.0)

    assert logp.shape == (N_TRIALS,), (
        f"Expected shape ({N_TRIALS},), got {logp.shape}"
    )
    assert jnp.all(jnp.isfinite(logp)), (
        f"model_a_logp contains non-finite values: {logp}"
    )


# ---------------------------------------------------------------------------
# Test 7: EV semantics — avoid favoured when shock dominates at uncertainty
# ---------------------------------------------------------------------------


def test_model_a_avoid_favoured_when_shock_dominates() -> None:
    """At mu2=0 (p_danger=0.5) with V_shk >> V_rew, P(approach) is very small.

    Convention: sigmoid(mu2) = P(state=1=dangerous).
    At mu2=0: P(dangerous)=0.5.
    EV_approach = 0.5 * V_rew - 0.5 * V_shk = 0.5*1 - 0.5*10 = -4.5.
    With beta=2: logit for approach = 2 * (-4.5) = -9 => P(approach) = exp(-9)/(1+exp(-9)) << 0.01.
    """
    mu2 = jnp.array([0.0])
    choices_approach = jnp.array([1], dtype=jnp.int32)  # approach
    reward_mag = jnp.array([1.0])
    shock_mag = jnp.array([10.0])

    logp_approach = model_a_logp(mu2, choices_approach, reward_mag, shock_mag, beta=2.0)
    p_approach = float(jnp.exp(logp_approach[0]))

    assert p_approach < 0.01, (
        f"Expected P(approach) < 0.01 at mu2=0, V_shk=10 >> V_rew=1, "
        f"but got P(approach)={p_approach:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Differentiability
# ---------------------------------------------------------------------------


def test_model_a_logp_is_differentiable() -> None:
    """model_a_logp is differentiable under jax.grad w.r.t. beta."""
    mu2 = jnp.array([0.0, 1.0, -1.0])
    choices = jnp.array([1, 1, 0], dtype=jnp.int32)
    reward_mag = jnp.array([5.0, 5.0, 5.0])
    shock_mag = jnp.array([5.0, 5.0, 5.0])

    grad = jax.grad(
        lambda b: jnp.sum(model_a_logp(mu2, choices, reward_mag, shock_mag, beta=b))
    )(2.0)

    assert jnp.isfinite(grad), (
        f"jax.grad of model_a_logp w.r.t. beta returned non-finite: {grad}"
    )


# ---------------------------------------------------------------------------
# Test 9: pick_best_cue regression — existing exports unchanged
# ---------------------------------------------------------------------------


def test_pick_best_cue_models_unchanged() -> None:
    """Existing pick_best_cue model exports are intact and INPUT_NODES==(0,2,4)."""
    from prl_hgf.models.hgf_2level import (  # noqa: PLC0415
        INPUT_NODES,
        build_2level_network,
    )

    assert INPUT_NODES == (0, 2, 4), (
        f"pick_best_cue INPUT_NODES changed: expected (0,2,4) got {INPUT_NODES}"
    )
    # Build the pick_best_cue 3-branch network to confirm it still works
    net = build_2level_network()
    assert len(net.edges) == 6, (
        f"pick_best_cue 2-level network should have 6 nodes, got {len(net.edges)}"
    )

    # PAT-RL input_idxs must remain distinct from pick_best_cue
    patrl_net = build_2level_network_patrl()
    assert patrl_net.input_idxs != net.input_idxs, (
        "PAT-RL and pick_best_cue topologies should have different input_idxs"
    )

    from prl_hgf.models.response import softmax_stickiness_surprise  # noqa: PLC0415

    assert callable(softmax_stickiness_surprise), (
        "softmax_stickiness_surprise import failed"
    )
