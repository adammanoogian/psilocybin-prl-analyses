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
from prl_hgf.models.response_patrl import model_a_logp, model_b_logp, model_c_logp

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


# ---------------------------------------------------------------------------
# Phase 20-02: Tests for model_a_logp(b), model_b_logp, model_c_logp
# ---------------------------------------------------------------------------


class TestPhase20ResponseModels:
    """Tests for the Plan 20-02 response model extensions.

    Covers model_a_logp with bias b, model_b_logp, and model_c_logp.
    """

    _rng = np.random.default_rng(20)
    _N = 10  # number of trials for shape tests

    @classmethod
    def _inputs(cls) -> tuple:
        """Return (mu2, choices, reward, shock) test arrays of length _N."""
        mu2 = jnp.array(
            cls._rng.standard_normal(cls._N).astype(np.float32)
        )
        choices = jnp.array(
            cls._rng.integers(0, 2, size=cls._N), dtype=jnp.int32
        )
        reward = jnp.full(cls._N, 3.0)
        shock = jnp.full(cls._N, 3.0)
        return mu2, choices, reward, shock

    def test_model_a_logp_accepts_b_default_zero(self) -> None:
        """model_a_logp with b=0.0 default must match Phase 18 behavior."""
        mu2, choices, reward, shock = self._inputs()
        logp_new_default = model_a_logp(mu2, choices, reward, shock, beta=2.0)
        logp_explicit_zero = model_a_logp(mu2, choices, reward, shock, beta=2.0, b=0.0)
        np.testing.assert_allclose(
            np.asarray(logp_new_default),
            np.asarray(logp_explicit_zero),
            atol=1e-6,
            err_msg="model_a_logp b=0 default must equal explicit b=0",
        )

    def test_model_a_logp_b_shifts_logits(self) -> None:
        """b=2.0 shifts approach logit at ev=0; P(approach) > 0.5.

        At mu2=0: P(danger)=0.5, EV = 0.5*V_rew - 0.5*V_shk.
        With V_rew=V_shk=3: EV = 0.  Approach logit before b: beta*EV=0.
        With b=2.0: approach logit = 2.0 -> P(approach) = sigmoid(2) ~ 0.88.
        With b=0.0: approach logit = 0 -> P(approach) = 0.5.
        """
        mu2 = jnp.zeros(1)
        choices_approach = jnp.array([1], dtype=jnp.int32)
        reward = jnp.array([3.0])
        shock = jnp.array([3.0])

        logp_b0 = model_a_logp(mu2, choices_approach, reward, shock, beta=2.0, b=0.0)
        logp_b2 = model_a_logp(mu2, choices_approach, reward, shock, beta=2.0, b=2.0)

        p_approach_b0 = float(jnp.exp(logp_b0[0]))
        p_approach_b2 = float(jnp.exp(logp_b2[0]))

        assert abs(p_approach_b0 - 0.5) < 1e-4, (
            f"Expected P(approach)~0.5 at EV=0,b=0; got {p_approach_b0:.6f}"
        )
        assert p_approach_b2 > 0.8, (
            f"Expected P(approach)>0.8 with b=2.0; got {p_approach_b2:.6f}"
        )

    def test_model_b_logp_shape_and_finite(self) -> None:
        """model_b_logp returns (n_trials,) all-finite array."""
        mu2, choices, reward, shock = self._inputs()
        dhr = jnp.array(
            self._rng.normal(0.0, 2.0, size=self._N).astype(np.float32)
        )
        logp = model_b_logp(mu2, choices, reward, shock, beta=2.0, b=0.3, gamma=0.5, delta_hr=dhr)
        assert logp.shape == (self._N,), (
            f"Expected shape ({self._N},), got {logp.shape}"
        )
        assert np.all(np.isfinite(np.asarray(logp))), "model_b_logp is not all-finite"

    def test_model_b_logp_gamma_zero_equals_model_a_plus_b(self) -> None:
        """model_b_logp with gamma=0 must equal model_a_logp with same b."""
        mu2, choices, reward, shock = self._inputs()
        dhr = jnp.array(
            self._rng.normal(0.0, 2.0, size=self._N).astype(np.float32)
        )
        logp_b = model_b_logp(
            mu2, choices, reward, shock, beta=2.0, b=0.3, gamma=0.0, delta_hr=dhr
        )
        logp_a = model_a_logp(mu2, choices, reward, shock, beta=2.0, b=0.3)
        np.testing.assert_allclose(
            np.asarray(logp_b),
            np.asarray(logp_a),
            atol=1e-6,
            err_msg="model_b with gamma=0 must equal model_a with same b",
        )

    def test_model_c_logp_shape_and_finite(self) -> None:
        """model_c_logp returns (n_trials,) all-finite array."""
        mu2, choices, reward, shock = self._inputs()
        dhr = jnp.array(
            self._rng.normal(0.0, 2.0, size=self._N).astype(np.float32)
        )
        logp = model_c_logp(
            mu2, choices, reward, shock, beta=2.0, b=0.3, alpha=0.1, gamma=0.5, delta_hr=dhr
        )
        assert logp.shape == (self._N,), (
            f"Expected shape ({self._N},), got {logp.shape}"
        )
        assert np.all(np.isfinite(np.asarray(logp))), "model_c_logp is not all-finite"

    def test_model_c_logp_alpha_zero_equals_model_b(self) -> None:
        """model_c_logp with alpha=0 must equal model_b_logp elementwise."""
        mu2, choices, reward, shock = self._inputs()
        dhr = jnp.array(
            self._rng.normal(0.0, 2.0, size=self._N).astype(np.float32)
        )
        logp_c = model_c_logp(
            mu2, choices, reward, shock, beta=2.0, b=0.3, alpha=0.0, gamma=0.5, delta_hr=dhr
        )
        logp_b = model_b_logp(
            mu2, choices, reward, shock, beta=2.0, b=0.3, gamma=0.5, delta_hr=dhr
        )
        np.testing.assert_allclose(
            np.asarray(logp_c),
            np.asarray(logp_b),
            atol=1e-6,
            err_msg="model_c with alpha=0 must equal model_b",
        )

    def test_model_c_alpha_effect_sign(self) -> None:
        """With alpha>0 and ΔHR<0, effective_beta < beta; approach logit decreases.

        At mu2 = -5 (very safe state): EV_approach > 0.
        With alpha>0 and negative dhr: effective_beta = beta + alpha*dhr < beta.
        Therefore approach logit at positive EV is smaller under Model C vs Model A.
        """
        mu2_safe = jnp.full(5, -5.0)  # P(danger) very low → EV > 0
        choices_approach = jnp.ones(5, dtype=jnp.int32)
        reward = jnp.full(5, 5.0)
        shock = jnp.full(5, 1.0)
        dhr_bradycardia = jnp.full(5, -3.0)  # freezing

        # Model A baseline (no dhr effect)
        logp_a = model_a_logp(mu2_safe, choices_approach, reward, shock, beta=2.0, b=0.0)
        # Model C with alpha=0.5: effective_beta = 2.0 + 0.5*(-3) = 0.5 < 2.0
        logp_c = model_c_logp(
            mu2_safe, choices_approach, reward, shock,
            beta=2.0, b=0.0, alpha=0.5, gamma=0.0, delta_hr=dhr_bradycardia,
        )

        # Lower effective_beta → lower approach logit → lower P(approach) → lower logp
        assert np.all(np.asarray(logp_c) < np.asarray(logp_a)), (
            "Expected model_c with alpha>0, dhr<0 to have lower approach logp than model_a"
        )
