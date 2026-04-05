"""Unit tests for HGF model construction, forward pass, and belief extraction.

Covers:
- 2-level and 3-level model construction (node count, input_idxs, parameters)
- Forward pass (no exceptions, finite beliefs)
- Partial feedback (unobserved cues do not update)
- Belief extraction correctness (p_reward in [0, 1], correct field keys)
- 3-level volatility trajectory
- prepare_input_data helper

Note: This is the first test file to exercise JAX/pyhgf code in this project.
JAX import issues would surface here as import errors or forward-pass failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import load_config
from prl_hgf.models import (
    BELIEF_NODES,
    INPUT_NODES,
    VOLATILITY_NODE,
    build_2level_network,
    build_3level_network,
    extract_beliefs,
    extract_beliefs_3level,
    prepare_input_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config():
    """Load the analysis configuration once per session."""
    return load_config()


@pytest.fixture(scope="session")
def trials(config):
    """Generate one full session from config (seed=42)."""
    return generate_session(config, seed=42)


@pytest.fixture(scope="session")
def simple_input():
    """Small synthetic input: 50 trials, cue 0 always chosen, ~80% reward.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(input_data, observed)`` arrays of shape ``(50, 3)`` with int
        observed dtype.
    """
    rng = np.random.default_rng(123)
    n_trials = 50
    input_data = np.zeros((n_trials, 3), dtype=float)
    observed = np.zeros((n_trials, 3), dtype=int)

    rewards = (rng.random(n_trials) < 0.8).astype(float)
    input_data[:, 0] = rewards
    observed[:, 0] = 1  # only cue 0 is ever observed

    return input_data, observed


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_2level_node_count():
    """2-level network has exactly 6 nodes (3 binary + 3 continuous)."""
    net = build_2level_network()
    assert len(net.edges) == 6, (
        f"Expected 6 nodes in 2-level network, got {len(net.edges)}"
    )


def test_3level_node_count():
    """3-level network has exactly 7 nodes (3 binary + 3 continuous + 1 volatility)."""
    net = build_3level_network()
    assert len(net.edges) == 7, (
        f"Expected 7 nodes in 3-level network, got {len(net.edges)}"
    )


def test_2level_input_idxs():
    """2-level network has input_idxs == (0, 2, 4)."""
    net = build_2level_network()
    assert net.input_idxs == INPUT_NODES, (
        f"Expected input_idxs={INPUT_NODES}, got {net.input_idxs}"
    )


def test_3level_input_idxs():
    """3-level network has input_idxs == (0, 2, 4)."""
    net = build_3level_network()
    assert net.input_idxs == INPUT_NODES, (
        f"Expected input_idxs={INPUT_NODES}, got {net.input_idxs}"
    )


def test_3level_volatility_parent_exists():
    """Node 6 exists and is wired as volatility parent to nodes 1, 3, 5.

    ``net.edges`` is a tuple of AdjacencyLists (indexed by position).
    Node 6 is the last element (index 6) and should have
    ``volatility_children=(1, 3, 5)``.
    """
    net = build_3level_network()
    assert len(net.edges) == 7, (
        f"Expected 7 nodes in 3-level network, got {len(net.edges)}"
    )
    # Node 6 is the last element of the edges tuple
    vol_edge = net.edges[VOLATILITY_NODE]
    assert vol_edge.volatility_children == (1, 3, 5), (
        f"Volatility node {VOLATILITY_NODE}: expected volatility_children=(1, 3, 5), "
        f"got {vol_edge.volatility_children}"
    )


def test_custom_omega2():
    """build_2level_network(omega_2=-6.0) sets tonic_volatility=-6.0 on all belief nodes."""
    net = build_2level_network(omega_2=-6.0)
    for node_idx in BELIEF_NODES:
        actual = net.attributes[node_idx]["tonic_volatility"]
        assert actual == -6.0, (
            f"Node {node_idx}: expected tonic_volatility=-6.0, got {actual}"
        )


def test_custom_kappa():
    """build_3level_network(kappa=0.5) sets coupling strength to 0.5 on all branches."""
    net = build_3level_network(kappa=0.5)
    # Verify network builds without error with non-default kappa
    assert len(net.edges) == 7, (
        f"Expected 7 nodes with kappa=0.5, got {len(net.edges)}"
    )
    # Check that the attributes for the belief nodes carry default omega_2
    for node_idx in BELIEF_NODES:
        actual = net.attributes[node_idx]["tonic_volatility"]
        assert actual == -4.0, (
            f"Node {node_idx}: expected tonic_volatility=-4.0 (default), got {actual}"
        )


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build_fn", [build_2level_network, build_3level_network])
def test_forward_pass_no_error(simple_input, build_fn):
    """Both model variants run input_data() without raising exceptions."""
    input_data, observed = simple_input
    net = build_fn()
    # Should not raise
    net.input_data(input_data=input_data, observed=observed)


def test_belief_trajectories_finite(simple_input):
    """After 2-level forward pass, all extracted beliefs are finite (no NaN/inf)."""
    input_data, observed = simple_input
    net = build_2level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs(net)
    for key, arr in beliefs.items():
        assert np.all(np.isfinite(arr)), (
            f"Non-finite values in {key}: "
            f"NaN count={np.sum(np.isnan(arr))}, "
            f"inf count={np.sum(np.isinf(arr))}"
        )


def test_belief_trajectory_shapes(simple_input):
    """Each belief trajectory in extract_beliefs has shape (n_trials,)."""
    input_data, observed = simple_input
    n_trials = input_data.shape[0]
    net = build_2level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs(net)
    for key, arr in beliefs.items():
        assert arr.shape == (n_trials,), (
            f"Trajectory '{key}': expected shape ({n_trials},), got {arr.shape}"
        )


# ---------------------------------------------------------------------------
# Partial feedback tests
# ---------------------------------------------------------------------------


def test_unobserved_cue_beliefs_constant(simple_input):
    """Cues 1 and 2 (never observed) should have constant mu1 across all trials."""
    input_data, observed = simple_input
    net = build_2level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs(net)

    for key in ("mu1_cue1", "mu1_cue2"):
        arr = beliefs[key]
        assert np.allclose(arr, arr[0], atol=1e-6), (
            f"Unobserved cue trajectory '{key}' is not constant — "
            f"expected all values ≈ {arr[0]:.6f}, "
            f"max deviation = {np.max(np.abs(arr - arr[0])):.2e}"
        )


def test_observed_cue_beliefs_change(simple_input):
    """Cue 0 (observed with variable rewards) should have non-constant mu1."""
    input_data, observed = simple_input
    net = build_2level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs(net)

    arr = beliefs["mu1_cue0"]
    assert not np.allclose(arr, arr[0], atol=1e-6), (
        "mu1_cue0 is unexpectedly constant — observed cue beliefs should update "
        "when variable rewards are received"
    )


# ---------------------------------------------------------------------------
# Belief extraction correctness tests
# ---------------------------------------------------------------------------


def test_p_reward_in_valid_range(simple_input):
    """p_reward values from binary nodes must be in [0, 1] for all three cues."""
    input_data, observed = simple_input
    net = build_2level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs(net)

    for key in ("p_reward_cue0", "p_reward_cue1", "p_reward_cue2"):
        arr = beliefs[key]
        assert np.all(arr >= 0.0), (
            f"'{key}' has values below 0.0: "
            f"min={arr.min():.6f}, expected min >= 0.0"
        )
        assert np.all(arr <= 1.0), (
            f"'{key}' has values above 1.0: "
            f"max={arr.max():.6f}, expected max <= 1.0"
        )


# ---------------------------------------------------------------------------
# 3-level specific tests
# ---------------------------------------------------------------------------


def test_3level_volatility_trajectory_finite(simple_input):
    """After 3-level forward pass, mu2_volatility is finite (no NaN/inf)."""
    input_data, observed = simple_input
    net = build_3level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs_3level(net)

    arr = beliefs["mu2_volatility"]
    assert np.all(np.isfinite(arr)), (
        f"mu2_volatility has non-finite values: "
        f"NaN count={np.sum(np.isnan(arr))}, "
        f"inf count={np.sum(np.isinf(arr))}"
    )


def test_3level_volatility_trajectory_shape(simple_input):
    """mu2_volatility has shape (n_trials,)."""
    input_data, observed = simple_input
    n_trials = input_data.shape[0]
    net = build_3level_network()
    net.input_data(input_data=input_data, observed=observed)
    beliefs = extract_beliefs_3level(net)

    arr = beliefs["mu2_volatility"]
    assert arr.shape == (n_trials,), (
        f"mu2_volatility: expected shape ({n_trials},), got {arr.shape}"
    )


# ---------------------------------------------------------------------------
# prepare_input_data tests
# ---------------------------------------------------------------------------


def test_prepare_input_data_shape():
    """prepare_input_data returns arrays with shape (n_trials, 3)."""
    n = 10
    trials = [None] * n
    choices = list(range(3)) * 3 + [0]  # 10 choices cycling through 0,1,2
    rewards = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    input_data, observed = prepare_input_data(trials, choices, rewards)
    assert input_data.shape == (n, 3), (
        f"input_data: expected shape ({n}, 3), got {input_data.shape}"
    )
    assert observed.shape == (n, 3), (
        f"observed: expected shape ({n}, 3), got {observed.shape}"
    )


def test_prepare_input_data_observed_dtype():
    """The observed array returned by prepare_input_data has integer dtype."""
    trials = [None] * 4
    choices = [0, 1, 2, 0]
    rewards = [1, 0, 1, 0]
    _, observed = prepare_input_data(trials, choices, rewards)
    assert np.issubdtype(observed.dtype, np.integer), (
        f"observed.dtype must be integer, got {observed.dtype}"
    )


def test_prepare_input_data_values():
    """For a known choice/reward pair, the correct cell is set in both arrays."""
    trials = [None] * 3
    choices = [0, 1, 2]
    rewards = [1, 0, 1]
    input_data, observed = prepare_input_data(trials, choices, rewards)

    # Trial 0: chose cue 0, reward=1
    assert input_data[0, 0] == 1.0, (
        f"Trial 0: input_data[0,0] expected 1.0, got {input_data[0, 0]}"
    )
    assert observed[0, 0] == 1, (
        f"Trial 0: observed[0,0] expected 1, got {observed[0, 0]}"
    )
    assert observed[0, 1] == 0 and observed[0, 2] == 0, (
        f"Trial 0: unchosen cues should have observed=0, "
        f"got observed[0,1]={observed[0, 1]}, observed[0,2]={observed[0, 2]}"
    )

    # Trial 1: chose cue 1, reward=0
    assert input_data[1, 1] == 0.0, (
        f"Trial 1: input_data[1,1] expected 0.0, got {input_data[1, 1]}"
    )
    assert observed[1, 1] == 1, (
        f"Trial 1: observed[1,1] expected 1, got {observed[1, 1]}"
    )
    assert observed[1, 0] == 0 and observed[1, 2] == 0, (
        f"Trial 1: unchosen cues should have observed=0, "
        f"got observed[1,0]={observed[1, 0]}, observed[1,2]={observed[1, 2]}"
    )

    # Trial 2: chose cue 2, reward=1
    assert input_data[2, 2] == 1.0, (
        f"Trial 2: input_data[2,2] expected 1.0, got {input_data[2, 2]}"
    )
    assert observed[2, 2] == 1, (
        f"Trial 2: observed[2,2] expected 1, got {observed[2, 2]}"
    )
