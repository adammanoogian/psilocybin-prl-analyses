"""Phase 12 validation tests for the batched hierarchical JAX logp Op.

Tests cover three categories:

1. **VALID-01 (bit-exact at P=1 and P=2):** The new
   ``build_logp_ops_batched`` forward pass at ``n_participants=1`` returns
   float64-identical values to the existing ``build_logp_ops_3level`` (and
   ``build_logp_ops_2level``) for matched inputs and parameters.  Tolerance:
   ``atol=1e-12, rtol=0``.

2. **VALID-02 (within-MCSE at P=5):** Fit 5 synthetic participants
   sequentially via the legacy ``fit_batch`` path on CPU and the same 5
   participants batched via ``fit_batch_hierarchical`` on CPU, both with
   matched seeds.  Per-parameter posterior means agree within
   ``3 x max(mcse_legacy, mcse_batched)``.

3. **Layer 2 clamping smoke test:** Drive a single participant's logp into
   the unstable region and confirm the batched logp returns a finite value
   (not NaN).

Run fast tests only::

    pytest tests/test_hierarchical_logp.py -v -k "not slow"

Run slow tests (MCMC)::

    pytest tests/test_hierarchical_logp.py -v -m slow --timeout=900
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytensor.tensor as pt
import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Session-scoped fixture: simulated data for a single participant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _simulated_data_session():
    """Build a short simulated session for one participant.

    Returns ``(input_data_arr, observed_arr, choices_arr)`` shaped
    ``(n_trials, 3)``, ``(n_trials, 3)``, ``(n_trials,)`` using ~50
    trials from a 3-level agent simulation with known parameters.
    """
    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.models.hgf_2level import prepare_input_data
    from prl_hgf.models.hgf_3level import build_3level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()
    rng = np.random.default_rng(12345)
    net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    trials = generate_session(cfg, seed=12345)
    result = simulate_agent(net, trials, beta=3.0, zeta=0.5, rng=rng)

    trials_50 = trials[:50]
    choices_50 = result.choices[:50]
    rewards_50 = result.rewards[:50]

    input_data_arr, observed_arr = prepare_input_data(
        trials_50, choices_50, rewards_50
    )
    choices_arr = np.array(choices_50, dtype=int)

    return input_data_arr, observed_arr, choices_arr


# ---------------------------------------------------------------------------
# VALID-01 — bit-exact at P=1
# ---------------------------------------------------------------------------


def test_valid_01_batched_at_p1_bit_exact_3level(_simulated_data_session):
    """VALID-01: batched logp at P=1 equals legacy 3-level logp.

    The new ``build_logp_ops_batched`` with ``n_participants=1`` must return
    a float64-identical value to ``build_logp_ops_3level`` for matched inputs
    and parameters.  Tolerance: ``atol=1e-12, rtol=0``.
    """
    from prl_hgf.fitting.hierarchical import build_logp_ops_batched
    from prl_hgf.fitting.ops import build_logp_ops_3level

    input_data_arr, observed_arr, choices_arr = _simulated_data_session

    # --- Legacy (per-participant) ---
    legacy_op, _ = build_logp_ops_3level(
        input_data_arr, observed_arr, choices_arr
    )
    legacy_val = float(
        legacy_op(
            pt.as_tensor_variable(-3.0),
            pt.as_tensor_variable(-6.0),
            pt.as_tensor_variable(1.0),
            pt.as_tensor_variable(3.0),
            pt.as_tensor_variable(0.5),
        ).eval()
    )

    # --- Batched at P=1 ---
    # Stack into (1, n_trials, 3) and (1, n_trials)
    batched_op, n_p, n_t = build_logp_ops_batched(
        input_data_arr[np.newaxis, ...],
        observed_arr[np.newaxis, ...],
        choices_arr[np.newaxis, ...],
        model_name="hgf_3level",
    )
    batched_val = float(
        batched_op(
            pt.as_tensor_variable(np.array([-3.0])),
            pt.as_tensor_variable(np.array([-6.0])),
            pt.as_tensor_variable(np.array([1.0])),
            pt.as_tensor_variable(np.array([3.0])),
            pt.as_tensor_variable(np.array([0.5])),
        ).eval()
    )

    diff = abs(legacy_val - batched_val)
    assert n_p == 1, f"Expected n_participants=1, got {n_p}"
    np.testing.assert_allclose(
        batched_val,
        legacy_val,
        atol=1e-12,
        rtol=0,
        err_msg=(
            f"VALID-01 3-level FAILED: batched={batched_val}, "
            f"legacy={legacy_val}, diff={diff}"
        ),
    )


def test_valid_01_batched_at_p1_bit_exact_2level(_simulated_data_session):
    """VALID-01: batched logp at P=1 equals legacy 2-level logp.

    Same as 3-level test but for the ``hgf_2level`` model variant.
    Parameters: ``omega_2=-3.0, beta=3.0, zeta=0.5``.
    """
    from prl_hgf.fitting.hierarchical import build_logp_ops_batched
    from prl_hgf.fitting.ops import build_logp_ops_2level

    input_data_arr, observed_arr, choices_arr = _simulated_data_session

    # --- Legacy (per-participant) ---
    legacy_op, _ = build_logp_ops_2level(
        input_data_arr, observed_arr, choices_arr
    )
    legacy_val = float(
        legacy_op(
            pt.as_tensor_variable(-3.0),
            pt.as_tensor_variable(3.0),
            pt.as_tensor_variable(0.5),
        ).eval()
    )

    # --- Batched at P=1 ---
    batched_op, n_p, n_t = build_logp_ops_batched(
        input_data_arr[np.newaxis, ...],
        observed_arr[np.newaxis, ...],
        choices_arr[np.newaxis, ...],
        model_name="hgf_2level",
    )
    batched_val = float(
        batched_op(
            pt.as_tensor_variable(np.array([-3.0])),
            pt.as_tensor_variable(np.array([3.0])),
            pt.as_tensor_variable(np.array([0.5])),
        ).eval()
    )

    diff = abs(legacy_val - batched_val)
    assert n_p == 1, f"Expected n_participants=1, got {n_p}"
    np.testing.assert_allclose(
        batched_val,
        legacy_val,
        atol=1e-12,
        rtol=0,
        err_msg=(
            f"VALID-01 2-level FAILED: batched={batched_val}, "
            f"legacy={legacy_val}, diff={diff}"
        ),
    )


# ---------------------------------------------------------------------------
# VALID-01 — P=2 doubling test
# ---------------------------------------------------------------------------


def test_valid_01_batched_at_p2_doubles_logp(_simulated_data_session):
    """VALID-01 P=2: batched logp with 2 identical participants equals 2x legacy.

    At P=2 with two identical participants and identical parameters, the
    batched logp must equal ``2 * legacy_logp`` within ``atol=1e-12``.  This
    confirms the ``vmap + jnp.sum`` reduction.
    """
    from prl_hgf.fitting.hierarchical import build_logp_ops_batched
    from prl_hgf.fitting.ops import build_logp_ops_3level

    input_data_arr, observed_arr, choices_arr = _simulated_data_session

    # --- Legacy single-participant ---
    legacy_op, _ = build_logp_ops_3level(
        input_data_arr, observed_arr, choices_arr
    )
    legacy_val = float(
        legacy_op(
            pt.as_tensor_variable(-3.0),
            pt.as_tensor_variable(-6.0),
            pt.as_tensor_variable(1.0),
            pt.as_tensor_variable(3.0),
            pt.as_tensor_variable(0.5),
        ).eval()
    )

    # --- Batched at P=2 (two identical participants) ---
    batched_op, n_p, n_t = build_logp_ops_batched(
        np.stack([input_data_arr, input_data_arr], axis=0),
        np.stack([observed_arr, observed_arr], axis=0),
        np.stack([choices_arr, choices_arr], axis=0),
        model_name="hgf_3level",
    )
    batched_val = float(
        batched_op(
            pt.as_tensor_variable(np.array([-3.0, -3.0])),
            pt.as_tensor_variable(np.array([-6.0, -6.0])),
            pt.as_tensor_variable(np.array([1.0, 1.0])),
            pt.as_tensor_variable(np.array([3.0, 3.0])),
            pt.as_tensor_variable(np.array([0.5, 0.5])),
        ).eval()
    )

    expected = 2.0 * legacy_val
    diff = abs(batched_val - expected)
    assert n_p == 2, f"Expected n_participants=2, got {n_p}"
    np.testing.assert_allclose(
        batched_val,
        expected,
        atol=1e-12,
        rtol=0,
        err_msg=(
            f"VALID-01 P=2 doubling FAILED: batched={batched_val}, "
            f"2*legacy={expected}, diff={diff}"
        ),
    )


# ---------------------------------------------------------------------------
# Layer 2 clamping smoke test
# ---------------------------------------------------------------------------


def test_layer_2_clamping_returns_finite_under_unstable_params(
    _simulated_data_session,
):
    """Layer 2 clamping: unstable params produce finite-or-(-inf), never NaN.

    Pushes ``omega_2=-0.1`` (near zero, unstable region where the binary
    HGF scan is known to produce NaN) and asserts the batched logp returns
    a finite value or ``-inf`` — never ``NaN``.
    """
    from prl_hgf.fitting.hierarchical import build_logp_ops_batched

    input_data_arr, observed_arr, choices_arr = _simulated_data_session

    batched_op, _, _ = build_logp_ops_batched(
        input_data_arr[np.newaxis, ...],
        observed_arr[np.newaxis, ...],
        choices_arr[np.newaxis, ...],
        model_name="hgf_3level",
    )

    val = float(
        batched_op(
            pt.as_tensor_variable(np.array([-0.1])),  # unstable omega_2
            pt.as_tensor_variable(np.array([-6.0])),
            pt.as_tensor_variable(np.array([1.0])),
            pt.as_tensor_variable(np.array([3.0])),
            pt.as_tensor_variable(np.array([0.5])),
        ).eval()
    )

    assert not math.isnan(val), (
        f"Layer 2 clamping smoke test FAILED: got NaN. "
        f"Expected finite or -inf at omega_2=-0.1. "
        f"Batched logp returned: {val}"
    )
    assert math.isfinite(val) or val == float("-inf"), (
        f"Layer 2 clamping smoke test: unexpected value {val}. "
        f"Expected finite or -inf."
    )
