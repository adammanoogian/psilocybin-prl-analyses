"""Unit and integration tests for the fitting module.

Tests cover:
- Op logp finiteness (2-level and 3-level)
- Gradient finiteness through the JAX scan
- NaN guard edge case (omega_2 >= 0)
- Single-participant MCMC fit convergence and output schema
- Summary row extraction schema
- Diagnostic flag logic (flag_fit)

Tests that run MCMC sampling are marked ``@pytest.mark.scientific`` (consistent
with test_batch.py convention).  Tests that only check Ops or gradients do
NOT need the slow marker.

Run fast tests only::

    pytest tests/test_fitting.py -v -k "not slow"

Run slow tests (MCMC)::

    pytest tests/test_fitting.py -v -k "slow" --timeout=120
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Session-scoped fixture: simulated data for Op tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _simulated_data():
    """Build a short simulated session for Op tests.

    Returns (input_data_arr, observed_arr, choices_arr) for a single
    participant-session using fixed seed and known parameters.

    Uses ~50 trials (first 50 of a full session) for speed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(input_data_arr, observed_arr, choices_arr)`` arrays.
    """
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.models.hgf_2level import prepare_input_data
    from prl_hgf.models.hgf_3level import build_3level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()

    # Use deterministic seed for reproducibility
    rng = np.random.default_rng(12345)
    net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    trials = generate_session(cfg, seed=12345)

    result = simulate_agent(net, trials, beta=3.0, zeta=0.5, rng=rng)

    # Slice to first 50 trials for speed in Op tests
    trials_50 = trials[:50]
    choices_50 = result.choices[:50]
    rewards_50 = result.rewards[:50]

    input_data_arr, observed_arr = prepare_input_data(
        trials_50, choices_50, rewards_50
    )
    choices_arr = np.array(choices_50, dtype=int)

    return input_data_arr, observed_arr, choices_arr


# ---------------------------------------------------------------------------
# Test 1: 2-level Op returns finite logp at realistic parameter values
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_2level_op_finite_logp(_simulated_data):
    """2-level Op returns a finite log-probability at canonical parameters."""
    from prl_hgf.fitting.ops import build_logp_ops_2level

    input_data_arr, observed_arr, choices_arr = _simulated_data
    logp_op, _ = build_logp_ops_2level(input_data_arr, observed_arr, choices_arr)

    import pytensor.tensor as pt

    o2 = pt.as_tensor_variable(-3.0)
    b = pt.as_tensor_variable(3.0)
    z = pt.as_tensor_variable(0.5)

    val = float(logp_op(o2, b, z).eval())
    assert math.isfinite(val), (
        f"Expected finite logp from 2-level Op, got {val}. "
        "Parameters: omega_2=-3.0, beta=3.0, zeta=0.5"
    )


# ---------------------------------------------------------------------------
# Test 2: 3-level Op returns finite logp at realistic parameter values
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_3level_op_finite_logp(_simulated_data):
    """3-level Op returns a finite log-probability at canonical parameters."""
    from prl_hgf.fitting.ops import build_logp_ops_3level

    input_data_arr, observed_arr, choices_arr = _simulated_data
    logp_op, _ = build_logp_ops_3level(input_data_arr, observed_arr, choices_arr)

    import pytensor.tensor as pt

    o2 = pt.as_tensor_variable(-3.0)
    o3 = pt.as_tensor_variable(-6.0)
    k = pt.as_tensor_variable(1.0)
    b = pt.as_tensor_variable(3.0)
    z = pt.as_tensor_variable(0.5)

    val = float(logp_op(o2, o3, k, b, z).eval())
    assert math.isfinite(val), (
        f"Expected finite logp from 3-level Op, got {val}. "
        "Parameters: omega_2=-3.0, omega_3=-6.0, kappa=1.0, beta=3.0, zeta=0.5"
    )


# ---------------------------------------------------------------------------
# Test 3: 2-level gradient components are all finite
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_2level_grad_finite(_simulated_data):
    """All gradient components of the 2-level Op are finite at realistic parameters.

    Uses pytensor.grad to compute the gradient through the symbolic graph,
    verifying that the gradient chain works end-to-end.
    """
    import pytensor
    import pytensor.tensor as pt

    from prl_hgf.fitting.ops import build_logp_ops_2level

    input_data_arr, observed_arr, choices_arr = _simulated_data
    logp_op, _ = build_logp_ops_2level(input_data_arr, observed_arr, choices_arr)

    # Create scalar variables
    o2 = pt.dscalar("o2")
    b = pt.dscalar("b")
    z = pt.dscalar("z")

    logp_val = logp_op(o2, b, z)
    grads = pytensor.grad(logp_val, [o2, b, z])

    # Compile and evaluate
    grad_fn = pytensor.function([o2, b, z], grads)
    g_vals = grad_fn(-3.0, 3.0, 0.5)

    for name, gval in zip(["omega_2", "beta", "zeta"], g_vals, strict=True):
        assert math.isfinite(float(gval)), (
            f"Gradient w.r.t. {name} is not finite: {gval}. "
            "Expected finite gradient at omega_2=-3.0, beta=3.0, zeta=0.5"
        )


# ---------------------------------------------------------------------------
# Test 4: omega_2 >= 0 triggers NaN guard and returns -inf
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_omega2_positive_returns_neginf(_simulated_data):
    """2-level Op returns -inf when omega_2 is positive (NaN guard activated).

    omega_2 >= 0 causes NaN in the binary HGF scan.  The NaN guard in the
    JAX logp function converts NaN to -jnp.inf, which is returned as -inf
    in the NumPy scalar output.
    """
    import pytensor.tensor as pt

    from prl_hgf.fitting.ops import build_logp_ops_2level

    input_data_arr, observed_arr, choices_arr = _simulated_data
    logp_op, _ = build_logp_ops_2level(input_data_arr, observed_arr, choices_arr)

    o2 = pt.as_tensor_variable(1.0)  # positive omega_2 — triggers NaN
    b = pt.as_tensor_variable(3.0)
    z = pt.as_tensor_variable(0.5)

    val = float(logp_op(o2, b, z).eval())
    assert val == float("-inf"), (
        f"Expected -inf from NaN guard at omega_2=+1.0, got {val}. "
        "omega_2 >= 0 should produce NaN in scan → -inf logp"
    )


# ---------------------------------------------------------------------------
# Test 5: Single-participant MCMC fit with 2-level model (slow)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_single_fit_2level():
    """Single-participant 2-level MCMC fit converges with expected schema.

    Uses n_draws=300, n_chains=2 for speed.  Asserts output schema,
    parameter constraints, and relaxed R-hat criterion (< 1.1).
    """
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.fitting.single import fit_participant
    from prl_hgf.models.hgf_2level import prepare_input_data
    from prl_hgf.models.hgf_3level import build_3level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()

    # Simulate with known parameters
    rng = np.random.default_rng(99)
    net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    trials = generate_session(cfg, seed=99)
    result = simulate_agent(net, trials, beta=3.0, zeta=0.5, rng=rng)

    input_data_arr, observed_arr = prepare_input_data(
        trials, result.choices, result.rewards
    )
    choices_arr = np.array(result.choices, dtype=int)

    _, summary_rows, flagged = fit_participant(
        input_data_arr=input_data_arr,
        observed_arr=observed_arr,
        choices_arr=choices_arr,
        participant_id="test_P001",
        group="placebo",
        session="baseline",
        model_name="hgf_2level",
        n_chains=2,
        n_draws=300,
        n_tune=300,
        target_accept=0.9,
        random_seed=42,
        cores=1,
    )

    # --- Schema check ---
    assert len(summary_rows) == 3, (
        f"Expected 3 summary rows (omega_2, beta, zeta), got {len(summary_rows)}. "
        f"Rows: {[r['parameter'] for r in summary_rows]}"
    )

    expected_keys = {
        "participant_id", "group", "session", "model", "parameter",
        "mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess",
    }
    for row in summary_rows:
        missing_keys = expected_keys - set(row.keys())
        assert not missing_keys, (
            f"Row for '{row.get('parameter', '?')}' missing keys: {missing_keys}"
        )

    # --- Convergence check (relaxed for short chain) ---
    for row in summary_rows:
        assert row["r_hat"] < 1.1, (
            f"R-hat for {row['parameter']} is {row['r_hat']:.4f}, "
            "expected < 1.1 for 2-chain 300-draw fit"
        )

    # --- Parameter sanity checks ---
    param_map = {r["parameter"]: r for r in summary_rows}
    omega_2_mean = param_map["omega_2"]["mean"]
    assert -6.0 <= omega_2_mean <= 0.0, (
        f"Posterior mean for omega_2 is {omega_2_mean:.3f}, "
        "expected in [-6, 0]"
    )

    beta_mean = param_map["beta"]["mean"]
    assert beta_mean > 0, (
        f"Posterior mean for beta is {beta_mean:.3f}, expected > 0"
    )


# ---------------------------------------------------------------------------
# Test 6: extract_summary_rows schema matches FIT-04 spec
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_extract_summary_rows_schema():
    """extract_summary_rows returns list[dict] matching FIT-04 schema."""
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.fitting.single import extract_summary_rows, fit_participant
    from prl_hgf.models.hgf_2level import prepare_input_data
    from prl_hgf.models.hgf_3level import build_3level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()
    rng = np.random.default_rng(101)
    net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    trials = generate_session(cfg, seed=101)
    result = simulate_agent(net, trials, beta=3.0, zeta=0.5, rng=rng)

    input_data_arr, observed_arr = prepare_input_data(
        trials, result.choices, result.rewards
    )
    choices_arr = np.array(result.choices, dtype=int)

    idata, _, _ = fit_participant(
        input_data_arr=input_data_arr,
        observed_arr=observed_arr,
        choices_arr=choices_arr,
        participant_id="test_P002",
        group="psilocybin",
        session="post_dose",
        model_name="hgf_2level",
        n_chains=2,
        n_draws=200,
        n_tune=200,
        random_seed=101,
        cores=1,
    )

    rows = extract_summary_rows(
        idata,
        participant_id="test_P002",
        group="psilocybin",
        session="post_dose",
        model="hgf_2level",
        var_names=["omega_2", "beta", "zeta"],
    )

    assert isinstance(rows, list), f"Expected list, got {type(rows)}"
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"

    expected_keys = {
        "participant_id", "group", "session", "model", "parameter",
        "mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess",
    }
    for row in rows:
        actual_keys = set(row.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        assert not missing, f"Missing keys in row: {missing}"
        assert not extra, f"Unexpected extra keys in row: {extra}"

        # Type checks
        assert isinstance(row["participant_id"], str)
        assert isinstance(row["mean"], float)
        assert isinstance(row["r_hat"], float)
        assert isinstance(row["ess"], float)


# ---------------------------------------------------------------------------
# Tests 7-9: flag_fit logic (fast, no MCMC)
# ---------------------------------------------------------------------------


def test_flag_fit_detects_bad_rhat():
    """flag_fit returns True when any parameter has r_hat > threshold."""
    from prl_hgf.fitting.single import flag_fit

    rows = [
        {"parameter": "omega_2", "r_hat": 1.2, "ess": 800.0},
        {"parameter": "beta", "r_hat": 1.01, "ess": 900.0},
        {"parameter": "zeta", "r_hat": 1.01, "ess": 750.0},
    ]
    result = flag_fit(rows)
    assert result is True, (
        f"Expected flag_fit to return True when r_hat=1.2 > 1.05, got {result}. "
        "Bad parameter: omega_2"
    )


def test_flag_fit_detects_low_ess():
    """flag_fit returns True when any parameter has ess < threshold."""
    from prl_hgf.fitting.single import flag_fit

    rows = [
        {"parameter": "omega_2", "r_hat": 1.01, "ess": 100.0},
        {"parameter": "beta", "r_hat": 1.01, "ess": 900.0},
        {"parameter": "zeta", "r_hat": 1.01, "ess": 800.0},
    ]
    result = flag_fit(rows)
    assert result is True, (
        f"Expected flag_fit to return True when ess=100 < 400, got {result}. "
        "Bad parameter: omega_2"
    )


def test_flag_fit_clean():
    """flag_fit returns False when all parameters pass diagnostic thresholds."""
    from prl_hgf.fitting.single import flag_fit

    rows = [
        {"parameter": "omega_2", "r_hat": 1.01, "ess": 800.0},
        {"parameter": "beta", "r_hat": 1.02, "ess": 650.0},
        {"parameter": "zeta", "r_hat": 1.005, "ess": 900.0},
    ]
    result = flag_fit(rows)
    assert result is False, (
        f"Expected flag_fit to return False for clean diagnostics, got {result}. "
        "All r_hat < 1.05 and ess >= 400"
    )
