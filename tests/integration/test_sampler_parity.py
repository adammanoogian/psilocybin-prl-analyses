"""AUDIT-02: BlackJAX vs NumPyro logp parity at fixed inputs.

Asserts that both backend paths produce identical log-density values
when evaluated at the same parameter vectors on the same data. This
guards against prior drift between _build_log_posterior (BlackJAX closure)
and _numpyro_model_{2,3}level (NumPyro model).

Both paths share build_logp_fn_batched -- logp parity is structurally
guaranteed at the likelihood level. This test verifies that the PRIOR
construction also matches numerically.

Run::

    pytest tests/integration/test_sampler_parity.py -v -m integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable regardless of install mode.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Mark all tests as integration (must run on M3 cluster with JAX/NumPyro)
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simulated_data_small():
    """Small synthetic dataset for parity checks (P=3, T=20).

    Returns
    -------
    dict
        Keys: ``input_data``, ``observed``, ``choices``, ``trial_mask``,
        ``n_participants``, ``n_trials``.  All arrays are NumPy (float32
        or int32); JAX conversion happens inside tests.
    """
    rng = np.random.default_rng(42)
    n_participants, n_trials = 3, 20

    # input_data: shape (P, T, 3) -- reward magnitudes per cue
    input_data = rng.uniform(0.0, 1.0, (n_participants, n_trials, 3)).astype(
        np.float32
    )

    # observed: shape (P, T, 3) -- binary mask (which cue was observed)
    # Exactly one cue observed per trial (partial feedback)
    observed = np.zeros((n_participants, n_trials, 3), dtype=np.int32)
    for p in range(n_participants):
        for t in range(n_trials):
            c = rng.integers(0, 3)
            observed[p, t, c] = 1

    # choices: shape (P, T) -- chosen cue index
    choices = np.argmax(observed, axis=-1).astype(np.int32)

    # trial_mask: shape (P, T) -- all trials valid
    trial_mask = np.ones((n_participants, n_trials), dtype=np.float32)

    return {
        "input_data": input_data,
        "observed": observed,
        "choices": choices,
        "trial_mask": trial_mask,
        "n_participants": n_participants,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_blackjax_numpyro_logp_parity_3level(simulated_data_small):
    """BlackJAX and NumPyro 3-level logp agree within fp32 tolerance."""
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from prl_hgf.fitting.hierarchical import (
        _build_log_posterior,
        _numpyro_model_3level,
        build_logp_fn_batched,
    )
    from prl_hgf.fitting.priors import HGFPriorSpec

    data = simulated_data_small
    n_participants = data["n_participants"]
    n_trials = data["n_trials"]

    # Convert to JAX arrays
    input_data = jnp.array(data["input_data"])
    observed = jnp.array(data["observed"])
    choices = jnp.array(data["choices"])
    trial_mask = jnp.array(data["trial_mask"])

    # Fixed parameter dict at known values (sampled params only)
    params = {
        "omega_2": jnp.full((n_participants,), -3.0),
        "omega_3": jnp.full((n_participants,), -6.0),
        "log_beta": jnp.full((n_participants,), 0.0),
        "zeta": jnp.full((n_participants,), 0.0),
    }

    # Shared prior spec and logp function
    prior_spec = HGFPriorSpec.default_3level()
    logp_fn, _n_params = build_logp_fn_batched("hgf_3level", n_trials)

    # --- BlackJAX path: _build_log_posterior returns a callable ---
    logdensity_fn = _build_log_posterior(
        logp_fn,
        input_data,
        observed,
        choices,
        trial_mask,
        n_participants,
        model_name="hgf_3level",
        prior_spec=prior_spec,
    )
    bj_logp = float(logdensity_fn(params))

    # --- NumPyro path: use numpyro.infer.util.log_density ---
    from functools import partial

    np_model = partial(
        _numpyro_model_3level,
        n_participants=n_participants,
        batched_logp_fn=logp_fn,
        prior_spec=prior_spec,
    )
    # log_density(model, model_args, model_kwargs, params)
    # _numpyro_model_3level positional args: input_data, observed, choices,
    # trial_mask (n_participants/batched_logp_fn/prior_spec via partial)
    np_logp_val, _ = log_density(
        np_model,
        (input_data, observed, choices, trial_mask),
        {},
        params,
    )
    np_logp_val = float(np_logp_val)

    np.testing.assert_allclose(
        bj_logp,
        np_logp_val,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            f"AUDIT-02 FAIL: BlackJAX logp={bj_logp:.6f} != "
            f"NumPyro logp={np_logp_val:.6f} (3-level)"
        ),
    )


def test_blackjax_numpyro_logp_parity_2level(simulated_data_small):
    """BlackJAX and NumPyro 2-level logp agree within fp32 tolerance."""
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from prl_hgf.fitting.hierarchical import (
        _build_log_posterior,
        _numpyro_model_2level,
        build_logp_fn_batched,
    )
    from prl_hgf.fitting.priors import HGFPriorSpec

    data = simulated_data_small
    n_participants = data["n_participants"]
    n_trials = data["n_trials"]

    # Convert to JAX arrays
    input_data = jnp.array(data["input_data"])
    observed = jnp.array(data["observed"])
    choices = jnp.array(data["choices"])
    trial_mask = jnp.array(data["trial_mask"])

    # Fixed parameter dict at known values (sampled params only, no omega_3)
    params = {
        "omega_2": jnp.full((n_participants,), -3.0),
        "log_beta": jnp.full((n_participants,), 0.0),
        "zeta": jnp.full((n_participants,), 0.0),
    }

    # Shared prior spec and logp function
    prior_spec = HGFPriorSpec.default_2level()
    logp_fn, _n_params = build_logp_fn_batched("hgf_2level", n_trials)

    # --- BlackJAX path: _build_log_posterior returns a callable ---
    logdensity_fn = _build_log_posterior(
        logp_fn,
        input_data,
        observed,
        choices,
        trial_mask,
        n_participants,
        model_name="hgf_2level",
        prior_spec=prior_spec,
    )
    bj_logp = float(logdensity_fn(params))

    # --- NumPyro path: use numpyro.infer.util.log_density ---
    from functools import partial

    np_model = partial(
        _numpyro_model_2level,
        n_participants=n_participants,
        batched_logp_fn=logp_fn,
        prior_spec=prior_spec,
    )
    # log_density(model, model_args, model_kwargs, params)
    np_logp_val, _ = log_density(
        np_model,
        (input_data, observed, choices, trial_mask),
        {},
        params,
    )
    np_logp_val = float(np_logp_val)

    np.testing.assert_allclose(
        bj_logp,
        np_logp_val,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            f"AUDIT-02 FAIL: BlackJAX logp={bj_logp:.6f} != "
            f"NumPyro logp={np_logp_val:.6f} (2-level)"
        ),
    )
