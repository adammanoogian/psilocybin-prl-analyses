"""Smoke tests for Mode B (hierarchical) fitting via both backends.

Exercises the full Mode B pipeline — hierarchical cohort simulation
through ``simulate_hierarchical_cohort``, followed by
``fit_batch_hierarchical`` in hierarchical pooling mode — with tiny
sample sizes (P=10, n_draws=50, n_warmup=50) to verify code paths
without requiring convergence.

Run::

    pytest tests/integration/test_mode_b_smoke.py -v -m slow
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_PARTICIPANTS = 10
N_GROUPS = 2
N_PER_GROUP = 5

# Ground-truth hyperparameters for tiny smoke cohort
_TRUE_MU = {
    "omega_2": np.array([-3.5, -2.5]),
    "log_beta": np.array([0.5, 1.0]),
    "zeta": np.array([0.2, -0.2]),
}
_TRUE_SIGMA = {
    "omega_2": 0.8,
    "log_beta": 0.5,
    "zeta": 0.6,
}


@pytest.fixture(scope="module")
def tiny_cohort():
    """Simulate a tiny hierarchical cohort for smoke testing.

    Returns
    -------
    tuple
        ``(sim_df, true_params, group_idx)`` with P=10 participants
        across 2 groups of 5.
    """
    from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort

    group_idx = np.repeat(np.arange(N_GROUPS), N_PER_GROUP)

    sim_df, true_params = simulate_hierarchical_cohort(
        n_participants=N_PARTICIPANTS,
        n_groups=N_GROUPS,
        true_mu=_TRUE_MU,
        true_sigma=_TRUE_SIGMA,
        true_beta=None,
        x_covariate=None,
        group_idx=group_idx,
        model_name="hgf_2level",
        seed=777,
    )

    return sim_df, true_params, group_idx


# ---------------------------------------------------------------------------
# BlackJAX Mode B smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_blackjax_mode_b_completes(tiny_cohort) -> None:
    """BlackJAX Mode B runs without exception and produces hyperparameters.

    Parameters
    ----------
    tiny_cohort : tuple
        Module-scoped fixture with sim_df, true_params, group_idx.
    """
    import arviz as az

    from prl_hgf.fitting.config import (
        CovariateConfig,
        FitConfig,
        MitigationConfig,
        SamplerConfig,
    )
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical

    sim_df, _, _ = tiny_cohort

    fit_config = FitConfig(
        model_name="hgf_2level",
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=2,
            n_draws=50,
            n_warmup=50,
            target_accept=0.8,
            random_seed=42,
        ),
        mitigation=MitigationConfig(
            non_centered=("omega_2", "log_beta", "zeta"),
        ),
        covariate=CovariateConfig(
            pooling="hierarchical",
            n_groups=N_GROUPS,
        ),
        progressbar=False,
    )

    result = fit_batch_hierarchical(sim_df, fit_config)

    # Unpack if tuple (cold call returns (idata, adapted_params))
    if isinstance(result, tuple):
        idata = result[0]
    else:
        idata = result

    # --- Assertion 1: result is InferenceData ---
    assert isinstance(idata, az.InferenceData), (
        f"Expected az.InferenceData, got {type(idata).__name__}"
    )

    # --- Assertion 2: posterior group exists ---
    assert hasattr(idata, "posterior"), (
        f"InferenceData missing 'posterior' group. Available: {list(idata._groups)}"
    )

    # --- Assertion 3: hyperparameter sites present ---
    posterior = idata.posterior
    # The naming convention uses underscore-free param names
    # (mu_omega_2 or mu_omega2). Check at least one mu_* key exists.
    posterior_vars = set(posterior.data_vars)
    has_mu = any(v.startswith("mu_") for v in posterior_vars)
    has_sigma = any(
        v.startswith("log_sigma_") or v.startswith("sigma_") for v in posterior_vars
    )
    assert has_mu, (
        f"No mu_* hyperparameters found in posterior. "
        f"Available vars: {sorted(posterior_vars)}"
    )
    assert has_sigma, (
        f"No sigma_* hyperparameters found in posterior. "
        f"Available vars: {sorted(posterior_vars)}"
    )

    # --- Assertion 4: provenance recorded ---
    assert "fit_config" in idata.attrs, (
        "idata.attrs missing 'fit_config' provenance key"
    )


# ---------------------------------------------------------------------------
# NumPyro Mode B smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_numpyro_mode_b_completes(tiny_cohort) -> None:
    """NumPyro Mode B runs without exception and produces hyperparameters.

    Parameters
    ----------
    tiny_cohort : tuple
        Module-scoped fixture with sim_df, true_params, group_idx.
    """
    import arviz as az

    from prl_hgf.fitting.config import (
        CovariateConfig,
        FitConfig,
        MitigationConfig,
        SamplerConfig,
    )
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical

    sim_df, _, _ = tiny_cohort

    fit_config = FitConfig(
        model_name="hgf_2level",
        sampler=SamplerConfig(
            backend="numpyro",
            n_chains=2,
            n_draws=50,
            n_warmup=50,
            target_accept=0.8,
            random_seed=42,
        ),
        mitigation=MitigationConfig(
            non_centered=("omega_2", "log_beta", "zeta"),
        ),
        covariate=CovariateConfig(
            pooling="hierarchical",
            n_groups=N_GROUPS,
        ),
        progressbar=False,
    )

    result = fit_batch_hierarchical(sim_df, fit_config)

    # NumPyro path returns idata directly (no adapted_params tuple)
    if isinstance(result, tuple):
        idata = result[0]
    else:
        idata = result

    # --- Assertion 1: result is InferenceData ---
    assert isinstance(idata, az.InferenceData), (
        f"Expected az.InferenceData, got {type(idata).__name__}"
    )

    # --- Assertion 2: posterior group exists ---
    assert hasattr(idata, "posterior"), (
        f"InferenceData missing 'posterior' group. Available: {list(idata._groups)}"
    )

    # --- Assertion 3: hyperparameter sites present ---
    posterior = idata.posterior
    posterior_vars = set(posterior.data_vars)
    has_mu = any(v.startswith("mu_") for v in posterior_vars)
    has_sigma = any(
        v.startswith("log_sigma_") or v.startswith("sigma_") for v in posterior_vars
    )
    assert has_mu, (
        f"No mu_* hyperparameters found in posterior. "
        f"Available vars: {sorted(posterior_vars)}"
    )
    assert has_sigma, (
        f"No sigma_* hyperparameters found in posterior. "
        f"Available vars: {sorted(posterior_vars)}"
    )

    # --- Assertion 4: provenance recorded ---
    assert "fit_config" in idata.attrs, (
        "idata.attrs missing 'fit_config' provenance key"
    )
