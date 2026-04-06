"""Unit tests for the Bayesian model comparison module.

Tests cover:
- groupBMC importability
- run_group_bms with synthetic data (clear winner)
- run_group_bms output shape
- compute_subject_waic smoke test (minimal 5-trial network)
- plot_exceedance_probabilities smoke test

Run::

    pytest tests/test_bms.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Test 1: groupBMC importability
# ---------------------------------------------------------------------------


def test_groupbmc_import():
    """groupBMC package must be importable."""
    from groupBMC.groupBMC import GroupBMC  # noqa: F401

    assert callable(GroupBMC), "GroupBMC should be callable"


# ---------------------------------------------------------------------------
# Test 2: run_group_bms with synthetic data where model 1 clearly wins
# ---------------------------------------------------------------------------


def test_run_group_bms_synthetic():
    """Model 1 should have higher EP when its log-evidence is much better."""
    from prl_hgf.analysis.bms import run_group_bms

    rng = np.random.default_rng(42)
    n_subjects = 10
    # elpd_matrix shape: (n_subjects, n_models)
    # Model 0: -100 per subject (better), Model 1: -200 per subject (worse)
    elpd = np.column_stack(
        [
            rng.normal(-100, 5, n_subjects),
            rng.normal(-200, 5, n_subjects),
        ]
    )
    result = run_group_bms(elpd, ["model_a", "model_b"], group_label="test")

    assert "alpha" in result
    assert "exp_r" in result
    assert "xp" in result
    assert "pxp" in result
    assert "bor" in result
    assert "model_names" in result
    assert result["model_names"] == ["model_a", "model_b"]
    assert result["n_subjects"] == n_subjects

    # Model 0 should win (higher EP)
    assert result["xp"][0] > 0.5, (
        f"Expected model_a EP > 0.5, got {result['xp'][0]:.4f}"
    )

    # exp_r should sum to ~1
    exp_r_sum = float(np.sum(result["exp_r"]))
    assert abs(exp_r_sum - 1.0) < 0.05, (
        f"Expected exp_r sum ~1.0, got {exp_r_sum:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: run_group_bms output array shapes
# ---------------------------------------------------------------------------


def test_run_group_bms_shape():
    """Output arrays must have length n_models."""
    from prl_hgf.analysis.bms import run_group_bms

    rng = np.random.default_rng(7)
    n_subjects = 8
    n_models = 3
    elpd = rng.normal(-150, 20, (n_subjects, n_models))
    model_names = ["m1", "m2", "m3"]
    result = run_group_bms(elpd, model_names)

    assert len(result["xp"]) == n_models, (
        f"xp length: expected {n_models}, got {len(result['xp'])}"
    )
    assert len(result["pxp"]) == n_models, (
        f"pxp length: expected {n_models}, got {len(result['pxp'])}"
    )
    assert len(result["exp_r"]) == n_models, (
        f"exp_r length: expected {n_models}, got {len(result['exp_r'])}"
    )


# ---------------------------------------------------------------------------
# Test 4: compute_subject_waic smoke test (2-level, minimal 5-trial network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_subject_waic_smoke():
    """compute_subject_waic should return a finite float for a 2-level model."""
    import arviz as az
    import xarray as xr

    from prl_hgf.analysis.bms import compute_subject_waic
    from prl_hgf.fitting.ops import build_logp_ops_2level

    # Minimal 5-trial dummy arrays (same pattern as _prewarm_jit in batch.py)
    n_trials = 5
    dummy_input = np.zeros((n_trials, 3), dtype=float)
    dummy_obs = np.zeros((n_trials, 3), dtype=int)
    dummy_choices = np.zeros(n_trials, dtype=int)

    # Mark cue 0 as chosen throughout (so at least one cue is "observed")
    dummy_obs[:, 0] = 1
    dummy_input[:, 0] = 1.0  # cue 0 always rewarded

    # Build Op to trigger JIT compilation so idata params are valid
    build_logp_ops_2level(dummy_input, dummy_obs, dummy_choices)

    # Build fake InferenceData with plausible posterior samples
    rng = np.random.default_rng(0)
    n_chains = 2
    n_draws = 10

    omega_2 = rng.normal(-3.0, 0.1, (n_chains, n_draws))
    beta = rng.normal(2.0, 0.1, (n_chains, n_draws))
    zeta = rng.normal(0.5, 0.05, (n_chains, n_draws))

    posterior_ds = xr.Dataset(
        {
            "omega_2": xr.DataArray(omega_2, dims=["chain", "draw"]),
            "beta": xr.DataArray(beta, dims=["chain", "draw"]),
            "zeta": xr.DataArray(zeta, dims=["chain", "draw"]),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
    )
    idata = az.InferenceData(posterior=posterior_ds)

    elpd = compute_subject_waic(
        dummy_input, dummy_obs, dummy_choices, idata, "hgf_2level"
    )

    assert np.isfinite(elpd), f"elpd_waic should be finite, got {elpd}"
    assert isinstance(elpd, float), f"elpd_waic should be float, got {type(elpd)}"


# ---------------------------------------------------------------------------
# Test 5: plot_exceedance_probabilities smoke test
# ---------------------------------------------------------------------------


def test_plot_exceedance_probabilities_runs():
    """EP bar plot should return a Figure without raising."""
    import matplotlib.pyplot as plt

    from prl_hgf.analysis.bms import plot_exceedance_probabilities

    # Build fake bms_results dict
    bms_results = {
        "all": {
            "xp": np.array([0.9, 0.1]),
            "pxp": np.array([0.85, 0.15]),
            "model_names": ["hgf_2level", "hgf_3level"],
            "n_subjects": 20,
            "bor": 0.05,
            "group_label": "all",
        },
        "healthy_control": {
            "xp": np.array([0.8, 0.2]),
            "pxp": np.array([0.77, 0.23]),
            "model_names": ["hgf_2level", "hgf_3level"],
            "n_subjects": 10,
            "bor": 0.12,
            "group_label": "healthy_control",
        },
    }

    fig = plot_exceedance_probabilities(bms_results)

    assert isinstance(fig, plt.Figure), (
        f"Expected plt.Figure, got {type(fig)}"
    )
    # Two subplots (one per group)
    n_axes = len(fig.get_axes())
    assert n_axes == 2, f"Expected 2 subplots, got {n_axes}"

    plt.close(fig)
