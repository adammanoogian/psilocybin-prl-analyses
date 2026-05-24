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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    import arviz as az

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


@pytest.mark.scientific
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
        "placebo": {
            "xp": np.array([0.8, 0.2]),
            "pxp": np.array([0.77, 0.23]),
            "model_names": ["hgf_2level", "hgf_3level"],
            "n_subjects": 10,
            "bor": 0.12,
            "group_label": "placebo",
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


# ---------------------------------------------------------------------------
# Phase 20-06 Tests: compute_batch_waic_patrl + compute_stratified_bms +
# export_peb_covariates
# ---------------------------------------------------------------------------


def _make_synthetic_idata(
    n_chains: int = 1,
    n_draws: int = 5,
    seed: int = 0,
    n_obs: int = 3,
) -> az.InferenceData:
    """Build a synthetic InferenceData with log_likelihood populated.

    Constructs an idata that compute_subject_waic_patrl can consume via its
    fast-path (log_likelihood already present) without triggering the slow
    JAX re-evaluation path.  This keeps tests fast and free of JAX compile
    overhead.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains.
    n_draws : int
        Number of posterior draws per chain.
    seed : int
        RNG seed.
    n_obs : int
        Trailing observation dimension for log_likelihood (n_trials proxy).

    Returns
    -------
    az.InferenceData
        Has ``posterior`` and ``log_likelihood`` groups.
    """
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(seed)
    posterior_ds = xr.Dataset(
        {
            "omega_2": xr.DataArray(
                rng.normal(-3.0, 0.1, (n_chains, n_draws)),
                dims=["chain", "draw"],
            ),
            "log_beta": xr.DataArray(
                rng.normal(0.7, 0.1, (n_chains, n_draws)),
                dims=["chain", "draw"],
            ),
            "b": xr.DataArray(
                rng.normal(0.0, 0.1, (n_chains, n_draws)),
                dims=["chain", "draw"],
            ),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
    )
    ll_vals = rng.normal(-50.0, 5.0, (n_chains, n_draws, n_obs))
    ll_da = xr.DataArray(
        ll_vals,
        dims=["chain", "draw", "loglike_dim_0"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
    )
    idata = az.InferenceData(posterior=posterior_ds)
    idata.add_groups({"log_likelihood": {"loglike": ll_da}})
    return idata


def _make_fit_df(
    participants: list[str],
    phenotypes: list[str],
    model_names: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal fit_df for BMS tests.

    Parameters
    ----------
    participants : list[str]
        Participant identifiers.
    phenotypes : list[str]
        Phenotype per participant (must have same length as participants).
    model_names : list[str]
        Model names to include.
    seed : int
        RNG seed for elpd values.

    Returns
    -------
    pandas.DataFrame
        Columns: participant_id, phenotype, model, elpd_waic.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid, phen in zip(participants, phenotypes, strict=True):
        for model in model_names:
            rows.append(
                {
                    "participant_id": pid,
                    "phenotype": phen,
                    "model": model,
                    "elpd_waic": float(rng.normal(-100, 10)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: compute_batch_waic_patrl
# ---------------------------------------------------------------------------


def test_compute_batch_waic_patrl_propagates_phenotype():
    """Output fit_df must carry the phenotype column from sim_df."""
    from prl_hgf.analysis.bms import compute_batch_waic_patrl

    participants = ["P0", "P1", "P2"]
    phenotypes = ["healthy", "high_anxiety", "healthy"]

    # Build sim_df with one row per participant (minimal; col set includes phenotype)
    sim_df = pd.DataFrame(
        {
            "participant_id": participants,
            "phenotype": phenotypes,
            "trial_idx": [0, 0, 0],
            "state": [0, 1, 0],
            "choice": [0, 1, 0],
            "reward_mag": [1.0, 0.0, 1.0],
            "shock_mag": [0.0, 1.0, 0.0],
            "delta_hr": [0.0, 0.0, 0.0],
            "outcome_time_s": [0.5, 0.5, 0.5],
        }
    )

    # Build idata_dict using synthetic idatas that have log_likelihood
    idata_dict = {
        "hgf_2level_patrl": {
            pid: _make_synthetic_idata(seed=i) for i, pid in enumerate(participants)
        }
    }

    result = compute_batch_waic_patrl(
        sim_df, idata_dict, model_names=["hgf_2level_patrl"]
    )

    assert "phenotype" in result.columns, (
        f"Expected 'phenotype' column in result, got columns: {list(result.columns)}"
    )
    assert len(result) == len(participants), (
        f"Expected {len(participants)} rows, got {len(result)}"
    )
    id_to_phen = dict(zip(participants, phenotypes, strict=True))
    for _, row in result.iterrows():
        expected_phen = id_to_phen[row["participant_id"]]
        assert row["phenotype"] == expected_phen, (
            f"phenotype mismatch for {row['participant_id']}: "
            f"expected {expected_phen!r}, got {row['phenotype']!r}"
        )


def test_compute_batch_waic_patrl_unknown_pid_sentinel():
    """Participant in idata_dict not in sim_df gets phenotype='unknown'."""
    from prl_hgf.analysis.bms import compute_batch_waic_patrl

    sim_df = pd.DataFrame(
        {
            "participant_id": ["P0"],
            "phenotype": ["healthy"],
            "trial_idx": [0],
            "state": [0],
            "choice": [0],
            "reward_mag": [1.0],
            "shock_mag": [0.0],
            "delta_hr": [0.0],
            "outcome_time_s": [0.5],
        }
    )
    # P99 is NOT in sim_df
    idata_dict = {
        "hgf_2level_patrl": {
            "P0": _make_synthetic_idata(seed=0),
            "P99": _make_synthetic_idata(seed=1),
        }
    }

    result = compute_batch_waic_patrl(
        sim_df, idata_dict, model_names=["hgf_2level_patrl"]
    )
    unknown_rows = result[result["participant_id"] == "P99"]
    assert len(unknown_rows) == 1, (
        f"Expected 1 row for P99, got {len(unknown_rows)}"
    )
    assert unknown_rows.iloc[0]["phenotype"] == "unknown", (
        f"Expected 'unknown', got {unknown_rows.iloc[0]['phenotype']!r}"
    )


def test_compute_batch_waic_patrl_missing_phenotype_raises():
    """ValueError raised when sim_df lacks the phenotype column."""
    from prl_hgf.analysis.bms import compute_batch_waic_patrl

    sim_df_no_phen = pd.DataFrame(
        {
            "participant_id": ["P0"],
            "trial_idx": [0],
            "state": [0],
            "choice": [0],
        }
    )
    with pytest.raises(ValueError, match="phenotype"):
        compute_batch_waic_patrl(sim_df_no_phen, {}, model_names=["hgf_2level_patrl"])


# ---------------------------------------------------------------------------
# Tests: compute_stratified_bms
# ---------------------------------------------------------------------------


def test_compute_stratified_bms_returns_all_plus_per_phenotype():
    """Result dict must have 'all' key plus one key per unique phenotype."""
    from prl_hgf.analysis.bms import compute_stratified_bms

    participants = [f"P{i}" for i in range(20)]
    phenotypes = ["healthy"] * 10 + ["high_anxiety"] * 10
    fit_df = _make_fit_df(
        participants, phenotypes, ["hgf_2level_patrl", "hgf_3level_patrl"], seed=7
    )

    result = compute_stratified_bms(
        fit_df, ["hgf_2level_patrl", "hgf_3level_patrl"]
    )
    assert set(result.keys()) == {"all", "healthy", "high_anxiety"}, (
        f"Expected keys {{'all', 'healthy', 'high_anxiety'}}, got {set(result.keys())}"
    )
    # Each BMS result should have the standard keys
    for key, bms in result.items():
        for required_key in ("alpha", "exp_r", "xp", "pxp", "bor", "n_subjects"):
            assert required_key in bms, (
                f"BMS result for '{key}' missing key '{required_key}'"
            )


def test_compute_stratified_bms_missing_column_raises():
    """ValueError raised when fit_df lacks the phenotype column."""
    from prl_hgf.analysis.bms import compute_stratified_bms

    df_no_phen = pd.DataFrame(
        {
            "participant_id": ["P0", "P1"],
            "model": ["hgf_2level_patrl", "hgf_3level_patrl"],
            "elpd_waic": [-100.0, -110.0],
        }
    )
    with pytest.raises(ValueError, match="phenotype"):
        compute_stratified_bms(df_no_phen, ["hgf_2level_patrl", "hgf_3level_patrl"])


def test_compute_stratified_bms_missing_model_raises():
    """ValueError raised when model_names contains a model absent from fit_df."""
    from prl_hgf.analysis.bms import compute_stratified_bms

    df_2level_only = pd.DataFrame(
        {
            "participant_id": ["P0", "P1"],
            "phenotype": ["healthy", "healthy"],
            "model": ["hgf_2level_patrl", "hgf_2level_patrl"],
            "elpd_waic": [-100.0, -110.0],
        }
    )
    with pytest.raises(ValueError, match="hgf_3level_patrl"):
        compute_stratified_bms(
            df_2level_only, ["hgf_2level_patrl", "hgf_3level_patrl"]
        )


def test_compute_stratified_bms_low_n_warning_fires(caplog: pytest.LogCaptureFixture):
    """A warning mentioning 'exploratory' fires for phenotypes with N < 20."""
    import logging

    from prl_hgf.analysis.bms import compute_stratified_bms

    # 3 participants in 'rare_phen', 20 in 'healthy'
    participants = [f"P{i}" for i in range(23)]
    phenotypes = ["healthy"] * 20 + ["rare_phen"] * 3
    fit_df = _make_fit_df(
        participants, phenotypes, ["hgf_2level_patrl", "hgf_3level_patrl"], seed=99
    )

    with caplog.at_level(logging.WARNING, logger="prl_hgf.analysis.bms"):
        compute_stratified_bms(fit_df, ["hgf_2level_patrl", "hgf_3level_patrl"])

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("exploratory" in str(m) for m in warning_msgs), (
        f"Expected a warning containing 'exploratory'; got: {warning_msgs}"
    )


def test_compute_stratified_bms_preserves_run_stratified_bms():
    """run_stratified_bms (pick_best_cue path) still produces correct output.

    Regression test: Phase 1 callers must not be affected by Phase 20 additions.
    """
    from prl_hgf.analysis.bms import run_stratified_bms

    rng = np.random.default_rng(77)
    rows = []
    for pid in range(10):
        for grp in ["placebo", "active"]:
            for sess in [1, 2]:
                for model in ["hgf_2level", "hgf_3level"]:
                    rows.append(
                        {
                            "participant_id": pid,
                            "group": grp,
                            "session": sess,
                            "model": model,
                            "elpd_waic": float(rng.normal(-100, 5)),
                        }
                    )
    waic_df = pd.DataFrame(rows)
    result = run_stratified_bms(waic_df, ["hgf_2level", "hgf_3level"])
    assert "all" in result, "run_stratified_bms must return 'all' key"
    assert "placebo" in result, "run_stratified_bms must return 'placebo' key"
    assert "active" in result, "run_stratified_bms must return 'active' key"
    # Shape checks
    assert result["all"]["n_subjects"] == 10


# ---------------------------------------------------------------------------
# Tests: export_peb_covariates
# ---------------------------------------------------------------------------


def test_export_peb_covariates_csv_columns(tmp_path: Path):
    """CSV has correct columns and delta_waic = elpd_waic(3level) - elpd_waic(2level)."""
    from prl_hgf.analysis.bms import export_peb_covariates

    df2 = pd.DataFrame(
        {
            "participant_id": ["P0", "P1", "P2"],
            "phenotype": ["healthy", "high_anxiety", "high_anxiety"],
            "elpd_waic": [-100.0, -110.0, -120.0],
        }
    )
    df3 = pd.DataFrame(
        {
            "participant_id": ["P0", "P1", "P2"],
            "phenotype": ["healthy", "high_anxiety", "high_anxiety"],
            "elpd_waic": [-95.0, -108.0, -118.0],
        }
    )
    out_path = tmp_path / "peb.csv"
    out = export_peb_covariates(df2, df3, out_path)

    assert set(out.columns) == {"participant_id", "phenotype", "delta_waic", "delta_f"}, (
        f"Unexpected columns: {set(out.columns)}"
    )
    # delta_waic = 3level - 2level
    assert out.loc[out.participant_id == "P0", "delta_waic"].iloc[0] == pytest.approx(5.0)
    assert out.loc[out.participant_id == "P1", "delta_waic"].iloc[0] == pytest.approx(2.0)
    # CSV was written
    assert out_path.exists(), f"Output CSV not found at {out_path}"
    written = pd.read_csv(out_path)
    assert set(written.columns) == {"participant_id", "phenotype", "delta_waic", "delta_f"}
    assert len(written) == 3


def test_export_peb_covariates_inner_join_alignment(tmp_path: Path):
    """Extra participant in fit_df_3level is dropped via inner join."""
    from prl_hgf.analysis.bms import export_peb_covariates

    df2 = pd.DataFrame(
        {
            "participant_id": ["P0", "P1"],
            "phenotype": ["healthy", "healthy"],
            "elpd_waic": [-100.0, -110.0],
        }
    )
    # P99 only in df3 — should be silently dropped
    df3 = pd.DataFrame(
        {
            "participant_id": ["P0", "P1", "P99"],
            "phenotype": ["healthy", "healthy", "high_anxiety"],
            "elpd_waic": [-95.0, -105.0, -200.0],
        }
    )
    out = export_peb_covariates(df2, df3, tmp_path / "peb_align.csv")
    assert "P99" not in out["participant_id"].values, (
        "P99 should have been dropped by inner join"
    )
    assert len(out) == 2


def test_export_peb_covariates_empty_intersection_raises(tmp_path: Path):
    """ValueError raised when no participant_ids are shared."""
    from prl_hgf.analysis.bms import export_peb_covariates

    df2 = pd.DataFrame(
        {
            "participant_id": ["P0"],
            "phenotype": ["healthy"],
            "elpd_waic": [-100.0],
        }
    )
    df3 = pd.DataFrame(
        {
            "participant_id": ["P99"],
            "phenotype": ["high_anxiety"],
            "elpd_waic": [-90.0],
        }
    )
    with pytest.raises(ValueError, match="no participants in common"):
        export_peb_covariates(df2, df3, tmp_path / "peb_empty.csv")


def test_export_peb_covariates_creates_parent_dir(tmp_path: Path):
    """Output CSV parent directory is created if it does not exist."""
    from prl_hgf.analysis.bms import export_peb_covariates

    df2 = pd.DataFrame(
        {
            "participant_id": ["P0"],
            "phenotype": ["healthy"],
            "elpd_waic": [-100.0],
        }
    )
    df3 = pd.DataFrame(
        {
            "participant_id": ["P0"],
            "phenotype": ["healthy"],
            "elpd_waic": [-95.0],
        }
    )
    # nested/nonexistent subdir
    out_path = tmp_path / "new_subdir" / "deep" / "peb.csv"
    assert not out_path.parent.exists(), "Parent should not exist before call"
    export_peb_covariates(df2, df3, out_path)
    assert out_path.exists(), f"CSV not found at {out_path}"
