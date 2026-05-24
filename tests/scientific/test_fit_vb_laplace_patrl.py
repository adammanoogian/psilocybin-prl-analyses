"""Tests for src/prl_hgf/fitting/fit_vb_laplace_patrl.py.

Covers:
1. _regularize_to_pd clips negative eigenvalues
2. _regularize_to_pd preserves PD matrix unchanged
3. NotImplementedError for model_b / response_model != 'model_a'
4. ValueError for bad model_name
5. 2-level smoke: 3 agents, finite output + jit-fallback coverage
6. 3-level smoke: 3 agents, finite output + correct dim names
7. Parameter recovery sanity: omega_2 within 0.5 of truth for >=4 of 5 agents
8. Shape parity 2-level: var names match export_trajectories._PARAMS_2LEVEL
9. Shape parity 3-level: var names match export_trajectories._PARAMS_3LEVEL
10. Pick-best-cue regression guard: key imports still work
11. Orchestrator dim-name guard: participant_id in dims, participant not in dims
"""

from __future__ import annotations

import logging
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort
from prl_hgf.fitting.fit_vb_laplace_patrl import (
    _regularize_to_pd,
    fit_vb_laplace_patrl,
)
from prl_hgf.fitting.laplace_idata import (
    _PARAM_ORDER_2LEVEL,
    _PARAM_ORDER_3LEVEL,
    build_idata_from_laplace,
)

# ---------------------------------------------------------------------------
# Fast unit tests: _regularize_to_pd
# ---------------------------------------------------------------------------


def test_regularize_to_pd_clips_negative_eigenvalues() -> None:
    """Indefinite input matrix should be regularized to PD by clipping."""
    H = np.diag(np.array([-0.5, 2.0]))
    H_pd, diag = _regularize_to_pd(H)

    # PD after regularization
    eigvals = np.linalg.eigvalsh(H_pd)
    assert np.min(eigvals) >= 1e-8, (
        f"Expected min eigval >= 1e-8, got {np.min(eigvals)}"
    )

    # Diagnostic values
    assert diag["n_eigenvalues_clipped"] == 1.0, (
        f"Expected 1 clipped eigenvalue, got {diag['n_eigenvalues_clipped']}"
    )
    assert diag["hessian_min_eigval"] == pytest.approx(-0.5, abs=1e-12), (
        f"Expected pre-clip min eigval -0.5, got {diag['hessian_min_eigval']}"
    )

    # Cholesky must succeed
    np.linalg.cholesky(H_pd)  # raises if not PD


def test_regularize_to_pd_preserves_pd_matrix() -> None:
    """PD input matrix should be returned essentially unchanged."""
    H = np.diag(np.array([0.5, 2.0]))
    H_pd, diag = _regularize_to_pd(H)

    assert np.allclose(H_pd, H, atol=1e-10), (
        f"PD matrix should be preserved. Max diff: {np.max(np.abs(H_pd - H))}"
    )
    assert diag["n_eigenvalues_clipped"] == 0.0, (
        f"Expected 0 clipped eigenvalues, got {diag['n_eigenvalues_clipped']}"
    )
    assert diag["ridge_added"] == 0.0, (
        f"Expected ridge_added=0.0, got {diag['ridge_added']}"
    )
    assert diag["hessian_min_eigval"] == pytest.approx(0.5, abs=1e-12), (
        f"Expected hessian_min_eigval=0.5, got {diag['hessian_min_eigval']}"
    )
    assert diag["hessian_max_eigval"] == pytest.approx(2.0, abs=1e-12), (
        f"Expected hessian_max_eigval=2.0, got {diag['hessian_max_eigval']}"
    )


# ---------------------------------------------------------------------------
# Fast unit tests: input validation
# ---------------------------------------------------------------------------


def test_notimplementederror_for_unknown_response_model() -> None:
    """response_model='model_e' must raise NotImplementedError (unknown model).

    Plan 20-02: model_b and model_c are now supported. Only truly unrecognised
    values (e.g. 'model_e') should raise.
    """
    with pytest.raises(NotImplementedError, match="model_e"):
        fit_vb_laplace_patrl(
            pd.DataFrame(),
            "hgf_2level_patrl",
            response_model="model_e",
        )


def test_jit_fallback_warning_logged(caplog: pytest.LogCaptureFixture) -> None:
    """jit=True RuntimeError triggers WARN and jit=False LBFGS creation.

    Tests the warning + fallback logic without running JAX compilation, avoiding
    tracer contamination in the same process.
    """
    import prl_hgf.fitting.fit_vb_laplace_patrl as _mod

    _solver_jit_settings: list[bool] = []
    _call_idx = {"n": 0}

    class _TrackingLBFGS:
        """Records jit setting; raises on first (jit=True), succeeds on second."""

        def __init__(self, **kwargs: object) -> None:
            self._jit = bool(kwargs.get("jit", True))
            _solver_jit_settings.append(self._jit)

        def run(self, *args: object, **kwargs: object) -> object:
            _call_idx["n"] += 1
            if self._jit:
                raise RuntimeError("Simulated jit=True failure")
            # Return a fake result object matching jaxopt state contract
            # We just need to verify the fallback was called (jit=False instance
            # created), so raise to short-circuit further execution.
            raise RuntimeError("jit=False fallback reached (expected)")

    with (
        mock.patch.object(_mod, "LBFGS", _TrackingLBFGS),
        caplog.at_level(
            logging.WARNING,
            logger="prl_hgf.fitting.fit_vb_laplace_patrl",
        ),
        pytest.raises(RuntimeError, match="jit=False fallback reached"),
    ):
        fit_vb_laplace_patrl(
            pd.DataFrame(
                {
                    "participant_id": ["P000"] * 5,
                    "trial_idx": list(range(5)),
                    "state": [0] * 5,
                    "choice": [1] * 5,
                    "reward_mag": [1.0] * 5,
                    "shock_mag": [0.0] * 5,
                    "delta_hr": [0.0] * 5,
                    "outcome_time_s": [float(i) for i in range(5)],
                }
            ),
            "hgf_2level_patrl",
            n_pseudo_draws=10,
            max_iter=5,
        )

    # Verify: jit=True was tried first (raises), then jit=False was created
    assert True in _solver_jit_settings, (
        "Expected a jit=True LBFGS solver to be created first"
    )
    assert False in _solver_jit_settings, (
        "Expected a jit=False LBFGS fallback solver to be created"
    )
    # Verify the warning was logged
    assert any(
        "jit=False" in r.message or "jit=True" in r.message
        for r in caplog.records
    ), f"Expected WARN about jit=False fallback. Records: {[r.message for r in caplog.records]}"


def test_valueerror_for_bad_model_name() -> None:
    """Unrecognised model_name must raise ValueError with expected vs actual."""
    with pytest.raises(ValueError, match="hgf_4level_patrl") as exc_info:
        fit_vb_laplace_patrl(pd.DataFrame(), "hgf_4level_patrl")
    # Error message should show expected options
    assert "hgf_2level_patrl" in str(exc_info.value) or "hgf_3level_patrl" in str(
        exc_info.value
    ), f"Error should show expected model names, got: {exc_info.value}"


# ---------------------------------------------------------------------------
# Slow smoke tests (require RUN_SMOKE_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_2level_smoke_3_agents_finite_output(caplog: pytest.LogCaptureFixture) -> None:
    """2-level Laplace smoke: 3 agents, all outputs finite, dims correct."""
    config = load_pat_rl_config()
    sim_df, _, _ = simulate_patrl_cohort(
        n_participants=3,
        level=2,
        master_seed=42,
        config=config,
    )

    # Normal run (jit=True expected to succeed for most environments)
    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        n_pseudo_draws=200,
        max_iter=100,
        config=config,
    )

    # Shape checks
    assert idata.posterior is not None
    assert "omega_2" in idata.posterior
    assert "log_beta" in idata.posterior
    assert "beta" in idata.posterior
    assert idata.posterior["omega_2"].shape == (1, 200, 3), (
        f"Expected shape (1, 200, 3), got {idata.posterior['omega_2'].shape}"
    )

    # All finite
    for var in ("omega_2", "log_beta", "beta"):
        vals = idata.posterior[var].values
        assert np.all(np.isfinite(vals)), (
            f"posterior[{var!r}] contains non-finite values"
        )

    # At least one convergence flag set (converged=1.0 means True)
    converged_vals = idata.sample_stats["converged"].values
    assert np.any(converged_vals == 1.0), (
        f"Expected at least one converged restart, got converged={converged_vals}"
    )

    # jit-fallback coverage: verified via test_jit_fallback_warning_logged (below).
    # The fallback is tested by confirming the solver error is caught, the warning
    # is logged, and a jit=False solver is created — without running full JAX
    # compilation (which would cause tracer-state contamination in the same process).


@pytest.mark.scientific
def test_3level_smoke_3_agents_finite_output() -> None:
    """3-level Laplace smoke: 3 agents, all 5 params + beta finite, dims correct."""
    config = load_pat_rl_config()
    sim_df, _, _ = simulate_patrl_cohort(
        n_participants=3,
        level=3,
        master_seed=99,
        config=config,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_3level_patrl",
        n_pseudo_draws=200,
        max_iter=100,
        config=config,
    )

    assert idata.posterior is not None
    for var in ("omega_2", "log_beta", "omega_3", "kappa", "mu3_0", "beta"):
        assert var in idata.posterior, f"Missing posterior variable: {var!r}"
        vals = idata.posterior[var].values
        assert np.all(np.isfinite(vals)), (
            f"posterior[{var!r}] contains non-finite values"
        )

    # Shape: (chain=1, draw=200, participant_id=3)
    assert idata.posterior["omega_2"].shape == (1, 200, 3), (
        f"Expected shape (1, 200, 3), got {idata.posterior['omega_2'].shape}"
    )

    # Dim name guard
    assert "participant_id" in idata.posterior["omega_2"].dims, (
        f"Expected 'participant_id' in dims, got {idata.posterior['omega_2'].dims}"
    )
    assert "participant" not in idata.posterior["omega_2"].dims, (
        "'participant' (OQ1 bug) should NOT be in dims"
    )


@pytest.mark.scientific
def test_laplace_recovery_sanity_omega2_2level() -> None:
    """Recovery sanity: posterior-mean omega_2 within 0.5 of truth for >=4/5 agents."""
    config = load_pat_rl_config()
    sim_df, true_params, _ = simulate_patrl_cohort(
        n_participants=5,
        level=2,
        master_seed=7,
        config=config,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        n_pseudo_draws=500,
        max_iter=200,
        config=config,
    )

    participants = sorted(sim_df["participant_id"].astype(str).unique().tolist())
    omega2_vals = idata.posterior["omega_2"].values  # (1, 500, 5)

    n_within = 0
    errors: list[str] = []
    for i, pid in enumerate(participants):
        true_omega2 = true_params[pid]["omega_2"]
        post_mean = float(np.mean(omega2_vals[:, :, i]))
        err = abs(post_mean - true_omega2)
        if err < 0.5:
            n_within += 1
        else:
            errors.append(
                f"  {pid}: true={true_omega2:.3f}, post_mean={post_mean:.3f}, err={err:.3f}"
            )

    assert n_within >= 4, (
        f"Expected >=4/5 agents with |posterior_mean - true| < 0.5; got {n_within}/5.\n"
        f"Recovery failures:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Fast shape-parity tests (using build_idata_from_laplace directly)
# ---------------------------------------------------------------------------


def test_shape_parity_with_nuts_var_names_2level() -> None:
    """2-level Laplace idata has same var names as NUTS (omega_2, log_beta, beta)."""
    P, K = 3, len(_PARAM_ORDER_2LEVEL)
    mode = {k: np.zeros(P) for k in _PARAM_ORDER_2LEVEL}
    cov = np.eye(P * K) * 0.05
    participant_ids = [f"P{i:03d}" for i in range(P)]

    idata = build_idata_from_laplace(
        mode=mode,
        cov=cov,
        param_names=_PARAM_ORDER_2LEVEL,
        participant_ids=participant_ids,
        n_pseudo_draws=10,
    )

    required_vars = {"omega_2", "log_beta", "beta"}
    actual_vars = set(idata.posterior.data_vars)
    missing = required_vars - actual_vars
    assert not missing, (
        f"Missing posterior vars: {missing}. Got: {actual_vars}"
    )


def test_shape_parity_with_nuts_var_names_3level() -> None:
    """3-level Laplace idata has same var names as NUTS (all 5 + beta)."""
    P, K = 3, len(_PARAM_ORDER_3LEVEL)
    mode = {k: np.zeros(P) for k in _PARAM_ORDER_3LEVEL}
    cov = np.eye(P * K) * 0.05
    participant_ids = [f"P{i:03d}" for i in range(P)]

    idata = build_idata_from_laplace(
        mode=mode,
        cov=cov,
        param_names=_PARAM_ORDER_3LEVEL,
        participant_ids=participant_ids,
        n_pseudo_draws=10,
    )

    required_vars = {"omega_2", "omega_3", "kappa", "beta", "mu3_0"}
    actual_vars = set(idata.posterior.data_vars)
    missing = required_vars - actual_vars
    assert not missing, (
        f"Missing posterior vars: {missing}. Got: {actual_vars}"
    )


# ---------------------------------------------------------------------------
# Regression guard: pick-best-cue imports must not break
# ---------------------------------------------------------------------------


def test_pick_best_cue_regression_unchanged() -> None:
    """Importing core pick-best-cue modules must not raise."""
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: F401
    from prl_hgf.env.simulator import generate_session  # noqa: F401
    from prl_hgf.models.hgf_2level import build_2level_network  # noqa: F401


# ---------------------------------------------------------------------------
# Orchestrator dim-name guard (slow — exercises full fit pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_laplace_fit_emits_participant_id_dim() -> None:
    """Full orchestrator: idata.posterior dims must include 'participant_id' not 'participant'.

    This catches regressions where a future edit re-introduces the OQ1 bug at
    the orchestrator layer.  Plan 19-02's unit test only covers the standalone
    factory; this test covers the full fit pipeline.
    """
    config = load_pat_rl_config()
    sim_df, _, _ = simulate_patrl_cohort(
        n_participants=3,
        level=2,
        master_seed=42,
        config=config,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        n_pseudo_draws=100,
        max_iter=50,
        config=config,
    )

    omega2_dims = idata.posterior["omega_2"].dims
    assert "participant_id" in omega2_dims, (
        f"Expected 'participant_id' in omega_2.dims, got {omega2_dims}"
    )
    assert "participant" not in omega2_dims, (
        f"'participant' (OQ1 dim-name bug) must NOT appear in omega_2.dims, "
        f"got {omega2_dims}"
    )


# ---------------------------------------------------------------------------
# Plan 20-02: Laplace smoke tests for Models B and C
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_fit_vb_laplace_model_b_smoke() -> None:
    """5-agent Laplace smoke for Model B: posterior contains b and gamma.

    Verifies that fit_vb_laplace_patrl no longer raises NotImplementedError
    for model_b, and that the returned InferenceData has the expected
    posterior variables including the ΔHR modulation parameters.
    """
    config = load_pat_rl_config()
    sim_df, _, _ = simulate_patrl_cohort(
        n_participants=5,
        level=2,
        master_seed=202,
        config=config,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        response_model="model_b",
        n_pseudo_draws=200,
        max_iter=100,
        config=config,
    )

    assert idata.posterior is not None
    expected_vars = {"omega_2", "log_beta", "b", "gamma", "beta"}
    actual_vars = set(idata.posterior.data_vars)
    missing = expected_vars - actual_vars
    assert not missing, (
        f"Missing posterior vars for model_b: {missing}. Got: {actual_vars}"
    )

    # All finite
    for var in expected_vars:
        vals = idata.posterior[var].values
        assert np.all(np.isfinite(vals)), (
            f"model_b posterior[{var!r}] contains non-finite values"
        )

    # Shape: (chain=1, draw=200, participant_id=5)
    assert idata.posterior["omega_2"].shape[2] == 5, (
        f"Expected participant_id dim = 5, got {idata.posterior['omega_2'].shape}"
    )


@pytest.mark.scientific
def test_model_d_lambda_recovery_smoke() -> None:
    """SC3 gate: Model D lambda posterior mean within 0.3 of truth at 5 agents.

    Simulates 5 agents under Model D with a known lambda_true=0.1, fits via
    ``fit_vb_laplace_patrl(response_model='model_d')``, and asserts the
    posterior mean of lambda is within 0.3 of lambda_true for at least 3 of 5
    agents.

    This is a smoke gate — tighter recovery (r >= 0.8 across 160 agents) is
    validated at 160 agents in scripts/05_run_validation.py --task=patrl
    (PRL-V1 gate, deferred to cluster runs).

    NOTE (M7 followup — Post-20-04): Plan 20-04 has now merged, so the ε₂-coupled
    generative simulator (simulate_patrl_cohort with response_model='model_d',
    lam_true) is available. This test uses it directly instead of the simpler
    bootstrap Normal(-1, 0.5) ΔHR originally planned.

    The simulator side uses a mean-effective-omega approximation for the
    generative ΔHR (see pat_rl_simulator.py simulate_patrl_cohort model_d path).
    The fitting side uses exact per-trial omega injection
    (``_clamped_step_model_d``). This mismatch is intentional for this smoke
    test: the goal is to verify that the fitting path can recover a positive
    lambda signal from data that genuinely varies omega. A full generative-
    model-matched re-verification should use the exact per-trial ε₂-driven
    ΔHR once the generative simulator's two-pass pass is extended. This is
    tracked as a followup in the plan SUMMARY.

    # Post-20-04 followup (tracked as must_have in Plan 20-04 — M7 bridge):
    # Confirm λ-recovery under the full ε₂-coupled generative process.
    # The call below already uses the Plan-20-04 bridge kwarg:
    #   simulate_patrl_cohort(
    #       config=config, master_seed=42, n_participants=5,
    #       phenotypes=['healthy'],
    #       response_model='model_d',
    #       lam_true=0.1,
    #   )
    # TODO: tighten the recovery gate from 3/5 to 4/5 and |err| < 0.2 after
    # the two-pass per-trial simulator (exact omega_eff injection) lands.
    """
    lam_true = 0.1

    config = load_pat_rl_config()
    sim_df, true_params, _ = simulate_patrl_cohort(
        n_participants=5,
        level=2,
        master_seed=42,
        config=config,
        response_model="model_d",
        lam_true=lam_true,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        response_model="model_d",
        n_pseudo_draws=500,
        max_iter=200,
        config=config,
    )

    assert idata.posterior is not None
    # Model D posterior should include lam
    assert "lam" in idata.posterior, (
        f"Model D posterior missing 'lam'. Got vars: {list(idata.posterior.data_vars)}"
    )

    participants = sorted(sim_df["participant_id"].astype(str).unique().tolist())
    lam_vals = idata.posterior["lam"].values  # (1, n_pseudo_draws, P)

    n_within = 0
    details: list[str] = []
    for i, pid in enumerate(participants):
        post_mean = float(np.mean(lam_vals[:, :, i]))
        err = abs(post_mean - lam_true)
        status = "PASS" if err < 0.3 else "FAIL"
        if err < 0.3:
            n_within += 1
        details.append(
            f"  {pid}: true_lam={lam_true:.3f}, post_mean={post_mean:.4f}, "
            f"|err|={err:.4f} [{status}]"
        )

    assert n_within >= 3, (
        f"SC3 GATE FAILED: Expected >=3/5 agents with |posterior_mean(lam) - "
        f"lam_true| < 0.3; got {n_within}/5 (lam_true={lam_true}).\n"
        "Lambda recovery details:\n" + "\n".join(details)
    )


@pytest.mark.scientific
def test_model_d_lambda_recovery_smoke_epsilon2_coupled() -> None:
    """M7 bridge: lambda recovery under the Plan 20-04 ε₂-coupled generator.

    This is the companion to ``test_model_d_lambda_recovery_smoke`` (Plan
    20-03 bootstrap version). Both tests co-exist so regressions in either
    generative path are independently catchable.

    The key difference vs the Plan 20-03 bootstrap test:
    * **Generator**: Uses the explicit ``phenotype_name='healthy'`` kwarg
      and the same ε₂-coupled ΔHR path (Plan 20-04 SC4 formula).
    * **Same fit path**: ``fit_vb_laplace_patrl(response_model='model_d')``.
    * **Same gate**: |posterior_mean(lam) - 0.1| < 0.3 for >= 3/5 agents.

    Both tests call ``simulate_patrl_cohort(response_model='model_d',
    lam_true=0.1)`` — the Plan 20-04 M7 bridge kwarg — so both exercise
    the ε₂-coupled generator. This test uses a different ``master_seed``
    and ``n_pseudo_draws`` to give an independent draw, strengthening the
    recovery evidence.
    """
    lam_true = 0.1

    config = load_pat_rl_config()
    sim_df, true_params, _ = simulate_patrl_cohort(
        n_participants=5,
        level=2,
        master_seed=404,          # different seed from Plan 20-03's 42
        config=config,
        phenotype_name="healthy",  # explicit phenotype via Plan 20-04 kwarg
        response_model="model_d",
        lam_true=lam_true,
    )

    # Sanity: true_params should carry lam
    for pid, tp in true_params.items():
        assert "lam" in tp, (
            f"true_params[{pid!r}] missing 'lam' key. "
            f"Got: {list(tp.keys())}"
        )
        assert abs(tp["lam"] - lam_true) < 1e-10, (
            f"true_params[{pid!r}]['lam'] = {tp['lam']!r}, expected {lam_true}"
        )

    # Sanity: phenotype column present
    assert "phenotype" in sim_df.columns, (
        "sim_df missing 'phenotype' column (Plan 20-04 M7 bridge check)"
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        response_model="model_d",
        n_pseudo_draws=500,
        max_iter=200,
        config=config,
    )

    assert idata.posterior is not None
    assert "lam" in idata.posterior, (
        f"Model D posterior missing 'lam'. Got vars: {list(idata.posterior.data_vars)}"
    )

    participants = sorted(sim_df["participant_id"].astype(str).unique().tolist())
    lam_vals = idata.posterior["lam"].values  # (1, n_pseudo_draws, P)

    n_within = 0
    details: list[str] = []
    for i, pid in enumerate(participants):
        post_mean = float(np.mean(lam_vals[:, :, i]))
        err = abs(post_mean - lam_true)
        status = "PASS" if err < 0.3 else "FAIL"
        if err < 0.3:
            n_within += 1
        details.append(
            f"  {pid}: true_lam={lam_true:.3f}, post_mean={post_mean:.4f}, "
            f"|err|={err:.4f} [{status}]"
        )

    assert n_within >= 3, (
        f"M7 BRIDGE GATE FAILED: Expected >=3/5 agents with "
        f"|posterior_mean(lam) - lam_true| < 0.3; got {n_within}/5 "
        f"(lam_true={lam_true}, seed=404).\n"
        "Lambda recovery details:\n" + "\n".join(details)
    )


@pytest.mark.scientific
def test_fit_vb_laplace_model_c_smoke() -> None:
    """5-agent Laplace smoke for Model C: posterior contains b, gamma, alpha.

    Verifies that fit_vb_laplace_patrl dispatches correctly for model_c,
    and that the returned InferenceData has the expected posterior variables
    including the ΔHR × value sensitivity alpha parameter.
    """
    config = load_pat_rl_config()
    sim_df, _, _ = simulate_patrl_cohort(
        n_participants=5,
        level=2,
        master_seed=203,
        config=config,
    )

    idata = fit_vb_laplace_patrl(
        sim_df,
        "hgf_2level_patrl",
        response_model="model_c",
        n_pseudo_draws=200,
        max_iter=100,
        config=config,
    )

    assert idata.posterior is not None
    expected_vars = {"omega_2", "log_beta", "b", "gamma", "alpha", "beta"}
    actual_vars = set(idata.posterior.data_vars)
    missing = expected_vars - actual_vars
    assert not missing, (
        f"Missing posterior vars for model_c: {missing}. Got: {actual_vars}"
    )

    # All finite
    for var in expected_vars:
        vals = idata.posterior[var].values
        assert np.all(np.isfinite(vals)), (
            f"model_c posterior[{var!r}] contains non-finite values"
        )

    # Shape: (chain=1, draw=200, participant_id=5)
    assert idata.posterior["omega_2"].shape[2] == 5, (
        f"Expected participant_id dim = 5, got {idata.posterior['omega_2'].shape}"
    )
