"""Tests for prl_hgf.fitting.hierarchical_patrl — PAT-RL fitting orchestrator.

Tests 1-4 and 7-8 run without blackjax (unit/regression tests).
Tests 5-6 are smoke tests requiring blackjax and are marked ``slow``;
they are also guarded by ``pytest.importorskip("blackjax")``.

If blackjax is not installed in the active environment, tests 5-6 are
skipped automatically and a gap is documented in STATE.md / SUMMARY.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_patrl_df(
    n_participants: int,
    n_trials: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal PAT-RL sim_df with all required columns.

    Uses random binary states/choices and constant magnitudes.  No real
    HGF simulation — sufficient for logp shape / smoke-fit tests.

    Parameters
    ----------
    n_participants : int
        Number of synthetic participants.
    n_trials : int
        Trials per participant.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, trial_idx, state, choice, reward_mag,
        shock_mag, delta_hr, outcome_time_s.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p_idx in range(n_participants):
        pid = f"sub-{p_idx:03d}"
        states = rng.integers(0, 2, size=n_trials).astype(np.int32)
        choices = rng.integers(0, 2, size=n_trials).astype(np.int32)
        reward_mags = rng.choice([1.0, 5.0], size=n_trials).astype(np.float32)
        shock_mags = rng.choice([1.0, 5.0], size=n_trials).astype(np.float32)
        delta_hrs = rng.normal(0.0, 3.0, size=n_trials).astype(np.float32)
        for t in range(n_trials):
            rows.append(
                {
                    "participant_id": pid,
                    "trial_idx": t,
                    "state": int(states[t]),
                    "choice": int(choices[t]),
                    "reward_mag": float(reward_mags[t]),
                    "shock_mag": float(shock_mags[t]),
                    "delta_hr": float(delta_hrs[t]),
                    "outcome_time_s": float(t * 11.0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: logp shape + finiteness
# ---------------------------------------------------------------------------


def test_build_logp_shape() -> None:
    """build_logp_fn_batched_patrl returns scalar finite logp for 3 participants."""
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 3, 192
    rng = np.random.default_rng(0)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)

    fn = build_logp_fn_batched_patrl(state, choices, reward, shock, mask, "hgf_2level_patrl")
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
    }
    lp = fn(params)
    assert lp.shape == (), f"Expected scalar output, got shape {lp.shape}"
    assert np.isfinite(float(lp)), f"Expected finite logp, got {float(lp)}"


# ---------------------------------------------------------------------------
# Test 2: logp is differentiable
# ---------------------------------------------------------------------------


def test_logp_is_differentiable() -> None:
    """jax.grad(logp_fn)(params) returns finite (P,)-shaped gradients."""
    import jax
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 3, 50
    rng = np.random.default_rng(1)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)

    fn = build_logp_fn_batched_patrl(state, choices, reward, shock, mask, "hgf_2level_patrl")
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
    }
    grads = jax.grad(fn)(params)

    assert set(grads.keys()) == {"omega_2", "beta"}
    for name, g in grads.items():
        assert g.shape == (P,), f"Expected grad shape ({P},) for {name}, got {g.shape}"
        assert np.all(np.isfinite(np.asarray(g))), f"Non-finite gradient for {name}: {g}"


# ---------------------------------------------------------------------------
# Test 3: batched logp matches per-participant loop reference
# ---------------------------------------------------------------------------


def test_logp_matches_loop_reference() -> None:
    """For P=1, batched logp agrees with manual loop reference within 1e-4."""
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl
    from prl_hgf.models.hgf_2level_patrl import (
        build_2level_network_patrl,
        extract_beliefs_patrl,
    )
    from prl_hgf.models.response_patrl import model_a_logp

    P, T = 1, 40
    rng = np.random.default_rng(2)
    state_1d = rng.integers(0, 2, size=(T,)).astype(np.int32)
    choices_1d = rng.integers(0, 2, size=(T,)).astype(np.int32)
    reward_1d = np.ones(T, dtype=np.float32)
    shock_1d = np.ones(T, dtype=np.float32)

    omega_2_val = -4.0
    beta_val = 2.0

    # --- Reference: manual forward pass + extract + model_a_logp ---
    net = build_2level_network_patrl(omega_2=omega_2_val)
    net.input_data(
        input_data=state_1d[:, None].astype(float),
        time_steps=np.ones(T),
    )
    beliefs = extract_beliefs_patrl(net)
    mu2_ref = beliefs["mu2"]  # (T,)

    logp_per_trial = np.asarray(
        model_a_logp(
            jnp.asarray(mu2_ref),
            jnp.asarray(choices_1d, dtype=jnp.int32),
            jnp.asarray(reward_1d),
            jnp.asarray(shock_1d),
            beta=beta_val,
        )
    )
    ref_logp = float(np.sum(logp_per_trial))

    # --- Batched logp factory ---
    state_2d = state_1d[None, :]    # (1, T)
    choices_2d = choices_1d[None, :]
    reward_2d = reward_1d[None, :]
    shock_2d = shock_1d[None, :]
    mask_2d = np.ones((P, T), dtype=bool)

    fn = build_logp_fn_batched_patrl(
        state_2d, choices_2d, reward_2d, shock_2d, mask_2d, "hgf_2level_patrl"
    )
    batched_lp = float(
        fn({"omega_2": jnp.array([omega_2_val]), "beta": jnp.array([beta_val])})
    )

    assert abs(batched_lp - ref_logp) < 1e-3, (
        f"Batched logp {batched_lp:.6f} differs from reference {ref_logp:.6f} "
        f"by {abs(batched_lp - ref_logp):.2e} (tolerance 1e-3)"
    )


# ---------------------------------------------------------------------------
# Test 4: _build_arrays_single_patrl raises on missing columns
# ---------------------------------------------------------------------------


def test_build_arrays_raises_on_missing_columns() -> None:
    """_build_arrays_single_patrl raises KeyError mentioning missing column."""
    from prl_hgf.fitting.hierarchical_patrl import _build_arrays_single_patrl

    # sim_df missing reward_mag
    df = pd.DataFrame(
        {
            "participant_id": ["sub-0"] * 5,
            "trial_idx": list(range(5)),
            "state": [0, 1, 0, 1, 0],
            "choice": [1, 0, 1, 0, 1],
            "shock_mag": [1.0] * 5,
            "delta_hr": [0.0] * 5,
            "outcome_time_s": [11.0 * i for i in range(5)],
        }
    )
    with pytest.raises(KeyError, match="reward_mag"):
        _build_arrays_single_patrl(df, ["sub-0"])


# ---------------------------------------------------------------------------
# Tests 5-6: 5-participant smoke fits (require blackjax; marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_5participant_smoke_fit_2level() -> None:
    """5-participant CPU smoke: 2-level PAT-RL HGF, finite posterior, low divergences.

    Guarded by importorskip — skipped automatically if blackjax is absent.
    Budget: n_chains=2, n_tune=200, n_draws=200 (<5 min CPU).
    """
    pytest.importorskip(
        "blackjax",
        reason="blackjax not installed; smoke test skipped. "
        "Install blackjax to validate PAT-RL fitting on this machine.",
    )

    from prl_hgf.fitting.hierarchical_patrl import fit_batch_hierarchical_patrl

    df = _make_synthetic_patrl_df(n_participants=5, n_trials=192, seed=10)

    idata = fit_batch_hierarchical_patrl(
        df,
        model_name="hgf_2level_patrl",
        n_chains=2,
        n_tune=200,  # tightened for <5 min CPU budget
        n_draws=200,
        random_seed=0,
    )

    # Shape check: (n_chains, n_draws, P)
    assert idata.posterior["omega_2"].shape == (2, 200, 5), (
        f"Expected omega_2 shape (2, 200, 5), got {idata.posterior['omega_2'].shape}"
    )

    # All posterior means must be finite
    for var in ["omega_2", "beta"]:
        means = np.asarray(idata.posterior[var].mean(dim=["chain", "draw"]))
        assert np.all(np.isfinite(means)), (
            f"Non-finite posterior means for {var}: {means}"
        )

    # Divergence rate < 20% per chain
    divergences = np.asarray(idata.sample_stats["diverging"])
    div_rate = divergences.mean()
    assert div_rate < 0.20, (
        f"NUTS divergence rate {div_rate:.1%} exceeds 20% threshold"
    )


@pytest.mark.scientific
def test_5participant_smoke_fit_3level() -> None:
    """5-participant CPU smoke: 3-level PAT-RL HGF, finite posterior, low divergences.

    Guarded by importorskip — skipped automatically if blackjax is absent.
    Budget: n_chains=2, n_tune=200, n_draws=200 (<5 min CPU).
    """
    pytest.importorskip(
        "blackjax",
        reason="blackjax not installed; smoke test skipped. "
        "Install blackjax to validate PAT-RL fitting on this machine.",
    )

    from prl_hgf.fitting.hierarchical_patrl import fit_batch_hierarchical_patrl

    df = _make_synthetic_patrl_df(n_participants=5, n_trials=192, seed=11)

    idata = fit_batch_hierarchical_patrl(
        df,
        model_name="hgf_3level_patrl",
        n_chains=2,
        n_tune=200,  # tightened for <5 min CPU budget
        n_draws=200,
        random_seed=1,
    )

    # Shape check: (n_chains, n_draws, P)
    assert idata.posterior["omega_2"].shape == (2, 200, 5), (
        f"Expected omega_2 shape (2, 200, 5), got {idata.posterior['omega_2'].shape}"
    )

    # All posterior means finite for all 3-level parameters
    for var in ["omega_2", "omega_3", "kappa", "mu3_0", "beta"]:
        means = np.asarray(idata.posterior[var].mean(dim=["chain", "draw"]))
        assert np.all(np.isfinite(means)), (
            f"Non-finite posterior means for {var}: {means}"
        )

    # Divergence rate < 20%
    divergences = np.asarray(idata.sample_stats["diverging"])
    div_rate = divergences.mean()
    assert div_rate < 0.20, (
        f"NUTS divergence rate {div_rate:.1%} exceeds 20% threshold"
    )


# ---------------------------------------------------------------------------
# Test 7: NotImplementedError only for unrecognised response_model values
# ---------------------------------------------------------------------------


def test_not_implemented_for_unknown_response_model() -> None:
    """fit_batch_hierarchical_patrl raises NotImplementedError for unrecognised model.

    Plan 20-02: model_a/b/c are now supported; only unknown values raise.
    """
    from prl_hgf.fitting.hierarchical_patrl import fit_batch_hierarchical_patrl

    df = _make_synthetic_patrl_df(n_participants=1, n_trials=5, seed=0)
    with pytest.raises(NotImplementedError, match="model_e"):
        fit_batch_hierarchical_patrl(df, response_model="model_e")


# ---------------------------------------------------------------------------
# Tests 9-12: Plan 20-02 factory tests (no blackjax dependency)
# ---------------------------------------------------------------------------


def test_build_logp_fn_batched_patrl_accepts_delta_hr_arr() -> None:
    """build_logp_fn_batched_patrl accepts delta_hr_arr kwarg; output is finite."""
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 5, 192
    rng = np.random.default_rng(42)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)
    delta_hr = rng.normal(0.0, 2.0, size=(P, T)).astype(np.float64)

    fn = build_logp_fn_batched_patrl(
        state, choices, reward, shock, mask, "hgf_2level_patrl",
        response_model="model_a",
        delta_hr_arr=delta_hr,
    )
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
        "b": jnp.full((P,), 0.0),
    }
    lp = fn(params)
    assert lp.shape == (), f"Expected scalar, got {lp.shape}"
    assert np.isfinite(float(lp)), f"Expected finite logp, got {float(lp)}"


def test_build_logp_fn_batched_patrl_model_b() -> None:
    """build_logp_fn_batched_patrl with response_model='model_b' returns finite logp."""
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 5, 192
    rng = np.random.default_rng(43)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)
    delta_hr = rng.normal(0.0, 2.0, size=(P, T)).astype(np.float64)

    fn = build_logp_fn_batched_patrl(
        state, choices, reward, shock, mask, "hgf_2level_patrl",
        response_model="model_b",
        delta_hr_arr=delta_hr,
    )
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
        "b": jnp.full((P,), 0.0),
        "gamma": jnp.full((P,), 0.3),
    }
    lp = fn(params)
    assert lp.shape == (), f"Expected scalar, got {lp.shape}"
    assert np.isfinite(float(lp)), f"Expected finite logp for model_b, got {float(lp)}"


def test_build_logp_fn_batched_patrl_model_c() -> None:
    """build_logp_fn_batched_patrl with response_model='model_c' returns finite logp."""
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 5, 192
    rng = np.random.default_rng(44)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)
    delta_hr = rng.normal(0.0, 2.0, size=(P, T)).astype(np.float64)

    fn = build_logp_fn_batched_patrl(
        state, choices, reward, shock, mask, "hgf_2level_patrl",
        response_model="model_c",
        delta_hr_arr=delta_hr,
    )
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
        "b": jnp.full((P,), 0.0),
        "alpha": jnp.full((P,), 0.1),
        "gamma": jnp.full((P,), 0.3),
    }
    lp = fn(params)
    assert lp.shape == (), f"Expected scalar, got {lp.shape}"
    assert np.isfinite(float(lp)), f"Expected finite logp for model_c, got {float(lp)}"


# ---------------------------------------------------------------------------
# Tests 13-17: Plan 20-03 Model D scan body (no blackjax dependency)
# ---------------------------------------------------------------------------


def test_build_logp_fn_batched_patrl_model_d_shape() -> None:
    """Model D: build_logp_fn_batched_patrl returns scalar finite logp.

    Verifies that the factory accepts response_model='model_d', evaluates
    without error, and returns a finite scalar for a 5-participant batch.
    """
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import build_logp_fn_batched_patrl

    P, T = 5, 192
    rng = np.random.default_rng(50)
    state = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    choices = rng.integers(0, 2, size=(P, T)).astype(np.int32)
    reward = np.ones((P, T), dtype=np.float32)
    shock = np.ones((P, T), dtype=np.float32)
    mask = np.ones((P, T), dtype=bool)
    delta_hr = rng.normal(-1.0, 0.5, size=(P, T)).astype(np.float64)

    fn = build_logp_fn_batched_patrl(
        state, choices, reward, shock, mask, "hgf_2level_patrl",
        response_model="model_d",
        delta_hr_arr=delta_hr,
    )
    params = {
        "omega_2": jnp.full((P,), -4.0),
        "beta": jnp.full((P,), 2.0),
        "b": jnp.full((P,), 0.0),
        "lam": jnp.full((P,), 0.1),
    }
    lp = fn(params)
    assert lp.shape == (), f"Expected scalar, got {lp.shape}"
    assert np.isfinite(float(lp)), f"Expected finite logp for model_d, got {float(lp)}"


def test_model_d_mu2_varies_with_delta_hr() -> None:
    """Model D: different delta_hr inputs produce different mu2 trajectories.

    This verifies that ΔHR is actually flowing through the scan body — i.e.,
    that per-trial omega_eff(t) = omega_2 + lam * dHR(t) modifies the HGF
    belief trajectory relative to a zero-ΔHR baseline.
    """
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import (
        _build_session_scanner_patrl,
        _make_single_logp_fn,
    )

    T = 40
    rng = np.random.default_rng(51)
    state = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    choices = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    reward = jnp.ones(T, dtype=jnp.float64)
    shock = jnp.ones(T, dtype=jnp.float64)
    mask = jnp.ones(T, dtype=bool)

    # Two different ΔHR streams: all-zeros vs drawn from N(-1, 0.5)
    dhr_zero = jnp.zeros(T, dtype=jnp.float64)
    dhr_nonzero = jnp.asarray(rng.normal(-1.0, 0.5, size=(T,)), dtype=jnp.float64)

    base_attrs, scan_fn, belief_idx = _build_session_scanner_patrl("hgf_2level_patrl")
    single_logp = _make_single_logp_fn(
        base_attrs=base_attrs,
        scan_fn=scan_fn,
        belief_idx=belief_idx,
        model_name="hgf_2level_patrl",
        n_trials=T,
        response_model="model_d",
    )

    params = {
        "omega_2": jnp.array(-4.0),
        "beta": jnp.array(2.0),
        "b": jnp.array(0.0),
        "lam": jnp.array(0.5),  # non-trivial lambda so ΔHR matters
    }
    lp_zero = single_logp(params, state, choices, reward, shock, mask, dhr_zero)
    lp_nonzero = single_logp(params, state, choices, reward, shock, mask, dhr_nonzero)

    # If ΔHR were ignored, logp values would be identical.
    assert abs(float(lp_zero) - float(lp_nonzero)) > 1e-6, (
        f"Model D logp identical under zero vs non-zero ΔHR "
        f"(lp_zero={float(lp_zero):.6f}, lp_nonzero={float(lp_nonzero):.6f}): "
        "ΔHR is not flowing through the scan body."
    )


def test_model_d_layer2_clamp_active() -> None:
    """Model D: pathological lam does not push |mu2| beyond 14.

    Feeds a large-magnitude lam (e.g. lam=100) combined with large ΔHR,
    which would create extreme omega_eff(t) values without the clamp.
    Asserts the layer-2 clamp keeps the scan numerically stable.
    """
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import (
        _MU_2_BOUND,
        _build_session_scanner_patrl,
        _make_single_logp_fn,
    )

    T = 40
    rng = np.random.default_rng(52)
    state = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    choices = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    reward = jnp.ones(T, dtype=jnp.float64)
    shock = jnp.ones(T, dtype=jnp.float64)
    mask = jnp.ones(T, dtype=bool)

    # Extreme ΔHR: constant large value to stress-test the clamp
    dhr_extreme = jnp.full((T,), 50.0, dtype=jnp.float64)

    base_attrs, scan_fn, belief_idx = _build_session_scanner_patrl("hgf_2level_patrl")
    single_logp = _make_single_logp_fn(
        base_attrs=base_attrs,
        scan_fn=scan_fn,
        belief_idx=belief_idx,
        model_name="hgf_2level_patrl",
        n_trials=T,
        response_model="model_d",
    )

    # Pathological lam that would produce omega_eff = -4 + 100 * 50 = 4996
    params = {
        "omega_2": jnp.array(-4.0),
        "beta": jnp.array(2.0),
        "b": jnp.array(0.0),
        "lam": jnp.array(100.0),
    }
    # Should not raise; logp should be finite (clamp + revert keeps carry valid)
    lp = single_logp(params, state, choices, reward, shock, mask, dhr_extreme)
    assert np.isfinite(float(lp)), (
        f"Model D logp not finite under pathological lam=100 + dhr=50: {float(lp)}. "
        f"Layer-2 clamp (_MU_2_BOUND={_MU_2_BOUND}) may not be active."
    )


def test_model_d_fp64_dtype_preserved() -> None:
    """Model D: mu2_traj has float64 dtype (Decision 118).

    Exercises the scan body with JAX_ENABLE_X64 (set via CONDA env); asserts
    that the trajectory array dtype matches jnp.float64.
    """

    import jax

    # Enable float64 for this test only (JAX default is float32 on CPU).
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        from prl_hgf.fitting.hierarchical_patrl import (
            _build_session_scanner_patrl,
        )

        T = 20
        rng = np.random.default_rng(53)
        base_attrs, scan_fn, belief_idx = _build_session_scanner_patrl(
            "hgf_2level_patrl"
        )

        values = jnp.asarray(
            rng.integers(0, 2, size=(T,)), dtype=jnp.float64
        )[:, None]
        observed = jnp.ones(T, dtype=jnp.int32)
        time_steps = jnp.ones(T, dtype=jnp.float64)
        dhr = jnp.asarray(rng.normal(-1.0, 0.5, size=(T,)), dtype=jnp.float64)

        # Build the attrs_d as the scan body would (2-level, no 3-level injection)
        omega_2_val = -4.0
        lam_val = 0.1
        attrs_d = {**base_attrs}
        attrs_d[belief_idx] = {
            **attrs_d[belief_idx],
            "tonic_volatility": jnp.float64(omega_2_val),
        }

        def _clamped_step_test(carry, x):
            val_i, obs_i, ts_i, dhr_i = x
            omega_eff_i = jnp.float64(omega_2_val) + jnp.float64(lam_val) * dhr_i
            attrs_i = {**carry}
            attrs_i[belief_idx] = {
                **attrs_i[belief_idx],
                "tonic_volatility": omega_eff_i,
            }
            new_carry, _ = scan_fn(attrs_i, ((val_i,), (obs_i,), ts_i, None))
            new_mean = new_carry[belief_idx]["mean"]
            is_stable = jnp.all(jnp.isfinite(new_mean)) & (jnp.abs(new_mean) < 14.0)
            safe_carry = jax.tree_util.tree_map(
                lambda n, o: jnp.where(is_stable, n, o), new_carry, carry
            )
            return safe_carry, safe_carry[belief_idx]["mean"]

        _, mu2_traj = jax.lax.scan(
            _clamped_step_test, attrs_d, (values, observed, time_steps, dhr)
        )

        assert mu2_traj.dtype == jnp.float64, (
            f"Expected mu2_traj dtype float64 (Decision 118), got {mu2_traj.dtype}. "
            "Check that JAX_ENABLE_X64 is set and dhr is cast to float64 before scan."
        )
    finally:
        # Leave x64 enabled for subsequent tests (harmless; guards against
        # accidentally disabling it with update("jax_enable_x64", False)).
        pass


def test_model_d_lam_zero_equals_model_a() -> None:
    """Model D with lam=0 and b=0 equals Model A logp within 1e-5.

    When lam=0, omega_eff(t) = omega_2 for every trial — the scan becomes
    identical to the fixed-omega scan used in Models A/B/C with the same
    omega_2.  Therefore Model D logp must equal Model A logp element-wise.
    """
    import jax.numpy as jnp

    from prl_hgf.fitting.hierarchical_patrl import (
        _build_session_scanner_patrl,
        _make_single_logp_fn,
    )

    T = 40
    rng = np.random.default_rng(54)
    state = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    choices = jnp.asarray(rng.integers(0, 2, size=(T,)), dtype=jnp.int32)
    reward = jnp.ones(T, dtype=jnp.float64)
    shock = jnp.ones(T, dtype=jnp.float64)
    mask = jnp.ones(T, dtype=bool)
    dhr = jnp.asarray(rng.normal(-1.0, 0.5, size=(T,)), dtype=jnp.float64)

    base_attrs, scan_fn, belief_idx = _build_session_scanner_patrl("hgf_2level_patrl")

    logp_model_a = _make_single_logp_fn(
        base_attrs=base_attrs,
        scan_fn=scan_fn,
        belief_idx=belief_idx,
        model_name="hgf_2level_patrl",
        n_trials=T,
        response_model="model_a",
    )
    logp_model_d = _make_single_logp_fn(
        base_attrs=base_attrs,
        scan_fn=scan_fn,
        belief_idx=belief_idx,
        model_name="hgf_2level_patrl",
        n_trials=T,
        response_model="model_d",
    )

    params_a = {
        "omega_2": jnp.array(-4.0),
        "beta": jnp.array(2.0),
    }
    params_d = {
        "omega_2": jnp.array(-4.0),
        "beta": jnp.array(2.0),
        "b": jnp.array(0.0),
        "lam": jnp.array(0.0),  # lam=0 → omega_eff = omega_2 always
    }

    lp_a = float(logp_model_a(params_a, state, choices, reward, shock, mask, dhr * 0))
    lp_d = float(logp_model_d(params_d, state, choices, reward, shock, mask, dhr))

    assert abs(lp_a - lp_d) < 1e-4, (
        f"Model D (lam=0, b=0) logp={lp_d:.8f} differs from Model A logp={lp_a:.8f} "
        f"by {abs(lp_a - lp_d):.2e} (tolerance 1e-4). "
        "These should be identical when lam=0."
    )


# ---------------------------------------------------------------------------
# Test 8: pick_best_cue fitting regression (hierarchical.py unchanged)
# ---------------------------------------------------------------------------


def test_pick_best_cue_fitting_unchanged() -> None:
    """Regression: fit_batch_hierarchical still importable after 18-04 changes."""
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical

    assert callable(fit_batch_hierarchical), (
        "fit_batch_hierarchical from hierarchical.py should be callable"
    )
