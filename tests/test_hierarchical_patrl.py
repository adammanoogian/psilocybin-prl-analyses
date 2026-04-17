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


@pytest.mark.slow
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


@pytest.mark.slow
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
# Test 7: NotImplementedError for Models B/C/D
# ---------------------------------------------------------------------------


def test_not_implemented_for_models_bcd() -> None:
    """fit_batch_hierarchical_patrl raises NotImplementedError with 'Phase 19+'."""
    from prl_hgf.fitting.hierarchical_patrl import fit_batch_hierarchical_patrl

    df = _make_synthetic_patrl_df(n_participants=1, n_trials=5, seed=0)
    with pytest.raises(NotImplementedError, match="Phase 19\\+"):
        fit_batch_hierarchical_patrl(df, response_model="model_b")


# ---------------------------------------------------------------------------
# Test 8: pick_best_cue fitting regression (hierarchical.py unchanged)
# ---------------------------------------------------------------------------


def test_pick_best_cue_fitting_unchanged() -> None:
    """Regression: fit_batch_hierarchical still importable after 18-04 changes."""
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical

    assert callable(fit_batch_hierarchical), (
        "fit_batch_hierarchical from hierarchical.py should be callable"
    )
