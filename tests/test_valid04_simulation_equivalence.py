"""VALID-04: Statistical equivalence between legacy and JAX simulation paths.

Verifies that ``simulate_agent`` (NumPy for-loop) and ``simulate_session_jax``
(JAX ``lax.scan``) produce statistically equivalent choice frequency
distributions per cue per phase over 100 replicates.

Per-trial exact match is NOT expected: the two paths use different RNG streams
(NumPy PCG64 vs JAX ThreeFry) and different floating-point accumulation orders.
Only aggregate statistics are compared via two-sample KS tests.

Runtime: ~5-10 minutes (100 replicates x 420 trials each).
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ks_2samp

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import load_config
from prl_hgf.models.hgf_3level import build_3level_network
from prl_hgf.simulation.agent import simulate_agent
from prl_hgf.simulation.jax_session import simulate_session_jax

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_REPLICATES = 100
N_CUES = 3

# Fixed parameters for all replicates.
# Moderate values that produce meaningful (non-uniform, non-degenerate) behavior.
OMEGA_2 = -3.0
OMEGA_3 = -6.0
KAPPA = 1.0
BETA = 2.0
ZETA = 0.5


# ---------------------------------------------------------------------------
# VALID-04 sanity check — both paths produce choices in {0, 1, 2}
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_valid04_choice_range() -> None:
    """Quick sanity check: both paths produce valid choices for one replicate.

    Catches gross implementation bugs (e.g. wrong output shape, out-of-range
    cue indices) before the long 100-replicate equivalence test runs.

    Notes
    -----
    Runs with seed=42 for both paths.  Per-trial match is NOT asserted —
    only that all choices fall in {0, 1, 2} and lengths equal n_trials.
    """
    config = load_config()
    trials = generate_session(config, seed=0)
    n_trials = len(trials)
    cue_probs_arr = jnp.array([t.cue_probs for t in trials])

    seed = 42

    # Legacy path
    net = build_3level_network(omega_2=OMEGA_2, omega_3=OMEGA_3, kappa=KAPPA)
    rng = np.random.default_rng(seed)
    result_legacy = simulate_agent(net, trials, beta=BETA, zeta=ZETA, rng=rng)
    choices_legacy = np.array(result_legacy.choices)

    # JAX path
    rng_key = jax.random.PRNGKey(seed)
    choices_jax, _, _ = simulate_session_jax(
        jnp.float32(OMEGA_2),
        jnp.float32(OMEGA_3),
        jnp.float32(KAPPA),
        jnp.float32(BETA),
        jnp.float32(ZETA),
        cue_probs_arr,
        rng_key,
    )
    choices_jax_np = np.array(choices_jax)

    # Both paths must produce choices in {0, 1, 2}
    assert set(choices_legacy).issubset({0, 1, 2}), (
        f"Legacy path produced out-of-range choices: {set(choices_legacy)}"
    )
    assert set(choices_jax_np.tolist()).issubset({0, 1, 2}), (
        f"JAX path produced out-of-range choices: {set(choices_jax_np.tolist())}"
    )

    # Both paths must produce exactly n_trials choices
    assert len(choices_legacy) == n_trials, (
        f"Legacy path: expected {n_trials} choices, got {len(choices_legacy)}"
    )
    assert len(choices_jax_np) == n_trials, (
        f"JAX path: expected {n_trials} choices, got {len(choices_jax_np)}"
    )


# ---------------------------------------------------------------------------
# VALID-04 main — KS-based statistical equivalence over 100 replicates
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_valid04_simulation_equivalence() -> None:
    """VALID-04: KS-based statistical equivalence over 100 replicates.

    Runs both simulation paths with seeds 0..99 and computes the per-cue
    per-phase choice frequency for each replicate.  The KS test is applied to
    the 100-sample frequency distributions; p > 0.05 is required for every
    (phase, cue) pair.

    Notes
    -----
    Using a single fixed trial sequence for all 100 replicates is a deliberate
    controlled-comparison design.  This isolates the effect of agent RNG
    (choice and reward sampling) from environment RNG (trial sequence
    generation).  By holding the environment constant, the KS test compares
    only the behavioral difference between the NumPy and JAX agent paths.
    Varying environment seeds across replicates would conflate environment
    variability with implementation differences, reducing the test's
    sensitivity to detect bias.
    """
    config = load_config()

    # Generate ONE trial sequence shared across all replicates.
    trials = generate_session(config, seed=0)
    cue_probs_arr = jnp.array([t.cue_probs for t in trials])

    # Extract phase labels and build trial-index lookup.
    phase_labels = [t.phase_label for t in trials]
    unique_phases = sorted(set(phase_labels))
    phase_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(phase_labels):
        phase_indices[label].append(idx)

    # Accumulate per-cue per-phase choice frequencies across replicates.
    # legacy_freqs[phase][cue] = list of N_REPLICATES frequency values
    legacy_freqs: dict[str, dict[int, list[float]]] = {
        ph: {c: [] for c in range(N_CUES)} for ph in unique_phases
    }
    jax_freqs: dict[str, dict[int, list[float]]] = {
        ph: {c: [] for c in range(N_CUES)} for ph in unique_phases
    }

    for seed in range(N_REPLICATES):
        # ------------------------------------------------------------------
        # Legacy path (simulate_agent — NumPy for-loop)
        # ------------------------------------------------------------------
        net = build_3level_network(omega_2=OMEGA_2, omega_3=OMEGA_3, kappa=KAPPA)
        rng = np.random.default_rng(seed)
        result = simulate_agent(net, trials, beta=BETA, zeta=ZETA, rng=rng)
        choices_legacy = np.array(result.choices)

        # ------------------------------------------------------------------
        # JAX path (simulate_session_jax — lax.scan)
        # ------------------------------------------------------------------
        rng_key = jax.random.PRNGKey(seed)
        choices_jax, _, _ = simulate_session_jax(
            jnp.float32(OMEGA_2),
            jnp.float32(OMEGA_3),
            jnp.float32(KAPPA),
            jnp.float32(BETA),
            jnp.float32(ZETA),
            cue_probs_arr,
            rng_key,
        )
        choices_jax_np = np.array(choices_jax)

        # ------------------------------------------------------------------
        # Compute per-cue per-phase choice frequencies for this replicate
        # ------------------------------------------------------------------
        for phase_label in unique_phases:
            trial_idxs = phase_indices[phase_label]
            n_phase = len(trial_idxs)
            for cue in range(N_CUES):
                legacy_freq = float(
                    np.sum(choices_legacy[trial_idxs] == cue) / n_phase
                )
                jax_freq = float(
                    np.sum(choices_jax_np[trial_idxs] == cue) / n_phase
                )
                legacy_freqs[phase_label][cue].append(legacy_freq)
                jax_freqs[phase_label][cue].append(jax_freq)

    # ------------------------------------------------------------------
    # KS tests: one per (phase, cue) pair
    # ------------------------------------------------------------------
    results: list[tuple[str, int, float, float]] = []

    for phase_label in unique_phases:
        for cue in range(N_CUES):
            # Sanity check: correct number of replicates accumulated
            assert len(legacy_freqs[phase_label][cue]) == N_REPLICATES, (
                f"Expected {N_REPLICATES} replicates for phase={phase_label!r} "
                f"cue={cue}, got {len(legacy_freqs[phase_label][cue])}"
            )

            leg = np.array(legacy_freqs[phase_label][cue])
            jx = np.array(jax_freqs[phase_label][cue])
            stat, p = ks_2samp(leg, jx)
            results.append((phase_label, cue, float(stat), float(p)))

    # ------------------------------------------------------------------
    # Assert equivalence: all KS p-values > 0.05
    # ------------------------------------------------------------------
    failures = [
        (ph, cue, stat, p) for ph, cue, stat, p in results if p <= 0.05
    ]
    assert len(failures) == 0, (
        f"VALID-04 FAILED: {len(failures)} KS tests had p <= 0.05. "
        f"Failures (phase, cue, KS_stat, p_value): {failures}"
    )
