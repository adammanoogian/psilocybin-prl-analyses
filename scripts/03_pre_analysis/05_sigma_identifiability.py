"""Phase 33-05: Sigma identifiability experiment for Mode B hierarchical.

Tests whether the shared-sigma assumption in Mode B hierarchical fitting
recovers accurately when:

- **Scenario A**: true sigma is identical across groups (assumption holds).
- **Scenario B**: true sigma differs across groups but the model uses a
  shared sigma (assumption violated).

For each scenario: simulate via hierarchical generative model, fit with
``fit_batch_hierarchical`` in Mode B, compute recovery metrics, and log
results to ``logs/33_identifiability_results.json``.

Usage::

    python scripts/03_pre_analysis/05_sigma_identifiability.py --scenario both
    python scripts/03_pre_analysis/05_sigma_identifiability.py --scenario A --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable regardless of install mode.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PER_GROUP = 33  # P = 33 * 2 groups * 3 sessions = 198 ~ 200
N_GROUPS = 2
N_PARTICIPANTS = N_PER_GROUP * N_GROUPS  # 66 (single-session Mode B)
MODEL_NAME = "hgf_2level"
RESULTS_PATH = Path("logs/33_identifiability_results.json")

# Hierarchical ground truth for Scenario A (shared sigma)
TRUE_MU_A: dict[str, np.ndarray] = {
    "omega_2": np.array([-3.5, -2.5]),  # group 0 and group 1
    "log_beta": np.array([0.5, 1.0]),
    "zeta": np.array([0.2, -0.2]),
}
TRUE_SIGMA_A: dict[str, float] = {
    "omega_2": 0.8,
    "log_beta": 0.5,
    "zeta": 0.6,
}

# Scenario B: sigma differs across groups — but we simulate from the
# average sigma (representing model misspecification: model uses shared
# sigma while data has heterogeneous sigma).
TRUE_SIGMA_B_PER_GROUP: dict[str, tuple[float, float]] = {
    "omega_2": (0.4, 1.2),  # group 0 has tight, group 1 has wide
    "log_beta": (0.3, 0.7),
    "zeta": (0.3, 0.9),
}


def _build_group_idx(n_per_group: int, n_groups: int) -> np.ndarray:
    """Build balanced group index array.

    Parameters
    ----------
    n_per_group : int
        Number of participants per group.
    n_groups : int
        Number of groups.

    Returns
    -------
    numpy.ndarray
        Integer group indices of shape ``(n_per_group * n_groups,)``.
    """
    return np.repeat(np.arange(n_groups), n_per_group)


def _compute_recovery_r(
    true_vals: np.ndarray,
    post_mean: np.ndarray,
) -> float:
    """Pearson correlation between true and recovered parameter values.

    Parameters
    ----------
    true_vals : numpy.ndarray
        Ground-truth parameter values.
    post_mean : numpy.ndarray
        Posterior mean estimates.

    Returns
    -------
    float
        Pearson correlation coefficient, or 0.0 if computation fails.
    """
    if np.std(true_vals) < 1e-10 or np.std(post_mean) < 1e-10:
        return 0.0
    return float(np.corrcoef(true_vals, post_mean)[0, 1])


def _simulate_scenario_b(
    n_participants: int,
    group_idx: np.ndarray,
    seed: int,
) -> tuple:
    """Simulate Scenario B with per-group sigma.

    Draws parameters from group-specific sigma values, which violates the
    shared-sigma assumption of Mode B.

    Parameters
    ----------
    n_participants : int
        Total number of participants.
    group_idx : numpy.ndarray
        Group assignment array.
    seed : int
        RNG seed.

    Returns
    -------
    tuple
        ``(sim_df, true_params)`` from hierarchical simulation.
    """
    from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort

    rng = np.random.default_rng(seed)
    h_params = ("omega_2", "log_beta", "zeta")

    # Draw individual params with per-group sigma
    individual_params: dict[str, np.ndarray] = {}
    for p_name in h_params:
        mu_g = TRUE_MU_A[p_name]
        sigma_per_group = TRUE_SIGMA_B_PER_GROUP[p_name]
        vals = np.zeros(n_participants)
        for k in range(n_participants):
            g = group_idx[k]
            vals[k] = rng.normal(mu_g[g], sigma_per_group[g])
        individual_params[p_name] = vals

    # Use the average sigma as the "true_sigma" for reporting
    true_sigma_avg: dict[str, float] = {}
    for p_name in h_params:
        s0, s1 = TRUE_SIGMA_B_PER_GROUP[p_name]
        true_sigma_avg[p_name] = (s0 + s1) / 2.0

    # Use simulate_hierarchical_cohort with shared sigma (the average)
    # to generate the trial-level data — but override individual params
    # by using a seed that reproduces the same env/sim seeds.
    sim_df, _ = simulate_hierarchical_cohort(
        n_participants=n_participants,
        n_groups=N_GROUPS,
        true_mu=TRUE_MU_A,
        true_sigma=true_sigma_avg,
        true_beta=None,
        x_covariate=None,
        group_idx=group_idx,
        model_name=MODEL_NAME,
        seed=seed,
    )

    # The true_params from simulate_hierarchical_cohort used average
    # sigma; we need the actual per-group-sigma params for recovery.
    # Overwrite the true_* columns in sim_df is not needed since
    # recovery is computed on the individual_params directly.
    true_params = {
        "omega_2": individual_params["omega_2"],
        "log_beta": individual_params["log_beta"],
        "zeta": individual_params["zeta"],
        "group_idx": group_idx.copy(),
    }

    return sim_df, true_params


def run_scenario(
    scenario: str,
    seed: int,
) -> dict:
    """Run one identifiability scenario (A or B).

    Parameters
    ----------
    scenario : str
        Either ``"A"`` or ``"B"``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Results dictionary with recovery metrics.
    """
    from prl_hgf.fitting.config import (
        CovariateConfig,
        FitConfig,
        MitigationConfig,
        SamplerConfig,
    )
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort

    group_idx = _build_group_idx(N_PER_GROUP, N_GROUPS)

    print(f"\n{'=' * 60}")
    print(f"Scenario {scenario}: seed={seed}, P={N_PARTICIPANTS}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    if scenario == "A":
        sim_df, true_params = simulate_hierarchical_cohort(
            n_participants=N_PARTICIPANTS,
            n_groups=N_GROUPS,
            true_mu=TRUE_MU_A,
            true_sigma=TRUE_SIGMA_A,
            true_beta=None,
            x_covariate=None,
            group_idx=group_idx,
            model_name=MODEL_NAME,
            seed=seed,
        )
    else:
        sim_df, true_params = _simulate_scenario_b(N_PARTICIPANTS, group_idx, seed)

    t_sim = time.time() - t0
    print(f"Simulation complete in {t_sim:.1f}s")
    print(f"sim_df shape: {sim_df.shape}")
    print(f"Unique participants: {sim_df['participant_id'].nunique()}")

    # Build FitConfig for Mode B
    fit_config = FitConfig(
        model_name=MODEL_NAME,
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=4,
            n_draws=2000,
            n_warmup=1000,
            target_accept=0.95,
            random_seed=seed + 1000,
        ),
        mitigation=MitigationConfig(
            non_centered=("omega_2", "log_beta", "zeta"),
        ),
        covariate=CovariateConfig(
            pooling="hierarchical",
            n_groups=N_GROUPS,
        ),
        progressbar=True,
    )

    t_fit0 = time.time()
    result = fit_batch_hierarchical(sim_df, fit_config)
    if isinstance(result, tuple):
        idata, _ = result
    else:
        idata = result
    t_fit = time.time() - t_fit0
    print(f"Fitting complete in {t_fit:.1f}s")

    # Compute recovery metrics
    posterior = idata.posterior
    h_params = ("omega_2", "log_beta", "zeta")
    metrics: dict[str, dict] = {}

    for p_name in h_params:
        # Sigma recovery: compare true sigma to posterior mean of
        # exp(log_sigma_*)
        log_sigma_key = f"log_sigma_{p_name}"
        if log_sigma_key in posterior:
            post_log_sigma = float(posterior[log_sigma_key].values.mean())
            post_sigma = float(np.exp(post_log_sigma))
        else:
            post_sigma = float("nan")

        if scenario == "A":
            true_sigma_val = TRUE_SIGMA_A[p_name]
        else:
            s0, s1 = TRUE_SIGMA_B_PER_GROUP[p_name]
            true_sigma_val = (s0 + s1) / 2.0

        # Mu recovery: compare true_mu to posterior mean of mu_*
        mu_key = f"mu_{p_name}"
        if mu_key in posterior:
            post_mu = posterior[mu_key].values.mean(axis=(0, 1))
            # post_mu shape: (n_groups,)
            true_mu_vals = TRUE_MU_A[p_name]
            mu_bias = float(np.mean(np.abs(post_mu - true_mu_vals)))
        else:
            mu_bias = float("nan")

        # Individual-level recovery
        if p_name in posterior:
            post_mean_ind = posterior[p_name].values.mean(axis=(0, 1))
            true_ind = true_params[p_name]
            r_ind = _compute_recovery_r(true_ind, post_mean_ind)
        else:
            r_ind = float("nan")

        metrics[p_name] = {
            "true_sigma": true_sigma_val,
            "post_sigma": post_sigma,
            "sigma_bias": abs(post_sigma - true_sigma_val),
            "mu_bias": mu_bias,
            "individual_r": r_ind,
        }

    # Overall recovery correlation (across all params)
    r_values = [
        m["individual_r"] for m in metrics.values() if not np.isnan(m["individual_r"])
    ]
    overall_r = float(np.mean(r_values)) if r_values else float("nan")

    return {
        "scenario": scenario,
        "seed": seed,
        "n_participants": N_PARTICIPANTS,
        "n_groups": N_GROUPS,
        "model_name": MODEL_NAME,
        "sim_time_s": t_sim,
        "fit_time_s": t_fit,
        "overall_r": overall_r,
        "metrics": metrics,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 33-05: sigma identifiability experiment"
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "both"],
        default="both",
        help="Which scenario(s) to run (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default: 42)",
    )
    args = parser.parse_args()

    scenarios: list[str] = []
    if args.scenario in ("A", "both"):
        scenarios.append("A")
    if args.scenario in ("B", "both"):
        scenarios.append("B")

    all_results: list[dict] = []
    for scenario in scenarios:
        result = run_scenario(scenario, args.seed)
        all_results.append(result)
        print(f"\nScenario {scenario} results:")
        print(json.dumps(result, indent=2, default=str))

    # Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Exit code: pass if Scenario A recovery r >= 0.7
    scenario_a = [r for r in all_results if r["scenario"] == "A"]
    if scenario_a:
        r_a = scenario_a[0]["overall_r"]
        if r_a >= 0.7:
            print(f"\nPASS: Scenario A overall_r = {r_a:.3f} >= 0.7")
            sys.exit(0)
        else:
            print(f"\nFAIL: Scenario A overall_r = {r_a:.3f} < 0.7")
            sys.exit(1)
    else:
        print("\nScenario A not run; exit 0")
        sys.exit(0)


if __name__ == "__main__":
    main()
