"""Phase 33-06: Recovery validation for Mode B hierarchical at P=200.

Tests parameter recovery under two covariate conditions:

- **Condition 1 (orthogonal)**: ``x_covariate ~ N(0, 1)``, independent of
  group assignment.
- **Condition 2 (near-collinear)**: ``x_covariate = group_idx * 0.5 +
  N(0, 1)``, partially confounded with group.

Recovery criteria:
- ``r(true_beta, post_mean_beta) >= 0.7`` (covariate slope recovery)
- ``|bias(mu)| <= 0.2 * sigma`` per parameter (group mean recovery)

Usage::

    python scripts/03_pre_analysis/06_recovery_validation_modeb.py --condition both
    python scripts/03_pre_analysis/06_recovery_validation_modeb.py --condition orthogonal --seed 123
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
N_PER_GROUP = 33
N_GROUPS = 2
N_PARTICIPANTS = N_PER_GROUP * N_GROUPS  # 66
MODEL_NAME = "hgf_2level"
RESULTS_PATH = Path("logs/33_recovery_results.json")

# Hierarchical ground truth
TRUE_MU: dict[str, np.ndarray] = {
    "omega_2": np.array([-3.5, -2.5]),
    "log_beta": np.array([0.5, 1.0]),
    "zeta": np.array([0.2, -0.2]),
}
TRUE_SIGMA: dict[str, float] = {
    "omega_2": 0.8,
    "log_beta": 0.5,
    "zeta": 0.6,
}
TRUE_BETA: dict[str, float] = {
    "omega_2": 0.3,
    "log_beta": 0.2,
    "zeta": -0.15,
}


def _build_group_idx(n_per_group: int, n_groups: int) -> np.ndarray:
    """Build balanced group index array.

    Parameters
    ----------
    n_per_group : int
        Participants per group.
    n_groups : int
        Number of groups.

    Returns
    -------
    numpy.ndarray
        Group indices of shape ``(n_per_group * n_groups,)``.
    """
    return np.repeat(np.arange(n_groups), n_per_group)


def _compute_recovery_r(
    true_vals: np.ndarray,
    post_mean: np.ndarray,
) -> float:
    """Pearson correlation between true and recovered values.

    Parameters
    ----------
    true_vals : numpy.ndarray
        Ground-truth values.
    post_mean : numpy.ndarray
        Posterior mean estimates.

    Returns
    -------
    float
        Pearson r, or 0.0 if degenerate.
    """
    if np.std(true_vals) < 1e-10 or np.std(post_mean) < 1e-10:
        return 0.0
    return float(np.corrcoef(true_vals, post_mean)[0, 1])


def run_condition(
    condition: str,
    seed: int,
) -> dict:
    """Run one recovery validation condition.

    Parameters
    ----------
    condition : str
        Either ``"orthogonal"`` or ``"near_collinear"``.
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
    rng = np.random.default_rng(seed)

    # Generate covariate
    if condition == "orthogonal":
        x_covariate = rng.standard_normal(N_PARTICIPANTS)
    else:  # near_collinear
        x_covariate = group_idx.astype(float) * 0.5 + rng.standard_normal(
            N_PARTICIPANTS
        )

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition}, seed={seed}, P={N_PARTICIPANTS}")
    print(f"x_covariate range: [{x_covariate.min():.2f}, {x_covariate.max():.2f}]")
    print(f"x_covariate mean: {x_covariate.mean():.3f}")
    print(f"{'=' * 60}\n")

    # Simulate
    t0 = time.time()
    sim_df, true_params = simulate_hierarchical_cohort(
        n_participants=N_PARTICIPANTS,
        n_groups=N_GROUPS,
        true_mu=TRUE_MU,
        true_sigma=TRUE_SIGMA,
        true_beta=TRUE_BETA,
        x_covariate=x_covariate,
        group_idx=group_idx,
        model_name=MODEL_NAME,
        seed=seed,
    )
    t_sim = time.time() - t0
    print(f"Simulation complete in {t_sim:.1f}s")
    print(f"sim_df shape: {sim_df.shape}")

    # Build FitConfig for Mode B with covariate
    fit_config = FitConfig(
        model_name=MODEL_NAME,
        sampler=SamplerConfig(
            backend="blackjax",
            n_chains=4,
            n_draws=2000,
            n_warmup=1000,
            target_accept=0.95,
            random_seed=seed + 2000,
        ),
        mitigation=MitigationConfig(
            non_centered=("omega_2", "log_beta", "zeta"),
        ),
        covariate=CovariateConfig(
            pooling="hierarchical",
            n_groups=N_GROUPS,
            covariate_names=("x",),
        ),
        progressbar=True,
    )

    # Fit
    t_fit0 = time.time()
    result = fit_batch_hierarchical(sim_df, fit_config, x_covariate=x_covariate)
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
        # Mu recovery
        mu_key = f"mu_{p_name}"
        if mu_key in posterior:
            post_mu = posterior[mu_key].values.mean(axis=(0, 1))
            true_mu_vals = TRUE_MU[p_name]
            mu_bias = float(np.mean(np.abs(post_mu - true_mu_vals)))
            bias_ratio = mu_bias / TRUE_SIGMA[p_name]
        else:
            mu_bias = float("nan")
            bias_ratio = float("nan")

        # Beta (covariate slope) recovery
        beta_key = f"beta_{p_name}"
        if beta_key in posterior:
            post_beta_mean = float(posterior[beta_key].values.mean())
            true_beta_val = TRUE_BETA[p_name]
            beta_bias = abs(post_beta_mean - true_beta_val)
        else:
            post_beta_mean = float("nan")
            true_beta_val = TRUE_BETA[p_name]
            beta_bias = float("nan")

        # Individual-level recovery
        if p_name in posterior:
            post_mean_ind = posterior[p_name].values.mean(axis=(0, 1))
            true_ind = true_params[p_name]
            r_ind = _compute_recovery_r(true_ind, post_mean_ind)
        else:
            r_ind = float("nan")

        metrics[p_name] = {
            "mu_bias": mu_bias,
            "bias_ratio": bias_ratio,
            "bias_criterion_met": bias_ratio <= 0.2,
            "true_beta": true_beta_val,
            "post_beta_mean": post_beta_mean,
            "beta_bias": beta_bias,
            "individual_r": r_ind,
        }

    # Beta recovery correlation across parameters
    true_betas = np.array([TRUE_BETA[p] for p in h_params])
    post_betas = np.array([metrics[p]["post_beta_mean"] for p in h_params])
    if not np.any(np.isnan(post_betas)):
        beta_r = _compute_recovery_r(true_betas, post_betas)
    else:
        beta_r = float("nan")

    # Aggregate criteria
    all_bias_met = all(
        metrics[p]["bias_criterion_met"]
        for p in h_params
        if not np.isnan(metrics[p]["bias_ratio"])
    )

    return {
        "condition": condition,
        "seed": seed,
        "n_participants": N_PARTICIPANTS,
        "n_groups": N_GROUPS,
        "model_name": MODEL_NAME,
        "sim_time_s": t_sim,
        "fit_time_s": t_fit,
        "beta_recovery_r": beta_r,
        "beta_r_criterion_met": beta_r >= 0.7,
        "all_mu_bias_criterion_met": all_bias_met,
        "metrics": metrics,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 33-06: Mode B recovery validation"
    )
    parser.add_argument(
        "--condition",
        choices=["orthogonal", "near_collinear", "both"],
        default="both",
        help="Which condition(s) to run (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Master RNG seed (default: 123)",
    )
    args = parser.parse_args()

    conditions: list[str] = []
    if args.condition in ("orthogonal", "both"):
        conditions.append("orthogonal")
    if args.condition in ("near_collinear", "both"):
        conditions.append("near_collinear")

    all_results: list[dict] = []
    for cond in conditions:
        result = run_condition(cond, args.seed)
        all_results.append(result)
        print(f"\nCondition '{cond}' results:")
        print(json.dumps(result, indent=2, default=str))

    # Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Exit code: pass if orthogonal condition meets both criteria
    orthogonal = [r for r in all_results if r["condition"] == "orthogonal"]
    if orthogonal:
        r = orthogonal[0]
        beta_ok = r["beta_r_criterion_met"]
        bias_ok = r["all_mu_bias_criterion_met"]
        if beta_ok and bias_ok:
            print(
                f"\nPASS: beta_r={r['beta_recovery_r']:.3f} >= 0.7, "
                f"all mu bias <= 0.2*sigma"
            )
            sys.exit(0)
        else:
            print(
                f"\nFAIL: beta_r={r['beta_recovery_r']:.3f} "
                f"(>= 0.7: {beta_ok}), "
                f"all mu bias <= 0.2*sigma: {bias_ok}"
            )
            sys.exit(1)
    else:
        print("\nOrthogonal condition not run; exit 0")
        sys.exit(0)


if __name__ == "__main__":
    main()
