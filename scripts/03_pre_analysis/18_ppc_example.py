# Demonstration script -- tiny data, runs locally in < 1 min.
"""PPC replay demonstration: simulate cohort, mock posterior, replay.

Simulates a tiny cohort (P=6) with known parameters, constructs a
mock ArviZ InferenceData posterior from the ground-truth values (with
small Gaussian perturbation to mimic posterior uncertainty), and runs
posterior_predictive_replay to produce per-trial P(choice_observed).

This is NOT a real inference pipeline -- the "posterior" is synthetic.
Its purpose is to verify the PPC machinery produces sensible output
and to illustrate the API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import arviz as az

# ---------------------------------------------------------------------------
# Cohort simulation
# ---------------------------------------------------------------------------

P = 6  # participants
N_GROUPS = 2
SEED = 42


def _simulate_tiny_cohort() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
):
    """Simulate a tiny cohort and return (input_data, observed, choices, params).

    Returns
    -------
    input_data_arr : numpy.ndarray, shape (P, n_trials, 3)
    observed_arr : numpy.ndarray, shape (P, n_trials, 3)
    choices_arr : numpy.ndarray, shape (P, n_trials)
    true_params : dict
        Ground-truth individual-level parameter arrays.
    """
    from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort

    group_idx = np.array([0, 0, 0, 1, 1, 1])

    true_mu = {
        "omega_2": np.array([-3.5, -4.0]),
        "log_beta": np.array([0.7, 1.0]),
        "zeta": np.array([0.2, 0.3]),
    }
    true_sigma = {
        "omega_2": 0.3,
        "log_beta": 0.2,
        "zeta": 0.1,
    }

    sim_df, true_params = simulate_hierarchical_cohort(
        n_participants=P,
        n_groups=N_GROUPS,
        true_mu=true_mu,
        true_sigma=true_sigma,
        true_beta=None,
        x_covariate=None,
        group_idx=group_idx,
        model_name="hgf_2level",
        task_config="pick_best_cue",
        seed=SEED,
    )

    # Convert DataFrame to stacked arrays (same logic as fit_batch)
    participant_ids = sim_df["participant_id"].unique().tolist()
    n_trials = sim_df.groupby("participant_id").size().iloc[0]

    input_data_list = []
    observed_list = []
    choices_list = []

    for pid in participant_ids:
        subset = sim_df[sim_df["participant_id"] == pid].sort_values("trial")
        choices = subset["cue_chosen"].to_numpy(dtype=int)
        rewards = subset["reward"].to_numpy(dtype=float)

        inp = np.zeros((n_trials, 3), dtype=float)
        obs = np.zeros((n_trials, 3), dtype=int)
        for t in range(n_trials):
            cue = choices[t]
            inp[t, cue] = rewards[t]
            obs[t, cue] = 1

        input_data_list.append(inp)
        observed_list.append(obs)
        choices_list.append(choices)

    input_data_arr = np.stack(input_data_list, axis=0)
    observed_arr = np.stack(observed_list, axis=0)
    choices_arr = np.stack(choices_list, axis=0)

    return input_data_arr, observed_arr, choices_arr, true_params


# ---------------------------------------------------------------------------
# Mock InferenceData
# ---------------------------------------------------------------------------

N_CHAINS = 2
N_DRAWS = 50


def _build_mock_idata(
    true_params: dict,
) -> az.InferenceData:
    """Build a mock InferenceData from ground-truth + Gaussian noise.

    Parameters
    ----------
    true_params : dict
        Must contain ``"omega_2"``, ``"beta"``, ``"zeta"`` arrays of
        shape ``(P,)``.

    Returns
    -------
    arviz.InferenceData
        Mock posterior with shape ``(chain, draw, participant)``.
    """
    import arviz as az

    rng = np.random.default_rng(123)

    omega_2_true = true_params["omega_2"]
    beta_true = true_params["beta"]
    zeta_true = true_params["zeta"]

    # Shape: (n_chains, n_draws, P)
    posterior_dict = {
        "omega_2": (
            omega_2_true[None, None, :]
            + rng.normal(0, 0.1, (N_CHAINS, N_DRAWS, P))
        ),
        "beta": np.maximum(
            0.1,
            beta_true[None, None, :]
            + rng.normal(0, 0.1, (N_CHAINS, N_DRAWS, P)),
        ),
        "zeta": (
            zeta_true[None, None, :]
            + rng.normal(0, 0.05, (N_CHAINS, N_DRAWS, P))
        ),
    }

    participant_ids = [f"P{k:04d}" for k in range(P)]
    dims_dict = {var: ["participant"] for var in posterior_dict}
    coords_dict = {"participant": participant_ids}

    return az.from_dict(
        posterior=posterior_dict,
        dims=dims_dict,
        coords=coords_dict,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the PPC demonstration."""
    from prl_hgf.simulation.ppc import posterior_predictive_replay

    print("Simulating tiny cohort (P=6, 2-level model)...")
    input_data_arr, observed_arr, choices_arr, true_params = (
        _simulate_tiny_cohort()
    )
    n_trials = input_data_arr.shape[1]
    print(f"  Cohort shape: P={P}, n_trials={n_trials}")

    print("Building mock InferenceData...")
    idata = _build_mock_idata(true_params)
    print(
        f"  Posterior shape: "
        f"chains={N_CHAINS}, draws={N_DRAWS}, P={P}"
    )

    print("Running posterior predictive replay...")
    p_choice = posterior_predictive_replay(
        idata=idata,
        input_data_arr=input_data_arr,
        observed_arr=observed_arr,
        choices_arr=choices_arr,
        model_name="hgf_2level",
        n_draws=20,
    )

    print(f"\nResult shape: {p_choice.shape}")
    print(f"  Expected: (n_draws_used=20, P={P}, n_trials={n_trials})")

    # Summary statistics (ignore NaN from masked trials)
    mean_p = np.nanmean(p_choice)
    median_p = np.nanmedian(p_choice)
    std_p = np.nanstd(p_choice)
    frac_above_chance = np.nanmean(p_choice > 1.0 / 3.0)

    print("\nPPC summary (across all draws, participants, trials):")
    print(f"  Mean P(choice_obs):     {mean_p:.4f}")
    print(f"  Median P(choice_obs):   {median_p:.4f}")
    print(f"  Std P(choice_obs):      {std_p:.4f}")
    print(f"  Frac above chance:      {frac_above_chance:.4f}")

    # Per-participant mean
    per_participant = np.nanmean(p_choice, axis=(0, 2))
    print("\nPer-participant mean P(choice_obs):")
    for k in range(P):
        print(f"  P{k:04d}: {per_participant[k]:.4f}")

    print("\nPPC replay demonstration complete.")


if __name__ == "__main__":
    main()
