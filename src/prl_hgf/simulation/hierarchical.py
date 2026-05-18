"""Hierarchical cohort simulator for Mode B recovery experiments.

Generates synthetic cohorts from a hierarchical generative model:

    theta_{k,p} = mu_{g(k),p} + beta_p * (x_k - mean(x)) + sigma_p * eps_k

where ``g(k)`` maps participant k to group index, ``x_k`` is an optional
continuous covariate, and ``eps_k ~ N(0, 1)``.

The output DataFrame matches the format expected by
:func:`~prl_hgf.fitting.hierarchical.fit_batch_hierarchical` (columns:
``participant_id``, ``group``, ``session``, ``cue_chosen``, ``reward``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import load_config
from prl_hgf.simulation.agent import PARAM_BOUNDS
from prl_hgf.simulation.jax_session import _build_session_scanner, _run_session

__all__ = ["simulate_hierarchical_cohort"]

#: Fixed omega_3 for 2-level model (effectively disables volatility coupling).
_OMEGA3_FIXED_2LEVEL: float = -6.0

#: Fixed kappa value (matches _KAPPA_FIXED in hierarchical.py).
_KAPPA_FIXED: float = 1.0


def simulate_hierarchical_cohort(
    n_participants: int,
    n_groups: int,
    true_mu: dict[str, np.ndarray],
    true_sigma: dict[str, float],
    true_beta: dict[str, float] | None,
    x_covariate: np.ndarray | None,
    group_idx: np.ndarray,
    model_name: str = "hgf_2level",
    task_config: str = "pick_best_cue",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a cohort from a hierarchical generative model.

    Draws individual parameters from a hierarchical distribution, runs each
    participant-session through the JAX-native HGF simulator, and assembles
    a tidy DataFrame suitable for ``fit_batch_hierarchical``.

    Parameters
    ----------
    n_participants : int
        Total number of participant-sessions ``P``.
    n_groups : int
        Number of experimental groups ``G``.
    true_mu : dict[str, numpy.ndarray]
        Group-level means for each hierarchical parameter.  Keys are
        parameter names (e.g. ``"omega_2"``, ``"log_beta"``, ``"zeta"``);
        values are arrays of shape ``(n_groups,)``.
    true_sigma : dict[str, float]
        Shared (across groups) standard deviation for each parameter.
    true_beta : dict[str, float] or None
        Covariate slope for each parameter, or ``None`` if no covariate.
    x_covariate : numpy.ndarray or None
        Continuous covariate of shape ``(P,)``, or ``None``.
    group_idx : numpy.ndarray
        Integer group assignment of shape ``(P,)`` with values in
        ``[0, n_groups)``.
    model_name : str, optional
        Model variant: ``"hgf_2level"`` (default) or ``"hgf_3level"``.
    task_config : str, optional
        Task configuration name.  Currently only ``"pick_best_cue"``
        is supported (default).
    seed : int, optional
        Master RNG seed for reproducibility.

    Returns
    -------
    sim_df : pandas.DataFrame
        Trial-level DataFrame with columns ``participant_id``, ``group``,
        ``session``, ``trial``, ``cue_chosen``, ``reward``, plus
        ``true_*`` columns for each parameter.
    true_params : dict
        Dictionary with keys ``"omega_2"``, ``"log_beta"``, ``"beta"``,
        ``"zeta"``, ``"kappa"``, ``"omega_3"`` (and ``"x_covariate"`` if
        provided), each mapping to a numpy array of shape ``(P,)``
        containing the individual-level ground-truth parameter values.

    Raises
    ------
    ValueError
        If ``group_idx`` shape does not match ``n_participants``, or if
        ``true_mu`` keys do not match ``true_sigma`` keys.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if group_idx.shape != (n_participants,):
        raise ValueError(
            f"group_idx shape mismatch: expected ({n_participants},), "
            f"got {group_idx.shape}"
        )
    if set(true_mu.keys()) != set(true_sigma.keys()):
        raise ValueError(
            f"true_mu and true_sigma must have the same keys. "
            f"Got true_mu={sorted(true_mu.keys())}, "
            f"true_sigma={sorted(true_sigma.keys())}"
        )
    if true_beta is not None and x_covariate is None:
        raise ValueError("true_beta provided but x_covariate is None")
    if x_covariate is not None and x_covariate.shape != (n_participants,):
        raise ValueError(
            f"x_covariate shape mismatch: expected ({n_participants},), "
            f"got {x_covariate.shape}"
        )

    # Determine hierarchical param names for the model
    if model_name == "hgf_3level":
        h_params = ("omega_2", "log_beta", "zeta", "omega_3")
    else:
        h_params = ("omega_2", "log_beta", "zeta")

    for p_name in h_params:
        if p_name not in true_mu:
            raise ValueError(
                f"true_mu missing required parameter '{p_name}' for "
                f"model_name={model_name!r}"
            )

    # ------------------------------------------------------------------
    # Draw individual parameters from hierarchical distribution
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # Mean-center covariate if provided
    if x_covariate is not None:
        x_centered = x_covariate - np.mean(x_covariate)
    else:
        x_centered = None

    individual_params: dict[str, np.ndarray] = {}
    for p_name in h_params:
        mu_g = true_mu[p_name]  # shape (n_groups,)
        sigma = true_sigma[p_name]
        eps = rng.standard_normal(n_participants)

        # Group-level mean for each participant
        mean_p = mu_g[group_idx]

        # Add covariate effect
        if true_beta is not None and p_name in true_beta and x_centered is not None:
            mean_p = mean_p + true_beta[p_name] * x_centered

        # Draw from Normal(mean_p, sigma)
        individual_params[p_name] = mean_p + sigma * eps

    # Clip to PARAM_BOUNDS
    if "omega_2" in individual_params:
        individual_params["omega_2"] = np.clip(
            individual_params["omega_2"], *PARAM_BOUNDS["omega_2"]
        )
    if "omega_3" in individual_params:
        individual_params["omega_3"] = np.clip(
            individual_params["omega_3"], *PARAM_BOUNDS["omega_3"]
        )
    if "zeta" in individual_params:
        individual_params["zeta"] = np.clip(
            individual_params["zeta"], *PARAM_BOUNDS["zeta"]
        )

    # log_beta -> beta via exp, then clip beta
    log_beta_arr = individual_params["log_beta"]
    beta_arr = np.exp(log_beta_arr)
    beta_arr = np.clip(beta_arr, *PARAM_BOUNDS["beta"])
    individual_params["beta"] = beta_arr

    # Fixed parameters
    kappa_arr = np.full(n_participants, _KAPPA_FIXED)
    individual_params["kappa"] = kappa_arr

    if model_name == "hgf_2level":
        omega_3_arr = np.full(n_participants, _OMEGA3_FIXED_2LEVEL)
        individual_params["omega_3"] = omega_3_arr

    # ------------------------------------------------------------------
    # Load task config and generate trial sequences
    # ------------------------------------------------------------------
    config = load_config()

    # Derive per-participant seeds for env and simulation
    all_seeds = rng.integers(0, 2**31, size=(n_participants, 2))

    # Collect trial sequences and cue_probs for each participant
    all_cue_probs: list[jnp.ndarray] = []
    all_rng_keys: list[jnp.ndarray] = []
    all_trials: list[list] = []

    for k in range(n_participants):
        env_seed = int(all_seeds[k, 0])
        sim_seed = int(all_seeds[k, 1])

        trials = generate_session(config, env_seed)
        cue_probs = jnp.array([t.cue_probs for t in trials], dtype=jnp.float32)

        all_cue_probs.append(cue_probs)
        all_rng_keys.append(jax.random.PRNGKey(sim_seed))
        all_trials.append(trials)

    # ------------------------------------------------------------------
    # Stack into batch arrays and run vmapped simulation
    # ------------------------------------------------------------------
    params_batch = {
        "omega_2": jnp.array(individual_params["omega_2"]),
        "omega_3": jnp.array(individual_params["omega_3"]),
        "kappa": jnp.array(individual_params["kappa"]),
        "beta": jnp.array(individual_params["beta"]),
        "zeta": jnp.array(individual_params["zeta"]),
    }
    cue_probs_batch = jnp.stack(all_cue_probs)
    rng_keys_batch = jnp.stack(all_rng_keys)

    scan_fn, base_attrs = _build_session_scanner()
    _vmapped = jax.vmap(
        lambda o2, o3, k, b, z, cp, rk: _run_session(
            scan_fn, base_attrs, o2, o3, k, b, z, cp, rk
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
    all_choices_batch, all_rewards_batch, all_diverged_batch = _vmapped(
        params_batch["omega_2"],
        params_batch["omega_3"],
        params_batch["kappa"],
        params_batch["beta"],
        params_batch["zeta"],
        cue_probs_batch,
        rng_keys_batch,
    )

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    group_labels = [f"group_{g}" for g in range(n_groups)]
    rows: list[dict] = []

    for k in range(n_participants):
        pid = f"P{k:04d}"
        grp = group_labels[group_idx[k]]
        trials = all_trials[k]
        choices_list = [int(c) for c in all_choices_batch[k]]
        rewards_list = [int(r) for r in all_rewards_batch[k]]

        for t_idx, trial in enumerate(trials):
            row: dict = {
                "participant_id": pid,
                "group": grp,
                "session": "baseline",
                "trial": trial.trial_idx,
                "cue_chosen": choices_list[t_idx],
                "reward": rewards_list[t_idx],
                "true_omega_2": float(individual_params["omega_2"][k]),
                "true_log_beta": float(individual_params["log_beta"][k]),
                "true_beta": float(individual_params["beta"][k]),
                "true_zeta": float(individual_params["zeta"][k]),
                "true_kappa": float(individual_params["kappa"][k]),
                "true_omega_3": float(individual_params["omega_3"][k]),
                "diverged": bool(all_diverged_batch[k]),
            }
            rows.append(row)

    sim_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Build true_params dict for recovery analysis
    # ------------------------------------------------------------------
    true_params: dict[str, np.ndarray] = {
        "omega_2": individual_params["omega_2"].copy(),
        "log_beta": individual_params["log_beta"].copy(),
        "beta": individual_params["beta"].copy(),
        "zeta": individual_params["zeta"].copy(),
        "kappa": individual_params["kappa"].copy(),
        "omega_3": individual_params["omega_3"].copy(),
        "group_idx": group_idx.copy(),
    }
    if x_covariate is not None:
        true_params["x_covariate"] = x_covariate.copy()

    return sim_df, true_params
