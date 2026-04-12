"""Batch simulation orchestrator for the PRL pick_best_cue pipeline.

Generates the full synthetic dataset: 2 groups × N participants × 3 sessions
per the study design.  Each participant-session is simulated by the JAX-native
:func:`~prl_hgf.simulation.jax_session._run_session` kernel, vmapped over the
full cohort in a single compiled XLA call.  Results are assembled into a tidy
DataFrame.

The output has one row per trial and includes ground-truth parameters as
``true_*`` columns, making it the direct input for Phase 4 (fitting) and
Phase 5 (parameter recovery).

Seed strategy
-------------
A master RNG derived from ``config.simulation.master_seed`` draws all
per-(participant, session) seeds upfront before any simulation begins.  This
ensures that changing ``n_participants_per_group`` does not alter seeds for
earlier participants.

JAX-native cohort path
-----------------------
The batch loop first collects trial sequences and sampled parameters for ALL
participant-sessions into batch arrays, then dispatches a single vmapped
``_run_session`` call across the entire cohort.  This runs the full cohort in
compiled JAX kernels rather than sequential NumPy loops.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import AnalysisConfig
from prl_hgf.simulation.agent import sample_participant_params
from prl_hgf.simulation.jax_session import _build_session_scanner, _run_session

__all__ = ["simulate_batch"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_batch(
    config: AnalysisConfig,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Simulate a full cohort of synthetic participants across all sessions.

    Uses the JAX-native cohort path: collects all participant-session trial
    sequences and parameters upfront, then dispatches a single vmapped
    ``_run_session`` call over the entire cohort for compiled XLA execution.

    Parameters
    ----------
    config : AnalysisConfig
        Validated analysis configuration loaded via
        :func:`~prl_hgf.env.task_config.load_config`.  The following fields
        are used:

        * ``config.simulation.master_seed`` — for reproducible seed derivation
        * ``config.simulation.n_participants_per_group`` — cohort size
        * ``config.simulation.groups`` — group parameter distributions
        * ``config.simulation.session_deltas`` — per-group session shifts

    output_path : Path or None, optional
        If provided, the resulting DataFrame is saved as CSV at this path.
        The parent directory must exist.

    Returns
    -------
    pandas.DataFrame
        Tidy trial-level DataFrame with one row per trial across all
        participant-sessions.  Columns:

        ``participant_id``, ``group``, ``session``, ``session_idx``,
        ``trial``, ``cue_chosen``, ``reward``,
        ``cue_0_prob``, ``cue_1_prob``, ``cue_2_prob``,
        ``phase_label``, ``phase_name``, ``best_cue``,
        ``true_omega_2``, ``true_omega_3``, ``true_kappa``,
        ``true_beta``, ``true_zeta``, ``model``, ``diverged``

    Notes
    -----
    Identical ``config.simulation.master_seed`` values always produce
    identical output — seed derivation is deterministic and the per-session
    seeds are drawn upfront before any simulation begins.

    Examples
    --------
    >>> from prl_hgf.env.task_config import load_config
    >>> from prl_hgf.simulation.batch import simulate_batch
    >>> config = load_config()
    >>> df = simulate_batch(config)
    >>> df.shape[1]  # columns
    21
    """
    sim_cfg = config.simulation
    n_per_group: int = sim_cfg.n_participants_per_group
    group_names: list[str] = sorted(sim_cfg.groups.keys())
    n_groups: int = len(group_names)
    n_sessions: int = 3
    n_total: int = n_groups * n_per_group * n_sessions

    # --- Derive all (env_seed, sim_seed) pairs upfront ---
    rng_master = np.random.default_rng(sim_cfg.master_seed)
    all_seeds = rng_master.integers(0, 2**31, size=(n_total, 2))

    # --- First pass: collect params, trial sequences, and metadata ---
    print("Collecting participant-session parameters and trial sequences…")
    t_start = time.time()

    all_omega_2: list[float] = []
    all_omega_3: list[float] = []
    all_kappa: list[float] = []
    all_beta: list[float] = []
    all_zeta: list[float] = []
    all_cue_probs: list[jnp.ndarray] = []
    all_rng_keys: list[jnp.ndarray] = []
    all_trials: list[list] = []
    all_metadata: list[tuple] = []  # (group_name, participant_id, session_label, session_idx, params)

    flat_idx: int = 0

    for group_name in group_names:
        group_cfg = sim_cfg.groups[group_name]
        session_cfg = sim_cfg.session_deltas[group_name]

        # Build full session label list: prepend "baseline" before the deltas
        session_labels = ["baseline"] + list(session_cfg.session_labels)

        for participant_idx in range(n_per_group):
            participant_id = f"{group_name}_{participant_idx:03d}"

            for session_idx, session_label in enumerate(session_labels):
                env_seed = int(all_seeds[flat_idx, 0])
                sim_seed = int(all_seeds[flat_idx, 1])
                rng_sim = np.random.default_rng(sim_seed)

                # Sample individual parameters (NumPy-based)
                params = sample_participant_params(
                    group_cfg, session_cfg, session_idx, rng_sim
                )

                # Generate trial sequence for this session
                trials = generate_session(config, env_seed)
                cue_probs = jnp.array(
                    [t.cue_probs for t in trials], dtype=jnp.float32
                )  # shape (n_trials, 3)

                all_omega_2.append(params["omega_2"])
                all_omega_3.append(params["omega_3"])
                all_kappa.append(params["kappa"])
                all_beta.append(params["beta"])
                all_zeta.append(params["zeta"])
                all_cue_probs.append(cue_probs)
                all_rng_keys.append(jax.random.PRNGKey(sim_seed))
                all_trials.append(trials)
                all_metadata.append(
                    (group_name, participant_id, session_label, session_idx, params)
                )

                flat_idx += 1

    print(f"Collection done in {time.time() - t_start:.1f}s. Running JAX vmap…")

    # --- Stack into batch arrays ---
    params_batch = {
        "omega_2": jnp.array(all_omega_2),
        "omega_3": jnp.array(all_omega_3),
        "kappa": jnp.array(all_kappa),
        "beta": jnp.array(all_beta),
        "zeta": jnp.array(all_zeta),
    }
    cue_probs_batch = jnp.stack(all_cue_probs)   # shape (P_total, n_trials, 3)
    rng_keys_batch = jnp.stack(all_rng_keys)      # shape (P_total, 2)

    # --- Single vmapped call over entire cohort ---
    scan_fn, base_attrs = _build_session_scanner()
    _vmapped = jax.vmap(
        lambda o2, o3, k, b, z, cp, rk: _run_session(
            scan_fn, base_attrs, o2, o3, k, b, z, cp, rk
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
    t_jax = time.time()
    all_choices_batch, all_rewards_batch, all_diverged_batch = _vmapped(
        params_batch["omega_2"],
        params_batch["omega_3"],
        params_batch["kappa"],
        params_batch["beta"],
        params_batch["zeta"],
        cue_probs_batch,
        rng_keys_batch,
    )
    print(f"JAX vmap completed in {time.time() - t_jax:.1f}s.")

    # --- DataFrame assembly ---
    rows: list[dict] = []

    for i in range(n_total):
        group_name, participant_id, session_label, session_idx, params = all_metadata[i]
        trials = all_trials[i]
        choices_list = [int(c) for c in all_choices_batch[i]]
        rewards_list = [int(r) for r in all_rewards_batch[i]]
        diverged_bool = bool(all_diverged_batch[i])

        for t_idx, trial in enumerate(trials):
            rows.append(
                {
                    "participant_id": participant_id,
                    "group": group_name,
                    "session": session_label,
                    "session_idx": session_idx,
                    "trial": trial.trial_idx,
                    "cue_chosen": choices_list[t_idx],
                    "reward": rewards_list[t_idx],
                    "cue_0_prob": trial.cue_probs[0],
                    "cue_1_prob": trial.cue_probs[1],
                    "cue_2_prob": trial.cue_probs[2],
                    "phase_label": trial.phase_label,
                    "phase_name": trial.phase_name,
                    "best_cue": trial.best_cue,
                    "true_omega_2": params["omega_2"],
                    "true_omega_3": params["omega_3"],
                    "true_kappa": params["kappa"],
                    "true_beta": params["beta"],
                    "true_zeta": params["zeta"],
                    "model": "hgf_3level",
                    "diverged": diverged_bool,
                }
            )

    df = pd.DataFrame(rows)

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved simulation output to: {output_path}")

    return df
