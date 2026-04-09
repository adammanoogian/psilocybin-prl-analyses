"""Batch simulation orchestrator for the PRL pick_best_cue pipeline.

Generates the full synthetic dataset: 2 groups × N participants × 3 sessions
per the study design.  Each participant-session is simulated by calling
:func:`~prl_hgf.simulation.agent.simulate_agent` with individually sampled
parameters, and the results are assembled into a single tidy DataFrame.

The output has one row per trial and includes ground-truth parameters as
``true_*`` columns, making it the direct input for Phase 4 (fitting) and
Phase 5 (parameter recovery).

Seed strategy
-------------
A master RNG derived from ``config.simulation.master_seed`` draws all
per-(participant, session) seeds upfront before any simulation begins.  This
ensures that changing ``n_participants_per_group`` does not alter seeds for
earlier participants.

JIT pre-warm
------------
Before the main loop the function builds a dummy network and calls
``input_data`` once.  This triggers JAX JIT compilation so that the first
real participant-session does not pay the compilation overhead.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from prl_hgf.env.simulator import generate_session
from prl_hgf.env.task_config import AnalysisConfig
from prl_hgf.models.hgf_3level import build_3level_network
from prl_hgf.simulation.agent import sample_participant_params, simulate_agent

__all__ = ["simulate_batch"]


# ---------------------------------------------------------------------------
# JIT pre-warm helper
# ---------------------------------------------------------------------------


def _prewarm_jit() -> None:
    """Trigger JAX JIT compilation before the main batch loop.

    Builds a minimal 3-level network and calls ``input_data`` once with a
    single dummy trial.  This pays the one-time compilation cost upfront so
    the first real participant-session starts immediately.
    """
    net = build_3level_network(omega_2=-3.0, omega_3=-6.0, kappa=1.0)
    dummy_input = np.zeros((1, 3), dtype=float)
    dummy_obs = np.zeros((1, 3), dtype=int)
    dummy_obs[0, 0] = 1
    net.input_data(input_data=dummy_input, observed=dummy_obs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_batch(
    config: AnalysisConfig,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Simulate a full cohort of synthetic participants across all sessions.

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
        ``true_beta``, ``true_zeta``, ``model``

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
    20
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

    # --- JIT pre-warm ---
    print("Pre-warming JAX JIT compilation…")
    _prewarm_jit()
    print("JIT pre-warm complete.")

    rows: list[dict] = []
    flat_idx: int = 0
    t_start = time.time()

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

                # Sample individual parameters
                params = sample_participant_params(
                    group_cfg, session_cfg, session_idx, rng_sim
                )

                # Build fresh network with participant's parameters
                net = build_3level_network(
                    omega_2=params["omega_2"],
                    omega_3=params["omega_3"],
                    kappa=params["kappa"],
                )

                # Generate trial sequence for this session
                trials = generate_session(config, env_seed)

                # Run simulation
                result = simulate_agent(
                    net, trials, params["beta"], params["zeta"], rng_sim
                )

                # Build one row per trial
                for t_idx, trial in enumerate(trials):
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "group": group_name,
                            "session": session_label,
                            "session_idx": session_idx,
                            "trial": trial.trial_idx,
                            "cue_chosen": result.choices[t_idx],
                            "reward": result.rewards[t_idx],
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
                            "diverged": result.diverged,
                        }
                    )

                flat_idx += 1

                # Progress logging every 10 participant-sessions
                if flat_idx % 10 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"[{flat_idx}/{n_total}] {group_name} "
                        f"P{participant_idx:03d} session {session_label} "
                        f"done ({elapsed:.1f}s)"
                    )

    df = pd.DataFrame(rows)

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved simulation output to: {output_path}")

    return df
