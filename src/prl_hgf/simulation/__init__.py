"""Simulation subpackage for the PRL pick_best_cue pipeline.

Provides single-agent simulation (trial-by-trial HGF forward pass),
parameter sampling from group distributions with session deltas,
batch simulation for generating full synthetic cohorts, and
hierarchical cohort simulation for Mode B recovery experiments.

Public API
----------
:func:`simulate_agent`
    Run one session of trial-by-trial HGF simulation for a single agent.
:func:`sample_participant_params`
    Draw individual parameters from group distributions with session deltas
    and clip to model bounds.
:class:`SimulationResult`
    Frozen dataclass holding choices, rewards, and prior beliefs.
:func:`simulate_batch`
    Orchestrate batch simulation over all groups, participants, and sessions,
    returning a tidy trial-level DataFrame with ground-truth parameters.
    Dispatches to the closed-loop path for criterion-based configs.
:func:`simulate_criterion_batch`
    Closed-loop (criterion-based reversal) cohort simulation with the same
    output schema as :func:`simulate_batch`.
:func:`simulate_criterion_session`
    Simulate one participant-session against the closed-loop environment.
:func:`simulate_hierarchical_cohort`
    Generate a cohort from a hierarchical generative model for Mode B
    recovery experiments.
"""

from __future__ import annotations

from prl_hgf.simulation.agent import (
    PARAM_BOUNDS,
    SimulationResult,
    sample_participant_params,
    simulate_agent,
)
from prl_hgf.simulation.batch import simulate_batch
from prl_hgf.simulation.criterion_sim import (
    CriterionEnvironment,
    CriterionSessionResult,
    simulate_criterion_batch,
    simulate_criterion_session,
)
from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort
from prl_hgf.simulation.jax_session import (
    simulate_cohort_jax,
    simulate_session_jax,
)
from prl_hgf.simulation.ppc import posterior_predictive_replay

__all__ = [
    "simulate_agent",
    "sample_participant_params",
    "SimulationResult",
    "PARAM_BOUNDS",
    "simulate_batch",
    "simulate_criterion_batch",
    "simulate_criterion_session",
    "CriterionEnvironment",
    "CriterionSessionResult",
    "simulate_hierarchical_cohort",
    "simulate_session_jax",
    "simulate_cohort_jax",
    "posterior_predictive_replay",
]
