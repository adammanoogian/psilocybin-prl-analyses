"""Simulation subpackage for the PRL pick_best_cue pipeline.

Provides single-agent simulation (trial-by-trial HGF forward pass),
parameter sampling from group distributions with session deltas, and
batch simulation for generating full synthetic cohorts.

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
"""

from __future__ import annotations

from prl_hgf.simulation.agent import (
    PARAM_BOUNDS,
    SimulationResult,
    sample_participant_params,
    simulate_agent,
)
from prl_hgf.simulation.batch import simulate_batch

__all__ = [
    "simulate_agent",
    "sample_participant_params",
    "SimulationResult",
    "PARAM_BOUNDS",
    "simulate_batch",
]
