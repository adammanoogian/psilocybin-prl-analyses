"""Simulation subpackage for the PRL pick_best_cue pipeline.

Provides single-agent simulation (trial-by-trial HGF forward pass) and
parameter sampling from group distributions with session deltas.

Public API
----------
:func:`simulate_agent`
    Run one session of trial-by-trial HGF simulation for a single agent.
:func:`sample_participant_params`
    Draw individual parameters from group distributions with session deltas
    and clip to model bounds.
:class:`SimulationResult`
    Frozen dataclass holding choices, rewards, and prior beliefs.
"""

from __future__ import annotations

from prl_hgf.simulation.agent import (
    PARAM_BOUNDS,
    SimulationResult,
    sample_participant_params,
    simulate_agent,
)

__all__ = [
    "simulate_agent",
    "sample_participant_params",
    "SimulationResult",
    "PARAM_BOUNDS",
]
