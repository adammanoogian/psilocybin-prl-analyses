"""Task environment module.

Provides PRL pick_best_cue task configuration loading, validation,
and trial sequence generation.
"""

from __future__ import annotations

from prl_hgf.env.simulator import Trial, generate_reward, generate_session
from prl_hgf.env.task_config import (
    AnalysisConfig,
    CriterionConfig,
    CueProbPair,
    ReversalConfig,
    load_config,
)

__all__ = [
    "AnalysisConfig",
    "CriterionConfig",
    "CueProbPair",
    "ReversalConfig",
    "Trial",
    "generate_reward",
    "generate_session",
    "load_config",
]
