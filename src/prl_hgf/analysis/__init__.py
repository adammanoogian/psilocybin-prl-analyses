"""Group-level analysis module.

Provides parameter recovery analysis (true vs recovered metrics and plots)
and Bayesian model selection utilities for Phases 5-6.
"""

from __future__ import annotations

from prl_hgf.analysis.bms import (
    compute_batch_waic,
    compute_subject_waic,
    plot_exceedance_probabilities,
    run_group_bms,
    run_stratified_bms,
)
from prl_hgf.analysis.recovery import (
    build_recovery_df,
    compute_correlation_matrix,
    compute_recovery_metrics,
)

__all__ = [
    "build_recovery_df",
    "compute_recovery_metrics",
    "compute_correlation_matrix",
    "compute_subject_waic",
    "compute_batch_waic",
    "run_group_bms",
    "run_stratified_bms",
    "plot_exceedance_probabilities",
]
