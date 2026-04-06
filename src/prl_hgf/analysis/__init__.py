"""Group-level analysis module.

Provides parameter recovery analysis (true vs recovered metrics and plots),
Bayesian model selection utilities, mixed-effects group analysis via bambi,
and frequentist effect size computation for Phases 5-6.
"""

from __future__ import annotations

from prl_hgf.analysis.bms import (
    compute_batch_waic,
    compute_subject_waic,
    plot_exceedance_probabilities,
    run_group_bms,
    run_stratified_bms,
)
from prl_hgf.analysis.effect_sizes import (
    compute_cohens_d,
    compute_effect_sizes_table,
)
from prl_hgf.analysis.group import (
    build_estimates_wide,
    extract_posterior_contrasts,
    fit_group_model,
    summarize_group_models,
)
from prl_hgf.analysis.group_plots import (
    plot_all_rainclouds,
    plot_interaction,
    plot_raincloud,
)
from prl_hgf.analysis.recovery import (
    build_recovery_df,
    compute_correlation_matrix,
    compute_recovery_metrics,
)

__all__ = [
    # Recovery
    "build_recovery_df",
    "compute_recovery_metrics",
    "compute_correlation_matrix",
    # BMS
    "compute_subject_waic",
    "compute_batch_waic",
    "run_group_bms",
    "run_stratified_bms",
    "plot_exceedance_probabilities",
    # Group analysis
    "build_estimates_wide",
    "fit_group_model",
    "extract_posterior_contrasts",
    "summarize_group_models",
    # Effect sizes
    "compute_cohens_d",
    "compute_effect_sizes_table",
    # Group plots
    "plot_raincloud",
    "plot_interaction",
    "plot_all_rainclouds",
]
