"""Pre-flight validation for FitConfig before launching sampler."""

from __future__ import annotations

from prl_hgf.fitting.config import FitConfig
from prl_hgf.fitting.priors import HGFPriorSpec


def validate_fit_config(
    fit_config: FitConfig,
    prior_spec: HGFPriorSpec,
) -> None:
    """Validate config + prior spec compatibility before launching fit.

    Parameters
    ----------
    fit_config : FitConfig
        Configuration to validate.
    prior_spec : HGFPriorSpec
        Prior specification to validate against config.

    Raises
    ------
    ValueError
        If configuration is invalid or incompatible.
    """
    # Phase 29 adds: memory pre-flight for dense mass matrix
    # Phase 30 adds: logical conflict detection (non_centered without hierarchical)
    pass
