"""Model fitting module for the PRL pick_best_cue HGF pipeline.

Exports the public API for Phase 4 (single-participant MCMC fitting):

* :func:`fit_participant` — run NUTS MCMC for one participant.
* :func:`extract_summary_rows` — convert InferenceData to summary rows.
* :func:`flag_fit` — check convergence diagnostics.
* :func:`build_pymc_model_2level` — build PyMC model (2-level HGF).
* :func:`build_pymc_model_3level` — build PyMC model (3-level HGF).
* :func:`build_logp_ops_2level` — build JAX-backed logp Op (2-level).
* :func:`build_logp_ops_3level` — build JAX-backed logp Op (3-level).
"""

from __future__ import annotations

from prl_hgf.fitting.models import (
    build_pymc_model_2level,
    build_pymc_model_3level,
)
from prl_hgf.fitting.ops import (
    build_logp_ops_2level,
    build_logp_ops_3level,
)
from prl_hgf.fitting.single import (
    extract_summary_rows,
    flag_fit,
    fit_participant,
)

__all__ = [
    "fit_participant",
    "extract_summary_rows",
    "flag_fit",
    "build_pymc_model_2level",
    "build_pymc_model_3level",
    "build_logp_ops_2level",
    "build_logp_ops_3level",
]
