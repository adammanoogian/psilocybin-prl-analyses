"""Model fitting module for the PRL pick_best_cue HGF pipeline.

Public API surface:

* Configuration (Phase 28):

  - :class:`FitConfig` -- single source of truth for fitting parameters
  - :class:`SamplerConfig` -- NUTS/Laplace backend settings
  - :class:`MitigationConfig` -- conditioning cliff mitigations
  - :class:`CovariateConfig` -- hierarchical pooling and covariates
  - :class:`HGFPriorSpec` -- prior distributions for model parameters
  - :class:`PriorDist` -- single prior distribution specification

* Legacy v1.1 per-participant sequential path (frozen, in
  :mod:`prl_hgf.fitting.legacy`):

  - :func:`fit_batch` -- sequential per-participant NUTS loop
  - :func:`fit_participant` -- single-participant NUTS fit
  - :func:`extract_summary_rows`, :func:`flag_fit`
  - :func:`build_pymc_model_2level`, :func:`build_pymc_model_3level`
  - :func:`build_logp_ops_2level`, :func:`build_logp_ops_3level`

* v1.2 batched hierarchical path:

  - :func:`build_logp_fn_batched` -- pure JAX logp factory (numpyro path)
  - :func:`fit_batch_hierarchical` -- single-call cohort MCMC via FitConfig
  - :func:`build_logp_ops_batched` -- (deprecated) PyTensor Op wrapper
  - :func:`build_pymc_model_batched` -- (deprecated) PyMC model factory

* VB-Laplace paths (TAPAS ``tapas_fitModel`` equivalent):

  - :func:`fit_vb_laplace_prl` -- pick_best_cue (3-cue PRL) Laplace fit
"""

from __future__ import annotations

# Legacy (frozen) per-participant path -- re-exported via the shim modules so
# existing call sites such as ``from prl_hgf.fitting import fit_batch`` keep
# resolving without code changes.
from prl_hgf.fitting.batch import fit_batch

# Configuration dataclasses (Phase 28)
from prl_hgf.fitting.config import (
    CovariateConfig,
    FitConfig,
    MitigationConfig,
    SamplerConfig,
)
from prl_hgf.fitting.diagnostics import emit_diagnostic_csv
from prl_hgf.fitting.fit_vb_laplace_prl import (
    compare_models_laplace,
    fit_vb_laplace_prl,
    idata_to_fit_df,
)
from prl_hgf.fitting.hierarchical import (  # noqa: F401
    build_logp_fn_batched,
    build_logp_ops_batched,
    build_pymc_model_batched,
    fit_batch_hierarchical,
)
from prl_hgf.fitting.models import (
    build_pymc_model_2level,
    build_pymc_model_3level,
)
from prl_hgf.fitting.ops import (
    build_logp_ops_2level,
    build_logp_ops_3level,
)
from prl_hgf.fitting.priors import HGFPriorSpec, PriorDist
from prl_hgf.fitting.single import (
    extract_summary_rows,
    fit_participant,
    flag_fit,
)

__all__ = [
    # Configuration (Phase 28)
    "FitConfig",
    "SamplerConfig",
    "MitigationConfig",
    "CovariateConfig",
    "HGFPriorSpec",
    "PriorDist",
    # Diagnostics (Phase 36)
    "emit_diagnostic_csv",
    # legacy v1.1 path
    "fit_batch",
    "fit_participant",
    "extract_summary_rows",
    "flag_fit",
    "build_pymc_model_2level",
    "build_pymc_model_3level",
    "build_logp_ops_2level",
    "build_logp_ops_3level",
    # v1.2 batched hierarchical path
    "build_logp_fn_batched",
    "build_logp_ops_batched",
    "build_pymc_model_batched",
    "fit_batch_hierarchical",
    # VB-Laplace paths
    "fit_vb_laplace_prl",
    "compare_models_laplace",
    "idata_to_fit_df",
]
