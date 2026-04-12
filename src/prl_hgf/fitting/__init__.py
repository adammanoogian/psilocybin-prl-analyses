"""Model fitting module for the PRL pick_best_cue HGF pipeline.

Public API surface:

* Legacy v1.1 per-participant sequential path (frozen, in
  :mod:`prl_hgf.fitting.legacy`):

  - :func:`fit_batch` -- sequential per-participant NUTS loop
  - :func:`fit_participant` -- single-participant NUTS fit
  - :func:`extract_summary_rows`, :func:`flag_fit`
  - :func:`build_pymc_model_2level`, :func:`build_pymc_model_3level`
  - :func:`build_logp_ops_2level`, :func:`build_logp_ops_3level`

* v1.2 batched hierarchical path (added by Plans 12-02 / 12-03):

  - :func:`build_logp_ops_batched` -- batched JAX logp Op factory
  - :func:`fit_batch_hierarchical` -- single-call cohort orchestrator
"""

from __future__ import annotations

# Legacy (frozen) per-participant path -- re-exported via the shim modules so
# existing call sites such as ``from prl_hgf.fitting import fit_batch`` keep
# resolving without code changes.
from prl_hgf.fitting.batch import fit_batch
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
    fit_participant,
    flag_fit,
)

# v1.2 batched hierarchical path -- populated by Plan 12-02
# (build_logp_ops_batched) and Plan 12-03 (fit_batch_hierarchical,
# build_pymc_model_batched).  Until those land the symbols are not part of the
# public API; importing from prl_hgf.fitting.hierarchical directly will raise
# ModuleNotFoundError.

__all__ = [
    "fit_batch",
    "fit_participant",
    "extract_summary_rows",
    "flag_fit",
    "build_pymc_model_2level",
    "build_pymc_model_3level",
    "build_logp_ops_2level",
    "build_logp_ops_3level",
]
