"""Backward-compatibility shim for pre-v1.2 imports.

The real per-participant batch loop now lives in
:mod:`prl_hgf.fitting.legacy.batch`. This module re-exports every public
name (and the private ``_prewarm_jit`` helper consumed by
``scripts/08_run_power_iteration.py``) so existing imports keep working.

# Frozen for v1.1 reproducibility — DO NOT MODIFY.
# See src/prl_hgf/fitting/hierarchical.py for the v1.2+ implementation.
"""

from __future__ import annotations

from prl_hgf.fitting.legacy.batch import (  # noqa: F401
    _RESULT_COLUMNS,
    _make_nan_rows,
    _prewarm_jit,
    fit_batch,
)

__all__ = ["fit_batch"]
