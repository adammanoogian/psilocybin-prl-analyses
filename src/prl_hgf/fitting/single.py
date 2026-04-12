"""Backward-compatibility shim for pre-v1.2 imports.

The real per-participant NUTS code now lives in
:mod:`prl_hgf.fitting.legacy.single`. This module re-exports every public
name so existing imports such as
``from prl_hgf.fitting.single import fit_participant`` keep working.

# Frozen for v1.1 reproducibility — DO NOT MODIFY.
# See src/prl_hgf/fitting/hierarchical.py for the v1.2+ implementation.
"""

from __future__ import annotations

from prl_hgf.fitting.legacy.single import (  # noqa: F401
    ESS_THRESHOLD,
    R_HAT_THRESHOLD,
    extract_summary_rows,
    fit_participant,
    flag_fit,
)

__all__ = [
    "fit_participant",
    "extract_summary_rows",
    "flag_fit",
    "R_HAT_THRESHOLD",
    "ESS_THRESHOLD",
]
