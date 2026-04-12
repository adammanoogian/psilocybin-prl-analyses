"""Frozen v1.1 per-participant sequential fitting code.

# Frozen for v1.1 reproducibility — DO NOT MODIFY.
# See src/prl_hgf/fitting/hierarchical.py for the v1.2+ implementation.

This subpackage preserves the exact per-participant NUTS fitting path that
was code-complete on 2026-04-07 so that v1.1 BFDA results remain bit-exactly
reproducible after the v1.2 batched hierarchical refactor lands.

Phase 14 will add a ``--legacy`` flag to ``scripts/08_run_power_iteration.py``
that routes through these functions; Phase 12 only freezes the code in place.
"""

from __future__ import annotations

from prl_hgf.fitting.legacy.batch import fit_batch
from prl_hgf.fitting.legacy.single import (
    extract_summary_rows,
    fit_participant,
    flag_fit,
)

__all__ = [
    "fit_batch",
    "fit_participant",
    "extract_summary_rows",
    "flag_fit",
]
