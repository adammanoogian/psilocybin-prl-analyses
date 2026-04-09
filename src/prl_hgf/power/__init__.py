"""Power analysis subpackage for the PRL HGF pipeline.

Provides config factory functions and dataclasses for BFDA-based power
analysis, SeedSequence-based RNG utilities, and parquet schema enforcement.
All public symbols are re-exported here.

Notes
-----
The ``power/`` subpackage wraps the existing pipeline without modifying any
existing modules. Config factories use :func:`dataclasses.replace` and never
perform file I/O (YAML loading is isolated to :func:`load_power_config`).
"""

from __future__ import annotations

from prl_hgf.power.config import PowerConfig, load_power_config, make_power_config
from prl_hgf.power.contrasts import (
    compute_all_contrasts,
    compute_did_contrast,
    compute_jzs_bf,
    compute_linear_trend_contrast,
)
from prl_hgf.power.curves import aggregate_parquets, compute_power_a, compute_power_b
from prl_hgf.power.grid import (
    chunk_task_ids,
    decode_sbf_task_id,
    decode_task_id,
    sbf_grid_size,
    total_grid_size,
)
from prl_hgf.power.iteration import (
    build_arrays_from_sim,
    run_power_iteration,
    run_sbf_iteration,
)
from prl_hgf.power.precheck import (
    PrecheckResult,
    build_eligibility_table,
    make_trial_config,
    run_recovery_precheck,
)
from prl_hgf.power.schema import POWER_SCHEMA, write_parquet_batch, write_parquet_row
from prl_hgf.power.seeds import make_child_rng, make_chunk_rngs

__all__ = [
    "make_power_config",
    "PowerConfig",
    "load_power_config",
    "chunk_task_ids",
    "decode_sbf_task_id",
    "decode_task_id",
    "sbf_grid_size",
    "total_grid_size",
    "make_child_rng",
    "make_chunk_rngs",
    "POWER_SCHEMA",
    "write_parquet_batch",
    "write_parquet_row",
    "PrecheckResult",
    "make_trial_config",
    "run_recovery_precheck",
    "build_eligibility_table",
    "compute_jzs_bf",
    "compute_did_contrast",
    "compute_linear_trend_contrast",
    "compute_all_contrasts",
    "build_arrays_from_sim",
    "run_power_iteration",
    "run_sbf_iteration",
    "aggregate_parquets",
    "compute_power_a",
    "compute_power_b",
]
