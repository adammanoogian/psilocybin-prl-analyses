"""Parquet schema enforcement for power analysis output rows.

Defines a fixed 13-column schema (``POWER_SCHEMA``) and a writer function
(:func:`write_parquet_row`) that enforces schema before persisting to disk.
All power sweep tasks must write through this module to guarantee a uniform
parquet layout across the entire result set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POWER_SCHEMA: dict[str, str] = {
    "sweep_type":    "string",
    "effect_size":   "float64",
    "n_per_group":   "int64",
    "trial_count":   "int64",
    "iteration":     "int64",
    "parameter":     "string",
    "bf_value":      "float64",
    "bf_exceeds":    "bool",
    "bms_xp":        "float64",
    "bms_correct":   "bool",
    "recovery_r":    "float64",
    "n_divergences": "int64",
    "mean_rhat":     "float64",
}


def write_parquet_row(row: dict, output_path: Path) -> None:
    """Write a single result row to a parquet file with schema enforcement.

    Creates all intermediate directories if necessary, then validates that
    ``row`` contains exactly the columns defined in :data:`POWER_SCHEMA` (no
    more, no fewer), casts each column to its declared dtype, enforces column
    order, and writes via the ``pyarrow`` engine.

    Parameters
    ----------
    row : dict
        Mapping of column name to scalar value.  Must contain exactly the 13
        keys defined in :data:`POWER_SCHEMA`.
    output_path : Path
        Destination ``.parquet`` file.  Parent directories are created
        automatically.

    Raises
    ------
    ValueError
        If ``row`` is missing any required columns.
    ValueError
        If ``row`` contains columns not present in :data:`POWER_SCHEMA`.

    Examples
    --------
    >>> from pathlib import Path
    >>> row = {
    ...     "sweep_type": "omega_2", "effect_size": 0.5, "n_per_group": 20,
    ...     "trial_count": 200, "iteration": 0, "parameter": "omega_2",
    ...     "bf_value": 12.3, "bf_exceeds": True, "bms_xp": 0.9,
    ...     "bms_correct": True, "recovery_r": 0.85, "n_divergences": 0,
    ...     "mean_rhat": 1.01,
    ... }
    >>> write_parquet_row(row, Path("/tmp/test.parquet"))  # doctest: +SKIP
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])

    missing = set(POWER_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected: {list(POWER_SCHEMA)}"
        )

    extra = set(df.columns) - set(POWER_SCHEMA)
    if extra:
        raise ValueError(
            f"Unexpected columns: {sorted(extra)}. "
            f"Expected: {list(POWER_SCHEMA)}"
        )

    for col, dtype in POWER_SCHEMA.items():
        df[col] = df[col].astype(dtype)

    df = df[list(POWER_SCHEMA)]
    df.to_parquet(output_path, index=False, engine="pyarrow")
