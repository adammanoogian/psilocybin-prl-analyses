"""Power curve computation for BFDA sweep results.

Provides functions to aggregate per-job parquet files into a master DataFrame
and compute P(BF > threshold) and P(correct BMS) power curves for publication
figures and sample-size recommendations.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from prl_hgf.power.config import load_power_config
from prl_hgf.power.grid import total_grid_size
from prl_hgf.power.schema import POWER_SCHEMA

__all__ = [
    "aggregate_parquets",
    "compute_power_a",
    "compute_power_b",
]


def aggregate_parquets(results_dir: Path) -> pd.DataFrame:
    """Concatenate all per-job parquet files in a directory into one DataFrame.

    Globs for ``*.parquet`` files under ``results_dir``, reads each with
    ``pyarrow``, concatenates them, and validates that the resulting columns
    match :data:`~prl_hgf.power.schema.POWER_SCHEMA`.  For each ``sweep_type``
    present, the actual row count is compared against the expected count
    derived from the power grid dimensions; missing cells trigger a
    :class:`UserWarning`.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``.parquet`` files produced by the power sweep.

    Returns
    -------
    pd.DataFrame
        Concatenated results with columns matching :data:`POWER_SCHEMA`.

    Raises
    ------
    FileNotFoundError
        If no ``.parquet`` files are found in ``results_dir``.
    ValueError
        If the concatenated DataFrame is missing required schema columns.

    Warns
    -----
    UserWarning
        For each ``sweep_type`` with fewer rows than expected based on the
        power grid dimensions.

    Examples
    --------
    >>> from pathlib import Path
    >>> from prl_hgf.power.curves import aggregate_parquets
    >>> df = aggregate_parquets(Path("results/power"))  # doctest: +SKIP
    """
    parquet_files = sorted(results_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {results_dir}. "
            f"Ensure the power sweep has been run and results are in "
            f"{results_dir}."
        )

    frames = [pd.read_parquet(f, engine="pyarrow") for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    # Validate schema columns
    missing_cols = set(POWER_SCHEMA) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Concatenated DataFrame is missing required columns: "
            f"{sorted(missing_cols)}. Expected: {list(POWER_SCHEMA)}"
        )

    # Check for missing cells per sweep_type
    power_cfg = load_power_config()
    n_expected = total_grid_size(
        power_cfg.n_per_group_grid,
        power_cfg.effect_size_grid,
        power_cfg.n_iterations,
    )

    for st in df["sweep_type"].unique():
        n_actual = int((df["sweep_type"] == st).sum())
        if n_actual < n_expected:
            warnings.warn(
                f"Missing {n_expected - n_actual} cells for sweep_type={st}: "
                f"expected {n_expected}, got {n_actual}",
                UserWarning,
                stacklevel=2,
            )

    return df


def compute_power_a(
    master_df: pd.DataFrame,
    bf_threshold: float = 10.0,
    sweep_type: str = "did_postdose",
) -> pd.DataFrame:
    """Compute P(BF > threshold) power curves for Power Analysis A.

    Filters to the specified ``sweep_type`` (default: ``"did_postdose"``,
    the primary psilocybin vs placebo contrast), then groups by
    ``(n_per_group, effect_size)`` to compute the empirical probability that
    the Bayes factor exceeds ``bf_threshold``.

    Parameters
    ----------
    master_df : pd.DataFrame
        Concatenated power sweep results as returned by
        :func:`aggregate_parquets`.
    bf_threshold : float, optional
        Bayes factor threshold for declaring evidence (default: 10.0).
        Rows are counted as a "success" when ``bf_value >= bf_threshold``.
        Note: the parquet schema stores the pre-computed boolean
        ``bf_exceeds``; this parameter documents intent but the boolean
        column is used directly.
    sweep_type : str, optional
        Which contrast to compute curves for (default: ``"did_postdose"``).
        Pass a different value to compute curves for other contrasts.

    Returns
    -------
    pd.DataFrame
        One row per ``(n_per_group, effect_size)`` cell with columns:

        - ``n_per_group`` : int
        - ``effect_size`` : float
        - ``p_bf_exceeds`` : float  — proportion of iterations where BF > threshold
        - ``n_iterations`` : int    — number of iterations in the cell

    Examples
    --------
    >>> import pandas as pd
    >>> from prl_hgf.power.curves import compute_power_a
    >>> df = pd.DataFrame({
    ...     "sweep_type": ["did_postdose"] * 4,
    ...     "n_per_group": [20, 20, 30, 30],
    ...     "effect_size": [0.5, 0.5, 0.5, 0.5],
    ...     "bf_exceeds": [True, False, True, True],
    ... })
    >>> compute_power_a(df)  # doctest: +SKIP
    """
    subset = master_df[master_df["sweep_type"] == sweep_type].copy()

    grouped = (
        subset.groupby(["n_per_group", "effect_size"])["bf_exceeds"]
        .agg(p_bf_exceeds="mean", n_iterations="count")
        .reset_index()
    )
    return grouped[["n_per_group", "effect_size", "p_bf_exceeds", "n_iterations"]]


def compute_power_b(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute P(correct BMS) power curves for Power Analysis B.

    BMS values are identical across the three ``sweep_type`` rows produced for
    each ``(n_per_group, iteration)`` task, so duplicate ``(n_per_group,
    iteration)`` combinations are dropped before computing the mean.  This
    avoids triple-counting BMS outcomes.

    Parameters
    ----------
    master_df : pd.DataFrame
        Concatenated power sweep results as returned by
        :func:`aggregate_parquets`.

    Returns
    -------
    pd.DataFrame
        One row per ``n_per_group`` level with columns:

        - ``n_per_group`` : int
        - ``p_bms_correct`` : float  — proportion of iterations with correct BMS
        - ``n_iterations`` : int     — number of unique iterations

    Examples
    --------
    >>> import pandas as pd
    >>> from prl_hgf.power.curves import compute_power_b
    >>> df = pd.DataFrame({
    ...     "sweep_type": ["did_postdose", "kappa", "omega_3"] * 2,
    ...     "n_per_group": [20] * 6,
    ...     "iteration": [0, 0, 0, 1, 1, 1],
    ...     "bms_correct": [True, True, True, False, False, False],
    ... })
    >>> compute_power_b(df)  # doctest: +SKIP
    """
    # Deduplicate: keep one row per (n_per_group, iteration)
    deduped = master_df.drop_duplicates(
        subset=["n_per_group", "iteration"]
    ).copy()

    grouped = (
        deduped.groupby("n_per_group")["bms_correct"]
        .agg(p_bms_correct="mean", n_iterations="count")
        .reset_index()
    )
    return grouped[["n_per_group", "p_bms_correct", "n_iterations"]]
