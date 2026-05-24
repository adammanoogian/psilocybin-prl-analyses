"""Unit tests for prl_hgf.power.curves aggregation and power computation.

All tests use ``tmp_path`` for file I/O and construct minimal synthetic
parquet fixtures via :func:`~prl_hgf.power.schema.write_parquet_row`.
No MCMC, JAX, or pyhgf is required.

Run with::

    pytest tests/test_power_curves.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from prl_hgf.power.curves import aggregate_parquets, compute_power_a, compute_power_b
from prl_hgf.power.schema import write_parquet_row

# ---------------------------------------------------------------------------
# Helper: minimal valid row factory
# ---------------------------------------------------------------------------


def _make_row(
    sweep_type: str = "did_postdose",
    effect_size: float = 0.5,
    n_per_group: int = 20,
    iteration: int = 0,
    bf_exceeds: bool = True,
    bms_correct: bool = True,
) -> dict[str, Any]:
    """Return a dict that satisfies POWER_SCHEMA with sensible defaults."""
    return {
        "sweep_type": sweep_type,
        "effect_size": float(effect_size),
        "n_per_group": int(n_per_group),
        "trial_count": 200,
        "iteration": int(iteration),
        "parameter": "omega_2",
        "bf_value": 12.3 if bf_exceeds else 2.0,
        "bf_exceeds": bf_exceeds,
        "bms_xp": 0.9 if bms_correct else 0.3,
        "bms_correct": bms_correct,
        "recovery_r": 0.85,
        "n_divergences": 0,
        "mean_rhat": 1.01,
    }


# ---------------------------------------------------------------------------
# Fake PowerConfig for monkeypatching
# ---------------------------------------------------------------------------


class _SmallPowerConfig:
    """Minimal mock returning a small grid so expected count == 2."""

    n_per_group_grid = [20]
    effect_size_grid = [0.5]
    n_iterations = 2


# ---------------------------------------------------------------------------
# aggregate_parquets
# ---------------------------------------------------------------------------


def test_aggregate_empty_dir_raises(tmp_path: Path) -> None:
    """aggregate_parquets raises FileNotFoundError for an empty directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory (empty).
    """
    with pytest.raises(FileNotFoundError, match="No .parquet files found"):
        aggregate_parquets(tmp_path)


def test_aggregate_single_parquet(tmp_path: Path) -> None:
    """aggregate_parquets reads a single parquet file correctly.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    row = _make_row()
    write_parquet_row(row, tmp_path / "job_0_task_0000_did_postdose.parquet")

    with patch(
        "prl_hgf.power.curves.load_power_config",
        return_value=_SmallPowerConfig(),
    ):
        df = aggregate_parquets(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["sweep_type"] == "did_postdose"
    assert df.iloc[0]["n_per_group"] == 20


def test_aggregate_multiple_parquets(tmp_path: Path) -> None:
    """aggregate_parquets concatenates 3 parquet files, giving 3 rows.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    for i in range(3):
        row = _make_row(iteration=i)
        write_parquet_row(row, tmp_path / f"job_0_task_{i:04d}.parquet")

    # Grid size = 1 * 1 * 2 = 2, but we have 3 rows (1 extra) — no warning
    # because warnings only fire when n_actual < n_expected
    class _LargeGrid:
        n_per_group_grid = [20]
        effect_size_grid = [0.5]
        n_iterations = 3  # exactly 3 expected

    with patch(
        "prl_hgf.power.curves.load_power_config",
        return_value=_LargeGrid(),
    ):
        df = aggregate_parquets(tmp_path)

    assert len(df) == 3


def test_aggregate_warns_missing_cells(tmp_path: Path) -> None:
    """aggregate_parquets emits UserWarning when fewer rows than expected.

    The mock PowerConfig expects 2 rows but only 1 parquet file is written.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    row = _make_row(iteration=0)
    write_parquet_row(row, tmp_path / "job_0_task_0000.parquet")

    with (
        patch(
            "prl_hgf.power.curves.load_power_config",
            return_value=_SmallPowerConfig(),
        ),
        pytest.warns(UserWarning, match="Missing"),
    ):
        aggregate_parquets(tmp_path)


# ---------------------------------------------------------------------------
# compute_power_a
# ---------------------------------------------------------------------------


def test_compute_power_a_basic() -> None:
    """compute_power_a correctly computes mean of bf_exceeds.

    With 2 rows, one True and one False, p_bf_exceeds should equal 0.5.
    """
    df = pd.DataFrame(
        [
            _make_row(n_per_group=20, effect_size=0.5, bf_exceeds=True),
            _make_row(n_per_group=20, effect_size=0.5, bf_exceeds=False),
        ]
    )
    result = compute_power_a(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_per_group"] == 20
    assert row["effect_size"] == pytest.approx(0.5)
    assert row["p_bf_exceeds"] == pytest.approx(0.5)
    assert row["n_iterations"] == 2


def test_compute_power_a_groups_correctly() -> None:
    """compute_power_a returns one row per (n_per_group, effect_size) cell.

    Two N levels times two effect sizes gives 4 output rows.
    """
    rows = []
    for n in [20, 30]:
        for d in [0.3, 0.5]:
            for i in range(5):
                rows.append(_make_row(n_per_group=n, effect_size=d, bf_exceeds=(i < 3)))

    df = pd.DataFrame(rows)
    result = compute_power_a(df)

    assert len(result) == 4
    assert set(result["n_per_group"]) == {20, 30}
    assert sorted(result["effect_size"].unique()) == pytest.approx([0.3, 0.5])
    # Each cell has 5 iterations, 3 True -> p = 0.6
    assert result["p_bf_exceeds"].tolist() == pytest.approx([0.6] * 4)
    assert result["n_iterations"].tolist() == [5, 5, 5, 5]


def test_compute_power_a_filters_sweep_type() -> None:
    """compute_power_a only uses rows matching sweep_type.

    Rows with a different sweep_type must be excluded.
    """
    df = pd.DataFrame(
        [
            _make_row(sweep_type="did_postdose", n_per_group=20, bf_exceeds=True),
            _make_row(sweep_type="kappa", n_per_group=20, bf_exceeds=False),
        ]
    )
    result = compute_power_a(df, sweep_type="did_postdose")

    assert len(result) == 1
    assert result.iloc[0]["p_bf_exceeds"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_power_b
# ---------------------------------------------------------------------------


def test_compute_power_b_basic() -> None:
    """compute_power_b correctly computes mean of bms_correct.

    Four rows, half True and half False, should give p_bms_correct = 0.5.
    """
    rows = [
        _make_row(n_per_group=20, iteration=0, bms_correct=True),
        _make_row(n_per_group=20, iteration=1, bms_correct=False),
        _make_row(n_per_group=30, iteration=0, bms_correct=True),
        _make_row(n_per_group=30, iteration=1, bms_correct=True),
    ]
    df = pd.DataFrame(rows)
    result = compute_power_b(df)

    assert len(result) == 2
    n20 = result[result["n_per_group"] == 20].iloc[0]
    n30 = result[result["n_per_group"] == 30].iloc[0]
    assert n20["p_bms_correct"] == pytest.approx(0.5)
    assert n30["p_bms_correct"] == pytest.approx(1.0)


def test_compute_power_b_deduplicates() -> None:
    """compute_power_b deduplicates on (n_per_group, iteration).

    Each SLURM task produces 3 rows (one per sweep_type) with identical BMS
    values.  The function must not triple-count them.
    """
    # Simulate 2 tasks, each producing 3 sweep_type rows (6 rows total)
    rows = []
    for sweep in ["did_postdose", "kappa", "omega_3"]:
        rows.append(
            _make_row(
                sweep_type=sweep,
                n_per_group=20,
                iteration=0,
                bms_correct=True,
            )
        )
        rows.append(
            _make_row(
                sweep_type=sweep,
                n_per_group=20,
                iteration=1,
                bms_correct=False,
            )
        )

    df = pd.DataFrame(rows)  # 6 rows

    result = compute_power_b(df)

    assert len(result) == 1
    row = result.iloc[0]
    # After dedup: 2 unique iterations — one True, one False -> p = 0.5
    assert row["p_bms_correct"] == pytest.approx(0.5)
    assert row["n_iterations"] == 2  # NOT 6
