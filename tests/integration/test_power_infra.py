"""Unit and integration tests for power analysis infrastructure.

Covers:
- :func:`~prl_hgf.power.seeds.make_child_rng`: RNG independence, state
  vectors, reproducibility, and input validation.
- :data:`~prl_hgf.power.schema.POWER_SCHEMA`: column count and names.
- :func:`~prl_hgf.power.schema.write_parquet_row`: parquet roundtrip,
  schema enforcement (missing/extra column rejection), and directory
  creation.
- Entry point ``scripts/08_run_power_iteration.py --dry-run``: end-to-end
  subprocess smoke test writing to ``tmp_path``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prl_hgf.power.schema import POWER_SCHEMA, write_parquet_row
from prl_hgf.power.seeds import make_child_rng

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_ENTRY_POINT = _SCRIPTS_DIR / "08_run_power_iteration.py"

_VALID_ROW: dict = {
    "sweep_type":    "omega_2",
    "effect_size":   0.5,
    "n_per_group":   20,
    "trial_count":   200,
    "iteration":     0,
    "parameter":     "omega_2",
    "bf_value":      12.3,
    "bf_exceeds":    True,
    "bms_xp":        0.9,
    "bms_correct":   True,
    "recovery_r":    0.85,
    "n_divergences": 0,
    "mean_rhat":     1.01,
}


# ---------------------------------------------------------------------------
# SeedSequence tests
# ---------------------------------------------------------------------------


def test_make_child_rng_returns_generator() -> None:
    """make_child_rng returns an np.random.Generator."""
    rng = make_child_rng(master_seed=42, n_jobs=10, job_index=0)
    assert isinstance(rng, np.random.Generator)


def test_make_child_rng_distinct_state_vectors() -> None:
    """Two tasks from the same master seed have distinct bit_generator states.

    Directly verifies Phase 8 Success Criterion 4: no two child seeds share
    a state vector. Compares the PCG64 internal 128-bit state integer.
    """
    rng_0 = make_child_rng(master_seed=42, n_jobs=10, job_index=0)
    rng_1 = make_child_rng(master_seed=42, n_jobs=10, job_index=1)
    state_0 = rng_0.bit_generator.state
    state_1 = rng_1.bit_generator.state
    assert state_0["state"]["state"] != state_1["state"]["state"], (
        "Job 0 and job 1 share the same PCG64 internal state — "
        "SeedSequence independence guarantee violated."
    )


def test_make_child_rng_independent_streams() -> None:
    """Streams from different tasks are statistically uncorrelated.

    Draws 1000 samples from each and asserts |correlation| < 0.1.
    This is a supplementary statistical check alongside the state vector test.
    """
    rng_0 = make_child_rng(master_seed=42, n_jobs=10, job_index=0)
    rng_1 = make_child_rng(master_seed=42, n_jobs=10, job_index=1)
    samples_0 = rng_0.random(1000)
    samples_1 = rng_1.random(1000)
    corr = np.corrcoef(samples_0, samples_1)[0, 1]
    assert abs(corr) < 0.1, (
        f"Streams from job 0 and job 1 are correlated: r={corr:.4f}. "
        "Expected |r| < 0.1 for independent SeedSequence children."
    )


def test_make_child_rng_reproducible() -> None:
    """Same (master_seed, n_jobs, job_index) always yields identical draws."""
    rng_a = make_child_rng(master_seed=42, n_jobs=10, job_index=3)
    rng_b = make_child_rng(master_seed=42, n_jobs=10, job_index=3)
    samples_a = rng_a.random(100)
    samples_b = rng_b.random(100)
    assert np.array_equal(samples_a, samples_b), (
        "make_child_rng is not reproducible: same inputs produced different draws."
    )


def test_make_child_rng_validates_job_index() -> None:
    """make_child_rng raises ValueError for out-of-range job_index."""
    with pytest.raises(ValueError, match="job_index"):
        make_child_rng(42, 10, 10)  # index == n_jobs (off by one)
    with pytest.raises(ValueError, match="job_index"):
        make_child_rng(42, 10, -1)  # negative index


def test_make_child_rng_validates_n_jobs() -> None:
    """make_child_rng raises ValueError for n_jobs < 1."""
    with pytest.raises(ValueError, match="n_jobs"):
        make_child_rng(42, 0, 0)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_power_schema_has_13_columns() -> None:
    """POWER_SCHEMA defines exactly 13 columns with the expected names."""
    assert len(POWER_SCHEMA) == 13, (
        f"POWER_SCHEMA has {len(POWER_SCHEMA)} columns, expected 13."
    )
    expected_columns = {
        "sweep_type", "effect_size", "n_per_group", "trial_count",
        "iteration", "parameter", "bf_value", "bf_exceeds", "bms_xp",
        "bms_correct", "recovery_r", "n_divergences", "mean_rhat",
    }
    assert set(POWER_SCHEMA.keys()) == expected_columns, (
        f"Column name mismatch. "
        f"Extra: {set(POWER_SCHEMA.keys()) - expected_columns}. "
        f"Missing: {expected_columns - set(POWER_SCHEMA.keys())}."
    )


def test_write_parquet_row_roundtrip(tmp_path: Path) -> None:
    """write_parquet_row produces a readable parquet with correct dtypes."""
    out = tmp_path / "test.parquet"
    write_parquet_row(_VALID_ROW, out)
    df = pd.read_parquet(out)

    assert list(df.columns) == list(POWER_SCHEMA), (
        f"Column order mismatch. Got: {list(df.columns)}"
    )
    assert len(df) == 1

    # Check numeric dtypes
    assert df["effect_size"].dtype == np.float64
    assert df["n_per_group"].dtype == np.int64
    assert df["trial_count"].dtype == np.int64
    assert df["iteration"].dtype == np.int64
    assert df["bf_value"].dtype == np.float64
    assert df["bms_xp"].dtype == np.float64
    assert df["recovery_r"].dtype == np.float64
    assert df["n_divergences"].dtype == np.int64
    assert df["mean_rhat"].dtype == np.float64

    # Check bool columns
    assert df["bf_exceeds"].dtype == bool
    assert df["bms_correct"].dtype == bool

    # Check values round-trip
    assert df["sweep_type"].iloc[0] == "omega_2"
    assert df["effect_size"].iloc[0] == pytest.approx(0.5)
    assert df["n_per_group"].iloc[0] == 20
    assert df["bf_exceeds"].iloc[0] == True  # noqa: E712  # np.True_ != True (is)


def test_write_parquet_row_rejects_missing_column(tmp_path: Path) -> None:
    """write_parquet_row raises ValueError when a required column is absent."""
    row = {k: v for k, v in _VALID_ROW.items() if k != "bf_value"}
    with pytest.raises(ValueError, match="Missing"):
        write_parquet_row(row, tmp_path / "missing.parquet")


def test_write_parquet_row_rejects_extra_column(tmp_path: Path) -> None:
    """write_parquet_row raises ValueError when an unexpected column is present."""
    row = {**_VALID_ROW, "bonus_col": 99}
    with pytest.raises(ValueError, match="Unexpected"):
        write_parquet_row(row, tmp_path / "extra.parquet")


def test_write_parquet_row_creates_parent_dirs(tmp_path: Path) -> None:
    """write_parquet_row creates intermediate directories automatically."""
    out = tmp_path / "deep" / "nested" / "out.parquet"
    write_parquet_row(_VALID_ROW, out)
    assert out.exists(), f"Expected parquet at {out} but file not found."


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_entry_point(tmp_path: Path) -> None:
    """Dry-run chunk entry point produces valid 13-column parquet rows."""
    result = subprocess.run(
        [
            sys.executable,
            str(_ENTRY_POINT),
            "--chunk-id", "0",
            "--job-id", "test_int",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Entry point failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    out_path = tmp_path / "job_test_int_chunk_0000.parquet"
    assert out_path.exists(), (
        f"Expected parquet at {out_path} but file not found.\n"
        f"stdout: {result.stdout}"
    )

    df = pd.read_parquet(out_path)
    assert list(df.columns) == list(POWER_SCHEMA), (
        f"Parquet column mismatch. Got: {list(df.columns)}"
    )
    assert len(df.columns) == 13, (
        f"Expected 13 columns, got {len(df.columns)}."
    )
    # Chunk 0 with 3 chunks should have ~1400 task_ids × 1 row each
    assert len(df) > 0, "Expected at least 1 row from dry-run chunk."
