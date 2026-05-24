"""Unit tests for power/precheck.py.

Tests cover make_trial_config (scaling, ratio, minimum), build_eligibility_table
(eligible/excluded/omega_3 cases), and PrecheckResult dataclass.

No calls to simulate_batch or fit_batch — all tests use synthetic DataFrames
or the real AnalysisConfig loaded from configs/prl_analysis.yaml.

Run with::

    conda run -n ds_env python -m pytest tests/test_precheck.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from prl_hgf.env.task_config import AnalysisConfig, load_config
from prl_hgf.power.precheck import (
    PrecheckResult,
    build_eligibility_table,
    make_trial_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base() -> AnalysisConfig:
    """Return the base AnalysisConfig loaded from configs/prl_analysis.yaml."""
    return load_config()


# ---------------------------------------------------------------------------
# make_trial_config — proportional scaling
# ---------------------------------------------------------------------------


def test_make_trial_config_scales_proportionally(base: AnalysisConfig) -> None:
    """make_trial_config produces a total close to target_total_trials.

    Due to integer rounding over 4 phases and 3 sets, the actual total may
    differ by a small amount. We allow ±4 trials.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    target = 200
    result = make_trial_config(base, target_total_trials=target)

    actual = result.task.n_trials_total
    assert abs(actual - target) <= 4, (
        f"Expected n_trials_total ≈ {target}, got {actual} (delta {actual - target})"
    )


def test_make_trial_config_preserves_transfer(base: AnalysisConfig) -> None:
    """Transfer n_trials must be unchanged after scaling.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    transfer_before = base.task.transfer.n_trials
    result = make_trial_config(base, target_total_trials=200)
    assert result.task.transfer.n_trials == transfer_before, (
        f"Transfer n_trials changed: expected {transfer_before}, "
        f"got {result.task.transfer.n_trials}"
    )


# ---------------------------------------------------------------------------
# make_trial_config — stable/volatile ratio
# ---------------------------------------------------------------------------


def test_make_trial_config_preserves_stable_volatile_ratio(
    base: AnalysisConfig,
) -> None:
    """After scaling, stable and volatile phase trial counts stay equal.

    The 4 phases are 2 stable + 2 volatile, all with the same base n_trials.
    Scaling must preserve the 1:1 ratio (each stable total == each volatile
    total per phase position).

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    result = make_trial_config(base, target_total_trials=200)
    phases = result.task.phases

    stable_trials = sum(p.n_trials for p in phases if p.phase_type == "stable")
    volatile_trials = sum(p.n_trials for p in phases if p.phase_type == "volatile")

    assert stable_trials == volatile_trials, (
        f"Stable/volatile ratio broken: stable={stable_trials}, "
        f"volatile={volatile_trials}"
    )


# ---------------------------------------------------------------------------
# make_trial_config — minimum one trial guard
# ---------------------------------------------------------------------------


def test_make_trial_config_minimum_one_trial(base: AnalysisConfig) -> None:
    """All phases must have at least 1 trial even for very small targets.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    # Very small target: 70 trials total / 3 sets = ~23 per set.
    # 20 are transfer, leaving only ~3 for 4 phases.
    # max(1, round(0.025 * 30)) should give 1 per phase.
    result = make_trial_config(base, target_total_trials=70)
    for phase in result.task.phases:
        assert phase.n_trials >= 1, (
            f"Phase '{phase.name}' has n_trials={phase.n_trials} (must be >= 1)"
        )


# ---------------------------------------------------------------------------
# build_eligibility_table — eligible case
# ---------------------------------------------------------------------------


def test_build_eligibility_table_eligible() -> None:
    """omega_2 with r=0.85 and passes_threshold=True → status 'power-eligible'."""
    metrics = pd.DataFrame(
        {
            "parameter": ["omega_2"],
            "r": [0.85],
            "passes_threshold": [True],
        }
    )
    elig = build_eligibility_table(metrics)

    assert len(elig) == 1
    row = elig.iloc[0]
    assert row["status"] == "power-eligible", (
        f"Expected 'power-eligible', got '{row['status']}'"
    )
    assert "0.85" in row["reason"] or "r=0.85" in row["reason"], (
        f"Reason should mention actual r value, got: {row['reason']}"
    )


# ---------------------------------------------------------------------------
# build_eligibility_table — excluded case
# ---------------------------------------------------------------------------


def test_build_eligibility_table_excluded() -> None:
    """zeta with r=0.55 and passes_threshold=False → status 'excluded', reason mentions '< 0.7'.

    Parameters are expected to follow the eligibility logic:
    excluded when r < threshold and not omega_3.
    """
    metrics = pd.DataFrame(
        {
            "parameter": ["zeta"],
            "r": [0.55],
            "passes_threshold": [False],
        }
    )
    elig = build_eligibility_table(metrics)

    assert len(elig) == 1
    row = elig.iloc[0]
    assert row["status"] == "excluded", (
        f"Expected 'excluded', got '{row['status']}'"
    )
    assert "< 0.7" in row["reason"], (
        f"Reason must mention '< 0.7' threshold, got: {row['reason']}"
    )


# ---------------------------------------------------------------------------
# build_eligibility_table — omega_3 always exploratory
# ---------------------------------------------------------------------------


def test_build_eligibility_table_omega3_always_exploratory() -> None:
    """omega_3 with r=0.90 and passes_threshold=True must still be 'exploratory -- upper bound'.

    This is a locked project decision: omega_3 recovery is known to be poor
    with binary PRL data and BFDA would be inflated 20-40pp.
    """
    metrics = pd.DataFrame(
        {
            "parameter": ["omega_3"],
            "r": [0.90],
            "passes_threshold": [True],
        }
    )
    elig = build_eligibility_table(metrics)

    assert len(elig) == 1
    row = elig.iloc[0]
    assert row["status"] == "exploratory -- upper bound", (
        f"omega_3 must be 'exploratory -- upper bound' regardless of r, "
        f"got '{row['status']}'"
    )
    # Must NOT be labelled power-eligible even though passes_threshold is True
    assert row["status"] != "power-eligible"


# ---------------------------------------------------------------------------
# build_eligibility_table — all 5 parameters
# ---------------------------------------------------------------------------


def test_build_eligibility_table_all_params() -> None:
    """Full 5-parameter metrics_df produces 5 rows with correct columns.

    Verifies the output schema: ['parameter', 'r', 'status', 'reason'].
    """
    metrics = pd.DataFrame(
        {
            "parameter": ["omega_2", "omega_3", "kappa", "beta", "zeta"],
            "r": [0.82, 0.67, 0.75, 0.80, 0.60],
            "passes_threshold": [True, False, True, True, False],
        }
    )
    elig = build_eligibility_table(metrics)

    assert len(elig) == 5, f"Expected 5 rows, got {len(elig)}"
    expected_cols = {"parameter", "r", "status", "reason"}
    assert set(elig.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {set(elig.columns)}"
    )

    # omega_3 must be exploratory despite passes_threshold=False — it's always overridden
    omega3_row = elig[elig["parameter"] == "omega_3"].iloc[0]
    assert omega3_row["status"] == "exploratory -- upper bound"

    # omega_2 should be power-eligible
    o2_row = elig[elig["parameter"] == "omega_2"].iloc[0]
    assert o2_row["status"] == "power-eligible"

    # zeta should be excluded
    zeta_row = elig[elig["parameter"] == "zeta"].iloc[0]
    assert zeta_row["status"] == "excluded"


# ---------------------------------------------------------------------------
# PrecheckResult dataclass
# ---------------------------------------------------------------------------


def test_precheck_result_dataclass() -> None:
    """PrecheckResult is frozen and has all expected fields."""
    empty_df = pd.DataFrame()
    result = PrecheckResult(
        metrics_df=empty_df,
        corr_df=empty_df,
        eligibility_df=empty_df,
        n_flagged=3,
        n_total=50,
        recovery_df=empty_df,
    )

    # All expected fields accessible
    assert result.n_flagged == 3
    assert result.n_total == 50
    assert isinstance(result.metrics_df, pd.DataFrame)
    assert isinstance(result.corr_df, pd.DataFrame)
    assert isinstance(result.eligibility_df, pd.DataFrame)
    assert isinstance(result.recovery_df, pd.DataFrame)

    # Frozen: mutation must raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        result.n_flagged = 99  # type: ignore[misc]
