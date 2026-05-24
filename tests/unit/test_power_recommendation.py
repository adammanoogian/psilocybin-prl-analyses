"""Unit tests for scripts/11_write_recommendation.py.

Tests cover ``generate_recommendation()`` across a range of scenarios using
minimal synthetic DataFrames.  No file I/O is performed by any test — all
I/O paths are exercised by passing ``None`` or constructing DataFrames directly.

The script is imported via ``importlib.util`` because pipeline scripts live
outside the ``src`` package tree and are not importable as normal modules.

Run with::

    conda run -n ds_env python -m pytest tests/test_power_recommendation.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Import the script module via importlib
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "03_pre_analysis" / "07_write_recommendation.py"
)

spec = importlib.util.spec_from_file_location("write_rec", str(_SCRIPT_PATH))
assert spec is not None and spec.loader is not None, (
    f"Could not build importlib spec for {_SCRIPT_PATH}. "
    f"Ensure the file exists at {_SCRIPT_PATH}."
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]

generate_recommendation = mod.generate_recommendation


# ---------------------------------------------------------------------------
# DataFrame fixture helpers
# ---------------------------------------------------------------------------


def _make_master_row(
    sweep_type: str = "did_postdose",
    effect_size: float = 0.5,
    n_per_group: int = 20,
    iteration: int = 0,
    bf_value: float = 12.0,
    bf_exceeds: bool = True,
    bms_correct: bool = True,
    mean_rhat: float = 1.01,
) -> dict[str, Any]:
    """Return one row matching the POWER_SCHEMA layout for master_df."""
    return {
        "sweep_type": sweep_type,
        "effect_size": effect_size,
        "n_per_group": n_per_group,
        "trial_count": 200,
        "iteration": iteration,
        "parameter": "omega_2",
        "bf_value": bf_value,
        "bf_exceeds": bf_exceeds,
        "bms_xp": 0.9 if bms_correct else 0.3,
        "bms_correct": bms_correct,
        "recovery_r": 0.85,
        "n_divergences": 0,
        "mean_rhat": mean_rhat,
    }


def _make_power_a_df(
    n_levels: list[int] | None = None,
    effect_sizes: list[float] | None = None,
    p_value: float = 0.80,
) -> pd.DataFrame:
    """Return a minimal power_a_df with specified N levels and effect sizes.

    Parameters
    ----------
    n_levels : list[int] or None
        N/group values to include (default: [20, 25, 30]).
    effect_sizes : list[float] or None
        Effect size values (default: [0.3, 0.5, 0.7]).
    p_value : float
        Uniform p_bf_exceeds value for all cells (default: 0.80).

    Returns
    -------
    pd.DataFrame
        Power analysis A summary DataFrame.
    """
    if n_levels is None:
        n_levels = [20, 25, 30]
    if effect_sizes is None:
        effect_sizes = [0.3, 0.5, 0.7]

    rows = []
    for n in n_levels:
        for d in effect_sizes:
            rows.append(
                {
                    "n_per_group": n,
                    "effect_size": d,
                    "p_bf_exceeds": p_value,
                    "n_iterations": 100,
                }
            )
    return pd.DataFrame(rows)


def _make_power_b_df(n_levels: list[int] | None = None, p_value: float = 0.85) -> pd.DataFrame:
    """Return a minimal power_b_df.

    Parameters
    ----------
    n_levels : list[int] or None
        N/group values (default: [20, 25, 30]).
    p_value : float
        Uniform p_bms_correct value (default: 0.85).

    Returns
    -------
    pd.DataFrame
        Power analysis B summary DataFrame.
    """
    if n_levels is None:
        n_levels = [20, 25, 30]
    rows = [
        {"n_per_group": n, "p_bms_correct": p_value, "n_iterations": 100}
        for n in n_levels
    ]
    return pd.DataFrame(rows)


def _make_master_df(
    n_levels: list[int] | None = None,
    p_exceed: float = 0.85,
    n_iterations: int = 10,
) -> pd.DataFrame:
    """Return a master_df with rows at multiple N and effect sizes.

    Parameters
    ----------
    n_levels : list[int] or None
        N/group values (default: [20, 25, 30]).
    p_exceed : float
        Fraction of iterations where bf_value exceeds threshold (default: 0.85).
    n_iterations : int
        Number of iterations per (N, d) cell (default: 10).

    Returns
    -------
    pd.DataFrame
        Synthetic master DataFrame.
    """
    if n_levels is None:
        n_levels = [20, 25, 30]

    rows = []
    for n in n_levels:
        for d in [0.3, 0.5, 0.7]:
            for i in range(n_iterations):
                exceeds = i < int(n_iterations * p_exceed)
                rows.append(
                    _make_master_row(
                        sweep_type="did_postdose",
                        effect_size=d,
                        n_per_group=n,
                        iteration=i,
                        bf_value=12.0 if exceeds else 2.0,
                        bf_exceeds=exceeds,
                        bms_correct=exceeds,
                    )
                )
    return pd.DataFrame(rows)


def _make_precheck_df(
    trial_counts: list[int] | None = None,
    params: list[str] | None = None,
    passing_tc: int = 250,
) -> pd.DataFrame:
    """Return a minimal trial sweep DataFrame.

    Parameters
    ----------
    trial_counts : list[int] or None
        Trial counts to include (default: [150, 200, 250, 300]).
    params : list[str] or None
        Parameter names (default: ["omega_2", "kappa", "omega_3"]).
    passing_tc : int
        Minimum trial count where all non-omega_3 params reach r=0.75 (default: 250).

    Returns
    -------
    pd.DataFrame
        Synthetic trial sweep DataFrame.
    """
    if trial_counts is None:
        trial_counts = [150, 200, 250, 300]
    if params is None:
        params = ["omega_2", "kappa", "omega_3"]

    rows = []
    for tc in trial_counts:
        for param in params:
            if param == "omega_3":
                r = 0.67  # always below 0.7 — excluded
            elif tc >= passing_tc:
                r = 0.75  # passes
            else:
                r = 0.60  # fails
            rows.append({"trial_count": tc, "parameter": param, "r": r})
    return pd.DataFrame(rows)


def _make_eligibility_df(include_omega3: bool = True) -> pd.DataFrame:
    """Return a minimal eligibility DataFrame.

    Parameters
    ----------
    include_omega3 : bool
        Whether to include omega_3 row (default: True).

    Returns
    -------
    pd.DataFrame
        Eligibility DataFrame with parameter, r, eligible columns.
    """
    rows: list[dict[str, Any]] = [
        {"parameter": "omega_2", "r": 0.88, "eligible": True},
        {"parameter": "kappa", "r": 0.81, "eligible": True},
    ]
    if include_omega3:
        rows.append({"parameter": "omega_3", "r": 0.67, "eligible": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recommendation_contains_n_per_group() -> None:
    """generate_recommendation reports 'N = 25 per group' when d=0.5 reaches 80% at N=25.

    Constructs master_df where N=20 has p=0.70 (below target) and N=25
    has p=0.90 (above target).  The function must select N=25 as the
    minimum N meeting the 80% power target and include that value in output.
    """
    rows: list[dict[str, Any]] = []
    for i in range(10):
        # N=20: 7/10 succeed -> p=0.70 < 0.80
        rows.append(
            _make_master_row(
                n_per_group=20,
                effect_size=0.5,
                iteration=i,
                bf_value=12.0 if i < 7 else 2.0,
                bf_exceeds=(i < 7),
            )
        )
        # N=25: 9/10 succeed -> p=0.90 >= 0.80
        rows.append(
            _make_master_row(
                n_per_group=25,
                effect_size=0.5,
                iteration=i,
                bf_value=12.0 if i < 9 else 2.0,
                bf_exceeds=(i < 9),
            )
        )

    master_df = pd.DataFrame(rows)
    power_a_df = compute_power_a_from_master(master_df)
    power_b_df = _make_power_b_df([20, 25])

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    assert "N = 25 per group" in report, (
        f"Expected 'N = 25 per group' in report. "
        f"First 1000 chars: {report[:1000]}"
    )


def compute_power_a_from_master(master_df: pd.DataFrame) -> pd.DataFrame:
    """Helper: compute power_a summary directly from a master DataFrame.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full power results.

    Returns
    -------
    pd.DataFrame
        Power analysis A summary.
    """
    subset = master_df[master_df["sweep_type"] == "did_postdose"].copy()
    subset["_exceeds"] = subset["bf_value"] >= 6.0
    return (
        subset.groupby(["n_per_group", "effect_size"])["_exceeds"]
        .agg(p_bf_exceeds="mean", n_iterations="count")
        .reset_index()
        .rename(columns={"_exceeds": "p_bf_exceeds"})
    )


def test_recommendation_no_crossing() -> None:
    """generate_recommendation notes that threshold is not reached when no N qualifies.

    All N levels have p=0.60 < 0.80, so the function must not report a clean
    crossing and should note that the threshold is not reached.
    """
    rows: list[dict[str, Any]] = []
    for n in [20, 25, 30]:
        for i in range(10):
            rows.append(
                _make_master_row(
                    n_per_group=n,
                    effect_size=0.5,
                    iteration=i,
                    bf_value=12.0 if i < 6 else 2.0,
                    bf_exceeds=(i < 6),
                )
            )

    master_df = pd.DataFrame(rows)
    power_a_df = compute_power_a_from_master(master_df)
    power_b_df = _make_power_b_df([20, 25, 30])

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    # "does not" should appear in the crossing note
    assert "does not" in report.lower(), (
        f"Expected 'does not' in report when no N crosses the power target. "
        f"Report excerpt: {report[:1500]}"
    )


def test_recommendation_contains_trial_count() -> None:
    """generate_recommendation reports '250' as the minimum passing trial count.

    precheck_sweep_df has params passing at trial_count=250 but failing below.
    """
    precheck_df = _make_precheck_df(
        trial_counts=[150, 200, 250, 300],
        params=["omega_2", "kappa", "omega_3"],
        passing_tc=250,
    )
    master_df = _make_master_df()
    power_a_df = _make_power_a_df()
    power_b_df = _make_power_b_df()

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=precheck_df,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    assert "250" in report, (
        f"Expected '250' in report for min passing trial count. "
        f"Relevant section: {report[report.find('Trial'):report.find('Trial') + 500]}"
    )


def test_recommendation_contains_omega3_caveat() -> None:
    """generate_recommendation includes omega_3 exploratory caveat.

    The caveats section must contain both 'exploratory' and 'omega_3'.
    """
    master_df = _make_master_df()
    power_a_df = _make_power_a_df()
    power_b_df = _make_power_b_df()

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    report_lower = report.lower()
    assert "exploratory" in report_lower, (
        "Expected 'exploratory' in report for omega_3 caveat."
    )
    assert "omega_3" in report_lower, (
        "Expected 'omega_3' in report for omega_3 caveat."
    )


def test_recommendation_contains_summary_table() -> None:
    """generate_recommendation includes a Markdown table with pipe delimiters.

    The power summary table section must contain '|' characters indicating
    a formatted table.
    """
    master_df = _make_master_df()
    power_a_df = _make_power_a_df(
        n_levels=[20, 30],
        effect_sizes=[0.3, 0.5, 0.7],
    )
    power_b_df = _make_power_b_df([20, 30])

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    # Count pipe characters — tables always have many
    pipe_count = report.count("|")
    assert pipe_count > 10, (
        f"Expected many '|' characters for Markdown tables, got {pipe_count}. "
        f"Report may be missing tables."
    )


def test_recommendation_exclusion_rate() -> None:
    """generate_recommendation reports exclusion rate with correct fraction.

    3 of 10 rows have mean_rhat > 1.05 -> 30% exclusion rate.
    The exclusion rate section must contain a percentage string.
    """
    rows: list[dict[str, Any]] = []
    for i in range(10):
        bad_rhat = i < 3  # 3 out of 10 flagged
        rows.append(
            _make_master_row(
                n_per_group=20,
                effect_size=0.5,
                iteration=i,
                mean_rhat=1.10 if bad_rhat else 1.01,
            )
        )

    master_df = pd.DataFrame(rows)
    power_a_df = compute_power_a_from_master(master_df)
    power_b_df = _make_power_b_df([20])

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    # 3/10 = 30.0%
    assert "30.0%" in report, (
        f"Expected '30.0%' exclusion rate in report. "
        f"Exclusion section: {report[report.find('Exclusion'):report.find('Exclusion') + 400]}"
    )


def test_recommendation_handles_missing_precheck() -> None:
    """generate_recommendation handles None precheck_sweep_df and eligibility_df gracefully.

    When both optional inputs are None, the report must contain 'pending'
    (for trial count) and 'not available' (for eligibility).
    """
    master_df = _make_master_df()
    power_a_df = _make_power_a_df()
    power_b_df = _make_power_b_df()

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=6.0,
        power_target=0.80,
    )

    report_lower = report.lower()
    assert "pending" in report_lower, (
        "Expected 'pending' when precheck_sweep_df is None."
    )
    assert "not available" in report_lower, (
        "Expected 'not available' when eligibility_df is None."
    )


def test_recommendation_bf_threshold_reflected() -> None:
    """generate_recommendation reflects non-default bf_threshold in the report.

    With bf_threshold=3.0, the report must contain 'BF > 3'.
    """
    master_df = _make_master_df()
    power_a_df = _make_power_a_df()
    power_b_df = _make_power_b_df()

    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=None,
        eligibility_df=None,
        bf_threshold=3.0,
        power_target=0.80,
    )

    assert "BF > 3" in report, (
        f"Expected 'BF > 3' in report when bf_threshold=3.0. "
        f"Report header: {report[:500]}"
    )
