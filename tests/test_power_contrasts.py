"""Unit tests for power/contrasts.py — JZS BF and contrast extraction.

Tests cover:

- BF computation accuracy against pingouin reference
- Default Cauchy prior scale r = sqrt(2)/2
- NaN return for small samples
- Positive BF10 for large effect sizes
- JASP reference match to <1% relative error
- DiD contrast extraction from synthetic fit_df
- ValueError on missing parameter
- Linear trend contrast computation
- compute_all_contrasts structure and threshold logic

Run with::

    pytest tests/test_power_contrasts.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
from scipy import stats

from prl_hgf.power.contrasts import (
    compute_all_contrasts,
    compute_did_contrast,
    compute_jzs_bf,
    compute_linear_trend_contrast,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_fit_df(
    n_per_group: int = 10,
    effect_size: float = 0.5,
    noise_sd: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic fit_df matching the fit_batch output schema.

    Psilocybin post_dose has a shift of *effect_size* relative to baseline;
    placebo remains constant across sessions.  Followup is set to
    ``baseline + effect_size / 2`` for psilocybin to give a non-trivial
    linear trend.

    Parameters
    ----------
    n_per_group : int
        Number of participants per group.
    effect_size : float
        Shift applied to psilocybin post_dose mean.
    noise_sd : float
        Per-session Gaussian noise added to each mean.  Use ``noise_sd > 0``
        when within-group variance is needed (e.g. for BF tests).  Use
        ``noise_sd = 0`` when exact DiD values are needed for deterministic
        checks.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching ``fit_batch`` output.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    pid = 0

    for group in ("psilocybin", "placebo"):
        for _ in range(n_per_group):
            pid += 1
            baseline_mean = rng.normal(-3.0, 0.3)
            for session in ("baseline", "post_dose", "followup"):
                if group == "psilocybin" and session == "post_dose":
                    mean_val = baseline_mean + effect_size
                elif group == "psilocybin" and session == "followup":
                    mean_val = baseline_mean + effect_size / 2.0
                else:
                    mean_val = baseline_mean

                # Add per-session noise for realistic within-group variance
                mean_val += rng.normal(0, noise_sd) if noise_sd > 0 else 0.0

                rows.append(
                    {
                        "participant_id": f"P{pid:03d}",
                        "group": group,
                        "session": session,
                        "model": "hgf_3level",
                        "parameter": "omega_2",
                        "mean": mean_val,
                        "sd": 0.1,
                        "hdi_3%": mean_val - 0.2,
                        "hdi_97%": mean_val + 0.2,
                        "r_hat": 1.01,
                        "ess": 500.0,
                        "flagged": False,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Known t-stat — wrapper matches direct pingouin call
# ---------------------------------------------------------------------------


def test_compute_jzs_bf_known_t_stat() -> None:
    """compute_jzs_bf output matches direct pg.bayesfactor_ttest call.

    Constructs two arrays with a known separation, computes Welch t via
    scipy, then checks that the wrapper produces the same BF10 as calling
    pingouin directly (within machine-precision tolerance).
    """
    rng = np.random.default_rng(123)
    a = rng.normal(1.0, 1.0, size=30)
    b = rng.normal(0.0, 1.0, size=30)

    t_val, _ = stats.ttest_ind(a, b, equal_var=False)
    bf_ref = float(
        pg.bayesfactor_ttest(
            t=t_val, nx=30, ny=30, paired=False,
            alternative="two-sided", r=math.sqrt(2) / 2,
        )
    )

    bf_wrapper = compute_jzs_bf(a, b)
    assert bf_wrapper == pytest.approx(bf_ref, rel=1e-10)


# ---------------------------------------------------------------------------
# Test 2: Default r matches sqrt(2)/2
# ---------------------------------------------------------------------------


def test_compute_jzs_bf_default_r() -> None:
    """Default r parameter equals sqrt(2)/2.

    Verifies that omitting r gives the same BF as passing r explicitly.
    """
    rng = np.random.default_rng(999)
    a = rng.normal(2.0, 1.0, size=20)
    b = rng.normal(0.0, 1.0, size=20)

    bf_default = compute_jzs_bf(a, b)
    bf_explicit = compute_jzs_bf(a, b, r=math.sqrt(2) / 2)

    assert bf_default == pytest.approx(bf_explicit, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 3: NaN for small samples
# ---------------------------------------------------------------------------


def test_compute_jzs_bf_nan_on_small_sample() -> None:
    """Return NaN when either group has fewer than 2 observations."""
    single = np.array([1.0])
    ok = np.array([1.0, 2.0, 3.0])

    assert math.isnan(compute_jzs_bf(single, ok))
    assert math.isnan(compute_jzs_bf(ok, single))
    assert math.isnan(compute_jzs_bf(np.array([]), ok))


# ---------------------------------------------------------------------------
# Test 4: Positive BF for large effect
# ---------------------------------------------------------------------------


def test_compute_jzs_bf_positive_for_large_effect() -> None:
    """BF10 > 1.0 for clearly separated group means."""
    psi = np.array([5.0, 6.0, 7.0, 8.0])
    plc = np.array([1.0, 2.0, 3.0, 4.0])

    bf = compute_jzs_bf(psi, plc)
    assert bf > 1.0


# ---------------------------------------------------------------------------
# Test 5: JASP reference match — <1% relative error
# ---------------------------------------------------------------------------


def test_compute_jzs_bf_matches_jasp_reference() -> None:
    """Wrapper agrees with pingouin reference BF10 to <1% relative error.

    Uses t=2.5, nx=20, ny=20 as the reference case.  Constructs two
    arrays that yield approximately t=2.5 under a Welch test, then
    verifies the wrapper's BF10 agrees with the direct
    ``pg.bayesfactor_ttest(t=2.5, nx=20, ny=20)`` call to within 1%.
    """
    # Reference BF10 from pingouin (equivalent to JASP default settings)
    n = 20
    bf_reference = float(
        pg.bayesfactor_ttest(
            t=2.5, nx=n, ny=n, paired=False,
            alternative="two-sided", r=math.sqrt(2) / 2,
        )
    )
    assert math.isfinite(bf_reference)

    # Construct arrays whose Welch t-stat is approximately 2.5.
    # Standardize random draws to mean=0, sd=1, then shift group_a so that
    # the Welch t-statistic lands near 2.5.
    rng = np.random.default_rng(7777)
    group_a = rng.normal(0, 1, size=n)
    group_b = rng.normal(0, 1, size=n)
    group_a = (group_a - group_a.mean()) / group_a.std(ddof=1)
    group_b = (group_b - group_b.mean()) / group_b.std(ddof=1)
    se = math.sqrt(2.0 / n)
    group_a = group_a + 2.5 * se

    # Verify our construction produces approximately t=2.5
    t_check, _ = stats.ttest_ind(group_a, group_b, equal_var=False)
    assert abs(t_check - 2.5) < 0.15  # loose check on construction

    bf_wrapper = compute_jzs_bf(group_a, group_b)
    assert abs(bf_wrapper - bf_reference) / bf_reference < 0.01


# ---------------------------------------------------------------------------
# Test 6: DiD contrast produces correct group arrays
# ---------------------------------------------------------------------------


def test_compute_did_contrast_correct_groups() -> None:
    """DiD arrays match hand-computed session_a - session_b per participant."""
    fit_df = _make_test_fit_df(n_per_group=5, effect_size=1.0)

    psi_did, plc_did = compute_did_contrast(
        fit_df, "omega_2", session_a="post_dose", session_b="baseline"
    )

    # Psilocybin: post_dose = baseline + 1.0 => DiD should be ~1.0
    assert len(psi_did) == 5
    np.testing.assert_allclose(psi_did, 1.0, atol=1e-10)

    # Placebo: post_dose = baseline => DiD should be ~0.0
    assert len(plc_did) == 5
    np.testing.assert_allclose(plc_did, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 7: Missing parameter raises ValueError
# ---------------------------------------------------------------------------


def test_compute_did_contrast_missing_parameter_raises() -> None:
    """ValueError is raised when parameter is not in fit_df."""
    fit_df = _make_test_fit_df(n_per_group=3)

    with pytest.raises(ValueError, match="not_a_parameter"):
        compute_did_contrast(
            fit_df, "not_a_parameter",
            session_a="post_dose", session_b="baseline",
        )


# ---------------------------------------------------------------------------
# Test 8: Linear trend contrast
# ---------------------------------------------------------------------------


def test_compute_linear_trend_contrast() -> None:
    """Trend = -1*baseline + 0*post_dose + 1*followup per participant.

    For psilocybin: followup = baseline + 0.25, so trend = 0.25.
    For placebo: all sessions equal, so trend = 0.0.
    """
    fit_df = _make_test_fit_df(n_per_group=5, effect_size=0.5)

    psi_trend, plc_trend = compute_linear_trend_contrast(fit_df, "omega_2")

    # psilocybin: followup = baseline + 0.25 => trend = followup - baseline = 0.25
    assert len(psi_trend) == 5
    np.testing.assert_allclose(psi_trend, 0.25, atol=1e-10)

    # placebo: all sessions equal => trend = 0.0
    assert len(plc_trend) == 5
    np.testing.assert_allclose(plc_trend, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 9: compute_all_contrasts returns three dicts
# ---------------------------------------------------------------------------


def test_compute_all_contrasts_returns_three_dicts() -> None:
    """Result is a list of 3 dicts with correct keys and sweep_type values."""
    fit_df = _make_test_fit_df(n_per_group=10, effect_size=0.5, noise_sd=0.1)

    result = compute_all_contrasts(fit_df, parameter="omega_2")

    assert len(result) == 3

    expected_types = {"did_postdose", "did_followup", "linear_trend"}
    actual_types = {r["sweep_type"] for r in result}
    assert actual_types == expected_types

    for entry in result:
        assert "sweep_type" in entry
        assert "bf_value" in entry
        assert "bf_exceeds" in entry
        assert isinstance(entry["bf_value"], float)
        assert isinstance(entry["bf_exceeds"], bool)


# ---------------------------------------------------------------------------
# Test 10: bf_exceeds respects threshold
# ---------------------------------------------------------------------------


def test_compute_all_contrasts_bf_exceeds_threshold() -> None:
    """Primary contrast bf_exceeds is True with large effect and low threshold."""
    fit_df = _make_test_fit_df(n_per_group=20, effect_size=2.0, noise_sd=0.15)

    result = compute_all_contrasts(
        fit_df, parameter="omega_2", bf_threshold=1.0,
    )

    # With a large effect and low threshold, the primary contrast should exceed
    primary = next(r for r in result if r["sweep_type"] == "did_postdose")
    assert primary["bf_exceeds"] is True
    assert primary["bf_value"] > 1.0
