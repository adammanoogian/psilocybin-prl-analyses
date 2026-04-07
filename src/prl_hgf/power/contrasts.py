"""JZS Bayes Factor computation and contrast extraction for BFDA power analysis.

Provides functions to compute difference-in-differences (DiD) contrasts from
fitted parameter DataFrames and wrap them in JZS Bayes Factor tests via
``pingouin.bayesfactor_ttest``.

Three contrast types are supported:

- **did_postdose**: post_dose minus baseline (primary contrast)
- **did_followup**: followup minus baseline
- **linear_trend**: weighted linear contrast [-1, 0, +1] across sessions

Notes
-----
The default Cauchy prior scale ``r = sqrt(2)/2`` matches the JASP default for
two-sample JZS BF tests (Rouder et al., 2009).  ``alternative='two-sided'``
is always used because ``pingouin.bayesfactor_ttest`` only supports two-sided
JZS BF.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


def compute_jzs_bf(
    did_psi: np.ndarray,
    did_plc: np.ndarray,
    r: float = math.sqrt(2) / 2,
) -> float:
    """Compute JZS BF10 from per-participant DiD vectors for two groups.

    Runs a Welch t-test on the two arrays, then passes the t-statistic to
    ``pingouin.bayesfactor_ttest`` to obtain the JZS Bayes Factor.

    Parameters
    ----------
    did_psi : np.ndarray
        1-D array of per-participant difference-in-differences values for the
        psilocybin group.
    did_plc : np.ndarray
        1-D array of per-participant difference-in-differences values for the
        placebo group.
    r : float, optional
        Cauchy prior scale for the JZS BF.  Default ``sqrt(2)/2`` matches the
        JASP default.

    Returns
    -------
    float
        BF10 value.  Returns ``float('nan')`` when either group has fewer than
        2 observations.
    """
    if len(did_psi) < 2 or len(did_plc) < 2:
        return float("nan")

    t_val, _ = stats.ttest_ind(did_psi, did_plc, equal_var=False)
    bf10 = pg.bayesfactor_ttest(
        t=t_val,
        nx=len(did_psi),
        ny=len(did_plc),
        paired=False,
        alternative="two-sided",
        r=r,
    )
    return float(bf10)


def compute_did_contrast(
    fit_df: pd.DataFrame,
    parameter: str,
    session_a: str,
    session_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-participant DiD vectors for psilocybin and placebo groups.

    Filters ``fit_df`` to the requested *parameter*, pivots to wide form, and
    computes ``DiD = session_a - session_b`` per participant.

    Parameters
    ----------
    fit_df : pd.DataFrame
        Output of ``fit_batch`` with columns: participant_id, group, session,
        model, parameter, mean, sd, hdi_3%, hdi_97%, r_hat, ess, flagged.
    parameter : str
        Name of the model parameter to extract (e.g. ``"omega_2"``).
    session_a : str
        Session label for the numerator (e.g. ``"post_dose"``).
    session_b : str
        Session label for the denominator (e.g. ``"baseline"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(psi_did, plc_did)`` — 1-D arrays of DiD values for the psilocybin
        and placebo groups respectively.

    Raises
    ------
    ValueError
        If *parameter* is not found in ``fit_df["parameter"]``.
    """
    available = fit_df["parameter"].unique()
    if parameter not in available:
        msg = (
            f"Parameter {parameter!r} not found in fit_df. "
            f"Available: {sorted(available)}"
        )
        raise ValueError(msg)

    param_df = fit_df[fit_df["parameter"] == parameter]
    wide = (
        param_df.pivot_table(
            index=["participant_id", "group"],
            columns="session",
            values="mean",
        )
        .reset_index()
    )
    wide["did"] = wide[session_a] - wide[session_b]

    psi_did = (
        wide.loc[wide["group"] == "psilocybin", "did"].dropna().to_numpy()
    )
    plc_did = (
        wide.loc[wide["group"] == "placebo", "did"].dropna().to_numpy()
    )
    return psi_did, plc_did


def compute_linear_trend_contrast(
    fit_df: pd.DataFrame,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a linear trend contrast across the three sessions.

    The contrast weights are ``[-1, 0, +1]`` applied to
    ``[baseline, post_dose, followup]`` per participant, yielding a scalar
    trend score.  This is then split by group.

    Parameters
    ----------
    fit_df : pd.DataFrame
        Output of ``fit_batch`` (same schema as :func:`compute_did_contrast`).
    parameter : str
        Name of the model parameter to extract.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(psi_trend, plc_trend)`` — 1-D arrays of linear trend scores for
        the psilocybin and placebo groups respectively.

    Raises
    ------
    ValueError
        If *parameter* is not found in ``fit_df["parameter"]``.
    """
    available = fit_df["parameter"].unique()
    if parameter not in available:
        msg = (
            f"Parameter {parameter!r} not found in fit_df. "
            f"Available: {sorted(available)}"
        )
        raise ValueError(msg)

    param_df = fit_df[fit_df["parameter"] == parameter]
    wide = (
        param_df.pivot_table(
            index=["participant_id", "group"],
            columns="session",
            values="mean",
        )
        .reset_index()
    )
    wide["trend"] = (
        -1.0 * wide["baseline"]
        + 0.0 * wide["post_dose"]
        + 1.0 * wide["followup"]
    )

    psi_trend = (
        wide.loc[wide["group"] == "psilocybin", "trend"].dropna().to_numpy()
    )
    plc_trend = (
        wide.loc[wide["group"] == "placebo", "trend"].dropna().to_numpy()
    )
    return psi_trend, plc_trend


def compute_all_contrasts(
    fit_df: pd.DataFrame,
    parameter: str = "omega_2",
    bf_threshold: float = 10.0,
) -> list[dict]:
    """Compute JZS BF10 for all three contrast types.

    Runs :func:`compute_did_contrast` for the *did_postdose* and
    *did_followup* contrasts and :func:`compute_linear_trend_contrast` for
    the *linear_trend* contrast.  Each result is passed through
    :func:`compute_jzs_bf`.

    Parameters
    ----------
    fit_df : pd.DataFrame
        Output of ``fit_batch``.
    parameter : str, optional
        Model parameter to test.  Default ``"omega_2"``.
    bf_threshold : float, optional
        BF10 threshold for the ``bf_exceeds`` flag.  Default ``10.0``.

    Returns
    -------
    list[dict]
        Three dicts with keys ``sweep_type`` (str), ``bf_value`` (float),
        and ``bf_exceeds`` (bool).  ``sweep_type`` values are
        ``"did_postdose"``, ``"did_followup"``, ``"linear_trend"``.
    """
    results: list[dict] = []

    # (a) did_postdose: post_dose - baseline
    psi_post, plc_post = compute_did_contrast(
        fit_df, parameter, session_a="post_dose", session_b="baseline"
    )
    bf_post = compute_jzs_bf(psi_post, plc_post)
    results.append(
        {
            "sweep_type": "did_postdose",
            "bf_value": bf_post,
            "bf_exceeds": bf_post > bf_threshold,
        }
    )

    # (b) did_followup: followup - baseline
    psi_fu, plc_fu = compute_did_contrast(
        fit_df, parameter, session_a="followup", session_b="baseline"
    )
    bf_fu = compute_jzs_bf(psi_fu, plc_fu)
    results.append(
        {
            "sweep_type": "did_followup",
            "bf_value": bf_fu,
            "bf_exceeds": bf_fu > bf_threshold,
        }
    )

    # (c) linear_trend: [-1, 0, +1] across sessions
    psi_trend, plc_trend = compute_linear_trend_contrast(fit_df, parameter)
    bf_trend = compute_jzs_bf(psi_trend, plc_trend)
    results.append(
        {
            "sweep_type": "linear_trend",
            "bf_value": bf_trend,
            "bf_exceeds": bf_trend > bf_threshold,
        }
    )

    return results
