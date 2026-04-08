#!/usr/bin/env python
"""Generate a structured power analysis recommendation report.

Reads pre-computed power CSV files (from ``09_aggregate_power.py`` and
``09_run_prechecks.py``) and produces ``results/power/recommendation.md``
containing:

- Concrete N/group and trial count recommendations
- P(BF > threshold) power summary table
- BMS discriminability table
- Eligible parameter table with recovery r and pass/fail status
- Exclusion rate summary
- omega_3 caveat and other caveats

Usage::

    python scripts/11_write_recommendation.py
    python scripts/11_write_recommendation.py --bf-threshold 6 --power-target 0.80
    python scripts/11_write_recommendation.py --output-dir /tmp/power_out
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Ensure project root is on sys.path so imports work without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config as _cfg
from prl_hgf.power.curves import compute_power_a, compute_power_b

# ---------------------------------------------------------------------------
# Core recommendation logic (pure function — testable without I/O)
# ---------------------------------------------------------------------------


def generate_recommendation(
    master_df: pd.DataFrame,
    power_a_df: pd.DataFrame,
    power_b_df: pd.DataFrame,
    precheck_sweep_df: pd.DataFrame | None,
    eligibility_df: pd.DataFrame | None,
    bf_threshold: float = 6.0,
    power_target: float = 0.80,
) -> str:
    """Generate a Markdown recommendation report from power analysis results.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full concatenated power results (all sweep types, N levels, iterations).
        Must contain columns: sweep_type, n_per_group, effect_size, bf_value,
        bf_exceeds, bms_correct, mean_rhat.
    power_a_df : pd.DataFrame
        P(BF > threshold) summary, one row per (n_per_group, effect_size).
        Columns: n_per_group, effect_size, p_bf_exceeds, n_iterations.
    power_b_df : pd.DataFrame
        P(correct BMS) summary, one row per n_per_group.
        Columns: n_per_group, p_bms_correct, n_iterations.
    precheck_sweep_df : pd.DataFrame or None
        Trial sweep recovery results.  If None, trial count section says
        "pending".  Expected columns include: trial_count, parameter, r.
    eligibility_df : pd.DataFrame or None
        Per-parameter eligibility table from precheck.  If None, the parameter
        table says "not available".  Expected columns: parameter, r, eligible.
    bf_threshold : float, optional
        Bayes factor threshold used for P(BF > threshold) computations
        (default 6.0).
    power_target : float, optional
        Minimum required power level (default 0.80).

    Returns
    -------
    str
        Formatted Markdown document as a single string.

    Notes
    -----
    The recommended N is the smallest n_per_group (d=0.5, sweep_type
    "did_postdose") where P(BF > bf_threshold) >= power_target.  If no
    crossing exists, the maximum N tested is reported.
    """
    sections: list[str] = []

    # ------------------------------------------------------------------
    # Section 1: Study Design Summary
    # ------------------------------------------------------------------
    sections.append(_section_study_design())

    # ------------------------------------------------------------------
    # Section 2: Recommended Sample Size
    # ------------------------------------------------------------------
    recommended_n, n_section = _section_sample_size(
        master_df, bf_threshold, power_target
    )
    sections.append(n_section)

    # ------------------------------------------------------------------
    # Section 3: Recommended Trial Count
    # ------------------------------------------------------------------
    sections.append(_section_trial_count(precheck_sweep_df))

    # ------------------------------------------------------------------
    # Section 4: Eligible Parameters
    # ------------------------------------------------------------------
    sections.append(_section_eligible_params(eligibility_df))

    # ------------------------------------------------------------------
    # Section 5: Power Summary Table — P(BF > threshold)
    # ------------------------------------------------------------------
    sections.append(_section_power_table(power_a_df, bf_threshold))

    # ------------------------------------------------------------------
    # Section 6: BMS Discriminability
    # ------------------------------------------------------------------
    sections.append(_section_bms_table(power_b_df))

    # ------------------------------------------------------------------
    # Section 7: Exclusion Rate
    # ------------------------------------------------------------------
    sections.append(_section_exclusion_rate(master_df))

    # ------------------------------------------------------------------
    # Section 8: Caveats
    # ------------------------------------------------------------------
    sections.append(_section_caveats(bf_threshold))

    # ------------------------------------------------------------------
    # Section 9: Evidence Generated
    # ------------------------------------------------------------------
    sections.append(_section_evidence(master_df, power_a_df))

    # Assemble final document
    header = textwrap.dedent(f"""\
        # Power Analysis Recommendation

        **Generated:** 2026-04-07
        **BF threshold:** BF > {bf_threshold:.0f}
        **Power target:** {power_target:.0%}

        ---
        """)
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _section_study_design() -> str:
    """Return Markdown for the Study Design Summary section."""
    return textwrap.dedent("""\
        ## 1. Study Design Summary

        | Item | Value |
        |------|-------|
        | Task | PRL pick_best_cue (3 cues, 4 phases: 2 acquisition + 2 reversal) |
        | Groups | Psilocybin vs. Placebo (both post-concussion syndrome) |
        | Sessions | 3 sessions per participant (baseline, post-dose, follow-up) |
        | Model | 3-level binary HGF with softmax + stickiness response model |
        | Primary parameter | omega_2 (tonic volatility / log-space learning rate) |
        | Primary contrast | Psilocybin vs. Placebo, did_postdose (DiD post-dose) |
        | Feedback type | Partial (only chosen cue updated per trial) |
        """)


def _section_sample_size(
    master_df: pd.DataFrame,
    bf_threshold: float,
    power_target: float,
) -> tuple[int | None, str]:
    """Return (recommended_n, markdown_section) for the sample-size section.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full power results.
    bf_threshold : float
        BF threshold for evidence declaration.
    power_target : float
        Minimum P(BF > threshold) to recommend N.

    Returns
    -------
    tuple[int | None, str]
        The recommended N (or None if not found) and the Markdown section.
    """
    # Filter to primary contrast, d = 0.5
    subset = master_df[
        (master_df["sweep_type"] == "did_postdose")
        & (master_df["effect_size"].round(3) == 0.5)
    ].copy()

    if subset.empty:
        section = textwrap.dedent("""\
            ## 2. Recommended Sample Size

            **No data available** for sweep_type='did_postdose' at d=0.5.
            Run the power sweep before generating this report.
            """)
        return None, section

    # Re-apply bf_threshold to bf_value for live computation
    subset = subset.copy()
    subset["_exceeds"] = subset["bf_value"] >= bf_threshold

    curve = (
        subset.groupby("n_per_group")["_exceeds"]
        .mean()
        .reset_index()
        .rename(columns={"_exceeds": "p_bf_exceeds"})
        .sort_values("n_per_group")
    )

    # Find first N where power >= target
    crossing = curve[curve["p_bf_exceeds"] >= power_target]
    if crossing.empty:
        recommended_n = int(curve["n_per_group"].max())
        crossing_note = (
            f"P(BF > {bf_threshold:.0f}) does not reach {power_target:.0%} "
            f"at any N tested. Maximum N tested: {recommended_n}/group. "
            f"Consider increasing the N range or using a lower threshold."
        )
    else:
        recommended_n = int(crossing["n_per_group"].min())
        achieved_power = float(crossing[crossing["n_per_group"] == recommended_n]["p_bf_exceeds"].iloc[0])
        crossing_note = (
            f"P(BF > {bf_threshold:.0f}) = **{achieved_power:.1%}** at N = {recommended_n}/group. "
            f"Power target {power_target:.0%} is achieved."
        )

    # Look up BMS power at recommended_n (from power_b / master)
    bms_rows = master_df[
        master_df["n_per_group"] == recommended_n
    ].drop_duplicates(subset=["n_per_group", "iteration"])
    p_bms = float(bms_rows["bms_correct"].mean()) if not bms_rows.empty else float("nan")

    section = textwrap.dedent(f"""\
        ## 2. Recommended Sample Size

        **Recommended N = {recommended_n} per group ({recommended_n * 2} total)**

        {crossing_note}

        At N = {recommended_n}/group, d = 0.5:

        | Metric | Value |
        |--------|-------|
        | P(BF > {bf_threshold:.0f}) | {curve[curve["n_per_group"] == recommended_n]["p_bf_exceeds"].iloc[0]:.1%} |
        | P(correct BMS) | {p_bms:.1%} |
        | Total participants | {recommended_n * 2} |
        """)
    return recommended_n, section


def _section_trial_count(precheck_sweep_df: pd.DataFrame | None) -> str:
    """Return Markdown for the Recommended Trial Count section.

    Parameters
    ----------
    precheck_sweep_df : pd.DataFrame or None
        Trial sweep DataFrame with columns: trial_count, parameter, r.
        omega_3 is excluded from the all-must-pass criterion.

    Returns
    -------
    str
        Markdown section.
    """
    if precheck_sweep_df is None:
        return textwrap.dedent("""\
            ## 3. Recommended Trial Count

            **Status: pending**

            Trial sweep results not found. Run ``09_run_prechecks.py --sweep``
            to generate ``trial_sweep_results.csv``.
            """)

    # Exclude omega_3 (exploratory) from the passing criterion
    non_omega3 = precheck_sweep_df[
        precheck_sweep_df["parameter"] != "omega_3"
    ].copy()

    if non_omega3.empty:
        return textwrap.dedent("""\
            ## 3. Recommended Trial Count

            **Status: pending**

            No non-omega_3 parameters found in trial sweep results.
            """)

    # Find minimum trial count where ALL non-omega_3 params pass r >= 0.7
    trial_counts = sorted(non_omega3["trial_count"].unique())
    min_passing = None
    for tc in trial_counts:
        tc_rows = non_omega3[non_omega3["trial_count"] == tc]
        if (tc_rows["r"] >= 0.7).all():
            min_passing = tc
            break

    if min_passing is None:
        max_tc = int(non_omega3["trial_count"].max())
        return textwrap.dedent(f"""\
            ## 3. Recommended Trial Count

            **No trial count tested achieves r >= 0.70 for all non-omega_3 parameters.**

            Maximum trial count tested: {max_tc}. Consider extending the sweep range.

            Note: omega_3 is excluded from this criterion (exploratory — upper bound only).
            """)

    return textwrap.dedent(f"""\
        ## 3. Recommended Trial Count

        **Recommended trial count: {min_passing} trials**

        This is the minimum trial count at which all non-omega_3 parameters
        achieve recovery r >= 0.70 (Pearson correlation between true and
        recovered parameter values).

        Note: omega_3 is excluded from this criterion (exploratory — see Caveats).
        """)


def _section_eligible_params(eligibility_df: pd.DataFrame | None) -> str:
    """Return Markdown for the Eligible Parameters section.

    Parameters
    ----------
    eligibility_df : pd.DataFrame or None
        Per-parameter eligibility table.  Expected columns: parameter, r,
        eligible (bool or str).

    Returns
    -------
    str
        Markdown section.
    """
    if eligibility_df is None:
        return textwrap.dedent("""\
            ## 4. Eligible Parameters

            **Status: not available**

            Run ``09_run_prechecks.py`` to generate ``power_eligible_params.csv``.
            """)

    # Build table
    rows_md = []
    for _, row in eligibility_df.iterrows():
        param = str(row["parameter"])
        r_val = float(row["r"])
        # Determine pass/fail: omega_3 is always exploratory
        if param == "omega_3":
            status = "EXPLORATORY"
        elif r_val >= 0.70:
            status = "PASS"
        else:
            status = "FAIL"
        rows_md.append(f"| {param} | {r_val:.3f} | {status} |")

    table = "\n".join(rows_md)
    return textwrap.dedent(f"""\
        ## 4. Eligible Parameters

        Parameters with recovery r >= 0.70 (Pearson) are included in the
        power analysis. omega_3 is always labelled EXPLORATORY regardless of r.

        | Parameter | Recovery r | Status |
        |-----------|-----------|--------|
        {table}

        Parameters marked FAIL are excluded from BFDA; their power estimates
        are unreliable and must not be used for confirmatory inference.
        """)


def _section_power_table(power_a_df: pd.DataFrame, bf_threshold: float) -> str:
    """Return Markdown for the P(BF > threshold) power summary table.

    Parameters
    ----------
    power_a_df : pd.DataFrame
        Power analysis A summary with columns: n_per_group, effect_size,
        p_bf_exceeds.
    bf_threshold : float
        BF threshold for the column header label.

    Returns
    -------
    str
        Markdown section with a formatted table.
    """
    if power_a_df.empty:
        return textwrap.dedent(f"""\
            ## 5. Power Summary Table — P(BF > {bf_threshold:.0f})

            No data available. Run ``09_aggregate_power.py`` first.
            """)

    # Pivot: rows = n_per_group, columns = effect_size
    effect_sizes = sorted(power_a_df["effect_size"].unique())
    n_levels = sorted(power_a_df["n_per_group"].unique())

    # Header
    d_headers = " | ".join(f"d={d:.1f}" for d in effect_sizes)
    header = f"| N/group | {d_headers} |"
    separator = "| ------- | " + " | ".join("------" for _ in effect_sizes) + " |"

    data_rows = []
    for n in n_levels:
        row_df = power_a_df[power_a_df["n_per_group"] == n]
        cells = []
        for d in effect_sizes:
            match = row_df[row_df["effect_size"].round(3) == round(d, 3)]
            if match.empty:
                cells.append("—")
            else:
                val = float(match["p_bf_exceeds"].iloc[0])
                cells.append(f"{val:.2f}")
        data_rows.append("| " + str(n) + " | " + " | ".join(cells) + " |")

    table = "\n".join([header, separator] + data_rows)
    return textwrap.dedent(f"""\
        ## 5. Power Summary Table — P(BF > {bf_threshold:.0f})

        Sweep type: did_postdose (psilocybin vs. placebo, DiD post-dose contrast).

        {table}

        Values are empirical proportions of iterations where BF > {bf_threshold:.0f}.
        """)


def _section_bms_table(power_b_df: pd.DataFrame) -> str:
    """Return Markdown for the BMS Discriminability section.

    Parameters
    ----------
    power_b_df : pd.DataFrame
        Power analysis B summary with columns: n_per_group, p_bms_correct,
        n_iterations.

    Returns
    -------
    str
        Markdown section with a formatted table.
    """
    if power_b_df.empty:
        return textwrap.dedent("""\
            ## 6. BMS Discriminability

            No data available. Run ``09_aggregate_power.py`` first.
            """)

    sorted_df = power_b_df.sort_values("n_per_group")
    rows_md = []
    for _, row in sorted_df.iterrows():
        n = int(row["n_per_group"])
        p = float(row["p_bms_correct"])
        iters = int(row["n_iterations"])
        rows_md.append(f"| {n} | {p:.2f} | {iters} |")

    table = "\n".join(rows_md)
    return textwrap.dedent(f"""\
        ## 6. BMS Discriminability — P(Correct Model Selected)

        Random-effects BMS (Rigoux et al. 2014) using exceedance probability
        as the decision criterion. P(correct) is the empirical probability that
        the true generative model achieves the highest exceedance probability.

        | N/group | P(correct BMS) | N iterations |
        | ------- | -------------- | ------------ |
        {table}
        """)


def _section_exclusion_rate(master_df: pd.DataFrame) -> str:
    """Return Markdown for the Exclusion Rate section.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full power results with a mean_rhat column.

    Returns
    -------
    str
        Markdown section reporting the fraction of rows with mean_rhat > 1.05.
    """
    if "mean_rhat" not in master_df.columns or master_df.empty:
        return textwrap.dedent("""\
            ## 7. Exclusion Rate

            MCMC convergence data not available.
            """)

    total = len(master_df)
    n_flagged = int((master_df["mean_rhat"] > 1.05).sum())
    rate = n_flagged / total if total > 0 else 0.0

    warning = ""
    if rate > 0.05:
        warning = (
            "\n> **Warning:** Exclusion rate exceeds 5%. Review MCMC settings "
            "(target_accept, n_tune, n_draws) or flag these cells for re-run.\n"
        )

    return textwrap.dedent(f"""\
        ## 7. Exclusion Rate

        Rows with mean Rhat > 1.05 are flagged as MCMC convergence failures
        and treated as exclusions (results from those fits are unreliable).

        | Metric | Value |
        |--------|-------|
        | Total fit rows | {total:,} |
        | Flagged (Rhat > 1.05) | {n_flagged:,} |
        | Exclusion rate | {rate:.1%} |
        {warning}
        """)


def _section_caveats(bf_threshold: float) -> str:
    """Return Markdown for the Caveats section.

    Parameters
    ----------
    bf_threshold : float
        BF threshold, used in the evidence threshold caveat.

    Returns
    -------
    str
        Markdown section.
    """
    return textwrap.dedent(f"""\
        ## 8. Caveats

        ### omega_3 (meta-volatility) — Exploratory Upper Bound

        omega_3 parameter recovery is known to be challenging with binary PRL
        data: reported recovery correlations are approximately r ~ 0.67 in the
        literature, and naive BFDA inflates power estimates by 20–40 percentage
        points compared to parameters with good recovery (r >= 0.85). All power
        estimates for omega_3 should be treated as **exploratory upper bounds**
        and must not be used for confirmatory sample-size decisions or
        pre-registration of omega_3-specific hypotheses.

        Primary confirmatory hypotheses are omega_2 and kappa.

        ### Effect Size Assumptions

        Simulations use omega_2 delta (psilocybin vs. placebo) calibrated to a
        standardised effect size of d = 0.3, 0.5, or 0.7. True population
        effect size is unknown. The d = 0.5 case is used as the primary planning
        assumption (medium effect, consistent with pilot data).

        ### Evidence Threshold

        BF > {bf_threshold:.0f} is used as the evidence threshold throughout this
        report. This is a conventional "moderate evidence" criterion (Jeffreys
        1961). The choice of threshold affects P(BF > threshold) non-linearly;
        verify sensitivity to threshold before pre-registering.

        ### MCMC Settings

        Power estimates are based on a fixed set of MCMC settings (n_tune,
        n_draws, target_accept). Performance on real data may differ if
        posterior geometry differs from the generative model.
        """)


def _section_evidence(
    master_df: pd.DataFrame,
    power_a_df: pd.DataFrame,
) -> str:
    """Return Markdown for the Evidence Generated section.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full power results.
    power_a_df : pd.DataFrame
        Power analysis A summary.

    Returns
    -------
    str
        Markdown section with counts.
    """
    total_fits = len(master_df)
    unique_cells = len(power_a_df) if not power_a_df.empty else 0

    if not master_df.empty:
        n_levels = master_df["n_per_group"].nunique()
        d_levels = master_df["effect_size"].nunique()
        sweep_types = master_df["sweep_type"].nunique()
        if "iteration" in master_df.columns:
            max_iter = int(master_df["iteration"].max()) + 1
        else:
            max_iter = 0
    else:
        n_levels = d_levels = sweep_types = max_iter = 0

    return textwrap.dedent(f"""\
        ## 9. Evidence Generated

        | Item | Count |
        |------|-------|
        | Total fit rows | {total_fits:,} |
        | Unique (N, d) cells | {unique_cells:,} |
        | N/group levels tested | {n_levels} |
        | Effect size levels tested | {d_levels} |
        | Sweep types | {sweep_types} |
        | Max iterations per cell | {max_iter} |

        All MCMC chains are saved to parquet for reproducibility. Raw results
        can be re-analysed with different BF thresholds using the
        ``power_master.csv`` file.
        """)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the recommendation script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: input_dir, precheck_dir,
        output_dir, bf_threshold, power_target.
    """
    default_power_dir = _cfg.RESULTS_DIR / "power"
    parser = argparse.ArgumentParser(
        description=(
            "Generate results/power/recommendation.md from pre-computed "
            "power CSV files. Reads power_master.csv, power_a_summary.csv, "
            "power_b_summary.csv, trial_sweep_results.csv, and "
            "power_eligible_params.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_power_dir,
        help="Directory containing power_master.csv, power_a_summary.csv, power_b_summary.csv.",
    )
    default_precheck_dir = _cfg.RESULTS_DIR / "power" / "prechecks"
    parser.add_argument(
        "--precheck-dir",
        type=Path,
        default=default_precheck_dir,
        help="Directory containing trial_sweep_results.csv and power_eligible_params.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_power_dir,
        help="Directory where recommendation.md will be written.",
    )
    parser.add_argument(
        "--bf-threshold",
        type=float,
        default=6.0,
        help="Bayes factor threshold for P(BF > threshold) computations.",
    )
    parser.add_argument(
        "--power-target",
        type=float,
        default=0.80,
        help="Minimum power level (proportion) to recommend N.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def _load_csv_or_none(path: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV file, returning None if missing or empty.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    label : str
        Human-readable label for warning messages.

    Returns
    -------
    pd.DataFrame or None
        Parsed DataFrame, or None if the file does not exist.
    """
    if not path.exists():
        print(f"[warn] {label} not found at {path} — section will say 'pending'.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[warn] {label} at {path} is empty — section will say 'pending'.")
        return None
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, read CSVs, generate recommendation.md.

    Raises
    ------
    SystemExit
        If power_master.csv is missing (no data to generate recommendation from).
    """
    args = parse_args()

    input_dir: Path = args.input_dir
    precheck_dir: Path = args.precheck_dir
    output_dir: Path = args.output_dir
    bf_threshold: float = args.bf_threshold
    power_target: float = args.power_target

    # Load required files
    master_path = input_dir / "power_master.csv"
    if not master_path.exists():
        print(
            f"[error] power_master.csv not found at {master_path}. "
            f"Run 09_aggregate_power.py first."
        )
        sys.exit(1)

    master_df = pd.read_csv(master_path)
    print(f"Loaded {len(master_df):,} rows from {master_path}")

    power_a_df = _load_csv_or_none(
        input_dir / "power_a_summary.csv", "power_a_summary.csv"
    )
    if power_a_df is None:
        # Recompute from master
        print("Recomputing power_a_summary from master_df...")
        power_a_df = compute_power_a(master_df, sweep_type="did_postdose")

    power_b_df = _load_csv_or_none(
        input_dir / "power_b_summary.csv", "power_b_summary.csv"
    )
    if power_b_df is None:
        print("Recomputing power_b_summary from master_df...")
        power_b_df = compute_power_b(master_df)

    precheck_sweep_df = _load_csv_or_none(
        precheck_dir / "trial_sweep_results.csv", "trial_sweep_results.csv"
    )
    eligibility_df = _load_csv_or_none(
        precheck_dir / "power_eligible_params.csv", "power_eligible_params.csv"
    )

    # Generate report
    report = generate_recommendation(
        master_df=master_df,
        power_a_df=power_a_df,
        power_b_df=power_b_df,
        precheck_sweep_df=precheck_sweep_df,
        eligibility_df=eligibility_df,
        bf_threshold=bf_threshold,
        power_target=power_target,
    )

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "recommendation.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote recommendation to {out_path}")
    print("\n" + "=" * 72)
    print(report)


if __name__ == "__main__":
    main()
