"""Aggregate Phase 32 sampler audit results into a summary CSV and report.

Reads individual audit result JSON files from
``models/power/audit_results/``, builds a summary DataFrame, writes a CSV,
and prints analysis including per-backend ESS/sec, divergent rates,
noise-floor variance, and head-to-head ESS/sec ratios.

Usage
-----
    python scripts/03_pre_analysis/16_aggregate_audit32.py
    python scripts/03_pre_analysis/16_aggregate_audit32.py --output-csv out.csv
    python scripts/03_pre_analysis/16_aggregate_audit32.py --results-dir path/to/jsons
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = Path("models/power/audit_results")

# Columns for the summary CSV (matches AUDIT_PROTOCOL.md Section 5)
CSV_COLUMNS = [
    "backend",
    "model",
    "n_participants",
    "mass_matrix",
    "seed",
    "job_id",
    "walltime_s",
    "compile_time_s",
    "ess_bulk_min",
    "ess_per_sec",
    "ess_per_grad_eval",
    "divergent_count",
    "divergent_rate",
    "rhat_max",
    "recovery_corr_omega2",
    "recovery_corr_beta",
    "recovery_corr_zeta",
    "memory_peak_mb",
    "status",
]

# AUDIT-04 threshold: minimum non-CRASH/TIMEOUT fits per backend
AUDIT_04_THRESHOLD = 30


def load_audit_results(results_dir: Path) -> pd.DataFrame:
    """Load all audit JSON files into a DataFrame.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``audit_*.json`` result files.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per audit result, columns per
        ``CSV_COLUMNS``. Empty DataFrame (with correct columns) if no
        files found.
    """
    json_files = sorted(results_dir.glob("audit_*.json"))

    if not json_files:
        return pd.DataFrame(columns=CSV_COLUMNS)

    records: list[dict] = []
    for jf in json_files:
        with jf.open() as f:
            data = json.load(f)
        # Extract only the columns we need (JSON may have extra fields)
        record = {col: data.get(col) for col in CSV_COLUMNS}
        # Map n_participants from JSON (key is "n_participants")
        if record["n_participants"] is None:
            record["n_participants"] = data.get("p_total")
        records.append(record)

    return pd.DataFrame(records, columns=CSV_COLUMNS)


def print_per_backend_ess(df: pd.DataFrame) -> None:
    """Print per-backend mean ESS/sec across PASS cells.

    Parameters
    ----------
    df : pd.DataFrame
        Audit summary DataFrame.
    """
    pass_df = df[df["status"] == "PASS"]
    if pass_df.empty:
        print("  No PASS cells; cannot compute per-backend ESS/sec.")
        return

    grouped = pass_df.groupby("backend")["ess_per_sec"]
    print(f"  {'Backend':<12} | {'Mean ESS/sec':>14} | {'Median':>10} | {'N':>3}")
    print(f"  {'-' * 50}")
    for backend, group in grouped:
        vals = group.dropna()
        if vals.empty:
            print(f"  {backend:<12} | {'--':>14} | {'--':>10} | {0:>3}")
        else:
            print(
                f"  {backend:<12} | {vals.mean():>14.2f} | "
                f"{vals.median():>10.2f} | {len(vals):>3}"
            )


def print_per_backend_divergent_rate(df: pd.DataFrame) -> None:
    """Print per-backend divergent rate across all completed cells.

    Parameters
    ----------
    df : pd.DataFrame
        Audit summary DataFrame.
    """
    completed = df[df["status"].isin(["PASS", "DIVERGENT", "INVALID"])]
    if completed.empty:
        print("  No completed cells; cannot compute divergent rates.")
        return

    grouped = completed.groupby("backend")["divergent_rate"]
    print(f"  {'Backend':<12} | {'Mean div%':>10} | {'Max div%':>10} | {'N':>3}")
    print(f"  {'-' * 45}")
    for backend, group in grouped:
        vals = group.dropna()
        if vals.empty:
            print(f"  {backend:<12} | {'--':>10} | {'--':>10} | {0:>3}")
        else:
            print(
                f"  {backend:<12} | {vals.mean():>10.4%} | "
                f"{vals.max():>10.4%} | {len(vals):>3}"
            )


def print_noise_floor(df: pd.DataFrame) -> None:
    """Print noise-floor ESS variance from A-vs-A seed runs.

    Noise-floor runs are identified as rows where the same
    (backend, model, n_participants, mass_matrix) tuple has multiple
    distinct seeds. The noise floor is the standard deviation of
    ESS/sec within each such group.

    Parameters
    ----------
    df : pd.DataFrame
        Audit summary DataFrame.
    """
    group_cols = ["backend", "model", "n_participants", "mass_matrix"]
    # Find groups with multiple seeds
    seed_counts = df.groupby(group_cols)["seed"].nunique()
    multi_seed = seed_counts[seed_counts > 1].index

    if multi_seed.empty:
        print("  No A-vs-A noise-floor runs detected (need multiple seeds "
              "with same config).")
        return

    print(f"  {'Backend':<10} | {'Model':<12} | {'P':>4} | "
          f"{'ESS/sec std':>12} | {'ESS/sec vals':>20}")
    print(f"  {'-' * 70}")

    for key in multi_seed:
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, key, strict=True):
            mask &= df[col] == val

        group = df[mask]
        ess_vals = group["ess_per_sec"].dropna()

        if len(ess_vals) < 2:
            continue

        std_val = float(np.std(ess_vals, ddof=1))
        vals_str = ", ".join(f"{v:.2f}" for v in ess_vals)
        backend, model, n_part, _ = key
        print(
            f"  {backend:<10} | {model:<12} | {n_part:>4} | "
            f"{std_val:>12.4f} | [{vals_str}]"
        )


def print_head_to_head(df: pd.DataFrame) -> None:
    """Print head-to-head ESS/sec ratio (BlackJAX / NumPyro) per cell.

    Matches cells by (model, n_participants, mass_matrix, seed) and
    computes the ratio of BlackJAX ESS/sec to NumPyro ESS/sec.

    Parameters
    ----------
    df : pd.DataFrame
        Audit summary DataFrame.
    """
    merge_cols = ["model", "n_participants", "mass_matrix", "seed"]

    bjx = df[df["backend"] == "blackjax"].set_index(merge_cols)
    npy = df[df["backend"] == "numpyro"].set_index(merge_cols)

    common = bjx.index.intersection(npy.index)
    if common.empty:
        print("  No matching head-to-head cells between backends.")
        return

    print(
        f"  {'Model':<12} | {'P':>4} | {'Mass':>9} | {'Seed':>5} | "
        f"{'BJX ESS/s':>10} | {'NPy ESS/s':>10} | {'Ratio BJX/NPy':>15}"
    )
    print(f"  {'-' * 80}")

    ratios: list[float] = []
    for key in sorted(common):
        bjx_ess = bjx.loc[key, "ess_per_sec"]
        npy_ess = npy.loc[key, "ess_per_sec"]

        # Handle potential duplicate index (take first)
        if isinstance(bjx_ess, pd.Series):
            bjx_ess = bjx_ess.iloc[0]
        if isinstance(npy_ess, pd.Series):
            npy_ess = npy_ess.iloc[0]

        if pd.isna(bjx_ess) or pd.isna(npy_ess) or npy_ess == 0:
            ratio_str = "--"
        else:
            ratio = bjx_ess / npy_ess
            ratios.append(ratio)
            ratio_str = f"{ratio:.3f}x"

        model, n_part, mass, seed = key
        bjx_str = f"{bjx_ess:.2f}" if not pd.isna(bjx_ess) else "--"
        npy_str = f"{npy_ess:.2f}" if not pd.isna(npy_ess) else "--"

        print(
            f"  {model:<12} | {n_part:>4} | {mass:>9} | {seed:>5} | "
            f"{bjx_str:>10} | {npy_str:>10} | {ratio_str:>15}"
        )

    if ratios:
        median_ratio = float(np.median(ratios))
        print(f"\n  Median ESS/sec ratio (BlackJAX / NumPyro): {median_ratio:.3f}x")
        n_bjx_wins = sum(1 for r in ratios if r > 1.0)
        print(
            f"  BlackJAX faster in {n_bjx_wins}/{len(ratios)} cells "
            f"({n_bjx_wins / len(ratios):.0%})"
        )


def print_audit04_check(df: pd.DataFrame) -> None:
    """Check AUDIT-04 threshold: >= 30 fits per backend with valid status.

    Parameters
    ----------
    df : pd.DataFrame
        Audit summary DataFrame.
    """
    valid = df[~df["status"].isin(["CRASH", "TIMEOUT"])]
    per_backend = valid.groupby("backend").size()

    print(f"  AUDIT-04 threshold: >= {AUDIT_04_THRESHOLD} valid fits per backend")
    print(f"  {'-' * 40}")
    for backend in ["blackjax", "numpyro"]:
        count = per_backend.get(backend, 0)
        passed = count >= AUDIT_04_THRESHOLD
        mark = "PASS" if passed else "FAIL"
        print(f"  {backend:<12}: {count:>3} valid fits  [{mark}]")


def main() -> int:
    """Aggregate Phase 32 audit results and print analysis.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Phase 32 sampler audit JSON results into "
            "a summary CSV and print analysis."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing audit_*.json result files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Path for summary CSV output. Default: "
            "{results_dir}/audit_summary.csv"
        ),
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_csv: Path | None = args.output_csv

    if output_csv is None:
        output_csv = results_dir / "audit_summary.csv"

    # ------------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------------
    if not results_dir.exists():
        print(
            "No audit results found. Submit cluster job first:\n"
            "  sbatch cluster/32_sampler_audit_gpu.slurm"
        )
        return 0

    df = load_audit_results(results_dir)

    if df.empty:
        print(
            "No audit results found. Submit cluster job first:\n"
            "  sbatch cluster/32_sampler_audit_gpu.slurm"
        )
        return 0

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Summary CSV written: {output_csv}")
    print(f"Total results loaded: {len(df)}")

    # ------------------------------------------------------------------
    # Status counts
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STATUS COUNTS")
    print(f"{'=' * 60}")
    counts = df["status"].value_counts()
    for status in ["PASS", "DIVERGENT", "INVALID", "TIMEOUT", "CRASH"]:
        print(f"  {status:<12}: {counts.get(status, 0):>3}")
    print(f"  {'─' * 30}")
    print(f"  Total:        {len(df):>3}")

    # ------------------------------------------------------------------
    # Per-backend mean ESS/sec (PASS cells only)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PER-BACKEND ESS/sec (PASS cells)")
    print(f"{'=' * 60}")
    print_per_backend_ess(df)

    # ------------------------------------------------------------------
    # Per-backend divergent rate
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PER-BACKEND DIVERGENT RATE")
    print(f"{'=' * 60}")
    print_per_backend_divergent_rate(df)

    # ------------------------------------------------------------------
    # Noise-floor ESS variance (A-vs-A seed runs)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("NOISE-FLOOR ESS VARIANCE (A-vs-A seed runs)")
    print(f"{'=' * 60}")
    print_noise_floor(df)

    # ------------------------------------------------------------------
    # Head-to-head ESS/sec ratio (BlackJAX / NumPyro)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("HEAD-TO-HEAD ESS/sec RATIO (BlackJAX / NumPyro)")
    print(f"{'=' * 60}")
    print_head_to_head(df)

    # ------------------------------------------------------------------
    # AUDIT-04 threshold check
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("AUDIT-04 SAMPLE SIZE CHECK")
    print(f"{'=' * 60}")
    print_audit04_check(df)

    print(f"\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
