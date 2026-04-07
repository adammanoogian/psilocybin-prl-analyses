"""Phase 9: Recovery precheck gate for the BFDA power analysis pipeline.

Runs PRE-01 (50-participant parameter recovery on baseline sessions),
PRE-02 (confound matrix), PRE-03 (eligibility table), and PRE-06 (MCMC
convergence gating). Optionally runs the PRE-04/PRE-05 trial count sweep
when ``--sweep`` is passed. Outputs eligibility CSV and recovery metrics CSV
to ``results/power/prechecks/`` by default.

Usage
-----
Run from the project root::

    conda run -n ds_env python scripts/09_run_prechecks.py
    conda run -n ds_env python scripts/09_run_prechecks.py --n-participants 30
    conda run -n ds_env python scripts/09_run_prechecks.py --model hgf_2level
    conda run -n ds_env python scripts/09_run_prechecks.py --sweep
    conda run -n ds_env python scripts/09_run_prechecks.py --sweep --sweep-grid 200 300 420

Options
-------
--n-participants INT
    Number of synthetic participants per group. Default 50.
--model STR
    HGF model variant. One of ``hgf_2level`` or ``hgf_3level``. Default
    ``hgf_3level``.
--seed INT
    Master RNG seed for reproducibility. Default 42.
--output-dir PATH
    Directory for output files. Default ``results/power/prechecks/``.
--sweep
    Run the trial count sweep (PRE-04/PRE-05) after the main precheck.
--sweep-grid INT [INT ...]
    Trial counts to evaluate in the sweep. Default: 150 200 250 300 420.
--n-per-group-sweep INT
    Participants per group for the sweep. Default 30.

Output
------
``results/power/prechecks/`` (or ``--output-dir``) directory with:

* ``power_eligible_params.csv``         — eligibility table
* ``recovery_metrics_precheck.csv``     — per-parameter r, p, bias, RMSE, n
* ``recovery_scatter_precheck_*.png``   — scatter plots (true vs recovered)
* ``correlation_matrix_precheck_*.png`` — inter-parameter heatmap
* ``trial_sweep_results.csv``           — long-form sweep results (--sweep only)
* ``trial_sweep_recovery_r.png``        — VIZ-01 recovery vs trial count (--sweep only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that config.py is importable when
# running as a script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from config import RESULTS_DIR  # noqa: E402
from prl_hgf.env.task_config import load_config  # noqa: E402
from prl_hgf.power.precheck import (  # noqa: E402
    build_eligibility_table,
    find_minimum_trial_count,
    plot_trial_sweep,
    run_recovery_precheck,
    run_trial_sweep,
)

_DEFAULT_OUTPUT_DIR = RESULTS_DIR / "power" / "prechecks"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 9: Recovery precheck gate for the BFDA power analysis. "
            "Establishes which HGF parameters are power-eligible before the "
            "expensive Phase 10 sweep."
        )
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=50,
        dest="n_participants",
        help="Number of synthetic participants per group (default: 50).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hgf_3level",
        choices=["hgf_2level", "hgf_3level"],
        help="HGF model variant to fit (default: hgf_3level).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help=f"Directory for output files (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help=(
            "Run the PRE-04/PRE-05 trial count sweep after the main precheck. "
            "Uses reduced MCMC settings (2 chains, 500 draws, 500 tune)."
        ),
    )
    parser.add_argument(
        "--sweep-grid",
        nargs="+",
        type=int,
        default=[150, 200, 250, 300, 420],
        dest="sweep_grid",
        help="Trial counts to evaluate in the sweep (default: 150 200 250 300 420).",
    )
    parser.add_argument(
        "--n-per-group-sweep",
        type=int,
        default=30,
        dest="n_per_group_sweep",
        help="Participants per group for the trial count sweep (default: 30).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_eligibility_table(eligibility_df) -> None:
    """Print eligibility table with clear formatting."""
    print("\n" + "=" * 70)
    print("PRE-03: Parameter Eligibility Table")
    print("=" * 70)
    print(
        f"{'Parameter':<12} {'r':>6}  {'Status':<30}  Reason"
    )
    print("-" * 70)
    for _, row in eligibility_df.iterrows():
        reason_short = row["reason"][:60] if len(row["reason"]) > 60 else row["reason"]
        print(
            f"{row['parameter']:<12} {row['r']:>6.3f}  "
            f"{row['status']:<30}  {reason_short}"
        )
    print("=" * 70)


def _print_confound_warnings(corr_df) -> None:
    """Print inter-parameter correlation pairs where |r| > 0.8."""
    arr = corr_df.to_numpy()
    n = arr.shape[0]
    cols = list(corr_df.columns)
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(arr[i, j]) > 0.8:
                high_pairs.append((cols[i], cols[j], float(arr[i, j])))

    if high_pairs:
        print("\nPRE-02: Confound Matrix — High-correlation pairs (|r| > 0.8):")
        for p1, p2, r_val in high_pairs:
            print(f"  WARNING: {p1} & {p2}  r={r_val:.3f}")
    else:
        print("\nPRE-02: Confound Matrix — No high-correlation pairs (|r| > 0.8).")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Phase 9 precheck pipeline."""
    args = _parse_args()

    print("=" * 70)
    print("Phase 9 — Recovery Precheck Gate")
    print("=" * 70)
    print(f"  n_participants/group : {args.n_participants}")
    print(f"  model                : {args.model}")
    print(f"  seed                 : {args.seed}")
    print(f"  output_dir           : {args.output_dir}")

    # --- 1. Load config
    config = load_config()
    print(f"\nConfig loaded: {config.task.n_trials_total} trials/session")

    # --- 2. Run recovery precheck
    print("\nRunning recovery precheck (baseline sessions only)...")
    result = run_recovery_precheck(
        config,
        n_participants=args.n_participants,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # --- 3. Build and print eligibility table
    eligibility_df = build_eligibility_table(result.metrics_df)
    _print_eligibility_table(eligibility_df)

    # --- 4. Convergence exclusion report (also printed inside run_recovery_precheck)
    print(
        f"\nPRE-06: {result.n_flagged}/{result.n_total} participants excluded "
        f"(R-hat>1.05 or ESS<400)"
    )

    # --- 5. Confound matrix report
    _print_confound_warnings(result.corr_df)

    # --- 6. Save eligibility CSV (also saved inside run_recovery_precheck)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    elig_path = args.output_dir / "power_eligible_params.csv"
    eligibility_df.to_csv(elig_path, index=False)
    print(f"\nSaved: {elig_path}")

    metrics_path = args.output_dir / "recovery_metrics_precheck.csv"
    result.metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    # --- 7. Summary banner
    n_eligible = int(
        (eligibility_df["status"] == "power-eligible").sum()
    )
    n_exploratory = int(
        (eligibility_df["status"] == "exploratory -- upper bound").sum()
    )
    n_excluded = int(
        (eligibility_df["status"] == "excluded").sum()
    )

    print("\n" + "=" * 70)
    print("Phase 9 Precheck — Summary")
    print("=" * 70)
    print(f"  Power-eligible parameters : {n_eligible}")
    print(f"  Exploratory (upper bound) : {n_exploratory}")
    print(f"  Excluded (|r| < 0.7)      : {n_excluded}")
    print(f"  Convergence exclusions    : {result.n_flagged}/{result.n_total}")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("=" * 70)

    # --- 8. Optional trial count sweep (PRE-04/PRE-05)
    if args.sweep:
        _run_trial_sweep(args, config, eligibility_df, output_dir=args.output_dir)


def _run_trial_sweep(args, config, eligibility_df, output_dir: Path) -> None:
    """Run PRE-04/PRE-05 trial count sweep and print minimum trial count."""
    print("\n" + "=" * 70)
    print("PRE-04/PRE-05: Trial Count Sweep")
    print("=" * 70)
    print(f"  sweep_grid      : {args.sweep_grid}")
    print(f"  n_per_group     : {args.n_per_group_sweep}")
    print(
        "  MCMC settings  : 2 chains, 500 draws, 500 tune "
        "(reduced; adequate for recovery)"
    )
    print()

    sweep_results = run_trial_sweep(
        config,
        trial_grid=args.sweep_grid,
        n_per_group=args.n_per_group_sweep,
        output_dir=output_dir,
    )

    # VIZ-01 sweep plot
    sweep_plot_path = output_dir / "trial_sweep_recovery_r.png"
    fig = plot_trial_sweep(sweep_results, save_path=sweep_plot_path)
    plt.close(fig)
    print(f"\nSaved: {sweep_plot_path}")

    # Eligible params from main precheck (power-eligible status only)
    eligible_params = eligibility_df.loc[
        eligibility_df["status"] == "power-eligible", "parameter"
    ].tolist()

    min_trials = find_minimum_trial_count(
        sweep_results,
        eligible_params=eligible_params if eligible_params else None,
    )

    if min_trials is not None:
        print(
            f"Minimum trial count for all power-eligible parameters: {min_trials}"
        )
    else:
        print(
            "WARNING: No trial count satisfies r >= 0.7 for all power-eligible "
            "parameters"
        )

    print(
        "NOTE: Sweep used reduced MCMC settings (2 chains, 500 draws, 500 tune). "
        "Main precheck used full settings."
    )


if __name__ == "__main__":
    main()
