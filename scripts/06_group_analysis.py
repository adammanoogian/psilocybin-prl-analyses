"""Phase 6: Group-level HGF parameter analysis.

Loads simulated data and batch-fitting results, builds wide-form per-subject
parameter estimates, fits Bayesian mixed-effects models (group × session
interaction via bambi), extracts posterior contrasts and frequentist effect
sizes, and saves all outputs to ``results/group_analysis/``.

Usage
-----
Run from the project root::

    conda run -n ds_env python scripts/06_group_analysis.py
    conda run -n ds_env python scripts/06_group_analysis.py --model 3level
    conda run -n ds_env python scripts/06_group_analysis.py --skip-plots

Options
-------
--model {2level,3level}
    HGF model variant to analyse. Default: ``2level``.
--skip-plots
    Skip saving figure outputs (CSV outputs are always saved).
--draws N
    Number of posterior draws per chain for bambi. Default: ``2000``.

Output
------
``results/group_analysis/`` directory with:

* ``estimates_wide_{model}.csv``   — wide-form per-subject parameter means
* ``effect_sizes_{model}.csv``     — Cohen's d and partial η² per param × session
* ``contrasts_{model}_{param}.csv`` — posterior contrasts per session (one file
  per parameter)
* ``raincloud_{param}.png``        — raincloud plots (unless ``--skip-plots``)
* ``interaction_{param}.png``      — interaction plots (unless ``--skip-plots``)

Phase-stratification note
-------------------------
ω₂ (tonic volatility) is a session-level parameter in the current pipeline —
it captures mean learning rate across the full session.  Trial-level
phase-stratified analysis (stable vs volatile learning rate per reversal phase)
requires saving per-trial belief trajectories and is deferred to the real-data
pipeline once participant data are collected.
"""

from __future__ import annotations

import argparse
import logging
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
import pandas as pd  # noqa: E402

from config import GROUP_ANALYSIS_DIR, RESULTS_DIR  # noqa: E402
from prl_hgf.analysis.effect_sizes import compute_effect_sizes_table  # noqa: E402
from prl_hgf.analysis.group import (  # noqa: E402
    build_estimates_wide,
    summarize_group_models,
)
from prl_hgf.analysis.group_plots import (  # noqa: E402
    plot_interaction,
    plot_raincloud,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Parameters of interest for each model variant
_PARAMS_2LEVEL = ["omega_2", "beta", "zeta"]
_PARAMS_3LEVEL = ["omega_2", "omega_3", "kappa", "beta", "zeta"]

_SESSION_LABELS = ["baseline", "session2", "session3"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 6: Group-level HGF parameter analysis."
    )
    parser.add_argument(
        "--model",
        choices=["2level", "3level"],
        default="2level",
        help="HGF model variant to analyse. Default: 2level.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=False,
        help="Skip saving figure outputs (CSV outputs are always saved).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Posterior draws per chain for bambi. Default: 2000.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _resolve_fit_path(model: str) -> Path | None:
    """Return path to fit-results CSV for *model*, or None if not found."""
    candidates = [
        RESULTS_DIR / f"fit_results_hgf_{model}.csv",
        RESULTS_DIR / f"fit_results_{model}.csv",
        _PROJECT_ROOT / "data" / "fitted" / f"hgf_{model}_results.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_sim_path() -> Path | None:
    """Return path to simulated-data CSV, or None if not found."""
    candidates = [
        RESULTS_DIR / "simulated_data.csv",
        RESULTS_DIR / "batch_simulation.csv",
        _PROJECT_ROOT / "data" / "simulated" / "batch_simulation.csv",
        _PROJECT_ROOT / "data" / "simulated" / "simulated_participants.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Phase 6 group-level analysis pipeline."""
    args = _parse_args()
    model = f"hgf_{args.model}"
    params = _PARAMS_3LEVEL if args.model == "3level" else _PARAMS_2LEVEL

    GROUP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 6 — Group-Level HGF Parameter Analysis")
    print(f"Model: {model}")
    print("=" * 60)

    # --- 1. Load fit results ---
    fit_path = _resolve_fit_path(args.model)
    if fit_path is None or not fit_path.exists():
        print(
            f"\nERROR: Fit results for model '{model}' not found.\n"
            "Run scripts/04_fit_participants.py first.\n"
            f"Expected at: {RESULTS_DIR / f'fit_results_{model}.csv'}"
        )
        sys.exit(1)

    print(f"\nLoading fit results from: {fit_path}")
    fit_df = pd.read_csv(fit_path)
    print(f"  {len(fit_df):,} rows loaded")

    # Optionally load sim_df for context (not strictly required for group analysis)
    sim_path = _resolve_sim_path()
    if sim_path is not None:
        print(f"Simulated data found at: {sim_path} (not required for group analysis)")

    # --- 2. Build wide-form estimates ---
    print("\nBuilding wide-form estimates table...")
    estimates_wide = build_estimates_wide(fit_df, model=model, exclude_flagged=True)
    n_participants = estimates_wide["participant_id"].nunique()
    print(f"  {len(estimates_wide)} participant-session rows ({n_participants} participants)")

    estimates_path = GROUP_ANALYSIS_DIR / f"estimates_wide_{model}.csv"
    estimates_wide.to_csv(estimates_path, index=False)
    print(f"  Saved: {estimates_path}")

    # --- 3. Effect sizes ---
    print("\nComputing effect sizes...")
    available_sessions = sorted(estimates_wide["session"].dropna().unique().tolist())
    available_params = [p for p in params if p in estimates_wide.columns]

    if len(available_params) == 0:
        print(f"  WARNING: None of {params} found in estimates_wide. Skipping effect sizes.")
    else:
        effect_df = compute_effect_sizes_table(
            estimates_wide,
            params=available_params,
            sessions=available_sessions,
        )
        effect_path = GROUP_ANALYSIS_DIR / f"effect_sizes_{model}.csv"
        effect_df.to_csv(effect_path, index=False)
        print(f"  Saved: {effect_path}")
        print(effect_df.to_string(index=False))

    # --- 4. Bayesian mixed-effects models ---
    print("\nFitting Bayesian mixed-effects models (bambi)...")
    print("  NOTE: omega_2 is a session-level parameter (mean learning rate across")
    print("  the full session). Trial-level phase-stratified analysis (stable vs")
    print("  volatile phases) requires per-trial belief trajectories and is")
    print("  deferred to the real-data pipeline.")

    if n_participants < 6:
        print(
            f"  WARNING: Only {n_participants} participants available. "
            "Group models require at least 6 participants (3 per group). Skipping."
        )
        group_results: dict = {}
    else:
        group_results = summarize_group_models(
            estimates_wide,
            params=available_params,
            draws=args.draws,
        )

        # Save posterior contrasts
        for param, res in group_results.items():
            contrasts_path = GROUP_ANALYSIS_DIR / f"contrasts_{model}_{param}.csv"
            res["contrasts"].to_csv(contrasts_path, index=False)
            print(f"  Saved contrasts: {contrasts_path}")
            print(res["contrasts"].to_string(index=False))

    # --- 5. Plots ---
    if args.skip_plots:
        print("\nSkipping plots (--skip-plots).")
    else:
        print("\nSaving plots...")
        for param in available_params:
            if param not in estimates_wide.columns:
                continue

            # Raincloud
            rc_path = GROUP_ANALYSIS_DIR / f"raincloud_{param}.png"
            fig = plot_raincloud(estimates_wide, outcome=param, save_path=rc_path)
            plt.close(fig)
            print(f"  Saved: {rc_path}")

            # Interaction
            int_path = GROUP_ANALYSIS_DIR / f"interaction_{param}.png"
            fig = plot_interaction(estimates_wide, outcome=param, save_path=int_path)
            plt.close(fig)
            print(f"  Saved: {int_path}")

    # --- 6. Summary ---
    print("\n" + "=" * 60)
    print("Phase 6 Group Analysis — Complete")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Participants analysed: {n_participants}")
    print(f"Parameters: {available_params}")
    print(f"Outputs saved to: {GROUP_ANALYSIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
