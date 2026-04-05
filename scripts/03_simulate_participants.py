"""Pipeline script — Stage 03: Simulate synthetic participants.

Generates the full synthetic cohort (2 groups × 30 participants × 3 sessions)
and saves a tidy trial-level CSV to ``data/simulated/simulated_participants.csv``.

This CSV is the input for Phase 4 (fitting) and Phase 5 (parameter recovery).

Usage
-----
Run from the project root::

    conda run -n ds_env python scripts/03_simulate_participants.py

Output
------
``data/simulated/simulated_participants.csv`` — one row per trial across all
participant-sessions, with ground-truth parameters as ``true_*`` columns.

Reproducibility
---------------
Set ``config.simulation.master_seed`` in ``configs/prl_analysis.yaml`` to
control all randomness.  The same seed always produces identical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that config.py is importable when
# running as a script (pytest adds it via pythonpath in pyproject.toml).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_DIR  # noqa: E402 (after sys.path adjustment)
from prl_hgf.env.task_config import load_config  # noqa: E402
from prl_hgf.simulation import simulate_batch  # noqa: E402


def main() -> None:
    """Load config, run batch simulation, save output CSV."""
    config = load_config()
    sim_cfg = config.simulation
    n_per_group = sim_cfg.n_participants_per_group
    n_groups = len(sim_cfg.groups)
    n_sessions = 3  # baseline + post_dose + followup
    n_participant_sessions = n_groups * n_per_group * n_sessions
    n_trials_per_session = config.task.n_trials_total
    expected_rows = n_participant_sessions * n_trials_per_session

    print("=" * 60)
    print("Stage 03 — Simulate Participants")
    print("=" * 60)
    print(f"Groups:               {sorted(sim_cfg.groups.keys())}")
    print(f"Participants/group:   {n_per_group}")
    print(f"Sessions:             {n_sessions}")
    print(f"Trials/session:       {n_trials_per_session}")
    print(f"Expected total rows:  {expected_rows:,}")
    print(f"Master seed:          {sim_cfg.master_seed}")
    print()

    output_dir = DATA_DIR / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulated_participants.csv"

    df = simulate_batch(config, output_path=output_path)

    print()
    print("=" * 60)
    print("Simulation complete.")
    print(f"Output rows:    {len(df):,}")
    print(f"Output columns: {len(df.columns)}")
    print(f"Output file:    {output_path}")
    print()
    print("Summary statistics (true parameters by group and session):")
    summary = (
        df.drop_duplicates(subset=["participant_id", "session"])
        .groupby(["group", "session"])[
            ["true_omega_2", "true_omega_3", "true_kappa", "true_beta", "true_zeta"]
        ]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
