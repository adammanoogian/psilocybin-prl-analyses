#!/usr/bin/env python
"""Entry point for a single SLURM array task in the power analysis sweep.

Each invocation corresponds to one row of the power analysis grid. The script:

1. Reads ``SLURM_ARRAY_TASK_ID`` via ``--task-id`` to build an independent
   RNG stream using :func:`~prl_hgf.power.seeds.make_child_rng`.
2. Loads base and power configurations from ``configs/prl_analysis.yaml``.
3. Writes a 13-column parquet row to ``<output_dir>/job_<JOB_ID>_task_<TASK_ID>.parquet``.

In ``--dry-run`` mode a placeholder row is written (all zero/false values)
to verify infrastructure without running the full simulate-fit pipeline.
The ``--output-dir`` flag lets integration tests redirect output to a
temporary directory instead of the real ``results/power/`` location.

Full simulation + fitting logic will be added in Phase 10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work on cluster
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as _cfg
from prl_hgf.env.task_config import load_config
from prl_hgf.power.config import load_power_config
from prl_hgf.power.schema import write_parquet_row
from prl_hgf.power.seeds import make_child_rng


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: task_id, job_id, dry_run,
        fit_chains, fit_draws, fit_tune, output_dir.
    """
    parser = argparse.ArgumentParser(
        description="Run one power analysis iteration (SLURM array task).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="SLURM_ARRAY_TASK_ID (zero-based index of this array task).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="SLURM_ARRAY_JOB_ID (used in output filename).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Write a placeholder parquet row instead of running the full "
            "pipeline. Used to verify infrastructure without MCMC."
        ),
    )
    parser.add_argument(
        "--fit-chains",
        type=int,
        default=2,
        help="Number of MCMC chains for power sweep fits.",
    )
    parser.add_argument(
        "--fit-draws",
        type=int,
        default=500,
        help="Posterior draws per chain.",
    )
    parser.add_argument(
        "--fit-tune",
        type=int,
        default=500,
        help="Tuning steps per chain.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. Defaults to RESULTS_DIR / 'power'. "
            "Use this in tests to redirect output to a tmp directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute one power sweep iteration.

    Raises
    ------
    SystemExit
        On argument parse error (via argparse).
    """
    args = parse_args()

    base_config = load_config()
    power_config = load_power_config()

    # Build independent RNG for this task
    rng = make_child_rng(
        master_seed=power_config.master_seed,
        n_jobs=power_config.n_jobs,
        job_index=args.task_id,
    )

    # Output path: <output_dir>/job_<JOB_ID>_task_<TASK_ID>.parquet
    output_dir = args.output_dir if args.output_dir is not None else _cfg.RESULTS_DIR / "power"
    output_path = output_dir / f"job_{args.job_id}_task_{args.task_id:04d}.parquet"

    if args.dry_run:
        # Write placeholder row to verify infrastructure end-to-end
        placeholder = {
            "sweep_type":    "smoke_test",
            "effect_size":   0.0,
            "n_per_group":   base_config.simulation.n_participants_per_group,
            "trial_count":   base_config.task.n_trials_total,
            "iteration":     args.task_id,
            "parameter":     "omega_2",
            "bf_value":      1.0,
            "bf_exceeds":    False,
            "bms_xp":        0.5,
            "bms_correct":   False,
            "recovery_r":    0.0,
            "n_divergences": 0,
            "mean_rhat":     1.0,
        }
        write_parquet_row(placeholder, output_path)
        print(f"Dry run: wrote placeholder to {output_path}")
        return

    # Full pipeline will be implemented in Phase 10
    print(f"Task {args.task_id}: full pipeline not yet implemented")
    print(f"RNG state ready, output would go to {output_path}")
    _ = rng  # suppress unused-variable warning


if __name__ == "__main__":
    main()
