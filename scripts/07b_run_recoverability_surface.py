#!/usr/bin/env python
"""Run the joint (N x sets x d) recoverability surface.

Sweeps number of task sets, sample size, and effect size to find
the minimum design that recovers DiD contrasts at r >= 0.7.

Usage::

    python scripts/07b_run_recoverability_surface.py
    python scripts/07b_run_recoverability_surface.py --n-iterations 10
    python scripts/07b_run_recoverability_surface.py --output-dir /tmp/surface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as _cfg
from prl_hgf.env.task_config import load_config
from prl_hgf.power.precheck import run_recoverability_surface


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``n_iterations`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run joint recoverability surface (N x sets x d).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help="Iterations per (sets, d) cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_cfg.RESULTS_DIR / "power" / "prechecks",
        help="Output directory for recoverability_surface.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the recoverability surface."""
    args = parse_args()
    config = load_config()

    print("Running joint recoverability surface...")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Output: {args.output_dir}")

    run_recoverability_surface(
        config,
        n_iterations=args.n_iterations,
        output_dir=args.output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
