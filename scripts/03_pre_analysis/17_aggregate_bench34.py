"""Aggregate Phase 34 Mode B grid-sweep results into a capability map table.

Reads cell JSON result files from ``models/power/bench_mode_b_results/``,
classifies missing cells as TIMEOUT (if started marker exists) or NOT_RUN,
and prints a summary table plus optional markdown rows for
``docs/CAPABILITY_MAP.md``.

Usage
-----
    python scripts/03_pre_analysis/17_aggregate_bench34.py
    python scripts/03_pre_analysis/17_aggregate_bench34.py --output-md capmap_b.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Grid constants (duplicated from 16_benchmark_hierarchical.py for standalone use)
# ---------------------------------------------------------------------------
MODELS = ["hgf_2level", "hgf_3level"]
N_PER_GROUP = [10, 17, 33]
MITIGATION_COMBOS = [
    "hier+M1",
    "hier+M1+M2",
    "hier+M1+M2+Laplace",
    "hier+M1+M2+Laplace+covariates",
]

TOTAL_CELLS = len(MODELS) * len(N_PER_GROUP) * len(MITIGATION_COMBOS)  # 24

# Default results directory (relative to project root)
DEFAULT_RESULTS_DIR = Path("models/power/bench_mode_b_results")

# Status emoji mapping for capability map markdown
STATUS_EMOJI = {
    "PASS": "✅",
    "TIMEOUT": "❌",
    "CRASH": "❌",
    "INVALID": "⚠",
    "NOT_RUN": "\U0001f7e8",
}


def decode_cell_id(cell_id: int) -> tuple[str, int, str]:
    """Decode a linear cell index into grid coordinates.

    Parameters
    ----------
    cell_id : int
        Linear index in range [0, 23].

    Returns
    -------
    tuple[str, int, str]
        (model_name, n_per_group, mitigation_combo).

    Raises
    ------
    ValueError
        If cell_id is out of range [0, 23].

    Notes
    -----
    Ordering: model outermost (12 cells each), then n_per_group (4 cells
    each), then mitigation innermost.
    """
    if cell_id < 0 or cell_id >= TOTAL_CELLS:
        raise ValueError(
            f"cell_id must be in [0, {TOTAL_CELLS - 1}], got {cell_id}"
        )
    n_mitigations = len(MITIGATION_COMBOS)  # 4
    n_n = len(N_PER_GROUP)  # 3
    cells_per_model = n_n * n_mitigations  # 12
    cells_per_n = n_mitigations  # 4

    model_idx = cell_id // cells_per_model
    remainder = cell_id % cells_per_model
    n_idx = remainder // cells_per_n
    m_idx = remainder % cells_per_n
    return MODELS[model_idx], N_PER_GROUP[n_idx], MITIGATION_COMBOS[m_idx]


def _format_walltime(seconds: float | None) -> str:
    """Format walltime seconds as human-readable string.

    Parameters
    ----------
    seconds : float | None
        Walltime in seconds.

    Returns
    -------
    str
        Formatted walltime string (e.g., "1h23m" or "--").
    """
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    if hours >= 24:
        return ">24h"
    return f"{hours:.1f}h"


def _model_display(model_name: str) -> str:
    """Format model name for capability map display.

    Parameters
    ----------
    model_name : str
        Raw model name (e.g., "hgf_2level").

    Returns
    -------
    str
        Display name (e.g., "2-level").
    """
    if "2level" in model_name:
        return "2-level"
    return "3-level"


def _format_recovery_diagnostics(r: dict) -> str:
    """Format recovery metrics for the diagnostics column.

    Parameters
    ----------
    r : dict
        Cell result dict with optional recovery fields.

    Returns
    -------
    str
        Compact diagnostic string.
    """
    rhat = r.get("rhat_max")
    ess = r.get("ess_min")
    div_rate = r.get("divergent_rate")

    status = r.get("status", "NOT_RUN")

    if status == "TIMEOUT":
        return "window_adaptation/sampling did not complete within 24h walltime"
    if status == "CRASH":
        error = r.get("error", "unknown error")
        return f"CRASH: {error[:80]}"

    parts: list[str] = []

    # Standard convergence diagnostics
    if rhat is not None and ess is not None and div_rate is not None:
        parts.append(
            f"R-hat={rhat:.4f}, ESS={ess:.0f}, div={div_rate:.1%}"
        )
    else:
        parts.append("--")

    # Individual-level recovery
    recovery_r_ind = r.get("recovery_r_individual")
    if recovery_r_ind:
        vals = [v for v in recovery_r_ind.values() if v is not None]
        if vals:
            avg_r = sum(vals) / len(vals)
            parts.append(f"r(theta)={avg_r:.3f}")

    # Group-level mu recovery
    recovery_r_mu = r.get("recovery_r_mu")
    if recovery_r_mu:
        vals = [v for v in recovery_r_mu.values() if v is not None]
        if vals:
            avg_r_mu = sum(vals) / len(vals)
            parts.append(f"r(mu)={avg_r_mu:.3f}")

    # Covariate beta recovery
    recovery_beta = r.get("recovery_beta")
    if recovery_beta:
        beta_parts: list[str] = []
        for pn, bd in recovery_beta.items():
            if bd and "posterior_mean" in bd and "true" in bd:
                beta_parts.append(f"{pn}:{bd['posterior_mean']:.2f}(true={bd['true']:.2f})")
        if beta_parts:
            parts.append(f"r(beta)=[{', '.join(beta_parts)}]")

    # Sigma recovery
    recovery_sigma = r.get("recovery_sigma")
    if recovery_sigma:
        rel_errs = [
            v["relative_error"]
            for v in recovery_sigma.values()
            if isinstance(v, dict) and "relative_error" in v
            and v["relative_error"] is not None
        ]
        if rel_errs:
            avg_rel_err = sum(rel_errs) / len(rel_errs)
            parts.append(f"relErr(sigma)={avg_rel_err:.3f}")

    return ", ".join(parts) if parts else "--"


def load_cell_results(
    results_dir: Path,
) -> list[dict]:
    """Load all Mode B cell results, classifying missing cells.

    Parameters
    ----------
    results_dir : Path
        Directory containing cell JSON files.

    Returns
    -------
    list[dict]
        List of 24 cell result dicts, one per cell_id (0-23).
    """
    results = []
    for cell_id in range(TOTAL_CELLS):
        model_name, n_per_group, mitigation_combo = decode_cell_id(cell_id)
        # pick_best_cue: 2 groups x 3 sessions
        p_total = n_per_group * 2 * 3

        result_path = results_dir / f"cell_{cell_id:04d}.json"
        started_path = results_dir / f"cell_{cell_id:04d}_started.json"

        if result_path.exists():
            with result_path.open() as f:
                data = json.load(f)
            results.append(data)
        elif started_path.exists():
            # Started but no result => TIMEOUT
            with started_path.open() as f:
                started_data = json.load(f)
            results.append({
                "cell_id": cell_id,
                "model": model_name,
                "n_per_group": n_per_group,
                "p_total": p_total,
                "mitigation_combo": mitigation_combo,
                "status": "TIMEOUT",
                "walltime_s": 86400.0,  # 24h walltime limit
                "rhat_max": None,
                "ess_min": None,
                "divergent_rate": None,
                "fit_config_yaml": started_data.get("fit_config_yaml", "unknown"),
                "job_id": started_data.get("job_id", "unknown"),
                "commit": started_data.get("commit", "unknown"),
            })
        else:
            # Not even started
            results.append({
                "cell_id": cell_id,
                "model": model_name,
                "n_per_group": n_per_group,
                "p_total": p_total,
                "mitigation_combo": mitigation_combo,
                "status": "NOT_RUN",
                "walltime_s": None,
                "rhat_max": None,
                "ess_min": None,
                "divergent_rate": None,
                "fit_config_yaml": None,
                "job_id": None,
                "commit": None,
            })

    return results


def print_summary_table(results: list[dict]) -> None:
    """Print a text summary table of all cell results to stdout.

    Parameters
    ----------
    results : list[dict]
        List of cell result dicts from load_cell_results.
    """
    header = (
        f"{'ID':>4} | {'Model':<11} | {'N/grp':>5} | {'P':>5} "
        f"| {'Mitigation':<30} | {'Status':<8} | {'Wall':>6} "
        f"| {'Rhat':>6} | {'ESS':>7} | {'Div%':>6}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in results:
        cell_id = r["cell_id"]
        model = r.get("model", "?")
        n_grp = r.get("n_per_group", 0)
        p_total = r.get("p_total", 0)
        mitigation = r.get("mitigation_combo", "?")
        status = r.get("status", "?")
        walltime = _format_walltime(r.get("walltime_s"))
        rhat = r.get("rhat_max")
        ess = r.get("ess_min")
        div_rate = r.get("divergent_rate")

        rhat_str = f"{rhat:.4f}" if rhat is not None else "--"
        ess_str = f"{ess:.0f}" if ess is not None else "--"
        div_str = f"{div_rate:.1%}" if div_rate is not None else "--"

        print(
            f"{cell_id:>4} | {model:<11} | {n_grp:>5} | {p_total:>5} "
            f"| {mitigation:<30} | {status:<8} | {walltime:>6} "
            f"| {rhat_str:>6} | {ess_str:>7} | {div_str:>6}"
        )

    print(sep)


def generate_markdown_rows(results: list[dict]) -> list[str]:
    """Generate capability map markdown table rows for Mode B results.

    Parameters
    ----------
    results : list[dict]
        List of cell result dicts from load_cell_results.

    Returns
    -------
    list[str]
        Markdown table rows (one per cell with a completed status).
    """
    rows = []
    for r in results:
        status = r.get("status", "NOT_RUN")
        if status == "NOT_RUN":
            continue  # Skip NOT_RUN cells in capability map output

        model = _model_display(r.get("model", "hgf_2level"))
        mitigation_combo = r.get("mitigation_combo", "hier+M1")
        p_total = r.get("p_total", 0)
        n_grp = r.get("n_per_group", 0)
        walltime = _format_walltime(r.get("walltime_s"))
        emoji = STATUS_EMOJI.get(status, "?")
        job_id = r.get("job_id", "unknown")
        commit = r.get("commit", "unknown")

        mitigation_tag = f"[{mitigation_combo}]"
        evidence = f"job {job_id}, commit `{commit}`"
        p_display = f"{p_total} (n/grp={n_grp})"
        diag = _format_recovery_diagnostics(r)

        row = (
            f"| {model} | BlackJAX | [dense] | [mode-b] | {mitigation_tag} "
            f"| {p_display} | {emoji} {status} | {walltime} "
            f"| {evidence} | {diag} |"
        )
        rows.append(row)

    return rows


def print_counts(results: list[dict]) -> None:
    """Print status counts summary grouped by model x n_per_group x mitigation.

    Parameters
    ----------
    results : list[dict]
        List of cell result dicts.
    """
    counts: dict[str, int] = {
        "PASS": 0,
        "TIMEOUT": 0,
        "CRASH": 0,
        "INVALID": 0,
        "NOT_RUN": 0,
    }
    for r in results:
        status = r.get("status", "NOT_RUN")
        counts[status] = counts.get(status, 0) + 1

    total_completed = sum(v for k, v in counts.items() if k != "NOT_RUN")

    print(f"\n{'=' * 40}")
    print("STATUS COUNTS (Mode B, 24 cells)")
    print(f"{'=' * 40}")
    print(f"  PASS:    {counts['PASS']:>3}")
    print(f"  TIMEOUT: {counts['TIMEOUT']:>3}")
    print(f"  CRASH:   {counts['CRASH']:>3}")
    print(f"  INVALID: {counts['INVALID']:>3}")
    print(f"  NOT_RUN: {counts['NOT_RUN']:>3}")
    print(f"  {'-' * 20}")
    print(f"  Completed (any status): {total_completed}/{TOTAL_CELLS}")
    print(f"{'=' * 40}")

    # Per-model breakdown
    print(f"\n{'Model':<12} | {'N/grp':>5} | {'Mitigation':<30} | Status")
    print("-" * 70)
    for r in results:
        model = _model_display(r.get("model", "hgf_2level"))
        n_grp = r.get("n_per_group", 0)
        mitigation = r.get("mitigation_combo", "?")
        status = r.get("status", "?")
        print(f"  {model:<10} | {n_grp:>5} | {mitigation:<30} | {status}")


def main() -> int:
    """Aggregate Phase 34 Mode B benchmark results and report.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 34 Mode B grid-sweep benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing cell_*.json result files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=(
            "Optional: write markdown table rows to this file "
            "instead of stdout."
        ),
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_md: Path | None = args.output_md

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"NOTE: Results directory created (was empty): {results_dir}",
            file=sys.stderr,
        )

    # Load and classify all cells
    results = load_cell_results(results_dir)

    # Print summary table
    print_summary_table(results)

    # Print counts
    print_counts(results)

    # Generate markdown rows
    md_rows = generate_markdown_rows(results)

    if md_rows:
        print(f"\n{'=' * 40}")
        print("CAPABILITY MAP ROWS (markdown, Mode B)")
        print(f"{'=' * 40}")

        md_output = "\n".join(md_rows)

        if output_md is not None:
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(md_output + "\n", encoding="utf-8")
            print(f"  Written to: {output_md}")
        else:
            print(md_output)
    else:
        print("\nNo completed Mode B cells to generate markdown rows for.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
