"""Aggregate Phase 14.2 variant-comparison runs into a single table.

Reads ``cluster/logs/bench14*.out`` files (or a user-supplied list), parses
the ``[hierarchical]`` and ``[diag ...]`` instrumentation lines, plus the
SLURM banner ``Sampler:`` / ``Max tree depth:`` / ``Chunk id:`` lines, and
writes a flat CSV + a Markdown summary keyed by variant.

The point is to compare variants 1–5 side-by-side on a shared cohort.
Each .out file represents one bench run with one set of knobs; this
script is a pure post-hoc parser, so adding a new variant (or re-running
an existing one) only requires submitting another SLURM job — no
changes here.

Usage
-----
::

    # All bench14 logs in cluster/logs/
    python scripts/14_2_aggregate_variants.py

    # Specific logs only
    python scripts/14_2_aggregate_variants.py \\
        cluster/logs/bench14_54969668.out \\
        cluster/logs/bench14arr_57000000_*.out

    # Custom output path
    python scripts/14_2_aggregate_variants.py \\
        --output results/diagnostics/variant_compare.csv

Outputs
-------
* ``results/diagnostics/variant_comparison.csv`` — one row per .out file.
* ``results/diagnostics/variant_comparison.md`` — same data as a
  Markdown table sorted by total wall time, highlighting the variant
  that completed fastest with acceptable diagnostics
  (saturated_frac < 0.5, divergent_count == 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# Importing config is convenient but optional — fall back to repo-relative paths
# when this script is launched from outside a configured environment.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import MODELS_DIR, PROJECT_ROOT  # noqa: E402
except Exception:  # noqa: BLE001
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# SLURM banner — emitted by cluster/14_benchmark*.slurm scripts
_RE_JOB_ID = re.compile(r"^Job ID:\s+(\S+)", re.MULTILINE)
_RE_ARRAY_JOB = re.compile(r"^Array job ID:\s+(\S+)", re.MULTILINE)
_RE_ARRAY_TASK = re.compile(r"^Array task ID:\s+(\S+)", re.MULTILINE)
_RE_NODE = re.compile(r"^Node:\s+(\S+)", re.MULTILINE)
# Primary partition signal: JAX device line printed at start of every bench.
# CudaDevice -> gpu partition, CpuDevice -> comp.  Falls back to node-prefix
# heuristic for older logs that pre-date the print.
_RE_JAX_DEVICES = re.compile(r"JAX devices:\s*\[([^\]]+)\]")
# Node-prefix fallback (m3g* = gpu, m3t* = T4 gpu, m3a* = mixed).  Note m3a
# nodes can be either gpu or comp depending on the SBATCH partition request,
# so this is best-effort only.  When in doubt the JAX-device signal wins.
_NODE_PARTITION_PREFIX = {
    "m3g": "gpu",
    "m3t": "gpu",
    "m3a": "comp",
}
_RE_SAMPLER = re.compile(r"^Sampler:\s+(\S+)", re.MULTILINE)
_RE_MAX_TREE = re.compile(r"^Max tree depth:\s+(\d+)", re.MULTILINE)
_RE_CHUNK_COUNT = re.compile(r"^Chunk count:\s+(\d+)", re.MULTILINE)
_RE_CHUNK_ID = re.compile(r"^Chunk id:\s+(\S+)", re.MULTILINE)

# fit_batch_hierarchical instrumentation (commit 0388afb)
_RE_FBH_ENTERED = re.compile(
    r"\[fit_batch_hierarchical\] entered: "
    r"model=(\S+) sampler=(\S+) n_chains=(\d+) n_tune=(\d+) n_draws=(\d+) "
    r"target_accept=([\d.]+).*?P=(\d+)",
    re.DOTALL,
)
_RE_COHORT = re.compile(
    r"\[fit_batch_hierarchical t=[\d.]+s\] cohort assembled: "
    r"P=(\d+) n_trials=(\d+)"
)

# _run_blackjax_nuts phase markers
_RE_WARMUP_START = re.compile(
    r"\[hierarchical t=[\d.]+s\] starting window_adaptation "
    r"\(num_steps=(\d+), target_accept=([\d.]+)\)"
)
_RE_WARMUP_DONE = re.compile(r"window_adaptation complete in ([\d.]+)s")
_RE_SAMPLE_DONE = re.compile(r"sample_loop complete in ([\d.]+)s")

# Diagnostic summary lines (from _log_nuts_diagnostics, commit 0055408)
_RE_DIAG_LINE = re.compile(r"\[diag (warmup|sample)\] (\S+):\s*(.*)")
_RE_DIAG_FIELDS = re.compile(r"(\w+)=([\-\d.eE]+)")

# Variant 4 chunking
_RE_CHUNK_SUMMARY = re.compile(
    r"Variant 4 chunking: chunk (\d+) of (\d+) -> (\d+) of (\d+)"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_diag_block(text: str, label: str) -> dict[str, float]:
    """Extract key=value pairs from all [diag <label>] lines in `text`.

    Multiple lines per label (e.g. ``[diag warmup] tree_depth: ...``,
    ``[diag warmup] leapfrog_per_step: ...``) get merged into one flat dict
    with the field name prefixed by the line key.  Window-partitioned lines
    (``[diag warmup window=slow4 ...]``) are dropped from the flat output
    here — they're a follow-on artifact, queried separately if needed.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.rstrip()
        if f"[diag {label}]" not in line:
            continue
        m = _RE_DIAG_LINE.search(line)
        if not m:
            continue
        section = m.group(2)  # e.g. "tree_depth", "leapfrog_per_step"
        rest = m.group(3)
        for fk, fv in _RE_DIAG_FIELDS.findall(rest):
            try:
                out[f"{section}_{fk}"] = float(fv)
            except ValueError:  # noqa: PERF203
                continue
    return out


def parse_log(path: Path) -> dict[str, Any]:
    """Parse one bench14 .out (and sibling .err) file into a flat record.

    The sibling .err file is checked for SLURM cancellation reasons (which
    are emitted by slurmctld, not the python process, so they don't reach
    .out).  Outcome flags ``TIME_LIMIT``/``SCANCEL`` come from .err; all
    other diagnostics come from .out.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    err_path = path.with_suffix(".err")
    err_text = ""
    if err_path.exists():
        err_text = err_path.read_text(encoding="utf-8", errors="replace")

    rec: dict[str, Any] = {
        "log_path": str(path),
        "log_size_lines": text.count("\n"),
    }

    # SLURM banner
    for key, pat in [
        ("job_id", _RE_JOB_ID),
        ("array_job_id", _RE_ARRAY_JOB),
        ("array_task_id", _RE_ARRAY_TASK),
        ("node", _RE_NODE),
        ("sampler_banner", _RE_SAMPLER),
        ("max_tree_depth", _RE_MAX_TREE),
        ("chunk_count_banner", _RE_CHUNK_COUNT),
        ("chunk_id_banner", _RE_CHUNK_ID),
    ]:
        m = pat.search(text)
        if m:
            val = m.group(1)
            try:
                rec[key] = int(val)
            except ValueError:
                rec[key] = val

    # Partition: prefer the JAX-device signal printed in the banner;
    # fall back to node-prefix heuristic for older logs.
    m = _RE_JAX_DEVICES.search(text)
    if m:
        devstr = m.group(1)
        if "Cuda" in devstr:
            rec["partition"] = "gpu"
        elif "Cpu" in devstr:
            rec["partition"] = "comp"
        else:
            rec["partition"] = "unknown"
    else:
        node_str = rec.get("node", "")
        if isinstance(node_str, str) and len(node_str) >= 3:
            rec["partition"] = _NODE_PARTITION_PREFIX.get(
                node_str[:3],
                "unknown",
            )

    # fit_batch_hierarchical entry
    m = _RE_FBH_ENTERED.search(text)
    if m:
        rec.update(
            {
                "model": m.group(1),
                "sampler": m.group(2),
                "n_chains": int(m.group(3)),
                "n_tune": int(m.group(4)),
                "n_draws": int(m.group(5)),
                "target_accept": float(m.group(6)),
                "P_at_entry": int(m.group(7)),
            }
        )
    m = _RE_COHORT.search(text)
    if m:
        rec["P_cohort"] = int(m.group(1))
        rec["n_trials"] = int(m.group(2))

    # Variant 4 chunking
    m = _RE_CHUNK_SUMMARY.search(text)
    if m:
        rec["chunk_id"] = int(m.group(1))
        rec["chunk_count"] = int(m.group(2))
        rec["chunk_n_PS"] = int(m.group(3))
        rec["cohort_n_PS_full"] = int(m.group(4))

    # Phase timings
    m = _RE_WARMUP_START.search(text)
    if m:
        rec["warmup_num_steps"] = int(m.group(1))
        rec["warmup_target_accept"] = float(m.group(2))
    m = _RE_WARMUP_DONE.search(text)
    if m:
        rec["window_adaptation_s"] = float(m.group(1))
    m = _RE_SAMPLE_DONE.search(text)
    if m:
        rec["sample_loop_s"] = float(m.group(1))

    # Diagnostic blocks
    rec.update({f"warmup_{k}": v for k, v in _parse_diag_block(text, "warmup").items()})
    rec.update({f"sample_{k}": v for k, v in _parse_diag_block(text, "sample").items()})

    # Outcome flags
    rec["completed_warmup"] = (
        "warmup_window_adaptation_s" in rec or "window_adaptation_s" in rec
    )
    rec["completed_sample"] = "sample_loop_s" in rec
    combined = text + "\n" + err_text
    if "DUE TO TIME LIMIT" in combined:
        rec["outcome"] = "TIME_LIMIT"
    elif "DUE to SIGNAL" in combined or "DUE TO SIGNAL" in combined:
        rec["outcome"] = "SCANCEL"
    elif "BENCHMARK COMPLETE" in text and rec["completed_sample"]:
        rec["outcome"] = "OK"
    elif "BENCHMARK FAILED" in text:
        rec["outcome"] = "FAILED"
    else:
        rec["outcome"] = "UNKNOWN"

    return rec


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "log_path",
    "job_id",
    "array_job_id",
    "array_task_id",
    "node",
    "partition",
    "outcome",
    "sampler",
    "model",
    "P_cohort",
    "n_trials",
    "n_chains",
    "n_tune",
    "n_draws",
    "target_accept",
    "max_tree_depth",
    "chunk_id",
    "chunk_count",
    "chunk_n_PS",
    "window_adaptation_s",
    "sample_loop_s",
    "warmup_tree_depth_mean",
    "warmup_tree_depth_max",
    "warmup_tree_depth_saturated_frac",
    "warmup_leapfrog_per_step_total",
    "warmup_leapfrog_per_step_mean",
    "warmup_accept_rate_mean",
    "warmup_accept_rate_final50",
    "warmup_divergent_count",
    "warmup_divergent_rate",
    "warmup_energy_dE_mean",
    "warmup_energy_dE_max",
    "sample_tree_depth_mean",
    "sample_tree_depth_saturated_frac",
    "sample_divergent_count",
    "sample_divergent_rate",
    "sample_accept_rate_mean",
]


def _to_csv_row(rec: dict[str, Any]) -> dict[str, str]:
    return {
        col: ("" if col not in rec or rec[col] is None else str(rec[col]))
        for col in _CSV_COLUMNS
    }


def _emit_markdown(records: list[dict[str, Any]]) -> str:
    rows = sorted(
        records,
        key=lambda r: (
            0 if r.get("outcome") == "OK" else 1,
            r.get("window_adaptation_s") or float("inf"),
        ),
    )
    md_cols = [
        "outcome",
        "partition",
        "P_cohort",
        "max_tree_depth",
        "chunk",
        "warmup_s",
        "sample_s",
        "saturated_frac",
        "divergent",
        "lf_total",
        "node",
        "log",
    ]
    out = [
        "# Phase 14.2 variant comparison",
        "",
        f"{len(rows)} bench runs aggregated.",
        "",
        "| " + " | ".join(md_cols) + " |",
        "|" + "|".join(["---"] * len(md_cols)) + "|",
    ]
    for r in rows:
        chunk = (
            f"{r.get('chunk_id', '?')}/{r.get('chunk_count', '?')}"
            if r.get("chunk_count") and r["chunk_count"] > 1
            else "full"
        )
        warm = r.get("window_adaptation_s")
        sample = r.get("sample_loop_s")
        sat = r.get("warmup_tree_depth_saturated_frac")
        div = r.get("warmup_divergent_count")
        lf = r.get("warmup_leapfrog_per_step_total")
        out.append(
            "| {outcome} | {part} | {P} | mtd={mtd} | {chunk} | {warm} | "
            "{sample} | {sat} | {div} | {lf} | {node} | {log} |".format(
                outcome=r.get("outcome", "-"),
                part=r.get("partition", "-"),
                P=r.get("P_cohort", "-"),
                mtd=r.get("max_tree_depth", "-"),
                chunk=chunk,
                warm=f"{warm:.0f}s" if warm is not None else "-",
                sample=f"{sample:.0f}s" if sample is not None else "-",
                sat=f"{sat:.3f}" if sat is not None else "-",
                div=f"{int(div)}" if div is not None else "-",
                lf=f"{int(lf):,}" if lf is not None else "-",
                node=r.get("node", "-"),
                log=Path(r["log_path"]).name,
            )
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 14.2 variant-comparison bench logs.",
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help=(
            "Specific .out files to parse.  If omitted, defaults to "
            "cluster/logs/bench14*.out (excludes failed-import logs by "
            "requiring the file have at least 30 lines)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "diagnostics",
        help="Directory to write variant_comparison.csv and .md.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=30,
        help=(
            "Skip log files shorter than this (filters out env-failure logs "
            "that died before any diagnostics — default 30)."
        ),
    )
    args = parser.parse_args()

    if args.logs:
        log_paths = list(args.logs)
    else:
        log_paths = sorted(
            (PROJECT_ROOT / "cluster" / "logs").glob("bench14*.out"),
        )

    if not log_paths:
        print("No log files matched.", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in log_paths:
        try:
            line_count = sum(
                1 for _ in path.open("r", encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            skipped.append(f"{path}: {exc}")
            continue
        if line_count < args.min_lines:
            skipped.append(f"{path}: only {line_count} lines (< {args.min_lines})")
            continue
        try:
            records.append(parse_log(path))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path}: parse failed: {exc!r}")

    if skipped:
        print(f"Skipped {len(skipped)} log(s):", file=sys.stderr)
        for s in skipped:
            print(f"  {s}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "variant_comparison.csv"
    md_path = args.output_dir / "variant_comparison.md"
    json_path = args.output_dir / "variant_comparison.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(_to_csv_row(rec))

    with md_path.open("w", encoding="utf-8") as f:
        f.write(_emit_markdown(records))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"Wrote {len(records)} records to:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
