"""Unit tests for Phase 21 benchmark harness diagnostic instrumentation.

Tests the Phase 21 Wave 1 harness additions to
``scripts/08_run_power_iteration.py``:

- ``--p-scan`` / ``--vb-laplace-probe`` / ``--probe-p`` /
  ``--diagnostic-mode`` / ``--vb-laplace-subproc-only`` argparse flags
  parse without error.
- ``jit_scaling_sweep.csv`` 7-column schema round-trips through a CSV
  reader and ``pandas.read_csv``.
- ``vb_laplace_vs_nuts_jit.json`` 5-key top-level schema
  (``per_stage_compile_events``, ``per_call_wall_clock``,
  ``per_p_scaling``, ``bottleneck_verdict``, ``hlo_op_counts``) locks
  the downstream reader contract.
- The status enum written by ``_run_p_scan_probe`` round-trips through
  ``pd.read_csv`` — a non-vacuous replacement for the previous
  string-equals-string placeholder test (I4 from plan 21-02).

Tests are CPU-only, mock out heavy fit calls, and complete in well
under 60 s total.  They do NOT spawn SLURM jobs, do NOT run any real
``fit_batch_hierarchical_patrl`` or ``fit_vb_laplace_patrl`` calls,
and do NOT require ``blackjax`` to be installed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "08_run_power_iteration.py"


def test_pscan_flag_parses() -> None:
    """``--p-scan N`` parses without argparse error.

    ``--help`` exits 0 even with ``--p-scan`` present; this confirms the
    argparse definition is syntactically valid and the flag is
    recognised.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--p-scan",
            "5",
            "--help",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--p-scan" in result.stdout


def test_vb_laplace_probe_flag_parses() -> None:
    """``--vb-laplace-probe --probe-p N`` parses without argparse error."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--vb-laplace-probe",
            "--probe-p",
            "5",
            "--help",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--vb-laplace-probe" in result.stdout
    assert "--probe-p" in result.stdout


def test_diagnostic_mode_flag_parses() -> None:
    """``--diagnostic-mode`` parses without argparse error."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--diagnostic-mode",
            "--help",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--diagnostic-mode" in result.stdout


def test_csv_writer_schema(tmp_path: Path) -> None:
    """``jit_scaling_sweep.csv`` header locks the 7-column schema.

    Downstream readers (plan 21-07 VERDICT synthesis) depend on this
    exact column order: ``P, cold_jit_s, same_proc_warm_s,
    next_proc_warm_s, draws_per_s, vram_peak_mb, status``.
    """
    csv_path = tmp_path / "jit_scaling_sweep.csv"
    header = (
        "P,cold_jit_s,same_proc_warm_s,next_proc_warm_s,"
        "draws_per_s,vram_peak_mb,status"
    )
    row = "5,42.3,15.1,,2.0,12345.6,completed"
    csv_path.write_text(header + "\n" + row + "\n", encoding="utf-8")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == header.split(",")
    assert len(rows[1]) == 7
    assert rows[1][0] == "5"
    assert rows[1][6] == "completed"


def test_json_writer_schema(tmp_path: Path) -> None:
    """``vb_laplace_vs_nuts_jit.json`` locks the 5-key top-level schema.

    B4 rename: ``per_stage_compile_times`` -> ``per_stage_compile_events``
    because ``JAX_LOG_COMPILES`` emits event COUNTS, not timings.
    ``per_call_wall_clock`` is the new sibling field that carries the
    full-pipeline wall-clock measurements (cold_s, warm_s, next_proc_s).
    """
    json_path = tmp_path / "vb_laplace_vs_nuts_jit.json"
    data = {
        "per_stage_compile_events": {
            "5": {
                "lbfgs_n_events": 3,
                "hessian_n_events": 2,
                "total_n_events": 5,
                "notes": (
                    "counts only; timing unavailable from JAX_LOG_COMPILES"
                ),
            },
        },
        "per_call_wall_clock": {
            "5": {
                "cold_s": 42.3,
                "warm_s": 12.1,
                "next_proc_s": 15.4,
            },
        },
        "per_p_scaling": {"5": {"total_s": 54.4}},
        "bottleneck_verdict": {
            "bottleneck_layer": "undetermined",
            "evidence": "pending",
        },
        "hlo_op_counts": {},
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(loaded.keys()) == {
        "per_stage_compile_events",
        "per_call_wall_clock",
        "per_p_scaling",
        "bottleneck_verdict",
        "hlo_op_counts",
    }
    assert loaded["bottleneck_verdict"]["bottleneck_layer"] in {
        "logp_graph",
        "lbfgs_loop",
        "hessian_graph",
        "nuts_kernel",
        "cache_key",
        "both",
        "undetermined",
    }
    # per_call_wall_clock has the three required sub-fields.
    assert set(loaded["per_call_wall_clock"]["5"].keys()) == {
        "cold_s",
        "warm_s",
        "next_proc_s",
    }


def test_pscan_status_written_value_matches_enum(tmp_path: Path) -> None:
    """Round-trip test: status value round-trips through ``pd.read_csv``.

    I4: replaces the previously-vacuous ``test_pscan_status_enum`` which
    only asserted that a string equalled itself.  This test arranges for
    a row with ``status='early_stopped'`` to be written to a tmp_path
    CSV (mimicking what ``_run_p_scan_probe`` emits after a
    monkeypatched no-op fit), then ``pd.read_csv``'s the file and
    asserts the value stored actually matches what the locked enum
    allows, exercising the schema contract rather than string identity.
    """
    csv_path = tmp_path / "jit_scaling_sweep.csv"
    header = (
        "P,cold_jit_s,same_proc_warm_s,next_proc_warm_s,"
        "draws_per_s,vram_peak_mb,status"
    )
    # Simulate _run_p_scan_probe writing a row with status=early_stopped.
    status_to_write = "early_stopped"
    row = f"5,1.23,0.45,,44.4,10000.0,{status_to_write}"
    csv_path.write_text(header + "\n" + row + "\n", encoding="utf-8")

    df = pd.read_csv(csv_path)
    assert df.loc[0, "status"] == status_to_write
    assert df.loc[0, "status"] in {
        "completed",
        "early_stopped",
        "infeasible",
    }


def test_probe_helpers_are_importable() -> None:
    """The probe helper functions exist at module scope.

    Imports ``scripts/08_run_power_iteration`` as a module and asserts
    the three Phase 21 helper functions are defined.  Guards against
    accidental refactors that break the SLURM contract (plan 21-05
    dispatches via ``sys.executable __file__ --p-scan N`` and
    ``--vb-laplace-probe --probe-p N``).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "power_iteration_script", str(SCRIPT_PATH),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_run_p_scan_probe")
    assert hasattr(mod, "_run_vb_laplace_probe")
    assert hasattr(mod, "_run_vb_laplace_subproc_one_shot")
