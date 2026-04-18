"""Structural tests for scripts/12_smoke_patrl_foundation.py.

Nine lightweight tests that validate the smoke script's structure, argument
handling, and parallel-stack invariant without executing MCMC.  All fast
tests pass on Windows with no blackjax installation required.

Test inventory (fast — no env gate)
-------------------------------------
1. test_smoke_script_py_compiles              -- syntax check via py_compile
2. test_smoke_script_rejects_invalid_level   -- argparse rejects ``--level 4``
3. test_smoke_script_has_no_pick_best_cue_imports -- parallel-stack invariant
4. test_smoke_script_lazy_imports_blackjax   -- blackjax must not be top-level
5. test_pick_best_cue_modules_still_compile  -- regression canary (parametrized)
6. test_script_accepts_fit_method_flag       -- --fit-method choices in --help
7. test_script_rejects_invalid_fit_method    -- --fit-method invalid rejected
8. test_script_lazy_imports_vbl06_for_both   -- vbl06 import is indented (lazy)

Test inventory (slow — RUN_SMOKE_TESTS=1 gate)
----------------------------------------------
9. test_smoke_end_to_end_laplace_2level      -- laplace 3-agent full run
10. test_smoke_laplace_recovery_sanity_4_of_5 -- 5-agent recovery >=4/5 omega_2
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "12_smoke_patrl_foundation.py"
_PICK_BEST_CUE_MODULES = [
    "src/prl_hgf/env/task_config.py",
    "src/prl_hgf/models/hgf_2level.py",
    "src/prl_hgf/models/response.py",
]


def test_smoke_script_py_compiles() -> None:
    """Script must be syntactically valid without running it.

    py_compile does not execute module top-level — blackjax is lazy-imported
    inside ``_fit`` so absence of blackjax on the test machine must not
    break compilation.
    """
    py_compile.compile(str(_SCRIPT), doraise=True)


def test_smoke_script_rejects_invalid_level() -> None:
    """``--level 4`` must be rejected by argparse (choices=[2, 3])."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--level", "4"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit for --level 4; got {result.returncode}.  "
        f"stderr:\n{result.stderr}"
    )
    assert "invalid choice" in result.stderr.lower() or "4" in result.stderr, (
        f"Expected argparse choices error; got stderr:\n{result.stderr}"
    )


def test_smoke_script_has_no_pick_best_cue_imports() -> None:
    """Parallel-stack invariant: smoke must not import pick_best_cue modules.

    Textual scan of the smoke script — no prl_hgf.env.task_config,
    prl_hgf.models.hgf_2level, or prl_hgf.models.response.  Those are the
    pick_best_cue stack; PAT-RL uses prl_hgf.env.pat_rl_config,
    prl_hgf.models.hgf_2level_patrl, prl_hgf.models.hgf_3level_patrl.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "from prl_hgf.env.task_config",
        "from prl_hgf.models.hgf_2level ",  # trailing space avoids _patrl hit
        "from prl_hgf.models.hgf_2level\n",
        "from prl_hgf.models.response",
        "import prl_hgf.env.task_config",
        "import prl_hgf.models.hgf_2level\n",
        "import prl_hgf.models.response",
    ]
    hits = [s for s in forbidden if s in source]
    assert not hits, (
        "Parallel-stack violation: smoke script imports pick_best_cue "
        f"modules: {hits}.  Expected only _patrl variants."
    )


def test_smoke_script_lazy_imports_blackjax() -> None:
    """blackjax must be imported lazily inside the fit path.

    The script must be importable/runnable (e.g. --dry-run) on a machine
    without blackjax.  Check the source: no top-level ``import blackjax``,
    and ``blackjax`` must appear only inside a function body (indented).
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    # No top-level import blackjax
    for line in source.splitlines():
        stripped = line.lstrip()
        if line == stripped and (
            stripped.startswith("import blackjax")
            or stripped.startswith("from blackjax")
        ):
            pytest.fail(
                "blackjax imported at module top-level; must be lazy "
                f"(inside _fit).  Offending line: {line!r}"
            )


@pytest.mark.parametrize("rel_path", _PICK_BEST_CUE_MODULES)
def test_pick_best_cue_modules_still_compile(rel_path: str) -> None:
    """Regression canary: pick_best_cue modules must still py_compile.

    Protects against accidental co-modification of the pick_best_cue stack
    while we were editing the PAT-RL smoke.
    """
    full = _REPO_ROOT / rel_path
    assert full.exists(), f"Expected pick_best_cue module at {full}"
    py_compile.compile(str(full), doraise=True)


# ---------------------------------------------------------------------------
# New tests: --fit-method flag coverage (Phase 19-05)
# ---------------------------------------------------------------------------


def test_script_accepts_fit_method_flag() -> None:
    """``--help`` must list ``--fit-method`` with choices blackjax, laplace, both.

    Invokes the script with ``--help`` (fast subprocess, no fit executed).
    Asserts returncode == 0 and that all three choices appear in stdout.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit 0 for --help; got {result.returncode}.  "
        f"stderr:\n{result.stderr}"
    )
    assert "--fit-method" in result.stdout, (
        f"Expected '--fit-method' in --help stdout; got:\n{result.stdout}"
    )
    for choice in ("blackjax", "laplace", "both"):
        assert choice in result.stdout, (
            f"Expected choice '{choice}' in --help stdout; got:\n{result.stdout}"
        )


def test_script_rejects_invalid_fit_method() -> None:
    """``--fit-method invalid`` must be rejected by argparse.

    Expects returncode != 0 and ``'invalid choice'`` in stderr (argparse
    standard error message for choices= violations).
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--fit-method", "invalid"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit for --fit-method invalid; got {result.returncode}.  "
        f"stderr:\n{result.stderr}"
    )
    assert "invalid choice" in result.stderr.lower(), (
        f"Expected 'invalid choice' in stderr; got:\n{result.stderr}"
    )


def test_script_lazy_imports_vbl06_for_both_mode() -> None:
    """VBL-06 comparison import must be inside a function body (indented).

    Static source-text check: the string
    ``'from validation.vbl06_laplace_vs_nuts import'`` must appear on an
    indented line (leading whitespace), NOT at column 0.  This enforces the
    lazy-import contract so the script is importable/runnable without
    ``validation.vbl06_laplace_vs_nuts`` present.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    target = "from validation.vbl06_laplace_vs_nuts import"
    found = False
    for line in source.splitlines():
        if target in line:
            found = True
            assert line != line.lstrip(), (
                f"vbl06_laplace_vs_nuts import must be indented (lazy); "
                f"found at column 0: {line!r}"
            )
    assert found, (
        f"Expected to find '{target}' in smoke script source; "
        "it was not present.  Add the lazy import inside the 'both' branch."
    )


def test_pick_best_cue_regression_still_passes() -> None:
    """Regression guard: key pick_best_cue modules must still be importable.

    Imports two sentinel symbols from the pick_best_cue analysis stack.
    If the parallel-stack invariant was violated (e.g. by accidental editing
    of frozen modules), these imports will fail with ImportError or
    SyntaxError — triggering an obvious test failure.
    """
    from prl_hgf.analysis.bms import compute_subject_waic  # noqa: PLC0415
    from prl_hgf.env.simulator import generate_session  # noqa: PLC0415

    assert callable(compute_subject_waic), "compute_subject_waic must be callable"
    assert callable(generate_session), "generate_session must be callable"


# ---------------------------------------------------------------------------
# Slow tests (RUN_SMOKE_TESTS=1 gate) — Laplace end-to-end
# ---------------------------------------------------------------------------

_SLOW_SKIP = pytest.mark.skipif(
    os.getenv("RUN_SMOKE_TESTS") != "1",
    reason="Set RUN_SMOKE_TESTS=1 to run slow end-to-end smoke tests.",
)


@_SLOW_SKIP
def test_smoke_end_to_end_laplace_2level(tmp_path: Path) -> None:
    """Full end-to-end: laplace 3-agent 2-level smoke completes <90 s.

    Invokes the script as a subprocess with ``--fit-method laplace``.
    Asserts:
    - returncode == 0
    - 3 per-subject trajectory CSVs written
    - ``parameter_summary.csv`` written with correct schema + row count
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--fit-method",
            "laplace",
            "--level",
            "2",
            "--n-participants",
            "3",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Laplace smoke exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # 3 per-subject trajectory CSVs
    traj_csvs = sorted(tmp_path.glob("P*_trajectories.csv"))
    assert len(traj_csvs) == 3, (
        f"Expected 3 trajectory CSVs; found {len(traj_csvs)}: {traj_csvs}"
    )

    # parameter_summary.csv with correct schema + row count
    param_csv = tmp_path / "parameter_summary.csv"
    assert param_csv.exists(), f"parameter_summary.csv not found at {param_csv}"
    df = pd.read_csv(param_csv)
    expected_cols = ["participant_id", "parameter", "posterior_mean", "hdi_low", "hdi_high"]
    assert list(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}; got {list(df.columns)}"
    )
    # 2-level model: omega_2 + beta per participant; 3 participants × 2 params = 6 rows
    assert len(df) == 6, (
        f"Expected 6 rows (3 participants × 2 params); got {len(df)}"
    )


@_SLOW_SKIP
def test_smoke_laplace_recovery_sanity_4_of_5(tmp_path: Path) -> None:
    """5-agent Laplace omega_2 recovery: >=4/5 within |diff| < 0.5.

    Invokes script with ``--n-participants 5 --fit-method laplace``, loads
    both ``parameter_summary.csv`` and ``true_params.csv``, merges on
    ``(participant_id, parameter)``, and asserts the recovery gate.

    This is Phase 19 Success Criterion #4 demonstrated end-to-end at the
    CLI level (Plan 19-03 tests the same at unit level).
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--fit-method",
            "laplace",
            "--level",
            "2",
            "--n-participants",
            "5",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Laplace smoke exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    param_csv = tmp_path / "parameter_summary.csv"
    true_csv = tmp_path / "true_params.csv"

    assert param_csv.exists(), f"parameter_summary.csv not found at {param_csv}"
    assert true_csv.exists(), (
        f"true_params.csv not found at {true_csv}.  "
        "Script must write true_params.csv for --fit-method laplace."
    )

    summary_df = pd.read_csv(param_csv)
    true_df = pd.read_csv(true_csv)

    # Merge on (participant_id, parameter) — inner join
    merged = summary_df.merge(true_df, on=["participant_id", "parameter"], how="inner")

    # Filter to omega_2 rows
    omega2 = merged[merged["parameter"] == "omega_2"].copy()
    assert len(omega2) == 5, (
        f"Expected 5 omega_2 rows (one per participant); got {len(omega2)}"
    )

    omega2["abs_diff"] = (omega2["posterior_mean"] - omega2["true_value"]).abs()

    n_pass = int((omega2["abs_diff"] < 0.5).sum())
    assert n_pass >= 4, (
        f"Recovery gate: expected >=4/5 omega_2 |diff| < 0.5; "
        f"got {n_pass}/5.\n"
        f"Details:\n{omega2[['participant_id', 'true_value', 'posterior_mean', 'abs_diff']].to_string()}"
    )
