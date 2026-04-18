"""Structural tests for scripts/12_smoke_patrl_foundation.py.

Five lightweight tests that validate the smoke script's structure, argument
handling, and parallel-stack invariant without executing MCMC.  All tests
pass on Windows with no blackjax installation required.

Test inventory
--------------
1. test_smoke_script_py_compiles              -- syntax check via py_compile
2. test_smoke_script_rejects_invalid_level   -- argparse rejects ``--level 4``
3. test_smoke_script_has_no_pick_best_cue_imports -- parallel-stack invariant
4. test_smoke_script_lazy_imports_blackjax   -- blackjax must not be top-level
5. test_pick_best_cue_modules_still_compile  -- regression canary (parametrized)
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

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
