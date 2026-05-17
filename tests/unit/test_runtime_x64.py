"""Unit tests for prl_hgf.runtime.set_x64 (GUARD-05, P6 prevention)."""

from __future__ import annotations

import subprocess
import sys


def test_set_x64_enables():
    """set_x64(True) enables fp64."""
    # Run in subprocess to avoid contaminating test process JAX state.
    code = (
        "from prl_hgf.runtime import set_x64; "
        "set_x64(True); "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_set_x64_disables():
    """set_x64(False) disables fp64."""
    code = (
        "from prl_hgf.runtime import set_x64; "
        "set_x64(False); "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_x64_stays_disabled_after_fitting_import():
    """P6 prevention: importing fitting.hierarchical does not flip x64 back on.

    PyTensor's JAX linker silently sets jax_enable_x64=True on import.
    set_x64(False) called AFTER that import must successfully force it
    back to False.
    """
    code = (
        "import prl_hgf.fitting.hierarchical; "  # triggers PyTensor import
        "from prl_hgf.runtime import set_x64; "
        "set_x64(False); "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False, "
        "'x64 was flipped back to True after set_x64(False)'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"P6 silent-flip detected!\nstdout={result.stdout}\nstderr={result.stderr}"
    )
