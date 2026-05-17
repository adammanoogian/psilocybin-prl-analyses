"""Runtime configuration for JAX environment.

Centralizes JAX config toggles that must be set BEFORE array creation
and asserted AFTER any imports that might silently flip them.
"""

from __future__ import annotations

import os


def set_x64(enabled: bool = True) -> None:
    """Set JAX fp64 mode and assert it took effect.

    Must be called before any ``jnp.array`` creation. Sets the env var
    (reliable pre-import channel) AND calls ``jax.config.update`` (for
    post-import flip-back after PyTensor import, Pitfall P6).

    After setting, asserts that ``jax.config.read("jax_enable_x64")``
    matches ``enabled``. If it does not, raises ``RuntimeError`` with a
    diagnostic message -- this catches the PyTensor silent-flip scenario.

    Parameters
    ----------
    enabled : bool, default True
        Whether to enable (True) or disable (False) fp64.

    Raises
    ------
    RuntimeError
        If the assertion fails (x64 flag does not match requested state).
    """
    os.environ["JAX_ENABLE_X64"] = "1" if enabled else "0"

    import jax

    jax.config.update("jax_enable_x64", enabled)

    actual = jax.config.read("jax_enable_x64")
    if actual != enabled:
        msg = (
            f"set_x64({enabled}) failed: jax.config.read('jax_enable_x64') "
            f"returned {actual!r}. This typically means another import "
            f"(e.g., PyTensor's jax linker) silently flipped x64. "
            f"Call set_x64() AFTER all such imports."
        )
        raise RuntimeError(msg)
