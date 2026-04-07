"""Power analysis subpackage for the PRL HGF pipeline.

Provides config factory functions and dataclasses for BFDA-based power
analysis. All public symbols are re-exported here.

Notes
-----
The ``power/`` subpackage wraps the existing pipeline without modifying any
existing modules. Config factories use :func:`dataclasses.replace` and never
perform file I/O (YAML loading is isolated to :func:`load_power_config`).
"""

from __future__ import annotations

from prl_hgf.power.config import PowerConfig, load_power_config, make_power_config

__all__ = ["make_power_config", "PowerConfig", "load_power_config"]
