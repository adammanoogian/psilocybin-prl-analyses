"""Path resolution for prl_hgf without a hard dependency on ``config.py``.

Historically the library did ``from config import CONFIGS_DIR``, which only
works when a repo-root ``config.py`` happens to be on ``sys.path`` (the
toolbox repo itself, a study repo's ``analysis/`` dir, or a sibling repo
shadowing it on purpose). Installed as a package with no such module on the
path, every ``prl_hgf.env`` import crashed.

Resolution order (first hit wins), per constant:

1. Environment variable (``PRL_HGF_CONFIGS_DIR`` / ``PRL_HGF_PROJECT_ROOT``)
   — explicit override.
2. A ``config`` module importable from ``sys.path`` exposing the constant —
   preserves the historical behaviour for the toolbox repo, study repos, and
   deliberate shadowing (e.g. dcm_hgf_mixed_models).
3. The checkout containing this package (``src`` layout: two levels above
   the package) — correct for editable installs of the toolbox itself.
"""

from __future__ import annotations

import os
from pathlib import Path

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, config_attr: str, fallback: Path) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    try:
        import config  # noqa: PLC0415 — deliberate late, optional import

        value = getattr(config, config_attr, None)
        if value is not None:
            return Path(value)
    except ImportError:
        pass
    return fallback


PROJECT_ROOT: Path = _resolve(
    "PRL_HGF_PROJECT_ROOT", "PROJECT_ROOT", _CHECKOUT_ROOT
)
CONFIGS_DIR: Path = _resolve(
    "PRL_HGF_CONFIGS_DIR", "CONFIGS_DIR", _CHECKOUT_ROOT / "configs"
)
