"""Power analysis configuration factory and dataclasses.

Provides :class:`PowerConfig` for loading BFDA grid parameters from YAML, and
:func:`make_power_config` for producing frozen :class:`AnalysisConfig` copies
with overridden sample size and effect size without any file I/O.

Notes
-----
- :func:`make_power_config` uses :func:`dataclasses.replace` bottom-up and
  never reads or writes files.
- :func:`load_power_config` reads only the ``power:`` top-level key from the
  YAML file; it does not re-parse task, simulation, or fitting sections.
- The existing :func:`~prl_hgf.env.task_config.load_config` is unaffected by
  the ``power:`` section because it simply ignores unknown top-level keys.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import config as _cfg
from prl_hgf.env.task_config import AnalysisConfig, SessionConfig, SimulationConfig

_DEFAULT_CONFIG_PATH = _cfg.CONFIGS_DIR / "prl_analysis.yaml"

# ---------------------------------------------------------------------------
# PowerConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerConfig:
    """Grid and seed parameters for the BFDA power analysis loop.

    Parameters
    ----------
    n_per_group_grid : list[int]
        Sample sizes per group to sweep over.
    effect_size_grid : list[float]
        Effect size deltas (in omega_2 units) to sweep over.
    n_iterations : int
        Number of simulated datasets per grid cell (must be >= 1).
    master_seed : int
        Master RNG seed for :class:`numpy.random.SeedSequence` spawning.
    n_jobs : int
        Maximum parallel jobs for the SLURM array throttle (must be >= 1).
    bf_threshold : float
        Bayes factor threshold for declaring evidence (must be > 0).
    """

    n_per_group_grid: list[int]
    effect_size_grid: list[float]
    n_iterations: int
    master_seed: int
    n_jobs: int
    bf_threshold: float

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(
                "PowerConfig: n_iterations must be >= 1, "
                f"got {self.n_iterations}."
            )
        if self.n_jobs < 1:
            raise ValueError(
                f"PowerConfig: n_jobs must be >= 1, got {self.n_jobs}."
            )
        if self.bf_threshold <= 0.0:
            raise ValueError(
                "PowerConfig: bf_threshold must be > 0, "
                f"got {self.bf_threshold}."
            )


# ---------------------------------------------------------------------------
# YAML loader for power section
# ---------------------------------------------------------------------------


def load_power_config(path: Path | None = None) -> PowerConfig:
    """Load the ``power:`` section from the PRL analysis YAML file.

    Parameters
    ----------
    path : Path or None, optional
        Path to the YAML config file. Defaults to
        ``CONFIGS_DIR / "prl_analysis.yaml"``.

    Returns
    -------
    PowerConfig
        Validated power analysis configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If the ``power:`` top-level key is absent or any field fails
        validation.

    Examples
    --------
    >>> from prl_hgf.power.config import load_power_config
    >>> pc = load_power_config()
    >>> pc.bf_threshold
    10.0
    """
    resolved = path if path is not None else _DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. "
            f"Expected at {_DEFAULT_CONFIG_PATH}."
        )
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if "power" not in raw:
        raise ValueError(
            f"Config file '{resolved}' is missing the required 'power:' "
            "top-level key. Add a 'power:' section to the YAML file."
        )

    pw = raw["power"]
    return PowerConfig(
        n_per_group_grid=[int(v) for v in pw["n_per_group_grid"]],
        effect_size_grid=[float(v) for v in pw["effect_size_grid"]],
        n_iterations=int(pw["n_iterations"]),
        master_seed=int(pw["master_seed"]),
        n_jobs=int(pw["n_jobs"]),
        bf_threshold=float(pw["bf_threshold"]),
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_power_config(
    base: AnalysisConfig,
    n_per_group: int,
    effect_size_delta: float,
    master_seed: int,
) -> AnalysisConfig:
    """Return a frozen AnalysisConfig with overridden sample size and effect.

    Applies ``n_per_group`` and ``master_seed`` to the simulation config and
    shifts the psilocybin group's ``omega_2_deltas`` by ``effect_size_delta``
    using :func:`dataclasses.replace` — no mutation of ``base`` occurs.

    This function performs no file I/O. All YAML loading must happen before
    calling this function (e.g. via :func:`~prl_hgf.env.task_config.load_config`).

    Parameters
    ----------
    base : AnalysisConfig
        The baseline frozen config from which to derive the power variant.
    n_per_group : int
        Override for ``simulation.n_participants_per_group``.
    effect_size_delta : float
        Additive shift applied to each element of the psilocybin group's
        ``omega_2_deltas``. The placebo group is left unchanged.
    master_seed : int
        Override for ``simulation.master_seed``.

    Returns
    -------
    AnalysisConfig
        A new frozen :class:`~prl_hgf.env.task_config.AnalysisConfig` with the
        requested overrides applied. ``base.task`` and ``base.fitting`` are
        identical to those in ``base`` (same objects, not copies).

    Examples
    --------
    >>> from prl_hgf.env.task_config import load_config
    >>> from prl_hgf.power.config import make_power_config
    >>> base = load_config()
    >>> variant = make_power_config(base, n_per_group=20, effect_size_delta=0.5,
    ...                             master_seed=9999)
    >>> variant.simulation.n_participants_per_group
    20
    """
    sim: SimulationConfig = base.simulation

    # Build new session_deltas dict without mutating the original
    new_deltas: dict[str, SessionConfig] = {}
    for group_name, sess in sim.session_deltas.items():
        if group_name == "psilocybin":
            shifted_omega_2 = [
                d + effect_size_delta for d in sess.omega_2_deltas
            ]
            new_deltas[group_name] = dataclasses.replace(
                sess, omega_2_deltas=shifted_omega_2
            )
        else:
            new_deltas[group_name] = sess

    new_sim = dataclasses.replace(
        sim,
        n_participants_per_group=n_per_group,
        master_seed=master_seed,
        session_deltas=new_deltas,
    )
    return dataclasses.replace(base, simulation=new_sim)
