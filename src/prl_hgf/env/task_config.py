"""Task configuration loading and validation for the PRL pick_best_cue pipeline.

Loads ``configs/prl_analysis.yaml`` and returns a validated, typed
:class:`AnalysisConfig` hierarchy. All task structure (phases, reward
probabilities, trial counts) and analysis parameters (simulation, fitting,
priors) live exclusively in that YAML file — nothing is hardcoded here.

Three-layer naming convention used throughout:
- Math symbols (``omega_2``, ``kappa``) appear in model internals only.
- Descriptive names (``tonic_volatility``, ``coupling_strength``) appear
  at API boundaries.
- Domain English appears in scripts and notebooks.
Within this module we use the YAML key names verbatim so that config keys
map directly to dataclass fields without translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from prl_hgf._paths import CONFIGS_DIR

_DEFAULT_CONFIG_PATH = CONFIGS_DIR / "prl_analysis.yaml"


# ---------------------------------------------------------------------------
# Task-level dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for one task phase.

    Parameters
    ----------
    name : str
        Phase identifier, e.g. ``"acquisition_1"``.
    phase_type : str
        Either ``"stable"`` or ``"volatile"``.
    n_trials : int
        Number of trials in this phase (must be >= 1).
    cue_probs : list[float]
        Reward probability for each cue. Must sum to > 0, each element in
        [0.0, 1.0], and the list length must equal the parent
        :attr:`TaskConfig.n_cues`.
    """

    name: str
    phase_type: str
    n_trials: int
    cue_probs: list[float]

    @property
    def phase_label(self) -> str:
        """Human-readable label identical to ``phase_type``.

        Returns
        -------
        str
            Either ``"stable"`` or ``"volatile"``.
        """
        return self.phase_type

    def __post_init__(self) -> None:
        valid_types = {"stable", "volatile"}
        if self.phase_type not in valid_types:
            raise ValueError(
                f"PhaseConfig '{self.name}': phase_type must be one of "
                f"{valid_types}, got '{self.phase_type}'."
            )
        if self.n_trials < 1:
            raise ValueError(
                f"PhaseConfig '{self.name}': n_trials must be >= 1, "
                f"got {self.n_trials}."
            )
        for i, p in enumerate(self.cue_probs):
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"PhaseConfig '{self.name}': cue_probs[{i}] must be in "
                    f"[0.0, 1.0], got {p}."
                )


@dataclass(frozen=True)
class TransferConfig:
    """Configuration for the transfer phase appended to each set.

    The transfer phase uses equal cue probabilities to assess generalisation
    without the reversal schedule.

    Parameters
    ----------
    phase_type : str
        Either ``"stable"`` or ``"volatile"``.
    n_trials : int
        Number of transfer trials per set (must be >= 1).
    cue_probs : list[float]
        Reward probability for each cue, each element in [0.0, 1.0].
    """

    phase_type: str
    n_trials: int
    cue_probs: list[float]

    @property
    def phase_label(self) -> str:
        """Human-readable label identical to ``phase_type``.

        Returns
        -------
        str
            Either ``"stable"`` or ``"volatile"``.
        """
        return self.phase_type

    def __post_init__(self) -> None:
        valid_types = {"stable", "volatile"}
        if self.phase_type not in valid_types:
            raise ValueError(
                f"TransferConfig: phase_type must be one of "
                f"{valid_types}, got '{self.phase_type}'."
            )
        if self.n_trials < 1:
            raise ValueError(
                f"TransferConfig: n_trials must be >= 1, got {self.n_trials}."
            )
        for i, p in enumerate(self.cue_probs):
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"TransferConfig: cue_probs[{i}] must be in [0.0, 1.0], got {p}."
                )


@dataclass(frozen=True)
class TaskConfig:
    """Complete task structure for one PRL pick_best_cue session.

    Parameters
    ----------
    name : str
        Task name.
    description : str
        Human-readable description.
    n_cues : int
        Number of cues (must be >= 2).
    cue_labels : list[str]
        Label for each cue. Length must equal ``n_cues``.
    n_sets : int
        Number of times the phase sequence repeats per session (must be >= 1).
    phases : list[PhaseConfig]
        Ordered list of task phases (must be non-empty).
    transfer : TransferConfig
        Transfer phase appended after each set.
    partial_feedback : bool
        If True, only the chosen cue receives a reward signal.
    task_seed : int
        RNG seed for reproducible trial sequence generation.
    """

    name: str
    description: str
    n_cues: int
    cue_labels: list[str]
    n_sets: int
    phases: list[PhaseConfig]
    transfer: TransferConfig
    partial_feedback: bool
    task_seed: int

    def __post_init__(self) -> None:
        if self.n_cues < 2:
            raise ValueError(f"TaskConfig: n_cues must be >= 2, got {self.n_cues}.")
        if len(self.cue_labels) != self.n_cues:
            raise ValueError(
                f"TaskConfig: cue_labels length must equal n_cues "
                f"(expected {self.n_cues}, got {len(self.cue_labels)})."
            )
        if self.n_sets < 1:
            raise ValueError(f"TaskConfig: n_sets must be >= 1, got {self.n_sets}.")
        if not self.phases:
            raise ValueError("TaskConfig: phases must be non-empty.")
        for phase in self.phases:
            if len(phase.cue_probs) != self.n_cues:
                raise ValueError(
                    f"TaskConfig: phase '{phase.name}' cue_probs length "
                    f"must equal n_cues "
                    f"(expected {self.n_cues}, got {len(phase.cue_probs)})."
                )
        if len(self.transfer.cue_probs) != self.n_cues:
            raise ValueError(
                f"TaskConfig: transfer cue_probs length must equal n_cues "
                f"(expected {self.n_cues}, "
                f"got {len(self.transfer.cue_probs)})."
            )

    @property
    def n_trials_per_set(self) -> int:
        """Number of trials in one set (phases + transfer).

        Returns
        -------
        int
            Sum of ``n_trials`` across all phases plus transfer phase trials.
        """
        return sum(p.n_trials for p in self.phases) + self.transfer.n_trials

    @property
    def n_trials_total(self) -> int:
        """Total number of trials for a full session (all sets).

        Returns
        -------
        int
            ``n_sets * n_trials_per_set``.
        """
        return self.n_sets * self.n_trials_per_set


# ---------------------------------------------------------------------------
# Simulation dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupParamDist:
    """Parameter distribution for one group and one parameter.

    Parameters
    ----------
    mean : float
        Distribution mean.
    sd : float
        Distribution standard deviation (must be > 0).
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0.0:
            raise ValueError(f"GroupParamDist: sd must be > 0, got {self.sd}.")


@dataclass(frozen=True)
class GroupConfig:
    """Parameter distributions for one synthetic group.

    Parameters
    ----------
    omega_2 : GroupParamDist
        Tonic volatility at level 1.
    omega_3 : GroupParamDist
        Meta-volatility at level 2 (3-level model).
    kappa : GroupParamDist
        Volatility coupling (3-level model). Mean must be > 0.
    beta : GroupParamDist
        Inverse temperature. Mean must be > 0.
    zeta : GroupParamDist
        Stickiness / choice perseveration.
    """

    omega_2: GroupParamDist
    omega_3: GroupParamDist
    kappa: GroupParamDist
    beta: GroupParamDist
    zeta: GroupParamDist

    def __post_init__(self) -> None:
        if self.kappa.mean <= 0.0:
            raise ValueError(
                f"GroupConfig: kappa.mean must be > 0, got {self.kappa.mean}."
            )
        if self.beta.mean <= 0.0:
            raise ValueError(
                f"GroupConfig: beta.mean must be > 0, got {self.beta.mean}."
            )


@dataclass(frozen=True)
class SessionConfig:
    """Session-level parameter deltas for one group.

    All delta lists must have the same length as ``session_labels``.

    Parameters
    ----------
    session_labels : list[str]
        Labels of sessions to which deltas apply (excludes baseline).
    omega_2_deltas : list[float]
        Additive shift in omega_2 for each non-baseline session.
    kappa_deltas : list[float]
        Additive shift in kappa for each non-baseline session.
    beta_deltas : list[float]
        Additive shift in beta for each non-baseline session.
    zeta_deltas : list[float]
        Additive shift in zeta for each non-baseline session.
    """

    session_labels: list[str]
    omega_2_deltas: list[float]
    kappa_deltas: list[float]
    beta_deltas: list[float]
    zeta_deltas: list[float]

    def __post_init__(self) -> None:
        n = len(self.session_labels)
        for attr_name in (
            "omega_2_deltas",
            "kappa_deltas",
            "beta_deltas",
            "zeta_deltas",
        ):
            val = getattr(self, attr_name)
            if len(val) != n:
                raise ValueError(
                    f"SessionConfig: '{attr_name}' length must equal "
                    f"len(session_labels) (expected {n}, got {len(val)})."
                )


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for synthetic participant generation.

    Parameters
    ----------
    n_participants_per_group : int
        Number of synthetic participants per group (must be >= 1).
    master_seed : int
        Master RNG seed for reproducibility.
    groups : dict[str, GroupConfig]
        Group name -> parameter distribution mapping.
    session_deltas : dict[str, SessionConfig]
        Group name -> session delta mapping.
    """

    n_participants_per_group: int
    master_seed: int
    groups: dict[str, GroupConfig]
    session_deltas: dict[str, SessionConfig]

    def __post_init__(self) -> None:
        if self.n_participants_per_group < 1:
            raise ValueError(
                "SimulationConfig: n_participants_per_group must be >= 1, "
                f"got {self.n_participants_per_group}."
            )
        if set(self.groups) != set(self.session_deltas):
            raise ValueError(
                "SimulationConfig: groups and session_deltas must have the "
                f"same keys. Groups: {set(self.groups)}, "
                f"session_deltas: {set(self.session_deltas)}."
            )


# ---------------------------------------------------------------------------
# Fitting dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittingConfig:
    """MCMC fitting parameters and diagnostic thresholds.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains per participant (must be >= 1).
    n_draws : int
        Posterior draws per chain after tuning (must be >= 1).
    n_tune : int
        Tuning (warm-up) steps per chain (must be >= 1).
    target_accept : float
        NUTS step-size adaptation target acceptance rate (must be in (0, 1)).
    random_seed : int
        Base RNG seed for reproducibility.
    r_hat_threshold : float
        R-hat flag threshold (default 1.05).
    ess_threshold : float
        ESS bulk flag threshold (default 400).
    """

    n_chains: int
    n_draws: int
    n_tune: int
    target_accept: float
    random_seed: int
    r_hat_threshold: float = 1.05
    ess_threshold: float = 400.0

    def __post_init__(self) -> None:
        if self.n_chains < 1:
            raise ValueError(
                f"FittingConfig: n_chains must be >= 1, got {self.n_chains}."
            )
        if self.n_draws < 1:
            raise ValueError(
                f"FittingConfig: n_draws must be >= 1, got {self.n_draws}."
            )
        if self.n_tune < 1:
            raise ValueError(f"FittingConfig: n_tune must be >= 1, got {self.n_tune}.")
        if not (0.0 < self.target_accept < 1.0):
            raise ValueError(
                f"FittingConfig: target_accept must be in (0, 1), "
                f"got {self.target_accept}."
            )


# ---------------------------------------------------------------------------
# Top-level AnalysisConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level analysis configuration.

    Aggregates all sub-configs loaded from ``prl_analysis.yaml``.

    Parameters
    ----------
    task : TaskConfig
        PRL task structure (phases, cue probs, trial counts).
    simulation : SimulationConfig
        Synthetic participant generation parameters.
    fitting : FittingConfig
        MCMC fitting parameters and diagnostic thresholds.
    """

    task: TaskConfig
    simulation: SimulationConfig
    fitting: FittingConfig


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_group_param_dist(raw: dict[str, Any], ctx: str) -> GroupParamDist:
    """Parse a ``{mean: float, sd: float}`` mapping into :class:`GroupParamDist`."""
    try:
        return GroupParamDist(mean=float(raw["mean"]), sd=float(raw["sd"]))
    except KeyError as exc:
        raise ValueError(
            f"{ctx}: missing required key {exc} in parameter distribution."
        ) from exc


def _parse_group_config(raw: dict[str, Any], group_name: str) -> GroupConfig:
    """Parse one group's parameter distributions into :class:`GroupConfig`."""
    ctx = f"simulation.groups.{group_name}"
    required = ("omega_2", "omega_3", "kappa", "beta", "zeta")
    for key in required:
        if key not in raw:
            raise ValueError(f"{ctx}: missing required parameter key '{key}'.")
    return GroupConfig(
        omega_2=_parse_group_param_dist(raw["omega_2"], f"{ctx}.omega_2"),
        omega_3=_parse_group_param_dist(raw["omega_3"], f"{ctx}.omega_3"),
        kappa=_parse_group_param_dist(raw["kappa"], f"{ctx}.kappa"),
        beta=_parse_group_param_dist(raw["beta"], f"{ctx}.beta"),
        zeta=_parse_group_param_dist(raw["zeta"], f"{ctx}.zeta"),
    )


def _parse_session_config(raw: dict[str, Any], group_name: str) -> SessionConfig:
    """Parse one group's session delta mapping into :class:`SessionConfig`."""
    ctx = f"simulation.session_deltas.{group_name}"
    try:
        return SessionConfig(
            session_labels=list(raw["session_labels"]),
            omega_2_deltas=[float(v) for v in raw["omega_2_deltas"]],
            kappa_deltas=[float(v) for v in raw["kappa_deltas"]],
            beta_deltas=[float(v) for v in raw["beta_deltas"]],
            zeta_deltas=[float(v) for v in raw["zeta_deltas"]],
        )
    except KeyError as exc:
        raise ValueError(
            f"{ctx}: missing required key {exc} in session_deltas."
        ) from exc


def _parse_task_config(raw: dict[str, Any]) -> TaskConfig:
    """Parse the ``task`` section of the YAML into :class:`TaskConfig`."""
    ctx = "task"
    try:
        raw_phases = raw["phases"]
        phases = [
            PhaseConfig(
                name=str(p["name"]),
                phase_type=str(p["phase_type"]),
                n_trials=int(p["n_trials"]),
                cue_probs=[float(v) for v in p["cue_probs"]],
            )
            for p in raw_phases
        ]
        raw_transfer = raw["transfer"]
        transfer = TransferConfig(
            phase_type=str(raw_transfer["phase_type"]),
            n_trials=int(raw_transfer["n_trials"]),
            cue_probs=[float(v) for v in raw_transfer["cue_probs"]],
        )
        return TaskConfig(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            n_cues=int(raw["n_cues"]),
            cue_labels=[str(s) for s in raw["cue_labels"]],
            n_sets=int(raw["n_sets"]),
            phases=phases,
            transfer=transfer,
            partial_feedback=bool(raw["partial_feedback"]),
            task_seed=int(raw["task_seed"]),
        )
    except KeyError as exc:
        raise ValueError(f"{ctx}: missing required key {exc} in task config.") from exc


def _parse_simulation_config(raw: dict[str, Any]) -> SimulationConfig:
    """Parse the ``simulation`` section of the YAML into :class:`SimulationConfig`."""
    ctx = "simulation"
    try:
        raw_groups: dict[str, Any] = raw["groups"]
        raw_deltas: dict[str, Any] = raw["session_deltas"]
        groups = {
            name: _parse_group_config(data, name) for name, data in raw_groups.items()
        }
        session_deltas = {
            name: _parse_session_config(data, name) for name, data in raw_deltas.items()
        }
        return SimulationConfig(
            n_participants_per_group=int(raw["n_participants_per_group"]),
            master_seed=int(raw["master_seed"]),
            groups=groups,
            session_deltas=session_deltas,
        )
    except KeyError as exc:
        raise ValueError(
            f"{ctx}: missing required key {exc} in simulation config."
        ) from exc


def _parse_fitting_config(raw: dict[str, Any]) -> FittingConfig:
    """Parse the ``fitting`` section of the YAML into :class:`FittingConfig`."""
    ctx = "fitting"
    try:
        raw_diag = raw.get("diagnostics", {})
        return FittingConfig(
            n_chains=int(raw["n_chains"]),
            n_draws=int(raw["n_draws"]),
            n_tune=int(raw["n_tune"]),
            target_accept=float(raw["target_accept"]),
            random_seed=int(raw["random_seed"]),
            r_hat_threshold=float(raw_diag.get("r_hat_threshold", 1.05)),
            ess_threshold=float(raw_diag.get("ess_threshold", 400.0)),
        )
    except KeyError as exc:
        raise ValueError(
            f"{ctx}: missing required key {exc} in fitting config."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AnalysisConfig:
    """Load and validate the PRL analysis configuration from a YAML file.

    Parameters
    ----------
    path : Path or None, optional
        Path to the YAML config file. Defaults to
        ``CONFIGS_DIR / "prl_analysis.yaml"``.

    Returns
    -------
    AnalysisConfig
        Fully validated, immutable configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If any field fails validation (wrong type, out-of-range, missing key).

    Examples
    --------
    >>> from prl_hgf.env.task_config import load_config
    >>> config = load_config()
    >>> config.task.n_cues
    3
    >>> len(config.task.phases)
    4
    """
    resolved = path if path is not None else _DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. Expected at {_DEFAULT_CONFIG_PATH}."
        )
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    task = _parse_task_config(raw["task"])
    simulation = _parse_simulation_config(raw["simulation"])
    fitting = _parse_fitting_config(raw.get("fitting", {}))

    return AnalysisConfig(task=task, simulation=simulation, fitting=fitting)
