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
class CriterionConfig:
    """Reversal criterion for criterion-based (closed-loop) task schedules.

    Two criterion types are supported:

    * ``"consecutive_correct"`` — a phase ends once the participant has
      chosen the objectively best cue on ``threshold`` consecutive trials,
      where ``threshold`` is drawn uniformly from
      ``[n_correct_min, n_correct_max]`` (inclusive) at each phase start
      (jittered criterion).
    * ``"window"`` — a phase ends once at least ``window_n_correct`` of the
      last ``window_size`` choices within the phase were of the best cue.

    Parameters
    ----------
    criterion_type : str
        Either ``"consecutive_correct"`` or ``"window"``. Maps to the YAML
        key ``criterion.type``.
    n_correct_min : int or None
        Lower bound (inclusive) of the jittered consecutive-correct
        threshold. Required for ``"consecutive_correct"``.
    n_correct_max : int or None
        Upper bound (inclusive) of the jittered consecutive-correct
        threshold. Required for ``"consecutive_correct"``.
    window_size : int or None
        Sliding window length in trials. Required for ``"window"``.
    window_n_correct : int or None
        Number of best-cue choices within the window required to end the
        phase. Required for ``"window"``.
    """

    criterion_type: str
    n_correct_min: int | None = None
    n_correct_max: int | None = None
    window_size: int | None = None
    window_n_correct: int | None = None

    def __post_init__(self) -> None:
        valid_types = {"consecutive_correct", "window"}
        if self.criterion_type not in valid_types:
            raise ValueError(
                f"CriterionConfig: criterion type must be one of {valid_types}, "
                f"got '{self.criterion_type}'."
            )
        if self.criterion_type == "consecutive_correct":
            if self.n_correct_min is None or self.n_correct_max is None:
                raise ValueError(
                    "CriterionConfig: type 'consecutive_correct' requires "
                    "n_correct_min and n_correct_max, got "
                    f"n_correct_min={self.n_correct_min}, "
                    f"n_correct_max={self.n_correct_max}."
                )
            if self.n_correct_min < 1:
                raise ValueError(
                    "CriterionConfig: n_correct_min must be >= 1, "
                    f"got {self.n_correct_min}."
                )
            if self.n_correct_max < self.n_correct_min:
                raise ValueError(
                    "CriterionConfig: n_correct_max must be >= n_correct_min "
                    f"(expected >= {self.n_correct_min}, "
                    f"got {self.n_correct_max})."
                )
        else:  # window
            if self.window_size is None or self.window_n_correct is None:
                raise ValueError(
                    "CriterionConfig: type 'window' requires window.size and "
                    f"window.n_correct, got size={self.window_size}, "
                    f"n_correct={self.window_n_correct}."
                )
            if self.window_size < 1:
                raise ValueError(
                    f"CriterionConfig: window.size must be >= 1, "
                    f"got {self.window_size}."
                )
            if not (1 <= self.window_n_correct <= self.window_size):
                raise ValueError(
                    "CriterionConfig: window.n_correct must be in "
                    f"[1, window.size={self.window_size}], "
                    f"got {self.window_n_correct}."
                )


@dataclass(frozen=True)
class ReversalConfig:
    """Criterion-based reversal schedule for a closed-loop task.

    Parameters
    ----------
    criterion : CriterionConfig
        The performance criterion that ends each learning phase.
    max_trials_per_phase : int
        Hard cap on trials per learning phase; a reversal fires when the
        criterion is met OR this cap is reached (must be >= 1).
    n_reversals_per_set : int
        Number of reversal phases after the acquisition phase in each set
        (must be >= 0).
    target_rule : str
        Rule for drawing the new best cue at each reversal. Only
        ``"random_nonbest"`` (uniform over the non-best cues) is supported.
    """

    criterion: CriterionConfig
    max_trials_per_phase: int
    n_reversals_per_set: int
    target_rule: str

    def __post_init__(self) -> None:
        valid_rules = {"random_nonbest"}
        if self.max_trials_per_phase < 1:
            raise ValueError(
                "ReversalConfig: max_trials_per_phase must be >= 1, "
                f"got {self.max_trials_per_phase}."
            )
        if self.n_reversals_per_set < 0:
            raise ValueError(
                "ReversalConfig: n_reversals_per_set must be >= 0, "
                f"got {self.n_reversals_per_set}."
            )
        if self.target_rule not in valid_rules:
            raise ValueError(
                f"ReversalConfig: target_rule must be one of {valid_rules}, "
                f"got '{self.target_rule}'."
            )


@dataclass(frozen=True)
class CueProbPair:
    """Reward probabilities for the best and non-best cues.

    Used by criterion-based tasks, where every learning phase assigns
    ``best`` to the current best cue and ``other`` to all remaining cues.

    Parameters
    ----------
    best : float
        Reward probability of the current best cue, in [0.0, 1.0].
    other : float
        Reward probability of each non-best cue, in [0.0, 1.0].
    """

    best: float
    other: float

    def __post_init__(self) -> None:
        for label in ("best", "other"):
            p = getattr(self, label)
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"CueProbPair: '{label}' must be in [0.0, 1.0], got {p}."
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
    Two schedule modes are supported, selected by which fields are present:

    * **Fixed-phase mode** (``reversal is None``): ``phases`` holds a
      non-empty list of fixed-length :class:`PhaseConfig` entries. This is
      the original open-loop schedule.
    * **Criterion-based mode** (``reversal is not None``): ``phases`` must
      be empty; phase lengths are determined at simulation time by the
      performance criterion in :class:`ReversalConfig`, with per-phase cue
      probabilities derived from ``cue_prob_pair`` and the current best cue.

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
        Ordered list of task phases. Must be non-empty in fixed-phase mode
        and empty in criterion-based mode.
    transfer : TransferConfig
        Transfer phase appended after each set.
    partial_feedback : bool
        If True, only the chosen cue receives a reward signal.
    task_seed : int
        RNG seed for reproducible trial sequence generation.
    reversal : ReversalConfig or None, optional
        Criterion-based reversal schedule. ``None`` selects fixed-phase mode.
    initial_best_per_set : list[int] or None, optional
        Best cue index for each set's acquisition phase. Required in
        criterion-based mode; length must equal ``n_sets`` and each entry
        must be in ``[0, n_cues)``.
    cue_prob_pair : CueProbPair or None, optional
        Best/other reward probabilities for criterion-based learning phases.
        Required in criterion-based mode.
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
    reversal: ReversalConfig | None = None
    initial_best_per_set: list[int] | None = None
    cue_prob_pair: CueProbPair | None = None

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
        if self.reversal is not None:
            self._validate_criterion_mode()
        elif not self.phases:
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

    def _validate_criterion_mode(self) -> None:
        """Validate criterion-based-mode field constraints.

        Raises
        ------
        ValueError
            If ``phases`` is non-empty, or ``cue_prob_pair`` /
            ``initial_best_per_set`` are missing or malformed.
        """
        if self.phases:
            raise ValueError(
                "TaskConfig: 'phases' and 'reversal' are mutually exclusive; "
                f"expected exactly one, got both ('phases' has "
                f"{len(self.phases)} entries and 'reversal' is set)."
            )
        if self.cue_prob_pair is None:
            raise ValueError(
                "TaskConfig: criterion-based mode requires 'cue_probs' as a "
                "{best, other} mapping (CueProbPair), got None."
            )
        if self.initial_best_per_set is None:
            raise ValueError(
                "TaskConfig: criterion-based mode requires "
                "'initial_best_per_set', got None."
            )
        if len(self.initial_best_per_set) != self.n_sets:
            raise ValueError(
                f"TaskConfig: initial_best_per_set length must equal n_sets "
                f"(expected {self.n_sets}, "
                f"got {len(self.initial_best_per_set)})."
            )
        for i, best in enumerate(self.initial_best_per_set):
            if not (0 <= best < self.n_cues):
                raise ValueError(
                    f"TaskConfig: initial_best_per_set[{i}] must be in "
                    f"[0, {self.n_cues - 1}], got {best}."
                )

    @property
    def is_criterion_based(self) -> bool:
        """Whether this task uses a criterion-based (closed-loop) schedule.

        Returns
        -------
        bool
            ``True`` if ``reversal`` is set (criterion-based mode), ``False``
            for fixed-phase mode.
        """
        return self.reversal is not None

    @property
    def n_trials_per_set(self) -> int:
        """Number of trials in one set (phases + transfer).

        Only defined for fixed-phase mode — criterion-based session lengths
        are determined at simulation time. Use :attr:`max_trials_per_set`
        for the criterion-based upper bound.

        Returns
        -------
        int
            Sum of ``n_trials`` across all phases plus transfer phase trials.

        Raises
        ------
        ValueError
            If the task is criterion-based (variable phase lengths).
        """
        if self.is_criterion_based:
            raise ValueError(
                "TaskConfig: n_trials_per_set is undefined for "
                "criterion-based tasks (expected fixed-phase mode, got "
                "reversal mode); use max_trials_per_set for the upper bound."
            )
        return sum(p.n_trials for p in self.phases) + self.transfer.n_trials

    @property
    def n_trials_total(self) -> int:
        """Total number of trials for a full session (all sets).

        Only defined for fixed-phase mode; see :attr:`n_trials_per_set`.

        Returns
        -------
        int
            ``n_sets * n_trials_per_set``.

        Raises
        ------
        ValueError
            If the task is criterion-based (variable phase lengths).
        """
        return self.n_sets * self.n_trials_per_set

    @property
    def max_trials_per_set(self) -> int:
        """Upper bound on the number of trials in one set.

        Returns
        -------
        int
            For criterion-based mode:
            ``(n_reversals_per_set + 1) * max_trials_per_phase`` plus
            transfer trials. For fixed-phase mode: :attr:`n_trials_per_set`.
        """
        if self.reversal is not None:
            n_learning_phases = self.reversal.n_reversals_per_set + 1
            return (
                n_learning_phases * self.reversal.max_trials_per_phase
                + self.transfer.n_trials
            )
        return self.n_trials_per_set


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

    @property
    def is_criterion_based(self) -> bool:
        """Whether the task uses a criterion-based (closed-loop) schedule.

        Returns
        -------
        bool
            ``True`` if ``task.reversal`` is set, ``False`` for the
            fixed-phase schedule.
        """
        return self.task.is_criterion_based


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


def _parse_criterion_config(raw: dict[str, Any]) -> CriterionConfig:
    """Parse the ``task.reversal.criterion`` mapping into :class:`CriterionConfig`."""
    ctx = "task.reversal.criterion"
    try:
        criterion_type = str(raw["type"])
    except KeyError as exc:
        raise ValueError(f"{ctx}: missing required key {exc}.") from exc

    raw_window = raw.get("window")
    window_size: int | None = None
    window_n_correct: int | None = None
    if raw_window is not None:
        if not isinstance(raw_window, dict):
            raise ValueError(
                f"{ctx}: 'window' must be a mapping with keys 'size' and "
                f"'n_correct' (or null), got {type(raw_window).__name__}."
            )
        try:
            window_size = int(raw_window["size"])
            window_n_correct = int(raw_window["n_correct"])
        except KeyError as exc:
            raise ValueError(f"{ctx}.window: missing required key {exc}.") from exc

    n_correct_min = raw.get("n_correct_min")
    n_correct_max = raw.get("n_correct_max")
    return CriterionConfig(
        criterion_type=criterion_type,
        n_correct_min=None if n_correct_min is None else int(n_correct_min),
        n_correct_max=None if n_correct_max is None else int(n_correct_max),
        window_size=window_size,
        window_n_correct=window_n_correct,
    )


def _parse_criterion_task_config(raw: dict[str, Any]) -> TaskConfig:
    """Parse a criterion-based (closed-loop) ``task`` section.

    Selected when ``task.reversal`` is present. ``phases`` must be absent
    (or empty); phase lengths are determined at simulation time.
    """
    ctx = "task"
    if raw.get("phases"):
        raise ValueError(
            f"{ctx}: 'phases' and 'reversal' are mutually exclusive; "
            f"expected exactly one, got both."
        )
    try:
        raw_reversal = raw["reversal"]
        reversal = ReversalConfig(
            criterion=_parse_criterion_config(raw_reversal["criterion"]),
            max_trials_per_phase=int(raw_reversal["max_trials_per_phase"]),
            n_reversals_per_set=int(raw_reversal["n_reversals_per_set"]),
            target_rule=str(raw_reversal["target_rule"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"{ctx}.reversal: missing required key {exc} in reversal config."
        ) from exc

    try:
        raw_cue_probs = raw["cue_probs"]
        if not isinstance(raw_cue_probs, dict):
            raise ValueError(
                f"{ctx}: criterion-based mode requires cue_probs to be a "
                f"{{best, other}} mapping, got {type(raw_cue_probs).__name__}."
            )
        cue_prob_pair = CueProbPair(
            best=float(raw_cue_probs["best"]),
            other=float(raw_cue_probs["other"]),
        )
        raw_transfer = raw["transfer"]
        transfer = TransferConfig(
            phase_type=str(raw_transfer["phase_type"]),
            n_trials=int(raw_transfer["n_trials"]),
            cue_probs=[float(v) for v in raw_transfer["cue_probs"]],
        )
        n_cues = int(raw["n_cues"])
        cue_labels = [str(s) for s in raw.get("cue_labels", [])] or [
            f"cue_{i}" for i in range(n_cues)
        ]
        return TaskConfig(
            name=str(raw.get("name", "pick_best_cue")),
            description=str(raw.get("description", "")),
            n_cues=n_cues,
            cue_labels=cue_labels,
            n_sets=int(raw["n_sets"]),
            phases=[],
            transfer=transfer,
            partial_feedback=bool(raw.get("partial_feedback", True)),
            task_seed=int(raw.get("task_seed", 0)),
            reversal=reversal,
            initial_best_per_set=[int(v) for v in raw["initial_best_per_set"]],
            cue_prob_pair=cue_prob_pair,
        )
    except KeyError as exc:
        raise ValueError(f"{ctx}: missing required key {exc} in task config.") from exc


def _parse_task_config(raw: dict[str, Any]) -> TaskConfig:
    """Parse the ``task`` section of the YAML into :class:`TaskConfig`.

    Dispatches on the presence of ``task.reversal``: if present, the task is
    criterion-based (closed-loop) and ``phases`` may be absent; otherwise the
    original fixed-phase schema is required.
    """
    if raw.get("reversal") is not None:
        return _parse_criterion_task_config(raw)
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
