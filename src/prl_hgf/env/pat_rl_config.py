"""PAT-RL task configuration loader (parallel stack from pick_best_cue).

Loads ``configs/pat_rl.yaml`` and returns a validated, typed
:class:`PATRLConfig` hierarchy.  This loader is completely independent of
:mod:`prl_hgf.env.task_config` — it does not import, subclass, or inspect
``AnalysisConfig``/``TaskConfig``.  That keeps pick_best_cue tests isolated
from PAT-RL changes.

PAT-RL callers import directly from this module::

    from prl_hgf.env.pat_rl_config import load_pat_rl_config

    cfg = load_pat_rl_config()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from prl_hgf._paths import CONFIGS_DIR

_DEFAULT_PATRL_CONFIG_PATH = CONFIGS_DIR / "pat_rl.yaml"

# ---------------------------------------------------------------------------
# Required phenotype keys (exactly the 4-cell 2x2 grid)
# ---------------------------------------------------------------------------
_REQUIRED_PHENOTYPE_KEYS: frozenset[str] = frozenset(
    {"healthy", "high_anxiety", "reward_susceptible", "anxious_reward"}
)

__all__ = [
    "DeltaHRDistribution",
    "DeltaHRStubConfig",
    "ContingencyConfig",
    "FittingPriorConfig",
    "HazardConfig",
    "MagnitudeConfig",
    "OutcomeProbs",
    "PATRLConfig",
    "PATRLFittingConfig",
    "PATRLSimulationConfig",
    "PATRLTaskConfig",
    "PhenotypeParams",
    "PriorGaussian",
    "PriorTruncated",
    "TimingConfig",
    "load_pat_rl_config",
    "PHENOTYPE_COLUMN_NAME",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Canonical column name for the phenotype label in simulation and fit
#: DataFrames. Use this constant everywhere to avoid silent name mismatches
#: when propagating the phenotype dimension from sim_df → fit_df → BMS.
#: (RESEARCH.md §12 Risk 3)
PHENOTYPE_COLUMN_NAME: str = "phenotype"


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HazardConfig:
    """Hazard rates for stable and volatile reversal regimes.

    Parameters
    ----------
    stable : float
        Hazard rate for stable runs (probability of state reversal per trial).
        Must be in the open interval (0, 1).
    volatile : float
        Hazard rate for volatile runs.  Must be in (0, 1) and strictly
        greater than *stable*.
    """

    stable: float
    volatile: float

    def __post_init__(self) -> None:
        if not (0.0 < self.stable < 1.0):
            raise ValueError(
                f"HazardConfig: stable must be in (0, 1), got {self.stable}"
            )
        if not (0.0 < self.volatile < 1.0):
            raise ValueError(
                f"HazardConfig: volatile must be in (0, 1), got {self.volatile}"
            )
        if self.stable >= self.volatile:
            raise ValueError(
                "HazardConfig: stable must be < volatile, "
                f"got stable={self.stable} >= volatile={self.volatile}"
            )


@dataclass(frozen=True)
class OutcomeProbs:
    """Outcome probabilities for one context state under approach action.

    Parameters
    ----------
    reward : float
        Probability of receiving a reward.  Must be in [0, 1].
    shock : float
        Probability of receiving a shock.  Must be in [0, 1].
    nothing : float
        Probability of no outcome.  Must be in [0, 1].

    Notes
    -----
    The three probabilities must sum to 1.0 within absolute tolerance 1e-6.
    """

    reward: float
    shock: float
    nothing: float

    def __post_init__(self) -> None:
        for name, val in (
            ("reward", self.reward),
            ("shock", self.shock),
            ("nothing", self.nothing),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"OutcomeProbs: {name} must be in [0, 1], got {val}")
        total = self.reward + self.shock + self.nothing
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"OutcomeProbs: probabilities must sum to 1.0, got {total:.7f}"
            )


@dataclass(frozen=True)
class ContingencyConfig:
    """Outcome contingencies for safe and dangerous context states.

    Parameters
    ----------
    safe : OutcomeProbs
        Probabilities for the safe state (state=0) under approach action.
    dangerous : OutcomeProbs
        Probabilities for the dangerous state (state=1) under approach action.
    avoid : OutcomeProbs
        Probabilities for the avoid action regardless of context state.
        Consumer spec: P(reward|avoid)=0.10, P(shock|avoid)=0.10,
        P(nothing|avoid)=0.80 (stochastic avoid; Klaassen 2024 Comms Bio).
        NOTE (Plan 20-01): This field exposes the avoid contingency at the
        config surface. Wiring stochastic avoid outcomes into the logp and
        scan body is deferred to Plans 20-02/20-03. Legacy configs without
        an ``avoid`` key default to P(nothing|avoid)=1.0 (deterministic
        no-outcome), preserving Phase 18/19 behavior.
    """

    safe: OutcomeProbs
    dangerous: OutcomeProbs
    avoid: OutcomeProbs = OutcomeProbs(reward=0.0, shock=0.0, nothing=1.0)


@dataclass(frozen=True)
class MagnitudeConfig:
    """2x2 reward/shock magnitude design.

    Parameters
    ----------
    reward_levels : tuple[float, ...]
        Two reward magnitude levels.  Both must be strictly positive.
        Length must be exactly 2.
    shock_levels : tuple[float, ...]
        Two shock magnitude levels.  Both must be strictly positive.
        Length must be exactly 2.
    """

    reward_levels: tuple[float, ...]
    shock_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, levels in (
            ("reward_levels", self.reward_levels),
            ("shock_levels", self.shock_levels),
        ):
            if len(levels) != 2:
                raise ValueError(
                    f"MagnitudeConfig: {name} must have exactly 2 elements, "
                    f"got {len(levels)}"
                )
            for i, v in enumerate(levels):
                if v <= 0.0:
                    raise ValueError(
                        f"MagnitudeConfig: {name}[{i}] must be > 0, got {v}"
                    )


@dataclass(frozen=True)
class TimingConfig:
    """Per-trial timing parameters in seconds.

    Parameters
    ----------
    cue_duration_s : float
        Duration of the cue presentation window.  Must be > 0.
    anticipation_s : float
        Duration of the anticipation window.  Must be > 0.
    outcome_duration_s : float
        Duration of the outcome window.  Must be > 0.
    iti_s : float
        Inter-trial interval.  Must be > 0.
    """

    cue_duration_s: float
    anticipation_s: float
    outcome_duration_s: float
    iti_s: float

    def __post_init__(self) -> None:
        for name, val in (
            ("cue_duration_s", self.cue_duration_s),
            ("anticipation_s", self.anticipation_s),
            ("outcome_duration_s", self.outcome_duration_s),
            ("iti_s", self.iti_s),
        ):
            if val <= 0.0:
                raise ValueError(f"TimingConfig: {name} must be > 0, got {val}")

    @property
    def trial_duration_s(self) -> float:
        """Total nominal trial duration in seconds."""
        return (
            self.cue_duration_s
            + self.anticipation_s
            + self.outcome_duration_s
            + self.iti_s
        )


@dataclass(frozen=True)
class DeltaHRDistribution:
    """Gaussian stub for Delta-HR per context state.

    Parameters
    ----------
    mean : float
        Mean anticipatory heart-rate change (bpm).
    sd : float
        Standard deviation.  Must be > 0.
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0.0:
            raise ValueError(f"DeltaHRDistribution: sd must be > 0, got {self.sd}")


@dataclass(frozen=True)
class DeltaHRStubConfig:
    """Literature-grounded stub distribution for anticipatory Delta-HR.

    Parameters
    ----------
    dangerous : DeltaHRDistribution
        Distribution for dangerous-state trials.
    safe : DeltaHRDistribution
        Distribution for safe-state trials.
    bounds : tuple[float, float]
        Hard lower and upper clamp bounds in bpm.
        bounds[0] must be strictly less than bounds[1].
    """

    dangerous: DeltaHRDistribution
    safe: DeltaHRDistribution
    bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                "DeltaHRStubConfig: bounds[0] must be < bounds[1], "
                f"got [{self.bounds[0]}, {self.bounds[1]}]"
            )


# ---------------------------------------------------------------------------
# Task-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PATRLTaskConfig:
    """Full task-structure specification for PAT-RL.

    Parameters
    ----------
    name : str
        Task identifier (``"pat_rl"``).
    description : str
        Human-readable description.
    n_trials : int
        Total number of trials across all runs.
    n_runs : int
        Number of runs.
    trials_per_run : int
        Trials per run.  ``n_runs * trials_per_run`` must equal ``n_trials``.
    hazards : HazardConfig
        Stable and volatile hazard rates.
    run_order : tuple[str, ...]
        Ordered list of run regime labels.  Length must equal ``n_runs``;
        each entry must be ``"stable"`` or ``"volatile"``.
    contingencies : ContingencyConfig
        Outcome probabilities per context state.
    magnitudes : MagnitudeConfig
        2x2 reward/shock magnitude levels.
    timing : TimingConfig
        Per-trial timing windows.
    delta_hr_stub : DeltaHRStubConfig
        Literature stub for Delta-HR covariate.
    """

    name: str
    description: str
    n_trials: int
    n_runs: int
    trials_per_run: int
    hazards: HazardConfig
    run_order: tuple[str, ...]
    contingencies: ContingencyConfig
    magnitudes: MagnitudeConfig
    timing: TimingConfig
    delta_hr_stub: DeltaHRStubConfig

    def __post_init__(self) -> None:
        expected_trials = self.n_runs * self.trials_per_run
        if expected_trials != self.n_trials:
            raise ValueError(
                "PATRLTaskConfig: n_runs * trials_per_run must equal n_trials; "
                f"expected {self.n_trials}, got {expected_trials}"
            )
        if len(self.run_order) != self.n_runs:
            raise ValueError(
                "PATRLTaskConfig: run_order length must equal n_runs; "
                f"expected {self.n_runs}, got {len(self.run_order)}"
            )
        valid_regimes = {"stable", "volatile"}
        for i, regime in enumerate(self.run_order):
            if regime not in valid_regimes:
                raise ValueError(
                    f"PATRLTaskConfig: run_order[{i}] must be one of "
                    f"{sorted(valid_regimes)}, got '{regime}'"
                )


# ---------------------------------------------------------------------------
# Simulation-level configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorGaussian:
    """Gaussian prior or generative distribution.

    Parameters
    ----------
    mean : float
        Mean value.
    sd : float
        Standard deviation.  Must be >= 0 (sd=0 means fixed/degenerate).
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd < 0.0:
            raise ValueError(f"PriorGaussian: sd must be >= 0, got {self.sd}")


@dataclass(frozen=True)
class PriorTruncated:
    """Truncated-normal fitting prior.

    Parameters
    ----------
    lower : float
        Lower truncation bound.  Must be strictly less than ``upper``.
    upper : float
        Upper truncation bound.
    mean : float
        Prior mean.  Must satisfy ``lower <= mean <= upper``.
    sd : float
        Prior standard deviation.  Must be > 0.
    """

    lower: float
    upper: float
    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                "PriorTruncated: lower must be < upper, "
                f"got lower={self.lower}, upper={self.upper}"
            )
        if not (self.lower <= self.mean <= self.upper):
            raise ValueError(
                "PriorTruncated: mean must be in [lower, upper], "
                f"got mean={self.mean}, lower={self.lower}, upper={self.upper}"
            )
        if self.sd <= 0.0:
            raise ValueError(f"PriorTruncated: sd must be > 0, got {self.sd}")


@dataclass(frozen=True)
class PhenotypeParams:
    """Parameter distribution for one phenotype in the 2x2 grid.

    Parameters
    ----------
    omega_2 : PriorGaussian
        Tonic volatility distribution (level-2 log-space learning rate).
    beta : PriorGaussian
        Inverse temperature distribution (decision noise).
    kappa : PriorGaussian
        Volatility-coupling distribution (sd=0 means fixed).
    mu3_0 : PriorGaussian
        Initial volatility prior distribution (sd=0 means fixed).
    omega_3 : PriorGaussian, optional
        Meta-volatility distribution (3-level HGF only; sd=0 means fixed).
        Defaults to ``PriorGaussian(mean=-4.0, sd=0.5)`` per consumer spec.
    theta : PriorGaussian, optional
        Volatility hyperparameter ϑ (scale factor for volatility updates).
        Defaults to ``PriorGaussian(mean=0.005, sd=0.002)`` per consumer spec.
    b : PriorGaussian, optional
        Response bias distribution (additive offset in decision logit).
        Defaults to ``PriorGaussian(mean=0.0, sd=0.5)`` (neutral bias).
        Prior for Models A+b, B, C. Validated: ``b.sd >= 0``.
    dhr_mean : float, optional
        Mean anticipatory heart-rate change (bpm) for ε₂-coupled ΔHR
        generative model. Negative values = bradycardia. Default -2.0 (healthy
        phenotype SC1 value from Klaassen et al. 2024 Comms Bio).
    dhr_sd : float, optional
        Standard deviation of ΔHR generative distribution (bpm). Must be
        > 0. Default 0.5.
    epsilon2_coupling_coef : float, optional
        Coefficient λ_dhr scaling level-2 prediction error ε₂(t) in the
        ΔHR generative model: ΔHR ~ N(dhr_mean, dhr_sd) + λ_dhr·ε₂(t).
        Positive = larger PE → more bradycardia (Klaassen 2024 direction).
        Must be >= 0. Default 0.3 (placeholder; TODO:consumer-spec).

    Notes
    -----
    The ``omega_3``, ``theta``, ``b``, ``dhr_mean``, ``dhr_sd``, and
    ``epsilon2_coupling_coef`` fields are new in Plan 20-01 and default
    to values that preserve Phase 18/19 simulation behavior on configs that
    omit them.
    """

    omega_2: PriorGaussian
    beta: PriorGaussian
    kappa: PriorGaussian
    mu3_0: PriorGaussian
    # --- Phase 20-01 additions (all have defaults for backward compat) ---
    omega_3: PriorGaussian = PriorGaussian(mean=-4.0, sd=0.5)
    theta: PriorGaussian = PriorGaussian(mean=0.005, sd=0.002)
    b: PriorGaussian = PriorGaussian(mean=0.0, sd=0.5)
    dhr_mean: float = -2.0
    dhr_sd: float = 0.5
    epsilon2_coupling_coef: float = 0.3

    def __post_init__(self) -> None:
        if self.dhr_sd <= 0.0:
            raise ValueError(f"PhenotypeParams: dhr_sd must be > 0, got {self.dhr_sd}")
        if self.epsilon2_coupling_coef < 0.0:
            raise ValueError(
                "PhenotypeParams: epsilon2_coupling_coef must be >= 0, "
                f"got {self.epsilon2_coupling_coef}"
            )


@dataclass(frozen=True)
class PATRLSimulationConfig:
    """Simulation configuration for PAT-RL synthetic participants.

    Parameters
    ----------
    n_participants_per_phenotype : int
        Number of synthetic participants per phenotype cell.  Must be >= 1.
    master_seed : int
        Master RNG seed for participant generation.
    phenotypes : dict[str, PhenotypeParams]
        Must contain exactly the four keys:
        ``healthy``, ``high_anxiety``, ``reward_susceptible``,
        ``anxious_reward``.
    """

    n_participants_per_phenotype: int
    master_seed: int
    phenotypes: dict[str, PhenotypeParams]

    def __post_init__(self) -> None:
        if self.n_participants_per_phenotype < 1:
            raise ValueError(
                "PATRLSimulationConfig: n_participants_per_phenotype must be "
                f">= 1, got {self.n_participants_per_phenotype}"
            )
        actual_keys = frozenset(self.phenotypes.keys())
        if actual_keys != _REQUIRED_PHENOTYPE_KEYS:
            missing = sorted(_REQUIRED_PHENOTYPE_KEYS - actual_keys)
            extra = sorted(actual_keys - _REQUIRED_PHENOTYPE_KEYS)
            raise ValueError(
                "PATRLSimulationConfig: phenotypes must have exactly the keys "
                f"{sorted(_REQUIRED_PHENOTYPE_KEYS)}; "
                f"missing={missing}, extra={extra}"
            )


# ---------------------------------------------------------------------------
# Fitting-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittingPriorConfig:
    """Prior distributions for all HGF and response-model parameters.

    Parameters
    ----------
    omega_2 : PriorGaussian
        Tonic volatility prior (Normal).
    omega_3 : PriorGaussian
        Meta-volatility prior (Normal).
    kappa : PriorTruncated
        Volatility-coupling prior (truncated Normal).
    beta : PriorTruncated
        Inverse temperature prior (truncated Normal).
    mu3_0 : PriorGaussian
        Initial volatility-state prior (Normal).
    b : PriorGaussian, optional
        Response bias prior (Normal). Used by Models A+b, B, C.
        Defaults to ``PriorGaussian(mean=0.0, sd=1.0)``.
    gamma : PriorGaussian, optional
        ΔHR additive weight prior (Normal). Used by Models B, C.
        Defaults to ``PriorGaussian(mean=0.0, sd=0.5)``.
    alpha : PriorGaussian, optional
        ΔHR × EV interaction weight prior (Normal). Used by Model C.
        Defaults to ``PriorGaussian(mean=0.0, sd=0.5)``.
    lam : PriorGaussian, optional
        Trial-varying omega coupling coefficient prior (Normal). Used by
        Model D (ω_eff(t) = ω + λ·ΔHR(t)). Defaults to
        ``PriorGaussian(mean=0.0, sd=0.1)``.

    Notes
    -----
    The ``b``, ``gamma``, ``alpha``, and ``lam`` fields are new in Plan
    20-01 and default to values consistent with no-effect (mean=0) that
    preserve Model A behavior on configs that omit them.
    """

    omega_2: PriorGaussian
    omega_3: PriorGaussian
    kappa: PriorTruncated
    beta: PriorTruncated
    mu3_0: PriorGaussian
    # --- Phase 20-01 additions (all have defaults for backward compat) ---
    b: PriorGaussian = PriorGaussian(mean=0.0, sd=1.0)
    gamma: PriorGaussian = PriorGaussian(mean=0.0, sd=0.5)
    alpha: PriorGaussian = PriorGaussian(mean=0.0, sd=0.5)
    lam: PriorGaussian = PriorGaussian(mean=0.0, sd=0.1)


@dataclass(frozen=True)
class PATRLFittingConfig:
    """MCMC fitting configuration for PAT-RL.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains.  Must be >= 1.
    n_tune : int
        Number of warmup / tuning steps.  Must be >= 1.
    n_draws : int
        Number of posterior draw steps.  Must be >= 1.
    target_accept : float
        Target acceptance rate for NUTS.  Must be in (0, 1).
    random_seed : int
        Base random seed for the sampler.
    priors : FittingPriorConfig
        Prior distributions for all parameters.
    """

    n_chains: int
    n_tune: int
    n_draws: int
    target_accept: float
    random_seed: int
    priors: FittingPriorConfig

    def __post_init__(self) -> None:
        for name, val in (
            ("n_chains", self.n_chains),
            ("n_tune", self.n_tune),
            ("n_draws", self.n_draws),
        ):
            if val < 1:
                raise ValueError(f"PATRLFittingConfig: {name} must be >= 1, got {val}")
        if not (0.0 < self.target_accept < 1.0):
            raise ValueError(
                "PATRLFittingConfig: target_accept must be in (0, 1), "
                f"got {self.target_accept}"
            )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PATRLConfig:
    """Root configuration object for PAT-RL.

    Parameters
    ----------
    task : PATRLTaskConfig
        Task structure (trials, runs, hazards, contingencies, magnitudes).
    simulation : PATRLSimulationConfig
        Synthetic participant generation parameters.
    fitting : PATRLFittingConfig
        MCMC settings and prior distributions.
    """

    task: PATRLTaskConfig
    simulation: PATRLSimulationConfig
    fitting: PATRLFittingConfig


# ---------------------------------------------------------------------------
# Private parse helpers
# ---------------------------------------------------------------------------


def _parse_hazards(d: dict[str, Any]) -> HazardConfig:
    return HazardConfig(
        stable=float(d["stable"]),
        volatile=float(d["volatile"]),
    )


def _parse_outcome_probs(d: dict[str, Any], state_name: str) -> OutcomeProbs:
    return OutcomeProbs(
        reward=float(d["reward"]),
        shock=float(d["shock"]),
        nothing=float(d["nothing"]),
    )


def _parse_contingencies(d: dict[str, Any]) -> ContingencyConfig:
    # NOTE (Plan 20-01): The ``avoid`` key is new in the consumer-spec YAML.
    # Legacy configs (Phase 18/19) that omit it receive the default
    # OutcomeProbs(reward=0.0, shock=0.0, nothing=1.0), preserving the
    # original deterministic no-outcome avoid behavior. Wiring stochastic
    # avoid outcomes into the logp/scan body is deferred to Plans 20-02/20-03.
    if "avoid" in d:
        avoid = _parse_outcome_probs(d["avoid"], "avoid")
    else:
        avoid = OutcomeProbs(reward=0.0, shock=0.0, nothing=1.0)
    return ContingencyConfig(
        safe=_parse_outcome_probs(d["safe"], "safe"),
        dangerous=_parse_outcome_probs(d["dangerous"], "dangerous"),
        avoid=avoid,
    )


def _parse_magnitudes(d: dict[str, Any]) -> MagnitudeConfig:
    return MagnitudeConfig(
        reward_levels=tuple(float(v) for v in d["reward_levels"]),
        shock_levels=tuple(float(v) for v in d["shock_levels"]),
    )


def _parse_timing(d: dict[str, Any]) -> TimingConfig:
    return TimingConfig(
        cue_duration_s=float(d["cue_duration_s"]),
        anticipation_s=float(d["anticipation_s"]),
        outcome_duration_s=float(d["outcome_duration_s"]),
        iti_s=float(d["iti_s"]),
    )


def _parse_delta_hr_distribution(d: dict[str, Any]) -> DeltaHRDistribution:
    return DeltaHRDistribution(
        mean=float(d["mean"]),
        sd=float(d["sd"]),
    )


def _parse_delta_hr_stub(d: dict[str, Any]) -> DeltaHRStubConfig:
    bounds_raw = d["bounds"]
    return DeltaHRStubConfig(
        dangerous=_parse_delta_hr_distribution(d["dangerous"]),
        safe=_parse_delta_hr_distribution(d["safe"]),
        bounds=(float(bounds_raw[0]), float(bounds_raw[1])),
    )


def _parse_task(d: dict[str, Any]) -> PATRLTaskConfig:
    return PATRLTaskConfig(
        name=str(d["name"]),
        description=str(d["description"]),
        n_trials=int(d["n_trials"]),
        n_runs=int(d["n_runs"]),
        trials_per_run=int(d["trials_per_run"]),
        hazards=_parse_hazards(d["hazards"]),
        run_order=tuple(str(r) for r in d["run_order"]),
        contingencies=_parse_contingencies(d["contingencies"]),
        magnitudes=_parse_magnitudes(d["magnitudes"]),
        timing=_parse_timing(d["timing"]),
        delta_hr_stub=_parse_delta_hr_stub(d["delta_hr_stub"]),
    )


def _parse_prior_gaussian(d: dict[str, Any]) -> PriorGaussian:
    return PriorGaussian(mean=float(d["mean"]), sd=float(d["sd"]))


def _parse_prior_truncated(d: dict[str, Any]) -> PriorTruncated:
    return PriorTruncated(
        lower=float(d["lower"]),
        upper=float(d["upper"]),
        mean=float(d["mean"]),
        sd=float(d["sd"]),
    )


def _parse_phenotype_params(d: dict[str, Any]) -> PhenotypeParams:
    # Required fields (present since Phase 18)
    kwargs: dict[str, Any] = {
        "omega_2": _parse_prior_gaussian(d["omega_2"]),
        "beta": _parse_prior_gaussian(d["beta"]),
        "kappa": _parse_prior_gaussian(d["kappa"]),
        "mu3_0": _parse_prior_gaussian(d["mu3_0"]),
    }
    # Phase 20-01 optional fields — fall back to PhenotypeParams defaults
    # when absent (preserves Phase 18/19 configs that lack these keys).
    if "omega_3" in d:
        kwargs["omega_3"] = _parse_prior_gaussian(d["omega_3"])
    if "theta" in d:
        kwargs["theta"] = _parse_prior_gaussian(d["theta"])
    if "b" in d:
        kwargs["b"] = _parse_prior_gaussian(d["b"])
    if "dhr_mean" in d:
        kwargs["dhr_mean"] = float(d["dhr_mean"])
    if "dhr_sd" in d:
        kwargs["dhr_sd"] = float(d["dhr_sd"])
    if "epsilon2_coupling_coef" in d:
        kwargs["epsilon2_coupling_coef"] = float(d["epsilon2_coupling_coef"])
    return PhenotypeParams(**kwargs)


def _parse_simulation(d: dict[str, Any]) -> PATRLSimulationConfig:
    phenotypes = {
        name: _parse_phenotype_params(params)
        for name, params in d["phenotypes"].items()
    }
    return PATRLSimulationConfig(
        n_participants_per_phenotype=int(d["n_participants_per_phenotype"]),
        master_seed=int(d["master_seed"]),
        phenotypes=phenotypes,
    )


def _parse_fitting_priors(d: dict[str, Any]) -> FittingPriorConfig:
    # Required fields (present since Phase 18)
    kwargs: dict[str, Any] = {
        "omega_2": _parse_prior_gaussian(d["omega_2"]),
        "omega_3": _parse_prior_gaussian(d["omega_3"]),
        "kappa": _parse_prior_truncated(d["kappa"]),
        "beta": _parse_prior_truncated(d["beta"]),
        "mu3_0": _parse_prior_gaussian(d["mu3_0"]),
    }
    # Phase 20-01 optional fields — fall back to FittingPriorConfig defaults
    # when absent (preserves Phase 18/19 configs that lack these keys).
    if "b" in d:
        kwargs["b"] = _parse_prior_gaussian(d["b"])
    if "gamma" in d:
        kwargs["gamma"] = _parse_prior_gaussian(d["gamma"])
    if "alpha" in d:
        kwargs["alpha"] = _parse_prior_gaussian(d["alpha"])
    if "lam" in d:
        kwargs["lam"] = _parse_prior_gaussian(d["lam"])
    return FittingPriorConfig(**kwargs)


def _parse_fitting(d: dict[str, Any]) -> PATRLFittingConfig:
    return PATRLFittingConfig(
        n_chains=int(d["n_chains"]),
        n_tune=int(d["n_tune"]),
        n_draws=int(d["n_draws"]),
        target_accept=float(d["target_accept"]),
        random_seed=int(d["random_seed"]),
        priors=_parse_fitting_priors(d["priors"]),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_pat_rl_config(path: Path | None = None) -> PATRLConfig:
    """Load and validate the PAT-RL configuration file.

    Parameters
    ----------
    path : Path or None, optional
        Path to the YAML configuration file.  Defaults to
        ``configs/pat_rl.yaml`` (relative to the project root as resolved
        by :data:`config.CONFIGS_DIR`).

    Returns
    -------
    PATRLConfig
        Fully validated, frozen configuration tree.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.  Message includes the expected path.
    ValueError
        If any field fails validation.  Message includes expected vs actual
        values.
    """
    resolved: Path = path if path is not None else _DEFAULT_PATRL_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"PAT-RL config not found at expected path: {resolved}")
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text())
    return PATRLConfig(
        task=_parse_task(raw["task"]),
        simulation=_parse_simulation(raw["simulation"]),
        fitting=_parse_fitting(raw["fitting"]),
    )
