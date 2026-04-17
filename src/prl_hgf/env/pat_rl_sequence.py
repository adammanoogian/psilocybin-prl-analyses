"""PAT-RL binary-state trial sequence generator (parallel stack).

Generates the 192-trial / 4-run structure described in
:mod:`prl_hgf.env.pat_rl_config`: hazard-driven binary state reversals,
uniform-random 2x2 reward/shock magnitudes, Delta-HR stub (N(-3,3) on
dangerous, N(0,3) on safe), and cumulative outcome timing in seconds.

This module is independent of :mod:`prl_hgf.env.simulator`; the PAT-RL
dataclass :class:`PATRLTrial` does not inherit from / share fields with
:class:`prl_hgf.env.simulator.Trial`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prl_hgf.env.pat_rl_config import (
    DeltaHRStubConfig,
    PATRLConfig,
    TimingConfig,
)

__all__ = [
    "PATRLTrial",
    "generate_state_sequence",
    "generate_magnitudes",
    "generate_delta_hr_stub",
    "compute_outcome_times_s",
    "generate_session_patrl",
]


# ---------------------------------------------------------------------------
# Trial dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PATRLTrial:
    """One PAT-RL trial record.

    Parameters
    ----------
    trial_idx : int
        Session-level trial index 0..n_trials-1.
    run_idx : int
        Run index 0..n_runs-1.
    trial_in_run : int
        Trial index within this run, 0..trials_per_run-1.
    regime : str
        ``"stable"`` or ``"volatile"`` from
        ``PATRLTaskConfig.run_order[run_idx]``.
    state : int
        Binary context state: 0 = safe, 1 = dangerous.
    reward_mag : float
        Magnitude if outcome == reward (one of
        ``task.magnitudes.reward_levels``).
    shock_mag : float
        Magnitude if outcome == shock (one of
        ``task.magnitudes.shock_levels``).
    delta_hr : float
        Per-trial Delta-HR covariate (anticipatory bradycardia, bpm).
    outcome_time_s : float
        Cumulative time of outcome onset within the session, in seconds.
    """

    trial_idx: int
    run_idx: int
    trial_in_run: int
    regime: str
    state: int
    reward_mag: float
    shock_mag: float
    delta_hr: float
    outcome_time_s: float


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------


def generate_state_sequence(
    n_trials: int,
    hazard: float,
    initial_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a binary state sequence with hazard-driven reversals.

    Parameters
    ----------
    n_trials : int
        Number of trials to generate.  Must be >= 1.
    hazard : float
        Per-trial probability of a state reversal.  Must be in (0, 1).
    initial_state : int
        Starting state: 0 (safe) or 1 (dangerous).
    rng : np.random.Generator
        Caller-managed RNG stream (for reproducibility).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_trials,)`` with values in {0, 1}.

    Notes
    -----
    ``state[0] = initial_state``.  For each subsequent trial ``t``,
    a Bernoulli flip is drawn; when it fires the state reverses, otherwise
    it is carried forward.
    """
    states = np.empty(n_trials, dtype=np.int64)
    states[0] = initial_state
    if n_trials > 1:
        flips = rng.random(n_trials - 1) < hazard
        for t in range(1, n_trials):
            states[t] = 1 - states[t - 1] if flips[t - 1] else states[t - 1]
    return states


def generate_magnitudes(
    n_trials: int,
    reward_levels: tuple[float, ...],
    shock_levels: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw 2x2 reward/shock magnitude pairs uniformly across trials.

    Parameters
    ----------
    n_trials : int
        Number of trials.  Must be >= 1.
    reward_levels : tuple[float, ...]
        Two reward magnitude levels.
    shock_levels : tuple[float, ...]
        Two shock magnitude levels.
    rng : np.random.Generator
        Caller-managed RNG stream.

    Returns
    -------
    reward_mag_arr : np.ndarray
        Float64 array of shape ``(n_trials,)`` with per-trial reward
        magnitudes drawn from *reward_levels*.
    shock_mag_arr : np.ndarray
        Float64 array of shape ``(n_trials,)`` with per-trial shock
        magnitudes drawn from *shock_levels*.

    Notes
    -----
    A cell index in 0..3 is drawn uniformly per trial.  The mapping is:
    ``reward = reward_levels[cell // 2]``,
    ``shock  = shock_levels[cell % 2]``.
    This covers all four 2x2 combinations with equal marginal probability.
    """
    cells = rng.integers(0, 4, size=n_trials)
    reward_arr = np.array(reward_levels, dtype=np.float64)
    shock_arr = np.array(shock_levels, dtype=np.float64)
    reward_mag_arr = reward_arr[cells // 2]
    shock_mag_arr = shock_arr[cells % 2]
    return reward_mag_arr, shock_mag_arr


def generate_delta_hr_stub(
    state_seq: np.ndarray,
    stub_config: DeltaHRStubConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample per-trial Delta-HR from state-conditioned Gaussian stub.

    Parameters
    ----------
    state_seq : np.ndarray
        Binary state array of shape ``(n_trials,)`` with values in {0, 1}.
    stub_config : DeltaHRStubConfig
        Specifies Gaussian parameters for safe/dangerous states and hard
        clamp bounds.
    rng : np.random.Generator
        Caller-managed RNG stream.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_trials,)`` with per-trial Delta-HR
        values clipped to ``stub_config.bounds``.

    Notes
    -----
    Dangerous-state (state=1) trials are sampled from
    ``N(stub_config.dangerous.mean, stub_config.dangerous.sd)``.
    Safe-state (state=0) trials are sampled from
    ``N(stub_config.safe.mean, stub_config.safe.sd)``.
    All values are clipped to ``[bounds[0], bounds[1]]``.
    """
    n_trials = len(state_seq)
    raw = rng.normal(size=n_trials)

    dangerous_mask = state_seq == 1
    delta_hr = np.where(
        dangerous_mask,
        stub_config.dangerous.mean + stub_config.dangerous.sd * raw,
        stub_config.safe.mean + stub_config.safe.sd * raw,
    )
    lo, hi = stub_config.bounds
    return np.clip(delta_hr, lo, hi)


def compute_outcome_times_s(
    n_runs: int,
    trials_per_run: int,
    timing: TimingConfig,
    run_gap_s: float = 15.0,
) -> np.ndarray:
    """Compute absolute cumulative outcome-onset times within a session.

    Parameters
    ----------
    n_runs : int
        Number of runs.  Must be >= 1.
    trials_per_run : int
        Number of trials per run.  Must be >= 1.
    timing : TimingConfig
        Per-trial timing parameters in seconds.
    run_gap_s : float, optional
        Additional gap between consecutive runs in seconds (default 15.0).

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_runs * trials_per_run,)`` with the
        cumulative outcome-onset time in seconds for each trial.

    Notes
    -----
    Within each run the trial onset is
    ``trial_in_run * timing.trial_duration_s``.
    The outcome onset within a trial is
    ``timing.cue_duration_s + timing.anticipation_s``.
    The run-level offset is
    ``run_idx * (trials_per_run * timing.trial_duration_s + run_gap_s)``.
    """
    n_trials = n_runs * trials_per_run
    outcome_offset_within_trial = timing.cue_duration_s + timing.anticipation_s
    run_duration_s = trials_per_run * timing.trial_duration_s
    times = np.empty(n_trials, dtype=np.float64)
    for run_idx in range(n_runs):
        run_start = run_idx * (run_duration_s + run_gap_s)
        for trial_in_run in range(trials_per_run):
            t = run_idx * trials_per_run + trial_in_run
            times[t] = (
                run_start
                + trial_in_run * timing.trial_duration_s
                + outcome_offset_within_trial
            )
    return times


# ---------------------------------------------------------------------------
# Top-level session generator
# ---------------------------------------------------------------------------


def generate_session_patrl(
    config: PATRLConfig,
    seed: int,
    delta_hr_override: np.ndarray | None = None,
) -> list[PATRLTrial]:
    """Generate a full PAT-RL session under *config*.

    Parameters
    ----------
    config : PATRLConfig
        Loaded PAT-RL configuration.
    seed : int
        Master seed; four child RNG streams are spawned via
        ``np.random.SeedSequence(seed).spawn(4)`` mapping to
        (state, magnitudes, delta_hr, reserved).
    delta_hr_override : np.ndarray or None, optional
        If provided (shape ``(n_trials,)``), bypass the stub and use the
        supplied values.  Values are still clipped to
        ``config.task.delta_hr_stub.bounds``.

    Returns
    -------
    list[PATRLTrial]
        Length ``config.task.n_trials``.  Trials are in session order with
        ``trial_idx`` 0..n_trials-1.

    Notes
    -----
    RNG stream split: ``SeedSequence(seed).spawn(4)`` produces four
    independent child sequences assigned to state generation, magnitude
    sampling, Delta-HR stub sampling, and a reserved stream (for future
    outcome draws).  This guarantees independence between streams even when
    *seed* values are close integers.
    """
    task = config.task
    n_trials: int = task.n_trials
    n_runs: int = task.n_runs
    trials_per_run: int = task.trials_per_run

    # Spawn 4 independent child RNG streams from a single master seed.
    ss = np.random.SeedSequence(seed)
    rng_state, rng_mag, rng_dhr, _reserved = (
        np.random.default_rng(s) for s in ss.spawn(4)
    )

    # --- Build full state sequence across all runs ---
    state_full = np.empty(n_trials, dtype=np.int64)
    initial_state: int = 0  # session always starts in safe state
    for run_idx in range(n_runs):
        regime = task.run_order[run_idx]
        hazard = (
            task.hazards.stable if regime == "stable" else task.hazards.volatile
        )
        state_run = generate_state_sequence(
            trials_per_run, hazard, initial_state, rng_state
        )
        start = run_idx * trials_per_run
        state_full[start : start + trials_per_run] = state_run
        initial_state = int(state_run[-1])  # carry forward last state

    # --- Magnitudes ---
    reward_mag_arr, shock_mag_arr = generate_magnitudes(
        n_trials,
        task.magnitudes.reward_levels,
        task.magnitudes.shock_levels,
        rng_mag,
    )

    # --- Delta-HR ---
    lo, hi = task.delta_hr_stub.bounds
    if delta_hr_override is not None:
        delta_hr_arr = np.clip(
            np.asarray(delta_hr_override, dtype=np.float64), lo, hi
        )
    else:
        delta_hr_arr = generate_delta_hr_stub(
            state_full, task.delta_hr_stub, rng_dhr
        )

    # --- Outcome onset times ---
    outcome_time_arr = compute_outcome_times_s(
        n_runs, trials_per_run, task.timing
    )

    # --- Assemble trial list ---
    trials: list[PATRLTrial] = []
    for trial_idx in range(n_trials):
        run_idx = trial_idx // trials_per_run
        trial_in_run = trial_idx % trials_per_run
        trials.append(
            PATRLTrial(
                trial_idx=trial_idx,
                run_idx=run_idx,
                trial_in_run=trial_in_run,
                regime=task.run_order[run_idx],
                state=int(state_full[trial_idx]),
                reward_mag=float(reward_mag_arr[trial_idx]),
                shock_mag=float(shock_mag_arr[trial_idx]),
                delta_hr=float(delta_hr_arr[trial_idx]),
                outcome_time_s=float(outcome_time_arr[trial_idx]),
            )
        )
    return trials
