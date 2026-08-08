"""Closed-loop (criterion-based) cohort simulation for pick_best_cue.

Simulates the DEPLOYED task variant in which reversals fire when a
performance criterion is met (or a per-phase trial cap is reached) rather
than after a fixed number of trials. Because phase transitions depend on the
agent's own choices, the environment and the agent must be stepped together
trial by trial — a plain NumPy loop, not the vmapped JAX kernel used for
fixed-length schedules.

Generative model fidelity
-------------------------
The agent is IDENTICAL to the fixed-phase simulators
(:func:`~prl_hgf.simulation.agent.simulate_agent` and
:mod:`~prl_hgf.simulation.jax_session`):

* same parameter set (``omega_2``, ``omega_3``, ``kappa``, ``beta``,
  ``zeta``),
* same 3-branch binary HGF belief update via the pyhgf network from
  :func:`~prl_hgf.models.hgf_3level.build_3level_network` with the attribute
  carry pattern,
* same softmax + stickiness response model
  ``logits = beta * p_reward + zeta * stick``,
* same partial feedback (only the chosen cue's outcome is observed),
* same tapas-style stability clamp as the JAX kernel: if a belief update
  produces non-finite values or ``|mu_2| >= 14``, the update is reverted and
  the session is flagged ``diverged`` (simulation continues).

Task structure per set
----------------------
``acquisition_1`` (initial best cue from ``task.initial_best_per_set``),
then ``n_reversals_per_set`` reversal phases (``reversal_1``,
``reversal_2``, ...) with the new best cue drawn per ``target_rule``, then a
``transfer`` phase in which no feedback is given: choices are still logged
but the HGF belief state is NOT updated.

Seed strategy
-------------
Mirrors :func:`~prl_hgf.simulation.batch.simulate_batch`: a master RNG
derived from ``config.simulation.master_seed`` draws one
``(env_seed, sim_seed)`` pair per participant-session upfront. ``env_seed``
drives all environment randomness (criterion jitter, reversal targets,
reward outcomes); ``sim_seed`` drives parameter sampling and choice
sampling. Identical master seeds therefore always reproduce identical
output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from prl_hgf.env.simulator import Trial
from prl_hgf.env.task_config import (
    AnalysisConfig,
    CriterionConfig,
    TaskConfig,
)
from prl_hgf.models.hgf_2level import INPUT_NODES
from prl_hgf.models.hgf_3level import BELIEF_NODES, build_3level_network
from prl_hgf.simulation.agent import sample_participant_params

__all__ = [
    "ConsecutiveCorrectTracker",
    "WindowTracker",
    "CriterionEnvironment",
    "CriterionSessionResult",
    "draw_new_best_cue",
    "make_criterion_tracker",
    "simulate_criterion_session",
    "simulate_criterion_batch",
]

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 means — must match
#: :data:`prl_hgf.simulation.jax_session._MU_2_BOUND`.
_MU_2_BOUND: float = 14.0


# ---------------------------------------------------------------------------
# Criterion trackers
# ---------------------------------------------------------------------------


class ConsecutiveCorrectTracker:
    """Consecutive-correct reversal criterion state for one learning phase.

    Counts consecutive choices of the objectively best cue; any non-best
    choice resets the count to zero.

    Parameters
    ----------
    threshold : int
        Number of consecutive best-cue choices required to meet the
        criterion (must be >= 1).
    """

    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError(
                f"ConsecutiveCorrectTracker: threshold must be >= 1, got {threshold}."
            )
        self.threshold = threshold
        self._n_consecutive = 0

    def update(self, chose_best: bool) -> bool:
        """Register one choice and report whether the criterion is met.

        Parameters
        ----------
        chose_best : bool
            ``True`` if the agent chose the objectively best cue.

        Returns
        -------
        bool
            ``True`` once the consecutive-correct count reaches
            ``threshold``.
        """
        self._n_consecutive = self._n_consecutive + 1 if chose_best else 0
        return self._n_consecutive >= self.threshold


class WindowTracker:
    """Sliding-window reversal criterion state for one learning phase.

    Counts best-cue choices within the last ``window_size`` trials of the
    current phase; the criterion is met when that count reaches
    ``n_correct_required``.

    Parameters
    ----------
    window_size : int
        Sliding window length in trials (must be >= 1).
    n_correct_required : int
        Best-cue choices required within the window (must be in
        ``[1, window_size]``).
    """

    def __init__(self, window_size: int, n_correct_required: int) -> None:
        if window_size < 1:
            raise ValueError(
                f"WindowTracker: window_size must be >= 1, got {window_size}."
            )
        if not (1 <= n_correct_required <= window_size):
            raise ValueError(
                f"WindowTracker: n_correct_required must be in "
                f"[1, {window_size}], got {n_correct_required}."
            )
        self.window_size = window_size
        self.n_correct_required = n_correct_required
        self._window: deque[bool] = deque(maxlen=window_size)

    def update(self, chose_best: bool) -> bool:
        """Register one choice and report whether the criterion is met.

        Parameters
        ----------
        chose_best : bool
            ``True`` if the agent chose the objectively best cue.

        Returns
        -------
        bool
            ``True`` once the window holds at least ``n_correct_required``
            best-cue choices.
        """
        self._window.append(chose_best)
        return sum(self._window) >= self.n_correct_required


def make_criterion_tracker(
    criterion: CriterionConfig,
    rng: np.random.Generator,
) -> ConsecutiveCorrectTracker | WindowTracker:
    """Build a fresh criterion tracker for one learning phase.

    For ``consecutive_correct`` the threshold is jittered: drawn uniformly
    from ``[n_correct_min, n_correct_max]`` (inclusive) using ``rng``.
    ``window`` criteria are deterministic and consume no random draws.

    Parameters
    ----------
    criterion : CriterionConfig
        Validated criterion configuration.
    rng : numpy.random.Generator
        Environment RNG used for the jittered threshold draw.

    Returns
    -------
    ConsecutiveCorrectTracker or WindowTracker
        Tracker with reset state for the new phase.
    """
    if criterion.criterion_type == "consecutive_correct":
        assert criterion.n_correct_min is not None  # validated in config
        assert criterion.n_correct_max is not None
        threshold = int(
            rng.integers(criterion.n_correct_min, criterion.n_correct_max + 1)
        )
        return ConsecutiveCorrectTracker(threshold)
    assert criterion.window_size is not None  # validated in config
    assert criterion.window_n_correct is not None
    return WindowTracker(criterion.window_size, criterion.window_n_correct)


# ---------------------------------------------------------------------------
# Reversal target rule
# ---------------------------------------------------------------------------


def draw_new_best_cue(
    current_best: int,
    n_cues: int,
    target_rule: str,
    rng: np.random.Generator,
) -> int:
    """Draw the best cue for the next learning phase at a reversal.

    Parameters
    ----------
    current_best : int
        Best cue index of the phase that just ended.
    n_cues : int
        Total number of cues (must be >= 2).
    target_rule : str
        Reversal target rule. Only ``"random_nonbest"`` is supported: the
        new best cue is drawn uniformly from the non-best cues.
    rng : numpy.random.Generator
        Environment RNG used for the draw.

    Returns
    -------
    int
        New best cue index, guaranteed different from ``current_best``.

    Raises
    ------
    ValueError
        If ``target_rule`` is not a supported rule.
    """
    if target_rule != "random_nonbest":
        raise ValueError(
            f"draw_new_best_cue: target_rule must be 'random_nonbest', "
            f"got '{target_rule}'."
        )
    candidates = [c for c in range(n_cues) if c != current_best]
    return int(rng.choice(candidates))


# ---------------------------------------------------------------------------
# Closed-loop environment state machine
# ---------------------------------------------------------------------------


class CriterionEnvironment:
    """Closed-loop pick_best_cue environment for one session.

    Steps through sets, learning phases, and transfer phases, advancing
    phase boundaries according to the reversal criterion and the per-phase
    trial cap. The agent interacts via :meth:`current_trial` (observe the
    trial context) and :meth:`step` (submit a choice, receive the outcome).

    All environment randomness (criterion jitter, reversal targets, reward
    outcomes) is drawn from the single generator ``rng`` in a fixed order,
    so identical seeds and identical choice sequences reproduce identical
    sessions.

    Parameters
    ----------
    task_cfg : TaskConfig
        Criterion-based task configuration (``is_criterion_based`` True).
    rng : numpy.random.Generator
        Seeded environment RNG.

    Raises
    ------
    ValueError
        If ``task_cfg`` is not criterion-based.
    """

    def __init__(self, task_cfg: TaskConfig, rng: np.random.Generator) -> None:
        if not task_cfg.is_criterion_based:
            raise ValueError(
                "CriterionEnvironment: expected a criterion-based TaskConfig "
                "(task.reversal set), got a fixed-phase config "
                "(task.reversal is None)."
            )
        assert task_cfg.reversal is not None  # narrowed by the check above
        assert task_cfg.initial_best_per_set is not None
        assert task_cfg.cue_prob_pair is not None
        self._cfg = task_cfg
        self._reversal = task_cfg.reversal
        self._rng = rng
        self._n_learning_phases = self._reversal.n_reversals_per_set + 1
        self._transfer_best = int(np.argmax(task_cfg.transfer.cue_probs))

        self._trial_idx = 0
        self._set_idx = 0
        self._phase_idx = 0  # 0 = acquisition; 1.. = reversals; == n -> transfer
        self._trial_in_phase = 0
        self._done = False
        self._best_cue = task_cfg.initial_best_per_set[0]
        self._tracker = make_criterion_tracker(self._reversal.criterion, rng)

    @property
    def done(self) -> bool:
        """Whether all sets (including their transfer phases) are complete.

        Returns
        -------
        bool
            ``True`` when the session has ended.
        """
        return self._done

    @property
    def in_transfer(self) -> bool:
        """Whether the current trial belongs to a transfer phase.

        Transfer trials give NO feedback: the outcome is logged but must not
        be shown to (or update) the agent.

        Returns
        -------
        bool
            ``True`` during transfer phases.
        """
        return self._phase_idx >= self._n_learning_phases

    def current_trial(self) -> Trial:
        """Build the trial context for the upcoming trial.

        Returns
        -------
        Trial
            Immutable trial record with the session-continuous 0-based
            ``trial_idx``, phase naming (``acquisition_1``, ``reversal_k``,
            ``transfer``), the current cue probabilities, and the best cue.

        Raises
        ------
        RuntimeError
            If the session is already complete.
        """
        if self._done:
            raise RuntimeError(
                "CriterionEnvironment: current_trial() called on a finished "
                "session (expected done == False, got True)."
            )
        if self.in_transfer:
            transfer = self._cfg.transfer
            return Trial(
                trial_idx=self._trial_idx,
                set_idx=self._set_idx,
                phase_name="transfer",
                phase_label=transfer.phase_label,
                cue_probs=tuple(transfer.cue_probs),
                best_cue=self._transfer_best,
            )
        pair = self._cfg.cue_prob_pair
        assert pair is not None  # validated at construction
        probs = [pair.other] * self._cfg.n_cues
        probs[self._best_cue] = pair.best
        if self._phase_idx == 0:
            phase_name = "acquisition_1"
            phase_label = "stable"
        else:
            phase_name = f"reversal_{self._phase_idx}"
            phase_label = "volatile"
        return Trial(
            trial_idx=self._trial_idx,
            set_idx=self._set_idx,
            phase_name=phase_name,
            phase_label=phase_label,
            cue_probs=tuple(probs),
            best_cue=self._best_cue,
        )

    def step(self, choice: int) -> int:
        """Submit the agent's choice, draw the outcome, and advance state.

        On learning trials the criterion state is updated with whether the
        choice hit the objectively best cue; the phase ends (reversal or
        entry into transfer) when the criterion is met OR
        ``max_trials_per_phase`` is reached. On transfer trials the outcome
        is drawn for logging only — no feedback is given.

        Parameters
        ----------
        choice : int
            Chosen cue index, in ``[0, n_cues)``.

        Returns
        -------
        int
            Binary outcome (1 rewarded, 0 not) drawn from the chosen cue's
            current reward probability.

        Raises
        ------
        ValueError
            If ``choice`` is out of range.
        RuntimeError
            If the session is already complete.
        """
        if self._done:
            raise RuntimeError(
                "CriterionEnvironment: step() called on a finished session "
                "(expected done == False, got True)."
            )
        if not (0 <= choice < self._cfg.n_cues):
            raise ValueError(
                f"CriterionEnvironment: choice must be in "
                f"[0, {self._cfg.n_cues - 1}], got {choice}."
            )

        trial = self.current_trial()
        reward = int(self._rng.random() < trial.cue_probs[choice])
        self._trial_idx += 1
        self._trial_in_phase += 1

        if self.in_transfer:
            if self._trial_in_phase >= self._cfg.transfer.n_trials:
                self._advance_set()
            return reward

        criterion_met = self._tracker.update(choice == self._best_cue)
        if criterion_met or self._trial_in_phase >= self._reversal.max_trials_per_phase:
            self._advance_phase()
        return reward

    def _advance_phase(self) -> None:
        """End the current learning phase: reverse or enter transfer."""
        self._phase_idx += 1
        self._trial_in_phase = 0
        if self._phase_idx < self._n_learning_phases:
            self._best_cue = draw_new_best_cue(
                self._best_cue,
                self._cfg.n_cues,
                self._reversal.target_rule,
                self._rng,
            )
            self._tracker = make_criterion_tracker(self._reversal.criterion, self._rng)
        elif self._cfg.transfer.n_trials == 0:
            self._advance_set()

    def _advance_set(self) -> None:
        """End the current set: start the next set or finish the session."""
        self._set_idx += 1
        self._trial_in_phase = 0
        self._phase_idx = 0
        if self._set_idx >= self._cfg.n_sets:
            self._done = True
            return
        assert self._cfg.initial_best_per_set is not None
        self._best_cue = self._cfg.initial_best_per_set[self._set_idx]
        self._tracker = make_criterion_tracker(self._reversal.criterion, self._rng)


# ---------------------------------------------------------------------------
# Session simulation (HGF agent in the loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionSessionResult:
    """Immutable output of one closed-loop participant-session.

    Parameters
    ----------
    trials : list[Trial]
        Trial contexts in session order (variable length).
    choices : list[int]
        Chosen cue index per trial.
    rewards : list[int]
        Binary outcome per trial. On transfer trials the outcome is drawn
        for logging but was NOT shown to the agent.
    beliefs : list[tuple[float, float, float]]
        Prior reward-probability belief per cue, read before each trial's
        update.
    diverged : bool
        ``True`` if any belief update was reverted by the stability clamp.
    """

    trials: list[Trial]
    choices: list[int]
    rewards: list[int]
    beliefs: list[tuple[float, float, float]]
    diverged: bool = False


def simulate_criterion_session(
    task_cfg: TaskConfig,
    params: dict[str, float],
    env_rng: np.random.Generator,
    choice_rng: np.random.Generator,
) -> CriterionSessionResult:
    """Simulate one participant-session against the closed-loop environment.

    Per trial: read prior HGF beliefs, sample a choice from the softmax +
    stickiness response model, draw the outcome from the environment, then —
    on learning trials only — update the HGF with partial feedback (only the
    chosen cue observed) and apply the tapas-style stability clamp. Transfer
    trials give no feedback, so the belief state is left untouched while
    choices are still logged.

    Parameters
    ----------
    task_cfg : TaskConfig
        Criterion-based task configuration with ``n_cues == 3`` (the HGF
        network topology is fixed at 3 branches).
    params : dict[str, float]
        Participant parameters with keys ``"omega_2"``, ``"omega_3"``,
        ``"kappa"``, ``"beta"``, ``"zeta"`` (e.g. from
        :func:`~prl_hgf.simulation.agent.sample_participant_params`).
    env_rng : numpy.random.Generator
        Seeded environment RNG (criterion jitter, reversal targets,
        outcomes).
    choice_rng : numpy.random.Generator
        Seeded agent RNG (softmax choice sampling).

    Returns
    -------
    CriterionSessionResult
        Trial contexts, choices, outcomes, prior beliefs, and the
        divergence flag for this session.

    Raises
    ------
    ValueError
        If ``task_cfg.n_cues != 3`` or the config is not criterion-based.
    """
    if task_cfg.n_cues != 3:
        raise ValueError(
            f"simulate_criterion_session: the 3-branch HGF network requires "
            f"n_cues == 3, got {task_cfg.n_cues}."
        )
    env = CriterionEnvironment(task_cfg, env_rng)
    net = build_3level_network(
        omega_2=params["omega_2"],
        omega_3=params["omega_3"],
        kappa=params["kappa"],
    )
    beta = params["beta"]
    zeta = params["zeta"]

    trials: list[Trial] = []
    choices: list[int] = []
    rewards: list[int] = []
    beliefs: list[tuple[float, float, float]] = []
    prev_choice = -1  # sentinel: no stickiness on the first trial
    diverged = False

    while not env.done:
        trial = env.current_trial()

        # --- Step 1: read PRIOR beliefs before this trial's update ---
        p_reward = np.array(
            [float(net.attributes[node]["expected_mean"]) for node in INPUT_NODES]
        )

        # --- Step 2: softmax with stickiness (identical response model) ---
        stick = np.zeros(3)
        if prev_choice >= 0:
            stick[prev_choice] = 1.0
        logits = beta * p_reward + zeta * stick
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()

        # --- Step 3: sample choice; environment draws the outcome ---
        choice = int(choice_rng.choice(3, p=probs))
        in_transfer = env.in_transfer
        reward = env.step(choice)

        trials.append(trial)
        choices.append(choice)
        rewards.append(reward)
        beliefs.append((float(p_reward[0]), float(p_reward[1]), float(p_reward[2])))

        # --- Step 4: HGF update with partial feedback (learning trials only) ---
        if not in_transfer:
            inp_t = np.zeros((1, 3), dtype=float)
            obs_t = np.zeros((1, 3), dtype=int)
            inp_t[0, choice] = float(reward)
            obs_t[0, choice] = 1
            net.input_data(input_data=inp_t, observed=obs_t)

            # Tapas-style stability clamp (mirrors jax_session._run_session):
            # revert the update if any leaf is non-finite or |mu_2| >= bound.
            new_attrs = net.last_attributes
            leaves = jax.tree_util.tree_leaves(new_attrs)
            all_finite = all(bool(np.all(np.isfinite(leaf))) for leaf in leaves)
            mu_2_vals = [float(new_attrs[node]["mean"]) for node in BELIEF_NODES]
            mu_2_ok = all(abs(v) < _MU_2_BOUND for v in mu_2_vals)
            if all_finite and mu_2_ok:
                # Carry posterior forward as the next trial's prior.
                net.attributes = new_attrs
            else:
                diverged = True  # keep previous attributes (revert)

        prev_choice = choice

    return CriterionSessionResult(
        trials=trials,
        choices=choices,
        rewards=rewards,
        beliefs=beliefs,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def simulate_criterion_batch(
    config: AnalysisConfig,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Simulate a full closed-loop cohort across all groups and sessions.

    Mirrors :func:`~prl_hgf.simulation.batch.simulate_batch` exactly in
    cohort structure, seed derivation, and output schema — only the
    per-session simulation differs (sequential closed-loop instead of the
    vmapped fixed-length kernel). Session lengths vary by participant.

    Parameters
    ----------
    config : AnalysisConfig
        Validated criterion-based configuration
        (``config.is_criterion_based`` must be ``True``).
    output_path : Path or None, optional
        If provided, the resulting DataFrame is saved as CSV at this path.
        The parent directory must exist.

    Returns
    -------
    pandas.DataFrame
        Tidy trial-level DataFrame, one row per trial, with the same
        columns as the fixed-phase path:

        ``participant_id``, ``group``, ``session``, ``session_idx``,
        ``trial``, ``cue_chosen``, ``reward``,
        ``cue_0_prob``, ``cue_1_prob``, ``cue_2_prob``,
        ``phase_label``, ``phase_name``, ``best_cue``,
        ``true_omega_2``, ``true_omega_3``, ``true_kappa``,
        ``true_beta``, ``true_zeta``, ``model``, ``diverged``

    Raises
    ------
    ValueError
        If ``config`` is not criterion-based.
    """
    if not config.is_criterion_based:
        raise ValueError(
            "simulate_criterion_batch: expected a criterion-based config "
            "(task.reversal set), got a fixed-phase config — use "
            "prl_hgf.simulation.batch.simulate_batch instead."
        )

    sim_cfg = config.simulation
    n_per_group: int = sim_cfg.n_participants_per_group
    group_names: list[str] = sorted(sim_cfg.groups.keys())
    n_sessions: int = 3
    n_total: int = len(group_names) * n_per_group * n_sessions

    # --- Derive all (env_seed, sim_seed) pairs upfront (same as batch) ---
    rng_master = np.random.default_rng(sim_cfg.master_seed)
    all_seeds = rng_master.integers(0, 2**31, size=(n_total, 2))

    rows: list[dict] = []
    flat_idx = 0

    for group_name in group_names:
        group_cfg = sim_cfg.groups[group_name]
        session_cfg = sim_cfg.session_deltas[group_name]
        session_labels = ["baseline"] + list(session_cfg.session_labels)

        for participant_idx in range(n_per_group):
            participant_id = f"{group_name}_{participant_idx:03d}"

            for session_idx, session_label in enumerate(session_labels):
                env_seed = int(all_seeds[flat_idx, 0])
                sim_seed = int(all_seeds[flat_idx, 1])
                flat_idx += 1

                # sim_seed drives params AND choices (as in the batch path).
                rng_sim = np.random.default_rng(sim_seed)
                params = sample_participant_params(
                    group_cfg, session_cfg, session_idx, rng_sim
                )
                env_rng = np.random.default_rng(env_seed)

                result = simulate_criterion_session(
                    config.task, params, env_rng, rng_sim
                )

                for t_idx, trial in enumerate(result.trials):
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "group": group_name,
                            "session": session_label,
                            "session_idx": session_idx,
                            "trial": trial.trial_idx,
                            "cue_chosen": result.choices[t_idx],
                            "reward": result.rewards[t_idx],
                            "cue_0_prob": trial.cue_probs[0],
                            "cue_1_prob": trial.cue_probs[1],
                            "cue_2_prob": trial.cue_probs[2],
                            "phase_label": trial.phase_label,
                            "phase_name": trial.phase_name,
                            "best_cue": trial.best_cue,
                            "true_omega_2": params["omega_2"],
                            "true_omega_3": params["omega_3"],
                            "true_kappa": params["kappa"],
                            "true_beta": params["beta"],
                            "true_zeta": params["zeta"],
                            "model": "hgf_3level",
                            "diverged": result.diverged,
                        }
                    )

    df = pd.DataFrame(rows)

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved simulation output to: {output_path}")

    return df
