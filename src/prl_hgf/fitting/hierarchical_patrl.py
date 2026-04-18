"""PAT-RL batched logp factory + BlackJAX NUTS orchestrator (parallel stack).

This module is a parallel fitting entry point for the PAT-RL task that
**reuses** generic BlackJAX helpers from :mod:`prl_hgf.fitting.hierarchical`
(``_run_blackjax_nuts``, ``_samples_to_idata``, ``_extract_nuts_stats``)
without modifying any of them.

Architecture
------------
* :func:`_build_session_scanner_patrl` — builds the pyhgf Network ONCE
  outside ``jax.vmap``/``jax.jit``, capturing ``scan_fn`` and ``base_attrs``
  in a closure (mirrors the factory pattern in
  ``prl_hgf.simulation.jax_session``).
* :func:`build_logp_fn_batched_patrl` — creates a batched ``logp_fn``
  that vmaps :func:`_single_logp` over the participant dimension.
* :func:`_build_arrays_single_patrl` — converts a PAT-RL ``sim_df`` into
  the ``(P, n_trials)`` NumPy arrays required by the logp factory.
* :func:`fit_batch_hierarchical_patrl` — top-level entry point; constructs
  a pure-JAX log-posterior and drives BlackJAX NUTS via the reused helpers.

Parallel-stack invariant
------------------------
``hierarchical.py`` is NOT modified.  Its functions are imported and called
unchanged.  PAT-RL fitting concerns live entirely in this module.

Scope (Phase 18)
----------------
Model A only.  Models B/C/D (Delta-HR bias, trial-varying omega) are
deferred to Phase 19+.

Notes
-----
* ``delta_hr`` is assembled by :func:`_build_arrays_single_patrl` for
  downstream CSV export (Phase 18-05) but is NOT consumed by the Model A
  logp.  It will enter the logp signature in Phase 19+ for Models B/C/D.
* Layer-2 clamping is implemented inline (``_MU_2_BOUND = 14.0``) and does
  NOT import from :mod:`prl_hgf.fitting.hierarchical` to preserve the
  parallel-stack invariant.
* kappa is stored in ``attrs[VOLATILITY_NODE]["volatility_coupling_children"]``
  as a JAX array and can be injected dynamically at runtime for the 3-level
  model.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from prl_hgf.env.pat_rl_config import PATRLConfig, load_pat_rl_config

# Reuse generic BlackJAX helpers from the pick_best_cue fitting module.
# These are logdensity-shape generic — no PAT-RL specific assumptions.
# hierarchical.py is NOT modified; these are pure imports.
from prl_hgf.fitting.hierarchical import (
    _run_blackjax_nuts,
    _samples_to_idata,
)
from prl_hgf.models.hgf_2level_patrl import build_2level_network_patrl
from prl_hgf.models.hgf_3level_patrl import build_3level_network_patrl
from prl_hgf.models.response_patrl import model_a_logp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 belief means (tapas_ehgf_binary.m).
#: Implemented inline here; do NOT import from hierarchical.py (that would
#: couple the parallel stacks).
_MU_2_BOUND: float = 14.0

#: Node index for the continuous-state value parent (level-1 belief).
_BELIEF_NODE: int = 1

#: Node index for the continuous-state volatility parent (3-level only).
_VOLATILITY_NODE: int = 2

#: Supported PAT-RL model names.
_PATRL_MODEL_NAMES: tuple[str, ...] = ("hgf_2level_patrl", "hgf_3level_patrl")

#: Required columns in the PAT-RL sim_df.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "participant_id",
        "trial_idx",
        "state",
        "choice",
        "reward_mag",
        "shock_mag",
        "delta_hr",
        "outcome_time_s",
    }
)

__all__ = [
    "build_logp_fn_batched_patrl",
    "fit_batch_hierarchical_patrl",
    "_build_arrays_single_patrl",
    "_build_session_scanner_patrl",
]


# ---------------------------------------------------------------------------
# Network factory (private)
# ---------------------------------------------------------------------------


def _build_session_scanner_patrl(
    model_name: str,
) -> tuple[dict, Any, int]:
    """Build ``(base_attrs, scan_fn, belief_idx)`` for the PAT-RL network.

    Constructs the pyhgf ``Network`` ONCE outside ``jax.vmap``/``jax.jit``
    so that ``scan_fn`` can be captured as a closed-over pure-JAX callable.

    Parameters
    ----------
    model_name : {"hgf_2level_patrl", "hgf_3level_patrl"}
        PAT-RL model variant.

    Returns
    -------
    base_attrs : dict
        Initial attributes pytree from the primed network.
    scan_fn : callable
        pyhgf ``Network.scan_fn`` (a ``jax.tree_util.Partial``).
    belief_idx : int
        Node index for the continuous-state value parent (always ``1``).

    Raises
    ------
    ValueError
        If ``model_name`` is not a recognised PAT-RL model.
    """
    if model_name not in _PATRL_MODEL_NAMES:
        msg = (
            f"model_name must be one of {_PATRL_MODEL_NAMES}, "
            f"got {model_name!r}"
        )
        raise ValueError(msg)

    if model_name == "hgf_2level_patrl":
        net = build_2level_network_patrl()
    else:  # hgf_3level_patrl
        net = build_3level_network_patrl()

    belief_idx = _BELIEF_NODE  # always node 1 for both variants

    # Prime scan_fn with one dummy observation so attributes are initialised.
    dummy = np.zeros((1, 1), dtype=np.float64)
    net.input_data(input_data=dummy, time_steps=np.ones(1, dtype=np.float64))

    base_attrs = net.attributes
    scan_fn = net.scan_fn
    return base_attrs, scan_fn, belief_idx


# ---------------------------------------------------------------------------
# Per-participant logp (private)
# ---------------------------------------------------------------------------


def _make_single_logp_fn(
    base_attrs: dict,
    scan_fn: Any,
    belief_idx: int,
    model_name: str,
    n_trials: int,
) -> Any:
    """Return a single-participant logp function closing over network state.

    The returned function is compatible with ``jax.vmap``.

    Parameters
    ----------
    base_attrs : dict
        Initial attributes pytree from :func:`_build_session_scanner_patrl`.
    scan_fn : callable
        pyhgf ``Network.scan_fn``.
    belief_idx : int
        Belief-node index (always ``1``).
    model_name : str
        PAT-RL model variant.
    n_trials : int
        Number of trials per participant.

    Returns
    -------
    callable
        ``(params, state, choices, reward, shock, mask) -> scalar``.
    """
    is_3level = model_name == "hgf_3level_patrl"

    def _single_logp(
        params: dict[str, jnp.ndarray],
        state: jnp.ndarray,
        choices: jnp.ndarray,
        reward: jnp.ndarray,
        shock: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute single-participant log-likelihood under Model A.

        Parameters
        ----------
        params : dict
            Scalar parameter values for this participant:
            ``omega_2`` (and ``omega_3``, ``kappa``, ``mu3_0`` for 3-level),
            ``beta``.
        state : jnp.ndarray
            Shape ``(n_trials,)`` int32 binary context states.
        choices : jnp.ndarray
            Shape ``(n_trials,)`` int32 binary choices (1=approach).
        reward : jnp.ndarray
            Shape ``(n_trials,)`` float32 reward magnitudes.
        shock : jnp.ndarray
            Shape ``(n_trials,)`` float32 shock magnitudes.
        mask : jnp.ndarray
            Shape ``(n_trials,)`` bool trial mask.

        Returns
        -------
        jnp.ndarray
            Scalar log-likelihood.
        """
        # 1) Inject omega_2 (and 3-level params) into a fresh copy of base_attrs.
        attrs = {**base_attrs}
        attrs[belief_idx] = {
            **attrs[belief_idx],
            "tonic_volatility": params["omega_2"],
        }
        if is_3level:
            attrs[_VOLATILITY_NODE] = {
                **attrs[_VOLATILITY_NODE],
                "tonic_volatility": params["omega_3"],
                # kappa stored as 1-element array in volatility_coupling_children
                "volatility_coupling_children": jnp.asarray([params["kappa"]]),
                "mean": params["mu3_0"],
            }

        # 2) Build scan inputs: binary state as float input, 1D time_steps.
        # Use float64 to match pyhgf attribute dtype (network primed with
        # float64 numpy arrays; jax.lax.cond inside pyhgf requires dtype
        # consistency between branches and the existing precision arrays).
        values = state.astype(jnp.float64)[:, None]  # (T, 1)
        observed = jnp.ones((n_trials,), dtype=jnp.int32)  # (T,)
        time_steps = jnp.ones((n_trials,), dtype=jnp.float64)  # (T,)

        # 3) Tapas-style Layer-2 clamped scan.
        #    _MU_2_BOUND = 14.0 (defined at module level; NOT imported from
        #    hierarchical.py to preserve the parallel-stack invariant).
        def _clamped_step(
            carry: dict,
            x: tuple,
        ) -> tuple[dict, jnp.ndarray]:
            val_i, obs_i, ts_i = x
            new_carry, _traj = scan_fn(carry, ((val_i,), (obs_i,), ts_i, None))

            # Layer-2 clamp: revert if |mu2| >= _MU_2_BOUND or non-finite.
            new_mean = new_carry[belief_idx]["mean"]
            is_stable = jnp.all(jnp.isfinite(new_mean)) & (
                jnp.abs(new_mean) < _MU_2_BOUND
            )
            safe_carry = jax.tree_util.tree_map(
                lambda n, o: jnp.where(is_stable, n, o),
                new_carry,
                carry,
            )
            # Per-trial output: belief-node mu2 after clamping.
            return safe_carry, safe_carry[belief_idx]["mean"]

        # 4) Scan over trials; collect mu2 per step.
        _, mu2_traj = jax.lax.scan(
            _clamped_step,
            attrs,
            (values, observed, time_steps),
        )
        # mu2_traj shape: (T,) — scalar mean per trial.

        # 5) Compute per-trial Model A log-likelihood.
        logp_per_trial = model_a_logp(
            mu2_traj,
            choices.astype(jnp.int32),
            reward.astype(jnp.float64),
            shock.astype(jnp.float64),
            params["beta"],
        )

        # 6) Apply trial mask and sum.
        return jnp.sum(jnp.where(mask, logp_per_trial, 0.0))

    return _single_logp


# ---------------------------------------------------------------------------
# Public logp factory
# ---------------------------------------------------------------------------


def build_logp_fn_batched_patrl(
    state_arr: np.ndarray,
    choices_arr: np.ndarray,
    reward_mag_arr: np.ndarray,
    shock_mag_arr: np.ndarray,
    trial_mask: np.ndarray,
    model_name: str,
) -> Any:
    """Build a batched (vmapped) pure-JAX logp function for PAT-RL Model A.

    Constructs the pyhgf network ONCE (via :func:`_build_session_scanner_patrl`),
    then wraps a per-participant logp in ``jax.vmap`` over the participant
    dimension.  The returned callable accepts a parameter dict with
    ``(P,)``-shaped arrays and returns a scalar summed log-likelihood.

    Parameters
    ----------
    state_arr : numpy.ndarray, shape (P, n_trials)
        Binary context states (0=safe, 1=dangerous) per participant.
    choices_arr : numpy.ndarray, shape (P, n_trials)
        Binary choices (0=avoid, 1=approach) per participant.
    reward_mag_arr : numpy.ndarray, shape (P, n_trials)
        Reward magnitudes per participant.
    shock_mag_arr : numpy.ndarray, shape (P, n_trials)
        Shock magnitudes per participant.
    trial_mask : numpy.ndarray, shape (P, n_trials)
        Boolean mask; ``True`` for valid trials, ``False`` for padding.
    model_name : {"hgf_2level_patrl", "hgf_3level_patrl"}
        PAT-RL HGF variant.

    Returns
    -------
    logp_fn : callable
        ``dict[str, jnp.ndarray] -> jnp.ndarray``.  Takes a params dict
        where each value has shape ``(P,)`` and returns a scalar.
        Compatible with ``jax.grad`` and BlackJAX NUTS.

    Raises
    ------
    ValueError
        If ``model_name`` is not a recognised PAT-RL variant.

    Notes
    -----
    ``delta_hr`` is NOT in the logp signature.  It is assembled by
    :func:`_build_arrays_single_patrl` for CSV export (Phase 18-05).
    Models B/C/D will extend this signature in Phase 19+.
    """
    base_attrs, scan_fn, belief_idx = _build_session_scanner_patrl(model_name)
    n_trials = state_arr.shape[1]

    _single_logp = _make_single_logp_fn(
        base_attrs, scan_fn, belief_idx, model_name, n_trials
    )

    # Convert to JAX arrays once (closure, not traced args — shapes fixed per call).
    state_jnp = jnp.asarray(state_arr, dtype=jnp.int32)
    choices_jnp = jnp.asarray(choices_arr, dtype=jnp.int32)
    reward_jnp = jnp.asarray(reward_mag_arr, dtype=jnp.float32)
    shock_jnp = jnp.asarray(shock_mag_arr, dtype=jnp.float32)
    mask_jnp = jnp.asarray(trial_mask, dtype=jnp.bool_)

    is_3level = model_name == "hgf_3level_patrl"

    def logp_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute batched log-likelihood summed across all participants.

        Parameters
        ----------
        params : dict[str, jnp.ndarray]
            Parameter dict.  Each value has shape ``(P,)``.  Required keys:
            ``omega_2``, ``beta``
            (plus ``omega_3``, ``kappa``, ``mu3_0`` for 3-level).

        Returns
        -------
        jnp.ndarray
            Scalar summed log-likelihood.
        """
        if is_3level:
            def _call_single(
                omega_2_i: jnp.ndarray,
                omega_3_i: jnp.ndarray,
                kappa_i: jnp.ndarray,
                mu3_0_i: jnp.ndarray,
                beta_i: jnp.ndarray,
                state_i: jnp.ndarray,
                choices_i: jnp.ndarray,
                reward_i: jnp.ndarray,
                shock_i: jnp.ndarray,
                mask_i: jnp.ndarray,
            ) -> Any:
                return _single_logp(  # type: ignore[return-value]
                    {
                        "omega_2": omega_2_i,
                        "omega_3": omega_3_i,
                        "kappa": kappa_i,
                        "mu3_0": mu3_0_i,
                        "beta": beta_i,
                    },
                    state_i, choices_i, reward_i, shock_i, mask_i,
                )

            per_participant = jax.vmap(_call_single)(
                params["omega_2"],
                params["omega_3"],
                params["kappa"],
                params["mu3_0"],
                params["beta"],
                state_jnp,
                choices_jnp,
                reward_jnp,
                shock_jnp,
                mask_jnp,
            )
        else:
            def _call_single_2(  # type: ignore[misc]
                omega_2_i: jnp.ndarray,
                beta_i: jnp.ndarray,
                state_i: jnp.ndarray,
                choices_i: jnp.ndarray,
                reward_i: jnp.ndarray,
                shock_i: jnp.ndarray,
                mask_i: jnp.ndarray,
            ) -> Any:
                return _single_logp(  # type: ignore[return-value]
                    {"omega_2": omega_2_i, "beta": beta_i},
                    state_i, choices_i, reward_i, shock_i, mask_i,
                )

            per_participant = jax.vmap(_call_single_2)(
                params["omega_2"],
                params["beta"],
                state_jnp,
                choices_jnp,
                reward_jnp,
                shock_jnp,
                mask_jnp,
            )

        return jnp.sum(per_participant)

    return logp_fn


# ---------------------------------------------------------------------------
# DataFrame array builder (private)
# ---------------------------------------------------------------------------


def _build_arrays_single_patrl(
    sim_df: pd.DataFrame,
    participants: list[str],
) -> dict[str, np.ndarray]:
    """Convert a PAT-RL sim_df to ``(P, n_trials)`` arrays needed by logp.

    Assembles stacked NumPy arrays for all participants listed in
    ``participants``.  The ``delta_hr`` column is included in the returned
    dict for downstream CSV export (Phase 18-05) but is NOT used by the
    Model A logp.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Must contain: ``participant_id``, ``trial_idx``, ``state``,
        ``choice``, ``reward_mag``, ``shock_mag``, ``delta_hr``,
        ``outcome_time_s``.
    participants : list[str]
        Ordered list of participant identifiers.  Determines row order in
        the stacked output.

    Returns
    -------
    dict[str, numpy.ndarray]
        Keys: ``state``, ``choice``, ``reward_mag``, ``shock_mag``,
        ``delta_hr``, ``trial_mask``.  Each value has shape
        ``(P, n_trials)``.

    Raises
    ------
    KeyError
        If any required column is missing from ``sim_df``.
        Message includes expected columns vs actual columns.
    ValueError
        If participants have inconsistent trial counts or a listed
        participant is absent from ``sim_df``.
    """
    actual_cols = frozenset(sim_df.columns)
    missing_cols = sorted(_REQUIRED_COLUMNS - actual_cols)
    if missing_cols:
        msg = (
            f"sim_df is missing required columns: {missing_cols}. "
            f"Expected: {sorted(_REQUIRED_COLUMNS)}. "
            f"Got: {sorted(actual_cols)}"
        )
        raise KeyError(msg)

    n_trials_per_participant: list[int] = []
    rows: dict[str, list[np.ndarray]] = {
        "state": [],
        "choice": [],
        "reward_mag": [],
        "shock_mag": [],
        "delta_hr": [],
    }

    for pid in participants:
        subset = sim_df[sim_df["participant_id"] == str(pid)]
        if subset.empty:
            msg = (
                f"Participant {pid!r} not found in sim_df. "
                f"Available: {sorted(sim_df['participant_id'].unique())}"
            )
            raise ValueError(msg)
        subset = subset.sort_values("trial_idx")
        n = len(subset)
        n_trials_per_participant.append(n)

        rows["state"].append(subset["state"].to_numpy(dtype=np.int32))
        rows["choice"].append(subset["choice"].to_numpy(dtype=np.int32))
        rows["reward_mag"].append(subset["reward_mag"].to_numpy(dtype=np.float32))
        rows["shock_mag"].append(subset["shock_mag"].to_numpy(dtype=np.float32))
        rows["delta_hr"].append(subset["delta_hr"].to_numpy(dtype=np.float32))

    # Validate uniform trial counts
    unique_counts = set(n_trials_per_participant)
    if len(unique_counts) != 1:
        msg = (
            f"Participants have inconsistent trial counts: {unique_counts}. "
            "All participants must have the same number of trials."
        )
        raise ValueError(msg)

    stacked: dict[str, np.ndarray] = {
        k: np.stack(v, axis=0) for k, v in rows.items()
    }
    n_trials = n_trials_per_participant[0]
    stacked["trial_mask"] = np.ones(
        (len(participants), n_trials), dtype=np.bool_
    )
    return stacked


# ---------------------------------------------------------------------------
# Prior log-probability helper (private)
# ---------------------------------------------------------------------------


def _build_patrl_log_posterior(
    logp_fn: Any,
    config: PATRLConfig,
    model_name: str,
) -> Any:
    """Build a pure-JAX log-posterior combining PAT-RL priors + logp.

    Parameters
    ----------
    logp_fn : callable
        Batched logp from :func:`build_logp_fn_batched_patrl`.
    config : PATRLConfig
        PAT-RL configuration (priors read from ``config.fitting.priors``).
    model_name : str
        PAT-RL model variant.

    Returns
    -------
    logdensity_fn : callable
        ``dict[str, jnp.ndarray] -> scalar``.
    """
    import numpyro.distributions as dist

    priors = config.fitting.priors
    is_3level = model_name == "hgf_3level_patrl"

    # omega_2: Gaussian (unrestricted); in practice always negative
    prior_omega_2 = dist.Normal(
        loc=priors.omega_2.mean,
        scale=priors.omega_2.sd,
    )
    # log_beta: parameterise beta in log-space so NUTS can explore freely.
    # Prior on log_beta ~ N(log(prior_beta.mean), prior_beta.sd / prior_beta.mean)
    # (delta method approximation; keeps prior centred near prior_beta.mean).
    _beta_mean = float(priors.beta.mean)
    _beta_sd = float(priors.beta.sd)
    prior_log_beta = dist.Normal(
        loc=float(np.log(_beta_mean)),
        scale=_beta_sd / _beta_mean,
    )

    if is_3level:
        prior_omega_3 = dist.Normal(
            loc=priors.omega_3.mean,
            scale=priors.omega_3.sd,
        )
        prior_kappa = dist.TruncatedNormal(
            loc=priors.kappa.mean,
            scale=priors.kappa.sd,
            low=priors.kappa.lower,
            high=priors.kappa.upper,
        )
        prior_mu3_0 = dist.Normal(
            loc=priors.mu3_0.mean,
            scale=priors.mu3_0.sd,
        )

    def logdensity_fn(params: dict[str, jnp.ndarray]) -> Any:
        """Compute log-posterior = prior log-prob + log-likelihood.

        Parameters
        ----------
        params : dict[str, jnp.ndarray]
            Parameter dict.  Each value has shape ``(P,)``.

        Returns
        -------
        jnp.ndarray
            Scalar log-posterior.
        """
        omega_2 = params["omega_2"]
        log_beta = params["log_beta"]
        beta = jnp.exp(log_beta)

        prior_lp = jnp.sum(prior_omega_2.log_prob(omega_2))
        prior_lp = prior_lp + jnp.sum(prior_log_beta.log_prob(log_beta))

        if is_3level:
            omega_3 = params["omega_3"]
            kappa = params["kappa"]
            mu3_0 = params["mu3_0"]
            prior_lp = prior_lp + jnp.sum(prior_omega_3.log_prob(omega_3))
            prior_lp = prior_lp + jnp.sum(prior_kappa.log_prob(kappa))
            prior_lp = prior_lp + jnp.sum(prior_mu3_0.log_prob(mu3_0))
            likelihood_lp = logp_fn(
                {
                    "omega_2": omega_2,
                    "omega_3": omega_3,
                    "kappa": kappa,
                    "mu3_0": mu3_0,
                    "beta": beta,
                }
            )
        else:
            likelihood_lp = logp_fn(
                {"omega_2": omega_2, "beta": beta}
            )

        return prior_lp + likelihood_lp

    return logdensity_fn


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_batch_hierarchical_patrl(
    sim_df: pd.DataFrame,
    model_name: str = "hgf_2level_patrl",
    response_model: str = "model_a",
    config: PATRLConfig | None = None,
    n_chains: int | None = None,
    n_tune: int | None = None,
    n_draws: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
) -> az.InferenceData:
    """Fit PAT-RL Model A to a batched cohort via BlackJAX NUTS.

    Reads participant data from ``sim_df``, builds a pure-JAX log-posterior
    (priors from ``config.fitting.priors`` + HGF log-likelihood), and runs
    NUTS with window adaptation via the generic :func:`_run_blackjax_nuts`
    helper from :mod:`prl_hgf.fitting.hierarchical`.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Trial-level data.  Must contain: ``participant_id``, ``trial_idx``,
        ``state``, ``choice``, ``reward_mag``, ``shock_mag``, ``delta_hr``,
        ``outcome_time_s``.
    model_name : {"hgf_2level_patrl", "hgf_3level_patrl"}
        PAT-RL HGF variant.
    response_model : str
        Only ``"model_a"`` is supported in Phase 18.  Models B/C/D are
        deferred to Phase 19+.
    config : PATRLConfig or None
        If ``None``, :func:`~prl_hgf.env.pat_rl_config.load_pat_rl_config`
        is called to load the default ``configs/pat_rl.yaml``.
    n_chains : int or None
        Number of MCMC chains.  Overrides ``config.fitting.n_chains``.
    n_tune : int or None
        Number of warmup steps.  Overrides ``config.fitting.n_tune``.
    n_draws : int or None
        Posterior draws per chain.  Overrides ``config.fitting.n_draws``.
    target_accept : float or None
        NUTS target acceptance rate.  Overrides
        ``config.fitting.target_accept``.
    random_seed : int or None
        Base random seed.  Overrides ``config.fitting.random_seed``.

    Returns
    -------
    arviz.InferenceData
        Posterior with a ``participant`` coordinate on every parameter.
        The ``log_beta`` samples are present; ``beta = exp(log_beta)`` is
        added as a deterministic variable.

    Raises
    ------
    NotImplementedError
        If ``response_model != 'model_a'``.
    ValueError
        If ``model_name`` is not a recognised PAT-RL variant or ``sim_df``
        is missing required columns.

    Notes
    -----
    ``_run_blackjax_nuts`` is imported from
    :mod:`prl_hgf.fitting.hierarchical` unchanged.  The PAT-RL
    ``logdensity_fn`` is closure-based (not the traced-arg sample loop),
    so the XLA cache warm-up optimisation from Phase 17 (Quick-003) does
    not apply here.  This is acceptable for Phase 18 smoke runs; the
    traced-arg extension is planned for Phase 19.
    """
    if response_model != "model_a":
        raise NotImplementedError(
            f"response_model={response_model!r}: only 'model_a' is supported "
            f"in Phase 18.  Models B/C/D are deferred to Phase 19+."
        )

    if model_name not in _PATRL_MODEL_NAMES:
        msg = (
            f"model_name must be one of {_PATRL_MODEL_NAMES}, "
            f"got {model_name!r}"
        )
        raise ValueError(msg)

    if config is None:
        config = load_pat_rl_config()

    # Merge MCMC overrides with config defaults.
    _n_chains = int(n_chains if n_chains is not None else config.fitting.n_chains)
    _n_tune = int(n_tune if n_tune is not None else config.fitting.n_tune)
    _n_draws = int(n_draws if n_draws is not None else config.fitting.n_draws)
    _target_accept = float(
        target_accept if target_accept is not None else config.fitting.target_accept
    )
    _random_seed = int(
        random_seed if random_seed is not None else config.fitting.random_seed
    )

    # ------------------------------------------------------------------
    # 1. Extract participant list and build arrays.
    # ------------------------------------------------------------------
    participants: list[str] = sorted(sim_df["participant_id"].astype(str).unique())
    n_participants = len(participants)

    arrays = _build_arrays_single_patrl(sim_df, participants)

    # ------------------------------------------------------------------
    # 2. Build batched logp.
    # ------------------------------------------------------------------
    logp_fn = build_logp_fn_batched_patrl(
        state_arr=arrays["state"],
        choices_arr=arrays["choice"],
        reward_mag_arr=arrays["reward_mag"],
        shock_mag_arr=arrays["shock_mag"],
        trial_mask=arrays["trial_mask"],
        model_name=model_name,
    )

    # ------------------------------------------------------------------
    # 3. Build pure-JAX log-posterior (prior + likelihood).
    # ------------------------------------------------------------------
    logdensity_fn = _build_patrl_log_posterior(logp_fn, config, model_name)

    # ------------------------------------------------------------------
    # 4. Construct initial parameter positions at prior means.
    # ------------------------------------------------------------------
    priors = config.fitting.priors
    is_3level = model_name == "hgf_3level_patrl"

    _beta_mean = float(priors.beta.mean)
    init_log_beta = float(np.log(max(_beta_mean, 1e-6)))

    initial_position: dict[str, jnp.ndarray] = {
        "omega_2": jnp.full((n_participants,), priors.omega_2.mean),
        "log_beta": jnp.full((n_participants,), init_log_beta),
    }
    if is_3level:
        initial_position["omega_3"] = jnp.full(
            (n_participants,), priors.omega_3.mean
        )
        # Clip kappa init into truncated-normal support.
        kappa_init = float(
            np.clip(priors.kappa.mean, priors.kappa.lower + 1e-6, priors.kappa.upper - 1e-6)
        )
        initial_position["kappa"] = jnp.full((n_participants,), kappa_init)
        initial_position["mu3_0"] = jnp.full(
            (n_participants,), priors.mu3_0.mean
        )

    # ------------------------------------------------------------------
    # 5. Run BlackJAX NUTS (reuse generic helper from hierarchical.py).
    # ------------------------------------------------------------------
    rng_key = jax.random.PRNGKey(_random_seed)

    positions, sample_stats, n_chains_actual, _adapted = _run_blackjax_nuts(
        logdensity_fn=logdensity_fn,
        initial_position=initial_position,
        rng_key=rng_key,
        n_tune=_n_tune,
        n_draws=_n_draws,
        n_chains=_n_chains,
        target_accept=_target_accept,
        # batched_logp_fn=None → fallback to closure-based legacy path
        # (traced-arg XLA cache reuse deferred to Phase 19).
        batched_logp_fn=None,
    )

    # ------------------------------------------------------------------
    # 6. Convert to ArviZ InferenceData.
    # ------------------------------------------------------------------
    # Build var_names list including derived beta.
    _sampled_keys = list(initial_position.keys())
    var_names = _sampled_keys + ["beta"]

    # _samples_to_idata expects participant_groups and participant_sessions;
    # supply defaults since PAT-RL sim_df may not have group/session columns.
    if "group" in sim_df.columns:
        pgroup_map = (
            sim_df[["participant_id", "group"]]
            .drop_duplicates()
            .set_index("participant_id")["group"]
            .to_dict()
        )
        participant_groups = [str(pgroup_map.get(p, "unknown")) for p in participants]
    else:
        participant_groups = ["unknown"] * n_participants

    if "session" in sim_df.columns:
        psession_map = (
            sim_df[["participant_id", "session"]]
            .drop_duplicates()
            .set_index("participant_id")["session"]
            .to_dict()
        )
        participant_sessions = [
            str(psession_map.get(p, "session_1")) for p in participants
        ]
    else:
        participant_sessions = ["session_1"] * n_participants

    idata = _samples_to_idata(
        positions=positions,
        sample_stats=sample_stats,
        var_names=var_names,
        participant_ids=participants,
        participant_groups=participant_groups,
        participant_sessions=participant_sessions,
        model_name=model_name,
        coord_name="participant_id",
    )

    return idata
