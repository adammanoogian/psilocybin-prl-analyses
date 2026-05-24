"""VB-Laplace fit path for pick_best_cue 3-cue PRL.

Analogous to ``fit_vb_laplace_patrl.py`` but for the standard 3-cue
partial-feedback reversal-learning task.  Uses the batched logp from
``hierarchical.py`` (identical math to the NUTS path) with the Mode A
log-posterior builder ``_build_log_posterior``.

Runs LBFGS MAP optimisation, computes the autodiff Hessian at the mode,
PD-regularizes via eigenvalue clipping, inverts to posterior covariance,
and packages the result as ``az.InferenceData`` via
``build_idata_from_laplace``.

Per-participant **log model evidence (LME)** is computed at the mode
using the Laplace free-energy approximation (same quantity as TAPAS
``est.optim.LME``).  This feeds directly into random-effects BMS via
:func:`compare_models_laplace`.

Reference: TAPAS ``tapas_fitModel`` with ``tapas_quasinewton_optim``
(conceptual analog).
"""

from __future__ import annotations

import logging
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree
from jaxopt import LBFGS

from prl_hgf.fitting.hierarchical import (
    _KAPPA_FIXED,
    _build_arrays_single,
    _build_log_posterior,
    build_logp_fn_batched,
)
from prl_hgf.fitting.laplace_idata import (
    _PARAM_ORDER_PRL_2LEVEL,
    _PARAM_ORDER_PRL_3LEVEL,
    build_idata_from_laplace,
)
from prl_hgf.fitting.priors import HGFPriorSpec

logger = logging.getLogger(__name__)

_MODEL_NAMES = ("hgf_2level", "hgf_3level")


def _regularize_to_pd(
    H: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip negative/tiny eigenvalues to eps; return (H_pd, diagnostics)."""
    w, V = np.linalg.eigh(H)
    n_clipped = int(np.sum(w < eps))
    w_clip = np.maximum(w, eps)
    H_pd = V @ np.diag(w_clip) @ V.T
    diag: dict[str, float] = {
        "hessian_min_eigval": float(np.min(w)),
        "hessian_max_eigval": float(np.max(w)),
        "n_eigenvalues_clipped": float(n_clipped),
        "ridge_added": float(max(0.0, eps - float(np.min(w)))),
    }
    return H_pd, diag


def _compute_per_participant_lme(
    H_pd: np.ndarray,
    P: int,
    K: int,
    per_participant_logpost: np.ndarray,
) -> np.ndarray:
    """Compute per-participant log model evidence from the Laplace approximation.

    Parameters
    ----------
    H_pd : np.ndarray, shape (P*K, P*K)
        PD-regularized Hessian of the negative log-posterior.
    P : int
        Number of participants.
    K : int
        Number of free parameters per participant.
    per_participant_logpost : np.ndarray, shape (P,)
        Per-participant log-posterior at the MAP.

    Returns
    -------
    np.ndarray, shape (P,)
        Per-participant LME (Laplace free energy).
    """
    lme = np.empty(P)
    for i in range(P):
        indices = [i + k * P for k in range(K)]
        H_i = H_pd[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(H_i)
        if sign <= 0:
            logdet = K * np.log(1e-8)
        lme[i] = (
            per_participant_logpost[i]
            + (K / 2.0) * np.log(2 * np.pi)
            - 0.5 * logdet
        )
    return lme


def fit_vb_laplace_prl(
    sim_df: pd.DataFrame,
    model_name: str = "hgf_3level",
    prior_spec: HGFPriorSpec | None = None,
    n_pseudo_draws: int = 1000,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_restarts: int = 1,
    random_seed: int = 0,
) -> az.InferenceData:
    """Fit pick_best_cue PRL via Laplace approximation at the MAP.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Trial-level DataFrame with columns ``participant_id``, ``group``,
        ``session``, ``cue_chosen``, ``reward``.
    model_name : str, default 'hgf_3level'
        ``'hgf_2level'`` or ``'hgf_3level'``.
    prior_spec : HGFPriorSpec or None, optional
        Prior specification.  Defaults derived from ``model_name``.
    n_pseudo_draws : int, default 1000
        Number of pseudo-samples from N(mode, Sigma).
    max_iter : int, default 200
        LBFGS maximum iterations.
    tol : float, default 1e-5
        LBFGS convergence tolerance.
    n_restarts : int, default 1
        MAP restarts from perturbed initial positions.
    random_seed : int, default 0
        RNG seed for pseudo-samples and restart perturbation.

    Returns
    -------
    az.InferenceData
        Posterior shaped ``(chain=1, draw=n_pseudo_draws, participant_id=P)``
        with ``sample_stats`` diagnostics including ``lme`` (per-participant
        log model evidence, shape ``(P,)``).
    """
    if model_name not in _MODEL_NAMES:
        msg = (
            f"model_name must be one of {_MODEL_NAMES}, got {model_name!r}"
        )
        raise ValueError(msg)

    is_3level = model_name == "hgf_3level"

    if prior_spec is None:
        prior_spec = (
            HGFPriorSpec.default_3level()
            if is_3level
            else HGFPriorSpec.default_2level()
        )

    # ------------------------------------------------------------------
    # 1. Build arrays from sim_df
    # ------------------------------------------------------------------
    group_keys = ["participant_id", "group", "session"]
    groups = list(sim_df.groupby(group_keys, sort=False))

    input_data_list: list[np.ndarray] = []
    observed_list: list[np.ndarray] = []
    choices_list: list[np.ndarray] = []
    participant_ids: list[str] = []

    for (pid, grp, sess), subset in groups:
        if "trial" in subset.columns:
            subset = subset.sort_values("trial")
        inp, obs, ch = _build_arrays_single(subset)
        input_data_list.append(inp)
        observed_list.append(obs)
        choices_list.append(ch)
        participant_ids.append(f"{pid}_{grp}_{sess}")

    trial_counts = [arr.shape[0] for arr in input_data_list]
    if len(set(trial_counts)) != 1:
        msg = (
            f"All participant-sessions must have the same trial count. "
            f"Got: {sorted(set(trial_counts))}"
        )
        raise ValueError(msg)

    n_trials = trial_counts[0]
    P = len(input_data_list)

    input_data_arr = jnp.array(np.stack(input_data_list), dtype=jnp.float32)
    observed_arr = jnp.array(np.stack(observed_list), dtype=jnp.int32)
    choices_arr = jnp.array(np.stack(choices_list), dtype=jnp.int32)
    trial_mask = jnp.ones((P, n_trials), dtype=jnp.float32)

    logger.info(
        "Built arrays: P=%d, n_trials=%d, model=%s",
        P,
        n_trials,
        model_name,
    )

    # ------------------------------------------------------------------
    # 2. Build logp functions
    # ------------------------------------------------------------------
    batched_logp_fn, _n_params = build_logp_fn_batched(model_name, n_trials)

    # Log-posterior (summed across participants) for MAP optimisation
    log_posterior_fn = _build_log_posterior(
        batched_logp_fn,
        input_data_arr,
        observed_arr,
        choices_arr,
        trial_mask,
        P,
        model_name=model_name,
        prior_spec=prior_spec,
    )

    # Per-participant log-likelihood (NOT summed) for LME computation.
    # Rebuild a vmapped-but-not-summed version from the same closure.
    _per_participant_logp_fn, _ = build_logp_fn_batched(model_name, n_trials)

    # ------------------------------------------------------------------
    # 3. Determine parameter order + initial position
    # ------------------------------------------------------------------
    if is_3level:
        param_order = _PARAM_ORDER_PRL_3LEVEL
    else:
        param_order = _PARAM_ORDER_PRL_2LEVEL

    K = len(param_order)

    init_position: dict[str, jnp.ndarray] = {
        "omega_2": jnp.full((P,), prior_spec.omega_2.loc),
        "log_beta": jnp.full((P,), prior_spec.log_beta.loc),
        "zeta": jnp.full((P,), prior_spec.zeta.loc),
    }
    if is_3level:
        init_position["omega_3"] = jnp.full(
            (P,), prior_spec.omega_3.loc,  # type: ignore[union-attr]
        )

    # Ensure insertion order matches param_order
    init_position = {k: init_position[k] for k in param_order}

    # ------------------------------------------------------------------
    # 4. LBFGS MAP optimization (with optional restarts)
    # ------------------------------------------------------------------
    def neg_log_posterior(params_dict: dict[str, jnp.ndarray]) -> Any:
        return -log_posterior_fn(params_dict)

    solver = LBFGS(fun=neg_log_posterior, maxiter=max_iter, tol=tol, jit=True)

    rng = np.random.default_rng(random_seed)
    best_logp = -np.inf
    best_mode: dict[str, jnp.ndarray] | None = None
    best_state: Any = None

    for restart_i in range(n_restarts):
        if restart_i == 0:
            start = init_position
        else:
            start = {
                k: v + jnp.array(rng.normal(0, 0.1, size=v.shape))
                for k, v in init_position.items()
            }

        res = solver.run(start)
        logp_val = float(-res.state.value)

        logger.info(
            "Restart %d/%d: logp=%.2f, converged=%s",
            restart_i + 1,
            n_restarts,
            logp_val,
            not bool(res.state.error),
        )

        if np.isfinite(logp_val) and logp_val > best_logp:
            best_logp = logp_val
            best_mode = {k: jnp.array(v) for k, v in res.params.items()}
            best_state = res.state

    if best_mode is None:
        msg = "All LBFGS restarts produced non-finite log-posterior."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # 5. Hessian of negative log-posterior at mode
    # ------------------------------------------------------------------
    ordered_mode = {k: best_mode[k] for k in param_order}
    flat_mode, unravel = ravel_pytree(ordered_mode)

    H = jax.hessian(
        lambda f: -log_posterior_fn(unravel(f)),
    )(flat_mode)
    H_np = np.asarray(H, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. PD regularization + covariance inversion
    # ------------------------------------------------------------------
    H_pd, pd_diag = _regularize_to_pd(H_np, eps=1e-8)
    cov = np.linalg.inv(H_pd)

    # ------------------------------------------------------------------
    # 7. Per-participant log model evidence (LME)
    # ------------------------------------------------------------------
    # Evaluate per-participant log-posterior at the MAP mode.
    # The log-posterior = log-likelihood + log-prior, both factorise
    # across participants for IID Mode A priors.
    prior_omega_2 = prior_spec.omega_2.to_numpyro_dist()
    prior_log_beta = prior_spec.log_beta.to_numpyro_dist()
    prior_zeta = prior_spec.zeta.to_numpyro_dist()

    beta_at_mode = jnp.exp(best_mode["log_beta"])

    # Per-participant log-likelihood at mode
    if is_3level:
        prior_omega_3 = prior_spec.omega_3.to_numpyro_dist()  # type: ignore[union-attr]
        kappa_arr = jnp.full_like(best_mode["omega_2"], _KAPPA_FIXED)
        per_p_loglik = _eval_per_participant_loglik(
            _per_participant_logp_fn,
            best_mode["omega_2"],
            best_mode["omega_3"],
            kappa_arr,
            beta_at_mode,
            best_mode["zeta"],
            input_data_arr,
            observed_arr,
            choices_arr,
            trial_mask,
            is_3level=True,
        )
        per_p_prior = (
            np.asarray(prior_omega_2.log_prob(best_mode["omega_2"]))
            + np.asarray(prior_log_beta.log_prob(best_mode["log_beta"]))
            + np.asarray(prior_zeta.log_prob(best_mode["zeta"]))
            + np.asarray(prior_omega_3.log_prob(best_mode["omega_3"]))
        )
    else:
        per_p_loglik = _eval_per_participant_loglik(
            _per_participant_logp_fn,
            best_mode["omega_2"],
            beta_at_mode,
            best_mode["zeta"],
            input_data_arr,
            observed_arr,
            choices_arr,
            trial_mask,
            is_3level=False,
        )
        per_p_prior = (
            np.asarray(prior_omega_2.log_prob(best_mode["omega_2"]))
            + np.asarray(prior_log_beta.log_prob(best_mode["log_beta"]))
            + np.asarray(prior_zeta.log_prob(best_mode["zeta"]))
        )

    per_p_logpost = np.asarray(per_p_loglik) + np.asarray(per_p_prior)
    per_p_lme = _compute_per_participant_lme(H_pd, P, K, per_p_logpost)

    # ------------------------------------------------------------------
    # 8. Build diagnostics dict
    # ------------------------------------------------------------------
    diagnostics: dict[str, float] = {
        "converged": float(not bool(best_state.error)),
        "n_iterations": float(best_state.iter_num),
        "logp_at_mode": best_logp,
        **pd_diag,
    }
    logger.info("Laplace diagnostics: %s", diagnostics)

    # ------------------------------------------------------------------
    # 9. Package as InferenceData
    # ------------------------------------------------------------------
    mode_native: dict[str, np.ndarray] = {
        k: np.asarray(best_mode[k]) for k in param_order
    }

    idata = build_idata_from_laplace(
        mode=mode_native,
        cov=cov,
        param_names=param_order,
        participant_ids=participant_ids,
        n_pseudo_draws=n_pseudo_draws,
        rng_key=random_seed,
        diagnostics=diagnostics,
    )

    # Attach per-participant LME as a custom attribute
    idata.attrs["lme"] = per_p_lme.tolist()
    idata.attrs["participant_ids"] = participant_ids
    idata.attrs["model_name"] = model_name

    return idata


def _eval_per_participant_loglik(
    batched_logp_fn: Any,
    *args: jnp.ndarray,
    is_3level: bool,
) -> np.ndarray:
    """Evaluate per-participant log-likelihood at the mode (not summed).

    ``build_logp_fn_batched`` internally vmaps the single-participant
    logp and sums.  We call it once to get the total, but to get
    per-participant values we leverage the linearity: call the batched
    function once per participant by slicing the arrays.
    """
    if is_3level:
        omega_2, omega_3, kappa, beta, zeta, inp, obs, ch, mask = args
        P = omega_2.shape[0]
        result = np.empty(P)
        for i in range(P):
            val = batched_logp_fn(
                omega_2[i : i + 1],
                omega_3[i : i + 1],
                kappa[i : i + 1],
                beta[i : i + 1],
                zeta[i : i + 1],
                inp[i : i + 1],
                obs[i : i + 1],
                ch[i : i + 1],
                mask[i : i + 1],
            )
            result[i] = float(val)
    else:
        omega_2, beta, zeta, inp, obs, ch, mask = args
        P = omega_2.shape[0]
        result = np.empty(P)
        for i in range(P):
            val = batched_logp_fn(
                omega_2[i : i + 1],
                beta[i : i + 1],
                zeta[i : i + 1],
                inp[i : i + 1],
                obs[i : i + 1],
                ch[i : i + 1],
                mask[i : i + 1],
            )
            result[i] = float(val)
    return result


def compare_models_laplace(
    idata_dict: dict[str, az.InferenceData],
) -> dict[str, Any]:
    """Compare models fitted with ``fit_vb_laplace_prl`` using LME + BMS.

    Uses per-participant log model evidence (LME) from the Laplace free
    energy — the same quantity as TAPAS ``est.optim.LME``.  Feeds the
    LME matrix to :func:`~prl_hgf.analysis.bms.run_group_bms` for
    random-effects Bayesian Model Selection (Rigoux et al. 2014).

    Parameters
    ----------
    idata_dict : dict[str, az.InferenceData]
        Mapping from model name to fitted InferenceData (output of
        :func:`fit_vb_laplace_prl`).  Each must have ``lme`` and
        ``model_name`` attributes.

    Returns
    -------
    dict
        BMS results with keys ``model_names``, ``xp`` (exceedance
        probability), ``pxp`` (protected exceedance probability),
        ``exp_r`` (expected model frequencies), ``bor`` (Bayesian
        Omnibus Risk), ``lme_matrix``, ``lme_summary``.
    """
    from prl_hgf.analysis.bms import run_group_bms

    model_names = list(idata_dict.keys())
    lme_arrays = []
    for name in model_names:
        idata = idata_dict[name]
        lme = np.array(idata.attrs["lme"])
        lme_arrays.append(lme)

    lme_matrix = np.column_stack(lme_arrays)

    bms_result = run_group_bms(lme_matrix, model_names)

    lme_summary = pd.DataFrame(
        {
            "model": model_names,
            "mean_lme": [float(np.mean(a)) for a in lme_arrays],
            "sum_lme": [float(np.sum(a)) for a in lme_arrays],
        }
    )

    return {
        **bms_result,
        "lme_matrix": lme_matrix,
        "lme_summary": lme_summary,
    }


def idata_to_fit_df(
    idata: az.InferenceData,
    param_names: list[str],
) -> pd.DataFrame:
    """Convert Laplace InferenceData to long-form fit DataFrame.

    Produces the same format as :func:`~prl_hgf.fitting.batch.fit_batch`
    so that :func:`~prl_hgf.analysis.recovery.build_recovery_df` works
    directly.

    Parameters
    ----------
    idata : az.InferenceData
        Output of :func:`fit_vb_laplace_prl`.
    param_names : list[str]
        Parameters to extract (e.g. ``["omega_2", "beta", "zeta"]``).

    Returns
    -------
    pd.DataFrame
        Long-form with columns ``participant_id``, ``group``, ``session``,
        ``parameter``, ``mean``, ``sd``, ``flagged``.
    """
    posterior = idata.posterior
    pids = list(posterior.participant_id.values)
    rows: list[dict] = []
    for idx, pid_key in enumerate(pids):
        parts = str(pid_key).rsplit("_", 2)
        pid, grp, sess = parts[0], parts[1], parts[2]
        for param in param_names:
            vals = posterior[param].values[:, :, idx]
            rows.append(
                {
                    "participant_id": pid,
                    "group": grp,
                    "session": sess,
                    "parameter": param,
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                    "flagged": False,
                }
            )
    return pd.DataFrame(rows)
