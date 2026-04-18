"""VB-Laplace fit path for PAT-RL (Phase 19).

Runs quasi-Newton MAP optimization via ``jaxopt.LBFGS`` on the imported
``_build_patrl_log_posterior`` (from ``hierarchical_patrl``, reused
unchanged — parallel-stack invariant), computes the autodiff Hessian at
the mode via ``jax.hessian`` on the ``ravel_pytree``-flattened parameter
vector, PD-regularizes indefinite Hessians via eigenvalue clipping,
inverts to posterior covariance, and packages the result as an
``az.InferenceData`` via ``build_idata_from_laplace`` from
``laplace_idata.py``.

The Hessian is block-diagonal across participants (priors IID,
likelihood vmap-summed — see research Q3); Phase 19 computes the dense
representation for simplicity. Block-structured ``jax.vmap(jax.hessian(
per_subject_logp))`` is a Phase 20+ optimization for cohorts >200.

Reference: laplax (arxiv 2507.17013), matlab tapas_fitModel /
tapas_riddersmatrix (conceptual analog).
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

from prl_hgf.env.pat_rl_config import PATRLConfig, load_pat_rl_config
from prl_hgf.fitting.hierarchical_patrl import (
    _build_arrays_single_patrl,
    _build_patrl_log_posterior,
    build_logp_fn_batched_patrl,
)
from prl_hgf.fitting.laplace_idata import (
    _PARAM_ORDER_2LEVEL,
    _PARAM_ORDER_3LEVEL,
    build_idata_from_laplace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-private PD regularization helper
# ---------------------------------------------------------------------------


def _regularize_to_pd(
    H: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip negative/tiny eigenvalues to eps; return (H_pd, diagnostics).

    Uses eigendecomposition (``numpy.linalg.eigh``) to clip the Hessian to
    be positive-definite at the MAP.  Cleaner than the ridge-add loop:
    deterministic, logs exactly how many eigenvalues were clipped.

    Returned diag dict uses the canonical Phase 19 key set:
    ``hessian_min_eigval``, ``hessian_max_eigval``,
    ``n_eigenvalues_clipped``, ``ridge_added``.  Plan 19-02 consumes
    these names directly.

    Parameters
    ----------
    H : np.ndarray
        Square symmetric matrix (Hessian of negative log-posterior).
        Shape ``(D, D)``.
    eps : float, default 1e-8
        Minimum eigenvalue threshold.  Eigenvalues below this value are
        clipped to ``eps``.

    Returns
    -------
    H_pd : np.ndarray
        PD-regularized Hessian, same shape as ``H``.
    diag : dict[str, float]
        Canonical diagnostic keys: ``hessian_min_eigval``,
        ``hessian_max_eigval``, ``n_eigenvalues_clipped``, ``ridge_added``.
    """
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_vb_laplace_patrl(
    sim_df: pd.DataFrame,
    model_name: str,
    response_model: str = "model_a",
    config: PATRLConfig | None = None,
    n_pseudo_draws: int = 1000,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_restarts: int = 1,
    random_seed: int = 0,
) -> az.InferenceData:
    """Fit PAT-RL via Laplace approximation at the MAP.

    Runs quasi-Newton MAP optimization (``jaxopt.LBFGS``) on the
    ``_build_patrl_log_posterior`` closure from ``hierarchical_patrl.py``
    (imported unchanged — parallel-stack invariant).  Computes the autodiff
    Hessian of the negated log-posterior at the mode, PD-regularizes via
    eigenvalue clipping, inverts to covariance, and packages the result as
    an ``az.InferenceData`` via ``build_idata_from_laplace``.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Simulated or observed data matching the hierarchical_patrl input
        contract (see ``_build_arrays_single_patrl`` docstring for
        required columns).
    model_name : str
        ``'hgf_2level_patrl'`` or ``'hgf_3level_patrl'``.
    response_model : str, default 'model_a'
        Only ``'model_a'`` is supported in Phase 19. Other values raise
        NotImplementedError (Phase 20+ scope).
    config : PATRLConfig, optional
        Analysis config. Defaults to ``load_pat_rl_config()``.
    n_pseudo_draws : int, default 1000
        Number of pseudo-samples from N(mode, Sigma). See
        ``build_idata_from_laplace`` for note on why this does not affect
        Laplace accuracy.
    max_iter : int, default 200
        jaxopt.LBFGS maximum iterations.
    tol : float, default 1e-5
        jaxopt.LBFGS convergence tolerance.
    n_restarts : int, default 1
        Number of MAP optimization restarts from perturbed initial
        positions. If >1, the restart with highest log-posterior at the
        mode is kept. Default 1 = single run from prior means.
    random_seed : int, default 0
        Seed for (a) MultivariateNormal pseudo-samples, (b) any
        restart-perturbation RNG.

    Returns
    -------
    az.InferenceData
        Posterior group with vars matching NUTS output (``omega_2,
        log_beta, beta`` for 2-level; +``omega_3, kappa, mu3_0`` for
        3-level) shaped ``(chain=1, draw=n_pseudo_draws,
        participant_id=P)``. ``sample_stats`` group includes diagnostic
        scalars (converged, n_iterations, logp_at_mode,
        hessian_min_eigval, ridge_added).

    Raises
    ------
    NotImplementedError
        If ``response_model != 'model_a'``.
    ValueError
        If ``model_name`` is not a recognised PAT-RL variant.
    RuntimeError
        If all LBFGS restarts fail to produce a finite log-posterior, or
        if Hessian PD-regularization fails.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if response_model != "model_a":
        raise NotImplementedError(
            f"response_model={response_model!r}: only 'model_a' is supported "
            f"in Phase 19. Models B/C/D are Phase 20+ scope."
        )
    if model_name not in ("hgf_2level_patrl", "hgf_3level_patrl"):
        raise ValueError(
            f"model_name must be one of {{'hgf_2level_patrl', 'hgf_3level_patrl'}}, "
            f"got {model_name!r}"
        )
    if config is None:
        config = load_pat_rl_config()

    # ------------------------------------------------------------------
    # 2. Build arrays + logp factory
    # ------------------------------------------------------------------
    # Derive participant list (sorted, str-cast) for both array builder and
    # the final InferenceData coord. Matches fit_batch_hierarchical_patrl:758.
    participants: list[str] = sorted(
        sim_df["participant_id"].astype(str).unique().tolist()
    )
    arrays = _build_arrays_single_patrl(sim_df, participants)
    logp_fn = build_logp_fn_batched_patrl(
        state_arr=arrays["state"],
        choices_arr=arrays["choice"],
        reward_mag_arr=arrays["reward_mag"],
        shock_mag_arr=arrays["shock_mag"],
        trial_mask=arrays["trial_mask"],
        model_name=model_name,
    )
    # log_posterior_fn returns a POSITIVE log-posterior (NOT negated).
    # Verified at hierarchical_patrl.py:650 — returns prior_lp + likelihood_lp.
    log_posterior_fn = _build_patrl_log_posterior(logp_fn, config, model_name)

    # ------------------------------------------------------------------
    # 3. Determine parameter order + initial position at prior means
    # ------------------------------------------------------------------
    if model_name == "hgf_2level_patrl":
        param_order = _PARAM_ORDER_2LEVEL
    else:
        param_order = _PARAM_ORDER_3LEVEL

    P = len(participants)
    prior = config.fitting.priors

    init_arrays: dict[str, jnp.ndarray] = {
        "omega_2": jnp.full((P,), prior.omega_2.mean),
        "log_beta": jnp.full((P,), jnp.log(prior.beta.mean)),  # log-space
    }
    if model_name == "hgf_3level_patrl":
        init_arrays["omega_3"] = jnp.full((P,), prior.omega_3.mean)
        # Clip kappa init into (lower+1e-6, upper-1e-6) — research Q2
        # TODO (OQ2 from research): if cluster VBL-06 shows kappa MAP landing
        # near lower/upper truncation bound (say within 5% of either), switch
        # to logit-reparametrization for kappa and apply Jacobian back-transform.
        # Phase 19 ships native-space kappa matching the existing TruncatedNormal
        # prior in _build_patrl_log_posterior.
        kappa_init = jnp.clip(
            prior.kappa.mean,
            prior.kappa.lower + 1e-6,
            prior.kappa.upper - 1e-6,
        )
        init_arrays["kappa"] = jnp.full((P,), kappa_init)
        init_arrays["mu3_0"] = jnp.full((P,), prior.mu3_0.mean)

    # Build init_position dict preserving param_order (dict-insertion order
    # MUST match ravel_pytree expectation — research Q10).
    init_position: dict[str, jnp.ndarray] = {k: init_arrays[k] for k in param_order}

    # ------------------------------------------------------------------
    # 4. Objective: negate log-posterior
    # ------------------------------------------------------------------
    def neg_log_posterior(
        params_dict: dict[str, jnp.ndarray],
    ) -> Any:
        # L-BFGS-B minimizes; we have a log-posterior maximization, so negate.
        # This is the SOLE negation site. log_posterior_fn returns a positive
        # log-posterior — see Step 2 spec.
        return -log_posterior_fn(params_dict)

    # ------------------------------------------------------------------
    # 5. Run jaxopt.LBFGS (with jit=True → jit=False fallback)
    # ------------------------------------------------------------------
    # Initialize ALL loop-output variables up front (ensures they exist even
    # if every restart fails to produce a finite log-posterior).
    best_mode_params: dict[str, jnp.ndarray] | None = None
    best_logp: float = float(-jnp.inf)
    best_state_info: dict[str, Any] = {}

    rng = np.random.default_rng(random_seed)

    for restart_idx in range(n_restarts):
        if restart_idx == 0:
            start = init_position
        else:
            # Perturb init by N(0, 0.5) in each parameter
            perturb = {
                k: jnp.asarray(rng.normal(0.0, 0.5, size=v.shape))
                for k, v in init_position.items()
            }
            start = {k: init_position[k] + perturb[k] for k in param_order}

        # jit=True is the jaxopt default; set explicitly for traceability.
        solver = LBFGS(
            fun=neg_log_posterior,
            maxiter=max_iter,
            tol=tol,
            jit=True,
        )
        try:
            res = solver.run(start)
        except Exception as exc:
            # If jit=True causes a tracing error (e.g. non-traceable objects in
            # config closed over by log_posterior_fn), retry with jit=False.
            logger.warning(
                "jaxopt.LBFGS jit=True tracing failed (%s); retrying with jit=False.",
                exc,
            )
            solver_nojit = LBFGS(
                fun=neg_log_posterior,
                maxiter=max_iter,
                tol=tol,
                jit=False,
            )
            res = solver_nojit.run(start)

        mode_params = res.params  # dict[str, jnp.ndarray]
        logp_at_mode = float(-res.state.value)

        if logp_at_mode > best_logp:
            best_logp = logp_at_mode
            best_mode_params = mode_params
            best_state_info = {
                "n_iterations": int(res.state.iter_num),
                "converged": bool(res.state.error < tol),
            }

    if best_mode_params is None:
        raise RuntimeError(
            f"All {n_restarts} LBFGS restarts failed to produce a finite "
            f"log-posterior; check priors and data. best_logp={best_logp}."
        )

    # Re-order best_mode_params to param_order so that ravel_pytree produces
    # a flat vector whose block layout matches the build_idata_from_laplace
    # convention (columns [i*P:(i+1)*P] belong to param_names[i]).
    # jaxopt.LBFGS returns params in JAX's sorted-key order; we must re-order
    # before flattening for the Hessian.
    best_mode_params_ordered: dict[str, jnp.ndarray] = {
        k: best_mode_params[k] for k in param_order
    }

    # ------------------------------------------------------------------
    # 6. Compute Hessian via flatten
    # ------------------------------------------------------------------
    flat_mode, unravel = ravel_pytree(best_mode_params_ordered)

    # TODO (Phase 20+): for cohorts P>200, switch to block-structured
    # jax.vmap(jax.hessian(per_subject_logp)) to avoid the (P*K)*(P*K)
    # dense intermediate. The Hessian is mathematically block-diagonal
    # across participants (priors IID, likelihood vmap-summed); current
    # implementation computes the full dense matrix for simplicity.

    # Hessian of NEGATIVE log-posterior is PSD at MAP (explicit negation
    # at the hessian callsite — do NOT reuse neg_log_posterior here to
    # avoid any doubt about sign conventions).
    H = jax.hessian(lambda f: -log_posterior_fn(unravel(f)))(flat_mode)
    H_np = np.asarray(H, dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. PD-regularize via eigh-clip
    # ------------------------------------------------------------------
    H_pd, pd_diag = _regularize_to_pd(H_np)
    if pd_diag["n_eigenvalues_clipped"] > 0:
        logger.warning(
            "Laplace Hessian had %d non-PD eigenvalues; regularized by "
            "clipping at 1e-8. ridge_added=%s.",
            int(pd_diag["n_eigenvalues_clipped"]),
            pd_diag["ridge_added"],
        )

    # ------------------------------------------------------------------
    # 8. Sanity check: Cholesky must succeed on H_pd
    # ------------------------------------------------------------------
    try:
        np.linalg.cholesky(H_pd)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Laplace Hessian regularization failed: Cholesky raised "
            f"{exc}. Diagnostics: {pd_diag}"
        ) from exc

    # ------------------------------------------------------------------
    # 9. Invert to covariance
    # ------------------------------------------------------------------
    cov = np.linalg.inv(H_pd)
    min_cov_eigval = float(np.min(np.linalg.eigvalsh(cov)))
    if min_cov_eigval <= 0:
        raise RuntimeError(
            f"Covariance matrix is not positive definite after Hessian "
            f"regularization. Expected min eigenvalue > 0, got "
            f"{min_cov_eigval:.6g}. Diagnostics: {pd_diag}"
        )

    # ------------------------------------------------------------------
    # 10. Unflatten mode for build_idata_from_laplace
    # ------------------------------------------------------------------
    # Extract mode in param_order (already ordered in best_mode_params_ordered).
    mode_native: dict[str, np.ndarray] = {
        k: np.asarray(best_mode_params_ordered[k]) for k in param_order
    }

    # ------------------------------------------------------------------
    # 11. Package diagnostics dict for sample_stats surfacing
    # ------------------------------------------------------------------
    # Keys already match the canonical set from _regularize_to_pd (no remap
    # needed — build_idata_from_laplace consumes these names directly).
    diagnostics: dict[str, Any] = {
        "converged": float(best_state_info["converged"]),
        "n_iterations": float(best_state_info["n_iterations"]),
        "logp_at_mode": float(best_logp),
        "hessian_min_eigval": float(pd_diag["hessian_min_eigval"]),
        "hessian_max_eigval": float(pd_diag["hessian_max_eigval"]),
        "n_eigenvalues_clipped": float(pd_diag["n_eigenvalues_clipped"]),
        "ridge_added": float(pd_diag["ridge_added"]),
    }

    # ------------------------------------------------------------------
    # 12. Call factory — participants is the sorted list from Step 2
    # ------------------------------------------------------------------
    idata = build_idata_from_laplace(
        mode=mode_native,
        cov=cov,
        param_names=param_order,
        participant_ids=participants,
        n_pseudo_draws=n_pseudo_draws,
        rng_key=random_seed,
        diagnostics=diagnostics,
    )
    return idata
