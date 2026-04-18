"""Package a Laplace approximation ``(mode, Σ)`` into an ``az.InferenceData``.

Draws ``n_pseudo_draws`` samples from ``N(mode, Σ)`` in the parameter space
used by the optimizer (log_beta for beta; native for omega_2, omega_3,
kappa, mu3_0), then adds ``beta = exp(log_beta)`` as a deterministic
transform.

Emits dim name ``'participant_id'`` natively — matches what
``src/prl_hgf/analysis/export_trajectories.py`` reads.  The NUTS path's
``_samples_to_idata`` uses the older dim name ``'participant'``; this
module's choice sidesteps that latent producer/consumer mismatch (OQ1
in Phase 19 research, to be hotfixed in a separate Phase 18 follow-up).

The flat covariance layout follows the ``jax.flatten_util.ravel_pytree``
convention for a dict keyed by ``param_names`` in insertion order with
values of shape ``(P,)``: variables are concatenated contiguously, so
columns ``[i*P : (i+1)*P]`` belong to ``param_names[i]``.  This allows
``fit_vb_laplace_patrl`` (Plan 19-03) to reuse the same unravel without
an extra reshape step.
"""

from __future__ import annotations

from typing import Any, cast

import arviz as az
import numpy as np

__all__ = ["build_idata_from_laplace"]

# ---------------------------------------------------------------------------
# Canonical parameter orders (module-level constants)
# ---------------------------------------------------------------------------

_PARAM_ORDER_2LEVEL: tuple[str, ...] = ("omega_2", "log_beta")
_PARAM_ORDER_3LEVEL: tuple[str, ...] = (
    "omega_2",
    "log_beta",
    "omega_3",
    "kappa",
    "mu3_0",
)

_VALID_PARAM_ORDERS: tuple[tuple[str, ...], ...] = (
    _PARAM_ORDER_2LEVEL,
    _PARAM_ORDER_3LEVEL,
)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_idata_from_laplace(
    mode: dict[str, np.ndarray],
    cov: np.ndarray,
    param_names: tuple[str, ...],
    participant_ids: list[str],
    n_pseudo_draws: int = 1000,
    rng_key: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> az.InferenceData:
    """Build an ArviZ InferenceData from a Laplace approximation.

    Parameters
    ----------
    mode : dict[str, np.ndarray]
        Posterior mode, one ``(P,)`` array per variable.  Keys must match
        ``param_names`` exactly.
    cov : np.ndarray
        Flat covariance of shape ``(P*K, P*K)`` where P is the number of
        participants and K is ``len(param_names)``.  Ordering matches
        ``jax.flatten_util.ravel_pytree`` of a dict keyed by
        ``param_names`` in insertion order: columns
        ``[i*P : (i+1)*P]`` belong to ``param_names[i]``.
    param_names : tuple[str, ...]
        Canonical parameter order.  Must be ``_PARAM_ORDER_2LEVEL`` or
        ``_PARAM_ORDER_3LEVEL``.
    participant_ids : list[str]
        Participant identifiers, length P.
    n_pseudo_draws : int, default 1000
        Number of pseudo-samples to draw from ``N(mode, cov)``.  Does NOT
        affect the Laplace approximation quality (the Gaussian is exact);
        only affects Monte-Carlo noise in sample-based summaries like HDI.
    rng_key : int, default 0
        Seed for the numpy Generator used to draw the pseudo-samples.
    diagnostics : dict[str, float] | None, default None
        Optional diagnostic summary.  Canonical keys (must use these
        exact strings so 19-03 and downstream introspection work):
        ``converged``, ``n_iterations``, ``logp_at_mode``,
        ``hessian_min_eigval``, ``hessian_max_eigval``,
        ``n_eigenvalues_clipped``, ``ridge_added``.  If provided, each
        value is broadcast to shape ``(1, n_pseudo_draws)`` in the
        ``sample_stats`` group.

    Returns
    -------
    az.InferenceData
        Posterior with vars ``omega_2, log_beta, beta`` (2-level) or
        ``omega_2, log_beta, beta, omega_3, kappa, mu3_0`` (3-level),
        each shaped ``(chain=1, draw=n_pseudo_draws, participant_id=P)``.
        ``sample_stats`` group present (populated from ``diagnostics``
        or with a ``"laplace"`` marker scalar).

    Raises
    ------
    ValueError
        If ``param_names`` is not one of the two canonical tuples, if
        ``mode`` keys do not match ``param_names``, if any mode array
        has the wrong shape, or if ``cov`` has the wrong shape.
    """
    # ------------------------------------------------------------------
    # 1. Validate param_names is canonical
    # ------------------------------------------------------------------
    if param_names not in _VALID_PARAM_ORDERS:
        raise ValueError(
            f"param_names expected one of {_VALID_PARAM_ORDERS!r}; "
            f"got {param_names!r}"
        )

    # ------------------------------------------------------------------
    # 2. Validate mode keys match param_names exactly
    # ------------------------------------------------------------------
    mode_keys = tuple(mode.keys())
    if mode_keys != param_names:
        raise ValueError(
            f"mode keys expected {param_names!r}; got {mode_keys!r}"
        )

    P = len(participant_ids)
    K = len(param_names)

    # ------------------------------------------------------------------
    # 4a. Validate per-variable mode shapes (m4 guard)
    # ------------------------------------------------------------------
    for k, v in mode.items():
        v_arr = np.asarray(v)
        if v_arr.ndim != 1 or v_arr.shape[0] != P:
            raise ValueError(
                f"mode[{k!r}] must be shape ({P},); got {v_arr.shape}"
            )

    # ------------------------------------------------------------------
    # 4. Validate cov shape
    # ------------------------------------------------------------------
    expected_cov_shape = (P * K, P * K)
    if cov.shape != expected_cov_shape:
        raise ValueError(
            f"cov.shape expected {expected_cov_shape}, got {cov.shape}"
        )

    # ------------------------------------------------------------------
    # 5. Build flat mode vector
    # ------------------------------------------------------------------
    # ravel_pytree convention: for dict {v1: (P,), v2: (P,), ...},
    # the flat vector is [v1_0, v1_1, ..., v1_{P-1}, v2_0, ..., v2_{P-1}, ...].
    # So mode_flat[:P] = mode[param_names[0]], mode_flat[P:2P] = mode[param_names[1]], etc.
    mode_flat = np.concatenate([np.asarray(mode[v]) for v in param_names])  # (P*K,)

    # ------------------------------------------------------------------
    # 6. Draw n_pseudo_draws samples from N(mode_flat, cov)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(rng_key)
    # samples shape: (n_pseudo_draws, P*K)
    samples = rng.multivariate_normal(mode_flat, cov, size=n_pseudo_draws)

    # ------------------------------------------------------------------
    # 7. Unflatten: columns [i*P : (i+1)*P] → param_names[i]
    # ------------------------------------------------------------------
    # Each var gets shape (1, n_pseudo_draws, P): (chain=1, draw=K, participant_id=P)
    posterior: dict[str, np.ndarray] = {}
    for i, var in enumerate(param_names):
        posterior[var] = samples[:, i * P : (i + 1) * P][None, :, :]

    # ------------------------------------------------------------------
    # 8. Add deterministic beta = exp(log_beta)
    # ------------------------------------------------------------------
    if "log_beta" in param_names:
        posterior["beta"] = np.exp(posterior["log_beta"])

    # ------------------------------------------------------------------
    # 9. Build coords + dims
    # ------------------------------------------------------------------
    coords: dict[str, Any] = {
        "chain": [0],
        "draw": np.arange(n_pseudo_draws),
        "participant_id": list(participant_ids),
    }
    dims: dict[str, list[str]] = {var: ["participant_id"] for var in posterior}

    # ------------------------------------------------------------------
    # 10. Build sample_stats
    # ------------------------------------------------------------------
    if diagnostics is not None:
        sample_stats: dict[str, np.ndarray] = {
            k: np.full((1, n_pseudo_draws), float(v))
            for k, v in diagnostics.items()
        }
    else:
        # Marker scalar so downstream code can check idata.sample_stats.laplace
        sample_stats = {"laplace": np.ones((1, n_pseudo_draws), dtype=bool)}

    # ------------------------------------------------------------------
    # 11. Assemble and return InferenceData
    # ------------------------------------------------------------------
    idata = cast(
        az.InferenceData,
        az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats,
            coords=coords,
            dims=dims,
        ),
    )
    return idata
