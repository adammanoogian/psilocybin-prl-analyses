"""PyMC model factory functions for 2-level and 3-level binary HGF fitting.

Each factory function builds a PyMC model with literature-informed priors and
hooks in the custom JAX-backed logp Op via ``pm.Potential``.

Prior specification (canonical — verified against research and literature):

* ``omega_2``: ``TruncatedNormal(mu=-3.0, sigma=2.0, upper=0.0)``
  — omega_2 >= 0 causes NaN in the binary HGF scan; bound is mandatory.
* ``omega_3``: ``TruncatedNormal(mu=-6.0, sigma=2.0, upper=0.0)``
  — same reasoning; binary HGF omega_3 clusters around -6 to -8.
* ``kappa``: ``TruncatedNormal(mu=1.0, sigma=0.5, lower=0.01, upper=2.0)``
  — must be positive; kappa > 2 is rare in the literature.
* ``log_beta``: ``Normal(mu=0.0, sigma=1.5)``
  — sampled in log-space to avoid the zero boundary; corresponds to
  ``beta ~ LogNormal(0, 1.5)``.
* ``beta``: ``Deterministic(exp(log_beta))``
  — ensures PyMC propagates the back-transformed value to InferenceData.
* ``zeta``: ``Normal(mu=0.0, sigma=2.0)``
  — unbounded; positive = persist, negative = switch.

Notes
-----
``pytensor.config.cxx = ""`` is set at module level to suppress the g++
warning on ds_env (no compiler installed; not needed for JAX-backed Ops).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor

from prl_hgf.fitting.ops import build_logp_ops_2level, build_logp_ops_3level

# Suppress PyTensor g++ compilation warning
pytensor.config.cxx = ""


def build_pymc_model_2level(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
) -> tuple[pm.Model, list[str]]:
    """Build a PyMC model for the 2-level binary HGF.

    Constructs the JAX-backed logp Op for the 2-level network and wraps it
    in a PyMC model with literature-informed priors.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array from :func:`~prl_hgf.models.hgf_2level.prepare_input_data`.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.

    Returns
    -------
    model : pm.Model
        Compiled PyMC model with priors for ``omega_2``, ``log_beta``,
        ``beta``, and ``zeta``.
    var_names : list[str]
        Names of the free parameters to pass to ``az.summary``:
        ``["omega_2", "beta", "zeta"]``.

    Examples
    --------
    >>> import numpy as np
    >>> inp = np.zeros((10, 3))
    >>> obs = np.zeros((10, 3), dtype=int)
    >>> obs[:, 0] = 1
    >>> ch = np.zeros(10, dtype=int)
    >>> model, var_names = build_pymc_model_2level(inp, obs, ch)
    >>> var_names
    ['omega_2', 'beta', 'zeta']
    """
    logp_op, _ = build_logp_ops_2level(input_data_arr, observed_arr, choices_arr)

    with pm.Model() as model:
        # Perceptual parameter: tonic volatility (must be < 0)
        omega_2 = pm.TruncatedNormal("omega_2", mu=-3.0, sigma=2.0, upper=0.0)

        # Response parameters
        log_beta = pm.Normal("log_beta", mu=0.0, sigma=1.5)
        beta = pm.Deterministic("beta", pm.math.exp(log_beta))
        zeta = pm.Normal("zeta", mu=0.0, sigma=2.0)

        # Hook the custom JAX logp into PyMC via pm.Potential
        pm.Potential("loglike", logp_op(omega_2, beta, zeta))

    var_names = ["omega_2", "beta", "zeta"]
    return model, var_names


def build_pymc_model_3level(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
) -> tuple[pm.Model, list[str]]:
    """Build a PyMC model for the 3-level binary HGF.

    Extends the 2-level model by adding priors for ``omega_3`` and ``kappa``.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.

    Returns
    -------
    model : pm.Model
        Compiled PyMC model with priors for ``omega_2``, ``omega_3``,
        ``kappa``, ``log_beta``, ``beta``, and ``zeta``.
    var_names : list[str]
        Names of the free parameters to pass to ``az.summary``:
        ``["omega_2", "omega_3", "kappa", "beta", "zeta"]``.

    Notes
    -----
    The primary hypotheses focus on ``omega_2`` and ``kappa``.  ``omega_3``
    recovery is expected to be poor (known literature issue).

    Examples
    --------
    >>> import numpy as np
    >>> inp = np.zeros((10, 3))
    >>> obs = np.zeros((10, 3), dtype=int)
    >>> obs[:, 0] = 1
    >>> ch = np.zeros(10, dtype=int)
    >>> model, var_names = build_pymc_model_3level(inp, obs, ch)
    >>> var_names
    ['omega_2', 'omega_3', 'kappa', 'beta', 'zeta']
    """
    logp_op, _ = build_logp_ops_3level(input_data_arr, observed_arr, choices_arr)

    with pm.Model() as model:
        # Perceptual parameters
        omega_2 = pm.TruncatedNormal("omega_2", mu=-3.0, sigma=2.0, upper=0.0)
        omega_3 = pm.TruncatedNormal("omega_3", mu=-6.0, sigma=2.0, upper=0.0)
        kappa = pm.TruncatedNormal(
            "kappa", mu=1.0, sigma=0.5, lower=0.01, upper=2.0
        )

        # Response parameters
        log_beta = pm.Normal("log_beta", mu=0.0, sigma=1.5)
        beta = pm.Deterministic("beta", pm.math.exp(log_beta))
        zeta = pm.Normal("zeta", mu=0.0, sigma=2.0)

        # Hook the custom JAX logp into PyMC via pm.Potential
        pm.Potential("loglike", logp_op(omega_2, omega_3, kappa, beta, zeta))

    var_names = ["omega_2", "omega_3", "kappa", "beta", "zeta"]
    return model, var_names
