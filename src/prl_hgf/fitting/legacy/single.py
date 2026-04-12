"""Single-participant MCMC fitting for 2-level and 3-level binary HGF models.

# Frozen for v1.1 reproducibility — DO NOT MODIFY.
# See src/prl_hgf/fitting/hierarchical.py for the v1.2+ implementation.

Provides the primary entry point :func:`fit_participant` and two helper
functions:

* :func:`extract_summary_rows` — converts an ArviZ InferenceData summary
  DataFrame to the structured row format used in batch fitting output.
* :func:`flag_fit` — checks convergence diagnostics against thresholds.

Output schema (one row per parameter per participant):

    participant_id, group, session, model, parameter,
    mean, sd, hdi_3%, hdi_97%, r_hat, ess

Notes
-----
PyTensor g++ compilation is suppressed at module level because the
performance-critical path delegates to JAX JIT.

``cores=1`` is used by default on Windows to avoid JAX process-isolation
issues with multi-core MCMC chains.  Pass ``cores=4`` only if JAX
cross-process state is confirmed stable in the target environment.
"""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pymc as pm
import pytensor

from prl_hgf.fitting.models import build_pymc_model_2level, build_pymc_model_3level

# Suppress PyTensor g++ compilation warning
pytensor.config.cxx = ""

# ---------------------------------------------------------------------------
# Diagnostic thresholds (mirror config fitting.diagnostics section)
# ---------------------------------------------------------------------------

R_HAT_THRESHOLD: float = 1.05
ESS_THRESHOLD: float = 400.0

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def extract_summary_rows(
    idata: az.InferenceData,
    participant_id: str,
    group: str,
    session: str,
    model: str,
    var_names: list[str],
) -> list[dict]:
    """Convert ArviZ summary to a list of structured dicts.

    Transforms the ``az.summary`` DataFrame into one dict per parameter,
    matching the FIT-04 output schema.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object returned by PyMC sampling.
    participant_id : str
        Participant identifier string.
    group : str
        Group label (e.g. ``"placebo"``).
    session : str
        Session label (e.g. ``"baseline"``).
    model : str
        Model name (e.g. ``"hgf_2level"``).
    var_names : list[str]
        Variables to include in the summary (passed directly to
        ``az.summary``).

    Returns
    -------
    list[dict]
        One dict per parameter with keys: ``participant_id``, ``group``,
        ``session``, ``model``, ``parameter``, ``mean``, ``sd``,
        ``hdi_3%``, ``hdi_97%``, ``r_hat``, ``ess``.

    Notes
    -----
    ``ess`` uses ``ess_bulk`` from ArviZ (robust to non-stationarity).
    """
    summary_df = az.summary(idata, var_names=var_names, round_to=8)
    rows: list[dict] = []
    for param_name in summary_df.index:
        row = summary_df.loc[param_name]
        rows.append(
            {
                "participant_id": participant_id,
                "group": group,
                "session": session,
                "model": model,
                "parameter": param_name,
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "hdi_3%": float(row["hdi_3%"]),
                "hdi_97%": float(row["hdi_97%"]),
                "r_hat": float(row["r_hat"]),
                "ess": float(row["ess_bulk"]),
            }
        )
    return rows


def flag_fit(
    summary_rows: list[dict],
    r_hat_threshold: float = R_HAT_THRESHOLD,
    ess_threshold: float = ESS_THRESHOLD,
) -> bool:
    """Return True if the fit should be flagged as potentially problematic.

    Flags a fit if any parameter exceeds the R-hat threshold or falls below
    the ESS threshold.  Flagged fits are still saved; the flag is a signal
    to re-run or inspect the trace.

    Parameters
    ----------
    summary_rows : list[dict]
        Output of :func:`extract_summary_rows`.
    r_hat_threshold : float, optional
        R-hat flag threshold.  Default ``1.05``.
    ess_threshold : float, optional
        ESS (bulk) flag threshold.  Default ``400``.

    Returns
    -------
    bool
        ``True`` if any parameter has ``r_hat > r_hat_threshold`` or
        ``ess < ess_threshold``.
    """
    for row in summary_rows:
        if row["r_hat"] > r_hat_threshold:
            return True
        if row["ess"] < ess_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Primary fitting function
# ---------------------------------------------------------------------------


def fit_participant(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    participant_id: str,
    group: str,
    session: str,
    model_name: str = "hgf_2level",
    n_chains: int = 4,
    n_draws: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42,
    cores: int = 1,
    sampler: str = "pymc",
) -> tuple[az.InferenceData, list[dict], bool]:
    """Fit a single participant's data via NUTS MCMC.

    Builds the requested PyMC model, runs NUTS sampling, and returns the
    posterior summary in the FIT-04 schema along with a convergence flag.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array from
        :func:`~prl_hgf.models.hgf_2level.prepare_input_data`.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary observed mask.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index for each trial.
    participant_id : str
        Participant identifier (used in summary rows).
    group : str
        Group label (used in summary rows).
    session : str
        Session label (used in summary rows).
    model_name : str, optional
        Which model to fit.  One of ``"hgf_2level"`` (default) or
        ``"hgf_3level"``.
    n_chains : int, optional
        Number of MCMC chains.  Default ``4``.
    n_draws : int, optional
        Number of posterior draws per chain (after tuning).  Default ``1000``.
    n_tune : int, optional
        Number of tuning (warm-up) steps per chain.  Default ``1000``.
    target_accept : float, optional
        Target acceptance rate for NUTS step-size adaptation.  Default
        ``0.9``.
    random_seed : int, optional
        RNG seed for reproducibility.  Default ``42``.
    cores : int, optional
        Number of parallel chains.  Use ``1`` on Windows to avoid JAX
        process-isolation issues.  Default ``1``.
    sampler : str, optional
        MCMC backend.  ``"pymc"`` (default) uses the PyMC/PyTensor NUTS
        sampler.  ``"numpyro"`` uses NumPyro's NUTS via JAX, which
        bypasses PyTensor compilation entirely and runs on CPU or GPU.

    Returns
    -------
    idata : az.InferenceData
        ArviZ InferenceData object with posterior samples and sample stats.
    summary_rows : list[dict]
        One dict per free parameter with the FIT-04 schema columns.
    flagged : bool
        ``True`` if any convergence diagnostic exceeds threshold.

    Raises
    ------
    ValueError
        If ``model_name`` is not ``"hgf_2level"`` or ``"hgf_3level"``.

    Examples
    --------
    >>> import numpy as np
    >>> inp = np.zeros((50, 3))
    >>> obs = np.zeros((50, 3), dtype=int)
    >>> obs[:, 0] = 1
    >>> ch = np.zeros(50, dtype=int)
    >>> idata, rows, flagged = fit_participant(
    ...     inp, obs, ch, "P001", "placebo", "baseline",
    ...     model_name="hgf_2level", n_chains=1, n_draws=50, n_tune=50,
    ... )
    >>> len(rows)
    3
    """
    if model_name == "hgf_2level":
        model, var_names = build_pymc_model_2level(
            input_data_arr, observed_arr, choices_arr
        )
    elif model_name == "hgf_3level":
        model, var_names = build_pymc_model_3level(
            input_data_arr, observed_arr, choices_arr
        )
    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}. "
            f"Expected 'hgf_2level' or 'hgf_3level'."
        )

    with model:
        if sampler == "numpyro":
            import pymc.sampling.jax as pmjax

            idata = pmjax.sample_numpyro_nuts(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
            )
        else:
            idata = pm.sample(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True,
            )

    summary_rows = extract_summary_rows(
        idata,
        participant_id=participant_id,
        group=group,
        session=session,
        model=model_name,
        var_names=var_names,
    )

    flagged = flag_fit(summary_rows)

    if flagged:
        bad_params = [
            r["parameter"]
            for r in summary_rows
            if r["r_hat"] > R_HAT_THRESHOLD or r["ess"] < ESS_THRESHOLD
        ]
        log.warning(
            "Fit flagged for %s (%s / %s / %s): problematic params = %s",
            participant_id,
            group,
            session,
            model_name,
            bad_params,
        )

    return idata, summary_rows, flagged
