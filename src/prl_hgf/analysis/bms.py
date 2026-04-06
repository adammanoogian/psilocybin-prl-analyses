"""Bayesian model comparison module for the PRL HGF pipeline.

Provides post-hoc WAIC computation for pm.Potential models (which do NOT
populate the log_likelihood group in InferenceData automatically), a
random-effects group BMS wrapper via the groupBMC package (Rigoux et al.
2014), and exceedance probability visualization.

Key challenge
-------------
PyMC models built with ``pm.Potential`` do not fill the ``log_likelihood``
group of ``az.InferenceData`` during sampling.  WAIC therefore cannot be
computed directly via ``az.waic``.  Instead, we re-evaluate the JAX JIT-
compiled logp function over all posterior samples and inject the resulting
array into a new ``log_likelihood`` group before calling ``az.waic``.

groupBMC API notes
------------------
``GroupBMC(L)`` expects ``L`` with shape ``(n_models, n_subjects)`` — the
TRANSPOSE of the ``(n_subjects, n_models)`` matrix used internally here.
``GroupBMCResult`` exposes:

* ``exceedance_probability``       — xp
* ``protected_exceedance_probability`` — pxp
* ``frequency_mean``               — exp_r (expected model frequencies)
* ``frequency_var``
* ``attribution``                  — per-subject posterior responsibility
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from groupBMC.groupBMC import GroupBMC

__all__ = [
    "compute_subject_waic",
    "compute_batch_waic",
    "run_group_bms",
    "run_stratified_bms",
    "plot_exceedance_probabilities",
]

log = logging.getLogger(__name__)

# ---- model names accepted by compute_subject_waic -------------------------
_MODEL_2LEVEL = "hgf_2level"
_MODEL_3LEVEL = "hgf_3level"


# ---------------------------------------------------------------------------
# WAIC computation (post-hoc, bypasses pm.Potential limitation)
# ---------------------------------------------------------------------------


def compute_subject_waic(
    input_data_arr: np.ndarray,
    observed_arr: np.ndarray,
    choices_arr: np.ndarray,
    idata: az.InferenceData,
    model_name: str,
) -> float:
    """Compute WAIC for one participant by re-evaluating the JAX logp.

    Because ``pm.Potential`` does not populate the ``log_likelihood`` group
    in ``az.InferenceData``, WAIC must be computed post-hoc by calling the
    JAX JIT-compiled logp function over every posterior sample.

    Parameters
    ----------
    input_data_arr : numpy.ndarray, shape (n_trials, 3)
        Float reward-value array.  ``input_data_arr[t, k]`` is the reward on
        trial ``t`` if cue ``k`` was chosen, else ``0.0``.
    observed_arr : numpy.ndarray, shape (n_trials, 3) int
        Binary mask: ``1`` when cue ``k`` was chosen on trial ``t``.
    choices_arr : numpy.ndarray, shape (n_trials,) int
        Chosen cue index ``(0, 1, or 2)`` for each trial.
    idata : az.InferenceData
        InferenceData object with ``posterior`` group from MCMC sampling.
    model_name : str
        ``"hgf_2level"`` or ``"hgf_3level"``.

    Returns
    -------
    float
        ``elpd_waic`` (Expected Log Predictive Density); higher is better.

    Notes
    -----
    Runtime budget: for 4 chains x 1000 draws the loop runs 4 000 times per
    participant.  Each call is JAX JIT-compiled after the first evaluation, so
    typical throughput is ~1 000–5 000 evaluations/second.  A progress log
    line is printed every 500 evaluations.
    """
    from prl_hgf.fitting.ops import build_logp_ops_2level, build_logp_ops_3level

    if model_name == _MODEL_2LEVEL:
        logp_op, _ = build_logp_ops_2level(
            input_data_arr, observed_arr, choices_arr
        )
    elif model_name == _MODEL_3LEVEL:
        logp_op, _ = build_logp_ops_3level(
            input_data_arr, observed_arr, choices_arr
        )
    else:
        raise ValueError(
            f"model_name must be {_MODEL_2LEVEL!r} or {_MODEL_3LEVEL!r},"
            f" got {model_name!r}"
        )

    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]

    # Collect per-sample log-likelihood values into (n_chains, n_draws) array
    loglike_vals = np.empty((n_chains, n_draws), dtype=float)
    total = n_chains * n_draws
    count = 0

    for chain_idx in range(n_chains):
        for draw_idx in range(n_draws):
            if model_name == _MODEL_2LEVEL:
                o2 = float(posterior["omega_2"].values[chain_idx, draw_idx])
                b = float(posterior["beta"].values[chain_idx, draw_idx])
                z = float(posterior["zeta"].values[chain_idx, draw_idx])
                lp = float(logp_op(o2, b, z).eval())
            else:
                o2 = float(posterior["omega_2"].values[chain_idx, draw_idx])
                o3 = float(posterior["omega_3"].values[chain_idx, draw_idx])
                k = float(posterior["kappa"].values[chain_idx, draw_idx])
                b = float(posterior["beta"].values[chain_idx, draw_idx])
                z = float(posterior["zeta"].values[chain_idx, draw_idx])
                lp = float(logp_op(o2, o3, k, b, z).eval())

            loglike_vals[chain_idx, draw_idx] = lp
            count += 1
            if count % 500 == 0:
                log.info(
                    "compute_subject_waic [%s]: %d / %d evaluations",
                    model_name,
                    count,
                    total,
                )

    # Build xarray DataArray and inject into a copy of idata
    ll_da = xr.DataArray(
        loglike_vals[:, :, np.newaxis],
        dims=["chain", "draw", "loglike_dim_0"],
        coords={
            "chain": posterior.coords["chain"].values,
            "draw": posterior.coords["draw"].values,
        },
    )

    # Add log_likelihood group (mutates idata in-place)
    idata.add_groups({"log_likelihood": {"loglike": ll_da}})

    waic_result = az.waic(idata, var_name="loglike")
    return float(waic_result.elpd_waic)


def compute_batch_waic(
    sim_df: pd.DataFrame,
    idata_dict: dict[str, dict[tuple, az.InferenceData]],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute WAIC for all participants across models.

    Iterates over every ``(participant_id, group, session)`` combination and
    every model, reconstructs trial arrays from ``sim_df``, and calls
    :func:`compute_subject_waic`.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulation DataFrame with columns ``participant_id``, ``group``,
        ``session``, ``input_data``, ``observed``, ``choice``.  One row per
        trial.
    idata_dict : dict[str, dict[tuple, az.InferenceData]]
        Nested mapping: ``idata_dict[model_name][(pid, group, session)]``.
        Must contain pre-loaded InferenceData objects for every participant.
    model_names : list[str] or None
        Models to compare.  Defaults to ``["hgf_2level", "hgf_3level"]``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``participant_id``, ``group``, ``session``, ``model``,
        ``elpd_waic``.

    Notes
    -----
    ``sim_df`` must have columns ``reward_c0``, ``reward_c1``, ``reward_c2``
    (float) for ``input_data_arr``, ``observed_c0``, ``observed_c1``,
    ``observed_c2`` (int) for ``observed_arr``, and ``choice`` (int) for
    ``choices_arr``.  These column names are produced by the batch simulation
    script.

    If a fit was flagged (convergence failure) or ``idata`` is missing, the
    row is skipped and a warning is logged.
    """
    if model_names is None:
        model_names = [_MODEL_2LEVEL, _MODEL_3LEVEL]

    keys = (
        sim_df[["participant_id", "group", "session"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    rows: list[dict] = []
    for pid, grp, sess in keys:
        mask = (
            (sim_df["participant_id"] == pid)
            & (sim_df["group"] == grp)
            & (sim_df["session"] == sess)
        )
        sub = sim_df.loc[mask].sort_values("trial").reset_index(drop=True)

        # Reconstruct trial-level arrays
        input_data_arr = sub[["reward_c0", "reward_c1", "reward_c2"]].to_numpy(
            dtype=float
        )
        observed_arr = sub[
            ["observed_c0", "observed_c1", "observed_c2"]
        ].to_numpy(dtype=int)
        choices_arr = sub["choice"].to_numpy(dtype=int)

        for model in model_names:
            idata_map = idata_dict.get(model, {})
            idata = idata_map.get((pid, grp, sess))
            if idata is None:
                log.warning(
                    "No idata for participant=%s group=%s session=%s model=%s"
                    " — skipping",
                    pid,
                    grp,
                    sess,
                    model,
                )
                continue

            try:
                elpd = compute_subject_waic(
                    input_data_arr, observed_arr, choices_arr, idata, model
                )
                rows.append(
                    {
                        "participant_id": pid,
                        "group": grp,
                        "session": sess,
                        "model": model,
                        "elpd_waic": elpd,
                    }
                )
            except Exception:
                log.exception(
                    "WAIC failed for participant=%s group=%s session=%s"
                    " model=%s",
                    pid,
                    grp,
                    sess,
                    model,
                )

    return pd.DataFrame(
        rows,
        columns=["participant_id", "group", "session", "model", "elpd_waic"],
    )


# ---------------------------------------------------------------------------
# Group BMS
# ---------------------------------------------------------------------------


def run_group_bms(
    elpd_matrix: np.ndarray,
    model_names: list[str],
    group_label: str = "all",
) -> dict:
    """Run random-effects Bayesian model selection using groupBMC.

    Parameters
    ----------
    elpd_matrix : numpy.ndarray, shape (n_subjects, n_models)
        Per-subject model evidence (elpd_waic).  Higher values indicate better
        model fit.
    model_names : list[str]
        Model name for each column of ``elpd_matrix``.
    group_label : str
        Label for this BMS run (used in plots and return dict).

    Returns
    -------
    dict
        Keys:

        ``alpha`` — Dirichlet posterior sufficient statistics (shape n_models).
        ``exp_r`` — Expected model frequencies (posterior Dirichlet mean).
        ``xp``    — Exceedance probabilities.
        ``pxp``   — Protected exceedance probabilities.
        ``bor``   — Bayesian Omnibus Risk.
        ``model_names`` — Echoed input.
        ``group_label`` — Echoed input.
        ``n_subjects`` — Number of subjects in this BMS run.

    Notes
    -----
    ``GroupBMC`` expects ``L`` with shape ``(n_models, n_subjects)`` so the
    matrix is transposed before passing.  The ``bor`` value is derived from
    the free-energy difference ``F1 - F0`` inside ``GroupBMC`` and is not
    directly exposed on ``GroupBMCResult``; we therefore compute it from the
    ``protected_exceedance_probability`` relationship::

        pxp = (1 - bor) * xp + bor / n_models
        => bor = (pxp - xp) / (1/n_models - xp)  [element-wise, averaged]

    In practice, we extract ``bor`` by calling ``get_result`` and checking
    the ``bor`` attribute directly since the package computes it internally.
    """
    assert (
        elpd_matrix.ndim == 2 and elpd_matrix.shape[1] == len(model_names)
    ), (
        f"elpd_matrix must be (n_subjects, n_models={len(model_names)}),"
        f" got {elpd_matrix.shape}"
    )

    n_subjects, n_models = elpd_matrix.shape

    # GroupBMC expects (n_models, n_subjects)
    L = elpd_matrix.T

    bmc = GroupBMC(L)
    result = bmc.get_result()

    # Extract bor from GroupBMC internal calculation
    from math import exp as _exp

    bor = float(1 / (1 + _exp(bmc.F1() - bmc.F0())))

    xp = np.asarray(result.exceedance_probability)
    pxp = np.asarray(result.protected_exceedance_probability)
    exp_r = np.asarray(result.frequency_mean)
    alpha = np.asarray(result.attribution.sum(axis=1)) + 1.0  # α_0=1 prior

    return {
        "alpha": alpha,
        "exp_r": exp_r,
        "xp": xp,
        "pxp": pxp,
        "bor": bor,
        "model_names": list(model_names),
        "group_label": group_label,
        "n_subjects": n_subjects,
    }


def run_stratified_bms(
    waic_df: pd.DataFrame,
    model_names: list[str],
) -> dict[str, dict]:
    """Run BMS for the full sample and per group.

    Parameters
    ----------
    waic_df : pandas.DataFrame
        Output of :func:`compute_batch_waic`.  Must have columns
        ``participant_id``, ``group``, ``session``, ``model``,
        ``elpd_waic``.
    model_names : list[str]
        Model names to include.  Order determines column order in the ELPD
        matrix.

    Returns
    -------
    dict[str, dict]
        Keys: ``"all"`` plus each unique value in ``waic_df["group"]``.
        Values: dicts returned by :func:`run_group_bms`.

    Notes
    -----
    The ELPD matrix for each BMS run is built by averaging ``elpd_waic``
    across sessions for each ``(participant_id, model)`` pair, then pivoting
    to ``(n_subjects, n_models)``.

    Per-group BMS with fewer than ~20 subjects has limited statistical power
    and should be treated as exploratory.
    """
    results: dict[str, dict] = {}

    def _build_matrix(df: pd.DataFrame) -> np.ndarray | None:
        """Average across sessions then pivot to (subjects, models)."""
        avg = (
            df.groupby(["participant_id", "model"])["elpd_waic"]
            .mean()
            .reset_index()
        )
        pivot = avg.pivot(
            index="participant_id", columns="model", values="elpd_waic"
        )
        # Ensure column order matches model_names
        missing = [m for m in model_names if m not in pivot.columns]
        if missing:
            log.warning(
                "Models %s missing from waic_df — cannot build ELPD matrix",
                missing,
            )
            return None
        pivot = pivot[model_names].dropna()
        if len(pivot) == 0:
            return None
        return pivot.to_numpy(dtype=float)

    # Full sample
    mat = _build_matrix(waic_df)
    if mat is not None:
        results["all"] = run_group_bms(mat, model_names, group_label="all")
    else:
        log.warning("Could not build ELPD matrix for full sample BMS")

    # Per group
    for grp in sorted(waic_df["group"].unique()):
        sub = waic_df[waic_df["group"] == grp]
        mat = _build_matrix(sub)
        if mat is None:
            log.warning("Could not build ELPD matrix for group=%s", grp)
            continue
        n = len(mat)
        if n < 20:
            log.warning(
                "Per-group BMS with N=%d subjects has limited power"
                " (exploratory) for group=%s",
                n,
                grp,
            )
        results[grp] = run_group_bms(mat, model_names, group_label=grp)

    return results


# ---------------------------------------------------------------------------
# Exceedance probability bar plot
# ---------------------------------------------------------------------------


def plot_exceedance_probabilities(
    bms_results: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot exceedance probabilities (EP) and protected EP for all groups.

    Parameters
    ----------
    bms_results : dict[str, dict]
        Output of :func:`run_stratified_bms` or a subset thereof.  Each value
        is a dict with keys ``xp``, ``pxp``, ``model_names``, ``n_subjects``,
        ``bor``, ``group_label``.
    save_path : Path or None
        If provided, figure is saved as PNG at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        Multi-panel figure (one subplot per group label).
    """
    group_labels = list(bms_results.keys())
    n_panels = len(group_labels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4 * n_panels, 4),
        squeeze=False,
    )
    axes = axes[0]  # squeeze the row dimension

    bar_width = 0.35

    for ax, label in zip(axes, group_labels, strict=True):
        res = bms_results[label]
        xp = np.asarray(res["xp"])
        pxp = np.asarray(res["pxp"])
        model_names = res["model_names"]
        n_subjects = res["n_subjects"]
        bor = res["bor"]
        n_models = len(model_names)

        x = np.arange(n_models)

        ax.bar(
            x - bar_width / 2,
            xp,
            width=bar_width,
            label="EP",
            color="#4C72B0",
            alpha=0.85,
        )
        ax.bar(
            x + bar_width / 2,
            pxp,
            width=bar_width,
            label="Protected EP",
            color="#DD8452",
            alpha=0.85,
        )

        # Chance level
        ax.axhline(
            1 / n_models,
            color="grey",
            linestyle="--",
            linewidth=1.2,
            label=f"Chance (1/{n_models})",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_title(
            f"{label}\n(N={n_subjects})",
            fontsize=10,
        )

        # Annotate BOR below bars
        ax.text(
            0.5,
            -0.22,
            f"BOR = {bor:.3f}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="dimgrey",
        )

        ax.legend(fontsize=7)

    fig.suptitle("Exceedance Probabilities — Bayesian Model Comparison", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved EP figure to %s", save_path)

    return fig
