"""Diagnostic CSV side-car emitter for MCMC fitting runs.

Writes a compact CSV with ArviZ parameter summaries and backend-agnostic
sampler diagnostics alongside each fitted InferenceData object.  The CSV
is human-readable, diff-friendly, and suitable for automated sweep
aggregation scripts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import arviz as az

    from prl_hgf.fitting.config import FitConfig


def emit_diagnostic_csv(
    idata: az.InferenceData,
    output_path: Path,
    fit_config: FitConfig | None = None,
    walltime_s: float | None = None,
) -> Path:
    """Write a diagnostic CSV summarising an MCMC fit.

    The CSV contains two sections separated by a blank row:

    1. **Parameter summary** -- ``az.summary(idata)`` exported as-is with
       an additional ``parameter`` column from the index.
    2. **Sampler diagnostics** -- key/value rows with backend-agnostic
       field resolution (BlackJAX field names vs NumPyro field names),
       optional ``FitConfig`` metadata, and walltime.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted inference data with ``posterior`` and (optionally)
        ``sample_stats`` groups.
    output_path : Path
        Destination CSV path.  Parent directories are created if needed.
    fit_config : FitConfig or None, optional
        If provided, sampler metadata (backend, model_name, n_chains,
        n_draws) is appended to the diagnostics section.
    walltime_s : float or None, optional
        Total wall-clock seconds for the fitting call.

    Returns
    -------
    Path
        The ``output_path`` that was written (pass-through for chaining).
    """
    import arviz as az
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Section 1: parameter summary from ArviZ                             #
    # ------------------------------------------------------------------ #
    summary_df = az.summary(idata)
    summary_cols = list(summary_df.columns)

    # ------------------------------------------------------------------ #
    # Section 2: sampler diagnostics (backend-agnostic field resolution)  #
    # ------------------------------------------------------------------ #
    diag_rows: list[tuple[str, str]] = []

    if hasattr(idata, "sample_stats"):
        ss = idata.sample_stats

        # Divergences: "diverging" (BlackJAX) / "divergence" (NumPyro)
        div_array = None
        if "diverging" in ss:
            div_array = ss["diverging"].values
        elif "divergence" in ss:
            div_array = ss["divergence"].values

        if div_array is not None:
            n_divergent = int(np.sum(div_array))
            divergent_rate = n_divergent / max(div_array.size, 1)
            diag_rows.append(("n_divergent", str(n_divergent)))
            diag_rows.append(("divergent_rate", f"{divergent_rate:.6f}"))

        # Acceptance rate: "acceptance_rate" (BlackJAX) /
        # "mean_accept_prob" (NumPyro)
        accept_array = None
        if "acceptance_rate" in ss:
            accept_array = ss["acceptance_rate"].values
        elif "mean_accept_prob" in ss:
            accept_array = ss["mean_accept_prob"].values

        if accept_array is not None:
            diag_rows.append(
                ("mean_accept_rate", f"{float(np.mean(accept_array)):.4f}")
            )

        # Leapfrog steps: "num_integration_steps" (BlackJAX) /
        # "num_steps" (NumPyro)
        lf_array = None
        if "num_integration_steps" in ss:
            lf_array = ss["num_integration_steps"].values
        elif "num_steps" in ss:
            lf_array = ss["num_steps"].values

        if lf_array is not None:
            total_lf = int(np.sum(lf_array))
            diag_rows.append(("total_leapfrog_steps", str(total_lf)))
            diag_rows.append(
                ("mean_leapfrog_per_draw", f"{float(np.mean(lf_array)):.2f}")
            )

        # Energy (both backends use "energy" when present)
        if "energy" in ss:
            energy_vals = ss["energy"].values
            finite_e = energy_vals[np.isfinite(energy_vals)]
            if finite_e.size > 0:
                diag_rows.append(
                    ("energy_mean", f"{float(np.mean(finite_e)):.4f}")
                )
                diag_rows.append(
                    ("energy_std", f"{float(np.std(finite_e)):.4f}")
                )

    # FitConfig metadata
    if fit_config is not None:
        diag_rows.append(("backend", fit_config.sampler.backend))
        diag_rows.append(("model_name", fit_config.model_name))
        diag_rows.append(("n_chains", str(fit_config.sampler.n_chains)))
        diag_rows.append(("n_draws", str(fit_config.sampler.n_draws)))
        diag_rows.append(("n_warmup", str(fit_config.sampler.n_warmup)))
        diag_rows.append(
            ("target_accept", str(fit_config.sampler.target_accept))
        )
        diag_rows.append(
            ("max_tree_depth", str(fit_config.sampler.max_tree_depth))
        )
        diag_rows.append(
            ("mass_matrix_kind", fit_config.mitigation.mass_matrix_kind)
        )
        diag_rows.append(("pooling", fit_config.covariate.pooling))

    # Walltime
    if walltime_s is not None:
        diag_rows.append(("walltime_s", f"{walltime_s:.2f}"))

    # ------------------------------------------------------------------ #
    # Write CSV                                                            #
    # ------------------------------------------------------------------ #
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Section 1 header + rows
        writer.writerow(["parameter"] + summary_cols)
        for param_name, row in summary_df.iterrows():
            writer.writerow([param_name] + [str(v) for v in row.values])

        # Blank separator
        writer.writerow([])

        # Section 2 header + rows
        writer.writerow(["diagnostic", "value"])
        for key, val in diag_rows:
            writer.writerow([key, val])

    return output_path
