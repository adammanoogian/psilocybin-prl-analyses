"""Phase 34 grid-sweep driver: benchmark hierarchical mode (Mode B).

Runs a single grid cell (model x n_per_group x mitigation_combo) per
invocation.  Accepts ``--cell-id`` (int 0-23) and optional ``--job-id``
from SLURM.  The 24-cell grid covers:

    2 models x 3 n_per_group values x 4 mitigation combos = 24 cells.

The driver decodes cell_id, loads the appropriate FitConfig YAML, simulates
a hierarchical cohort via simulate_hierarchical_cohort (with true group-level
hyperparameters), fits via fit_batch_hierarchical, and writes a JSON result
file with recovery diagnostics.

CRITICAL two-level seed strategy (Pitfall 2 prevention):
- Cohort seed: deterministic per (model, n_per_group) -- SAME across all
  mitigation combos so recovery comparisons are apples-to-apples.
- MCMC seed: cell_id-based -- different chain initialisation per cell.
- Covariate vector: generated with cohort seed so identical across combos
  for the same model x n_per_group.

No fp64 in Mode B grid -- use non-centering (M2) instead.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
MODELS = ["hgf_2level", "hgf_3level"]
N_PER_GROUP = [10, 17, 33]
MITIGATION_COMBOS = [
    "hier+M1",
    "hier+M1+M2",
    "hier+M1+M2+Laplace",
    "hier+M1+M2+Laplace+covariates",
]

TOTAL_CELLS = len(MODELS) * len(N_PER_GROUP) * len(MITIGATION_COMBOS)  # 24

# Output directory for benchmark results
RESULTS_DIR = Path("models/power/bench_mode_b_results")

# Seed base for reproducibility
_SEED_BASE = 99000

# ---------------------------------------------------------------------------
# True hyperparameters used for simulation
# ---------------------------------------------------------------------------
# 2-level model: omega_2, log_beta, zeta
# 3-level model: omega_2, log_beta, zeta, omega_3
_TRUE_MU_2LEVEL = {
    "omega_2": [[-3.0, -2.5]],       # list form; converted to np later
    "log_beta": [[0.0, 0.5]],
    "zeta": [[0.0, 0.1]],
}
_TRUE_MU_3LEVEL_EXTRA = {
    "omega_3": [[-6.0, -5.5]],
}
_TRUE_SIGMA_2LEVEL = {"omega_2": 0.5, "log_beta": 0.5, "zeta": 0.5}
_TRUE_SIGMA_3LEVEL_EXTRA = {"omega_3": 0.5}
_TRUE_BETA_2LEVEL = {"omega_2": 0.8, "log_beta": 0.3, "zeta": 0.2}
_TRUE_BETA_3LEVEL_EXTRA = {"omega_3": 0.4}


def decode_cell_id(cell_id: int) -> tuple[str, int, str]:
    """Decode a linear cell index into grid coordinates.

    Parameters
    ----------
    cell_id : int
        Linear index in range [0, 23].

    Returns
    -------
    tuple[str, int, str]
        (model_name, n_per_group, mitigation_combo).

    Raises
    ------
    ValueError
        If cell_id is out of range [0, 23].

    Notes
    -----
    Ordering: model outermost (12 cells each), then n_per_group (4 cells
    each), then mitigation innermost.

    Examples
    --------
    >>> decode_cell_id(0)
    ('hgf_2level', 10, 'hier+M1')
    >>> decode_cell_id(23)
    ('hgf_3level', 33, 'hier+M1+M2+Laplace+covariates')
    """
    if cell_id < 0 or cell_id >= TOTAL_CELLS:
        raise ValueError(
            f"cell_id must be in [0, {TOTAL_CELLS - 1}], got {cell_id}"
        )
    n_mitigations = len(MITIGATION_COMBOS)  # 4
    n_n = len(N_PER_GROUP)  # 3
    cells_per_model = n_n * n_mitigations  # 12
    cells_per_n = n_mitigations  # 4

    model_idx = cell_id // cells_per_model
    remainder = cell_id % cells_per_model
    n_idx = remainder // cells_per_n
    m_idx = remainder % cells_per_n
    return MODELS[model_idx], N_PER_GROUP[n_idx], MITIGATION_COMBOS[m_idx]


def mitigation_to_yaml_path(combo: str, model_name: str) -> Path:
    """Map mitigation combo + model to the appropriate FitConfig YAML path.

    Parameters
    ----------
    combo : str
        Mitigation combo label from MITIGATION_COMBOS.
    model_name : str
        Model name (``"hgf_2level"`` or ``"hgf_3level"``).

    Returns
    -------
    Path
        Path to the FitConfig YAML file.

    Raises
    ------
    KeyError
        If combo is not in the known mapping.
    """
    level = "2level" if "2level" in model_name else "3level"
    mapping = {
        "hier+M1": f"configs/fit/hier_m1_{level}.yaml",
        "hier+M1+M2": f"configs/fit/hier_m1_m2_{level}.yaml",
        "hier+M1+M2+Laplace": f"configs/fit/hier_m1_m2_laplace_{level}.yaml",
        "hier+M1+M2+Laplace+covariates": (
            f"configs/fit/hier_m1_m2_laplace_cov_{level}.yaml"
        ),
    }
    return Path(mapping[combo])


def _get_git_commit() -> str:
    """Get current git short commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_json(path: Path, data: dict) -> None:
    """Write JSON data to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def _compute_recovery_r(true_arr: np.ndarray, est_arr: np.ndarray) -> float:
    """Compute Pearson r between true and estimated parameter arrays.

    Parameters
    ----------
    true_arr : np.ndarray
        True parameter values, shape (P,).
    est_arr : np.ndarray
        Posterior mean parameter values, shape (P,).

    Returns
    -------
    float
        Pearson correlation coefficient, or NaN if degenerate.
    """
    import numpy as np

    if len(true_arr) < 2:
        return float("nan")
    std_true = np.std(true_arr)
    std_est = np.std(est_arr)
    if std_true < 1e-10 or std_est < 1e-10:
        return float("nan")
    r = float(np.corrcoef(true_arr, est_arr)[0, 1])
    return r


def main() -> None:
    """Execute a single Mode B hierarchical benchmark grid cell."""
    # -----------------------------------------------------------------------
    # Step 1: Parse CLI args
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Phase 34 grid-sweep: run one Mode B hierarchical benchmark cell."
    )
    parser.add_argument(
        "--cell-id",
        type=int,
        required=True,
        help="Cell index [0-23] encoding (model, n_per_group, mitigation).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="local",
        help="SLURM job ID for provenance (optional).",
    )
    args = parser.parse_args()

    cell_id = args.cell_id
    job_id = args.job_id

    # -----------------------------------------------------------------------
    # Step 2: Decode cell
    # -----------------------------------------------------------------------
    model_name, n_per_group, mitigation_combo = decode_cell_id(cell_id)

    # Two groups x 3 sessions for pick_best_cue
    n_groups = 2
    n_sessions = 3
    p_total = n_per_group * n_groups * n_sessions

    # Determine indices for seed computation
    model_idx = MODELS.index(model_name)
    n_idx = N_PER_GROUP.index(n_per_group)

    # Two-level seed strategy (Pitfall 2 prevention):
    # - cohort_seed: deterministic per (model, n_per_group), same across mitigations
    # - mcmc_seed: different per cell
    cohort_seed = _SEED_BASE + n_idx * 1000 + model_idx * 100
    mcmc_seed = _SEED_BASE + cell_id

    has_covariate = "covariates" in mitigation_combo

    print("=" * 60)
    print(f"Phase 34 Grid Sweep (Mode B) — Cell {cell_id:04d}")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  n_per_group: {n_per_group}")
    print(f"  P_total:     {p_total}")
    print(f"  Mitigation:  {mitigation_combo}")
    print(f"  Has cov:     {has_covariate}")
    print(f"  Cohort seed: {cohort_seed}")
    print(f"  MCMC seed:   {mcmc_seed}")
    print(f"  Job ID:      {job_id}")
    print("=" * 60)
    print(flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Import JAX-dependent modules (no fp64 in Mode B)
    # -----------------------------------------------------------------------
    import arviz as az
    import numpy as np

    from prl_hgf.fitting.config import FitConfig
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.simulation.hierarchical import simulate_hierarchical_cohort

    # -----------------------------------------------------------------------
    # SIGTERM handler: write partial result on SLURM wall-time pre-kill
    # -----------------------------------------------------------------------
    t_start = time.perf_counter()
    result_path = RESULTS_DIR / f"cell_{cell_id:04d}.json"
    yaml_path = mitigation_to_yaml_path(mitigation_combo, model_name)

    def _sigterm_handler(signum: int, frame: object) -> None:
        """Write TIMEOUT result on SIGTERM from SLURM."""
        walltime = time.perf_counter() - t_start
        timeout_result = {
            "cell_id": cell_id,
            "model": model_name,
            "n_per_group": n_per_group,
            "p_total": p_total,
            "mitigation_combo": mitigation_combo,
            "has_covariate": has_covariate,
            "cohort_seed": cohort_seed,
            "mcmc_seed": mcmc_seed,
            "status": "TIMEOUT",
            "walltime_s": round(walltime, 2),
            "rhat_max": None,
            "ess_min": None,
            "divergent_rate": None,
            "fit_config_yaml": str(yaml_path),
            "job_id": job_id,
            "commit": _get_git_commit(),
        }
        _write_json(result_path, timeout_result)
        print(
            f"\n[SIGTERM] Wrote TIMEOUT result after {walltime:.1f}s",
            flush=True,
        )
        sys.exit(143)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # -----------------------------------------------------------------------
    # Step 4: Load FitConfig
    # -----------------------------------------------------------------------
    fit_config = FitConfig.from_yaml(yaml_path)
    print(f"Loaded FitConfig: {yaml_path}")
    print(
        f"  n_draws={fit_config.sampler.n_draws}, "
        f"mass={fit_config.mitigation.mass_matrix_kind}, "
        f"laplace={fit_config.mitigation.use_laplace_warmup}, "
        f"non_centered={fit_config.mitigation.non_centered}, "
        f"pooling={fit_config.covariate.pooling}"
    )
    print(flush=True)

    # -----------------------------------------------------------------------
    # Step 5: Write "started" marker
    # -----------------------------------------------------------------------
    started_path = RESULTS_DIR / f"cell_{cell_id:04d}_started.json"
    started_marker = {
        "cell_id": cell_id,
        "model": model_name,
        "n_per_group": n_per_group,
        "p_total": p_total,
        "mitigation_combo": mitigation_combo,
        "has_covariate": has_covariate,
        "cohort_seed": cohort_seed,
        "mcmc_seed": mcmc_seed,
        "fit_config_yaml": str(yaml_path),
        "job_id": job_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(started_path, started_marker)
    print(f"Wrote started marker: {started_path}")

    # -----------------------------------------------------------------------
    # Step 6: Build true hyperparameters
    # -----------------------------------------------------------------------
    true_mu: dict[str, np.ndarray] = {
        "omega_2": np.array([-3.0, -2.5]),
        "log_beta": np.array([0.0, 0.5]),
        "zeta": np.array([0.0, 0.1]),
    }
    true_sigma: dict[str, float] = {"omega_2": 0.5, "log_beta": 0.5, "zeta": 0.5}
    true_beta: dict[str, float] | None = (
        {"omega_2": 0.8, "log_beta": 0.3, "zeta": 0.2} if has_covariate else None
    )

    if model_name == "hgf_3level":
        true_mu["omega_3"] = np.array([-6.0, -5.5])
        true_sigma["omega_3"] = 0.5
        if has_covariate:
            assert true_beta is not None  # type: ignore[unreachable]
            true_beta["omega_3"] = 0.4

    # -----------------------------------------------------------------------
    # Step 7: Build group assignment and optional covariate
    # -----------------------------------------------------------------------
    # Balanced groups: n_per_group participants per group, n_sessions sessions
    # Group assignment repeats by session: [0,0,...,1,1,...] per session block
    group_idx_session = np.repeat(np.arange(n_groups), n_per_group)
    group_idx = np.tile(group_idx_session, n_sessions)

    # Covariate: generated with cohort_seed so identical across mitigations
    x_covariate: np.ndarray | None = None
    if has_covariate:
        rng_cov = np.random.default_rng(cohort_seed)
        x_covariate = rng_cov.standard_normal(p_total)

    # -----------------------------------------------------------------------
    # Step 8: Simulate hierarchical cohort
    # -----------------------------------------------------------------------
    print(f"\nSimulating hierarchical cohort (n_per_group={n_per_group})...", flush=True)
    sim_df, true_params = simulate_hierarchical_cohort(
        n_participants=p_total,
        n_groups=n_groups,
        true_mu=true_mu,
        true_sigma=true_sigma,
        true_beta=true_beta,
        x_covariate=x_covariate,
        group_idx=group_idx,
        model_name=model_name,
        task_config="pick_best_cue",
        seed=cohort_seed,
    )
    sim_time = time.perf_counter() - t_start
    print(f"  Simulation done in {sim_time:.1f}s ({len(sim_df)} trial rows)", flush=True)

    # -----------------------------------------------------------------------
    # Step 9: Fit
    # -----------------------------------------------------------------------
    print(f"\nFitting (cell {cell_id:04d})...", flush=True)
    t_fit_start = time.perf_counter()

    try:
        idata = fit_batch_hierarchical(
            sim_df,
            fit_config=fit_config,
            x_covariate=x_covariate,
        )

        # If returned as tuple (idata, adapted_params), extract idata
        if isinstance(idata, tuple):
            idata, _ = idata

        fit_time = time.perf_counter() - t_fit_start
        total_time = time.perf_counter() - t_start

        # -------------------------------------------------------------------
        # Step 10: Extract diagnostics
        # -------------------------------------------------------------------
        summary = az.summary(idata)
        rhat_max = float(summary["r_hat"].max())
        ess_min = float(summary["ess_bulk"].min())

        # Divergences
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            div_array = idata.sample_stats["diverging"].values
            n_divergent = int(np.sum(div_array))
            n_total_samples = div_array.size
            divergent_rate = (
                n_divergent / n_total_samples if n_total_samples > 0 else 0.0
            )
        else:
            n_divergent = 0
            n_total_samples = 0
            divergent_rate = 0.0

        # -------------------------------------------------------------------
        # Step 11: Recovery metrics
        # -------------------------------------------------------------------
        # Parameters to recover (subset to those actually in model)
        if model_name == "hgf_3level":
            h_params = ["omega_2", "log_beta", "zeta", "omega_3"]
        else:
            h_params = ["omega_2", "log_beta", "zeta"]

        # Individual-level recovery: r(true_theta_k, posterior_mean_theta_k)
        recovery_r_individual: dict[str, float] = {}
        for p_name in h_params:
            true_key = f"true_{p_name}"
            if true_key in true_params and hasattr(idata, "posterior"):
                true_arr = true_params[p_name]
                # Try to extract per-participant posterior means
                # Variable names in idata may be e.g. omega_2_k
                var_candidates = [
                    p_name,
                    f"{p_name}_k",
                    f"{p_name}_participant",
                ]
                post_mean = None
                for var in var_candidates:
                    if var in idata.posterior:
                        post_mean = (
                            idata.posterior[var]
                            .mean(dim=("chain", "draw"))
                            .values
                        )
                        break
                if post_mean is not None and post_mean.shape == true_arr.shape:
                    recovery_r_individual[p_name] = _compute_recovery_r(
                        true_arr, post_mean
                    )

        # Group-level mean recovery: r(true_mu_g, posterior_mean_mu_g)
        recovery_r_mu: dict[str, float] = {}
        for p_name in h_params:
            true_mu_arr = true_mu[p_name]  # shape (n_groups,)
            # Variable names may be e.g. mu_omega_2, mu_omega_2_g
            mu_candidates = [
                f"mu_{p_name}",
                f"mu_{p_name}_g",
                f"group_mu_{p_name}",
            ]
            post_mean_mu = None
            if hasattr(idata, "posterior"):
                for var in mu_candidates:
                    if var in idata.posterior:
                        post_mean_mu = (
                            idata.posterior[var]
                            .mean(dim=("chain", "draw"))
                            .values
                        )
                        break
            if post_mean_mu is not None and post_mean_mu.shape == true_mu_arr.shape:
                recovery_r_mu[p_name] = _compute_recovery_r(
                    true_mu_arr, post_mean_mu
                )

        # Sigma recovery: relative error |est - true| / true
        recovery_sigma: dict[str, dict] = {}
        for p_name in h_params:
            true_s = true_sigma[p_name]
            sigma_candidates = [
                f"sigma_{p_name}",
                f"sigma_{p_name}_g",
                f"log_sigma_{p_name}",
            ]
            post_mean_sigma = None
            if hasattr(idata, "posterior"):
                for var in sigma_candidates:
                    if var in idata.posterior:
                        raw = (
                            idata.posterior[var]
                            .mean(dim=("chain", "draw"))
                            .values
                        )
                        # If log-space, exp-transform
                        if "log_sigma" in var:
                            raw = float(np.exp(raw))
                        else:
                            raw = float(raw) if raw.ndim == 0 else float(np.mean(raw))
                        post_mean_sigma = raw
                        break
            if post_mean_sigma is not None:
                rel_err = abs(post_mean_sigma - true_s) / true_s if true_s > 0 else float("nan")
                recovery_sigma[p_name] = {
                    "true": true_s,
                    "posterior_mean": post_mean_sigma,
                    "relative_error": rel_err,
                }

        # Covariate beta recovery (only for covariate cells)
        recovery_beta: dict[str, dict] | None = None
        if has_covariate and true_beta is not None:
            recovery_beta = {}
            for p_name in h_params:
                true_b = true_beta.get(p_name)
                if true_b is None:
                    continue
                beta_candidates = [
                    f"beta_{p_name}",
                    f"beta_x_{p_name}",
                    f"beta_p_{p_name}",
                ]
                post_mean_beta = None
                if hasattr(idata, "posterior"):
                    for var in beta_candidates:
                        if var in idata.posterior:
                            post_mean_beta = float(
                                idata.posterior[var]
                                .mean(dim=("chain", "draw"))
                                .values
                            )
                            break
                if post_mean_beta is not None:
                    recovery_beta[p_name] = {
                        "true": true_b,
                        "posterior_mean": post_mean_beta,
                    }

        # -------------------------------------------------------------------
        # Step 12: Determine status
        # -------------------------------------------------------------------
        if rhat_max <= 1.05 and ess_min >= 400 and divergent_rate < 0.05:
            status = "PASS"
        else:
            status = "INVALID"

        result: dict = {
            "cell_id": cell_id,
            "model": model_name,
            "n_per_group": n_per_group,
            "p_total": p_total,
            "mitigation_combo": mitigation_combo,
            "has_covariate": has_covariate,
            "cohort_seed": cohort_seed,
            "mcmc_seed": mcmc_seed,
            "status": status,
            "walltime_s": round(total_time, 2),
            "fit_time_s": round(fit_time, 2),
            "rhat_max": round(rhat_max, 4),
            "ess_min": round(ess_min, 1),
            "divergent_rate": round(divergent_rate, 4),
            "n_divergent": n_divergent,
            "n_total_samples": n_total_samples,
            "recovery_r_individual": recovery_r_individual,
            "recovery_r_mu": recovery_r_mu,
            "recovery_sigma": recovery_sigma,
            "fit_config_yaml": str(yaml_path),
            "job_id": job_id,
            "commit": _get_git_commit(),
        }
        if recovery_beta is not None:
            result["recovery_beta"] = recovery_beta

        print(f"\n{'=' * 60}")
        print(f"RESULT: {status}")
        print(f"  Walltime:    {total_time:.1f}s (fit: {fit_time:.1f}s)")
        print(f"  R-hat max:   {rhat_max:.4f}")
        print(f"  ESS min:     {ess_min:.1f}")
        print(f"  Div rate:    {divergent_rate:.4f} ({n_divergent}/{n_total_samples})")
        if recovery_r_individual:
            for pn, rv in recovery_r_individual.items():
                print(f"  r(indiv) {pn}: {rv:.3f}")
        if recovery_r_mu:
            for pn, rv in recovery_r_mu.items():
                print(f"  r(mu) {pn}: {rv:.3f}")
        print(f"{'=' * 60}", flush=True)

    except Exception as exc:
        # -------------------------------------------------------------------
        # CRASH handler
        # -------------------------------------------------------------------
        total_time = time.perf_counter() - t_start
        result = {
            "cell_id": cell_id,
            "model": model_name,
            "n_per_group": n_per_group,
            "p_total": p_total,
            "mitigation_combo": mitigation_combo,
            "has_covariate": has_covariate,
            "cohort_seed": cohort_seed,
            "mcmc_seed": mcmc_seed,
            "status": "CRASH",
            "walltime_s": round(total_time, 2),
            "rhat_max": None,
            "ess_min": None,
            "divergent_rate": None,
            "recovery_r_individual": None,
            "recovery_r_mu": None,
            "recovery_sigma": None,
            "fit_config_yaml": str(yaml_path),
            "job_id": job_id,
            "commit": _get_git_commit(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"\n{'=' * 60}")
        print(f"CRASH after {total_time:.1f}s: {exc}")
        print(traceback.format_exc())
        print(f"{'=' * 60}", flush=True)

    # -----------------------------------------------------------------------
    # Step 13: Write result JSON
    # -----------------------------------------------------------------------
    _write_json(result_path, result)
    print(f"\nResult written: {result_path}", flush=True)


if __name__ == "__main__":
    main()
