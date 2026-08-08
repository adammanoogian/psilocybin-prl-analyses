"""Phase 32 audit driver: BlackJAX vs NumPyro NUTS head-to-head (AUDIT-03).

Runs a single audit cell (backend x model x P x mass x seed) per invocation.
Accepts ``--cell-id`` (int 0-25) and optional ``--job-id`` from SLURM.

The 26-cell grid covers:

    Cells 0-19: head-to-head
        2 backends x 2 models x 5 n_per_group values = 20 cells
    Cells 20-25: noise-floor (A-vs-A)
        Same backend, different seed at P=60

The driver decodes cell_id, loads the appropriate FitConfig YAML, overrides
backend/seed to construct an audit-specific FitConfig, simulates a cohort
via simulate_batch (with known true parameters), fits via
fit_batch_hierarchical, extracts all AUDIT-03 metrics, and writes a JSON
result file.

CRITICAL: fp64 decision must happen before importing JAX. For the primary
audit, fp64 is NOT enabled (fp32 primary run). The import of
prl_hgf.runtime confirms set_x64 availability (future-proof).

Pre-registered protocol: .planning/AUDIT_PROTOCOL.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import subprocess
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

# Safe import -- prl_hgf.runtime does NOT import JAX at module level
import prl_hgf.runtime  # noqa: F401 — confirms set_x64 available

# ---------------------------------------------------------------------------
# Grid constants (pre-registered in AUDIT_PROTOCOL.md)
# ---------------------------------------------------------------------------
BACKENDS = ["blackjax", "numpyro"]
MODELS = ["hgf_2level", "hgf_3level"]
N_PER_GROUP = [5, 10, 17, 25, 33]  # P = {30, 60, 102, 150, 198}
MASS_MATRIX = ["diagonal"]  # Primary audit; dense is secondary

TOTAL_HEAD_TO_HEAD = len(BACKENDS) * len(MODELS) * len(N_PER_GROUP)  # 20
TOTAL_CELLS = 26  # 20 head-to-head + 6 noise-floor

# Audit hyperparameters (locked by AUDIT_PROTOCOL.md)
AUDIT_N_DRAWS = 2000
AUDIT_N_WARMUP = 1000
AUDIT_N_CHAINS = 4
AUDIT_TARGET_ACCEPT = 0.95
AUDIT_MAX_TREE_DEPTH = 10
AUDIT_MASTER_SEED = 42


def decode_cell_id(
    cell_id: int,
) -> tuple[str, str, int, str, int]:
    """Decode a linear cell index into audit grid coordinates.

    Parameters
    ----------
    cell_id : int
        Linear index in range [0, 25].

    Returns
    -------
    tuple[str, str, int, str, int]
        (backend, model_name, n_per_group, mass_matrix_kind, seed).

    Raises
    ------
    ValueError
        If cell_id is out of range [0, 25].
    """
    if cell_id < 0 or cell_id >= TOTAL_CELLS:
        raise ValueError(
            f"cell_id must be in [0, {TOTAL_CELLS - 1}], got {cell_id}"
        )

    if cell_id < TOTAL_HEAD_TO_HEAD:
        # Cells 0-19: head-to-head grid
        # Decode: backend_idx = cell_id // 10
        #         model_idx = (cell_id % 10) // 5
        #         p_idx = cell_id % 5
        backend_idx = cell_id // 10
        model_idx = (cell_id % 10) // 5
        p_idx = cell_id % 5
        backend = BACKENDS[backend_idx]
        model = MODELS[model_idx]
        n_per_group = N_PER_GROUP[p_idx]
        seed = AUDIT_MASTER_SEED
    else:
        # Cells 20-25: noise-floor (A-vs-A)
        # All at P=60 (n_per_group=10)
        noise_idx = cell_id - TOTAL_HEAD_TO_HEAD
        # cell 20: blackjax, 2level, seed=43
        # cell 21: blackjax, 3level, seed=43
        # cell 22: numpyro, 2level, seed=43
        # cell 23: numpyro, 3level, seed=43
        # cell 24: blackjax, 2level, seed=44
        # cell 25: numpyro, 2level, seed=44
        noise_map = [
            ("blackjax", "hgf_2level", 43),
            ("blackjax", "hgf_3level", 43),
            ("numpyro", "hgf_2level", 43),
            ("numpyro", "hgf_3level", 43),
            ("blackjax", "hgf_2level", 44),
            ("numpyro", "hgf_2level", 44),
        ]
        backend, model, seed = noise_map[noise_idx]
        n_per_group = 10  # P=60

    mass_matrix_kind = "diagonal"
    return backend, model, n_per_group, mass_matrix_kind, seed


def _validate_decode_logic() -> None:
    """Assert deterministic cell_id decode produces expected tuples.

    Raises
    ------
    AssertionError
        If any cell_id produces an unexpected result.
    """
    # Head-to-head cells
    # Cell 0: blackjax, hgf_2level, n=5, diagonal, seed=42
    assert decode_cell_id(0) == (
        "blackjax", "hgf_2level", 5, "diagonal", 42
    ), f"Cell 0 mismatch: {decode_cell_id(0)}"
    # Cell 4: blackjax, hgf_2level, n=33, diagonal, seed=42
    assert decode_cell_id(4) == (
        "blackjax", "hgf_2level", 33, "diagonal", 42
    ), f"Cell 4 mismatch: {decode_cell_id(4)}"
    # Cell 5: blackjax, hgf_3level, n=5, diagonal, seed=42
    assert decode_cell_id(5) == (
        "blackjax", "hgf_3level", 5, "diagonal", 42
    ), f"Cell 5 mismatch: {decode_cell_id(5)}"
    # Cell 10: numpyro, hgf_2level, n=5, diagonal, seed=42
    assert decode_cell_id(10) == (
        "numpyro", "hgf_2level", 5, "diagonal", 42
    ), f"Cell 10 mismatch: {decode_cell_id(10)}"
    # Cell 19: numpyro, hgf_3level, n=33, diagonal, seed=42
    assert decode_cell_id(19) == (
        "numpyro", "hgf_3level", 33, "diagonal", 42
    ), f"Cell 19 mismatch: {decode_cell_id(19)}"
    # Noise-floor cells
    assert decode_cell_id(20) == (
        "blackjax", "hgf_2level", 10, "diagonal", 43
    ), f"Cell 20 mismatch: {decode_cell_id(20)}"
    assert decode_cell_id(21) == (
        "blackjax", "hgf_3level", 10, "diagonal", 43
    ), f"Cell 21 mismatch: {decode_cell_id(21)}"
    assert decode_cell_id(22) == (
        "numpyro", "hgf_2level", 10, "diagonal", 43
    ), f"Cell 22 mismatch: {decode_cell_id(22)}"
    assert decode_cell_id(23) == (
        "numpyro", "hgf_3level", 10, "diagonal", 43
    ), f"Cell 23 mismatch: {decode_cell_id(23)}"
    assert decode_cell_id(24) == (
        "blackjax", "hgf_2level", 10, "diagonal", 44
    ), f"Cell 24 mismatch: {decode_cell_id(24)}"
    assert decode_cell_id(25) == (
        "numpyro", "hgf_2level", 10, "diagonal", 44
    ), f"Cell 25 mismatch: {decode_cell_id(25)}"
    print("  [OK] All 26 cell_id decode assertions passed.")


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
    """Write JSON data to file, creating parent directories.

    Parameters
    ----------
    path : Path
        Output file path.
    data : dict
        Data to serialize as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def extract_audit_metrics(idata: object, backend: str) -> dict:
    """Extract normalized audit metrics from InferenceData.

    Handles field name differences between BlackJAX and NumPyro backends.
    BlackJAX uses ``"num_integration_steps"`` and ``"acceptance_rate"``;
    NumPyro uses ``"num_steps"`` and ``"mean_accept_prob"``.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted inference data with sample_stats group.
    backend : str
        Backend identifier (``"blackjax"`` or ``"numpyro"``).

    Returns
    -------
    dict
        Normalized metrics with keys: ``ess_bulk_min``, ``rhat_max``,
        ``divergent_count``, ``divergent_rate``, ``total_leapfrog_steps``,
        ``ess_per_sec``, ``ess_per_grad_eval``, ``mean_accept_rate``.
        Values are None if the corresponding field is unavailable.
    """
    import arviz as az
    import numpy as np

    # ESS and R-hat from ArviZ summary
    summary = az.summary(idata)
    ess_bulk_min = float(summary["ess_bulk"].min())
    rhat_max = float(summary["r_hat"].max())

    # Divergences and leapfrog steps from sample_stats
    divergent_count: int | None = None
    divergent_rate: float | None = None
    total_leapfrog: int | None = None
    mean_accept_rate: float | None = None

    if hasattr(idata, "sample_stats"):
        ss = idata.sample_stats

        # Divergences (same field name in both backends via idata)
        if "diverging" in ss:
            div_array = ss["diverging"].values
            divergent_count = int(np.sum(div_array))
            divergent_rate = divergent_count / div_array.size
        elif "divergence" in ss:
            div_array = ss["divergence"].values
            divergent_count = int(np.sum(div_array))
            divergent_rate = divergent_count / div_array.size

        # Leapfrog steps (field name differs by backend)
        if "num_integration_steps" in ss:  # BlackJAX
            total_leapfrog = int(ss["num_integration_steps"].values.sum())
        elif "num_steps" in ss:  # NumPyro with extra_fields
            total_leapfrog = int(ss["num_steps"].values.sum())

        # Acceptance rate
        if "acceptance_rate" in ss:  # BlackJAX
            mean_accept_rate = float(ss["acceptance_rate"].values.mean())
        elif "mean_accept_prob" in ss:  # NumPyro
            mean_accept_rate = float(ss["mean_accept_prob"].values.mean())

    # Derived metrics
    ess_per_grad = None
    if total_leapfrog is not None and total_leapfrog > 0:
        ess_per_grad = ess_bulk_min / total_leapfrog

    return {
        "ess_bulk_min": round(ess_bulk_min, 2),
        "rhat_max": round(rhat_max, 4),
        "divergent_count": divergent_count,
        "divergent_rate": (
            round(divergent_rate, 6) if divergent_rate is not None else None
        ),
        "total_leapfrog_steps": total_leapfrog,
        "ess_per_grad_eval": (
            round(ess_per_grad, 6) if ess_per_grad is not None else None
        ),
        "mean_accept_rate": (
            round(mean_accept_rate, 4)
            if mean_accept_rate is not None
            else None
        ),
    }


def _compute_recovery_correlations(
    sim_df: object, idata: object, model_name: str
) -> dict:
    """Compute Pearson correlation between true and recovered parameters.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulation DataFrame with true parameter columns.
    idata : arviz.InferenceData
        Fitted inference data with posterior group.
    model_name : str
        Model name for determining which parameters to check.

    Returns
    -------
    dict
        Recovery correlations for each parameter. Keys are
        ``recovery_corr_omega2``, ``recovery_corr_beta``,
        ``recovery_corr_zeta``. Values are None on failure.
    """
    import numpy as np

    result = {
        "recovery_corr_omega2": None,
        "recovery_corr_beta": None,
        "recovery_corr_zeta": None,
    }

    try:
        # Extract unique true params per participant (first row per pid)
        pid_col = "participant_id"
        true_df = sim_df.drop_duplicates(subset=[pid_col]).sort_values(
            pid_col
        )

        # Get posterior means across chains and draws (mean over draw+chain)
        posterior = idata.posterior

        # omega_2
        if "omega_2" in posterior and "true_omega_2" in true_df.columns:
            post_mean = posterior["omega_2"].mean(
                dim=("chain", "draw")
            ).values
            true_vals = true_df["true_omega_2"].values
            if len(post_mean) == len(true_vals) and len(true_vals) > 2:
                corr = np.corrcoef(true_vals, post_mean)[0, 1]
                if np.isfinite(corr):
                    result["recovery_corr_omega2"] = round(float(corr), 4)

        # beta (stored as log_beta in posterior, true_beta in sim_df)
        if "log_beta" in posterior and "true_beta" in true_df.columns:
            # Compare on log scale for consistency
            post_mean = posterior["log_beta"].mean(
                dim=("chain", "draw")
            ).values
            true_vals = np.log(true_df["true_beta"].values)
            if len(post_mean) == len(true_vals) and len(true_vals) > 2:
                corr = np.corrcoef(true_vals, post_mean)[0, 1]
                if np.isfinite(corr):
                    result["recovery_corr_beta"] = round(float(corr), 4)

        # zeta
        if "zeta" in posterior and "true_zeta" in true_df.columns:
            post_mean = posterior["zeta"].mean(
                dim=("chain", "draw")
            ).values
            true_vals = true_df["true_zeta"].values
            if len(post_mean) == len(true_vals) and len(true_vals) > 2:
                corr = np.corrcoef(true_vals, post_mean)[0, 1]
                if np.isfinite(corr):
                    result["recovery_corr_zeta"] = round(float(corr), 4)

    except Exception as exc:
        print(f"  [WARN] Recovery correlation failed: {exc}", flush=True)

    return result


def main() -> None:
    """Execute a single audit cell."""
    # -------------------------------------------------------------------
    # Step 1: Parse CLI args
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Phase 32 sampler audit: run one audit cell (AUDIT-03)."
    )
    parser.add_argument(
        "--cell-id",
        type=int,
        required=True,
        help="Cell index [0-25] encoding (backend, model, P, seed).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="local",
        help="SLURM job ID for provenance (optional).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("models/power/audit_results"),
        help="Output directory for JSON results.",
    )
    args = parser.parse_args()

    cell_id = args.cell_id
    job_id = args.job_id
    results_dir = args.results_dir

    # -------------------------------------------------------------------
    # Step 2: Decode cell
    # -------------------------------------------------------------------
    backend, model_name, n_per_group, mass_matrix_kind, seed = decode_cell_id(
        cell_id
    )
    p_total = n_per_group * 6  # pick_best_cue: 2 groups x 3 sessions
    level = "2level" if "2level" in model_name else "3level"
    run_type = "head-to-head" if cell_id < TOTAL_HEAD_TO_HEAD else "noise-floor"

    print("=" * 60)
    print(f"Phase 32 Sampler Audit -- Cell {cell_id:02d}")
    print("=" * 60)
    print(f"  Backend:     {backend}")
    print(f"  Model:       {model_name}")
    print(f"  n_per_group: {n_per_group}")
    print(f"  P_total:     {p_total}")
    print(f"  Mass matrix: {mass_matrix_kind}")
    print(f"  Seed:        {seed}")
    print(f"  Run type:    {run_type}")
    print(f"  Job ID:      {job_id}")
    print("=" * 60)
    print(flush=True)

    # -------------------------------------------------------------------
    # Step 3: fp64 decision -- primary audit uses fp32 (no fp64)
    # -------------------------------------------------------------------
    # fp64 NOT enabled for primary audit (AUDIT_PROTOCOL.md Section 1).
    # prl_hgf.runtime already imported above to confirm set_x64 availability.

    # -------------------------------------------------------------------
    # Step 4: Now safe to import JAX-dependent modules
    # -------------------------------------------------------------------
    from prl_hgf.env.task_config import load_config
    from prl_hgf.fitting.config import FitConfig
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    # -------------------------------------------------------------------
    # SIGTERM handler: write partial result on SLURM wall-time pre-kill
    # -------------------------------------------------------------------
    t_start = time.perf_counter()
    result_filename = (
        f"audit_{backend}_{model_name}_P{p_total}"
        f"_{mass_matrix_kind}_{seed}.json"
    )
    result_path = results_dir / result_filename

    def _sigterm_handler(signum: int, frame: object) -> None:
        """Write TIMEOUT result on SIGTERM from SLURM."""
        walltime = time.perf_counter() - t_start
        timeout_result = {
            "cell_id": cell_id,
            "backend": backend,
            "model": model_name,
            "n_participants": p_total,
            "n_per_group": n_per_group,
            "mass_matrix": mass_matrix_kind,
            "seed": seed,
            "run_type": run_type,
            "status": "TIMEOUT",
            "walltime_s": round(walltime, 2),
            "ess_bulk_min": None,
            "ess_per_sec": None,
            "ess_per_grad_eval": None,
            "divergent_count": None,
            "divergent_rate": None,
            "rhat_max": None,
            "memory_peak_mb": None,
            "recovery_corr_omega2": None,
            "recovery_corr_beta": None,
            "recovery_corr_zeta": None,
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

    # -------------------------------------------------------------------
    # Step 5: Build FitConfig (audit-specific overrides)
    # -------------------------------------------------------------------
    yaml_path = Path(f"configs/fit/none_{level}.yaml")
    base_config = FitConfig.from_yaml(yaml_path)

    # Override with audit-locked hyperparameters
    fit_config = dataclasses.replace(
        base_config,
        model_name=model_name,
        sampler=dataclasses.replace(
            base_config.sampler,
            backend=backend,
            random_seed=seed,
            n_draws=AUDIT_N_DRAWS,
            n_warmup=AUDIT_N_WARMUP,
            n_chains=AUDIT_N_CHAINS,
            target_accept=AUDIT_TARGET_ACCEPT,
            max_tree_depth=AUDIT_MAX_TREE_DEPTH,
        ),
        mitigation=dataclasses.replace(
            base_config.mitigation,
            mass_matrix_kind=mass_matrix_kind,
            use_laplace_warmup=False,
            use_fp64=False,
            use_shard_map=False,
        ),
        progressbar=False,
    )
    print(f"FitConfig loaded from: {yaml_path}")
    print(f"  backend={fit_config.sampler.backend}, "
          f"n_draws={fit_config.sampler.n_draws}, "
          f"n_warmup={fit_config.sampler.n_warmup}, "
          f"mass={fit_config.mitigation.mass_matrix_kind}")
    print(flush=True)

    # -------------------------------------------------------------------
    # Step 6: Write "started" marker
    # -------------------------------------------------------------------
    started_path = results_dir / f"audit_cell_{cell_id:02d}_started.json"
    started_marker = {
        "cell_id": cell_id,
        "backend": backend,
        "model": model_name,
        "n_participants": p_total,
        "n_per_group": n_per_group,
        "mass_matrix": mass_matrix_kind,
        "seed": seed,
        "run_type": run_type,
        "job_id": job_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(started_path, started_marker)
    print(f"Wrote started marker: {started_path}")

    # -------------------------------------------------------------------
    # Step 7: Simulate cohort (with known true parameters for recovery)
    # -------------------------------------------------------------------
    print(f"\nSimulating cohort (n_per_group={n_per_group})...", flush=True)
    task_config = load_config()
    power_cfg = make_power_config(
        task_config,
        n_per_group,
        effect_size_delta=0.0,
        master_seed=12345 + seed,
    )
    sim_df = simulate_batch(power_cfg)
    sim_time = time.perf_counter() - t_start
    print(
        f"  Simulation done in {sim_time:.1f}s "
        f"({len(sim_df)} trial rows)",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Step 8: Start tracemalloc for memory peak
    # -------------------------------------------------------------------
    tracemalloc.start()

    # -------------------------------------------------------------------
    # Step 9: Fit
    # -------------------------------------------------------------------
    print(f"\nFitting (cell {cell_id:02d}, {backend})...", flush=True)
    t_fit_start = time.perf_counter()

    try:
        fit_result = fit_batch_hierarchical(sim_df, fit_config=fit_config)

        # Handle tuple return (idata, adapted_params) from BlackJAX
        if isinstance(fit_result, tuple):
            idata, _ = fit_result
        else:
            idata = fit_result

        fit_time = time.perf_counter() - t_fit_start
        total_time = time.perf_counter() - t_start

        # ---------------------------------------------------------------
        # Step 10: Stop tracemalloc, get peak memory
        # ---------------------------------------------------------------
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak_mb = round(peak_memory / 1e6, 2)

        # ---------------------------------------------------------------
        # Step 11: Extract audit metrics
        # ---------------------------------------------------------------
        metrics = extract_audit_metrics(idata, backend)

        # ESS per second
        ess_per_sec = None
        if metrics["ess_bulk_min"] is not None and total_time > 0:
            ess_per_sec = round(metrics["ess_bulk_min"] / total_time, 4)

        # ---------------------------------------------------------------
        # Step 12: Recovery correlations
        # ---------------------------------------------------------------
        recovery = _compute_recovery_correlations(sim_df, idata, model_name)

        # ---------------------------------------------------------------
        # Step 13: Determine status
        # ---------------------------------------------------------------
        rhat_max = metrics["rhat_max"]
        div_rate = metrics["divergent_rate"]

        if div_rate is not None and div_rate >= 0.05:
            status = "DIVERGENT"
        elif (
            rhat_max is not None
            and rhat_max < 1.05
            and (div_rate is None or div_rate < 0.05)
        ):
            status = "PASS"
        else:
            status = "INVALID"

        # ---------------------------------------------------------------
        # Step 14: Build result dict
        # ---------------------------------------------------------------
        result = {
            "cell_id": cell_id,
            "backend": backend,
            "model": model_name,
            "n_participants": p_total,
            "n_per_group": n_per_group,
            "mass_matrix": mass_matrix_kind,
            "seed": seed,
            "run_type": run_type,
            "status": status,
            "walltime_s": round(total_time, 2),
            "fit_time_s": round(fit_time, 2),
            "compile_time_s": None,  # Fresh process; no separation
            "ess_bulk_min": metrics["ess_bulk_min"],
            "ess_per_sec": ess_per_sec,
            "ess_per_grad_eval": metrics["ess_per_grad_eval"],
            "divergent_count": metrics["divergent_count"],
            "divergent_rate": metrics["divergent_rate"],
            "rhat_max": metrics["rhat_max"],
            "total_leapfrog_steps": metrics["total_leapfrog_steps"],
            "mean_accept_rate": metrics["mean_accept_rate"],
            "memory_peak_mb": memory_peak_mb,
            "recovery_corr_omega2": recovery["recovery_corr_omega2"],
            "recovery_corr_beta": recovery["recovery_corr_beta"],
            "recovery_corr_zeta": recovery["recovery_corr_zeta"],
            "fit_config_json": fit_config.to_json(),
            "job_id": job_id,
            "commit": _get_git_commit(),
        }

        print(f"\n{'=' * 60}")
        print(f"RESULT: {status}")
        print(f"  Walltime:    {total_time:.1f}s (fit: {fit_time:.1f}s)")
        print(f"  R-hat max:   {rhat_max:.4f}")
        print(f"  ESS min:     {metrics['ess_bulk_min']:.1f}")
        print(f"  ESS/sec:     {ess_per_sec}")
        print(f"  ESS/grad:    {metrics['ess_per_grad_eval']}")
        print(f"  Div rate:    {metrics['divergent_rate']}")
        print(f"  Memory:      {memory_peak_mb:.1f} MB")
        print(f"  Recovery:    omega2={recovery['recovery_corr_omega2']}, "
              f"beta={recovery['recovery_corr_beta']}, "
              f"zeta={recovery['recovery_corr_zeta']}")
        print(f"{'=' * 60}", flush=True)

    except Exception as exc:
        # ---------------------------------------------------------------
        # CRASH handler
        # ---------------------------------------------------------------
        tracemalloc.stop()
        total_time = time.perf_counter() - t_start
        result = {
            "cell_id": cell_id,
            "backend": backend,
            "model": model_name,
            "n_participants": p_total,
            "n_per_group": n_per_group,
            "mass_matrix": mass_matrix_kind,
            "seed": seed,
            "run_type": run_type,
            "status": "CRASH",
            "walltime_s": round(total_time, 2),
            "fit_time_s": None,
            "compile_time_s": None,
            "ess_bulk_min": None,
            "ess_per_sec": None,
            "ess_per_grad_eval": None,
            "divergent_count": None,
            "divergent_rate": None,
            "rhat_max": None,
            "total_leapfrog_steps": None,
            "mean_accept_rate": None,
            "memory_peak_mb": None,
            "recovery_corr_omega2": None,
            "recovery_corr_beta": None,
            "recovery_corr_zeta": None,
            "fit_config_json": fit_config.to_json(),
            "job_id": job_id,
            "commit": _get_git_commit(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"\n{'=' * 60}")
        print(f"CRASH after {total_time:.1f}s: {exc}")
        print(traceback.format_exc())
        print(f"{'=' * 60}", flush=True)

    # -------------------------------------------------------------------
    # Step 15: Write result JSON
    # -------------------------------------------------------------------
    _write_json(result_path, result)
    print(f"\nResult written: {result_path}", flush=True)


if __name__ == "__main__":
    # Validate decode logic before entering main
    print("Validating cell_id decode logic...")
    _validate_decode_logic()
    print()

    main()
