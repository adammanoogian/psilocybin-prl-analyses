"""Phase 31 grid-sweep driver: benchmark no-pooling mode (Mode A).

Runs a single grid cell (model x n_per_group x mitigation_combo) per
invocation.  Accepts ``--cell-id`` (int 0-47) and optional ``--job-id``
from SLURM.  The 48-cell grid covers:

    2 models x 6 n_per_group values x 4 mitigation combos = 48 cells.

The driver decodes cell_id, loads the appropriate FitConfig YAML, simulates
a cohort via simulate_batch (with effect_size_delta=0.0 for pure benchmark),
fits via fit_batch_hierarchical, and writes a JSON result file.

CRITICAL: fp64 must be set before importing JAX.  This script conditionally
calls set_x64(True) after decoding cell_id but BEFORE importing jax or
prl_hgf.fitting.
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

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
MODELS = ["hgf_2level", "hgf_3level"]
N_PER_GROUP = [5, 10, 17, 25, 33, 50]
MITIGATION_COMBOS = ["none", "M1", "M1+Laplace", "M1+Laplace+fp64"]

TOTAL_CELLS = len(MODELS) * len(N_PER_GROUP) * len(MITIGATION_COMBOS)  # 48

# Output directory for benchmark results
RESULTS_DIR = Path("models/power/bench_mode_a_results")


def decode_cell_id(cell_id: int) -> tuple[str, int, str]:
    """Decode a linear cell index into grid coordinates.

    Parameters
    ----------
    cell_id : int
        Linear index in range [0, 47].

    Returns
    -------
    tuple[str, int, str]
        (model_name, n_per_group, mitigation_combo).

    Raises
    ------
    ValueError
        If cell_id is out of range [0, 47].
    """
    if cell_id < 0 or cell_id >= TOTAL_CELLS:
        raise ValueError(
            f"cell_id must be in [0, {TOTAL_CELLS - 1}], got {cell_id}"
        )
    model_idx = cell_id // 24
    n_idx = (cell_id % 24) // 4
    m_idx = cell_id % 4
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
    """
    level = "2level" if "2level" in model_name else "3level"
    mapping = {
        "none": f"configs/fit/none_{level}.yaml",
        "M1": f"configs/fit/benchmark_dense_{level}.yaml",
        "M1+Laplace": f"configs/fit/m1_laplace_{level}.yaml",
        "M1+Laplace+fp64": f"configs/fit/m1_laplace_fp64_{level}.yaml",
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


def main() -> None:
    """Execute a single benchmark grid cell."""
    # -----------------------------------------------------------------------
    # Step 1: Parse CLI args
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Phase 31 grid-sweep: run one benchmark cell."
    )
    parser.add_argument(
        "--cell-id",
        type=int,
        required=True,
        help="Cell index [0-47] encoding (model, n_per_group, mitigation).",
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
    p_total = n_per_group * 6  # pick_best_cue: 2 groups x 3 sessions

    print("=" * 60)
    print(f"Phase 31 Grid Sweep — Cell {cell_id:04d}")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  n_per_group: {n_per_group}")
    print(f"  P_total:     {p_total}")
    print(f"  Mitigation:  {mitigation_combo}")
    print(f"  Job ID:      {job_id}")
    print("=" * 60)
    print(flush=True)

    # -----------------------------------------------------------------------
    # Step 3: fp64 BEFORE any JAX import
    # -----------------------------------------------------------------------
    if mitigation_combo == "M1+Laplace+fp64":
        from prl_hgf.runtime import set_x64

        set_x64(True)

    # -----------------------------------------------------------------------
    # Step 4: Now safe to import JAX-dependent modules
    # -----------------------------------------------------------------------
    import arviz as az
    import numpy as np

    from prl_hgf.env.task_config import load_config
    from prl_hgf.fitting.config import FitConfig
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    # -----------------------------------------------------------------------
    # SIGTERM handler: write partial result on SLURM wall-time pre-kill
    # -----------------------------------------------------------------------
    t_start = time.perf_counter()
    result_path = RESULTS_DIR / f"cell_{cell_id:04d}.json"

    def _sigterm_handler(signum: int, frame: object) -> None:
        """Write TIMEOUT result on SIGTERM from SLURM."""
        walltime = time.perf_counter() - t_start
        timeout_result = {
            "cell_id": cell_id,
            "model": model_name,
            "n_per_group": n_per_group,
            "p_total": p_total,
            "mitigation_combo": mitigation_combo,
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
    # Step 5: Load FitConfig
    # -----------------------------------------------------------------------
    yaml_path = mitigation_to_yaml_path(mitigation_combo, model_name)
    fit_config = FitConfig.from_yaml(yaml_path)
    print(f"Loaded FitConfig: {yaml_path}")
    print(f"  n_draws={fit_config.sampler.n_draws}, "
          f"mass={fit_config.mitigation.mass_matrix_kind}, "
          f"laplace={fit_config.mitigation.use_laplace_warmup}, "
          f"fp64={fit_config.mitigation.use_fp64}")
    print(flush=True)

    # -----------------------------------------------------------------------
    # Step 6: Write "started" marker
    # -----------------------------------------------------------------------
    started_path = RESULTS_DIR / f"cell_{cell_id:04d}_started.json"
    started_marker = {
        "cell_id": cell_id,
        "model": model_name,
        "n_per_group": n_per_group,
        "p_total": p_total,
        "mitigation_combo": mitigation_combo,
        "fit_config_yaml": str(yaml_path),
        "job_id": job_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(started_path, started_marker)
    print(f"Wrote started marker: {started_path}")

    # -----------------------------------------------------------------------
    # Step 7: Simulate cohort
    # -----------------------------------------------------------------------
    print(f"\nSimulating cohort (n_per_group={n_per_group})...", flush=True)
    base_config = load_config()
    cfg = make_power_config(
        base_config, n_per_group, effect_size_delta=0.0,
        master_seed=12345 + cell_id,
    )
    sim_df = simulate_batch(cfg)
    sim_time = time.perf_counter() - t_start
    print(f"  Simulation done in {sim_time:.1f}s "
          f"({len(sim_df)} trial rows)", flush=True)

    # -----------------------------------------------------------------------
    # Step 8: Fit
    # -----------------------------------------------------------------------
    print(f"\nFitting (cell {cell_id:04d})...", flush=True)
    t_fit_start = time.perf_counter()

    try:
        idata = fit_batch_hierarchical(sim_df, fit_config=fit_config)

        # If returned as tuple (idata, adapted_params), extract idata
        if isinstance(idata, tuple):
            idata, _ = idata

        fit_time = time.perf_counter() - t_fit_start
        total_time = time.perf_counter() - t_start

        # -------------------------------------------------------------------
        # Step 9: Extract diagnostics
        # -------------------------------------------------------------------
        summary = az.summary(idata)
        rhat_max = float(summary["r_hat"].max())
        ess_min = float(summary["ess_bulk"].min())

        # Divergences
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            div_array = idata.sample_stats["diverging"].values
            n_divergent = int(np.sum(div_array))
            n_total_samples = div_array.size
            divergent_rate = n_divergent / n_total_samples if n_total_samples > 0 else 0.0
        else:
            n_divergent = 0
            n_total_samples = 0
            divergent_rate = 0.0

        # -------------------------------------------------------------------
        # Step 10: Determine status
        # -------------------------------------------------------------------
        if rhat_max <= 1.05 and ess_min >= 400 and divergent_rate < 0.05:
            status = "PASS"
        else:
            status = "INVALID"

        result = {
            "cell_id": cell_id,
            "model": model_name,
            "n_per_group": n_per_group,
            "p_total": p_total,
            "mitigation_combo": mitigation_combo,
            "status": status,
            "walltime_s": round(total_time, 2),
            "fit_time_s": round(fit_time, 2),
            "rhat_max": round(rhat_max, 4),
            "ess_min": round(ess_min, 1),
            "divergent_rate": round(divergent_rate, 4),
            "n_divergent": n_divergent,
            "n_total_samples": n_total_samples,
            "fit_config_yaml": str(yaml_path),
            "job_id": job_id,
            "commit": _get_git_commit(),
        }

        print(f"\n{'=' * 60}")
        print(f"RESULT: {status}")
        print(f"  Walltime:    {total_time:.1f}s (fit: {fit_time:.1f}s)")
        print(f"  R-hat max:   {rhat_max:.4f}")
        print(f"  ESS min:     {ess_min:.1f}")
        print(f"  Div rate:    {divergent_rate:.4f} ({n_divergent}/{n_total_samples})")
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
            "status": "CRASH",
            "walltime_s": round(total_time, 2),
            "rhat_max": None,
            "ess_min": None,
            "divergent_rate": None,
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
    # Step 11: Write result JSON
    # -----------------------------------------------------------------------
    _write_json(result_path, result)
    print(f"\nResult written: {result_path}", flush=True)


if __name__ == "__main__":
    main()
