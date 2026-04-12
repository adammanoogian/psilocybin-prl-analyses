"""VALID-03: Cross-platform posterior consistency check.

This script validates that :func:`prl_hgf.fitting.hierarchical.fit_batch_hierarchical`
produces posterior means that agree within 1% relative error regardless of
whether JAX runs on CPU or GPU.

**Design:** JAX's platform (CPU vs GPU) cannot be changed within a single Python
process — ``jax.config.update("jax_platform_name", ...)`` is a one-time global
setting applied at import time. Therefore, VALID-03 is structured as **two
separate script invocations** followed by a comparison step:

Usage
-----
Run on CPU (forces CPU even if GPU is present)::

    JAX_PLATFORM_NAME=cpu python validation/valid03_cross_platform.py run \\
        --output results/valid03_cpu.json

Run on GPU (default on GPU node; uses whatever JAX detects)::

    python validation/valid03_cross_platform.py run \\
        --output results/valid03_gpu.json

Compare results (1% relative error threshold)::

    python validation/valid03_cross_platform.py compare \\
        results/valid03_cpu.json results/valid03_gpu.json

**Dev-machine behaviour:** On a CPU-only machine both runs execute on CPU and
produce identical results (trivially pass). This is by design — the MCMC
results depend on the seed and are deterministic on the same platform. The
actual CPU vs GPU comparison is performed manually on the cluster.

**Cluster usage:** Run the first invocation on a CPU node (or with
``JAX_PLATFORM_NAME=cpu``) and the second on a GPU node. Save both JSON files
to shared storage (e.g., Lustre project scratch) then run the compare step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the project root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def run_fit_and_save(
    output_path: Path,
    n_per_group: int = 3,
    n_chains: int = 2,
    n_draws: int = 200,
    n_tune: int = 200,
    seed: int = 42,
) -> None:
    """Fit a small cohort and save posterior means to JSON.

    Builds a small power config, simulates a cohort, fits the 3-level HGF
    model via :func:`~prl_hgf.fitting.hierarchical.fit_batch_hierarchical`,
    and writes per-participant posterior means for every parameter to
    ``output_path`` as a JSON file.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination file for the JSON results.
    n_per_group : int, optional
        Participants per group (psilocybin + placebo).  Default ``3``.
    n_chains : int, optional
        MCMC chains.  Default ``2``.
    n_draws : int, optional
        Posterior draws per chain.  Default ``200``.
    n_tune : int, optional
        Tuning steps per chain.  Default ``200``.
    seed : int, optional
        Master RNG seed for both simulation and MCMC.  Default ``42``.

    Returns
    -------
    None
        Results are written to ``output_path``.
    """
    import jax

    from prl_hgf.env.task_config import load_config
    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    # Log which platform JAX is running on.
    devices = jax.devices()
    platform = str(devices[0].platform)
    print(f"[VALID-03] JAX devices: {devices}")
    print(f"[VALID-03] Platform: {platform}")

    # Build a small power config.
    base_cfg = load_config()
    cfg = make_power_config(
        base_cfg,
        n_per_group=n_per_group,
        effect_size_delta=0.5,
        master_seed=seed,
    )

    print(
        f"[VALID-03] Simulating cohort: "
        f"{n_per_group} per group, seed={seed}"
    )
    t0 = time.perf_counter()
    sim_df = simulate_batch(cfg)
    sim_time = time.perf_counter() - t0
    print(f"[VALID-03] Simulation complete in {sim_time:.2f}s, "
          f"rows={len(sim_df)}")

    print(
        f"[VALID-03] Fitting hgf_3level: "
        f"chains={n_chains}, draws={n_draws}, tune={n_tune}"
    )
    t1 = time.perf_counter()
    idata = fit_batch_hierarchical(
        sim_df,
        model_name="hgf_3level",
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        target_accept=0.9,
        random_seed=seed,
        sampler="numpyro",
        progressbar=False,
    )
    fit_time = time.perf_counter() - t1
    print(f"[VALID-03] Fit complete in {fit_time:.2f}s")

    # Extract per-participant posterior means.
    posterior = idata.posterior

    # Determine participant IDs from coords (ground truth order).
    if "participant" in posterior.coords:
        participant_ids: list[str] = list(posterior.coords["participant"].values)
    else:
        # Fallback: infer from the number of participants in shape.
        first_var = list(posterior.data_vars)[0]
        n_participants = posterior[first_var].shape[-1]
        participant_ids = [str(i) for i in range(n_participants)]

    # Build posterior_means: {param_name: {participant_id: float}}.
    posterior_means: dict[str, dict[str, float]] = {}
    for param in list(posterior.data_vars):
        arr = posterior[param]
        # Compute mean over chain and draw dims.
        param_mean = arr.mean(dim=["chain", "draw"])
        param_posterior: dict[str, float] = {}
        # Handle both shaped (participant dim) and scalar parameters.
        if "participant" in param_mean.dims:
            for i, pid in enumerate(participant_ids):
                val = float(param_mean.isel(participant=i).values)
                param_posterior[pid] = val
        else:
            # Scalar parameter: same value for all participants.
            val = float(param_mean.values)
            for pid in participant_ids:
                param_posterior[pid] = val
        posterior_means[param] = param_posterior

    result = {
        "platform": platform,
        "n_per_group": n_per_group,
        "seed": seed,
        "n_chains": n_chains,
        "n_draws": n_draws,
        "n_tune": n_tune,
        "simulation_time_s": round(sim_time, 2),
        "fit_time_s": round(fit_time, 2),
        "participant_ids": participant_ids,
        "parameters": list(posterior_means.keys()),
        "posterior_means": posterior_means,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"[VALID-03] Saved to {output_path}\n"
        f"  platform={platform}, "
        f"n_participants={len(participant_ids)}, "
        f"params={list(posterior_means.keys())}"
    )


def compare_results(
    path_a: Path,
    path_b: Path,
    rtol: float = 0.01,
) -> bool:
    """Compare posterior means from two JSON files within a relative tolerance.

    Loads both JSON files (produced by :func:`run_fit_and_save`) and computes
    the relative error for every (parameter, participant) pair::

        rel_err = abs(mean_a - mean_b) / (abs(mean_a) + 1e-8)

    Prints a summary of max and mean relative errors, and a per-parameter
    breakdown. Returns ``True`` if **all** relative errors are below ``rtol``,
    ``False`` otherwise.

    Parameters
    ----------
    path_a : pathlib.Path
        Path to the first JSON file (e.g. CPU results).
    path_b : pathlib.Path
        Path to the second JSON file (e.g. GPU results).
    rtol : float, optional
        Relative tolerance threshold.  Default ``0.01`` (1%).

    Returns
    -------
    bool
        ``True`` if all relative errors are below ``rtol``, else ``False``.
    """
    with Path(path_a).open() as fh:
        data_a = json.load(fh)
    with Path(path_b).open() as fh:
        data_b = json.load(fh)

    platform_a = data_a.get("platform", "unknown")
    platform_b = data_b.get("platform", "unknown")
    print(f"[VALID-03] Comparing: platform_a={platform_a}, platform_b={platform_b}")

    posterior_a = data_a["posterior_means"]
    posterior_b = data_b["posterior_means"]

    params_a = set(posterior_a.keys())
    params_b = set(posterior_b.keys())
    common_params = sorted(params_a & params_b)
    if params_a != params_b:
        only_a = params_a - params_b
        only_b = params_b - params_a
        print(
            f"[VALID-03] WARNING: parameter mismatch. "
            f"Only in A: {only_a}. Only in B: {only_b}. "
            f"Comparing {len(common_params)} common parameters."
        )

    all_relative_errors: list[float] = []
    per_param: dict[str, dict[str, float]] = {}
    passed = True

    for param in common_params:
        means_a = posterior_a[param]
        means_b = posterior_b[param]
        pids_a = set(means_a.keys())
        pids_b = set(means_b.keys())
        common_pids = sorted(pids_a & pids_b)
        if pids_a != pids_b:
            print(
                f"[VALID-03] WARNING: participant mismatch for {param}. "
                f"Only in A: {pids_a - pids_b}. Only in B: {pids_b - pids_a}."
            )

        param_errors: dict[str, float] = {}
        for pid in common_pids:
            mean_a = float(means_a[pid])
            mean_b = float(means_b[pid])
            rel_err = abs(mean_a - mean_b) / (abs(mean_a) + 1e-8)
            param_errors[pid] = rel_err
            all_relative_errors.append(rel_err)
            if rel_err >= rtol:
                passed = False

        per_param[param] = param_errors

    if not all_relative_errors:
        print("[VALID-03] ERROR: no common (parameter, participant) pairs to compare.")
        return False

    max_err = max(all_relative_errors)
    mean_err = sum(all_relative_errors) / len(all_relative_errors)

    print(f"\n[VALID-03] Relative error summary (rtol={rtol:.1%}):")
    print(f"  max rel_err  = {max_err:.6f} ({max_err:.3%})")
    print(f"  mean rel_err = {mean_err:.6f} ({mean_err:.3%})")
    print(f"  n_comparisons = {len(all_relative_errors)}")
    print()

    print("[VALID-03] Per-parameter max relative error:")
    for param in common_params:
        errors = list(per_param[param].values())
        if errors:
            p_max = max(errors)
            status = "PASS" if p_max < rtol else "FAIL"
            print(f"  [{status}] {param:20s}  max={p_max:.6f} ({p_max:.3%})")

    if passed:
        print(f"\n[VALID-03] PASS: all {len(all_relative_errors)} comparisons "
              f"within {rtol:.1%} relative tolerance.")
    else:
        n_fail = sum(1 for e in all_relative_errors if e >= rtol)
        print(
            f"\n[VALID-03] FAIL: {n_fail}/{len(all_relative_errors)} comparisons "
            f"exceed {rtol:.1%} relative tolerance."
        )

    return passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for VALID-03."""
    parser = argparse.ArgumentParser(
        prog="valid03_cross_platform.py",
        description=(
            "VALID-03: Cross-platform posterior consistency check.\n\n"
            "Run twice (once on CPU, once on GPU) then compare results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # CPU run:\n"
            "  JAX_PLATFORM_NAME=cpu python valid03_cross_platform.py run"
            " --output results/valid03_cpu.json\n\n"
            "  # GPU run (on GPU node):\n"
            "  python valid03_cross_platform.py run"
            " --output results/valid03_gpu.json\n\n"
            "  # Compare:\n"
            "  python valid03_cross_platform.py compare"
            " results/valid03_cpu.json results/valid03_gpu.json\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run subcommand -------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Fit a small cohort and save posterior means to JSON.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the JSON results file.",
    )
    run_parser.add_argument(
        "--n-per-group",
        type=int,
        default=3,
        metavar="N",
        help="Participants per group (default: %(default)s).",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default: %(default)s).",
    )
    run_parser.add_argument(
        "--n-chains",
        type=int,
        default=2,
        help="MCMC chains (default: %(default)s).",
    )
    run_parser.add_argument(
        "--n-draws",
        type=int,
        default=200,
        help="Posterior draws per chain (default: %(default)s).",
    )
    run_parser.add_argument(
        "--n-tune",
        type=int,
        default=200,
        help="Tuning steps per chain (default: %(default)s).",
    )

    # -- compare subcommand ---------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two JSON result files within a relative tolerance.",
    )
    compare_parser.add_argument(
        "path_a",
        type=Path,
        help="Path to the first JSON file (e.g. CPU results).",
    )
    compare_parser.add_argument(
        "path_b",
        type=Path,
        help="Path to the second JSON file (e.g. GPU results).",
    )
    compare_parser.add_argument(
        "--rtol",
        type=float,
        default=0.01,
        help="Relative error tolerance (default: %(default)s = 1%%).",
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_fit_and_save(
            output_path=args.output,
            n_per_group=args.n_per_group,
            n_chains=args.n_chains,
            n_draws=args.n_draws,
            n_tune=args.n_tune,
            seed=args.seed,
        )
        sys.exit(0)

    elif args.command == "compare":
        ok = compare_results(
            path_a=args.path_a,
            path_b=args.path_b,
            rtol=args.rtol,
        )
        sys.exit(0 if ok else 1)
