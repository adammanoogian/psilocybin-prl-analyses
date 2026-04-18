"""PAT-RL foundation smoke: simulate -> fit -> export agents on CPU.

Runs the full parallel PAT-RL stack end-to-end to verify all Phase 18+19
modules compose correctly.  Writes trajectory CSVs + parameter summary CSV
under ``output/patrl_smoke/`` (or the dir passed via ``--output-dir``).

Three fit paths are supported via ``--fit-method``:

``blackjax`` (default)
    Full BlackJAX NUTS posterior (Phase 18 behavior, unchanged).
``laplace``
    VB-Laplace MAP + Hessian approximation from Phase 19.  Fast (~<60 s for
    5 agents on CPU).  Writes ``true_params.csv`` alongside
    ``parameter_summary.csv`` so downstream tests can compare recovery
    without re-simulating.
``both``
    Runs both fits on the same seed-determined cohort. Primary idata is
    laplace (used for export); secondary is blackjax.  Optionally invokes
    ``validation/vbl06_laplace_vs_nuts.py::compare_posteriors`` if that
    module is importable.

# TODO (OQ7, Phase 19 closure): after cluster NUTS smoke returns,
# write .planning/phases/19-vb-laplace-fit-path-patrl/19-CLOSURE-MEMO.md
# summarizing observed Laplace-vs-NUTS agreement on real cluster data
# and recommending keep-both vs consolidate. Defer the actual writing
# until numbers are in.

Usage
-----
    python scripts/12_smoke_patrl_foundation.py [--level {2,3}]
        [--output-dir DIR] [--n-tune N] [--n-draws N] [--seed N]
        [--n-participants N] [--dry-run]
        [--fit-method {blackjax,laplace,both}]

Exit codes
----------
0  All steps completed successfully (full path or dry-run path).
1  Runtime error (exception during simulate/fit/export/sanity-check).
2  blackjax not installed in the current environment.  Install it and retry.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402
from prl_hgf.analysis.export_trajectories import (  # noqa: E402
    export_subject_parameters,
    export_subject_trajectories,
)
from prl_hgf.env.pat_rl_config import load_pat_rl_config  # noqa: E402

# simulate_patrl_cohort was extracted to prl_hgf.env.pat_rl_simulator in
# Phase 19-01 so that the Laplace smoke (Plan 19-05) and VBL-06 comparison
# harness can share the same synthetic cohort on a fixed master_seed.
from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort  # noqa: E402
from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: level, output_dir, n_tune, n_draws,
        seed, n_participants, dry_run, fit_method.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[2, 3],
        default=2,
        metavar="{2,3}",
        help="HGF model level: level must be 2 or 3 (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs (default: output/patrl_smoke/).",
    )
    parser.add_argument(
        "--n-tune",
        type=int,
        default=200,
        help="Number of NUTS warmup steps (default: 200).",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=200,
        help="Number of posterior draws per chain (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default: 42).",
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=5,
        help="Number of synthetic participants to simulate (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run simulate + forward pass only, skip fit/export/sanity.  "
            "Validates module wiring without requiring blackjax or jaxopt.  "
            "Exits 0 on success, 1 on exception."
        ),
    )
    parser.add_argument(
        "--fit-method",
        choices=("blackjax", "laplace", "both"),
        default="blackjax",
        help=(
            "Which posterior fit path to run. Default 'blackjax' "
            "preserves Phase 18 behavior. 'laplace' uses "
            "fit_vb_laplace_patrl (Phase 19). 'both' runs both on the "
            "same seeded cohort and invokes the VBL-06 comparison if "
            "validation/vbl06_laplace_vs_nuts.py is importable."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fitting step helpers
# ---------------------------------------------------------------------------


def _fit_blackjax(
    sim_df: pd.DataFrame,
    level: int,
    n_tune: int,
    n_draws: int,
    seed: int,
    config: object,
) -> object:
    """Run PAT-RL hierarchical MCMC fit via BlackJAX NUTS.

    Imports and calls :func:`fit_batch_hierarchical_patrl` at call-time so
    that the script remains syntactically importable even when blackjax is not
    installed.  If blackjax is absent the call will raise ``ImportError``
    which is caught by :func:`main` and mapped to exit code 2.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulated trial data from
        :func:`~prl_hgf.env.pat_rl_simulator.simulate_patrl_cohort`.
    level : int
        HGF level (2 or 3).
    n_tune : int
        Number of NUTS warmup steps.
    n_draws : int
        Number of posterior draws per chain.
    seed : int
        Base random seed for the sampler.
    config : PATRLConfig
        Loaded PAT-RL configuration.

    Returns
    -------
    arviz.InferenceData
        MCMC posterior.
    """
    from prl_hgf.fitting.hierarchical_patrl import (  # noqa: PLC0415
        fit_batch_hierarchical_patrl,
    )

    model_name = "hgf_2level_patrl" if level == 2 else "hgf_3level_patrl"
    logger.info(
        "Fitting %s via BlackJAX NUTS  n_tune=%d  n_draws=%d  seed=%d",
        model_name,
        n_tune,
        n_draws,
        seed,
    )
    t0 = time.perf_counter()
    idata = fit_batch_hierarchical_patrl(
        sim_df,
        model_name=model_name,
        response_model="model_a",
        config=config,  # type: ignore[arg-type]
        n_chains=2,
        n_tune=n_tune,
        n_draws=n_draws,
        random_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    logger.info("BlackJAX fit complete in %.1f s", elapsed)
    return idata


def _fit_laplace(
    sim_df: pd.DataFrame,
    level: int,
    n_pseudo_draws: int,
    seed: int,
    config: object,
) -> object:
    """Run PAT-RL VB-Laplace fit via jaxopt.LBFGS MAP + Hessian.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulated trial data from
        :func:`~prl_hgf.env.pat_rl_simulator.simulate_patrl_cohort`.
    level : int
        HGF level (2 or 3).
    n_pseudo_draws : int
        Number of MultivariateNormal pseudo-samples from
        N(mode, Sigma).
    seed : int
        Base random seed for pseudo-sample draws and restart perturbations.
    config : PATRLConfig
        Loaded PAT-RL configuration.

    Returns
    -------
    arviz.InferenceData
        Laplace-approximated posterior (``sample_stats.converged`` present).
    """
    model_name = "hgf_2level_patrl" if level == 2 else "hgf_3level_patrl"
    logger.info(
        "Fitting %s via VB-Laplace  n_pseudo_draws=%d  seed=%d",
        model_name,
        n_pseudo_draws,
        seed,
    )
    t0 = time.perf_counter()
    idata = fit_vb_laplace_patrl(
        sim_df,
        model_name=model_name,
        response_model="model_a",
        config=config,  # type: ignore[arg-type]
        n_pseudo_draws=n_pseudo_draws,
        random_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Laplace fit complete in %.1f s", elapsed)
    return idata


def _fit(
    sim_df: pd.DataFrame,
    level: int,
    method: str,
    n_tune: int,
    n_draws: int,
    seed: int,
    config: object,
    n_pseudo_draws: int = 1000,
) -> tuple[object, object | None]:
    """Dispatch fit by method; returns (primary_idata, optional_secondary).

    Primary idata is what downstream export consumes. For ``'both'``,
    primary = laplace, secondary = blackjax (reflecting Phase 19 as the
    "new" path under test while blackjax is the baseline).

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulated trial data.
    level : int
        HGF level (2 or 3).
    method : str
        One of ``'blackjax'``, ``'laplace'``, ``'both'``.
    n_tune : int
        Number of NUTS warmup steps (blackjax path).
    n_draws : int
        Number of posterior draws per chain (blackjax path).
    seed : int
        Master RNG seed.
    config : PATRLConfig
        Loaded PAT-RL configuration.
    n_pseudo_draws : int, default 1000
        Number of pseudo-samples for Laplace path.

    Returns
    -------
    primary_idata : arviz.InferenceData
        Main posterior used for export (Laplace when method='both').
    secondary_idata : arviz.InferenceData or None
        Second posterior for comparison (BlackJAX when method='both').
    """
    if method == "blackjax":
        idata = _fit_blackjax(sim_df, level, n_tune, n_draws, seed, config)
        return idata, None
    if method == "laplace":
        idata = _fit_laplace(sim_df, level, n_pseudo_draws, seed, config)
        return idata, None
    # method == "both": primary = laplace, secondary = blackjax
    idata_lap = _fit_laplace(sim_df, level, n_pseudo_draws, seed, config)
    idata_nuts = _fit_blackjax(sim_df, level, n_tune, n_draws, seed, config)
    return idata_lap, idata_nuts


# ---------------------------------------------------------------------------
# Export step
# ---------------------------------------------------------------------------


def _export(
    idata: object,
    trials_by_participant: dict[str, list],
    choices_by_participant: dict[str, np.ndarray],
    level: int,
    output_dir: Path,
) -> list[Path]:
    """Export per-subject trajectory CSVs and the parameter summary.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`_fit` (primary idata).
    trials_by_participant : dict[str, list]
        Mapping from participant_id to :class:`PATRLTrial` list.
    choices_by_participant : dict[str, np.ndarray]
        Mapping from participant_id to choice array.
    level : int
        HGF level (2 or 3).
    output_dir : Path
        Directory to write CSVs.

    Returns
    -------
    list[Path]
        All paths written (per-subject trajectories + parameter summary).
    """
    model_name = "hgf_2level_patrl" if level == 2 else "hgf_3level_patrl"
    paths: list[Path] = []

    for pid, trials in trials_by_participant.items():
        choices = choices_by_participant[pid]
        out_path = export_subject_trajectories(
            participant_id=pid,
            idata=idata,  # type: ignore[arg-type]
            trials=trials,
            choices=choices,
            model_name=model_name,
            output_dir=output_dir,
        )
        paths.append(out_path)
        logger.info("Wrote trajectory: %s", out_path)

    param_path = export_subject_parameters(
        idata=idata,  # type: ignore[arg-type]
        model_name=model_name,
        output_dir=output_dir,
    )
    paths.append(param_path)
    logger.info("Wrote parameter summary: %s", param_path)
    return paths


def _write_true_params_csv(
    true_params: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Write true generative parameters to CSV for recovery comparison.

    Columns: ``participant_id``, ``parameter``, ``true_value``.  Written
    alongside ``parameter_summary.csv`` when ``--fit-method`` is ``laplace``
    or ``both``; skipped for ``blackjax`` to preserve Phase 18 output
    bit-for-bit.

    Parameters
    ----------
    true_params : dict[str, dict[str, float]]
        Mapping from participant_id to dict of true parameter values.
    output_dir : Path
        Directory where the CSV is written.

    Returns
    -------
    Path
        Absolute path to the written ``true_params.csv`` file.
    """
    rows: list[dict[str, object]] = []
    for pid, params in sorted(true_params.items()):
        for param_name, value in params.items():
            rows.append(
                {
                    "participant_id": pid,
                    "parameter": param_name,
                    "true_value": value,
                }
            )
    df = pd.DataFrame(rows, columns=["participant_id", "parameter", "true_value"])
    out_path = output_dir / "true_params.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote true params: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Sanity-check step
# ---------------------------------------------------------------------------


def _sanity_check(
    idata: object,
    method: str,
) -> None:
    """Method-specific sanity check on fit diagnostics.

    Dispatches on ``method`` explicitly — NO bare try/except around
    ``sample_stats`` access.  Laplace and BlackJAX InferenceData have
    different ``sample_stats`` keys; generic access would mask real bugs.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fit result to check. For ``method='both'``, pass each constituent
        idata to separate ``_sanity_check`` calls (one for ``'laplace'``,
        one for ``'blackjax'``).
    method : str
        One of ``'blackjax'``, ``'laplace'``, ``'both'``.

    Raises
    ------
    RuntimeError
        If the BlackJAX divergence fraction exceeds the 20% smoke gate.
    ValueError
        If ``method`` is not one of the three expected values.
    """
    if method == "blackjax":
        div = idata.sample_stats.diverging  # type: ignore[attr-defined]
        frac = float(div.sum()) / float(div.size)
        logger.info(
            "NUTS divergence fraction: %.3f  (%d / %d)",
            frac,
            int(div.sum()),
            int(div.size),
        )
        if frac >= 0.20:
            raise RuntimeError(
                f"NUTS divergence fraction {frac:.3f} >= 0.20 gate.  "
                f"Expected < 0.20."
            )
    elif method == "laplace":
        converged = bool(
            idata.sample_stats.converged.values.any()  # type: ignore[attr-defined]
        )
        if not converged:
            logger.warning(
                "Laplace fit did not converge for any subject. "
                "Check diagnostics (logp_at_mode, hessian_min_eigval, "
                "n_eigenvalues_clipped) for the failing subjects. "
                "Continuing — Laplace may fail per-subject; this is "
                "a soft gate at the smoke level."
            )
        else:
            logger.info("Laplace convergence check: at least one subject converged.")
    elif method == "both":
        # 'both' branch: caller should call _sanity_check on each leaf
        # idata individually.  This branch exists only for programmatic
        # callers that pass a single method string without separate idatas.
        raise ValueError(
            "For method='both', call _sanity_check(primary, 'laplace') "
            "and _sanity_check(secondary, 'blackjax') separately.  "
            "Do NOT pass method='both' directly to _sanity_check."
        )
    else:
        raise ValueError(
            f"method must be one of {{'blackjax', 'laplace', 'both'}}, "
            f"got {method!r}"
        )


def _log_recovery_table(
    idata: object,
    true_params: dict[str, dict[str, float]],
    config: object,
) -> None:
    """Log posterior-mean vs true parameter comparison per participant.

    Performs hard assertions on posterior finiteness.  Soft checks
    (directional sign) are logged as warnings only.

    Parameters
    ----------
    idata : arviz.InferenceData
        Laplace or NUTS posterior.
    true_params : dict[str, dict[str, float]]
        Mapping from participant_id to true parameter values.
    config : PATRLConfig
        PAT-RL configuration (provides prior means for directional check).
    """
    post = idata.posterior  # type: ignore[attr-defined]
    participant_ids = list(post.coords["participant_id"].values)
    prior_omega_2_mean = config.fitting.priors.omega_2.mean  # type: ignore[attr-defined]
    sign_match_omega2: list[bool] = []

    for pid in participant_ids:
        true_p = true_params.get(pid, {})

        omega2_post = float(
            post["omega_2"].sel(participant_id=pid).mean().values
        )

        # log_beta is sampled; beta may be a deterministic added downstream
        if "beta" in post:
            beta_post = float(
                post["beta"].sel(participant_id=pid).mean().values
            )
        else:
            log_beta_post = float(
                post["log_beta"].sel(participant_id=pid).mean().values
            )
            beta_post = float(np.exp(log_beta_post))

        # Hard: must be finite.
        assert np.isfinite(omega2_post), (
            f"participant {pid}: posterior_mean omega_2 is non-finite.  "
            f"Got {omega2_post!r}."
        )
        assert np.isfinite(beta_post), (
            f"participant {pid}: posterior_mean beta is non-finite.  "
            f"Got {beta_post!r}."
        )

        true_omega2 = true_p.get("omega_2", float("nan"))
        true_beta = true_p.get("beta", float("nan"))

        logger.info(
            "  %s  omega_2: true=%.3f  post_mean=%.3f  diff=%.3f",
            pid,
            true_omega2,
            omega2_post,
            omega2_post - true_omega2,
        )
        logger.info(
            "  %s  beta:    true=%.3f  post_mean=%.3f  diff=%.3f",
            pid,
            true_beta,
            beta_post,
            beta_post - true_beta,
        )

        # Soft: sign of update direction (posterior moved toward true?).
        if np.isfinite(true_omega2):
            moved = np.sign(omega2_post - prior_omega_2_mean) == np.sign(
                true_omega2 - prior_omega_2_mean
            )
            sign_match_omega2.append(bool(moved))

    if sign_match_omega2:
        n_correct = sum(sign_match_omega2)
        n_total = len(sign_match_omega2)
        if n_correct < n_total:
            logger.warning(
                "Directional check: %d/%d participants have posterior "
                "omega_2 moving toward true.",
                n_correct,
                n_total,
            )
        else:
            logger.info(
                "Directional check: %d/%d participants passed (omega_2 "
                "posterior moved toward true).",
                n_correct,
                n_total,
            )

    logger.info("Recovery table logged.")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the PAT-RL foundation smoke end-to-end.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = runtime error, 2 = blackjax missing.
    """
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    method: str = args.fit_method

    print("=" * 60)
    print("PAT-RL foundation smoke — Phase 18/19")
    print("=" * 60)
    print(f"  level={args.level}  n_participants={args.n_participants}")
    print(f"  n_tune={args.n_tune}  n_draws={args.n_draws}  seed={args.seed}")
    print(f"  fit_method={method}")
    if args.dry_run:
        print("  mode=DRY-RUN (simulate + forward pass only; fit skipped)")

    output_dir: Path = (
        args.output_dir if args.output_dir is not None else OUTPUT_DIR / "patrl_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    t_total = time.perf_counter()

    try:
        # 1. Load config.
        logger.info("Loading PAT-RL config...")
        config = load_pat_rl_config()

        # 2. Simulate cohort.
        logger.info(
            "Simulating %d synthetic participants at level %d...",
            args.n_participants,
            args.level,
        )
        t0 = time.perf_counter()
        sim_df, true_params, trials_by_participant = simulate_patrl_cohort(
            n_participants=args.n_participants,
            level=args.level,
            master_seed=args.seed,
            config=config,
        )
        logger.info(
            "Simulation done in %.1f s  rows=%d",
            time.perf_counter() - t0,
            len(sim_df),
        )

        # Collect per-participant choice arrays for export.
        choices_by_participant: dict[str, np.ndarray] = {}
        for pid in sorted(true_params.keys()):
            subset = sim_df[sim_df["participant_id"] == pid].sort_values("trial_idx")
            choices_by_participant[pid] = subset["choice"].to_numpy(dtype=np.int32)

        # --- Dry-run early exit ---
        if args.dry_run:
            elapsed_dry = time.perf_counter() - t_total
            n_participants = len(true_params)
            n_trials_total = len(sim_df)
            n_trials_per = n_trials_total // n_participants if n_participants else 0
            print()
            print("DRY-RUN COMPLETE")
            print(f"  fit_method:            {method}")
            print(f"  n_participants:        {n_participants}")
            print(f"  n_trials/participant:  {n_trials_per}")
            print(f"  total rows in sim_df:  {n_trials_total}")
            print(f"  elapsed:               {elapsed_dry:.2f} s")
            logger.info("PAT-RL foundation smoke DRY-RUN PASSED in %.2f s", elapsed_dry)
            return 0

        # 3. Fit.
        logger.info("Starting fit (method=%s)...", method)
        try:
            primary, secondary = _fit(
                sim_df=sim_df,
                level=args.level,
                method=method,
                n_tune=args.n_tune,
                n_draws=args.n_draws,
                seed=args.seed,
                config=config,
            )
        except ImportError as exc:
            logger.error(
                "blackjax is not installed.  "
                "Install it with: pip install blackjax>=1.2.4\n"
                "Original error: %s",
                exc,
            )
            return 2

        # 4. Sanity-check fit diagnostics.
        logger.info("Running sanity checks...")
        if method == "both":
            # Call each leaf separately — do NOT recurse via 'both' branch.
            _sanity_check(primary, "laplace")
            _sanity_check(secondary, "blackjax")
        else:
            _sanity_check(primary, method)

        # 5. Write true_params.csv for laplace / both paths.
        if method in ("laplace", "both"):
            _write_true_params_csv(true_params, output_dir)

        # 6. Export trajectories + parameter summary from PRIMARY idata.
        logger.info(
            "Exporting trajectories and parameter summary (primary=%s)...",
            "laplace" if method in ("laplace", "both") else "blackjax",
        )
        paths = _export(
            idata=primary,
            trials_by_participant=trials_by_participant,
            choices_by_participant=choices_by_participant,
            level=args.level,
            output_dir=output_dir,
        )

        # 7. Write secondary idata netcdf for 'both' mode.
        if method == "both" and secondary is not None:
            lap_nc = output_dir / "idata_laplace.nc"
            nuts_nc = output_dir / "idata_nuts.nc"
            primary.to_netcdf(str(lap_nc))  # type: ignore[attr-defined]
            secondary.to_netcdf(str(nuts_nc))  # type: ignore[attr-defined]
            logger.info("Wrote idata_laplace.nc: %s", lap_nc)
            logger.info("Wrote idata_nuts.nc:    %s", nuts_nc)

            # Optional VBL-06 comparison (lazy import so absence is non-fatal).
            try:
                from validation.vbl06_laplace_vs_nuts import (  # noqa: PLC0415
                    _apply_hard_gates,
                    compare_posteriors,
                )
            except ImportError:
                logger.warning(
                    "validation.vbl06_laplace_vs_nuts not importable — "
                    "skipping Laplace-vs-NUTS comparison."
                )
            else:
                diff = compare_posteriors(primary, secondary)
                diff_path = output_dir / "laplace_vs_nuts_diff.csv"
                diff.to_csv(diff_path, index=False)
                logger.info("Wrote Laplace-vs-NUTS diff to %s", diff_path)
                all_pass, msgs = _apply_hard_gates(diff)
                for msg in msgs:
                    if msg.startswith("HARD FAIL"):
                        logger.error(msg)
                    else:
                        logger.warning(msg)
                if not all_pass:
                    logger.warning(
                        "Laplace-vs-NUTS comparison failed hard gates. "
                        "Check %s for details.",
                        diff_path,
                    )

        # 8. Log recovery table.
        logger.info("Recovery table:")
        _log_recovery_table(idata=primary, true_params=true_params, config=config)

        # 9. Summary.
        elapsed_total = time.perf_counter() - t_total
        n_csvs = len(paths)
        total_bytes = sum(p.stat().st_size for p in paths if p.exists())
        print()
        print(f"WROTE {n_csvs} CSVs at {output_dir}")
        print(f"  fit_method:  {method}")
        print(f"  Total size: {total_bytes / 1024:.1f} KB")
        print(f"  Wall time:  {elapsed_total:.1f} s")
        logger.info("PAT-RL foundation smoke PASSED in %.1f s", elapsed_total)
        return 0

    except Exception:
        logger.exception("PAT-RL foundation smoke FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
