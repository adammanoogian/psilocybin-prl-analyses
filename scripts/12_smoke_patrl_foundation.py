"""PAT-RL foundation smoke: simulate -> fit -> export 5 agents on CPU.

Runs the full parallel PAT-RL stack end-to-end to verify all Phase 18 modules
compose correctly.  Writes trajectory CSVs + parameter summary CSV under
``output/patrl_smoke/`` (or the dir passed via ``--output-dir``).

Usage
-----
    python scripts/12_smoke_patrl_foundation.py [--level {2,3}]
        [--output-dir DIR] [--n-tune N] [--n-draws N] [--seed N]
        [--n-participants N] [--dry-run]

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
from prl_hgf.env.pat_rl_sequence import generate_session_patrl  # noqa: E402
from prl_hgf.models.hgf_2level_patrl import build_2level_network_patrl  # noqa: E402
from prl_hgf.models.hgf_3level_patrl import build_3level_network_patrl  # noqa: E402

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
        seed, n_participants.
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
            "Run simulate + forward pass only, skip MCMC fit/export/sanity.  "
            "Validates module wiring without requiring blackjax.  Exits 0 on "
            "success, 1 on exception."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------


def _run_hgf_forward(
    trials: list,
    omega_2: float,
    level: int,
    omega_3: float = -6.0,
    kappa: float = 1.0,
    mu3_0: float = 1.0,
) -> np.ndarray:
    """Run HGF forward pass and return per-trial mu2 trajectory.

    Parameters
    ----------
    trials : list[PATRLTrial]
        Trial list from :func:`generate_session_patrl`.
    omega_2 : float
        Tonic volatility for the value parent node.
    level : int
        HGF level (2 or 3).
    omega_3 : float, optional
        Tonic volatility for the volatility parent (3-level only).
    kappa : float, optional
        Volatility coupling strength (3-level only).
    mu3_0 : float, optional
        Initial volatility state mean (3-level only).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials,)`` float64 array of posterior belief means mu2.
    """
    u = np.array([t.state for t in trials], dtype=np.float64)
    n_trials = len(trials)

    if level == 2:
        net = build_2level_network_patrl(omega_2=omega_2)
    else:
        net = build_3level_network_patrl(
            omega_2=omega_2, omega_3=omega_3, kappa=kappa, mu3_0=mu3_0
        )

    net.input_data(
        input_data=u[:, None],
        time_steps=np.ones(n_trials, dtype=np.float64),
    )
    traj = net.node_trajectories[1]  # BELIEF_NODE = 1
    return np.asarray(traj["mean"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Simulation step
# ---------------------------------------------------------------------------


def _simulate_cohort(
    n_participants: int,
    level: int,
    master_seed: int,
    config: object,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, list]]:
    """Simulate synthetic PAT-RL participants at known true parameters.

    Draws true parameters from the ``healthy`` phenotype distribution for each
    participant, runs an HGF forward pass, and samples binary choices via the
    Model A softmax.

    Parameters
    ----------
    n_participants : int
        Number of synthetic agents.
    level : int
        HGF level (2 or 3).
    master_seed : int
        Master RNG seed; spawns per-participant child seeds.
    config : PATRLConfig
        Loaded PAT-RL configuration.

    Returns
    -------
    sim_df : pandas.DataFrame
        Trial-level data with required columns for
        :func:`fit_batch_hierarchical_patrl`.
    true_params : dict[str, dict[str, float]]
        Mapping from participant_id to dict of true parameter values.
    trials_by_participant : dict[str, list]
        Mapping from participant_id to list of :class:`PATRLTrial`.
    """
    phenotype = config.simulation.phenotypes["healthy"]  # type: ignore[attr-defined]

    # Spawn n_participants independent child seeds from master.
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n_participants)

    all_rows: list[dict] = []
    true_params: dict[str, dict[str, float]] = {}
    trials_by_participant: dict[str, list] = {}

    for i, child_ss in enumerate(child_seeds):
        pid = f"P{i:03d}"
        rng = np.random.default_rng(child_ss)

        # Sample true parameters from the healthy phenotype distribution.
        omega_2_true = float(rng.normal(phenotype.omega_2.mean, phenotype.omega_2.sd))
        beta_true = float(
            max(0.01, rng.normal(phenotype.beta.mean, phenotype.beta.sd))
        )

        true_p: dict[str, float] = {"omega_2": omega_2_true, "beta": beta_true}

        if level == 3:
            omega_3_true = float(
                rng.normal(phenotype.kappa.mean + 1e-3, max(0.01, phenotype.kappa.sd))
                if phenotype.kappa.sd > 0
                else phenotype.kappa.mean
            )
            # Use omega_3 from config priors
            omega_3_true = float(
                rng.normal(
                    config.fitting.priors.omega_3.mean,  # type: ignore[attr-defined]
                    config.fitting.priors.omega_3.sd,  # type: ignore[attr-defined]
                )
            )
            kappa_true = float(phenotype.kappa.mean)
            mu3_0_true = float(phenotype.mu3_0.mean)
            true_p["omega_3"] = omega_3_true
            true_p["kappa"] = kappa_true
            true_p["mu3_0"] = mu3_0_true

        # Draw a trial seed for environment (separate from parameter RNG).
        env_seed = int(rng.integers(0, 2**31))

        # Generate trial sequence.
        trials = generate_session_patrl(config, seed=env_seed)
        n_trials = len(trials)
        trials_by_participant[pid] = trials

        # HGF forward pass at true parameters to get mu2 trajectory.
        if level == 3:
            mu2_traj = _run_hgf_forward(
                trials,
                omega_2_true,
                level=3,
                omega_3=omega_3_true,
                kappa=kappa_true,
                mu3_0=mu3_0_true,
            )
        else:
            mu2_traj = _run_hgf_forward(trials, omega_2_true, level=2)

        # Sample choices: approach=1, avoid=0 via softmax over [EV_avoid=0, EV_approach].
        reward_mag = np.array([t.reward_mag for t in trials], dtype=np.float64)
        shock_mag = np.array([t.shock_mag for t in trials], dtype=np.float64)

        # Compute EV_approach using numpy (no JAX in simulation loop).
        mu2_clip = np.clip(mu2_traj, -30.0, 30.0)
        p_danger = 1.0 / (1.0 + np.exp(-mu2_clip))
        ev_approach = (1.0 - p_danger) * reward_mag - p_danger * shock_mag

        # Softmax choice probabilities: p_approach = sigmoid(beta * ev_approach).
        logit_approach = beta_true * ev_approach
        p_approach = 1.0 / (1.0 + np.exp(-logit_approach))

        choice_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        choices = choice_rng.random(n_trials) < p_approach
        choices_int = choices.astype(np.int32)

        delta_hr = np.array([t.delta_hr for t in trials], dtype=np.float64)

        # Assemble rows.
        for t_idx, trial in enumerate(trials):
            all_rows.append(
                {
                    "participant_id": pid,
                    "trial_idx": trial.trial_idx,
                    "state": trial.state,
                    "choice": int(choices_int[t_idx]),
                    "reward_mag": trial.reward_mag,
                    "shock_mag": trial.shock_mag,
                    "delta_hr": float(delta_hr[t_idx]),
                    "outcome_time_s": trial.outcome_time_s,
                }
            )

        true_params[pid] = true_p
        logger.info(
            "Simulated participant %s: omega_2=%.3f  beta=%.3f  "
            "approach_rate=%.2f",
            pid,
            omega_2_true,
            beta_true,
            float(np.mean(choices_int)),
        )

    sim_df = pd.DataFrame(all_rows)
    return sim_df, true_params, trials_by_participant


# ---------------------------------------------------------------------------
# Fitting step  (blackjax imported lazily inside call)
# ---------------------------------------------------------------------------


def _fit(
    sim_df: pd.DataFrame,
    level: int,
    n_tune: int,
    n_draws: int,
    seed: int,
    config: object,
) -> object:
    """Run PAT-RL hierarchical MCMC fit.

    Imports and calls :func:`fit_batch_hierarchical_patrl` at call-time so
    that the script remains syntactically importable even when blackjax is not
    installed.  If blackjax is absent the call will raise ``ImportError``
    which is caught by :func:`main` and mapped to exit code 2.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulated trial data from :func:`_simulate_cohort`.
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
        "Fitting %s  n_tune=%d  n_draws=%d  seed=%d",
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
    logger.info("Fit complete in %.1f s", elapsed)
    return idata


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
        MCMC posterior from :func:`_fit`.
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


# ---------------------------------------------------------------------------
# Sanity-check step
# ---------------------------------------------------------------------------


def _sanity_check(
    idata: object,
    true_params: dict[str, dict[str, float]],
    config: object,
) -> None:
    """Log posterior-mean vs true parameter comparison per participant.

    Performs hard assertions on posterior finiteness and divergence rate.
    Soft checks (sign match) are logged as warnings only.

    Parameters
    ----------
    idata : arviz.InferenceData
        MCMC posterior from :func:`_fit`.
    true_params : dict[str, dict[str, float]]
        Mapping from participant_id to true parameter values.
    config : PATRLConfig
        PAT-RL configuration (provides prior means for directional check).

    Raises
    ------
    AssertionError
        If any posterior mean is non-finite or the divergence rate exceeds
        20%.
    """
    post = idata.posterior  # type: ignore[attr-defined]
    participant_ids = list(post.coords["participant_id"].values)

    # --- Hard assertion: divergence rate < 20% ---
    diverging = idata.sample_stats.diverging  # type: ignore[attr-defined]
    div_arr = np.asarray(diverging)
    div_rate = float(div_arr.sum()) / float(div_arr.size)
    logger.info("Divergence rate: %.3f  (%d / %d)", div_rate, int(div_arr.sum()), div_arr.size)
    assert div_rate < 0.20, (
        f"Divergence rate {div_rate:.3f} exceeds 20% smoke gate.  "
        f"Expected < 0.20."
    )

    # --- Per-participant: log posterior mean vs true ---
    prior_omega_2_mean = config.fitting.priors.omega_2.mean  # type: ignore[attr-defined]
    sign_match_omega2: list[bool] = []

    for pid in participant_ids:
        true_p = true_params.get(pid, {})

        omega2_post = float(
            post["omega_2"].sel(participant_id=pid).mean().values
        )

        # log_beta is sampled; beta is a deterministic added by _samples_to_idata
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

        # Soft: sign of update direction (posterior moved towards true?).
        if np.isfinite(true_omega2):
            moved_toward_true = np.sign(omega2_post - prior_omega_2_mean) == np.sign(
                true_omega2 - prior_omega_2_mean
            )
            sign_match_omega2.append(bool(moved_toward_true))

    if sign_match_omega2:
        n_correct = sum(sign_match_omega2)
        n_total = len(sign_match_omega2)
        if n_correct < n_total:
            logger.warning(
                "Directional check: %d/%d participants have posterior "
                "omega_2 moving toward true.  This is a smoke test; "
                "full recovery gate deferred to Phase 19.",
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

    logger.info("Sanity check passed.")


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

    print("=" * 60)
    print("PAT-RL foundation smoke — Phase 18")
    print("=" * 60)
    print(f"  level={args.level}  n_participants={args.n_participants}")
    print(f"  n_tune={args.n_tune}  n_draws={args.n_draws}  seed={args.seed}")
    if args.dry_run:
        print("  mode=DRY-RUN (simulate + forward pass only; MCMC skipped)")

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
        sim_df, true_params, trials_by_participant = _simulate_cohort(
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
            print(f"  n_participants:        {n_participants}")
            print(f"  n_trials/participant:  {n_trials_per}")
            print(f"  total rows in sim_df:  {n_trials_total}")
            print(f"  elapsed:               {elapsed_dry:.2f} s")
            logger.info("PAT-RL foundation smoke DRY-RUN PASSED in %.2f s", elapsed_dry)
            return 0

        # 3. Fit.
        logger.info("Starting MCMC fit...")
        try:
            idata = _fit(
                sim_df=sim_df,
                level=args.level,
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

        # 4. Export.
        logger.info("Exporting trajectories and parameter summary...")
        paths = _export(
            idata=idata,
            trials_by_participant=trials_by_participant,
            choices_by_participant=choices_by_participant,
            level=args.level,
            output_dir=output_dir,
        )

        # 5. Sanity check.
        logger.info("Running sanity checks...")
        _sanity_check(idata=idata, true_params=true_params, config=config)

        # 6. Summary.
        elapsed_total = time.perf_counter() - t_total
        n_csvs = len(paths)
        total_bytes = sum(p.stat().st_size for p in paths if p.exists())
        print()
        print(f"WROTE {n_csvs} CSVs at {output_dir}")
        print(f"  Total size: {total_bytes / 1024:.1f} KB")
        print(f"  Wall time:  {elapsed_total:.1f} s")
        logger.info("PAT-RL foundation smoke PASSED in %.1f s", elapsed_total)
        return 0

    except Exception:
        logger.exception("PAT-RL foundation smoke FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
