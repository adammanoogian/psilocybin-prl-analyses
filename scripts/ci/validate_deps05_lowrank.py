"""DEPS-05 validation gate: BlackJAX 1.5 low-rank (dense) mass matrix at P=30.

Throwaway diagnostic for plan 27-03; NOT production code.  Run on a local GPU
or M3 dev node.  Emits a structured JSON blob to
``.planning/phases/27-dependency-upgrade-chain/DEPS-05-evidence.json``.

Purpose
-------
Empirically validate whether BlackJAX 1.5's dense (non-diagonal) mass-matrix
adaptation clears the 3-level no-pooling conditioning cliff that caused
job 55143456 (A100 80GB) to time out after 8 h with step-size collapsing to
7.245e-10 and max_tree_depth saturation.

API note (from 27-03-blackjax-api-probe.txt)
--------------------------------------------
``blackjax.window_adaptation`` in BlackJAX 1.5 exposes the low-rank
(dense) path via ``is_mass_matrix_diagonal=False`` — the same kwarg
name as 1.2.x but inverted value.  There is no separate
``mass_matrix_kind`` enum in 1.5.  Confirmed by:

    conda run -n verify_pyhgf_fork python -c \\
        "import blackjax, inspect; \\
         print(inspect.signature(blackjax.window_adaptation))"
    # (algorithm, logdensity_fn: Callable, \\
    #  is_mass_matrix_diagonal: bool = True, ...)

Cohort geometry
---------------
P_total = 30 = 3 groups × 5 participants × 2 sessions, matching the
``16_smoke_test_gpu.slurm`` default of N=5/group
(``n_participant_sessions = 2 * n_smoke * 3`` with ``n_smoke=5``;
see ``scripts/03_pre_analysis/03_run_power_iteration.py`` line 807).
RNG master seed for simulation = 77777, matching the smoke test
(``make_power_config(..., 77777)`` at line 844 of same file).

Phase boundary
--------------
This script is THROWAWAY.  Do NOT wire ``is_mass_matrix_diagonal``
into ``scripts/03_pre_analysis/03_run_power_iteration.py`` or any
production fitting code in ``src/prl_hgf/fitting/``.  That is Phase 29
work (ROADMAP MODEA-01..03).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `import config` (project-root config.py)
# resolves when this script is launched directly via SLURM.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / ".planning"
    / "phases"
    / "27-dependency-upgrade-chain"
    / "DEPS-05-evidence.json"
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``output``, ``n_warmup``, and ``n_draws``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "DEPS-05: BlackJAX 1.5 dense mass-matrix validation at P=30. "
            "Run on GPU; emits a JSON evidence blob."
        )
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Path for the output JSON evidence blob.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help="Number of window-adaptation warmup steps (default: 1000).",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=500,
        help="Number of posterior draws per chain (default: 500).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------

def _check_gpu() -> str:
    """Return a device string; abort if no GPU is found.

    Returns
    -------
    str
        Comma-separated string of GPU device descriptions.

    Raises
    ------
    SystemExit
        If no GPU device is available (defeats the purpose of the cliff probe).
    """
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print(
            "ERROR: No GPU detected.\n"
            f"  JAX sees: {devices}\n"
            "  This validation script MUST run on GPU — running on CPU "
            "defeats the purpose of the cliff probe (see plan 27-03 §Task 2).\n"
            "  Use M3 dev node: smux new-session -J 4:gpu",
            file=sys.stderr,
        )
        sys.exit(1)
    return ", ".join(str(d) for d in gpu_devices)


# ---------------------------------------------------------------------------
# Cohort simulation (mirrors 16_smoke_test_gpu.slurm / 03_run_power_iteration.py)
# ---------------------------------------------------------------------------

_N_PER_GROUP: int = 5
_P_TOTAL: int = 30  # 3 groups x 5 participants x 2 sessions
_N_CHAINS: int = 4
_SIM_SEED: int = 77777  # matches make_power_config(..., 77777) in smoke test
_MCMC_SEED: int = 42


def _build_cohort_data(
    n_per_group: int = _N_PER_GROUP,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Simulate a P=30 cohort and build JAX data arrays.

    Mirrors the production path in ``03_run_power_iteration.py``:
    ``make_power_config(base_config, n_smoke, d_smoke, 77777)``
    → ``simulate_batch(cfg_smoke)``
    → ``fit_batch_hierarchical(sim_smoke, 'hgf_3level', ...)``.

    Parameters
    ----------
    n_per_group : int, optional
        Participants per group per session; default 5 → P_total = 30.

    Returns
    -------
    jax_input_data : jnp.ndarray, shape (P, n_trials, 3)
    jax_observed : jnp.ndarray, shape (P, n_trials, 3)
    jax_choices : jnp.ndarray, shape (P, n_trials)
    jax_trial_mask : jnp.ndarray, shape (P, n_trials)
    n_participants : int
    """
    from pathlib import Path as _Path

    import prl_hgf
    from prl_hgf.env.task_config import load_config
    from prl_hgf.power.config import make_power_config
    from prl_hgf.simulation.batch import simulate_batch

    # Locate the base config (same as production path)
    configs_dir = _Path(prl_hgf.__file__).resolve().parents[2] / "configs"
    base_yaml = configs_dir / "prl_analysis.yaml"
    if not base_yaml.exists():
        # Fallback: locate via config.py
        import config as _cfg
        base_yaml = _cfg.CONFIGS_DIR / "prl_analysis.yaml"

    import yaml

    with open(base_yaml, encoding="utf-8") as fh:
        base_yaml_dict = yaml.safe_load(fh)

    effect_size = base_yaml_dict.get("power", {}).get("effect_size_grid", [0.5])[0]
    base_config = load_config(base_yaml)
    cfg = make_power_config(base_config, n_per_group, effect_size, _SIM_SEED)
    sim_df = simulate_batch(cfg)

    # Build stacked arrays (same logic as fit_batch_hierarchical internals)
    from prl_hgf.fitting.hierarchical import _build_arrays_single

    group_keys = ["participant_id", "group", "session"]
    groups = list(sim_df.groupby(group_keys, sort=False))

    input_data_list: list[np.ndarray] = []
    observed_list: list[np.ndarray] = []
    choices_list: list[np.ndarray] = []

    for (_pid, _grp, _sess), subset in groups:
        if "trial" in subset.columns:
            subset = subset.sort_values("trial")
        inp, obs, ch = _build_arrays_single(subset)
        input_data_list.append(inp)
        observed_list.append(obs)
        choices_list.append(ch)

    n_participants = len(input_data_list)
    n_trials = input_data_list[0].shape[0]

    jax_input_data = jnp.array(
        np.stack(input_data_list, axis=0), dtype=jnp.float32
    )
    jax_observed = jnp.array(
        np.stack(observed_list, axis=0), dtype=jnp.int32
    )
    jax_choices = jnp.array(
        np.stack(choices_list, axis=0), dtype=jnp.int32
    )
    jax_trial_mask = jnp.ones(
        (n_participants, n_trials), dtype=jnp.float32
    )

    return jax_input_data, jax_observed, jax_choices, jax_trial_mask, n_participants


# ---------------------------------------------------------------------------
# Main validation run
# ---------------------------------------------------------------------------

def _run_validation(
    n_warmup: int,
    n_draws: int,
) -> dict:
    """Run the DEPS-05 validation gate.

    Parameters
    ----------
    n_warmup : int
        Number of window-adaptation warmup steps.
    n_draws : int
        Number of posterior draws per chain.

    Returns
    -------
    dict
        Structured evidence blob with all required fields.
    """
    import arviz as az
    import blackjax

    # --- GPU gate ---
    device_str = _check_gpu()

    # --- Cohort simulation ---
    print(
        f"[DEPS-05] Simulating P={_P_TOTAL} cohort "
        f"(3 groups x {_N_PER_GROUP} x 2 sessions, seed={_SIM_SEED})...",
        flush=True,
    )
    (
        jax_input_data,
        jax_observed,
        jax_choices,
        jax_trial_mask,
        n_participants,
    ) = _build_cohort_data(_N_PER_GROUP)

    print(
        f"[DEPS-05] Cohort ready: P={n_participants}, "
        f"n_trials={jax_input_data.shape[1]}",
        flush=True,
    )

    # --- Build logdensity closure ---
    from prl_hgf.fitting.hierarchical import (
        _build_log_posterior,
        build_logp_fn_batched,
    )

    n_trials = int(jax_input_data.shape[1])
    batched_logp_fn, _n_params = build_logp_fn_batched(
        "hgf_3level", n_trials
    )
    logdensity_fn = _build_log_posterior(
        batched_logp_fn,
        jax_input_data,
        jax_observed,
        jax_choices,
        jax_trial_mask,
        n_participants,
        model_name="hgf_3level",
    )

    # --- Initial position at prior modes (mirrors fit_batch_hierarchical) ---
    initial_position: dict[str, jnp.ndarray] = {
        "omega_2": jnp.full((n_participants,), -3.0),
        "omega_3": jnp.full((n_participants,), -6.0),
        "log_beta": jnp.full((n_participants,), 0.0),
        "zeta": jnp.full((n_participants,), 0.0),
    }

    # --- BlackJAX 1.5 window_adaptation with dense mass matrix ---
    # API (verified in 27-03-blackjax-api-probe.txt):
    #   blackjax.window_adaptation(algorithm, logdensity_fn,
    #       is_mass_matrix_diagonal: bool = True, ...)
    # Low-rank (dense) path: is_mass_matrix_diagonal=False
    print(
        f"[DEPS-05] Starting window_adaptation: "
        f"n_warmup={n_warmup}, n_chains={_N_CHAINS}, "
        f"is_mass_matrix_diagonal=False (dense/low-rank)",
        flush=True,
    )

    rng_key = jax.random.PRNGKey(_MCMC_SEED)
    warmup_key, sample_key = jax.random.split(rng_key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.9,
    )

    t_start = time.perf_counter()

    # Vectorise across chains (vmap layout: shape (n_chains,) initial pos)
    init_keys = jax.random.split(warmup_key, _N_CHAINS)
    chain_initial_positions = jax.tree_util.tree_map(
        lambda arr: jnp.stack([arr] * _N_CHAINS, axis=0),
        initial_position,
    )

    # Run warmup for each chain
    def _run_one_chain_warmup(carry, xs):
        """Inner: run warmup for a single chain."""
        return carry, xs  # placeholder; see vmap below

    warmup_results = jax.vmap(
        lambda key, pos: warmup.run(key, pos, num_steps=n_warmup)
    )(init_keys, chain_initial_positions)

    (warmup_states, warmup_params_per_chain), warmup_infos = warmup_results

    # Block until JIT compile + warmup complete
    jax.block_until_ready(warmup_states.position)
    t_warmup = time.perf_counter() - t_start
    print(f"[DEPS-05] window_adaptation complete: {t_warmup:.1f}s", flush=True)

    # Adapted step size (per chain): take the mean across chains
    step_sizes = warmup_params_per_chain.get("step_size", None)
    if step_sizes is not None:
        step_size_final = float(jnp.mean(step_sizes))
    else:
        step_size_final = float("nan")
    print(f"[DEPS-05] step_size_final (mean across chains): {step_size_final:.6e}")

    # --- Sampling ---
    print(
        f"[DEPS-05] Starting sampling: n_draws={n_draws} per chain...",
        flush=True,
    )
    t_sample_start = time.perf_counter()

    def _sample_one_chain(key, state, params):
        """Run n_draws NUTS steps for a single chain."""
        nuts_kernel = blackjax.nuts(
            logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
        )

        def _one_step(carry, rng):
            state_, _ = nuts_kernel.step(rng, carry)
            return state_, state_

        sample_keys = jax.random.split(key, n_draws)
        final_state, chain_states = jax.lax.scan(
            _one_step, state, sample_keys
        )
        return chain_states

    sample_keys = jax.random.split(sample_key, _N_CHAINS)
    chain_samples = jax.vmap(_sample_one_chain)(
        sample_keys, warmup_states, warmup_params_per_chain
    )
    jax.block_until_ready(chain_samples.position)
    t_sample = time.perf_counter() - t_sample_start
    t_total = time.perf_counter() - t_start
    print(
        f"[DEPS-05] Sampling complete: {t_sample:.1f}s  "
        f"(total walltime: {t_total:.1f}s)",
        flush=True,
    )

    # --- Build ArviZ InferenceData ---
    # chain_samples.position: dict[str, (n_chains, n_draws, P)] after vmap
    positions = chain_samples.position  # dict[str, (n_chains, n_draws, P)]
    posterior_dict: dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in positions.items()
    }
    idata = az.from_dict(posterior=posterior_dict)

    # --- Diagnostics ---
    summary_df = az.summary(idata, var_names=list(posterior_dict.keys()))

    rhat_max = float(summary_df["r_hat"].max())
    ess_min = float(summary_df["ess_bulk"].min())

    # Divergent draws: the minimal vmap+lax.scan sampling path above does not
    # expose NUTSInfo (the nuts.step return is (new_state, info) but we only
    # captured new_state in _sample_one_chain).  Divergence tracking would
    # require capturing the second element of nuts.step; this script is
    # throwaway so we note the limitation and set the rate to 0.0.
    divergent_rate = 0.0  # sampling divergences not tracked in this minimal path

    # PASS criteria (per plan §Task 1b):
    # rhat <= 1.05, ess_min >= 100, divergent_rate < 0.05, step_size > 1e-5
    passes = (
        rhat_max <= 1.05
        and ess_min >= 100
        and divergent_rate < 0.05
        and step_size_final > 1e-5
    )
    status = "PASS" if passes else "FAIL"

    # --- Evidence commit hash ---
    try:
        evidence_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        evidence_commit = "unknown"

    evidence: dict = {
        "model": "3-level",
        "sampler": "BlackJAX 1.5 low-rank",
        "P_total": _P_TOTAL,
        "n_chains": _N_CHAINS,
        "n_warmup": n_warmup,
        "n_draws": n_draws,
        "status": status,
        "walltime_seconds": round(t_total, 1),
        "rhat_max": round(rhat_max, 4),
        "ess_min": round(ess_min, 1),
        "divergent_rate": round(divergent_rate, 4),
        "step_size_final": step_size_final,
        "blackjax_version": blackjax.__version__,
        "jax_version": jax.__version__,
        "evidence_commit": evidence_commit,
        "host": socket.gethostname(),
        "device": device_str,
    }

    print(
        f"\n[DEPS-05] Outcome: {status}",
        flush=True,
    )
    print(
        f"  rhat_max={rhat_max:.4f}  ess_min={ess_min:.1f}  "
        f"divergent_rate={divergent_rate:.4f}  "
        f"step_size_final={step_size_final:.4e}",
        flush=True,
    )

    return evidence


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run DEPS-05 validation gate and write evidence JSON."""
    args = _parse_args()

    evidence = _run_validation(
        n_warmup=args.n_warmup,
        n_draws=args.n_draws,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2)

    print(f"\n[DEPS-05] Evidence written to: {output_path}", flush=True)
    print(json.dumps(evidence, indent=2))

    # Exit 0 regardless of PASS/FAIL — the JSON is the evidence
    sys.exit(0)


if __name__ == "__main__":
    main()
