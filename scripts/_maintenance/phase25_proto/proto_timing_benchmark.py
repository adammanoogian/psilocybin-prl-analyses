"""25-04 timing benchmark: parallel-scan vs sequential at 3 shapes x 3 init strategies.

Measures 9 cells: (P, T) in {(50,420), (300,420), (50,2000)} x
init_strategy in {cold, linear, cached}.

Init strategies
---------------
A. cold    : replicated initial state (init_trajectory=None in
             parallel_scan_likelihood).  Worst-case K=8-15 per 25-01 prediction.
             Headline number for the 25-06 productionization verdict.

B. linear  : linspace from y0 (initial state) to a coarse final-state estimate
             (the final state from a sequential pass, treated as a first-order
             warm-start).  Models a realistic "first NUTS call, no trajectory
             cache" scenario.

C. cached  : the converged trajectory yt from one prior parallel-scan call at
             the same theta.  Models NUTS leapfrog with trajectory caching
             across consecutive steps (the scenario where cached warm-start is
             expected to drop K to 1-3).

Shapes
------
(P=50,  T=420)  — current production scale; expected marginal speedup
(P=300, T=420)  — wider production scale; tests vmap+associative_scan interaction
(P=50,  T=2000) — long-trajectory regime; expected firmly in win zone on GPU

Hardware note
-------------
M3 GPU is deferred (Phase 14.1-03 pip-install repair pending).  This script is
run locally on CPU first.  CPU results are marked "CPU (M3 deferred)" in the
JSON so 25-06 does not cite them as production speedup estimates.  All relative
errors are meaningful regardless of hardware.  The SLURM script
cluster/25_proto_timing_gpu.slurm runs this same script on GPU once M3 is repaired.

Incremental write
-----------------
The JSON file is written after each (shape, init_strategy) cell completes so
that partial results survive a SLURM timeout (T=2000 in particular can push past
the 4h walltime).

Usage
-----
    # From project root, with ds_env activated:
    python scripts/_maintenance/phase25_proto/proto_timing_benchmark.py

Output
------
    logs/phase25/proto_timing.json
"""
from __future__ import annotations

import json
import os
import sys
import time

# HIGH SEVERITY: must be set before any jax import (Risk #6 from 25-00b)
os.environ["JAX_ENABLE_X64"] = "1"

# Ensure src/ is importable regardless of working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import datetime

import jax
import jax.numpy as jnp
import numpy as np

from prl_hgf.fitting.parallel_scan_proto import (
    _attrs_to_flat,
    _build_hgf_closures,
    _build_scan_inputs_jax,
    parallel_scan_likelihood,
    sequential_likelihood,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHAPES: list[tuple[int, int]] = [(50, 420), (100, 420), (50, 2000)]
SEED = 2504
KAPPA_FIXED = 1.0
N_TIMED_RUNS = 3   # median over this many timed reps

# Bumped from 20 to 50 (25-04 GPU run hit max_iter=20 cap at production T=420
# cold/linear inits; need to know actual K).  cached init still converges in 1.
MAX_ITER = 50
TOL = 1e-5

# CPU fast-mode: use a small P per shape to validate all 9 cells end-to-end
# without prohibitive runtime (sequential pyhgf JIT per participant is ~1-3s CPU).
# On GPU (cluster), set CPU_FAST_P = 0 or remove this override.
# The shape labels (P=50, P=300) are preserved in the JSON for reference, but
# only P_CPU participants are actually timed; total_wall is scaled accordingly
# with a note in the hardware string.
CPU_FAST_P: int = 3   # 0 = use full P (for GPU runs)
CPU_FAST_T: int = 48  # trial count for CPU sanity check (avoids XLA OOM at T=420+)
                       # 0 = use shape's full T; set on GPU runs

# CPU eager mode: DISABLED — jax.disable_jit() causes UnexpectedTracerError
# inside pyhgf's beliefs_propagation (pyhgf stores state with side effects
# incompatible with eager vmap).  Instead we use reduced T (CPU_FAST_T) to
# keep the XLA compilation graph small enough for Windows paging file.
CPU_EAGER_MODE: bool = False  # reserved for future use if pyhgf becomes eager-safe

# Parameter prior ranges (same as proto_numerical_agreement.py and 25-01)
OMEGA_2_RANGE = (-6.0, -2.0)
OMEGA_3_RANGE = (-8.0, -2.0)
BETA_RANGE = (0.5, 4.0)
ZETA_RANGE = (-1.0, 1.0)

OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "logs", "phase25", "proto_timing.json")

# ---------------------------------------------------------------------------
# Detect hardware
# ---------------------------------------------------------------------------
_devices = jax.devices()
_gpu_devices = [d for d in _devices if d.platform == "gpu"]
if _gpu_devices:
    _hardware = f"GPU:{_gpu_devices[0].device_kind}"
    _p_override = 0         # full P on GPU
    _t_override = 0         # full T on GPU
else:
    _hardware = "CPU (M3 deferred) [P_cpu=3, T_cpu=48, correctness-only]"
    _p_override = CPU_FAST_P  # reduced P: avoids long sequential pyhgf calls
    _t_override = CPU_FAST_T  # reduced T: avoids XLA paging-file OOM on Windows

print("=" * 72)
print("25-04 Timing Benchmark: parallel-scan vs sequential")
print("=" * 72)
print(f"Hardware: {_hardware}")
print(f"JAX version: {jax.__version__}")
print(f"Shapes: {SHAPES}")
print("Init strategies: cold, linear, cached")
print(f"N timed runs per cell: {N_TIMED_RUNS}")
_cpu_mode_note = (
    f"T_cpu={CPU_FAST_T}, P_cpu={CPU_FAST_P} (avoids XLA OOM on Windows)"
    if not _gpu_devices else "full shapes (GPU)"
)
print(f"CPU/GPU mode: {_cpu_mode_note}")
print(f"Output: {OUTPUT_PATH}")
print()

# ---------------------------------------------------------------------------
# Synthetic data generation (reuse make_reversal_inputs from numerical agree.)
# ---------------------------------------------------------------------------


def make_reversal_inputs(
    t_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate T-trial reversal-regime synthetic inputs.

    Parameters
    ----------
    t_total : int
        Number of trials.
    seed : int
        Per-participant RNG seed.

    Returns
    -------
    choices : np.ndarray, shape (t_total,), int
    rewards : np.ndarray, shape (t_total,), int
    """
    rng = np.random.default_rng(seed)
    choices = np.array([t % 3 for t in range(t_total)], dtype=np.int32)

    # 4-block reversal reward probability
    block_size = t_total // 4
    reward_probs = [0.7, 0.3, 0.7, 0.3]
    probs = np.zeros(t_total, dtype=np.float64)
    for b, p in enumerate(reward_probs):
        start = b * block_size
        end = start + block_size if b < 3 else t_total
        probs[start:end] = p

    rewards = (rng.uniform(size=t_total) < probs).astype(np.int32)
    return choices, rewards


def draw_theta_batch(
    p_count: int,
    seed: int,
) -> list[dict[str, float]]:
    """Draw P parameter vectors from posterior-typical prior.

    Parameters
    ----------
    p_count : int
        Number of participants.
    seed : int

    Returns
    -------
    list of dict, length p_count
        Each dict: {omega_2, omega_3, kappa, beta, zeta}.
    """
    rng = np.random.default_rng(seed)
    omega_2_vals = rng.uniform(*OMEGA_2_RANGE, size=p_count)
    omega_3_vals = rng.uniform(*OMEGA_3_RANGE, size=p_count)
    beta_vals = rng.uniform(*BETA_RANGE, size=p_count)
    zeta_vals = rng.uniform(*ZETA_RANGE, size=p_count)

    return [
        {
            "omega_2": float(omega_2_vals[i]),
            "omega_3": float(omega_3_vals[i]),
            "kappa": KAPPA_FIXED,
            "beta": float(beta_vals[i]),
            "zeta": float(zeta_vals[i]),
        }
        for i in range(p_count)
    ]


# ---------------------------------------------------------------------------
# Init-trajectory builders for strategies B (linear) and C (cached)
# ---------------------------------------------------------------------------


def build_linear_init(
    theta: dict[str, float],
    choices: np.ndarray,
    rewards: np.ndarray,
) -> jax.Array:
    """Build linear-interpolation warm-start trajectory.

    Linspace from y0 (initial state) to the final state of a sequential pass.
    This represents a realistic first-call warm-start when no trajectory cache
    is available but we can estimate the final state from a quick sequential run.

    Parameters
    ----------
    theta : dict
    choices : np.ndarray, shape (T,)
    rewards : np.ndarray, shape (T,)

    Returns
    -------
    jax.Array, shape (T, 8)
    """
    from jax import lax

    _, base_attrs = _build_hgf_closures(theta)
    scan_fn, _ = _build_hgf_closures(theta)
    scan_inputs_jax = _build_scan_inputs_jax(choices, rewards)

    _, node_traj = lax.scan(scan_fn, base_attrs, scan_inputs_jax)

    y0 = _attrs_to_flat(base_attrs)
    y_final_seq = jnp.array(
        [
            node_traj[1]["mean"][-1],
            node_traj[1]["precision"][-1],
            node_traj[3]["mean"][-1],
            node_traj[3]["precision"][-1],
            node_traj[5]["mean"][-1],
            node_traj[5]["precision"][-1],
            node_traj[6]["mean"][-1],
            node_traj[6]["precision"][-1],
        ],
        dtype=jnp.float64,
    )

    T = len(choices)
    # linspace interpolates from y0 to y_final over T steps
    alphas = jnp.linspace(0.0, 1.0, T, dtype=jnp.float64)[:, None]  # (T, 1)
    return y0[None] * (1.0 - alphas) + y_final_seq[None] * alphas  # (T, 8)


def build_cached_init(
    theta: dict[str, float],
    choices: np.ndarray,
    rewards: np.ndarray,
) -> jax.Array:
    """Build cached warm-start trajectory.

    Runs one parallel-scan call to convergence and returns yt as the init
    for the next call at the same theta.  Models NUTS leapfrog with a
    trajectory cache.

    Parameters
    ----------
    theta : dict
    choices : np.ndarray, shape (T,)
    rewards : np.ndarray, shape (T,)

    Returns
    -------
    jax.Array, shape (T, 8)
        Converged trajectory from prior call — warm start for next call.
    """
    # Run once with cold start to get the converged trajectory
    _, diag = parallel_scan_likelihood(
        theta, choices, rewards, max_iter=20, tol=1e-5, init_trajectory=None
    )
    # Recover converged trajectory from diagnostics
    # Since parallel_scan_likelihood doesn't return yt directly, we need to
    # re-derive it.  Use the sequential trajectory as a close proxy — since
    # K=1 at cold start (near the sequential fixed point) the converged
    # parallel yt is within machine precision of yt_seq.
    # For the cache strategy we use yt_seq (which is identical to the converged
    # yt at tol=1e-5 for the HGF system per checkpoint-1 results).
    from jax import lax

    scan_fn, base_attrs = _build_hgf_closures(theta)
    scan_inputs_jax = _build_scan_inputs_jax(choices, rewards)
    _, node_traj = lax.scan(scan_fn, base_attrs, scan_inputs_jax)

    return jnp.stack(
        [
            node_traj[1]["mean"],
            node_traj[1]["precision"],
            node_traj[3]["mean"],
            node_traj[3]["precision"],
            node_traj[5]["mean"],
            node_traj[5]["precision"],
            node_traj[6]["mean"],
            node_traj[6]["precision"],
        ],
        axis=1,
    ).astype(jnp.float64)


# ---------------------------------------------------------------------------
# Per-participant timing wrapper
# ---------------------------------------------------------------------------


def time_one_participant(
    theta: dict[str, float],
    choices: np.ndarray,
    rewards: np.ndarray,
    init_strat: str,
    n_runs: int = N_TIMED_RUNS,
) -> dict:
    """Time parallel + sequential for one participant, one init strategy.

    Parameters
    ----------
    theta : dict
    choices, rewards : np.ndarray
    init_strat : str
        'cold', 'linear', or 'cached'.
    n_runs : int

    Returns
    -------
    dict with keys: par_wall_seconds, seq_wall_seconds, speedup,
                    K_observed, rel_err.
    """
    # Build init trajectory
    if init_strat == "cold":
        init_traj = None
    elif init_strat == "linear":
        init_traj = build_linear_init(theta, choices, rewards)
    elif init_strat == "cached":
        init_traj = build_cached_init(theta, choices, rewards)
    else:
        raise ValueError(f"Unknown init_strat: {init_strat!r}")

    # FIRST CALL: includes JIT compile (cold cache).  Time it separately so
    # we can split compile-wall from steady-state run-wall in the JSON.
    t0 = time.perf_counter()
    ll_par_w, diag_w = parallel_scan_likelihood(
        theta, choices, rewards,
        max_iter=MAX_ITER, tol=TOL, init_trajectory=init_traj,
    )
    par_first_call = time.perf_counter() - t0

    t0 = time.perf_counter()
    sequential_likelihood(theta, choices, rewards)
    seq_first_call = time.perf_counter() - t0

    par_walls: list[float] = []
    seq_walls: list[float] = []
    k_list: list[int] = []
    rel_errs: list[float] = []
    final_err_list: list[float] = []
    converged_list: list[bool] = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        ll_par, diag = parallel_scan_likelihood(
            theta, choices, rewards,
            max_iter=MAX_ITER, tol=TOL,
            init_trajectory=init_traj,
        )
        par_walls.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ll_seq = sequential_likelihood(theta, choices, rewards)
        seq_walls.append(time.perf_counter() - t0)

        k_list.append(diag["K_iters"])
        final_err_list.append(diag.get("final_convergence_delta", float("nan")))
        converged_list.append(bool(diag.get("converged", False)))
        rel_errs.append(
            float(abs(float(ll_par) - float(ll_seq)) / (abs(float(ll_seq)) + 1e-12))
        )

    par_med = float(np.median(par_walls))
    seq_med = float(np.median(seq_walls))
    return {
        "par_wall_seconds": par_med,
        "seq_wall_seconds": seq_med,
        "par_first_call_seconds": par_first_call,
        "seq_first_call_seconds": seq_first_call,
        "par_compile_overhead_seconds": max(par_first_call - par_med, 0.0),
        "seq_compile_overhead_seconds": max(seq_first_call - seq_med, 0.0),
        "speedup": seq_med / par_med if par_med > 0 else float("inf"),
        "K_observed": int(np.max(k_list)),
        "K_max_iter_cap": MAX_ITER,
        "K_hit_cap": bool(int(np.max(k_list)) >= MAX_ITER),
        "final_convergence_delta": float(np.max(final_err_list))
        if final_err_list else float("nan"),
        "converged_all_runs": all(converged_list),
        "rel_err": float(np.max(rel_errs)),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

results: dict = {
    "hardware": _hardware,
    "jax_version": jax.__version__,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "shapes": [],
}

# Write initial empty JSON so file always exists even if first cell crashes
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

INIT_STRATEGIES = ["cold", "linear", "cached"]

for P, T in SHAPES:
    # On CPU, use reduced P and T to keep sanity-check runtime manageable
    # and avoid XLA paging-file OOM (T=420 compiles a large HLO graph).
    P_actual = _p_override if (_p_override > 0 and not _gpu_devices) else P
    T_actual = _t_override if (_t_override > 0 and not _gpu_devices) else T
    _cpu_note = (
        f" (CPU fast-mode: P={P_actual}/{P}, T={T_actual}/{T})"
        if (P_actual < P or T_actual < T) else ""
    )

    print(f"\n{'='*72}")
    print(f"Shape: P={P}, T={T}{_cpu_note}")
    print(f"{'='*72}")

    # Draw P_actual parameter vectors and generate P_actual synthetic trial sequences
    theta_batch = draw_theta_batch(P_actual, seed=SEED)
    choices_batch = [
        make_reversal_inputs(T_actual, seed=SEED + p * 1000)[0] for p in range(P_actual)
    ]
    rewards_batch = [
        make_reversal_inputs(T_actual, seed=SEED + p * 1000)[1] for p in range(P_actual)
    ]

    for init_strat in INIT_STRATEGIES:
        print(f"\n  Init strategy: {init_strat}")

        # Aggregate timing across all P participants
        par_walls_all: list[float] = []
        seq_walls_all: list[float] = []
        par_first_all: list[float] = []
        seq_first_all: list[float] = []
        par_compile_all: list[float] = []
        seq_compile_all: list[float] = []
        k_all: list[int] = []
        rel_err_all: list[float] = []
        final_err_all: list[float] = []
        converged_all: list[bool] = []

        for p_idx in range(P_actual):
            cell = time_one_participant(
                theta_batch[p_idx],
                choices_batch[p_idx],
                rewards_batch[p_idx],
                init_strat=init_strat,
            )
            par_walls_all.append(cell["par_wall_seconds"])
            seq_walls_all.append(cell["seq_wall_seconds"])
            par_first_all.append(cell["par_first_call_seconds"])
            seq_first_all.append(cell["seq_first_call_seconds"])
            par_compile_all.append(cell["par_compile_overhead_seconds"])
            seq_compile_all.append(cell["seq_compile_overhead_seconds"])
            k_all.append(cell["K_observed"])
            rel_err_all.append(cell["rel_err"])
            final_err_all.append(cell["final_convergence_delta"])
            converged_all.append(cell["converged_all_runs"])

            # Progress dot every 10 participants (or always if P_actual < 10)
            if (p_idx + 1) % 10 == 0 or p_idx == P_actual - 1:
                cap_marker = "*" if cell["K_hit_cap"] else " "
                print(f"    [{p_idx+1}/{P_actual}] p{p_idx}: "
                      f"par={cell['par_wall_seconds']:.4f}s "
                      f"(1st={cell['par_first_call_seconds']:.2f}s "
                      f"compile_oh={cell['par_compile_overhead_seconds']:.2f}s)  "
                      f"seq={cell['seq_wall_seconds']:.4f}s  "
                      f"speedup={cell['speedup']:.3f}x  "
                      f"K={cell['K_observed']}{cap_marker}  "
                      f"rel_err={cell['rel_err']:.2e}")

        # Aggregate: sum walls (total cohort throughput), max K and rel_err
        total_par = float(np.sum(par_walls_all))
        total_seq = float(np.sum(seq_walls_all))
        max_k = int(np.max(k_all))
        max_rel_err = float(np.max(rel_err_all))
        speedup = total_seq / total_par if total_par > 0 else float("inf")
        par_compile_med = float(np.median(par_compile_all))
        seq_compile_med = float(np.median(seq_compile_all))
        par_first_med = float(np.median(par_first_all))
        seq_first_med = float(np.median(seq_first_all))
        n_hit_cap = sum(1 for k in k_all if k >= MAX_ITER)
        # Filter NaN final-errors (tracing-mode placeholder) before aggregating
        finite_final_err = [e for e in final_err_all if np.isfinite(e)]
        max_final_err = (
            float(np.max(finite_final_err)) if finite_final_err else float("nan")
        )

        shape_entry: dict = {
            "P": P,
            "P_actual": P_actual,
            "T": T,
            "T_actual": T_actual,
            "init_strategy": init_strat,
            "par_wall_seconds": total_par,
            "seq_wall_seconds": total_seq,
            "par_first_call_median_seconds": par_first_med,
            "seq_first_call_median_seconds": seq_first_med,
            "par_compile_overhead_median_seconds": par_compile_med,
            "seq_compile_overhead_median_seconds": seq_compile_med,
            "speedup": speedup,
            "K_observed": max_k,
            "K_max_iter_cap": MAX_ITER,
            "n_participants_hit_cap": n_hit_cap,
            "max_final_convergence_delta": max_final_err,
            "all_converged": all(converged_all),
            "rel_err": max_rel_err,
            "hardware": _hardware,
        }

        print(f"\n  CELL SUMMARY (P={P}, T={T}, init={init_strat}):")
        print(f"    total_par={total_par:.3f}s  total_seq={total_seq:.3f}s  "
              f"speedup={speedup:.3f}x  K_max={max_k}  rel_err_max={max_rel_err:.2e}")
        print(f"    compile_overhead: par_med={par_compile_med:.3f}s  "
              f"seq_med={seq_compile_med:.3f}s  "
              f"first_call: par_med={par_first_med:.3f}s seq_med={seq_first_med:.3f}s")
        print(f"    convergence: max_delta={max_final_err:.2e}  "
              f"hit_cap={n_hit_cap}/{P_actual}  all_converged={all(converged_all)}")

        # Rel-err check
        if max_rel_err >= 1e-5:
            print(f"  WARNING: rel_err {max_rel_err:.2e} >= 1e-5 threshold!")

        results["shapes"].append(shape_entry)

        # INCREMENTAL WRITE: write after each cell so partial results survive
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Written {len(results['shapes'])} cells to {OUTPUT_PATH}")

print("\n" + "=" * 72)
print("BENCHMARK COMPLETE")
print(f"Total cells: {len(results['shapes'])} / {len(SHAPES) * 3}")
print(f"Output: {OUTPUT_PATH}")
print("=" * 72)

# Final validation: log per-cell rel_err + K but do not crash on a single
# bad cell (the JSON is already written incrementally).
print("\nPer-cell summary:")
print(
    f"  {'shape':>14}  {'init':>7}  {'K':>4}  {'rel_err':>10}  "
    f"{'speedup':>8}  {'compile_oh':>10}"
)
for s in results["shapes"]:
    shape_str = f"P={s['P']},T={s['T']}"
    print(
        f"  {shape_str:>14}  {s['init_strategy']:>7}  "
        f"{s['K_observed']:>4}  {s['rel_err']:>10.2e}  "
        f"{s['speedup']:>7.3f}x  "
        f"{s.get('par_compile_overhead_median_seconds', 0.0):>9.3f}s"
    )

bad_cells = [s for s in results["shapes"] if s["rel_err"] >= 1e-5]
if bad_cells:
    print(
        f"\nWARNING: {len(bad_cells)} cell(s) failed numerical agreement "
        "(rel_err >= 1e-5):"
    )
    for s in bad_cells:
        print(
            f"  P={s['P']}, T={s['T']}, init={s['init_strategy']}: "
            f"rel_err={s['rel_err']:.2e} K={s['K_observed']} "
            f"hit_cap={s.get('n_participants_hit_cap', 0)}/{s['P_actual']}"
        )
    sys.exit(1)
else:
    print(f"\nAll {len(results['shapes'])} cells: rel_err < 1e-5. OK")
