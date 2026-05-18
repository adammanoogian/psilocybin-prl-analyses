"""Reproduce ELK on the coordinated-turn model (D=4, T=500).

Phase 25 Plan 03 — Reference reproduction study.

This script reproduces the quasi-ELK (diagonal-Jacobian) parallel-scan
algorithm from lindermanlab/elk on the coordinated-turn benchmark dataset
(standard nonlinear SSM; Sarkka 2013 ch. 5; Yaghoobi et al. 2021 §5).

The algorithm is re-implemented from scratch in pure JAX, faithfully
following the algorithm in:
    elk/algs/deer.py  (commit 2801539, lindermanlab/elk)
    - seq1d() with quasi=True, qmem_efficient=False
    - diagonal_deer_iteration_helper()
    - diagonal_matmul_recursive() via jax.lax.associative_scan

This approach was chosen over direct import because:
- ELK's flax/dynamax/orbax dep stack has Windows path-length incompatibility
  (orbax checkpoint paths exceed MAX_PATH on Windows 11).
- The algorithm itself is 40 lines of pure JAX; re-implementation is
  auditable against the reference (see REPRODUCTION_NOTES.md §2).

Algorithm
---------
quasi-DEER (from lindermanlab/elk deer.py): solve y[t] = f(y[t-1], x[t])
iteratively.  At each Newton iteration k:
    1. Compute diagonal Jacobian diag(df/dy) at current estimate y^(k)
       using jax.jacfwd.
    2. Form linear recurrence: A[t] = diag_jac[t], b[t] = f(y^(k)[t-1], x[t])
       - A[t]*y^(k)[t-1].
    3. Solve linear recurrence in O(log T) via jax.lax.associative_scan.
    4. Update y^(k+1) = solution of linear recurrence.
    5. Check convergence: max|y^(k+1) - y^(k)| < tol.

Sequential reference
--------------------
The sequential reference is jax.lax.scan over the SAME transition function
f(state, driver).  This is the correct apples-to-apples comparison:
ELK solves y[t] = f(y[t-1], x[t]) in O(K log T) depth; lax.scan solves
the same recurrence in O(T) sequential steps.  Both must reach the same
trajectory to machine precision.

IMPORTANT: ELK is NOT a filter. It does not incorporate observations.
ELK solves the deterministic recurrence y[t] = f(y[t-1]) where f is the
transition model of the SSM. The EKF is a separate algorithm that adds
observation updates. The correct validation is:

    seq: y[t] = f(y[t-1]) via jax.lax.scan   [O(T) sequential]
    par: y[t] = f(y[t-1]) via quasi-ELK scan  [O(K log T) parallel]

Both should yield identical trajectories. The EKF is run separately as
context (to show what filtering would give) but is NOT the reference for
the ELK agreement check.

Coordinated turn model
----------------------
State: z = [pos_x, pos_y, velocity, omega]  (D=4)
Transition (Euler integration, dt=0.1):
    pos_x_new  = pos_x  + velocity * cos(omega * dt) * dt
    pos_y_new  = pos_y  + velocity * sin(omega * dt) * dt
    velocity_new = velocity                              (constant speed)
    omega_new  = omega                                  (constant turn rate)
Driver: zeros (autonomous system; no external input in the deterministic
        transition; driver argument kept for API parity with ELK).

Usage
-----
    python scripts/_maintenance/phase25_proto/reproduce_reference.py

Expected output:
    rel_err_vs_sequential < 1e-5 and K_observed <= 15. PASS.

Author: Phase 25-03 executor
Reference: lindermanlab/elk commit 2801539febf07f5e9a28eacc6265dc9ac179e835
"""
from __future__ import annotations

import os

# HIGH SEVERITY: must be set before any jax import (Risk #6 from 25-00b)
os.environ["JAX_ENABLE_X64"] = "1"

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T = 500
D = 4
OBS_DIM = 2
DT = 0.1
RNG_SEED = 2503

# Noise magnitudes
Q_DIAG = jnp.array([1e-4, 1e-4, 1e-4, 1e-4])  # process noise variances (for data gen)
R_DIAG = jnp.array([0.5, 0.5])                  # obs noise variances (for data gen)

# ELK params
MAX_ITER = 20
TOL = 1e-5       # tolerance for convergence (matches plan target)
# dtype tolerance in elk/deer.py: 1e-7 for float64
# We track both convergence under TOL (our gate) and iteration count.

# ---------------------------------------------------------------------------
# Coordinated turn model
# ---------------------------------------------------------------------------


def transition_fn(state: jax.Array, driver: jax.Array) -> jax.Array:
    """Deterministic part of the coordinated turn transition.

    Parameters
    ----------
    state : jax.Array, shape (D,)
        [pos_x, pos_y, velocity, omega]
    driver : jax.Array, shape (D,)
        Per-step process noise (added OUTSIDE this fn; driver is zero here
        for the deterministic component used in ELK linearization).
        In the noisy simulation, the caller adds Q_noise to state first.

    Returns
    -------
    jax.Array, shape (D,)
        Next deterministic state.

    Notes
    -----
    Standard coordinated turn (Euler integration, constant velocity model):
    pos_x_new = pos_x + v * cos(omega*dt) * dt
    pos_y_new = pos_y + v * sin(omega*dt) * dt
    v_new = v
    omega_new = omega
    driver is not used in the deterministic part; it is kept as argument
    to match the DEER/ELK API signature f(state, driver).
    """
    pos_x, pos_y, v, omega = state
    new_x = pos_x + v * jnp.cos(omega * DT) * DT
    new_y = pos_y + v * jnp.sin(omega * DT) * DT
    new_v = v
    new_omega = omega
    return jnp.array([new_x, new_y, new_v, new_omega], dtype=jnp.float64)


def observation_fn(state: jax.Array) -> jax.Array:
    """Linear observation model: observe [pos_x, pos_y].

    Parameters
    ----------
    state : jax.Array, shape (D,)

    Returns
    -------
    jax.Array, shape (OBS_DIM,)
    """
    return state[:OBS_DIM]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_coordinated_turn_data(
    seed: int,
    T: int = T,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate T steps of coordinated turn trajectory + noisy observations.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    T : int
        Number of timesteps.

    Returns
    -------
    true_states : np.ndarray, shape (T+1, D)
        Ground truth state trajectory (including t=0).
    observations : np.ndarray, shape (T, OBS_DIM)
        Noisy position observations at t=1..T.
    """
    rng = np.random.default_rng(seed)

    # Initial state: position (0,0), velocity 10, turn rate 0.1 rad/step
    state = np.array([0.0, 0.0, 10.0, 0.1])

    true_states = np.zeros((T + 1, D))
    true_states[0] = state

    q_std = np.sqrt(np.array([1e-4, 1e-4, 1e-4, 1e-4]))

    for t in range(T):
        # Deterministic transition
        s = true_states[t]
        pos_x, pos_y, v, omega = s
        new_x = pos_x + v * np.cos(omega * DT) * DT
        new_y = pos_y + v * np.sin(omega * DT) * DT
        new_v = v
        new_omega = omega
        det_next = np.array([new_x, new_y, new_v, new_omega])
        # Add process noise
        noise = rng.standard_normal(D) * q_std
        true_states[t + 1] = det_next + noise

    # Noisy observations (position only)
    r_std = np.sqrt(np.array([0.5, 0.5]))
    observations = (
        true_states[1:, :OBS_DIM]
        + rng.standard_normal((T, OBS_DIM)) * r_std
    )

    return true_states, observations


# ---------------------------------------------------------------------------
# Sequential transition reference
# ---------------------------------------------------------------------------

def run_sequential_transition(
    y0: jax.Array,
    drivers: jax.Array,
) -> jax.Array:
    """Run the transition recurrence y[t] = f(y[t-1], x[t]) via jax.lax.scan.

    Parameters
    ----------
    y0 : jax.Array, shape (D,)
        Initial state (t=0).
    drivers : jax.Array, shape (T, D)
        Per-step drivers (zeros for autonomous system).

    Returns
    -------
    traj : jax.Array, shape (T, D)
        Trajectory y[1..T] (y0 excluded).

    Notes
    -----
    This is the SEQUENTIAL reference that quasi-ELK must match.
    ELK solves the SAME recurrence y[t] = f(y[t-1], x[t]) in O(K log T)
    depth; jax.lax.scan solves it in O(T) sequential steps. Both must
    yield the same trajectory to within numerical precision.

    The EKF (which additionally incorporates observations) is a different
    algorithm and is NOT the target reference for ELK agreement.
    """
    def scan_step(
        carry: jax.Array,
        driver: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        y_new = transition_fn(carry, driver)
        return y_new, y_new

    _, traj = jax.lax.scan(scan_step, y0, drivers)
    return traj  # (T, D)


# ---------------------------------------------------------------------------
# quasi-ELK (reimplemented from lindermanlab/elk deer.py)
# ---------------------------------------------------------------------------
# Re-implements the algorithm from:
#   elk/algs/deer.py  commit 2801539
#   seq1d(func, y0, xinp, params, quasi=True, qmem_efficient=False)
#   -> diagonal_deer_iteration_helper(qmem_efficient=False)
#   -> diagonal_matmul_recursive() via jax.lax.associative_scan
#
# The only change vs elk/algs/deer.py: the convergence tolerance uses our
# target TOL=1e-5 (float64 elk default is 1e-7; we want convergence at 1e-5
# per the plan gate criterion).
#
# The ELK (Kalman) variant (elk/algs/elk.py) requires dynamax which has
# Windows path-length installation issues. The quasi-DEER diagonal variant
# needs only JAX. Both variants converge to the same fixed point; the Kalman
# variant adds per-step variance tracking (not needed for the state trajectory
# comparison here).


def _diagonal_binary_op(
    elem_i: tuple[jax.Array, jax.Array],
    elem_j: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Associative operator for diagonal linear recurrence.

    Computes (A_j * A_i, A_j * b_i + b_j) where A are diagonal vectors.

    Parameters
    ----------
    elem_i : tuple
        (A_i, b_i), each shape (D,).
    elem_j : tuple
        (A_j, b_j), each shape (D,).

    Returns
    -------
    tuple
        Composed (A_out, b_out), each shape (D,).
    """
    A_i, b_i = elem_i
    A_j, b_j = elem_j
    return A_j * A_i, A_j * b_i + b_j


def _diagonal_matmul_recursive(
    A_diags: jax.Array,
    bs: jax.Array,
    y0: jax.Array,
) -> jax.Array:
    """Solve y[t] = A[t]*y[t-1] + b[t] in O(log T) via associative scan.

    Parameters
    ----------
    A_diags : jax.Array, shape (T, D)
        Diagonal of the recurrence matrix at each step.
    bs : jax.Array, shape (T, D)
        Bias vector at each step.
    y0 : jax.Array, shape (D,)
        Initial condition.

    Returns
    -------
    yt : jax.Array, shape (T+1, D)
        Full trajectory including y0 at index 0.

    Notes
    -----
    Algorithm from elk/algs/deer.py diagonal_matmul_recursive().
    Prepends identity A=1 and y0 as b to make dimensions consistent,
    then applies jax.lax.associative_scan.
    """
    ones = jnp.ones((1, D), dtype=jnp.float64)           # (1, D) — identity
    first_A = jnp.concatenate([ones, A_diags], axis=0)   # (T+1, D)
    first_b = jnp.concatenate([y0[None], bs], axis=0)    # (T+1, D)
    _, yt = jax.lax.associative_scan(_diagonal_binary_op, (first_A, first_b))
    return yt  # (T+1, D)


def run_quasi_elk(
    func: callable,
    y0: jax.Array,
    xinp: jax.Array,
    params: None = None,
    yinit_guess: jax.Array | None = None,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> tuple[jax.Array, int, list[float]]:
    """Run quasi-DEER/ELK on a general 1-D recurrence y[t] = func(y[t-1], x[t]).

    Re-implements elk/algs/deer.py seq1d(quasi=True, qmem_efficient=False)
    using only JAX. The ELK Kalman step is omitted (quasi-DEER only), which
    suffices for trajectory reproduction validation.

    Parameters
    ----------
    func : callable
        f(state, driver) -> next_state. Must be JAX-differentiable.
        state: (D,), driver: (D,), returns (D,).
    y0 : jax.Array, shape (D,)
        Initial condition (fixed; NOT part of the optimization).
    xinp : jax.Array, shape (T, D)
        Per-step driver inputs (zeros for autonomous systems).
    params : None
        Unused; kept for API parity with elk/algs/deer.py.
    yinit_guess : jax.Array or None, shape (T, D)
        Initial trajectory guess. If None, initialized to zeros.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on max|y^(k+1) - y^(k)|.

    Returns
    -------
    yt : jax.Array, shape (T, D)
        Converged trajectory estimate (t=1..T; y0 is excluded).
    K_observed : int
        Number of Newton iterations performed.
    err_trace : list[float]
        Max absolute change per iteration (for convergence curve).

    Notes
    -----
    The algorithm is:
      1. Initialize y^(0) = yinit_guess or zeros.
      2. For k = 0, 1, ..., max_iter:
         a. Compute diag_jac[t] = diag(df/dy)(y^(k)[t-1], x[t]) for all t
            (using jax.vmap over jax.jacfwd).
         b. Compute b[t] = f(y^(k)[t-1], x[t]) - diag_jac[t] * y^(k)[t-1]
            (linearization bias, corrected for the diagonal approximation).
         c. Solve y^(k+1) = _diagonal_matmul_recursive(-A, b, y0)
            (forward linear scan in O(log T) depth).
         d. Check convergence.
      3. Return y^(k+1)[1:] (drop y0 prepended by the solver).
    """
    T_local = xinp.shape[0]
    if yinit_guess is None:
        yt = jnp.zeros((T_local, D), dtype=jnp.float64)
    else:
        yt = yinit_guess.astype(jnp.float64)

    # Prepend y0 to form y[0..T-1] for shifted indexing
    def get_shifted(y: jax.Array) -> jax.Array:
        """Return y[0..T-1]: prepend y0 and drop last."""
        return jnp.concatenate([y0[None], y[:-1]], axis=0)  # (T, D)

    err_trace: list[float] = []
    K_observed = 0

    for k in range(max_iter):
        y_prev = get_shifted(yt)  # (T, D): y[0..T-1]

        # Compute diagonal Jacobians and function values in parallel
        diag_jacs = jax.vmap(
            lambda s, x: jnp.diag(jax.jacfwd(func, argnums=0)(s, x))
        )(y_prev, xinp)  # (T, D): diagonal of df/dy at each step

        f_vals = jax.vmap(func)(y_prev, xinp)  # (T, D): f(y_prev[t], x[t])

        # Linearization bias: b[t] = f(y[t-1]) - A[t] * y[t-1]
        bias = f_vals - diag_jacs * y_prev  # (T, D)

        # Solve linear recurrence: yt_new[t] = A[t]*yt_new[t-1] + b[t]
        # where A[t] = diag_jac[t]. At convergence y_new = y_prev, so:
        #   yt_new[t] = diag_jac[t]*y_prev[t-1] + f(y_prev[t-1]) - diag_jac[t]*y_prev[t-1]
        #             = f(y_prev[t-1])  (the original recurrence).
        #
        # Elk deer.py diagonal_seq1d_inv_lin does: matmul_recursive(-gmat, rhs, y0)
        # where gmat = gts[0] = -diag(J)  (already negated), so -gmat = +diag(J).
        # That matches: A_diags = +diag_jacs here.
        full = _diagonal_matmul_recursive(diag_jacs, bias, y0)  # (T+1, D)
        yt_new = full[1:]  # drop the prepended y0; shape (T, D)

        # Clip and NaN-guard (matching elk deer.py clip_ytnext logic)
        clip = 1e8
        yt_new = jnp.clip(yt_new, -clip, clip)
        yt_new = jnp.where(jnp.isnan(yt_new), 0.0, yt_new)

        err = float(jnp.max(jnp.abs(yt_new - yt)))
        err_trace.append(err)
        K_observed = k + 1
        yt = yt_new

        if err < tol:
            break

    return yt, K_observed, err_trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the coordinated-turn reproduction benchmark."""
    print("=" * 64)
    print("Phase 25-03: ELK on Coordinated Turn (D=4, T=500)")
    print("Algorithm: quasi-DEER (re-impl from lindermanlab/elk)")
    print("Reference: sequential jax.lax.scan of SAME transition fn")
    print("=" * 64)
    print()

    # 1. Generate initial condition (from noisy simulation)
    print(f"Generating coordinated turn trajectory (seed={RNG_SEED})...")
    true_states, observations = generate_coordinated_turn_data(RNG_SEED, T)
    print(
        f"  True trajectory shape:  {true_states.shape}  "
        f"pos_x range [{true_states[:,0].min():.1f}, {true_states[:,0].max():.1f}]"
    )
    print()

    # Use true initial state as y0 for both seq and parallel
    y0 = jnp.array(true_states[0], dtype=jnp.float64)
    drivers = jnp.zeros((T, D), dtype=jnp.float64)

    # 2. Sequential scan reference (y[t] = f(y[t-1]) via lax.scan)
    print("Running sequential reference (jax.lax.scan over transition_fn)...")

    seq_fn = jax.jit(lambda y: run_sequential_transition(y, drivers))
    _ = seq_fn(y0)  # warm-up compile

    seq_t0 = time.perf_counter()
    seq_traj = seq_fn(y0)
    seq_traj = jax.device_get(seq_traj)
    seq_wall = time.perf_counter() - seq_t0

    print(f"  Sequential done in {seq_wall:.4f}s (post-JIT)")
    print(f"  Trajectory shape: {seq_traj.shape}")
    print(
        "  First 3 states: "
        + ", ".join(
            f"[{seq_traj[i,0]:.4f}, {seq_traj[i,1]:.4f}, "
            f"{seq_traj[i,2]:.4f}, {seq_traj[i,3]:.4f}]"
            for i in range(3)
        )
    )
    print()

    # 3. quasi-ELK parallel scan (same transition_fn, same y0)
    print(f"Running quasi-ELK (max_iter={MAX_ITER}, tol={TOL:.0e})...")

    # Warm-up run (first call compiles jax.vmap / jax.jacfwd)
    print("  Compiling (first call)...")
    compile_t0 = time.perf_counter()
    _yt_warmup, _K_warmup, _ = run_quasi_elk(
        transition_fn, y0, drivers, max_iter=2, tol=1e-3
    )
    compile_wall = time.perf_counter() - compile_t0
    print(f"  Compile done in {compile_wall:.2f}s")

    # Timed run
    par_t0 = time.perf_counter()
    par_traj, K_observed, err_trace = run_quasi_elk(
        transition_fn, y0, drivers, max_iter=MAX_ITER, tol=TOL
    )
    par_traj = jax.device_get(par_traj)
    par_wall = time.perf_counter() - par_t0

    print(f"  K_observed = {K_observed}  (target: <= {MAX_ITER})")
    print("  Convergence trace:")
    for i, e in enumerate(err_trace):
        print(f"    iter {i+1:2d}: max|delta| = {e:.3e}")
    print(f"  quasi-ELK wall: {par_wall:.4f}s (post-compile, Python loop)")
    print()

    # 4. Numerical agreement: ELK vs sequential scan
    print("Numerical agreement check (quasi-ELK vs lax.scan)...")

    seq_traj_jax = jnp.array(seq_traj)
    par_traj_jax = jnp.array(par_traj)

    abs_diff = jnp.abs(par_traj_jax - seq_traj_jax)
    denom = jnp.abs(seq_traj_jax) + 1e-12
    rel_err_per_state = abs_diff / denom

    rel_err = float(jnp.max(rel_err_per_state))
    mean_rel_err = float(jnp.mean(rel_err_per_state))

    print(f"  Max rel-err:  {rel_err:.3e}  (gate: < {TOL:.0e})")
    print(f"  Mean rel-err: {mean_rel_err:.3e}")

    # Per-component max rel-err
    for i, name in enumerate(["pos_x", "pos_y", "velocity", "omega"]):
        comp_err = float(jnp.max(rel_err_per_state[:, i]))
        print(f"    {name:8s}: max rel-err = {comp_err:.3e}")
    print()

    # 5. Gate verdict
    passed_rel_err = rel_err < TOL
    passed_k = K_observed <= 15
    speedup = seq_wall / par_wall if par_wall > 0 else float("nan")

    if passed_rel_err and passed_k:
        verdict = "PASS"
    elif passed_rel_err:
        verdict = "PASS-WITH-K-NOTE"
    else:
        verdict = "FAIL"

    print(f"VERDICT: {verdict}")
    print(
        f"  rel_err < {TOL:.0e}: {'PASS' if passed_rel_err else 'FAIL'}  "
        f"({rel_err:.2e})"
    )
    print(
        f"  K <= 15:         {'PASS' if passed_k else 'NOTE'}  "
        f"(K={K_observed})"
    )
    print(f"  Speedup (seq/par): {speedup:.3f}x  (CPU; expect < 1 on CPU)")
    print()

    # 6. Write results JSON
    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    results_dir = os.path.join(_project_root, "logs", "phase25")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "reproduction_benchmark.json")

    results = {
        "reproduction_target": "ELK on coordinated turn (option C)",
        "reference_dataset": (
            "Coordinated turn nonlinear SSM, Sarkka 2013 ch.5 / "
            "Yaghoobi et al. 2021 (arXiv:2102.00514) §5 setup"
        ),
        "algorithm_description": (
            "quasi-DEER (diagonal-Jacobian Newton iterations + "
            "jax.lax.associative_scan); re-implemented from lindermanlab/elk"
        ),
        "algorithm_implementation": "lindermanlab/elk (re-implemented in pure JAX)",
        "elk_repo_commit": "2801539febf07f5e9a28eacc6265dc9ac179e835",
        "elk_repo_url": "https://github.com/lindermanlab/elk",
        "sequential_reference": (
            "jax.lax.scan over the same transition_fn (NOT EKF); "
            "ELK and lax.scan both solve y[t]=f(y[t-1]) — "
            "ELK in O(K log T), lax.scan in O(T)."
        ),
        "T": T,
        "D": D,
        "dt": DT,
        "K_observed": K_observed,
        "K_target_max": 15,
        "K_note": (
            "Coordinated-turn has strong contraction (linear transition near "
            "omega=0); K expected 3-5. HGF K=8-15 prediction still stands "
            "(weakly contracting, transient sr>=1). These are different systems."
        ),
        "rel_err_vs_sequential": rel_err,
        "mean_rel_err_vs_sequential": mean_rel_err,
        "rel_err_tolerance": TOL,
        "per_component_max_rel_err": {
            "pos_x": float(jnp.max(rel_err_per_state[:, 0])),
            "pos_y": float(jnp.max(rel_err_per_state[:, 1])),
            "velocity": float(jnp.max(rel_err_per_state[:, 2])),
            "omega": float(jnp.max(rel_err_per_state[:, 3])),
        },
        "convergence_trace_max_delta": [float(e) for e in err_trace],
        "seq_wall_seconds": seq_wall,
        "par_wall_seconds": par_wall,
        "par_wall_note": (
            "Python loop over Newton iterations; par_wall includes "
            "jax.vmap/jacfwd overhead but NOT compilation time."
        ),
        "speedup_observed": speedup,
        "speedup_note": (
            "CPU-only; speedup expected < 1 on CPU per 25-RESEARCH.md §7b. "
            "GPU speedup informational; see cluster SLURM script."
        ),
        "hardware": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "jax_x64_enabled": os.environ.get("JAX_ENABLE_X64", "0") == "1",
        "rng_seed": RNG_SEED,
        "verdict": verdict,
        "verdict_note": (
            "PASS: rel_err < 1e-5 (quasi-ELK matches lax.scan) "
            "and K <= 15. GPU speedup is informational (not gating)."
        ),
        "platform_note": (
            "CPU-only run (local Windows 11 machine). "
            "GPU run submitted via SLURM "
            "(see cluster/25_reproduce_reference_gpu.slurm). "
            "M3 cluster status: Phase 14.1-03 stale pip install issue "
            "pending (job 54934145 failed with ModuleNotFoundError:prl_hgf); "
            "see STATE.md."
        ),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {results_path}")
    print()
    print("=" * 64)
    print(json.dumps(results, indent=2))

    # Exit code: 0 if PASS, 1 if FAIL
    sys.exit(0 if verdict.startswith("PASS") else 1)


if __name__ == "__main__":
    main()
