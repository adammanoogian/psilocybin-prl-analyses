"""JAX-native single-session HGF simulator using ``lax.scan``.

Provides ``simulate_session_jax``, the compiled XLA kernel that replaces the
NumPy for-loop in ``simulate_agent``.  The simulator runs one 420-trial
session via ``jax.lax.scan`` using pyhgf's ``scan_fn`` for HGF belief updates,
``jax.random.categorical`` for choice sampling, and ``jax.random.bernoulli``
for reward sampling.

Architecture: factory pattern
------------------------------
Building a pyhgf ``Network()`` involves Python-side side effects (graph
construction, JAX JIT compilation) that cannot be traced by JAX.  The
factory function :func:`_build_session_scanner` builds the network ONCE and
returns the captured ``scan_fn`` and ``base_attrs``.  The pure-JAX function
:func:`_run_session` accepts these as arguments and can be safely vmapped.

This mirrors the pattern used by :func:`~prl_hgf.fitting.hierarchical.build_logp_ops_batched`.

For batch/vmap usage (Plan 02)::

    scan_fn, base_attrs = _build_session_scanner()
    simulate_cohort = jax.vmap(
        _run_session,
        in_axes=(None, None, 0, 0, 0, 0, 0, None, 0),
    )

Layer 2 NaN clamping
---------------------
Each trial step applies tapas-style stability check on the pyhgf belief
update.  If any leaf in the updated attributes pytree is non-finite, or if
any level-2 mean exceeds the magnitude bound ``|mu_2| < 14.0``, the belief
state is reverted to the previous trial's values.  This matches the
implementation in :mod:`prl_hgf.fitting.hierarchical`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from prl_hgf.models.hgf_3level import build_3level_network

__all__ = [
    "simulate_session_jax",
    "simulate_cohort_jax",
    "_build_session_scanner",
    "_run_session",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tapas magnitude bound on level-2 means (``tapas_ehgf_binary.m``).
#: Must match :data:`prl_hgf.fitting.hierarchical._MU_2_BOUND`.
_MU_2_BOUND: float = 14.0


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def _build_session_scanner() -> tuple[object, dict]:
    """Build pyhgf ``scan_fn`` and ``base_attrs`` once.

    Constructs a 3-level binary HGF network, seeds it with a single dummy
    trial to initialise the ``scan_fn`` JIT, and returns the captured
    ``scan_fn`` callable and ``base_attrs`` attribute pytree.

    .. warning::
        This function performs Python-side side effects (pyhgf graph
        construction) and **must not** be called inside ``jax.vmap`` or
        ``jax.jit``.  Call it once outside those contexts, then pass the
        returned values into :func:`_run_session`.

    Returns
    -------
    scan_fn : callable
        The pyhgf per-trial update function, suitable for ``lax.scan``.
    base_attrs : dict
        Initial attribute pytree with default parameters.  Shallow-copy and
        inject real parameters before passing to :func:`_run_session`.
    """
    net = build_3level_network()
    net.input_data(
        input_data=np.zeros((1, 3)),
        observed=np.ones((1, 3), dtype=int),
    )
    return net.scan_fn, net.attributes


# ---------------------------------------------------------------------------
# Pure-JAX simulation kernel
# ---------------------------------------------------------------------------


def _run_session(
    scan_fn: object,
    base_attrs: dict,
    omega_2: jnp.ndarray,
    omega_3: jnp.ndarray,
    kappa: jnp.ndarray,
    beta: jnp.ndarray,
    zeta: jnp.ndarray,
    cue_probs_arr: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX single-session simulator kernel.

    Runs one complete session of ``n_trials`` trials via ``lax.scan``.  All
    random sampling uses the functional JAX PRNG pattern: the PRNG key is
    threaded through the carry and split at each step.

    This function is designed to be vmapped across participants for
    cohort-level simulation (Plan 02).  Because it is pure JAX (no
    Python-side side effects), it is compatible with ``jax.vmap``,
    ``jax.jit``, and ``jax.grad``.

    Parameters
    ----------
    scan_fn : callable
        Captured pyhgf per-trial update function from
        :func:`_build_session_scanner`.
    base_attrs : dict
        Captured initial attribute pytree from :func:`_build_session_scanner`.
    omega_2 : jnp.ndarray
        Scalar tonic volatility for level-1 belief nodes (1, 3, 5).
    omega_3 : jnp.ndarray
        Scalar meta-volatility for the shared level-2 node (6).
    kappa : jnp.ndarray
        Scalar volatility coupling strength between node 6 and nodes 1, 3, 5.
    beta : jnp.ndarray
        Scalar inverse temperature for the softmax choice model.
    zeta : jnp.ndarray
        Scalar stickiness / choice perseveration weight.
    cue_probs_arr : jnp.ndarray, shape (n_trials, 3)
        Ground-truth reward probability for each cue on each trial.
        Produced from ``jnp.array([t.cue_probs for t in trials])``.
    rng_key : jnp.ndarray
        JAX PRNG key, shape ``(2,)``.  Use ``jax.random.PRNGKey(seed)``
        or a pre-split subkey from ``jax.random.split``.

    Returns
    -------
    choices : jnp.ndarray, shape (n_trials,)
        Chosen cue index (0, 1, or 2) per trial, dtype int32.
    rewards : jnp.ndarray, shape (n_trials,)
        Binary reward outcome (0 or 1) per trial, dtype int32.
    diverged : jnp.ndarray
        Scalar boolean.  ``True`` if any trial's belief state was clamped.

    Notes
    -----
    For batch/vmap usage, call :func:`_build_session_scanner` once and vmap
    this function directly — do not vmap :func:`simulate_session_jax` (which
    would rebuild the network per call).

    The ``-1`` sentinel for ``prev_choice`` on trial 0 is safe: the
    stickiness comparison ``(prev_choice == jnp.arange(3))`` evaluates to all-
    False for ``-1``, giving zero stickiness on the first trial.
    """
    # ------------------------------------------------------------------
    # Parameter injection (shallow-copy pattern from hierarchical.py)
    # ------------------------------------------------------------------
    attrs = dict(base_attrs)

    # omega_2 into level-1 belief nodes (1, 3, 5)
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["tonic_volatility"] = omega_2
        attrs[idx] = node

    # omega_3 and kappa children-side into shared volatility node 6
    node6 = dict(attrs[6])
    node6["tonic_volatility"] = omega_3
    node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
    attrs[6] = node6

    # kappa parents-side into nodes 1, 3, 5
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["volatility_coupling_parents"] = jnp.array([kappa])
        attrs[idx] = node

    # ------------------------------------------------------------------
    # Per-trial step function for lax.scan
    # ------------------------------------------------------------------

    def _sim_step(
        carry: tuple,
        x: jnp.ndarray,
    ) -> tuple[tuple, tuple]:
        """Single-trial simulation step.

        Parameters
        ----------
        carry : tuple
            ``(attrs, rng_key, prev_choice)`` — HGF belief state, current
            PRNG key, and the previous trial's chosen cue index.
        x : jnp.ndarray, shape (3,)
            Ground-truth reward probabilities for this trial.

        Returns
        -------
        new_carry : tuple
            ``(safe_attrs, next_key, choice)`` for the next step.
        outputs : tuple
            ``(choice, reward, is_stable)`` emitted per trial.
        """
        step_attrs, step_rng_key, prev_choice = carry
        cue_probs = x  # shape (3,)

        # Step 1: Read prior beliefs from carry BEFORE HGF update
        # INPUT_NODES = (0, 2, 4) — binary-state nodes, expected_mean in [0,1]
        p_reward = jnp.array([
            step_attrs[0]["expected_mean"],
            step_attrs[2]["expected_mean"],
            step_attrs[4]["expected_mean"],
        ])

        # Step 2: Split PRNG key — step_key for this trial, next_key for carry
        step_key, next_key = jax.random.split(step_rng_key)
        choice_key, reward_key = jax.random.split(step_key)

        # Step 3: Softmax-stickiness choice sampling
        # prev_choice == -1 (sentinel) → stick is all-False → zero stickiness
        stick = (prev_choice == jnp.arange(3)).astype(jnp.float32)
        logits = beta * p_reward + zeta * stick
        choice = jax.random.categorical(choice_key, logits)

        # Step 4: Bernoulli reward sampling for the chosen cue
        reward = jax.random.bernoulli(reward_key, cue_probs[choice])
        reward = jnp.int32(reward)

        # Step 5: Build per-trial scan input for pyhgf scan_fn
        # Shape contract (from _build_scan_inputs in hierarchical.py):
        #   values elements: shape (1,)   — matches n_trials[:, 0:1] sliced to (1,)
        #   observed elements: scalar     — matches n_trials[:, 0] sliced to scalar
        #   time_step: scalar float32
        reward_f = jnp.float32(reward)
        values_t = (
            jnp.where(choice == 0, reward_f, jnp.float32(0.0)).reshape(1),
            jnp.where(choice == 1, reward_f, jnp.float32(0.0)).reshape(1),
            jnp.where(choice == 2, reward_f, jnp.float32(0.0)).reshape(1),
        )
        observed_t = (
            jnp.where(choice == 0, jnp.float32(1.0), jnp.float32(0.0)),
            jnp.where(choice == 1, jnp.float32(1.0), jnp.float32(0.0)),
            jnp.where(choice == 2, jnp.float32(1.0), jnp.float32(0.0)),
        )
        time_step_t = jnp.float32(1.0)
        scan_input_t = (values_t, observed_t, time_step_t, None)

        # Step 6: HGF belief update via pyhgf scan_fn
        new_attrs, _ = scan_fn(step_attrs, scan_input_t)

        # Step 7: Layer 2 clamping (tapas-style, identical to hierarchical.py)
        leaves = jax.tree_util.tree_leaves(new_attrs)
        all_finite = jnp.all(
            jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        )
        mu_2_vals = jnp.array([
            new_attrs[1]["mean"],
            new_attrs[3]["mean"],
            new_attrs[5]["mean"],
        ])
        mu_2_ok = jnp.all(jnp.abs(mu_2_vals) < _MU_2_BOUND)
        is_stable = all_finite & mu_2_ok

        # Revert belief state on instability (no Python if on traced value)
        safe_attrs = jax.tree_util.tree_map(
            lambda n, o: jnp.where(is_stable, n, o),
            new_attrs,
            step_attrs,
        )

        return (safe_attrs, next_key, choice), (choice, reward, is_stable)

    # ------------------------------------------------------------------
    # Run lax.scan over all trials
    # ------------------------------------------------------------------
    # Initial carry: attrs with injected params, starting PRNG key,
    # and prev_choice = -1 sentinel (gives zero stickiness on trial 0)
    init_carry = (attrs, rng_key, jnp.int32(-1))
    _, (choices, rewards, stability_flags) = lax.scan(
        _sim_step, init_carry, cue_probs_arr
    )

    # Diverged if any trial was clamped (stability_flag was False)
    diverged = jnp.any(~stability_flags)

    return choices, rewards, diverged


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------


def simulate_session_jax(
    omega_2: jnp.ndarray,
    omega_3: jnp.ndarray,
    kappa: jnp.ndarray,
    beta: jnp.ndarray,
    zeta: jnp.ndarray,
    cue_probs_arr: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate one HGF session via ``lax.scan`` (public convenience wrapper).

    Builds the pyhgf network once via :func:`_build_session_scanner`, then
    delegates to :func:`_run_session` for the compiled XLA forward pass.

    .. note::
        For batch/vmap usage (e.g. cohort-level simulation), call
        :func:`_build_session_scanner` once outside the vmap and vmap
        :func:`_run_session` directly.  Vmapping this function would
        rebuild the network per call, which is inefficient.

    Parameters
    ----------
    omega_2 : jnp.ndarray
        Scalar tonic volatility for level-1 belief nodes.
    omega_3 : jnp.ndarray
        Scalar meta-volatility for the shared level-2 node.
    kappa : jnp.ndarray
        Scalar volatility coupling strength.
    beta : jnp.ndarray
        Scalar inverse temperature.
    zeta : jnp.ndarray
        Scalar stickiness / choice perseveration weight.
    cue_probs_arr : jnp.ndarray, shape (n_trials, 3)
        Ground-truth reward probability per cue per trial.
    rng_key : jnp.ndarray
        JAX PRNG key, shape ``(2,)``.

    Returns
    -------
    choices : jnp.ndarray, shape (n_trials,)
        Chosen cue index per trial, dtype int32.
    rewards : jnp.ndarray, shape (n_trials,)
        Binary reward outcome per trial, dtype int32.
    diverged : jnp.ndarray
        Scalar bool — ``True`` if any trial's belief state was clamped.
    """
    scan_fn, base_attrs = _build_session_scanner()
    return _run_session(
        scan_fn,
        base_attrs,
        omega_2,
        omega_3,
        kappa,
        beta,
        zeta,
        cue_probs_arr,
        rng_key,
    )


# ---------------------------------------------------------------------------
# Cohort-level vmap wrapper
# ---------------------------------------------------------------------------


def simulate_cohort_jax(
    params_batch: dict[str, jnp.ndarray],
    cue_probs_arr: jnp.ndarray,
    rng_keys_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate a full cohort of participants via ``jax.vmap``.

    Builds the pyhgf network once via :func:`_build_session_scanner`, then
    vmaps :func:`_run_session` across participants.  All participants share
    the same trial sequence (``cue_probs_arr``) but have distinct parameter
    vectors and PRNG keys.

    .. note::
        For cohorts with **per-participant** trial sequences (different
        ``env_seed`` per participant), use the :func:`simulate_batch`
        function in :mod:`prl_hgf.simulation.batch`, which stacks
        per-participant ``cue_probs_arr`` arrays and vmaps over them.

    Parameters
    ----------
    params_batch : dict[str, jnp.ndarray]
        Dictionary with keys ``"omega_2"``, ``"omega_3"``, ``"kappa"``,
        ``"beta"``, ``"zeta"``, each of shape ``(P,)`` — one scalar value
        per participant.
    cue_probs_arr : jnp.ndarray, shape (n_trials, 3)
        Shared ground-truth reward probability for each cue on each trial.
        Same trial sequence used for all ``P`` participants.
    rng_keys_batch : jnp.ndarray, shape (P, 2)
        One distinct JAX PRNG key per participant.  Generate with
        ``jax.random.split(master_key, P)``.

    Returns
    -------
    choices : jnp.ndarray, shape (P, n_trials)
        Chosen cue index per participant per trial, dtype int32.
    rewards : jnp.ndarray, shape (P, n_trials)
        Binary reward outcome per participant per trial, dtype int32.
    diverged : jnp.ndarray, shape (P,)
        Per-participant boolean — ``True`` if any trial's belief state was
        clamped for that participant.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from prl_hgf.simulation.jax_session import simulate_cohort_jax
    >>> params = {
    ...     "omega_2": jnp.array([-3.0, -4.0]),
    ...     "omega_3": jnp.array([-6.0, -5.0]),
    ...     "kappa": jnp.array([1.0, 0.5]),
    ...     "beta": jnp.array([2.0, 3.0]),
    ...     "zeta": jnp.array([0.5, 0.0]),
    ... }
    >>> # cue_probs_arr shape (n_trials, 3) — from generate_session
    >>> keys = jax.random.split(jax.random.PRNGKey(0), 2)
    >>> choices, rewards, diverged = simulate_cohort_jax(params, cue_probs_arr, keys)
    >>> choices.shape
    (2, n_trials)
    """
    scan_fn, base_attrs = _build_session_scanner()
    _vmapped = jax.vmap(
        lambda o2, o3, k, b, z, rk: _run_session(
            scan_fn, base_attrs, o2, o3, k, b, z, cue_probs_arr, rk
        ),
        in_axes=(0, 0, 0, 0, 0, 0),
    )
    return _vmapped(
        params_batch["omega_2"],
        params_batch["omega_3"],
        params_batch["kappa"],
        params_batch["beta"],
        params_batch["zeta"],
        rng_keys_batch,
    )
