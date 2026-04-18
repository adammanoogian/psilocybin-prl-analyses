# Phase 19: VB-Laplace Fit Path for PAT-RL — Research

**Researched:** 2026-04-18
**Domain:** Variational Bayes / Laplace approximation at the MAP, JAX-side
**Confidence:** HIGH on the core architectural decisions; MEDIUM on
optimizer-library choice (two valid options); HIGH on consumer-surface parity.

---

## Summary

Phase 19 is bounded: build `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` as
a new sibling to `hierarchical_patrl.py`, reuse `_build_patrl_log_posterior`
unmodified, run a quasi-Newton MAP, take an autodiff Hessian, draw
pseudo-samples from `N(mode, Σ)`, and emit an `az.InferenceData` shaped
identically to the NUTS path so Plan 18-05's exporters consume it without
change.

Every primitive needed already lives in installed packages (`jaxopt 0.8.5`,
`scipy 1.16.3`, `optax 0.2.5`, `jax 0.4.31`, `arviz 0.22.0`, `numpyro 0.19.0`).
**No new dependencies are required.** The `_build_patrl_log_posterior` function
returns a JAX-traceable scalar `logdensity_fn(params: dict[str, jnp.ndarray])`
where each value is shape `(P,)`, and is fully differentiable / Hessianable
via `jax.grad` / `jax.hessian` after a `jax.flatten_util.ravel_pytree` round
trip. The Hessian is exactly block-diagonal across participants (verified by
inspection of the prior + likelihood structure), so `(P*K)×(P*K)` Hessians
are tractable up to several hundred participants without exploiting block
structure, and trivially scalable beyond that with per-subject Hessians if
ever needed.

**One pre-existing latent bug surfaces and the planner needs to know about
it (it is NOT a Phase 19 fix):** `hierarchical.py::_samples_to_idata` (line
1606) emits dim name `"participant"`, but `export_subject_trajectories`
(`export_trajectories.py:181`) and `export_subject_parameters`
(`export_trajectories.py:345-358`) read the coord name `"participant_id"`.
Test `tests/test_export_trajectories.py:54-78` builds idatas directly with
`participant_id`, hiding the inconsistency; the smoke script
`scripts/12_smoke_patrl_foundation.py:_export` would fail in real use because
it forwards the raw NUTS-produced idata to the exporter. Phase 19's Laplace
output **MUST use `participant_id`** to be export-compatible. This is
architecturally correct (matches what Plan 18-05 froze) and naturally
documents the bug in `_samples_to_idata` for a future Phase-18 hotfix
without taking on that fix here.

**Primary recommendation:** Use `jaxopt.LBFGS` (already installed,
PyTree-native, smaller integration surface) for the MAP; use
`jax.flatten_util.ravel_pytree` to bridge to flat-vector `jax.hessian`; use
the eigh-clip PD regularization fallback (cleaner and more deterministic
than a ridge loop in JAX); draw `K=1000` pseudo-samples from
`N(mode, Σ_native)` with `chain=1, draw=K, participant_id=[ids]`; package as
`az.from_dict` with the same posterior var names as the NUTS path
(`omega_2, log_beta, beta` for 2-level; `+ omega_3, kappa, mu3_0` for
3-level). Live the comparison harness in `validation/vbl06_laplace_vs_nuts.py`
(matches `validation/valid03_cross_platform.py` convention).

---

## 1. Architecture Recommendations Keyed to the 10 Questions

### Q1 — MAP optimizer: jaxopt.LBFGS vs scipy L-BFGS-B vs optax.lbfgs

**Recommendation: `jaxopt.LBFGS`.** All three are available; `jaxopt` wins
on minimum integration surface and PyTree-native ergonomics.

Evidence:
- `jaxopt 0.8.5` installed (`pip metadata`), `LBFGS` exposed via
  `jaxopt.LBFGS` and accepts a PyTree (dict-of-arrays) initial position
  directly — no flattening needed for the optimizer itself
  (verified by smoke test in this research).
- `optax.lbfgs` is the strategically-future-proofed choice (jaxopt is
  effectively in maintenance, archived as of 0.8.5 — see WebSearch result
  citing JAXopt's official position) but its API is lower-level (you write
  the optimization loop yourself with `value_and_grad_from_state`,
  `linesearch`, and a `while_loop`), which is more code than `jaxopt.LBFGS`
  for the same outcome.
- `scipy.optimize.minimize(method="L-BFGS-B")` works but requires
  numpy/jax-array round-tripping at every step; tolerable but ugly.

Concrete implementation in `fit_vb_laplace_patrl.py`:
```python
from jaxopt import LBFGS

def neg_log_posterior(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return -_build_patrl_log_posterior(logp_fn, config, model_name)(params)

solver = LBFGS(fun=neg_log_posterior, maxiter=500, tol=1e-5)
res = solver.run(initial_position)  # initial_position is dict[str, (P,)]
mode_params = res.params  # same PyTree shape as input
```

**Caveat the planner should record in the plan:** if `jaxopt` ever stops
working with a future `jax` upgrade, swapping to `optax.lbfgs` is the
escape hatch — the rest of the architecture (objective, Hessian, ArviZ
packaging) is library-agnostic. Cite `pyproject.toml:13` (`jax>=0.4.26,<0.4.32`)
which already pins jax narrowly enough that jaxopt 0.8.5 will continue to
work for the foreseeable lifetime of this milestone.

### Q2 — Parameter exposure of `_build_patrl_log_posterior`

**Verified at `src/prl_hgf/fitting/hierarchical_patrl.py:609-650`:**
the returned `logdensity_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray`
takes a dict with keys:
- 2-level: `{"omega_2": (P,), "log_beta": (P,)}` (2 params × P)
- 3-level: `{"omega_2": (P,), "log_beta": (P,), "omega_3": (P,), "kappa": (P,), "mu3_0": (P,)}` (5 params × P)

`beta = jnp.exp(log_beta)` is computed inside the closure (line 624) before
calling `logp_fn` — so the optimizer parameter is `log_beta` (real-line
unconstrained), not `beta`. The kappa parameter is sampled in the bounded
interval `[priors.kappa.lower, priors.kappa.upper]` via TruncatedNormal
prior log-prob (line 598-603); since the optimizer works on the real line,
the planner should NOT reparametrize kappa to logit — instead let the
truncated prior contribute `−∞` outside support, with optimizer init clipped
into `(lower+1e-6, upper-1e-6)` (mirrors `hierarchical_patrl.py:799` exact
pattern).

**Flatten/unflatten helper (REQUIRED for the Hessian):**
```python
from jax.flatten_util import ravel_pytree

flat_init, unravel = ravel_pytree(initial_position)  # one-time setup

def neg_logp_flat(flat: jnp.ndarray) -> jnp.ndarray:
    return -logdensity_fn(unravel(flat))

H = jax.hessian(neg_logp_flat)(flat_at_mode)
```

The optimizer can run on the dict directly (jaxopt handles PyTrees); the
Hessian needs the flat representation. Verified working in this research.
**No existing precedent — novel addition** (`ravel_pytree` is JAX-builtin
but not used elsewhere in `prl_hgf`).

### Q3 — Hessian dimensionality and block-diagonal structure

**The Hessian IS exactly block-diagonal across participants — verified by
code inspection.**

Evidence:
- `hierarchical_patrl.py:626-635` — every prior term is
  `jnp.sum(prior_X.log_prob(X))` (IID per-participant, no coupling).
- `hierarchical_patrl.py:444` — `jnp.sum(per_participant)` where
  `per_participant = jax.vmap(_call_single)(...)` (also IID per-participant).
- No cross-participant terms anywhere in the closure.

Dimension table:
| P (subjects) | K (params) | Full H size | Block diag block size |
|--------------|------------|-------------|-----------------------|
| 5            | 2 (2lvl)   | 10 × 10     | 2 × 2                 |
| 5            | 5 (3lvl)   | 25 × 25     | 5 × 5                 |
| 32           | 5 (3lvl)   | 160 × 160   | 5 × 5                 |
| 200          | 5 (3lvl)   | 1000 × 1000 | 5 × 5                 |

`jax.hessian` of the full PyTree produces the dense `(P*K) × (P*K)` matrix
in seconds for P ≤ 32. **Recommendation: use the dense Hessian for Phase 19
implementation simplicity.** Document the block-diagonal property as an
optimization opportunity for Phase 20+ if cohort sizes grow past ~500.

A more efficient pattern (deferred): `jax.vmap(jax.hessian(per_subject_logp))`
returns a `(P, K, K)` block stack directly. The planner should leave a
TODO/comment in the implementation noting this — not implement it.

**Memory at largest realistic scale:** P=200, K=5, dense H = 1000² × 8 bytes
= 8 MB, eigendecomposition O(N³) ≈ 10⁹ flops ≈ <1 s on CPU. No memory or
performance concerns at any cohort size we plan to fit.

### Q4 — PD regularization fallback when Hessian is indefinite or
ill-conditioned

**Recommendation: eigh + clip eigenvalues + reconstruct.** Cleaner JAX
pattern than the ridge loop, deterministic, single-shot.

Pattern:
```python
def regularize_to_pd(H: jnp.ndarray, eps: float = 1e-8) -> tuple[jnp.ndarray, dict]:
    """Force Hessian to PD by clipping eigenvalues.

    Returns the regularized Hessian and a diagnostic dict with min/max
    eigenvalues and the number of clipped negative directions.
    """
    w, V = jnp.linalg.eigh(H)
    n_clipped = int(jnp.sum(w < eps))
    w_clip = jnp.maximum(w, eps)
    H_pd = V @ jnp.diag(w_clip) @ V.T
    diag = {
        "eigval_min_pre": float(jnp.min(w)),
        "eigval_max_pre": float(jnp.max(w)),
        "n_clipped": n_clipped,
        "ridge_added": float(eps - jnp.min(w)) if jnp.min(w) < eps else 0.0,
    }
    return H_pd, diag
```

Why eigh-clip beats the ridge loop:
1. **Deterministic** — single matrix decomposition instead of a Python while
   loop with try/except (ugly inside JAX traces).
2. **Same algorithmic intent** as matlab tapas `tapas_riddersmatrix.m`'s
   final PD enforcement step (which conceptually achieves the same thing
   via successive Cholesky retries) but expressed in linear algebra terms.
3. **Diagnoses the failure** — the returned dict tells the user how many
   negative directions were present and how much regularization was needed,
   logged as a warning per VBL-02.
4. **Matches established best practice** — see WebSearch result on `laplax`
   (arxiv 2507.17013) which uses rank-truncation / eigenvalue regularization
   for the same problem.

Verified working in this research (smoke test on a 2×2 indefinite matrix).

The downstream covariance is then `Σ_unconstrained = jnp.linalg.inv(H_pd)`.

**Sanity gate before returning:** assert `jnp.linalg.cholesky(H_pd)` succeeds
(no exception); assert `jnp.all(jnp.linalg.eigvalsh(Σ_unconstrained) > 0)`.
Per VBL-02, the function "does NOT silently return garbage" — failing this
gate means raising an explicit `RuntimeError` with the diagnostic dict in
the message, NOT returning a finite-but-wrong result.

### Q5 — ArviZ InferenceData shape parity with the NUTS path

**The Laplace output target shape matches `_samples_to_idata`'s output
EXCEPT for the dim name (see critical caveat below).**

Verified at `hierarchical.py:1555-1624` (`_samples_to_idata` source):
- `posterior_dict["beta"] = np.exp(positions["log_beta"])` — beta as deterministic
- All sampled vars share `dims = ["participant"]`
- `coords = {"participant": participant_ids}`
- `sample_stats` includes `diverging`, `acceptance_rate`,
  `num_integration_steps`, `num_trajectory_expansions`, `energy`
  (NUTS-specific; Laplace will skip these).

**Critical pre-existing inconsistency the planner needs to record:**

| Producer (NUTS, `_samples_to_idata`) | Consumer (`export_trajectories.py`) |
|--------------------------------------|--------------------------------------|
| dim/coord name = `"participant"`     | reads `"participant_id"` (`:181, 345`)|
| line 1606, 1608                      | line 181: `coords.get("participant_id", None)` |

The test fixture (`test_export_trajectories.py:54, 73`) builds idatas with
`participant_id` directly, hiding the bug. The Phase 18 smoke
(`scripts/12_smoke_patrl_foundation.py:_export`) would fail in real cluster
use because it forwards the raw NUTS idata. **Phase 19 sidesteps the bug
by emitting `participant_id` natively** (matches what 18-05 froze and what
the consumer reads), and explicitly documents in the new module's
docstring that this differs from the NUTS path's dim name. The fix-up of
`_samples_to_idata` belongs to a separate Phase 18 hotfix (track in
STATE.md), NOT in Phase 19.

**Recommended Laplace InferenceData factory (DO NOT call `_samples_to_idata`):**
```python
import arviz as az

def _laplace_to_idata(
    mode_params_native: dict[str, np.ndarray],   # {var: (P,)}
    cov_native: np.ndarray,                       # (P*K, P*K), block diag
    var_order: list[str],                         # e.g. ['omega_2','log_beta'] (or 5 vars)
    participant_ids: list[str],
    n_pseudo_draws: int = 1000,
    rng_key: jnp.ndarray | None = None,
) -> az.InferenceData:
    """Draw K pseudo-samples from N(mode, Σ); package as ArviZ InferenceData.

    Output shape parity with NUTS path:
    - chain=1, draw=K, participant_id=[ids]
    - var names: omega_2, log_beta, beta (deterministic), [omega_3, kappa, mu3_0]
    """
    P = len(participant_ids)
    K = len(var_order)

    # Block-diagonal sampling: per-subject K×K cov → independent multivariate normals.
    # Layout: flat = concatenate([params for var in var_order]) per subject? Or per var?
    # Use ravel_pytree's natural layout (matches Hessian).
    ...
    posterior_dict = {var: ...  # shape (1, n_pseudo_draws, P)}
    posterior_dict['beta'] = np.exp(posterior_dict['log_beta'])  # deterministic
    return az.from_dict(
        posterior=posterior_dict,
        coords={
            'chain': [0],
            'draw': list(range(n_pseudo_draws)),
            'participant_id': participant_ids,
        },
        dims={var: ['participant_id'] for var in posterior_dict},
    )
```

**HDI verification:** `az.hdi(post[params], hdi_prob=0.94)` works on
single-chain (chain=1) posteriors; returns coord values `["lower", "higher"]`
(verified by smoke test in this research, on ArviZ 0.22.0). Consumes
identically to NUTS-path output for `export_subject_parameters`.

**`sample_stats` for Laplace:** include a synthetic minimal block so the
exporter can introspect `idata.sample_stats` without crashing. Recommended
keys: `{"converged": bool, "n_iterations": int, "logp_at_mode": float,
"hessian_min_eigval": float, "ridge_added": float}`. None of these are
read by 18-05's exporters, so they're informational only — but matching
the structure (a `sample_stats` group exists) keeps the InferenceData
shape consistent with what downstream code expects.

### Q6 — Laplace-vs-NUTS comparison harness location

**Recommendation: `validation/vbl06_laplace_vs_nuts.py`** (new module, mirrors
`validation/valid03_cross_platform.py` convention).

Evidence:
- `validation/valid03_cross_platform.py` is the established precedent: a
  per-validation script with a `compare_results(path_a, path_b, rtol)`
  function (line 56, 70, 90, 109 in `tests/test_valid03.py` — six tests
  covering identical / within-tolerance / exceeds-tolerance / near-zero
  cases).
- Mirroring this convention gives the planner a known testing pattern to
  follow without inventing new structure.

Module sketch:
```python
"""VBL-06: Laplace-vs-NUTS posterior comparison harness.

Loads two InferenceData (one Laplace, one NUTS) for the same sim_df
and produces a per-subject per-parameter diff table.

Tolerance gates (from .planning/quick/004 VB_LAPLACE_FEASIBILITY.md sec 6):
- |Δ posterior_mean(omega_2)| < 0.3
- |Δ log_sd(omega_2)| < 0.5
"""

def compare_posteriors(
    idata_laplace: az.InferenceData,
    idata_nuts: az.InferenceData,
    params: list[str] = ('omega_2', 'beta'),
) -> pd.DataFrame:
    """Returns long-format DataFrame:
    columns = [participant_id, parameter, mean_laplace, mean_nuts,
               sd_laplace, sd_nuts, abs_diff_mean, abs_diff_log_sd, within_gate]
    """
```

Tests live in `tests/test_vbl06_laplace_vs_nuts.py` (mirrors
`tests/test_valid03.py`).

**Do NOT** put the comparison helper in `src/prl_hgf/fitting/compare_fits.py` —
that would create a third location for cross-fit utilities and not match
the validation/ convention. The comparison is a validation activity, not a
fitting activity.

### Q7 — Smoke + recovery data: reuse `_simulate_cohort` from
`scripts/12_smoke_patrl_foundation.py`?

**Recommendation: extract `_simulate_cohort` to a new module
`src/prl_hgf/env/pat_rl_simulator.py`** — it's reusable infrastructure that
multiple consumers will want, and module-private helpers in `scripts/` are
not importable cleanly.

Evidence for extraction:
- `_simulate_cohort` signature at `scripts/12_smoke_patrl_foundation.py:176-181`:
  `(n_participants, level, master_seed, config) ->
  (sim_df, true_params, trials_by_participant)`
  — this is a clean, reusable signature with no script-specific state.
- It already uses `_run_hgf_forward` (line 123) and
  `generate_session_patrl` (line 254) — both already public/importable.
- The Phase 19 Laplace smoke needs the IDENTICAL data to what Plan 18-06's
  smoke uses to make `(Laplace ↔ NUTS)` comparison meaningful on shared
  subjects (VBL-06's success criterion #5).

Constraint check: extracting `_simulate_cohort` from a script to a module
**does NOT touch any of the parallel-stack-invariant files** listed in the
phase brief (`hierarchical.py, hierarchical_patrl.py, task_config.py,
simulator.py, hgf_2level.py, hgf_3level.py, response.py,
configs/prl_analysis.yaml`). It only adds a new module and turns the
script function into a thin wrapper around the module function. Safe to do.

**Alternative (acceptable but inferior):** import the underscore-prefixed
function directly from the script path. This works but encourages
test/code coupling to script internals — discouraged by Python style.

### Q8 — Tolerance gates

Verbatim from `.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md`:
- `|Δ mean(omega_2)| < 0.3` (Section 6, "Downgrade triggers")
- `|Δ log_sd(omega_2)| < 0.5`

Use these directly as assertion thresholds in
`tests/test_vbl06_laplace_vs_nuts.py` and as the gate in
`compare_posteriors(...)["within_gate"]`. **No re-derivation.**

The memo also implicitly establishes a third gate by reference (Section 6,
"Downgrade to Option B"): "Laplace covariance underestimates true
posterior width by > 2× on omega_2" → VBL-06 should also report
`sd_laplace / sd_nuts` per subject and warn if any subject has ratio < 0.5
or > 2.0 (silent precision inflation/deflation). This is a softer warning,
not a hard gate.

### Q9 — Matlab tapas parity vs JAX autodiff Hessian

**Recommendation: state in the docstring that JAX autodiff Hessian is
mathematically exact in finite precision (modulo float64 rounding) and is
strictly better than Ridders' numerical method at the mode of a smooth,
differentiable objective.** This is established practice — no literature
citation needed beyond noting the equivalence.

Specifically:
- `_build_patrl_log_posterior` is composed entirely of JAX-traceable
  primitives (Normal log-prob, TruncatedNormal log-prob, sigmoid, scan,
  vmap, exp). Every primitive has a registered `jvp` and `transpose` rule.
- `jax.hessian = jax.jacrev(jax.grad)` produces the exact second derivative
  in float64 / float32 arithmetic; the only error is roundoff, bounded by
  `O(eps_machine * cond(H))`.
- Ridders' method (matlab `tapas_riddersmatrix.m`) is `O(h^6)` after
  Richardson extrapolation, but with `h ~ eps_machine^(1/3)`, the absolute
  error is `O(eps_machine^2)` — comparable to autodiff in well-conditioned
  cases, **strictly worse** in ill-conditioned ones because numerical
  cancellation degrades the accuracy as the Hessian condition number grows.

The literature reference for this comparison (cited in the WebSearch result
for "jax.hessian Laplace approximation"): the `laplax` library
(arxiv 2507.17013, "laplax — Laplace Approximations with JAX") explicitly
relies on `jax.hessian` and discusses the same Cholesky / regularization
considerations. **Use this as the secondary reference in the module
docstring** alongside the matlab tapas references for context.

### Q10 — Parameter ordering for flatten/unflatten and InferenceData
variable names

**Recommendation: define a single canonical parameter order at module top
and use it everywhere.**

Concrete:
```python
_PARAM_ORDER_2LEVEL: tuple[str, ...] = ("omega_2", "log_beta")
_PARAM_ORDER_3LEVEL: tuple[str, ...] = (
    "omega_2", "log_beta", "omega_3", "kappa", "mu3_0",
)
```

`jax.flatten_util.ravel_pytree` flattens dict keys in **insertion order**
on Python 3.7+ (CPython, PyPy, and JAX-internal representations all
guarantee this). The planner should construct `initial_position` as a
plain `dict` populated in canonical order, NOT rely on alphabetical
sorting (which would put `beta < kappa < log_beta < mu3_0 < omega_2 <
omega_3` — confusing and fragile).

The InferenceData posterior dict uses the same names PLUS `beta` as the
deterministic transform of `log_beta` (matches NUTS path
`_samples_to_idata:1602`). Order in the posterior dict does not affect
ArviZ behavior — `from_dict` builds an `xarray.Dataset` keyed by name —
but consistency aids debugging.

Match what the consumer reads (`export_trajectories.py:288-289`):
- 2-level expects `["omega_2", "beta"]`
- 3-level expects `["omega_2", "omega_3", "kappa", "beta", "mu3_0"]`

Both lists use `beta` (not `log_beta`) — the deterministic transform must
be present in the Laplace posterior dict.

---

## 2. Consumer Surface Audit — Exact NUTS InferenceData Shape

The following is the contract that Phase 19's Laplace output MUST satisfy
to pass the exporters from Plan 18-05 unchanged.

| Group | Variable | Shape | Coord/Dim | Source |
|-------|----------|-------|-----------|--------|
| `posterior` | `omega_2` | `(chain, draw, P)` | `dims=['chain','draw','participant_id']` | `_samples_to_idata:1606` (writes "participant"; consumer reads "participant_id" — see bug note) |
| `posterior` | `log_beta` | `(chain, draw, P)` | same | `_samples_to_idata:1599-1604` |
| `posterior` | `beta` (deterministic) | `(chain, draw, P)` | same | `_samples_to_idata:1602` `np.exp(positions["log_beta"])` |
| `posterior` | `omega_3` (3-level only) | `(chain, draw, P)` | same | `_samples_to_idata:1599-1604` |
| `posterior` | `kappa` (3-level only) | `(chain, draw, P)` | same | same |
| `posterior` | `mu3_0` (3-level only) | `(chain, draw, P)` | same | same |
| `posterior.coords` | `participant_id` (Laplace) / `participant` (NUTS) | `(P,)` strings | — | see bug note above |
| `posterior.coords` | `participant_group` | `(P,)` strings | dim `participant` | `_samples_to_idata:1620` (extra metadata; not strictly required by exporter but harmless to add) |
| `posterior.coords` | `participant_session` | `(P,)` strings | dim `participant` | `_samples_to_idata:1621` (same) |
| `sample_stats` | `diverging`, `acceptance_rate`, `num_integration_steps`, `num_trajectory_expansions`, `energy` | `(chain, draw)` | NUTS-only | `_extract_nuts_stats:899-905` (Laplace skips; substitute synthetic minimal stats) |

**Exporter assumptions (`export_trajectories.py`, verified by reading source):**

- `export_subject_trajectories:189-195` uses
  `post[var].sel(participant_id=pid).mean()` — needs `participant_id` coord
  and computes mean across `chain` and `draw` (both dims). For Laplace
  with `chain=1, draw=K`, the mean is well-defined: `(1*K) → 1` value.
- `export_subject_parameters:349, 353` calls `az.hdi(post[params], hdi_prob=0.94)`
  → returns Dataset shape `(participant_id, hdi)` with `hdi` coord values
  `["lower", "higher"]`. **Verified working on single-chain idata in this
  research** (ArviZ 0.22.0).
- `export_subject_parameters:353` also uses `mean(dim=["chain", "draw"])`
  — works for Laplace single-chain.

**Implication:** Laplace path emits `participant_id` (matching consumer);
puts `chain=1, draw=K`; populates the same var names; adds `beta` as
deterministic exp(log_beta). Pass.

---

## 3. Dependency Status

| Library | Version installed | Phase 19 use | Action |
|---------|-------------------|--------------|--------|
| `jaxopt` | 0.8.5 | `LBFGS` for MAP | **Add to `pyproject.toml`** — currently not in `dependencies`, only available because numpyro/arviz pulled it in transitively. Add explicitly so the build is reproducible. |
| `scipy` | 1.16.3 | (alternative to jaxopt) — not used if jaxopt chosen | Already in `pyproject.toml:19` |
| `optax` | 0.2.5 | (future-proof alternative) — not used in Phase 19 | Already pulled by numpyro |
| `jax` | 0.4.31 | `jax.hessian`, `jax.flatten_util.ravel_pytree`, `jax.linalg.eigh`, `jax.linalg.cholesky` | Already in `pyproject.toml:13` |
| `arviz` | 0.22.0 | `az.from_dict`, `az.hdi` | Already in `pyproject.toml:22` |
| `numpyro` | 0.19.0 | `numpyro.distributions` (already used by `_build_patrl_log_posterior`) | Already pulled in transitively |

**Add to `[project.dependencies]` in `pyproject.toml:11`** (insert between
`blackjax>=1.2.4` and `numpy>=2.0.0,<3.0`):
```toml
"jaxopt>=0.8.5,<0.9",
```

**No GPU dependencies.** Laplace fits are CPU-friendly and fast (<60 s for
5 agents per VBL-04). No SLURM template needed in Phase 19.

**Verification commands (run during plan-validation):**
```bash
python -c "from jaxopt import LBFGS; print('LBFGS ok')"
python -c "import jax; from jax.flatten_util import ravel_pytree; print('jax', jax.__version__)"
python -c "import arviz as az; print('arviz', az.__version__)"
```

---

## 4. Planner Handoff — Concrete Plans

Phase 19 should decompose into **5 plans**, with parallelization waves as
indicated. All plans assume Phase 18 is code-complete (which it is —
ROADMAP shows Phase 18 at 6/6).

### Wave 1 (parallel — independent sub-modules, can run concurrently)

**Plan 19-01: Simulator extraction + dependency add** (small)
- Extract `_simulate_cohort` from `scripts/12_smoke_patrl_foundation.py` to
  new `src/prl_hgf/env/pat_rl_simulator.py::simulate_patrl_cohort` (with
  type hints, NumPy-style docstring, public API).
- Replace the script's local `_simulate_cohort` with a thin wrapper
  importing the new module function (preserves the existing smoke).
- Also extract `_run_hgf_forward` (line 123) since the Laplace MAP-evaluation
  needs the same forward-pass-at-true-params utility.
- Add `jaxopt>=0.8.5,<0.9` to `pyproject.toml:11`.
- 4–6 unit tests on the extracted module (ensure the smoke still works).

**Plan 19-02: Laplace InferenceData factory module** (small/medium)
- Create `src/prl_hgf/fitting/_laplace_idata.py::laplace_to_idata`
  (module-private helper, name underscore-prefixed).
- Implements the pseudo-sample drawing from `N(mode, Σ_native)` and the
  `az.from_dict` packaging.
- Inputs: mode params dict + flat covariance + var order + participant_ids
  + n_pseudo_draws + rng_key.
- Output: `az.InferenceData` with the consumer-surface contract from
  Section 2 above.
- Keep this private — `fit_vb_laplace_patrl` from Plan 19-03 is the single
  public entry point.
- 5–7 unit tests covering: shape parity with a fake NUTS idata, single-chain
  HDI works, `participant_id` coord exists, `beta = exp(log_beta)`
  deterministic correct, var names match consumer expectations,
  `sample_stats` group present.

### Wave 2 (depends on Wave 1 — both modules above)

**Plan 19-03: Core Laplace fit pipeline** (medium/large — the heart of the phase)
- Create `src/prl_hgf/fitting/fit_vb_laplace_patrl.py`.
- Public API: `fit_vb_laplace_patrl(sim_df, model_name, response_model,
  config, n_pseudo_draws=1000, max_iter=500, tol=1e-5, random_seed=0) ->
  az.InferenceData`.
- Steps inside (each verifiable):
  1. Validate inputs (model_name, response_model — same NotImplementedError
     pattern as `hierarchical_patrl.py:728`).
  2. Build arrays via `_build_arrays_single_patrl` (reuse from
     `hierarchical_patrl.py` — public-via-import).
  3. Build batched logp via `build_logp_fn_batched_patrl` (reuse).
  4. Build log posterior via `_build_patrl_log_posterior` (reuse, **NOT
     copied or modified — IMPORTED**).
  5. Build initial position dict at prior means (mirror
     `hierarchical_patrl.py:783-804`).
  6. Run `jaxopt.LBFGS` MAP.
  7. Compute Hessian via `jax.hessian` of the negated log posterior
     (flat-vector via `ravel_pytree`).
  8. PD-regularize via eigh-clip; raise `RuntimeError` with diagnostics if
     not recoverable.
  9. Compute `Σ_unconstrained = inv(H)`.
  10. Apply delta-rule transform for `log_beta → beta` (the only constrained
      → unconstrained reparam in scope for Phase 19; see note below on
      kappa).
  11. Call `_laplace_to_idata` from Plan 19-02 to package output.
- 10+ tests including: 2-level smoke (3 agents, 50 trials, <30 s),
  3-level smoke (3 agents, 50 trials, <60 s), parameter recovery sanity
  (omega_2 within 0.5 of truth for 4/5), PD regularization triggers
  correctly on a constructed indefinite case, NotImplementedError for
  Models B/C/D, kappa init clipping handles edge cases.

**Note on kappa reparam:** the matlab tapas convention reparametrizes
`kappa` to logit. `_build_patrl_log_posterior` uses TruncatedNormal in the
*native* space (line 598-603), so the optimizer would land at the mode in
native space. **Recommendation: DO NOT reparametrize kappa in Phase 19.**
The truncated prior contributes `−∞` outside support; init clipped into
`(lower+1e-6, upper-1e-6)` keeps the optimizer interior. Document this as
an acceptable simplification — the Laplace covariance for kappa will be
computed in native space, which is what the consumer expects (the NUTS
posterior is also in native space). Adding logit reparam is a future-Phase
refinement and only matters if the MAP lands close to a kappa boundary
(unlikely with reasonable priors). Cite line 598-603, 799 as the existing
truncated-prior pattern this matches.

### Wave 3 (depends on Wave 2 — needs Plan 19-03 output to compare against)

**Plan 19-04: Laplace-vs-NUTS comparison harness** (small/medium)
- Create `validation/vbl06_laplace_vs_nuts.py::compare_posteriors`.
- Mirrors `validation/valid03_cross_platform.py` structure (module + CLI
  with `run` and `compare` subcommands).
- The `run` subcommand: takes a sim_df parquet, runs both
  `fit_batch_hierarchical_patrl` and `fit_vb_laplace_patrl`, saves both
  idatas as netCDF.
- The `compare` subcommand: loads both idatas, calls `compare_posteriors`,
  prints the diff table, asserts the tolerance gates.
- 6+ tests in `tests/test_vbl06_laplace_vs_nuts.py` mirroring
  `tests/test_valid03.py`'s six test cases (identical / within-tolerance /
  exceeds-tolerance / near-zero / missing-coord / wrong-model-name).

**Plan 19-05: Smoke + recovery integration test** (small)
- New script `scripts/13_smoke_vb_laplace_patrl.py` mirroring
  `scripts/12_smoke_patrl_foundation.py` structure: simulate → fit (Laplace)
  → export → sanity check → optionally compare to NUTS if a NUTS idata is
  passed in.
- 5-agent CPU smoke <60 s total per VBL-04 success criterion #4.
- Recovery sanity assertion: posterior-mean omega_2 within 0.5 of truth
  for ≥4/5 agents.
- Integration test in `tests/test_smoke_vb_laplace_patrl.py` that runs the
  whole pipeline end-to-end on a tiny problem (3 agents, 24 trials, <10 s)
  and asserts CSVs land in the output dir with the frozen schema.

### Parallelization waves summary

```
Wave 1 (parallel):
  ├─ 19-01: simulator extraction + dep add
  └─ 19-02: laplace InferenceData factory

Wave 2 (depends on 1):
  └─ 19-03: core Laplace fit pipeline (the substantial plan)

Wave 3 (parallel, depends on 2):
  ├─ 19-04: Laplace-vs-NUTS comparison harness
  └─ 19-05: smoke + recovery integration
```

### Files created (NONE of these touch the parallel-stack invariant set)

```
src/prl_hgf/fitting/fit_vb_laplace_patrl.py     [new — Plan 19-03]
src/prl_hgf/fitting/_laplace_idata.py           [new — Plan 19-02]
src/prl_hgf/env/pat_rl_simulator.py             [new — Plan 19-01]
validation/vbl06_laplace_vs_nuts.py             [new — Plan 19-04]
scripts/13_smoke_vb_laplace_patrl.py            [new — Plan 19-05]
tests/test_fit_vb_laplace_patrl.py              [new — Plan 19-03]
tests/test_laplace_idata.py                     [new — Plan 19-02]
tests/test_pat_rl_simulator.py                  [new — Plan 19-01]
tests/test_vbl06_laplace_vs_nuts.py             [new — Plan 19-04]
tests/test_smoke_vb_laplace_patrl.py            [new — Plan 19-05]
```

### Files modified (must verify `git diff` is empty for invariants)

```
pyproject.toml                                   [Plan 19-01: add jaxopt dep]
scripts/12_smoke_patrl_foundation.py             [Plan 19-01: replace inline _simulate_cohort with import]
```

NONE of: `hierarchical.py`, `hierarchical_patrl.py`, `task_config.py`,
`env/simulator.py`, `models/hgf_2level.py`, `models/hgf_3level.py`,
`models/hgf_2level_patrl.py`, `models/hgf_3level_patrl.py`,
`models/response.py`, `models/response_patrl.py`,
`configs/prl_analysis.yaml`, `configs/pat_rl.yaml`,
`analysis/export_trajectories.py`. **Phase 19 success criterion #6 holds
trivially under this file plan.**

---

## 5. Open Questions the Planner Should NOT Assume Away

### OQ1 — The participant/participant_id dim-name mismatch in `_samples_to_idata`

**What we know:** `_samples_to_idata` writes `"participant"`,
`export_subject_trajectories` reads `"participant_id"`. Test fixtures hide
this. Real cluster runs of Phase 18 smoke have not happened yet.

**Phase 19 stance:** the Laplace path emits `"participant_id"` natively
(matches the consumer; documented choice in module docstring). Do NOT fix
`_samples_to_idata` in Phase 19 — that's a Phase 18 hotfix. **The planner
MUST add a note to `.planning/STATE.md` flagging this for follow-up.**

If the cluster smoke hits this bug before Phase 19 lands, the user may
choose to fix it in `_samples_to_idata` (rename to `participant_id`), in
which case the Laplace path's choice is the same — no change required to
Phase 19's plans.

### OQ2 — The kappa reparametrization choice in 3-level Laplace

**What we know:** matlab tapas convention is logit-transform of bounded
parameters (kappa). Our existing `_build_patrl_log_posterior` uses native-
space TruncatedNormal. The Laplace MAP will land in native space.

**What's unclear:** if the cluster smoke shows kappa MAPs landing close to
the upper or lower truncation bound, the native-space Hessian will be
poorly conditioned, leading to large pseudo-sample variance estimates that
don't reflect the truncation. The matlab tapas logit reparam fixes this.

**Recommendation for Phase 19:** ship the simpler native-space
implementation. Add a TODO/comment in `fit_vb_laplace_patrl.py` noting
that if VBL-06's per-subject diff for kappa shows `|Δ posterior_mean| > 0.3`
on the cluster smoke, a follow-up can add logit reparam (small additive
change to the optimizer + Jacobian back-transform; no new architecture).

### OQ3 — `n_pseudo_draws=1000` parameter choice

**What we know:** the Laplace posterior is a Gaussian; in principle one
could compute moments analytically (mean = mode, sd = sqrt(diag(Σ))) and
skip pseudo-sampling. But `az.hdi` and downstream consumers expect a
sample-based posterior with `chain` and `draw` dims.

**What's unclear:** the optimal `K = n_pseudo_draws`. Too few (e.g., 100)
gives noisy HDI estimates; too many (e.g., 10000) wastes memory. Default
of 1000 chosen by analogy to typical NUTS draw count.

**Recommendation:** make it a configurable kwarg with default 1000.
Document in the docstring that it does NOT affect the Laplace approximation
quality (the Gaussian is exact); it only affects MC noise in the
sample-based summaries (HDI, percentiles). The mean and Σ are exact.

### OQ4 — Should the Laplace path support trial-varying parameters
(Models B/C/D)?

**Phase 19 brief stance:** Phase 19 focuses on Model A only (matches Phase
18 scope). Models B/C/D are explicitly deferred per ROADMAP Phase 18
"Option A Minimum Viable scope" note (line 235).

**Recommendation:** NotImplementedError for `response_model != "model_a"`
(mirror `hierarchical_patrl.py:728-732` exact pattern). The phase brief
does not list Models B/C/D as in-scope; do not add them.

### OQ5 — Multi-modal posterior detection

**What we know:** Laplace silently collapses multimodal posteriors to one
mode — this is the well-known Laplace failure mode. The feasibility memo
(Section 5, "Key honest caveat") names this.

**What's unclear:** should `fit_vb_laplace_patrl` detect multimodality (e.g.,
random restarts from different initial positions) and warn? The matlab
tapas `tapas_fitModel.m` does not — it trusts the user to know the
posterior shape.

**Recommendation:** Phase 19 ships a single optimization run from
prior-mean init. Add a `n_restarts: int = 1` parameter to the public API
(default 1, no extra cost) with a docstring note that values >1 will
re-run the MAP from perturbed initial positions and pick the highest-logp
mode. Implement only if the planner has spare scope; otherwise leave
`n_restarts=1` only and add the parameter in a follow-up. **Not a Phase 19
must-have.**

### OQ6 — Diagnostic logging interface

**What we know:** the Laplace path computes useful diagnostics (logp at
mode, n_iterations, eigval min/max, ridge added). The NUTS path returns
these via `sample_stats`. ArviZ has no standardized location for non-MCMC
diagnostics.

**What's unclear:** where to surface these for downstream users.
Recommendation:
- Stash them in `idata.sample_stats` under custom keys
  (`converged, n_iterations, logp_at_mode, hessian_min_eigval, ridge_added`).
- Also log to `logging.INFO` per subject during the fit.
- Do NOT define a new dataclass return type — the brief specifies
  `az.InferenceData` and that's what the exporters expect.

### OQ7 — Phase 19 closure decision memo

The ROADMAP entry (Phase 19 "Scope notes") states: "A decision memo at
phase close reports Laplace-vs-NUTS agreement on real cluster data and
recommends whether to keep both paths or consolidate on one." This is
**out of Phase 19's coding scope** but in its planning scope — the planner
should include a final small plan (or note in 19-04) committing to:

> Plan 19-05 (or a sixth plan): write
> `.planning/phases/19-vb-laplace-fit-path-patrl/19-CLOSURE-MEMO.md` after
> the cluster smoke runs, summarizing observed Δ-mean and Δ-log-sd per
> parameter, recovery quality, and a Yes/No recommendation on
> consolidation. Defer the actual writing until cluster numbers are in.

This is a placeholder planning task — content cannot be researched without
real cluster results.

---

## Sources

### Primary (HIGH confidence) — repo source code, directly read

- `src/prl_hgf/fitting/hierarchical_patrl.py:1-868` — full module, especially:
  - `_build_patrl_log_posterior:552-652` (the function Phase 19 reuses)
  - `_build_arrays_single_patrl:454-544` (data prep, also reused)
  - `build_logp_fn_batched_patrl:304-446` (the batched logp factory)
  - `fit_batch_hierarchical_patrl:660-867` (the NUTS orchestrator to mimic)
- `src/prl_hgf/fitting/hierarchical.py:869-905` (`_extract_nuts_stats`),
  `:908-925` (`_run_blackjax_nuts` signature), `:1555-1624`
  (`_samples_to_idata` — output shape spec)
- `src/prl_hgf/analysis/export_trajectories.py:1-379` — the consumer
  surface, especially:
  - `:181, 345-358` (reads coord name `participant_id`)
  - `:189-195` (`_post_mean` uses `mean(("chain","draw"))`)
  - `:288-289` (`_PARAMS_2LEVEL`, `_PARAMS_3LEVEL` lists)
  - `:332-340` (`hdi` coord handling — `lower`/`higher`)
- `src/prl_hgf/env/pat_rl_config.py:348-545` (priors structure:
  PriorGaussian, PriorTruncated, FittingPriorConfig)
- `src/prl_hgf/models/response_patrl.py:1-141` (Model A logp; numerics)
- `src/prl_hgf/models/hgf_2level_patrl.py:54-94`,
  `src/prl_hgf/models/hgf_3level_patrl.py:80-142` (network builders;
  signature reuse patterns)
- `scripts/12_smoke_patrl_foundation.py:123-316` (`_simulate_cohort`,
  `_run_hgf_forward` — Phase 19 simulator extractions)
- `tests/test_export_trajectories.py:54-78` (idata fixture proves consumer
  expects `participant_id` coord)
- `tests/test_hierarchical_patrl.py:231-319` (5-participant smoke
  precedent for Phase 19's smoke)
- `tests/test_valid03.py:56-125` (precedent for Plan 19-04's comparison
  harness tests)
- `validation/valid03_cross_platform.py:1-40` (precedent for Plan 19-04's
  module structure)
- `pyproject.toml:11-31` (verified jaxopt NOT in explicit dependencies but
  available transitively)
- `.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md`
  — entire memo, especially Section 6 (tolerance gates) and Section 5
  (multimodal caveat)
- `.planning/ROADMAP.md:256-282` (Phase 19 brief; success criteria; scope notes)
- `.planning/phases/18-pat-rl-task-adaptation/18-RESEARCH.md` (precedent
  for RESEARCH.md format)

### Primary (HIGH confidence) — installed package introspection

- `python -c "import jaxopt; ...; print(version('jaxopt'))"` →
  `jaxopt 0.8.5` installed, `LBFGS`, `LBFGSB`, `ScipyMinimize`, `BFGS`
  exposed
- `python -c "import scipy; print(scipy.__version__)"` → `1.16.3`
- `python -c "import optax; print(optax.__version__)"` → `0.2.5` with
  `optax.lbfgs` available
- `python -c "import jax; print(jax.__version__)"` → `0.4.31`
- `python -c "import arviz; print(arviz.__version__)"` → `0.22.0`
- `python -c "import numpyro; print(numpyro.__version__)"` → `0.19.0`
- Smoke test (this research): `jaxopt.LBFGS` works on dict-of-arrays
  PyTree initial position; `jax.hessian(neg_logp_flat)(flat)` produces a
  symmetric `(N, N)` matrix; `jax.flatten_util.ravel_pytree` round-trips
  dict ↔ flat correctly; `az.hdi` on `chain=1, draw=K` data returns coord
  values `["lower", "higher"]`.

### Secondary (MEDIUM confidence) — WebSearch verified with project pages

- `jaxopt 0.8.5 maintenance status` → JAXopt is no longer maintained;
  features migrated to optax. Confirmed via official jaxopt PyPI page and
  GitHub issue google-deepmind/optax#1201 ("Jaxopt-equivalent L-BFGS").
  Implication: `jaxopt.LBFGS` works for Phase 19's lifetime but a future
  jax upgrade may force migration to `optax.lbfgs`. Documented in Q1.
- `optax.lbfgs API` → confirmed `optax.lbfgs()` returns a gradient
  transformation that can be wrapped with linesearch via
  `optax.scale_by_zoom_linesearch` and run inside a manual `jax.lax.while_loop`.
  More code than `jaxopt.LBFGS`; not chosen for Phase 19.
- `Laplace approximation in JAX` → laplax (arxiv 2507.17013, 2025) confirms
  the architectural pattern Phase 19 will follow (`jax.hessian` + Cholesky
  + eigenvalue clipping for ill-conditioned Hessians). Recommended cite in
  module docstring.

### Tertiary (LOW confidence — flagged for validation)

- None used — every architectural claim is backed by either repo source
  or installed-package introspection (HIGH) or by an official package page
  cross-checked against an independent source (MEDIUM).

---

## Metadata

**Confidence breakdown:**
- Standard stack (libraries, versions): HIGH — verified by direct
  introspection
- Architecture patterns (Laplace InferenceData, eigh-clip PD, ravel_pytree
  flatten, jaxopt.LBFGS optimizer): HIGH — patterns verified end-to-end in
  research smoke tests
- Consumer surface (export_trajectories shape contract): HIGH — read source
  line by line
- Optimizer-library choice (jaxopt vs optax vs scipy): MEDIUM — multiple
  valid choices; jaxopt recommended on minimal-integration grounds but
  optax is the long-term forward path (jaxopt is in maintenance mode)
- Pre-existing `participant`/`participant_id` dim-name bug: HIGH — verified
  by reading both producer and consumer source. Not a Phase 19 fix; record
  in STATE.md.

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days; stable scientific Python ecosystem,
no fast-moving APIs in scope)
