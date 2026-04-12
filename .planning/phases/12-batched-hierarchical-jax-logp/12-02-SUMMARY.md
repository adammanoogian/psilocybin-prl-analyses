# Phase 12 Plan 02: Batched JAX Logp Op Factory Summary

**One-liner:** vmap'd cohort-level JAX logp Op with Layer 2 NaN clamping, trial_mask plumbing, and two-Op split for PyMC gradients -- bit-exact with ops.py at P=1.

## What Was Done

### Task 1: Create hierarchical.py (9e9f4a6)

Created `src/prl_hgf/fitting/hierarchical.py` (580 lines) containing:

- `build_logp_ops_batched(input_data_arr, observed_arr, choices_arr, model_name, trial_mask)` -- public factory accepting `(P, n_trials, 3)` shaped arrays
- `_single_logp_3level` / `_single_logp_2level` -- per-participant logp closures reusing pyhgf's `Network.scan_fn` (no HGF math reimplemented)
- `_clamped_scan` -- Layer 2 NaN-clamping wrapper using `jnp.where` + `jax.tree_util.tree_map` (no Python `if` on traced values)
- `_build_scan_inputs` / `_compute_logp` -- shared helpers factored out for reuse between 2-level and 3-level variants
- `_BatchedLogpOp` / `_BatchedGradOp` -- two-Op split with `@jax_funcify.register` for numpyro dispatch
- Constants: `_MU_2_BOUND = 14.0`, `_MODEL_NAMES = ("hgf_2level", "hgf_3level")`

## Design Decisions

### Option A chosen: data as runtime arguments (not closure-over-data)

The `single_participant_logp` functions accept data as arguments (not closed over). This enables `jax.vmap` with `in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)` across all args. The batched function `_jax_logp_batched` closes over the pre-converted JAX arrays and passes them alongside the parameter arrays to the vmapped function. This approach:

- Keeps the vmap signature clean and explicit
- Avoids re-creating closures when data changes
- Lets the XLA compiler see the full data flow

### Level-2 mean attribute key: `attrs[i]["mean"]` (not `"expected_mean"`)

The pyhgf attribute pytree uses `"mean"` on continuous-state nodes 1, 3, 5 for the level-2 posterior mean (a scalar `()` shape in the carry). This is distinct from `"expected_mean"` on binary input nodes 0, 2, 4 (which is the sigmoid-transformed reward probability used for `mu1` in the softmax logp).

### Separate named functions for 2-level and 3-level (not conditional redefinition)

Originally used `if/else` branches defining `_single_participant_logp` with different signatures. Mypy flagged this as incompatible types. Refactored to define `_single_logp_3level` and `_single_logp_2level` as separate functions, then assign to `_single_participant_logp` at runtime.

## Deviations from ops.py Math

**None beyond the plan-specified additions:**

1. Layer 2 clamping wrapper inside `lax.scan` (reverts belief state on instability)
2. `stability_mask` multiplication on per-trial logp (unstable trials contribute 0)
3. `trial_mask` multiplication on per-trial logp (padded trials contribute 0)

The softmax-stickiness formula, `expected_mean` readout from nodes 0/2/4, parameter injection via shallow-copy pattern, `-jnp.inf` NaN sentinel, and scan-input tuple structure are all identical to ops.py.

## Smoke-Test Logp Values

Parameters: omega_2=-3.0, omega_3=-6.0, kappa=1.0, beta=3.0, zeta=0.5. Data: 50 trials, seed 42.

| P | model_name | logp |
|---|-----------|------|
| 1 | hgf_3level | -59.960130 |
| 2 | hgf_3level | -119.920261 |
| 3 | hgf_3level | -179.880391 |

Logp scales linearly with P (identical participants), confirming correct vmap reduction.

### Bit-Exact Comparison at P=1

| Model | ops.py logp | hierarchical.py logp | Difference |
|-------|------------|---------------------|-----------|
| hgf_3level | -59.96013031946589 | -59.96013031946588 | 7.11e-15 |
| hgf_2level | -59.980806838532175 | -59.980806838532175 | 0.00e+00 |

Both within float64 machine epsilon. VALID-01 (Plan 12-04) will formalize this as a regression test.

## Verification Results

- Import check: `from prl_hgf.fitting.hierarchical import build_logp_ops_batched` succeeds
- Factory build: P=1, P=2, P=3 all succeed for both model variants
- Forward pass: P=2 returns finite scalar for both hgf_3level and hgf_2level
- trial_mask: zeroing last 10 trials changes logp from -80.046 to -64.159
- Layer 2 clamping: `jnp.where` pattern present on lines 174, 254
- JAX funcify: `@jax_funcify.register(_BatchedLogpOp)` on line 566
- ruff: all checks passed
- mypy: no issues found (3 type: ignore comments for JAX/PyTensor API patterns)
- Existing tests: 6/7 pass; 1 pre-existing failure in test_omega2_positive_returns_neginf (unrelated)

## Decisions Made

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Data as runtime args (Option A) | Clean vmap signature; XLA sees full data flow; no closure recreation on data change | 12-02 |
| Separate named functions for 2-level/3-level logp | Avoids mypy error from conditional redefinition with different signatures | 12-02 |
| Level-2 mean key is `attrs[i]["mean"]` | Confirmed via runtime inspection of pyhgf attribute pytree structure | 12-02 |
| type: ignore on 3 lines for JAX/PyTensor API patterns | jax.vmap changes callable arity; PyTensor Op.__call__ returns Variable or list; both correct at runtime | 12-02 |

## Key Files

### Created

- `src/prl_hgf/fitting/hierarchical.py` (580 lines)

### Not Modified

- `src/prl_hgf/fitting/ops.py` (reference implementation, unchanged)
- `src/prl_hgf/fitting/__init__.py` (12-03 will add the re-export)

## Next Phase Readiness

Plan 12-03 (hierarchical PyMC model wrapper) can proceed immediately. It will:
- Import `build_logp_ops_batched` from this module
- Wrap the Op in a PyMC model with group-level priors
- Add the re-export to `src/prl_hgf/fitting/__init__.py`

Plan 12-04 (VALID-01) will formalize the bit-exact comparison as a regression test.

## Metrics

- Duration: ~15 minutes
- Tasks: 1/1
- Commits: 1
