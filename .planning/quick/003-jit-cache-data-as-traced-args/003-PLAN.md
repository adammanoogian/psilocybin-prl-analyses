---
phase: quick-003
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/prl_hgf/fitting/hierarchical.py
autonomous: true

must_haves:
  truths:
    - "Power-sweep iterations with same-shape data reuse the persistent JIT compilation cache (no ~1600s recompilation per call)"
    - "Warmup phase still works correctly (closure-based logdensity is fine for single-shot warmup)"
    - "Posterior samples are numerically identical to the current implementation given the same RNG seed"
    - "Both 2-level and 3-level model variants work through the new sample loop"
    - "Both vmap (single-GPU) and pmap (multi-GPU) chain strategies still work"
  artifacts:
    - path: "src/prl_hgf/fitting/hierarchical.py"
      provides: "_build_sample_loop factory + _run_vmap_chains/_run_pmap_chains refactored to take data as traced args"
      contains: "_build_sample_loop"
  key_links:
    - from: "fit_batch_hierarchical BlackJAX path"
      to: "_build_sample_loop"
      via: "passes data arrays as explicit arguments to JIT'd sampling function"
      pattern: "_build_sample_loop"
    - from: "_build_sample_loop inner logdensity"
      to: "batched_logp_fn"
      via: "constructs logdensity_fn inside JIT boundary with data as traced args"
      pattern: "batched_logp_fn.*input_data.*observed.*choices.*trial_mask"
---

<objective>
Restructure BlackJAX sampling so data arrays flow as traced JIT arguments
instead of being captured as constants in a closure, enabling persistent
XLA compilation cache hits across power-sweep iterations with same data shapes.

Purpose: Eliminate ~1600s recompilation per power-sweep call when data values
differ but shapes are identical. The XLA persistent cache keys on HLO hash --
closure-captured data becomes HLO constants (different values = different hash),
while traced arguments become shape-dependent placeholders (same shapes = same hash).

Output: Modified `hierarchical.py` where the BlackJAX sampling phase passes
data as traced args through a JIT boundary, while warmup remains closure-based
(runs once, no cache benefit needed).
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/prl_hgf/fitting/hierarchical.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add _build_sample_loop factory and refactor _run_vmap_chains / _run_pmap_chains to accept data as traced args</name>
  <files>src/prl_hgf/fitting/hierarchical.py</files>
  <action>
The core problem: `_build_log_posterior` captures data arrays (input_data, observed,
choices, trial_mask) in a closure. When JAX JIT-compiles, these become HLO constants.
Different data values = different HLO hash = persistent cache miss = full recompilation.

**Step A: Create `_build_sample_loop` factory function (insert after `_run_pmap_chains`, before `_samples_to_idata` around line 1098)**

This factory builds a JIT'd sampling function where data flows as traced arguments:

```python
def _build_sample_loop(
    batched_logp_fn,
    model_name: str,
    n_chains: int,
    n_draws: int,
    use_pmap: bool,
):
```

- Takes `batched_logp_fn` (the pure JAX logp from `build_logp_fn_batched` -- this
  closes over `base_attrs`, `scan_fn`, `n_trials` which are static per model shape,
  so capturing it is fine).
- Takes `model_name` as a static config value (controls prior structure).
- Takes `n_chains`, `n_draws`, `use_pmap` as static config values.
- Returns a function `sample_loop(warmup_state, warmup_params, sample_key, input_data, observed, choices, trial_mask)`.

Inside `_build_sample_loop`, define the returned function. Inside THAT function:

1. **Reconstruct logdensity_fn from traced data args** -- this is the key change.
   Inline the prior+likelihood combination (same math as `_build_log_posterior`'s
   `logdensity_fn`, but data arrays come from function args, not closure):

   ```python
   is_3level = model_name == "hgf_3level"
   # Prior distributions (constructed once, captured by closure -- these are
   # parameterless JAX objects, not data)
   prior_omega_2 = dist.TruncatedNormal(loc=-3.0, scale=2.0, high=0.0)
   prior_log_beta = dist.Normal(0.0, 1.5)
   prior_zeta = dist.Normal(0.0, 2.0)
   if is_3level:
       prior_omega_3 = dist.TruncatedNormal(loc=-6.0, scale=2.0, high=0.0)
       prior_kappa = dist.TruncatedNormal(loc=1.0, scale=0.5, low=0.01, high=2.0)
   ```

   Then inside the returned function, define a local `logdensity_fn(params)` that:
   - Computes prior_lp the same way as `_build_log_posterior`
   - Calls `batched_logp_fn(omega_2, omega_3, kappa, beta, zeta, input_data, observed, choices, trial_mask)` using the TRACED data args from the enclosing function scope
   - Returns `prior_lp + likelihood_lp`

2. **Build the NUTS kernel inside the function:**
   ```python
   nuts = blackjax.nuts(logdensity_fn, **warmup_params)
   ```
   This is fine -- `blackjax.nuts` returns a NamedTuple with pure JAX `.step` method.

3. **Run the sampling loop** (same logic as current `_run_vmap_chains` / `_run_pmap_chains`):
   - If `use_pmap`: replicate state, pmap across chains, lax.scan per chain
   - Else (vmap): replicate state, vmap the step across chains, lax.scan draws

4. Return `(positions, infos)` as JAX arrays (NOT converted to numpy -- stay in JAX land inside JIT).

**The returned function must be JIT'd.** Apply `@jax.jit` to the inner function.
Since `warmup_params` is a dict of scalars (step_size float, inverse_mass_matrix array),
these are fine as traced args. The `warmup_state` is a BlackJAX NUTSState pytree -- also fine.

**Important:** The pmap path inside a JIT boundary is tricky. If `use_pmap` is True,
the factory should NOT wrap the inner function with `@jax.jit` and should instead
let pmap handle compilation. Use `@jax.jit` only for the vmap path. Make `use_pmap`
control which inner function variant gets returned:

- `use_pmap=False`: return a `@jax.jit`-decorated vmap sampling function
- `use_pmap=True`: return a function that uses `jax.pmap` internally (pmap handles its own JIT)

For the vmap path (the common case on single GPU), the JIT boundary is:
```
sample_loop(warmup_state, warmup_params, sample_key,
            input_data, observed, choices, trial_mask)
            ^^^^^^^^^^ ^^^^^^^^ ^^^^^^^ ^^^^^^^^^^
            traced     traced   traced  traced  <-- data is VALUE-independent in HLO
```

**Step B: Modify `_run_blackjax_nuts` to use the new sample loop**

Change `_run_blackjax_nuts` signature to also accept:
- `batched_logp_fn` (the raw logp function from `build_logp_fn_batched`)
- `input_data`, `observed`, `choices`, `trial_mask` (the data arrays)
- `model_name` (for prior construction inside JIT)

The function body changes:
1. **Warmup stays as-is** (lines 912-922): uses `logdensity_fn` from `_build_log_posterior`.
   This is fine -- warmup runs once, no cache benefit needed.
2. **Remove line 925** (`nuts = blackjax.nuts(logdensity_fn, **warmup_params)`).
3. **Replace the sampling phase** (lines 928-948):
   ```python
   n_devices = jax.device_count()
   use_pmap = n_devices >= n_chains

   sample_loop = _build_sample_loop(
       batched_logp_fn, model_name, n_chains, n_draws, use_pmap,
   )

   all_states, all_infos = sample_loop(
       warmup_state, warmup_params, sample_key,
       input_data, observed, choices, trial_mask,
   )
   ```
4. **Post-process positions/stats to numpy** (moved OUT of _run_vmap_chains/_run_pmap_chains
   and into _run_blackjax_nuts, since the sample_loop returns JAX arrays):
   - For vmap: positions are `(n_draws, n_chains, P)` -> transpose to `(n_chains, n_draws, P)` -> np.asarray
   - For pmap: positions are `(n_chains, n_draws, P)` -> np.asarray directly
   - Extract `is_divergent` and `acceptance_rate` from infos, same transpose logic

**Step C: Update `fit_batch_hierarchical` BlackJAX path call site (lines 1710-1760)**

Pass the additional arguments to `_run_blackjax_nuts`:
```python
positions, sample_stats, n_chains_actual = _run_blackjax_nuts(
    logdensity_fn,       # still needed for warmup
    initial_position,
    rng_key,
    n_tune=n_tune,
    n_draws=n_draws,
    n_chains=n_chains,
    target_accept=target_accept,
    batched_logp_fn=logp_fn,          # NEW
    input_data=jax_input_data,        # NEW
    observed=jax_observed,            # NEW
    choices=jax_choices,              # NEW
    trial_mask=jax_trial_mask,        # NEW
    model_name=model_name,            # NEW
)
```

**Step D: Keep `_run_vmap_chains` and `_run_pmap_chains` as-is (DO NOT delete)**

These functions are still valid helper functions and their logic is reused conceptually
inside `_build_sample_loop`. However, `_run_blackjax_nuts` will no longer call them
directly -- the new sample_loop replaces that call path. Keep them around for now as
they serve as documentation and could be used by other callers. If ruff flags them
as unused, add `# noqa: F811` or keep them in `__all__` -- but more likely they
won't be flagged since they're module-level functions, not local variables.

**What NOT to change:**
- `_build_log_posterior` -- keep as-is, still used for warmup
- `build_logp_fn_batched` -- untouched
- `_clamped_scan`, `_compute_logp` -- untouched
- NumPyro path in `fit_batch_hierarchical` -- untouched
- `_samples_to_idata` -- untouched
- `_build_arrays_single` -- untouched
- Any test files -- this is a quick task, cluster smoke test validates

**Docstring updates:**
- `_run_blackjax_nuts`: update docstring to document new parameters and explain
  that warmup uses closure-based logdensity while sampling uses traced-arg sample loop
- `_build_sample_loop`: full NumPy-style docstring explaining the factory pattern,
  why data must be traced (not closure-captured), and the JIT cache implications
  </action>
  <verify>
1. `python -c "from prl_hgf.fitting.hierarchical import fit_batch_hierarchical, _build_sample_loop; print('import OK')"` succeeds
2. `python -m ruff check src/prl_hgf/fitting/hierarchical.py` passes (no lint errors)
3. `python -m ruff format --check src/prl_hgf/fitting/hierarchical.py` passes
4. `python -m pytest tests/test_blackjax_smoke.py -x -v` passes (existing smoke tests exercise the BlackJAX path end-to-end with tiny data; they will exercise the new sample_loop since it replaces the sampling phase)
5. If smoke tests are not available locally (GPU-only), verify at minimum that the function constructs without error: `python -c "from prl_hgf.fitting.hierarchical import _build_sample_loop; print('factory importable')"` 
  </verify>
  <done>
- `_build_sample_loop` factory exists and returns a JIT'd sampling function that takes data as traced arguments
- `_run_blackjax_nuts` uses `_build_sample_loop` for the sampling phase (warmup unchanged)
- `fit_batch_hierarchical` passes data arrays and model config to `_run_blackjax_nuts`
- All existing imports, tests, and lint pass
- Data arrays are no longer baked as HLO constants in the sampling compilation -- they flow as traced args, enabling persistent cache reuse across power-sweep iterations with same shapes
  </done>
</task>

</tasks>

<verification>
- Import chain works: `from prl_hgf.fitting.hierarchical import fit_batch_hierarchical, _build_sample_loop`
- Ruff lint + format pass on hierarchical.py
- Existing smoke tests pass (if GPU available) or at minimum import-level verification
- Manual code review confirms: inside the JIT'd sample_loop, data arrays are function parameters (traced) not closure captures (constant)
</verification>

<success_criteria>
1. `_build_sample_loop` exists as a factory that returns a JIT'd function accepting data as traced args
2. The sampling phase of BlackJAX NUTS passes data as explicit arguments through the JIT boundary
3. Warmup phase is unchanged (closure-based logdensity_fn, runs once)
4. Both vmap and pmap chain strategies work through the new sample_loop
5. All lint passes, all existing tests pass
6. Ready for cluster GPU validation (same-shape data across iterations should produce cache hits instead of recompilation)
</success_criteria>

<output>
After completion, create `.planning/quick/003-jit-cache-data-as-traced-args/003-SUMMARY.md`
</output>
