# Phase 13: JAX-Native Cohort Simulation - Research

**Researched:** 2026-04-12
**Domain:** JAX lax.scan + vmap simulation pipeline, PRNG key threading, pyhgf scan_fn reuse
**Confidence:** HIGH

## Summary

Phase 13 wraps the existing pyhgf `scan_fn` (proven working in Phase 12 for logp) inside a simulation scan loop that also threads PRNG keys through `lax.scan` carries, samples choices via `jax.random.categorical`, and samples rewards via `jax.random.bernoulli`. The output replaces `simulate_agent`'s NumPy loop as the internal engine of `simulate_batch`, preserving the exact same DataFrame schema (including `diverged`). The Phase 12 `_clamped_scan` pattern ports directly: only the per-trial step body changes from "compute logp contribution" to "sample choice, sample reward, build HGF input".

The key structural difference from the Phase 12 logp path is that simulation requires PRNG keys in the carry: `lax.scan` carry becomes `(attrs, rng_key, prev_choice)` instead of just `attrs`. Each step splits the key into a step-key and continuation-key, uses the step-key for `jax.random.categorical` (choice) and `jax.random.bernoulli` (reward), and advances the carry with the continuation-key. Cohort-level parallelism follows the same `jax.vmap` pattern as Phase 12: split the master key into per-participant keys outside the vmap, vmap over participant params and keys with `in_axes=0`.

The simulation differs from logp in a crucial input-structure way: the logp path takes real observed `(input_data, observed)` arrays as scan inputs and computes the likelihood of observed choices. The simulation path uses pre-computed per-trial cue probability arrays as scan inputs (shape `(n_trials, 3)`, float), generates choices and rewards stochastically, and writes those back as scan outputs. The HGF update after each trial requires constructing `(values, observed_cols, time_steps, None)` inside the step function from the sampled `choice` and `reward`.

**Primary recommendation:** Build `simulate_session_jax` as a thin scan wrapper around the existing `_clamped_scan` infrastructure from `hierarchical.py`. The carry is `(attrs, rng_key, prev_choice)`. Scan outputs are `(choice, reward, is_stable_flag)` per trial. Vmap with pre-split keys for the cohort path. `simulate_batch` gets a drop-in wrapper that converts the DataFrame build loop to use the JAX path.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| jax | >=0.4.26,<0.4.32 | lax.scan, vmap, jax.random | Project constraint (pyproject.toml) |
| jax.numpy | same | Array ops inside scan | Same as jax |
| pyhgf | >=0.2.8,<0.3 | scan_fn for HGF belief updates | Project constraint; no HGF math reimplementation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2.0.0,<3.0 | Post-vmap array conversion to Python/DataFrame | Outside JAX boundary |
| pandas | >=2.0 | DataFrame assembly from vmap output | Same as current simulate_batch |
| scipy.stats | >=1.10 | ks_2samp for VALID-04 statistical equivalence | Validation test only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| jax.random.categorical | Custom softmax + numpy.random.choice | Don't hand-roll: categorical accepts logits directly, works inside lax.scan |
| jax.random.bernoulli | Custom uniform comparison | Don't hand-roll: bernoulli is cleaner and accepts probability arrays |
| lax.scan carry for PRNG | jax.random.split all keys upfront outside scan | Upfront split requires O(n_trials) keys in memory; carry pattern is idiomatic and avoids that |

## Architecture Patterns

### Recommended Project Structure
```
src/prl_hgf/simulation/
├── agent.py          # Unchanged (legacy NumPy loop, kept for VALID-04 reference)
├── batch.py          # Updated: simulate_batch calls jax_session.py internally
├── jax_session.py    # NEW: simulate_session_jax, simulate_cohort_jax
└── __init__.py       # Updated: export new public functions
```

The new file `jax_session.py` is the natural location. It mirrors the split in `fitting/` where `ops.py` (per-participant) and `hierarchical.py` (batched) are separate files.

### Pattern 1: PRNG Key Threading Through lax.scan Carry

**What:** Include the PRNG key as part of the carry tuple. Split at each step to get a step-subkey for sampling and a carry-key for the next step.

**When to use:** Any lax.scan that samples random values per step. This is the canonical JAX pattern — confirmed by JAX GitHub discussion #7342 and the JAX PRNG design doc.

**Example (adapted from JAX community pattern):**
```python
# Source: JAX GitHub discussion #7342 + project codebase patterns
def _sim_step(carry, x):
    attrs, rng_key, prev_choice = carry
    cue_probs = x  # shape (3,), float — from scan input

    # Split key: step_key for this trial, next_key for carry
    step_key, next_key = jax.random.split(rng_key)
    choice_key, reward_key = jax.random.split(step_key)

    # ... HGF update and choice sampling here ...

    return (new_attrs, next_key, choice), (choice, reward, is_stable)

final_carry, outputs = lax.scan(_sim_step, init_carry, scan_inputs)
```

### Pattern 2: Reusing _clamped_scan Logic Inline

**What:** The Layer 2 NaN clamping from `hierarchical.py::_clamped_scan` is a wrapper that runs `scan_fn` once per step and applies `jnp.where` to revert on instability. The exact same logic applies here — the HGF update inside the simulation step is structurally identical to the logp step.

**When to use:** Any scan step that calls pyhgf's `scan_fn`. The clamping is mandatory (JSIM-02 requirement).

**Pattern (from `src/prl_hgf/fitting/hierarchical.py`, lines 155-193):**
```python
# Source: src/prl_hgf/fitting/hierarchical.py _clamped_step
new_attrs, new_node = scan_fn(prev_attrs, x)

leaves = jax.tree_util.tree_leaves(new_attrs)
all_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))

mu_2_vals = jnp.array([new_attrs[1]["mean"], new_attrs[3]["mean"], new_attrs[5]["mean"]])
mu_2_ok = jnp.all(jnp.abs(mu_2_vals) < _MU_2_BOUND)  # _MU_2_BOUND = 14.0

is_stable = all_finite & mu_2_ok

safe_attrs = jax.tree_util.tree_map(
    lambda n, o: jnp.where(is_stable, n, o),
    new_attrs,
    prev_attrs,
)
```

### Pattern 3: Building scan_inputs from Choice and Reward Inside the Step

**What:** The logp path receives pre-built `(values, observed_cols, time_steps, None)` as static scan inputs. The simulation path generates `choice` and `reward` stochastically each step, then constructs the scan input dynamically inside the step function.

**Critical:** pyhgf `scan_fn` expects `(values, observed_cols, time_steps, None)` as the second argument where `values` is a tuple of 3 scalars (one per cue branch) and `observed_cols` is a tuple of 3 binary scalars.

**Pattern:**
```python
# Source: src/prl_hgf/fitting/hierarchical.py _build_scan_inputs (adapted for per-trial use)
# Inside the scan step, after sampling choice and reward:
reward_float = jnp.float32(reward)
values_t = (
    jnp.where(choice == 0, reward_float, 0.0),
    jnp.where(choice == 1, reward_float, 0.0),
    jnp.where(choice == 2, reward_float, 0.0),
)
observed_t = (
    jnp.float32(choice == 0),
    jnp.float32(choice == 1),
    jnp.float32(choice == 2),
)
time_step = jnp.ones(())  # scalar 1.0
scan_input_t = (values_t, observed_t, time_step, None)
```

### Pattern 4: vmap Over Participants With Pre-split Keys

**What:** vmap requires each vmapped function call to receive a distinct PRNG key. The master key is split into `n_participants` subkeys before calling vmap.

**When to use:** Cohort-level simulation. Confirmed correct by JAX GitHub #7342.

**Example:**
```python
# Source: JAX GitHub discussion #7342 + hierarchical.py in_axes pattern
master_key = jax.random.PRNGKey(seed)
per_participant_keys = jax.random.split(master_key, n_participants)  # shape (P, 2)

simulate_cohort_jax = jax.vmap(simulate_session_jax, in_axes=(0, None, 0))
# in_axes: params_batch axis-0, trial_inputs None (shared), keys axis-0
```

### Pattern 5: Softmax-Stickiness Choice Sampling

**What:** The existing `simulate_agent` computes `logits = beta * p_reward + zeta * stick` and then calls `np.exp(logits) / np.exp(logits).sum()` + `rng.choice`. JAX replaces this with `jax.random.categorical(key, logits)` which accepts unnormalized logits directly.

**When to use:** Choice sampling step in the simulation scan.

```python
# jax.random.categorical takes logits (not probabilities), axis=-1 by default
# Source: JAX docs (jax.random.categorical documentation)
stick = (prev_choice == jnp.arange(3)).astype(jnp.float32)
logits = beta * p_reward + zeta * stick  # p_reward from node_traj or current attrs
choice = jax.random.categorical(choice_key, logits)  # returns int32 scalar
```

### Pattern 6: Reading Prior Beliefs Before Scan Update

**What:** In `simulate_agent`, beliefs are read from `net.attributes[INPUT_NODES[k]]["expected_mean"]` BEFORE calling `input_data`. In the JAX scan step, the equivalent is reading from `carry["attrs"]` before calling `scan_fn`.

**Critical distinction:** The logp path reads `expected_mean` from `node_traj` (post-update trajectory). The simulation path must read beliefs from the carry attrs BEFORE the scan_fn call to drive the choice model — matching the `simulate_agent` semantics.

```python
# Read expected_mean (sigmoid P in [0,1]) from INPUT_NODES (0, 2, 4)
# Source: src/prl_hgf/simulation/agent.py lines 251-255
p_reward = jnp.array([
    attrs[0]["expected_mean"],   # INPUT_NODES = (0, 2, 4) from hgf_2level.py
    attrs[2]["expected_mean"],
    attrs[4]["expected_mean"],
])
```

### Pattern 7: diverged Flag as jnp.any Reduction

**What:** JSIM-02 and JSIM-05 require a per-session `diverged` boolean flag. The logp path collects a `stability_mask` per trial. For simulation, `diverged` is `jnp.any(~stability_mask)` — True if any trial was clamped.

```python
choices, rewards, stability_flags = outputs  # from lax.scan
diverged = jnp.any(~stability_flags)
```

### Anti-Patterns to Avoid

- **Calling pyhgf scan_fn with Python for-loop per trial:** This defeats the entire purpose of Phase 13. Use `lax.scan` end-to-end.
- **Reusing the same PRNG key for all vmap calls:** `jax.vmap(f, in_axes=(0, None))(params, key)` gives identical random draws per participant. Always split first.
- **Python branching inside scan step on traced values:** `if is_stable:` fails under `jax.jit`. Use `jnp.where`/`jax.tree_util.tree_map` exactly as in `_clamped_scan`.
- **Building scan_inputs as Python list inside the step:** pyhgf expects a specific tuple structure `(values, observed_cols, time_steps, None)`. Use tuples, not lists.
- **Constructing jnp.array([kappa, kappa, kappa]) inside vmap'd body:** This is fine for scalars (confirmed working in `_single_logp_3level`), but verify with concrete test since jnp.array construction inside traced code can sometimes cause issues with older JAX versions.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Categorical sampling from logits | Custom softmax + argmax loop | `jax.random.categorical(key, logits)` | Handles numerical stability, is XLA-compiled, works inside lax.scan |
| Bernoulli reward sampling | `jnp.where(jax.random.uniform(...) < prob, ...)` | `jax.random.bernoulli(key, p)` | Cleaner, directly returns bool/int, works inside lax.scan |
| HGF belief update math | Re-implement binary HGF equations | `pyhgf scan_fn` | Phase 12 proof: scan_fn works correctly in lax.scan; reimplementing would break scientific validity |
| Stability clamping | Custom divergence detection | `_clamped_scan` pattern from `hierarchical.py` | Already validated in Phase 12 VALID-01/02 |
| Per-participant PRNG independence in vmap | Manual seeding with participant index | `jax.random.split(master_key, n_participants)` | Standard JAX pattern; integer folding is not independent |

**Key insight:** The simulation path is an extension of the logp path, not a rewrite. Almost every utility from `hierarchical.py` can be reused or ported with minimal changes.

## Common Pitfalls

### Pitfall 1: scan_fn Input Structure Mismatch
**What goes wrong:** `scan_fn` raises an error or silently produces wrong beliefs because the per-trial input tuple structure doesn't match what pyhgf expects.
**Why it happens:** In the logp path, `_build_scan_inputs` constructs the full `(n_trials,)` structure upfront and passes it as scan xs. In the simulation path, each step must construct a single-trial equivalent.
**How to avoid:** Mirror `_build_scan_inputs` logic but for a single trial (scalar, not 1D slice). Check that `values` is a tuple of 3 scalar `jnp.float32` values, `observed_cols` is a tuple of 3 binary scalar `jnp.float32` values, and `time_steps` is a scalar `1.0`. Verify by running `scan_fn(attrs, single_trial_input)` manually before embedding in lax.scan.
**Warning signs:** `scan_fn` output `new_attrs` has NaN in `expected_mean` from trial 1 even with finite parameters.

### Pitfall 2: Reading Beliefs After vs Before HGF Update
**What goes wrong:** Choices are driven by post-update beliefs instead of prior beliefs, causing the softmax to be applied to the wrong probability estimates.
**Why it happens:** The `node_traj` from scan_fn reflects posterior beliefs (after the update). The simulation loop should drive choices from the carry's prior beliefs, not from `node_traj`.
**How to avoid:** Read `attrs[INPUT_NODES[k]]["expected_mean"]` from the carry BEFORE calling `scan_fn`. The `node_traj` output of the scan step is only needed for diagnostics, not for choice generation.
**Warning signs:** VALID-04 statistical equivalence fails because choice distributions are systematically off.

### Pitfall 3: RNG Key Reuse Across vmap Calls
**What goes wrong:** All participants in the cohort make identical choices (or identically correlated choices) because they share the same PRNG key.
**Why it happens:** Passing `in_axes=(0, None)` for the key argument means every vmap call sees the same key, producing identical random draws.
**How to avoid:** Always call `jax.random.split(master_key, n_participants)` before vmapping, giving each participant a distinct subkey. Verify by checking that participant choices differ even for identical parameters.
**Warning signs:** `simulate_cohort_jax` with identical params for all participants returns identical choice sequences.

### Pitfall 4: Cross-Device Floating-Point Differences Breaking Strict Equality
**What goes wrong:** VALID-04 equivalence test fails because per-trial beliefs differ by small floating-point amounts between CPU and GPU, causing choice probabilities to differ.
**Why it happens:** XLA floating-point order of operations differs across hardware backends. Float32 accumulation in HGF updates can differ by 1-2 ULPs.
**How to avoid:** VALID-04 does not require per-trial exact match — only aggregate statistics (mean choice frequency per cue per phase) over 100 replicates. Use `scipy.stats.ks_2samp` with p > 0.05 threshold. Do NOT assert `np.allclose` on per-trial outputs.
**Warning signs:** Test passes on CPU, fails on GPU (or vice versa) — this is expected and not a bug if the aggregate test passes.

### Pitfall 5: simulate_batch DataFrame Schema Drift
**What goes wrong:** Downstream code (`run_sbf_iteration`, `run_power_iteration`) breaks because `simulate_batch` output has different columns.
**Why it happens:** The new JAX path emits numpy arrays from vmap output; if the DataFrame assembly code in batch.py is not updated carefully, column names or dtypes can drift.
**How to avoid:** Keep the existing column list from `batch.py` lines 95-115 as the specification. Add `diverged` as the only new column (it was already there — verify it is preserved). Run `test_batch.py::test_batch_output_shape` and `test_batch_column_values` against the new path.
**Warning signs:** `test_batch.py` fails after the batch.py update.

### Pitfall 6: Parameter Injection in Simulation (Same as Logp Path)
**What goes wrong:** Parameters (omega_2, omega_3, kappa) are not injected into `base_attrs` before the scan, so the HGF runs with default parameters.
**Why it happens:** Forgetting to apply the shallow-copy injection pattern from `ops.py` and `hierarchical.py`.
**How to avoid:** Copy the exact parameter injection block from `_single_logp_3level` in `hierarchical.py` (lines 404-425). The same nodes (1, 3, 5 for omega_2; node 6 for omega_3 and kappa children; nodes 1, 3, 5 for kappa parents) need injection in the simulation path.
**Warning signs:** Choice distributions don't vary with omega_2 parameter changes.

## Code Examples

### Complete scan step skeleton for simulate_session_jax

```python
# Source: synthesized from hierarchical.py _clamped_step (lines 155-193)
# + simulate_agent.py (lines 247-302) + JAX PRNG discussion #7342

def _sim_step(carry, x):
    """Single-trial simulation step for lax.scan.

    Parameters
    ----------
    carry : tuple
        (attrs, rng_key, prev_choice) — HGF belief state, PRNG key, last choice
    x : tuple
        (cue_probs,) — shape (3,) float array of true reward probabilities
        for this trial (pre-computed from generate_session output)

    Returns
    -------
    new_carry : tuple
        (safe_attrs, next_key, choice)
    outputs : tuple
        (choice, reward, is_stable)
    """
    attrs, rng_key, prev_choice = carry
    cue_probs = x  # shape (3,)

    # --- Step 1: Read prior beliefs from carry ---
    p_reward = jnp.array([
        attrs[0]["expected_mean"],  # INPUT_NODES = (0, 2, 4)
        attrs[2]["expected_mean"],
        attrs[4]["expected_mean"],
    ])

    # --- Step 2: Split PRNG key ---
    step_key, next_key = jax.random.split(rng_key)
    choice_key, reward_key = jax.random.split(step_key)

    # --- Step 3: Softmax-stickiness choice sampling ---
    stick = (prev_choice == jnp.arange(3)).astype(jnp.float32)
    logits = beta * p_reward + zeta * stick  # beta, zeta from outer closure
    choice = jax.random.categorical(choice_key, logits)

    # --- Step 4: Bernoulli reward sampling ---
    reward = jax.random.bernoulli(reward_key, cue_probs[choice])
    reward = jnp.int32(reward)

    # --- Step 5: Build per-trial scan input for scan_fn ---
    reward_f = jnp.float32(reward)
    values_t = (
        jnp.where(choice == 0, reward_f, jnp.float32(0.0)),
        jnp.where(choice == 1, reward_f, jnp.float32(0.0)),
        jnp.where(choice == 2, reward_f, jnp.float32(0.0)),
    )
    observed_t = (
        jnp.float32(choice == 0),
        jnp.float32(choice == 1),
        jnp.float32(choice == 2),
    )
    scan_input_t = (values_t, observed_t, jnp.float32(1.0), None)

    # --- Step 6: HGF update via scan_fn (from pyhgf Network) ---
    new_attrs, _ = scan_fn(attrs, scan_input_t)

    # --- Step 7: Layer 2 clamping (tapas-style, from hierarchical.py) ---
    leaves = jax.tree_util.tree_leaves(new_attrs)
    all_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    mu_2_vals = jnp.array([
        new_attrs[1]["mean"], new_attrs[3]["mean"], new_attrs[5]["mean"],
    ])
    mu_2_ok = jnp.all(jnp.abs(mu_2_vals) < _MU_2_BOUND)  # 14.0
    is_stable = all_finite & mu_2_ok

    safe_attrs = jax.tree_util.tree_map(
        lambda n, o: jnp.where(is_stable, n, o),
        new_attrs,
        attrs,
    )

    return (safe_attrs, next_key, choice), (choice, reward, is_stable)
```

### How scan_fn input structure differs between logp and simulation paths

```python
# LOGP PATH (hierarchical.py): full session arrays sliced by scan
# Each xs element is a tuple of (n_trials,) arrays; scan slices at each step
values = (input_data[:, 0:1], input_data[:, 1:2], input_data[:, 2:3])
observed_cols = (observed[:, 0], observed[:, 1], observed[:, 2])
time_steps = jnp.ones(n_trials)
scan_inputs = (values, observed_cols, time_steps, None)  # lax.scan slices these

# SIMULATION PATH: per-trial scalars built inside the step function
# After sampling choice=1, reward=1:
values_t = (jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0))  # reward in chosen slot
observed_t = (jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0))  # 1 for chosen cue
scan_input_t = (values_t, observed_t, jnp.float32(1.0), None)
# NOTE: This is NOT sliced by lax.scan — it is constructed fresh inside the step.
# The scan xs input for simulation is just cue_probs array shape (n_trials, 3).
```

### vmap cohort simulation

```python
# Source: hierarchical.py _batched_logp pattern (lines 486-491)
# adapted for simulation

master_key = jax.random.PRNGKey(master_seed)
# Split into one key per participant
rng_keys_batch = jax.random.split(master_key, n_participants)  # shape (P, 2)

# simulate_session_jax has signature: (params_dict, cue_probs_arr, rng_key) -> outputs
_simulate_cohort = jax.vmap(
    simulate_session_jax,
    in_axes=(0, None, 0),  # params per-participant, cue_probs shared, keys per-participant
)

all_choices, all_rewards, all_diverged = _simulate_cohort(
    params_batch,   # dict of shape-(P,) arrays (omega_2, omega_3, kappa, beta, zeta)
    cue_probs_arr,  # shape (n_trials, 3) — same trial sequence for all
    rng_keys_batch, # shape (P, 2) — distinct key per participant
)
```

### Parameter injection for simulate_session_jax

```python
# Source: hierarchical.py _single_logp_3level (lines 404-425) — identical for simulation
def _inject_params_3level(base_attrs, omega_2, omega_3, kappa):
    """Inject 3-level parameters into a shallow-copied attrs dict."""
    attrs = dict(base_attrs)

    # omega_2 into level-1 belief nodes (1, 3, 5)
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["tonic_volatility"] = omega_2
        attrs[idx] = node

    # omega_3 and kappa children-side into volatility node 6
    node6 = dict(attrs[6])
    node6["tonic_volatility"] = omega_3
    node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
    attrs[6] = node6

    # kappa parents-side into nodes 1, 3, 5
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["volatility_coupling_parents"] = jnp.array([kappa])
        attrs[idx] = node

    return attrs
```

### VALID-04 equivalence test structure

```python
# Source: test_hierarchical_logp.py VALID-02 pattern adapted for simulation
from scipy.stats import ks_2samp

n_replicates = 100
legacy_freq = []    # mean choice frequency per cue per phase across replicates
jax_freq = []

for seed in range(n_replicates):
    # Legacy path (simulate_agent with matched NumPy seed)
    # ... run simulate_agent with np.random.default_rng(seed) ...
    
    # JAX path (simulate_session_jax with matched JAX key)
    # ... run simulate_session_jax with jax.random.PRNGKey(seed) ...
    
    # Aggregate: mean choice frequency per cue (0, 1, 2) per phase label
    # Exact per-trial ordering will differ — only aggregate matters

# KS test per cue per phase: p > 0.05 means distributions are statistically equivalent
for cue in range(3):
    stat, p = ks_2samp(legacy_freq_cue[cue], jax_freq_cue[cue])
    assert p > 0.05, f"cue {cue}: KS p={p:.4f} (expected > 0.05)"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NumPy for-loop over trials (simulate_agent) | lax.scan over trials | Phase 13 | Single compiled XLA kernel; vmap across participants |
| Sequential participant loop in simulate_batch | jax.vmap across participants | Phase 13 | O(P) sequential calls → 1 compiled kernel call |
| np.random.Generator for stochastic sampling | jax.random.categorical / bernoulli | Phase 13 | Different RNG sequence; VALID-04 allows aggregate equivalence only |
| JIT pre-warm via dummy input_data call | JIT compilation happens at first jax.jit call | Phase 13 | Pre-warm pattern still valid but optional with JAX-native path |

## Open Questions

1. **scan_fn single-trial input format**
   - What we know: The logp path passes full `(n_trials,)` arrays as xs and lax.scan slices at each step. The simulation step must call scan_fn with a single-trial tuple constructed inside the step.
   - What's unclear: Whether pyhgf's `scan_fn` expects the single-trial element to be a slice (1D array of shape (1,)) or a scalar. The logp path passes `input_data[:, 0:1]` (shape `(n_trials, 1)`) — after scan slicing this becomes shape `(1,)` per step, not a scalar. The simulation path must replicate this exactly.
   - Recommendation: In the first task, verify scan_fn input format by running it manually with a single-trial tuple before embedding in lax.scan. Add a smoke test that calls `scan_fn(attrs, single_trial_input)` and confirms `new_attrs` has finite `expected_mean` values.

2. **Carrying prev_choice through first trial**
   - What we know: `simulate_agent` uses `prev_choice = -1` as a sentinel so the stickiness term is zero on trial 0. JAX needs a JAX-compatible sentinel.
   - What's unclear: Whether `jnp.int32(-1)` is safe, or whether a special `jnp.int32(3)` (out-of-range) sentinel is safer for the `jnp.where` comparison.
   - Recommendation: Use `jnp.int32(-1)` as the initial carry value. The stickiness comparison `(prev_choice == jnp.arange(3))` evaluates to all-False for `-1`, which is the correct zero-stickiness behavior. Verify in a unit test.

3. **params_batch input structure for vmap**
   - What we know: The logp path uses separate shape-`(P,)` arrays per parameter (`omega_2_batch`, `beta_batch`, etc.) and vmap axes are `(0, 0, 0, 0, 0, ...)`. The simulation path could use either a dict of arrays or separate positional args.
   - What's unclear: Whether to use a dict pytree or flat positional arguments for `params_batch`. Dicts work as vmap pytrees in JAX when the axis is specified.
   - Recommendation: Use a flat params dict (e.g., `{"omega_2": jnp.array(...), ...}`) with `in_axes=(0, None, 0)` where `0` for the dict means all leaves are vmapped over axis 0. This is cleaner than 5 positional args. Verify with a minimal vmap test before building the full cohort path.

4. **scan_fn compatibility with single-trial construction inside vmap**
   - What we know: Phase 12 confirmed `scan_fn` works under `jax.vmap` for the logp path. The simulation path adds PRNG key operations inside the vmap body.
   - What's unclear: Whether `jax.random.bernoulli` on a scalar probability extracted from a vmapped array (e.g., `cue_probs[choice]`) traces correctly. The index `choice` is a traced int32 — JAX supports dynamic indexing via `jnp.take` or `cue_probs.at[choice]` but direct `cue_probs[choice]` also works for 1D arrays.
   - Recommendation: Use `cue_probs[choice]` for direct indexing — JAX's `__getitem__` with a traced integer scalar uses `lax.dynamic_slice` under the hood and is safe inside jit/vmap.

## Sources

### Primary (HIGH confidence)
- `src/prl_hgf/fitting/hierarchical.py` — clamped_scan pattern, vmap pattern, scan_fn usage confirmed working in Phase 12
- `src/prl_hgf/simulation/agent.py` — legacy simulate_agent: the exact semantics to replicate (belief reads before update, stickiness sentinel, diverged flag)
- `src/prl_hgf/simulation/batch.py` — DataFrame schema that must be preserved
- JAX GitHub discussion #7342 — vmap + PRNG split pattern (verified, multiple maintainer responses)
- JAX GitHub discussion #8503 — lax.scan with PRNG key in carry pattern (verified)

### Secondary (MEDIUM confidence)
- JAX docs (jax.random.categorical) — accepts logits, returns int32 index; verified via search result snippet
- JAX docs (jax.lax.scan) — carry + xs + outputs structure; verified via search result snippet

### Tertiary (LOW confidence)
- pyproject.toml dependencies — JAX >=0.4.26,<0.4.32 and pyhgf >=0.2.8,<0.3 confirmed from project file (HIGH confidence for these specific versions)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are pinned in pyproject.toml; no new dependencies needed
- Architecture patterns: HIGH — scan_fn + clamping pattern proven in Phase 12; PRNG key carry pattern is canonical JAX; vmap pattern mirrors hierarchical.py exactly
- Pitfalls: HIGH for scan_fn input structure and RNG key reuse (common JAX mistakes); MEDIUM for scan_fn single-trial format (open question 1)

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (pyhgf and JAX APIs are stable; Phase 12 validation gives HIGH confidence in the core patterns)
