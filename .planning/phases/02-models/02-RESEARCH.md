# Phase 02: Models - Research

**Researched:** 2026-04-05
**Domain:** pyhgf 0.2.8 Network API — custom HGF topologies, partial feedback, response functions
**Confidence:** HIGH (all findings verified from installed source code)

## Summary

This research investigated the pyhgf 0.2.8 source code installed in
`C:/Users/aman0087/AppData/Local/miniforge3/envs/ds_env/Lib/site-packages/pyhgf/`
to determine the exact API for building 3-parallel-branch binary HGF networks
with a shared volatility parent. All findings are HIGH confidence — verified
directly from installed source, not from training memory or web results.

The `Network` class (not `HGF`) is the correct base for custom topologies.
`HGF` is a convenience subclass that only supports a single-branch 2- or
3-level filter. For 3 parallel cue branches, use `Network` directly with
`add_nodes()` calls. Partial feedback (unchosen cues) is handled natively via
the `observed` mask passed to `input_data()` — setting `observed=0` cancels the
prediction error contribution for that input node on that trial, and the code
verifies this by multiplying `value_prediction_error *= attributes[node_idx]["observed"]`
in both the binary and continuous posterior updates.

The `HGFDistribution` PyMC Op is hardwired to the single-branch `HGF` class with
a fixed set of named parameters (`tonic_volatility_1/2/3`, `volatility_coupling_1/2`,
etc.). It **cannot** be used directly with a custom multi-branch `Network`. For
Phase 2 (forward pass / belief extraction) this is irrelevant — `HGFDistribution`
belongs to Phase 4 (fitting). Phase 2 only needs `Network.input_data()` and
`Network.surprise()`.

**Primary recommendation:** Build both models as `Network` subclasses using
`add_nodes(kind="binary-state")` and `add_nodes(kind="continuous-state")` with
explicit edge wiring. Use `input_data(input_data=..., observed=...)` for partial
feedback. Custom response function signature is `fn(hgf, response_function_inputs,
response_function_parameters) -> float`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyhgf | 0.2.8 | JAX-backed HGF — Network API for custom topologies | Installed; chosen by project |
| JAX / jax.numpy | installed with pyhgf | Array ops, scan, JIT | Backend for pyhgf |
| numpy | installed | Input data arrays, mask arrays | pyhgf `input_data` expects np.ndarray |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jax.nn.sigmoid | jax built-in | Convert level-1 mean → probability | Response function: μ₁ₖ → P(reward) |
| jax.numpy.inf | jax built-in | Flag NaN/invalid fits | Response fn: return jnp.inf on NaN |
| pytensor | installed with pyhgf | PyMC bridge (HGFDistribution) | Phase 4 only, not Phase 2 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `Network` (custom) | `HGF` subclass | `HGF` only supports 1-branch, is too restrictive |
| `observed` mask for partial feedback | NaN input values | NaN breaks JAX scan; `observed=0` is the correct mechanism |
| `input_custom_sequence` | `input_data` | Only needed if update sequence must change per trial; not needed here |

**Installation:** already installed in `ds_env` conda environment.

---

## Architecture Patterns

### Recommended Module Structure
```
src/prl_hgf/models/
├── __init__.py          # exports HGF2Level, HGF3Level, softmax_stickiness_surprise
├── hgf_2level.py        # Network subclass: 3-branch 2-level binary HGF
├── hgf_3level.py        # Network subclass: 3-branch 3-level binary HGF
└── response.py          # softmax_stickiness_surprise response function
```

### Pattern 1: Node Numbering for 3-Branch Binary HGF

When `add_nodes()` is called sequentially, nodes receive sequential integer
indexes starting from 0. Plan the index layout before coding.

**2-level model node layout (9 nodes total):**
```
Node 0: binary-state  — cue 0 input
Node 1: continuous-state — cue 0 level-1 (value parent of node 0)
Node 2: binary-state  — cue 1 input
Node 3: continuous-state — cue 1 level-1 (value parent of node 2)
Node 4: binary-state  — cue 2 input
Node 5: continuous-state — cue 2 level-1 (value parent of node 4)
```
Input nodes: (0, 2, 4) — these are what `input_idxs` must be set to.
Belief nodes: (1, 3, 5) — μ₁ and σ₁ extracted from `node_trajectories[1/3/5]`.

**3-level model node layout (10 nodes total):**
```
Node 0: binary-state  — cue 0 input
Node 1: continuous-state — cue 0 level-1 (value parent of node 0)
Node 2: binary-state  — cue 1 input
Node 3: continuous-state — cue 1 level-1 (value parent of node 2)
Node 4: binary-state  — cue 2 input
Node 5: continuous-state — cue 2 level-1 (value parent of node 4)
Node 6: continuous-state — shared volatility parent (volatility parent of nodes 1, 3, 5)
```
Input nodes: (0, 2, 4).
Volatility node: 6 — μ₂ and σ₂ extracted from `node_trajectories[6]`.

### Pattern 2: Building a 2-Level Network

```python
# Source: pyhgf/model/network.py + pyhgf/model/hgf.py (binary model construction)
from pyhgf.model import Network
import numpy as np

def build_2level_network(omega_2: float = -4.0) -> Network:
    net = Network()

    # Branch 0: binary input (node 0) + continuous state (node 1)
    net.add_nodes(kind="binary-state")                          # node 0
    net.add_nodes(
        kind="continuous-state",
        value_children=0,                                       # node 1, parent of node 0
        node_parameters={"tonic_volatility": omega_2},
    )

    # Branch 1: binary input (node 2) + continuous state (node 3)
    net.add_nodes(kind="binary-state")                          # node 2
    net.add_nodes(
        kind="continuous-state",
        value_children=2,                                       # node 3, parent of node 2
        node_parameters={"tonic_volatility": omega_2},
    )

    # Branch 2: binary input (node 4) + continuous state (node 5)
    net.add_nodes(kind="binary-state")                          # node 4
    net.add_nodes(
        kind="continuous-state",
        value_children=4,                                       # node 5, parent of node 4
        node_parameters={"tonic_volatility": omega_2},
    )

    # Explicitly name the input nodes (avoids auto-detection ambiguity)
    net.input_idxs = (0, 2, 4)

    return net
```

### Pattern 3: Building a 3-Level Network with Shared Volatility Parent

```python
# Source: pyhgf/model/hgf.py (binary 3-level) + pyhgf/model/network.py add_nodes
from pyhgf.model import Network

def build_3level_network(
    omega_2: float = -4.0,
    omega_3: float = -4.0,
    kappa: float = 1.0,
) -> Network:
    net = Network()

    # Branch 0
    net.add_nodes(kind="binary-state")                          # node 0
    net.add_nodes(
        kind="continuous-state",
        value_children=0,
        node_parameters={"tonic_volatility": omega_2},
    )                                                           # node 1

    # Branch 1
    net.add_nodes(kind="binary-state")                          # node 2
    net.add_nodes(
        kind="continuous-state",
        value_children=2,
        node_parameters={"tonic_volatility": omega_2},
    )                                                           # node 3

    # Branch 2
    net.add_nodes(kind="binary-state")                          # node 4
    net.add_nodes(
        kind="continuous-state",
        value_children=4,
        node_parameters={"tonic_volatility": omega_2},
    )                                                           # node 5

    # Shared volatility parent: volatility children = nodes 1, 3, 5
    # kappa is the coupling strength — passed as list of strengths
    net.add_nodes(
        kind="continuous-state",
        volatility_children=([1, 3, 5], [kappa, kappa, kappa]),
        node_parameters={"tonic_volatility": omega_3},
    )                                                           # node 6

    net.input_idxs = (0, 2, 4)
    return net
```

**Critical detail for `volatility_children` with multiple children and coupling
strengths:** from `get_couplings()` source, when passing a tuple it must be
`(list_of_indexes, list_of_strengths)`. The call
`volatility_children=([1, 3, 5], [kappa, kappa, kappa])` produces
`coupling_idxs=(1, 3, 5)` and `coupling_strengths=(kappa, kappa, kappa)`.

### Pattern 4: Running Forward Pass with Partial Feedback

```python
# Source: pyhgf/model/network.py input_data(), pyhgf/utils/beliefs_propagation.py
import numpy as np

# input_data shape: (n_trials, n_input_nodes) — columns map to input_idxs order
# observed shape: (n_trials, n_input_nodes) — 1=observed, 0=not observed

n_trials = 420
# Each row: [reward_cue0, reward_cue1, reward_cue2]
# For unchosen cues: value can be anything (0.0 is fine), but observed=0
input_data = np.zeros((n_trials, 3), dtype=float)
observed = np.zeros((n_trials, 3), dtype=int)

for t, trial in enumerate(trials):
    chosen = trial.chosen_cue        # 0, 1, or 2
    reward = trial.reward            # 1 or 0
    input_data[t, chosen] = reward
    observed[t, chosen] = 1         # only the chosen cue is "observed"
    # unchosen cues: observed stays 0 — their PE is zeroed out automatically

net = build_2level_network()
net.input_data(input_data=input_data, observed=observed)
```

The `observed` parameter in `input_data()` accepts either:
- `None` (defaults to all-ones mask — all inputs observed every trial)
- A 2D numpy array of shape `(n_trials, n_input_nodes)` with int dtype
- Or a tuple of 1D arrays

When `observed[t, k] = 0`, `set_observation` is called with `observed=0` for
that node, and both the binary prediction error step and the continuous posterior
update multiply the prediction error by `attributes[node_idx]["observed"]`,
effectively setting PE=0. The parent's mean and precision are NOT updated for
that cue on that trial — precisely what partial feedback requires.

### Pattern 5: Extracting Belief Trajectories

```python
# Source: pyhgf/utils/to_pandas.py + pyhgf/model/network.py
# After calling net.input_data(...), node_trajectories is populated.

# node_trajectories is a dict keyed by node index.
# Each entry is itself a dict of arrays, length n_trials.

# 2-level model:
mu1_cue0 = net.node_trajectories[1]["mean"]           # shape (n_trials,)
sigma1_cue0 = 1.0 / net.node_trajectories[1]["precision"]  # precision → variance
mu1_cue1 = net.node_trajectories[3]["mean"]
mu1_cue2 = net.node_trajectories[5]["mean"]

# The binary input node's expected_mean is the sigmoid-transformed level-1 mean:
p_reward_cue0 = net.node_trajectories[0]["expected_mean"]   # P(reward | cue 0)

# 3-level model adds:
mu2_shared = net.node_trajectories[6]["mean"]         # volatility estimate
sigma2_shared = 1.0 / net.node_trajectories[6]["precision"]

# Columns available in node_trajectories[i]:
# binary nodes (node_type=1):  mean, expected_mean, precision, expected_precision, observed
# continuous nodes (node_type=2): mean, expected_mean, precision, expected_precision,
#                                  tonic_volatility, tonic_drift, autoconnection_strength,
#                                  observed, temp{...}
```

### Pattern 6: Custom Response Function

```python
# Source: pyhgf/model/network.py surprise() + pyhgf/response.py examples
# Signature is mandatory: fn(hgf, response_function_inputs, response_function_parameters)
import jax.numpy as jnp
from jax.nn import sigmoid

def softmax_stickiness_surprise(
    hgf,
    response_function_inputs,    # choices: np.ndarray shape (n_trials,), values 0/1/2
    response_function_parameters,  # [beta, zeta] or jnp.array([beta, zeta])
) -> float:
    """Surprise (negative log-likelihood) under softmax + stickiness response model.

    P(choice=k | t) ∝ exp(beta * mu1k + zeta * I[prev_choice=k])
    where mu1k is the continuous-state node mean (log-odds) for cue k.
    """
    beta = response_function_parameters[0]
    zeta = response_function_parameters[1]

    choices = response_function_inputs  # shape (n_trials,)
    n_trials = len(choices)

    # Collect mu1 for each cue from continuous-state nodes (1, 3, 5)
    mu1_0 = hgf.node_trajectories[1]["expected_mean"]   # shape (n_trials,)
    mu1_1 = hgf.node_trajectories[3]["expected_mean"]
    mu1_2 = hgf.node_trajectories[5]["expected_mean"]

    # Stack: shape (n_trials, 3)
    mu1 = jnp.stack([mu1_0, mu1_1, mu1_2], axis=1)

    # Build stickiness term: I[prev_choice=k]
    # First trial: no previous choice, stickiness = 0
    prev_choices = jnp.concatenate([jnp.array([-1]), choices[:-1]])
    stick = jnp.stack([
        (prev_choices == 0).astype(float),
        (prev_choices == 1).astype(float),
        (prev_choices == 2).astype(float),
    ], axis=1)                                           # shape (n_trials, 3)

    # Logits
    logits = beta * mu1 + zeta * stick                  # (n_trials, 3)

    # Log-softmax for numerical stability
    log_probs = logits - jnp.log(jnp.sum(jnp.exp(logits), axis=1, keepdims=True))

    # Per-trial log-likelihood of the chosen cue
    chosen_idx = choices.astype(int)
    trial_loglik = log_probs[jnp.arange(n_trials), chosen_idx]

    surprise = -jnp.sum(trial_loglik)

    # Return inf if NaN (model failed to fit)
    return jnp.where(jnp.isnan(surprise), jnp.inf, surprise)
```

**Call pattern:**
```python
net.input_data(input_data=input_data, observed=observed)
total_surprise = net.surprise(
    response_function=softmax_stickiness_surprise,
    response_function_inputs=choices_array,
    response_function_parameters=jnp.array([2.0, 0.5]),
)
```

### Pattern 7: Parameter Mapping (MOD-05)

| Project name | pyhgf attribute | Node(s) | How to set |
|---|---|---|---|
| ω₂ | `tonic_volatility` | Continuous-state nodes 1, 3, 5 | `node_parameters={"tonic_volatility": omega_2}` at creation |
| ω₃ | `tonic_volatility` | Shared volatility node 6 | `node_parameters={"tonic_volatility": omega_3}` |
| κ | `volatility_coupling_children` / `volatility_coupling_parents` | Edge 6→(1,3,5) | `volatility_children=([1,3,5],[kappa,kappa,kappa])` |
| β | response parameter | n/a | `response_function_parameters[0]` |
| ζ | response parameter | n/a | `response_function_parameters[1]` |

To update parameters after construction (for fitting), modify
`net.attributes[node_idx]["tonic_volatility"]` directly and call
`net.create_belief_propagation_fn(overwrite=True)` before re-running.

### Anti-Patterns to Avoid

- **Using `HGF` instead of `Network` for multi-branch models:** `HGF.__init__`
  builds a fixed single-branch topology; there is no way to extend it to 3
  parallel branches.
- **Using NaN as the "no observation" value:** `input_data` calls
  `set_observation(values=..., observed=...)`. The value for an unobserved
  input is stored in `attributes[node_idx]["mean"]` but zeroed out by `observed=0`.
  NaN values would propagate through JAX and corrupt the computation graph.
- **Setting `observed=False` (bool) instead of `observed=0` (int):** The scan
  function passes `observed` as a traced JAX array; use `int` dtype (0/1) not bool.
- **Calling `net.surprise()` without calling `net.input_data()` first:**
  `node_trajectories` will be empty and the response function will fail.
- **Assuming node index ordering differs from add_nodes insertion order:** Nodes
  are indexed sequentially starting from 0 in insertion order. Index -1 is the
  special time-step node (`attributes[-1]`).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Partial feedback masking | Custom "skip node" logic | `observed` mask in `input_data()` | Native: PE is multiplied by `attributes[node_idx]["observed"]` in every update step |
| Sigmoid transform of level-1 belief | Custom sigmoid | `jax.nn.sigmoid` (already used internally) | Already applied in `binary_state_node_prediction` — `expected_mean` of binary node IS sigmoid(μ₁) |
| Update sequence construction | Hand-build update order | `get_update_sequence()` called by `create_belief_propagation_fn()` | Automatically infers correct topological order from edges |
| JAX scan over time series | Manual Python loop | `jax.lax.scan` (called inside `input_data`) | Required for JIT-compilation and gradient computation |
| Log-softmax computation | `log(softmax(x))` | `logits - log(sum(exp(logits)))` inline | Numerically stable; jnp.log(jax.nn.softmax(x)) also works |

**Key insight:** The `observed` mask is the ONLY correct mechanism for partial
feedback in pyhgf 0.2.8. Any attempt to manipulate the update sequence per
trial introduces non-static structure that breaks `jit`-compiled scan.

---

## Common Pitfalls

### Pitfall 1: JAX Static Argument Violations
**What goes wrong:** `beliefs_propagation` is decorated with `@partial(jit, static_argnames=("update_sequence", "edges", "input_idxs", "observations"))`. Modifying `edges` or `input_idxs` after `create_belief_propagation_fn()` has been called will silently use stale compiled code.
**Why it happens:** JAX traces and compiles against static args at first call; subsequent calls with different static args require recompilation.
**How to avoid:** Call `create_belief_propagation_fn(overwrite=True)` after any structural change, or set `net.scan_fn = None` to force recompilation on next `input_data()` call.
**Warning signs:** Network produces incorrect trajectories after parameter/structure changes.

### Pitfall 2: Wrong `expected_mean` vs `mean` in Response Function
**What goes wrong:** The binary node has both `mean` (posterior, the actual 0/1 observation) and `expected_mean` (the predicted probability from sigmoid). The response function should use `expected_mean` of binary nodes (the sigmoid-transformed prediction), NOT `mean` (which is the observation itself).
**Why it happens:** `set_observation` sets `mean = observed_value` (0 or 1). `expected_mean` is set during the prediction step as `sigmoid(mu_parent)`.
**How to avoid:** Use `hgf.node_trajectories[binary_idx]["expected_mean"]` for the predicted reward probability, and `hgf.node_trajectories[continuous_idx]["expected_mean"]` for the log-odds belief.
**Warning signs:** Surprise is always 0 or always infinite.

### Pitfall 3: `input_data` Array Shape
**What goes wrong:** `input_data` expects a 2D array `(n_trials, n_features)`. If a 1D array is passed, it is auto-expanded with `input_data[:, jnp.newaxis]`, but this makes it a single-input-node array, which conflicts with a 3-input-node network.
**Why it happens:** The network has 3 input nodes, so `input_data` must have 3 columns. The split is done by `np.split(input_data, split_indices, axis=1)`.
**How to avoid:** Always pass `input_data` as shape `(n_trials, 3)` and `observed` as shape `(n_trials, 3)`.
**Warning signs:** `IndexError` or shape mismatch in `beliefs_propagation`.

### Pitfall 4: Stickiness Term on First Trial
**What goes wrong:** On trial 0, there is no previous choice. If `prev_choices[0]` is set to any cue index, the stickiness term is incorrectly non-zero on the first trial.
**Why it happens:** Simple array indexing produces a nonsensical "previous choice."
**How to avoid:** Use sentinel value -1 for the first trial's previous choice. Since no cue index is -1, `(prev_choices == k)` is False for all k on trial 0. This is exactly RSP-04.
**Warning signs:** Surprise on trial 0 is biased toward whichever cue has index 0.

### Pitfall 5: Volatility Coupling Strength for Shared Parent
**What goes wrong:** If `volatility_children=([1, 3, 5], [1.0, 1.0, 1.0])` is not passed as a tuple-of-lists, `get_couplings()` may interpret the argument differently.
**Why it happens:** `get_couplings()` checks `isinstance(indexes, tuple)` and interprets the first element as index list and second as strength list. If only a plain list `[1, 3, 5]` is passed, it defaults all coupling strengths to 1.0 — this is fine for the default case but κ must be set explicitly if it differs from 1.0.
**How to avoid:** Always pass `([1, 3, 5], [kappa, kappa, kappa])` as the tuple form when coupling strengths matter.
**Warning signs:** κ parameter has no effect on model behavior.

### Pitfall 6: `HGFDistribution` Cannot Be Used with Custom `Network`
**What goes wrong:** `HGFDistribution.__init__` creates its own `HGF()` instance internally and wires a fixed `logp()` function to it. It cannot accept an externally built `Network` with 3 branches.
**Why it happens:** `HGFDistribution` is designed for the simple 1-branch `HGF` only.
**How to avoid:** Phase 2 uses `Network.input_data()` + `Network.surprise()` directly. Phase 4 (fitting) will need a custom PyMC Op that wraps the multi-branch network's `logp`.
**Warning signs:** Attempting to pass a custom `Network` to `HGFDistribution` constructor will fail immediately.

### Pitfall 7: `observed` dtype must be int, not bool
**What goes wrong:** JAX traced arrays for `observed` pass through `attributes[node_idx]["observed"]` which is multiplied against the prediction error. Boolean arrays may not behave correctly under JAX tracing.
**Why it happens:** `set_observation(observed=observed)` sets the attribute; downstream it is multiplied as `value_prediction_error *= attributes[node_idx]["observed"]`. With bool dtype this is technically fine in numpy but can cause issues under JAX tracing.
**How to avoid:** Use `np.ones(..., dtype=int)` and `np.zeros(..., dtype=int)` for the observed mask. Explicitly cast: `observed.astype(int)`.

---

## Code Examples

### Minimal Forward Pass Smoke Test (2-level)

```python
# Source: pyhgf/model/network.py + pyhgf/model/hgf.py patterns
from pyhgf.model import Network
import numpy as np

net = Network()
net.add_nodes(kind="binary-state")
net.add_nodes(kind="continuous-state", value_children=0,
              node_parameters={"tonic_volatility": -4.0})
net.add_nodes(kind="binary-state")
net.add_nodes(kind="continuous-state", value_children=2,
              node_parameters={"tonic_volatility": -4.0})
net.add_nodes(kind="binary-state")
net.add_nodes(kind="continuous-state", value_children=4,
              node_parameters={"tonic_volatility": -4.0})
net.input_idxs = (0, 2, 4)

n_trials = 100
input_data = np.random.randint(0, 2, size=(n_trials, 3)).astype(float)
observed = np.zeros((n_trials, 3), dtype=int)
observed[:, 0] = 1  # always observe cue 0 (test only)

net.input_data(input_data=input_data, observed=observed)

# Extract beliefs
mu1_cue0 = net.node_trajectories[1]["mean"]
assert mu1_cue0.shape == (n_trials,)
assert not np.any(np.isnan(mu1_cue0))
```

### Confirming Partial Feedback: Unchosen Cue Belief is Unchanged

```python
# After running with observed[:,1]=0, observed[:,2]=0:
# Nodes 3 and 5 should have constant mean = initial mean (0.0)
mu1_cue1 = net.node_trajectories[3]["mean"]
# All values should equal initial mean because PE was always zeroed out
assert np.allclose(mu1_cue1, 0.0), f"Expected constant, got {mu1_cue1[:5]}"
```

### Extracting Volatility Node (3-level)

```python
# After building 3-level net and running input_data:
mu2 = net.node_trajectories[6]["mean"]        # volatility estimate time series
pi2 = net.node_trajectories[6]["precision"]   # precision of volatility
# Volatility should increase near reversal points
```

### response_function_parameters as JAX array for PyMC compatibility

```python
import jax.numpy as jnp

# In Phase 4, beta and zeta will be PyMC random variables.
# Already in Phase 2, match the expected array form:
params = jnp.array([2.0, 0.5])  # [beta, zeta]
surprise = net.surprise(
    response_function=softmax_stickiness_surprise,
    response_function_inputs=choices,
    response_function_parameters=params,
)
assert jnp.isfinite(surprise)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standard HGF update | eHGF update (default in 0.2.x) | ~0.2.0 | Fewer NaN/invalid precision updates; better sampling |
| `HGF(model_type="binary", n_levels=3)` | `Network` with `add_nodes` for custom topologies | 0.2.x | Custom multi-branch architectures are now first-class |
| Manual update sequences | Auto-generated by `get_update_sequence()` | 0.2.x | Topology-aware update order derived automatically from edges |

**Deprecated/outdated:**
- Old Matlab HGF Toolbox API naming (`x1`, `x2`, `mu1`, `mu2` as top-level attributes): replaced by `node_trajectories[idx]["mean"]` dict access.
- Single `model_type` parameter in old versions: now use explicit `add_nodes(kind=...)` for fine-grained control.

---

## Open Questions

1. **How exactly does the shared volatility parent interact when one branch has `observed=0`?**
   - What we know: The binary node PE is zeroed. The continuous-state level-1 node (e.g., node 1) still computes its volatility PE (`volatility_prediction_error`) based on its own precision ratio — this is NOT gated by `observed`. The shared volatility parent (node 6) will still receive volatility PE from all 3 level-1 nodes even when only 1 cue was observed.
   - What's unclear: Whether the volatility PE from an unobserved branch is meaningfully non-zero or effectively zero due to precision not changing.
   - Recommendation: Verify empirically in unit test: check that the volatility node's mean does NOT change when all 3 cues have `observed=0`.

2. **`volatility_prediction_error` and `observed` flag interaction for level-1 → level-2**
   - What we know: In `posterior_update_mean_continuous_node.py`, volatility coupling updates multiply by `attributes[volatility_child_idx]["observed"]` (line confirmed: `precision_weigthed_prediction_error *= attributes[volatility_child_idx]["observed"]`). The `volatility_child_idx` here is the continuous-state node (1, 3, or 5), not the binary node. The continuous-state nodes do not have their `observed` flag set by `input_data` — only the binary input nodes do.
   - What's unclear: The continuous-state nodes (1, 3, 5) always have `observed=1` (their default). Only the binary nodes (0, 2, 4) have `observed` set per trial. So the volatility PE from level-1 to level-2 is NOT gated by whether the cue was chosen.
   - Recommendation: This is the correct scientific behavior for this model — the agent always updates its volatility estimate based on "time passing" even for unchosen cues (the precision decreases over time). Confirm with pyhgf documentation or author's examples. Accept this behavior for Phase 2; document the distinction.

3. **`input_data` `observed` parameter for 3-input network: 2D array vs tuple**
   - What we know: The code path shows `isinstance(observed, np.ndarray)` branches to `tuple(observed[:, i] for i in range(observed.shape[1]))`. A 2D numpy array `(n_trials, 3)` should work.
   - What's unclear: Whether int dtype is required or float works.
   - Recommendation: Use `np.zeros/ones(..., dtype=int)` to be safe.

---

## Sources

### Primary (HIGH confidence)
- Installed source: `pyhgf/model/network.py` — `Network` class, `add_nodes()`, `input_data()`, `surprise()`, `add_edges()`
- Installed source: `pyhgf/model/add_nodes.py` — `add_binary_state()`, `add_continuous_state()`, `get_couplings()`, `insert_nodes()`
- Installed source: `pyhgf/model/hgf.py` — `HGF.__init__` binary 3-level construction (reference pattern)
- Installed source: `pyhgf/utils/beliefs_propagation.py` — scan loop, `observed` mask handling
- Installed source: `pyhgf/updates/observation.py` — `set_observation()` sets `mean` and `observed`
- Installed source: `pyhgf/updates/prediction_error/binary.py` — `value_prediction_error *= attributes[node_idx]["observed"]`
- Installed source: `pyhgf/updates/posterior/continuous/posterior_update_mean_continuous_node.py` — volatility PE also multiplied by `observed`
- Installed source: `pyhgf/updates/prediction/binary.py` — confirms `expected_mean = sigmoid(mu_parent)`
- Installed source: `pyhgf/response.py` — response function signature examples
- Installed source: `pyhgf/distribution.py` — `HGFDistribution` (Phase 4 reference; confirmed incompatible with custom Network)
- Installed source: `pyhgf/utils/to_pandas.py` — how to access `node_trajectories[i]["mean"]`
- Installed source: `pyhgf/typing.py` — `AdjacencyLists`, `UpdateSequence` data structures

### Secondary (MEDIUM confidence)
- None needed — all findings verified from source.

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — version confirmed via `python -c "import pyhgf; print(pyhgf.__version__)"` → `0.2.8`
- Architecture (node layout, add_nodes API): HIGH — verified from `network.py` and `add_nodes.py` source
- Partial feedback mechanism: HIGH — verified from `beliefs_propagation.py`, `observation.py`, `binary.py` prediction_error, and `posterior_update_mean_continuous_node.py`
- Response function signature: HIGH — verified from `network.py` `surprise()` and `response.py` examples
- HGFDistribution incompatibility with custom Network: HIGH — verified from `distribution.py` source showing it creates its own `HGF()` internally
- Open question about volatility PE gating: MEDIUM — code read confirms behavior but scientific implication requires empirical verification

**Research date:** 2026-04-05
**Valid until:** 2026-06-05 (pyhgf 0.2.8 is pinned in ds_env; stable until intentional upgrade)
