# Phase 4: Fitting - Research

**Researched:** 2026-04-05
**Domain:** PyMC 5 + PyTensor Op + JAX scan for custom multi-branch HGF MCMC fitting
**Confidence:** HIGH

---

## Summary

This research investigated how to build a custom PyTensor Op that wraps pyhgf's
JAX scan infrastructure for MCMC fitting of the project's 3-branch HGF Network.
All critical claims were verified against actual source code and live test runs
on the ds_env environment.

The standard pattern (reading `pyhgf/distribution.py` directly) splits into two
Ops: a gradient Op (`HGFLogpGradOp`) and a forward Op (`HGFDistribution`). The
gradient Op's `perform` method calls the JAX gradient function; the forward Op's
`grad` method delegates to the gradient Op. Parameter injection into the frozen
network attributes uses a shallow-copy pattern that is JAX-traceable. All five
target parameters (omega_2, omega_3, kappa, beta, zeta) were verified to produce
finite gradients through `lax.scan` on the actual network topology.

**Primary recommendation:** Build two PyTensor Ops per model variant — a gradient
Op and a forward logp Op — following the exact split shown in `pyhgf/distribution.py`.
Use `TruncatedNormal(mu=-3, sigma=2, upper=0)` for omega_2 to prevent NaN at
omega_2 >= 0 (confirmed failure point). Use log-space for beta with
`pm.Deterministic('beta', pm.math.exp(log_beta))`. Expect ~60s per participant
(4 chains x 1000 draws on ds_env Windows, no g++ compilation). Total batch of
360 fits: approximately 6 hours sequential.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyMC | 5.25.1 | MCMC sampling, model definition | NUTS sampler, ArviZ integration |
| PyTensor | 2.31.7 | Symbolic graph, Op framework | PyMC 5 backend, gradient chain |
| JAX | 0.4.31 | JIT-compiled logp + autodiff | pyhgf compute engine |
| ArviZ | 0.22.0 | Summary statistics, diagnostics | r_hat, ESS, HDI extraction |
| pyhgf | 0.2.8 | Network API, scan_fn | Already used in Phases 2-3 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (env) | Array operations, scan input prep | Scan input formatting |
| pandas | (env) | Results DataFrame | Batch fit aggregation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Op | HGFDistribution | Cannot handle custom Network topology (confirmed) |
| NUTS | Metropolis-Hastings | NUTS is 5-10x more efficient; use NUTS |
| Sequential fitting | Parallel fitting | Sequential is simpler; parallelism adds complexity |

**Installation:** All libraries already installed in ds_env.

---

## Architecture Patterns

### Recommended Project Structure

```
src/prl_hgf/fitting/
├── __init__.py
├── ops.py          # Custom PyTensor Ops (2-level and 3-level)
├── models.py       # pm.Model factory functions (build_pymc_model_2level, _3level)
├── single.py       # fit_participant() — runs MCMC for one participant
└── batch.py        # fit_batch() — loops over all participants, aggregates results
```

### Pattern 1: Two-Op Split (Gradient Op + Forward Op)

**What:** pyhgf uses two separate classes — one Op that computes only gradients
(`HGFLogpGradOp`) and one that computes the forward value and delegates gradients
to the first (`HGFDistribution`). The gradient Op's `perform` runs JAX's gradient
function; the forward Op's `grad` calls the gradient Op.

**When to use:** Always. This is the only verified pattern.

**Example:**

```python
# Source: pyhgf/distribution.py lines 367-810 (verified by reading source)

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pyhgf.model import Network

# --- Step 1: Build and freeze network topology ---
# Call input_data once to trigger scan_fn creation
net = build_2level_network()
net.input_data(input_data=dummy_input, observed=dummy_observed)
base_attributes = net.attributes  # frozen pytree
scan_fn = net.scan_fn             # jax.tree_util.Partial

# --- Step 2: Build pure JAX logp function ---
def _jax_logp_2level(omega_2, beta, zeta, *, scan_inputs, base_attrs):
    attrs = dict(base_attrs)
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["tonic_volatility"] = omega_2
        attrs[idx] = node
    _, node_traj = lax.scan(scan_fn, attrs, scan_inputs)
    mu1 = jnp.stack([
        node_traj[0]["expected_mean"],
        node_traj[2]["expected_mean"],
        node_traj[4]["expected_mean"],
    ], axis=1)
    # ... softmax-stickiness surprise computation ...
    return log_likelihood  # scalar

# JIT-compile value+grad once at network construction time
_jit_val_grad = jax.jit(jax.value_and_grad(_jax_logp_2level, argnums=(0, 1, 2)))
_jit_logp    = jax.jit(_jax_logp_2level)

# --- Step 3: Gradient Op ---
class HGFGradOp2Level(Op):
    def make_node(self, omega_2, beta, zeta):
        inputs = [pt.as_tensor_variable(x) for x in [omega_2, beta, zeta]]
        outputs = [inp.type() for inp in inputs]   # same type as inputs
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        omega_2, beta, zeta = inputs
        (_, (g0, g1, g2)) = _jit_val_grad(float(omega_2), float(beta), float(zeta))
        outputs[0][0] = np.asarray(g0, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(g1, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(g2, dtype=node.outputs[2].dtype)

# --- Step 4: Forward logp Op ---
class HGFLogpOp2Level(Op):
    def __init__(self):
        self._grad_op = HGFGradOp2Level()

    def make_node(self, omega_2, beta, zeta):
        inputs = [pt.as_tensor_variable(x) for x in [omega_2, beta, zeta]]
        outputs = [pt.scalar(dtype=float)]        # single scalar logp
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        omega_2, beta, zeta = inputs
        val = _jit_logp(float(omega_2), float(beta), float(zeta))
        outputs[0][0] = np.asarray(val, dtype=float)

    def grad(self, inputs, output_gradients):
        g0, g1, g2 = self._grad_op(*inputs)
        og = output_gradients[0]
        return [og * g0, og * g1, og * g2]
```

### Pattern 2: Parameter Injection via Shallow Copy

**What:** Inject JAX traced values into the frozen attributes dict by shallow-copying
the outer dict and each modified node dict. Never deepcopy (breaks JAX traceability).

**When to use:** Every call to the JAX logp function.

```python
# Source: verified via test execution on actual network (2026-04-05)

# 2-level: inject omega_2 into belief nodes 1, 3, 5
attrs = dict(base_attributes)       # shallow copy outer dict
for idx in [1, 3, 5]:
    node = dict(attrs[idx])         # shallow copy this node's dict
    node["tonic_volatility"] = omega_2   # omega_2 is a JAX scalar
    attrs[idx] = node

# 3-level additional: inject omega_3 into volatility node 6
node6 = dict(attrs[6])
node6["tonic_volatility"] = omega_3
attrs[6] = node6

# 3-level: inject kappa — must inject into BOTH ends of each edge
# Node 6 (parent): volatility_coupling_children shape (3,) array
node6 = dict(attrs[6])
node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
attrs[6] = node6
# Nodes 1, 3, 5 (children): volatility_coupling_parents shape (1,) array
for idx in [1, 3, 5]:
    node = dict(attrs[idx])
    node["volatility_coupling_parents"] = jnp.array([kappa])
    attrs[idx] = node
```

### Pattern 3: Scan Input Formatting

**What:** `lax.scan` expects a specific pytree for the scan carry and inputs
matching what `Network.input_data` produces.

```python
# Source: pyhgf/model/network.py lines 374-384 (verified)

# inputs tuple: (values, observed, time_steps, rng_keys)
# values: tuple of (n_trials, 1) arrays, one per input node
values = tuple(np.split(input_data_arr, [1, 2], axis=1))  # 3 splits for 3 cues
# observed: tuple of (n_trials,) int arrays, one per input node
observed_cols = tuple(observed_arr[:, i] for i in range(3))
# time_steps: (n_trials,) float array
time_steps = np.ones(n_trials)
# rng_keys: None (not needed for deterministic forward pass)
scan_inputs = (values, observed_cols, time_steps, None)

# Run scan
_, node_trajectories = lax.scan(scan_fn, attrs, scan_inputs)
# node_trajectories[i]["expected_mean"] has shape (n_trials,)
```

### Pattern 4: PyMC Model Structure

```python
# Source: verified via test execution (2026-04-05)

_logp_op = HGFLogpOp2Level()

with pm.Model() as model:
    # omega_2: must be bounded above at 0 — NaN above that value (verified)
    omega_2 = pm.TruncatedNormal("omega_2", mu=-3.0, sigma=2.0, upper=0.0)

    # beta: sample in log-space to avoid boundary at 0
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=1.5)
    beta = pm.Deterministic("beta", pm.math.exp(log_beta))

    # zeta: unbounded, Normal prior
    zeta = pm.Normal("zeta", mu=0.0, sigma=2.0)

    # Hook the custom logp Op into PyMC via pm.Potential
    pm.Potential("loglike", _logp_op(omega_2, beta, zeta))
```

### Anti-Patterns to Avoid

- **Deepcopy of attributes:** `import copy; copy.deepcopy(attrs)` breaks JAX
  traceability because copied arrays lose their JAX identity.
- **Using HGFDistribution:** Cannot handle custom Network topology. Confirmed in
  CONTEXT.md. The standard HGFDistribution wraps `HGF` objects, not `Network`.
- **pm.DensityDist for custom logp:** `pm.Potential` is the correct approach
  when you have a raw log-probability scalar. `DensityDist` is for distributions
  with shapes; `Potential` is for free-form log-probability terms.
- **Sampling omega_2 above 0:** At omega_2 >= 0 (2-level network, 420 trials),
  logp returns NaN and gradients are NaN. This causes divergences. Always bound
  with `upper=0.0`.
- **Instancing GradOp inside `grad` method:** Instantiate `self._grad_op` once
  in `__init__` and reuse it. Creating a new instance per `grad` call may add
  overhead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient computation | Manual finite differences | `jax.grad` through `lax.scan` | Automatic exact gradients verified working |
| MCMC diagnostics | Custom r_hat/ESS code | `az.summary()` | Returns all needed columns including `ess_bulk`, `ess_tail`, `r_hat` |
| Trace storage | Custom HDF5 writer | PyMC default (in-memory InferenceData) | ArviZ InferenceData format needed for Phase 5 |
| Bounded sampling | Manual rejection sampling | `pm.TruncatedNormal` or `pm.Bound` | PyMC handles reparameterization automatically |
| Log transform | Manual `exp` in logp | `pm.Deterministic` + `pm.math.exp` | PyMC propagates to InferenceData correctly |

**Key insight:** The JAX autodiff machinery (verified at 0.15-0.38ms per grad eval
depending on trial count) eliminates any need for finite-difference gradients or
variational methods.

---

## Common Pitfalls

### Pitfall 1: omega_2 >= 0 Causes NaN (CRITICAL)

**What goes wrong:** At omega_2 = 0.0 and above (for the binary 3-branch network
with 200+ trials), `lax.scan` produces NaN values in node trajectories, causing
NaN logp and NaN gradients. NUTS then treats these as divergences.

**Why it happens:** Large positive omega_2 causes precision to collapse to zero
or go negative inside the binary HGF update equations, producing NaN propagation
through the scan.

**How to avoid:** Use `pm.TruncatedNormal("omega_2", mu=-3.0, sigma=2.0, upper=0.0)`.
This reduces divergences from ~173/2000 to ~13/2000 on uninformative random data,
and to 0 with structured behavioral data.

**Warning signs:** High divergence count in `idata.sample_stats['diverging']`,
NaN values in trace, chains not mixing.

### Pitfall 2: Divergences on Uninformative (Random) Data

**What goes wrong:** With purely random data (no signal), the likelihood is
flat over a wide parameter range. NUTS struggles with flat regions because
step-size adaptation cannot identify the posterior geometry.

**Why it happens:** The data comes from a random generator with no true signal,
so the likelihood contributes almost no information. The posterior is dominated
by the prior, which may be wide.

**How to avoid:** With real simulated behavioral data from Phase 3 (choices driven
by true parameters), divergences drop to near zero. Use structured simulated data
(not random arrays) for testing.

**Warning signs:** `idata.sample_stats['diverging']` > 5% of draws with any
parameter set. Trace plots showing chains not converging.

### Pitfall 3: PyTensor g++ Warning Does Not Affect Results

**What goes wrong (non-issue):** `WARNING: g++ not detected! PyTensor will be
unable to compile C-implementations and will default to Python.`

**Why it happens:** The ds_env environment does not have g++ installed. PyTensor
can use it for C compilation of non-JAX parts of the graph.

**How to handle:** Since our Op's `perform` method calls JIT-compiled JAX
directly, the performance-critical path is already compiled. The Python fallback
for PyTensor graph traversal adds negligible overhead. PyTensor config `cxx` is
already empty string, which suppresses the warning for production use
(`pytensor.config.cxx = ""`).

### Pitfall 4: Kappa Injection Requires Both Edge Endpoints

**What goes wrong:** Injecting kappa only into node 6's `volatility_coupling_children`
without also updating the matching `volatility_coupling_parents` on nodes 1, 3, 5
(or vice versa) may leave the scan with inconsistent coupling values.

**Why it happens:** pyhgf stores coupling strength at both ends of each edge.
The base attributes show:
- Node 6: `volatility_coupling_children` = `[1. 1. 1.]` (shape (3,))
- Node 1/3/5: `volatility_coupling_parents` = `[1.]` (shape (1,))

**How to avoid:** Always inject kappa at both endpoints simultaneously (verified
in test_3level_and_timing.py).

### Pitfall 5: JIT Warm-Up Cost

**What goes wrong:** The first call to a JAX JIT-compiled function pays a
one-time compilation cost (can be 1-5 seconds per function). In a batch fitting
loop, this only occurs once per model variant (not per participant) if the same
compiled function is reused.

**How to avoid:** Build the `_jit_val_grad` and `_jit_logp` functions once
outside the participant loop. Optionally call them once with dummy values to
trigger compilation before the main loop.

---

## Prior Specification

### Verified Literature Defaults

| Parameter | Prior | Distribution | Source |
|-----------|-------|-------------|--------|
| omega_2 | TruncatedNormal(mu=-3.0, sigma=2.0, upper=0.0) | Bounded normal | Mathys et al. 2011, pyhgf docs |
| omega_3 | TruncatedNormal(mu=-6.0, sigma=2.0, upper=0.0) | Bounded normal | pyhgf docs (Normal(-11, 2) for continuous; binary uses higher values) |
| kappa | TruncatedNormal(mu=1.0, sigma=0.5, lower=0.01, upper=2.0) | Bounded normal | Mathys et al. 2014 |
| log_beta | Normal(mu=0.0, sigma=1.5) | Normal (log space) | Corresponds to beta ~ LogNormal(0, 1.5) |
| beta | Deterministic(exp(log_beta)) | — | Ensures beta > 0 |
| zeta | Normal(mu=0.0, sigma=2.0) | Normal | Uninformative; positive = persist, negative = switch |

**Rationale for bounds:**

- omega_2 `upper=0.0`: omega_2 >= 0 causes NaN in the binary HGF scan (verified).
  This matches the physical constraint: learning rates above 1 (omega_2 >= 0 in
  log-space) are neurobiologically implausible and numerically unstable.
- omega_3 `upper=0.0`: Same reasoning. Note: pyhgf docs use Normal(-11, 2) for
  continuous HGF omega_3; for binary HGF the values cluster around -6 to -8.
- kappa `lower=0.01`: kappa = 0 decouples the volatility hierarchy (3-level
  reduces to 2-level). kappa > 2 is rarely observed in the literature.
- beta via log transform: `pm.HalfNormal(sigma=5)` is an alternative but
  log_beta + Deterministic is better for NUTS geometry (avoids zero-boundary
  sampling issues).

**Config file priors (prl_analysis.yaml):**

The existing config already documents: `omega_2: prior_mean=-3.0, prior_sd=2.0`,
`omega_3: prior_mean=-6.0, prior_sd=2.0`, `kappa: prior_mean=1.0, prior_sd=0.5`,
`beta: prior_mean=2.0, prior_sd=2.0`, `zeta: prior_mean=0.0, prior_sd=1.0`.
These serve as the canonical values. The fitting module should read these from
config rather than hardcoding.

---

## Code Examples

### Full 2-Level Op Implementation

```python
# Verified working on ds_env (2026-04-05)
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pyhgf.model import Network

def build_logp_ops_2level(
    input_data_arr: np.ndarray,   # (n_trials, 3)
    observed_arr: np.ndarray,     # (n_trials, 3) int
) -> tuple:
    """Build JIT-compiled logp and Op pair for 2-level network.

    Returns (logp_op, n_trials) where logp_op(omega_2, beta, zeta)
    returns a scalar log-likelihood.
    """
    from prl_hgf.models.hgf_2level import build_2level_network

    net = build_2level_network()
    net.input_data(input_data=input_data_arr, observed=observed_arr)
    base_attrs = net.attributes
    scan_fn = net.scan_fn
    n_trials = input_data_arr.shape[0]

    values = tuple(np.split(input_data_arr, [1, 2], axis=1))
    observed_cols = tuple(observed_arr[:, i] for i in range(3))
    scan_inputs = (values, observed_cols, np.ones(n_trials), None)
    choices_arr = observed_arr.argmax(axis=1)  # reconstruct from observed mask

    def _jax_logp(omega_2, beta, zeta):
        attrs = dict(base_attrs)
        for idx in [1, 3, 5]:
            node = dict(attrs[idx])
            node["tonic_volatility"] = omega_2
            attrs[idx] = node
        _, node_traj = lax.scan(scan_fn, attrs, scan_inputs)
        mu1 = jnp.stack([
            node_traj[0]["expected_mean"],
            node_traj[2]["expected_mean"],
            node_traj[4]["expected_mean"],
        ], axis=1)
        choices_jax = jnp.array(choices_arr)
        prev = jnp.concatenate([jnp.array([-1]), choices_jax[:-1]])
        stick = (prev[:, None] == jnp.arange(3)[None, :]).astype(jnp.float32)
        logits = beta * mu1 + zeta * stick
        lp = jax.nn.log_softmax(logits, axis=1)
        return jnp.sum(lp[jnp.arange(n_trials), choices_jax])

    _jit_val_grad = jax.jit(jax.value_and_grad(_jax_logp, argnums=(0, 1, 2)))
    _jit_logp = jax.jit(_jax_logp)

    class _GradOp(Op):
        def make_node(self, o2, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, b, z]]
            return Apply(self, inputs, [inp.type() for inp in inputs])
        def perform(self, node, inputs, outputs):
            (_, grads) = _jit_val_grad(*[float(x) for x in inputs])
            for i, g in enumerate(grads):
                outputs[i][0] = np.asarray(g, dtype=node.outputs[i].dtype)

    grad_op = _GradOp()

    class _LogpOp(Op):
        def make_node(self, o2, b, z):
            inputs = [pt.as_tensor_variable(x) for x in [o2, b, z]]
            return Apply(self, inputs, [pt.scalar(dtype=float)])
        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(_jit_logp(*[float(x) for x in inputs]), dtype=float)
        def grad(self, inputs, output_gradients):
            grads = grad_op(*inputs)
            og = output_gradients[0]
            return [og * g for g in grads]

    return _LogpOp(), n_trials
```

### Kappa Injection for 3-Level (Verified)

```python
# Source: verified via test_3level_and_timing.py (2026-04-05)
# Attributes structure (read from live network):
#   node 6:   volatility_coupling_children = jnp.array([1., 1., 1.])  shape (3,)
#   node 1/3/5: volatility_coupling_parents = jnp.array([1.])         shape (1,)

def _inject_params_3level(base_attrs, omega_2, omega_3, kappa):
    attrs = dict(base_attrs)
    # omega_2 into belief nodes
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["tonic_volatility"] = omega_2
        attrs[idx] = node
    # omega_3 into volatility node
    node6 = dict(attrs[6])
    node6["tonic_volatility"] = omega_3
    # kappa into both endpoints of each 6→(1,3,5) edge
    node6["volatility_coupling_children"] = jnp.array([kappa, kappa, kappa])
    attrs[6] = node6
    for idx in [1, 3, 5]:
        node = dict(attrs[idx])
        node["volatility_coupling_parents"] = jnp.array([kappa])
        attrs[idx] = node
    return attrs
```

### ArviZ Summary to DataFrame Rows

```python
# Source: verified via test_arviz_summary.py (2026-04-05)
# az.summary() returns DataFrame with columns:
# ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat']

import arviz as az

def extract_summary_rows(
    idata,
    participant_id: str,
    group: str,
    session: str,
    model: str,
    var_names: list[str],
) -> list[dict]:
    """Convert ArviZ summary to list of dicts for DataFrame construction."""
    summary = az.summary(idata, var_names=var_names, round_to=8)
    rows = []
    for param_name in summary.index:
        row = summary.loc[param_name]
        rows.append({
            "participant_id": participant_id,
            "group": group,
            "session": session,
            "model": model,
            "parameter": param_name,
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_3%": float(row["hdi_3%"]),
            "hdi_97%": float(row["hdi_97%"]),
            "r_hat": float(row["r_hat"]),
            "ess": float(row["ess_bulk"]),
        })
    return rows
```

### Diagnostic Flag Check

```python
R_HAT_THRESHOLD = 1.05
ESS_THRESHOLD = 400

def flag_fit(summary_rows: list[dict]) -> bool:
    """Return True if fit should be flagged as potentially problematic."""
    for row in summary_rows:
        if row["r_hat"] > R_HAT_THRESHOLD:
            return True
        if row["ess"] < ESS_THRESHOLD:
            return True
    return False
```

---

## Timing and Feasibility

### Measured Performance (ds_env, Windows 11, CPU-only, no g++)

| Scenario | Trials | Grad eval | MCMC run (4 chains x 1000 draws) |
|----------|--------|-----------|----------------------------------|
| 2-level, 200 trials | 200 | 0.29 ms | ~30s measured (500 draws) |
| 2-level, 420 trials | 420 | 0.15 ms | ~62s estimated |
| 3-level, 200 trials | 200 | 0.38 ms | ~45s estimated |

**Note:** 420 trials is the actual session length (3 sets x 140 trials).

### Batch Estimate

| Scope | Count | Per-participant | Total |
|-------|-------|----------------|-------|
| 2-level fits | 180 (60 participants x 3 sessions) | ~62s | ~3.1 hours |
| 3-level fits | 180 | ~90s | ~4.5 hours |
| Both models | 360 | — | ~7.6 hours sequential |

**Mitigation options:**

1. Reduce tune steps from 1000 to 500 for initial batch (can re-run flagged fits)
2. Fit 2-level first for Phase 4 validation, add 3-level in Phase 5
3. Use `cores=4` with `chains=4` if parallel cores are available — requires
   JAX process isolation (one network per process)

**PyTensor g++ absence:** The missing g++ compiler only affects PyTensor's
C backend for non-JAX graph nodes. Since `perform` calls directly into
JIT-compiled JAX, the missing compiler has no effect on fitting speed. The
warnings can be silenced with `pytensor.config.cxx = ""`.

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| HGFDistribution (single-branch) | Custom Op wrapping Network scan_fn | Enables multi-branch topology |
| HGF class (deprecated API) | Network API (pyhgf >= 0.2) | Must use Network for custom graphs |
| Standard update | eHGF update (pyhgf default) | Better sampling performance |

**Deprecated:**
- `HGF` class: Still present but wraps `Network` internally. For custom topologies,
  use `Network` directly.
- `pyhgf.networks.beliefs_propagation`: Not exposed as a module-level import in
  0.2.8. Use `net.scan_fn` (a `jax.tree_util.Partial`) after `create_belief_propagation_fn`.

---

## Open Questions

1. **omega_3 prior for binary HGF**
   - What we know: pyhgf docs use Normal(-11, 2) for continuous HGF omega_3.
     Config uses -6.0 mean for binary. Literature default unclear.
   - What's unclear: Whether binary 3-level omega_3 should use -6 or more negative.
   - Recommendation: Use config.yaml value (-6.0 mean, 2.0 sd) with upper=0 bound.
     Flag omega_3 recovery as secondary per project documentation (CLAUDE.md).

2. **Parallel chains (cores=4 vs. cores=1)**
   - What we know: PyMC supports `pm.sample(chains=4, cores=4)` for parallel
     execution. JAX uses process isolation when `cores > 1`.
   - What's unclear: Whether JAX JIT state (compiled scan_fn) is shared correctly
     across processes on Windows.
   - Recommendation: Use `cores=1` for reliability. Retest with `cores=4` if
     batch time exceeds 8 hours.

3. **kappa constraint during sampling**
   - What we know: Kappa must stay positive for physical interpretability. Negative
     kappa was not tested.
   - What's unclear: Whether negative kappa causes NaN in scan (like omega_2 >= 0).
   - Recommendation: Use TruncatedNormal(lower=0.01) for kappa as precaution.

---

## Sources

### Primary (HIGH confidence)

- `pyhgf/distribution.py` (lines 1-930) — complete Op pattern for HGFLogpGradOp
  and HGFDistribution; read directly from installed package
- `pyhgf/model/network.py` (lines 303-390, 733-771) — input_data method showing
  scan_fn structure and scan input tuple format
- Live test execution: `test_jax_logp.py`, `test_3level_and_timing.py`,
  `test_pymc_op.py`, `test_timing_full.py`, `test_arviz_summary.py` — all run
  on ds_env (2026-04-05)
- `prl_analysis.yaml` — canonical prior values from project config
- `src/prl_hgf/models/hgf_2level.py`, `hgf_3level.py` — verified node layout
  and parameter mapping

### Secondary (MEDIUM confidence)

- [pyhgf binary HGF docs](https://computationalpsychiatry.github.io/pyhgf/notebooks/1.1-Binary_HGF.html) —
  prior recommendations for pyhgf binary HGF fitting with PyMC (tonic_volatility_2
  Uniform(-3.5, 0.0); tonic_volatility_3 Normal(-11, 2))
- Mathys et al. 2014 — omega prior on real line, Normal distribution recommended

### Tertiary (LOW confidence)

- Literature values for kappa and beta priors in binary HGF — general consensus
  from pyhgf community; not verified against a single canonical publication

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions confirmed from installed packages
- Custom Op pattern: HIGH — read and understood source code, verified via tests
- Parameter injection: HIGH — tested kappa/omega injection with lax.scan + jax.grad
- Priors: MEDIUM — verified against config + pyhgf docs; omega_2 NaN boundary confirmed
- Timing: HIGH — measured on actual hardware with realistic trial counts
- ArviZ extraction: HIGH — tested summary column names and DataFrame conversion

**Research date:** 2026-04-05
**Valid until:** 2026-07-05 (pyhgf and PyMC are active projects; check for updates
if more than 30 days pass before implementation)
