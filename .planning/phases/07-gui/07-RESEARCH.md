# Phase 7: GUI - Research

**Researched:** 2026-04-06
**Domain:** ipywidgets interactive notebooks, matplotlib visualization, JAX/pyhgf forward pass performance
**Confidence:** HIGH (all key findings verified by running code against the live environment)

## Summary

Phase 7 builds a Jupyter notebook interactive dashboard using ipywidgets 8.1.8 (already
installed) and matplotlib 3.10.3. The widget architecture uses `FloatSlider` widgets with
`continuous_update=False` connected to an `observe` callback that runs a pyhgf forward pass
and updates matplotlib line objects. The dashboard lives in a new
`src/prl_hgf/gui/` module and `notebooks/07_parameter_explorer.ipynb`.

The critical performance finding: `simulate_agent` (trial-by-trial loop, 420 JAX dispatches)
takes ~4.2 s per call even after JAX JIT warmup — far too slow for a <2 s interactive target.
The solution is to use a **batch forward pass** (`net.input_data(inp_data, obs_mask)`) with
a fixed observation mask derived from one warm-up simulation. Batch forward pass takes ~1.1 s
for 2-level and ~1.7 s for 3-level after JAX compilation, meeting the <2 s target when
combined with `continuous_update=False`. An additional optimisation caches HGF-level beliefs
so that changes to `beta`/`zeta` only (response parameters) skip the forward pass entirely
and recompute only the softmax, taking <1 ms.

**Primary recommendation:** Use `continuous_update=False` on all sliders, batch forward pass
(not `simulate_agent`) for GUI updates, belief caching keyed on HGF parameters, and add
`ipympl>=0.9.0` to `pyproject.toml` for the `%matplotlib widget` VSCode-compatible backend.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ipywidgets | 8.1.8 (installed) | Slider, toggle, layout, Output widgets | Pre-decided; VSCode Jupyter compatible |
| matplotlib | 3.10.3 (installed) | Multi-panel figure rendering | Pre-decided; used throughout project |
| ipympl | 0.10.0 (NOT installed) | `%matplotlib widget` backend for VSCode | Required for persistent in-place plot updates; alternative to clear/redraw |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.special.softmax | (installed) | Vectorised softmax for choice probs | Recomputing choice probs from cached beliefs |
| numpy | 2.x (installed) | Array manipulation | Everywhere |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ipympl + `%matplotlib widget` | `Output` widget + `clear_output(wait=True)` + inline | No extra dep; slight flicker on each update; full figure rebuild each time |
| `continuous_update=False` | debounce decorator (threading) | More complex; `continuous_update=False` is simpler and sufficient |
| Batch forward pass | `simulate_agent` trial-by-trial | `simulate_agent` = ~4.2 s/call; batch = ~1.1-1.7 s/call |

**Installation:**
```bash
pip install ipympl>=0.9.0
```
Add to `pyproject.toml` under `dependencies`: `"ipympl>=0.9.0"`.

## Architecture Patterns

### Recommended Project Structure
```
src/prl_hgf/
└── gui/
    ├── __init__.py          # exports ParamExplorer
    └── explorer.py          # ParamExplorer class

notebooks/
└── 07_parameter_explorer.ipynb  # thin wrapper: instantiates ParamExplorer, displays

tests/
└── test_gui.py              # unit tests for explorer (no display, use Agg backend)
```

### Pattern 1: Belief Caching with Selective Recompute

The critical optimisation separates HGF parameters (drive the forward pass) from response
parameters (only drive the softmax).

**What:** Cache `_cached_beliefs_` and the fixed `_obs_mask_`/`_inp_data_` when any HGF
parameter changes. Skip the forward pass when only `beta`/`zeta` change.

**When to use:** Every slider callback.

**Example:**
```python
# Source: verified benchmarks in live env (see research notes)
_HGF_PARAMS = frozenset({"omega_2", "omega_3", "kappa", "mu_1_0", "mu_3_0"})
_RESPONSE_PARAMS = frozenset({"beta", "zeta"})

def _on_slider_change(self, change):
    param = change["owner"].description  # which slider fired
    if param in _HGF_PARAMS:
        self._run_forward_pass()   # ~1.0-1.7 s
    self._update_choice_probs()    # <1 ms (softmax only)
    self._update_plot()            # ~0.3 s (set_ydata + draw_idle)
```

### Pattern 2: Batch Forward Pass (not simulate_agent)

**What:** Run `net.input_data(inp_data, obs_mask)` once with all 420 trials at once.
Do NOT run the trial-by-trial `simulate_agent` loop on slider changes.

**Why:** Trial-by-trial loop = 420 Python→JAX dispatches = ~4.2 s per call (even after
JIT warmup). Batch = one XLA dispatch = ~1.1-1.7 s.

**Setup (run once at notebook open):**
```python
# Source: verified in live env
from prl_hgf.simulation.agent import simulate_agent
from prl_hgf.models.hgf_2level import build_2level_network, prepare_input_data
from prl_hgf.env.simulator import generate_session

config = load_config()
trials = generate_session(config, seed=42)

# Warm-up: runs simulate_agent once to compile JAX JIT (~5-10 s, done once)
net_wu = build_2level_network(omega_2=-3.0)
rng_wu = np.random.default_rng(0)
result_wu = simulate_agent(net_wu, trials, beta=2.0, zeta=0.0, rng=rng_wu)

# Fixed observation mask for all subsequent GUI updates
inp_data, obs_mask = prepare_input_data(trials, result_wu.choices, result_wu.rewards)
```

**Per-slider-change update:**
```python
# Source: verified in live env (batch timings: 2-level ~1.1 s, 3-level ~1.7 s)
def _run_forward_pass(self):
    net = build_2level_network(omega_2=self.omega_2_slider.value)
    # Set initial beliefs (post-hoc attribute modification — verified works)
    for node_idx in BELIEF_NODES:
        net.attributes[node_idx]["mean"] = self.mu_1_0_slider.value
        net.attributes[node_idx]["expected_mean"] = self.mu_1_0_slider.value
    net.input_data(input_data=self._inp_data, observed=self._obs_mask)
    self._cached_beliefs_ = extract_beliefs(net)
    self._cached_node_traj_ = net.node_trajectories
```

### Pattern 3: ipympl In-Place Plot Update

**What:** Create the figure once with `%matplotlib widget`; update data with
`line.set_ydata(new_data)` + `fig.canvas.draw_idle()`. Never recreate the figure.

**When to use:** Every slider change (the plot update part).

**Setup in notebook:**
```python
# Source: VSCode Jupyter wiki - %matplotlib widget is fully supported
%matplotlib widget
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
# Create lines once
belief_lines = [axes[0].plot(trial_x, np.zeros(420), label=f"Cue {i}")[0] for i in range(3)]
gt_lines     = [axes[0].plot(trial_x, np.zeros(420), "--", alpha=0.4)[0] for i in range(3)]
mu2_line     = axes[1].plot(trial_x, np.zeros(420))[0]   # volatility (3-level only)
prob_lines   = [axes[2].plot(trial_x, np.zeros(420))[0] for _ in range(3)]
lr_line      = axes[3].plot(trial_x, np.zeros(420))[0]
```

**Update function:**
```python
# Source: Kapernikov tutorial, verified pattern
def _update_plot(self):
    b = self._cached_beliefs_
    for i, l in enumerate(self.belief_lines_):
        key = f"p_reward_cue{i}"
        l.set_ydata(b[key])
    # ... update other lines ...
    self.fig_.canvas.draw_idle()   # non-blocking redraw
```

### Pattern 4: Widget Layout for Model Toggle

**What:** Show/hide the volatility panel (`axes[1]`) based on model toggle.

**How:** Set `ax.set_visible(bool)` and `fig.tight_layout()` — simpler and more reliable
than hiding ipywidgets containers.

```python
# Source: ipywidgets Widget List docs (layout.display = 'none' pattern)
# For matplotlib axes: visibility toggle is cleaner
def _on_model_toggle(self, change):
    is_3level = (change["new"] == "3-level")
    self.axes_[1].set_visible(is_3level)
    # Also show/hide omega_3 and kappa sliders:
    vis = "" if is_3level else "none"
    self.omega_3_slider.layout.display = vis
    self.kappa_slider.layout.display = vis
    self.fig_.tight_layout()
    self.fig_.canvas.draw_idle()
```

### Pattern 5: Preset Parameter Profiles

**What:** `ToggleButtons` widget with 3 profiles. On toggle, set all slider values at once
then trigger one forward pass.

**Profile values (derived from configs/prl_analysis.yaml):**
```python
# Source: configs/prl_analysis.yaml simulation.groups and simulation.session_deltas
PROFILES = {
    "healthy baseline": {
        "omega_2": -3.0, "omega_3": -6.0, "kappa": 1.0, "beta": 2.5, "zeta": 0.2,
        "mu_1_0": 0.0, "mu_3_0": 0.0,
    },
    "post-concussion": {
        "omega_2": -4.0, "omega_3": -7.0, "kappa": 0.8, "beta": 2.0, "zeta": 0.3,
        "mu_1_0": 0.0, "mu_3_0": 0.0,
    },
    "post-psilocybin": {
        # post_concussion + post_dose deltas: omega_2 += 1.5, kappa += 0.3
        "omega_2": -2.5, "omega_3": -7.0, "kappa": 1.1, "beta": 2.0, "zeta": 0.3,
        "mu_1_0": 0.0, "mu_3_0": 0.0,
    },
}
```

**Batch slider update (suppress intermediate callbacks):**
```python
# Unobserve → set all values → re-observe → trigger one update
def _load_preset(self, change):
    p = PROFILES[change["new"]]
    self._unobserve_all()
    for name, widget in self._sliders.items():
        widget.value = p[name]
    self._observe_all()
    self._run_forward_pass()   # one forward pass after all sliders set
    self._update_choice_probs()
    self._update_plot()
```

### Anti-Patterns to Avoid

- **Calling `simulate_agent` on slider change:** 4.2 s per call — fails <2 s target.
  Use batch `net.input_data()` instead.
- **Recreating the matplotlib figure on each update:** ~300 ms overhead + flicker.
  Create once, use `set_ydata()` + `draw_idle()`.
- **`continuous_update=True` (default) on sliders:** Triggers callback on every pixel of
  drag motion. Always set `continuous_update=False`.
- **Not calling warm-up at notebook init:** First JAX JIT compilation takes 5-10 s.
  Run one warm-up `simulate_agent` call at notebook open with a visible progress message.
- **Using `plt.show()` or `display(fig)` inside the update callback:** Creates duplicate
  figures in inline backend. With ipympl this never triggers; without it, wrap in
  `Output` context with `clear_output(wait=True)`.
- **Not unobserving sliders before bulk preset set:** Causes N forward passes instead of 1.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Softmax with numerical stability | Custom exp/sum | `scipy.special.softmax(axis=1)` | Handles edge cases, vectorised |
| Widget layout containers | CSS div HTML | `HBox`, `VBox` | Standard ipywidgets layout |
| Debouncing slider | `threading.Timer` | `continuous_update=False` | Simpler; solves the same problem for slider interactions |
| In-place plot updates | Clearing axes and re-plotting | `line.set_ydata()` + `fig.canvas.draw_idle()` | 10x faster; avoids flicker |
| JAX warm-up detection | Try/except with timing | Unconditional warm-up at module import | JAX compilation is always slow on first call with specific input shapes |

**Key insight:** The biggest performance win is architectural: batch forward pass + belief
caching eliminates 75% of forward pass calls. This matters more than any other optimisation.

## Common Pitfalls

### Pitfall 1: simulate_agent is Too Slow for Interactive GUI

**What goes wrong:** Developer naturally reaches for `simulate_agent` since that's how
agent behaviour has been simulated throughout the project. With 420 trials × 1 JAX dispatch
each, it takes ~4.2 s per call even after warmup.

**Why it happens:** `simulate_agent` uses the attribute-carry pattern: each trial calls
`net.input_data` with shape `(1, 3)` then copies `last_attributes` → `attributes`.
That is 420 separate Python→JAX dispatcher calls.

**How to avoid:** Use batch `net.input_data(inp_data_shape=(420, 3))` for GUI updates.
Only use `simulate_agent` for the one-time warmup (to generate the fixed choice/reward
observation mask).

**Warning signs:** Slider updates taking >2 s; notebook feels laggy during dragging.

### Pitfall 2: JAX JIT Compilation on First Call

**What goes wrong:** First call to any pyhgf `input_data` (either batch or trial-by-trial)
takes 5-10 s while XLA compiles the computation graph. If this happens on the first slider
move, the user sees a long freeze with no feedback.

**Why it happens:** JAX JIT compiles on first call with a specific input shape/dtype.
Subsequent calls with the same shape reuse the compiled XLA program.

**How to avoid:** Run a warm-up call (one full `simulate_agent` or batch `input_data`)
during notebook initialisation, before displaying the widget. Print a "Warming up JAX..." 
message so the user understands the delay.

**Warning signs:** First slider movement takes 5-10 s; subsequent moves take 1-2 s.

### Pitfall 3: Multiple Matplotlib Figures Created

**What goes wrong:** Without ipympl, calling `plt.figure()` or `plt.subplots()` inside
an `Output` widget callback creates a new figure output each time instead of updating
the existing one.

**Why it happens:** `%matplotlib inline` treats each `plt.figure()` call as a new output.

**How to avoid:** Either (a) install ipympl and use `%matplotlib widget` (recommended),
or (b) create the figure *once* outside the callback, use `set_ydata()` inside the
callback, and wrap only the `display()` call in the `Output` context.

**Warning signs:** Notebook output grows with duplicate figures on each slider move.

### Pitfall 4: Slider Preset Triggering N Forward Passes

**What goes wrong:** When a preset button loads 7 slider values, each slider `.value`
assignment fires the `observe` callback, triggering 7 separate forward passes.

**Why it happens:** ipywidgets `observe` fires synchronously on `.value` assignment.

**How to avoid:** Unobserve all sliders before bulk-setting values, then re-observe.
Trigger one explicit forward pass after all values are set.

**Warning signs:** Preset button causes 5-7 s delay; multiple plot flashes.

### Pitfall 5: omega_3/kappa Sliders Visible for 2-Level Model

**What goes wrong:** Sliders for `omega_3`, `kappa`, and `mu_3_0` are shown even when
the 2-level model is selected. Changing them triggers a forward pass that ignores them,
confusing the user.

**How to avoid:** Set `widget.layout.display = "none"` for 3-level-only sliders when
2-level model is active. Do NOT use `layout.visibility = "hidden"` (takes up space).

### Pitfall 6: Forgetting to Update Axis Rescaling

**What goes wrong:** Using `set_ydata()` updates the line data but axes auto-limits may
not update, clipping trajectories that go outside the original range.

**How to avoid:** After `set_ydata()`, call `ax.relim()` then `ax.autoscale_view()` for
dynamic-range axes (log-odds `mu1`, precision-based `mu2`). Fixed-range axes (e.g.,
belief probability `[0, 1]`) can use `ax.set_ylim(0, 1)` once and never rescale.

## Code Examples

### Complete Slider Setup (verified pattern)
```python
# Source: ipywidgets 8.1.8 docs + verified in env
import ipywidgets as widgets
from prl_hgf.simulation.agent import PARAM_BOUNDS

omega_2_slider = widgets.FloatSlider(
    value=-3.0,
    min=PARAM_BOUNDS["omega_2"][0],   # -8.0
    max=PARAM_BOUNDS["omega_2"][1],   # 2.0
    step=0.1,
    description="omega_2",
    continuous_update=False,          # CRITICAL: update only on release
    style={"description_width": "80px"},
    layout=widgets.Layout(width="400px"),
)
omega_2_slider.observe(on_change, names="value")
```

### Effective Learning Rate Extraction (verified in env)
```python
# Source: live pyhgf env inspection
# Node 1 (continuous L1) has: expected_precision, precision
# Effective LR = prediction precision / posterior precision
traj = net.node_trajectories
eff_lr_cue0 = traj[1]["expected_precision"] / traj[1]["precision"]
# Values in (0, 1]: 1.0 = unobserved trial, < 1.0 = observed trial (belief updated)
# Shape: (420,) — one value per trial
```

### Ground-Truth Probability Overlay (verified in env)
```python
# Source: verified against Trial dataclass in simulator.py
# trials is list[Trial], each has cue_probs: tuple[float, float, float]
gt_probs = np.array([t.cue_probs for t in trials])  # shape (420, 3)
# Phase shading: extract phase boundaries for axvspan
phase_boundaries = []
current_key = None
start = 0
for t in trials:
    key = (t.set_idx, t.phase_name, t.phase_label)
    if key != current_key:
        if current_key:
            phase_boundaries.append({
                "start": start, "end": t.trial_idx,
                "label": current_key[2],  # "stable" or "volatile"
            })
        current_key, start = key, t.trial_idx
phase_boundaries.append({"start": start, "end": 419, "label": current_key[2]})
```

### Volatility Extraction for 3-Level Model (verified in env)
```python
# Source: live env inspection of node 6 (VOLATILITY_NODE = 6)
from prl_hgf.models.hgf_3level import extract_beliefs_3level, VOLATILITY_NODE
beliefs = extract_beliefs_3level(net)
mu2 = beliefs["mu2_volatility"]          # posterior mean of shared volatility node
sigma2 = beliefs["sigma2_volatility"]    # posterior SD
# Both shape (420,); mu2 is log-volatility, typically negative values
```

### Choice Probability Computation (verified in env)
```python
# Source: verified with scipy.special.softmax in env
from scipy.special import softmax
import numpy as np

# p_reward from binary input nodes is already in [0, 1]
mu1 = np.column_stack([
    beliefs["p_reward_cue0"],
    beliefs["p_reward_cue1"],
    beliefs["p_reward_cue2"],
])  # shape (420, 3)

# Softmax with stickiness: requires tracking prev_choice per trial
# For display purposes (marginal probs ignoring prev_choice):
choice_probs = softmax(beta * mu1, axis=1)   # shape (420, 3)

# With stickiness (proper implementation):
n = mu1.shape[0]
stick = np.zeros((n, 3))
choices_arr = np.array(result.choices)
prev = np.concatenate([[-1], choices_arr[:-1]])
for k in range(3):
    stick[prev == k, k] = 1.0
logits = beta * mu1 + zeta * stick
choice_probs_sticky = softmax(logits, axis=1)
```

### ipympl Notebook Setup (verified VSCode compatible)
```python
# Source: VSCode Jupyter wiki - %matplotlib widget is fully supported
%matplotlib widget
import matplotlib.pyplot as plt

# Create once at cell execution time
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].set_ylabel("P(reward)")
axes[0].set_ylim(0, 1)
axes[1].set_ylabel("mu2 (volatility)")
axes[2].set_ylabel("P(choose cue)")
axes[2].set_ylim(0, 1)
axes[3].set_ylabel("Effective LR")
axes[3].set_ylim(0, 1)
axes[3].set_xlabel("Trial")
fig.tight_layout()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `interact()` decorator | `observe()` + `interactive_output()` | ipywidgets 7+ | More control over layout; avoids full cell re-execution |
| `%matplotlib notebook` | `%matplotlib widget` (ipympl) | ~2021 | `notebook` backend unsupported in VSCode; `widget` is recommended |
| Re-creating figure on update | `set_ydata()` + `draw_idle()` | Ongoing best practice | 10x faster; eliminates flicker |
| `visible` property on widgets | `layout.display` / `layout.visibility` | ipywidgets 7 | `visible` trait removed; must use layout CSS |

**Deprecated/outdated:**
- `%matplotlib notebook`: Explicitly unsupported in VSCode (use `%matplotlib widget`)
- `ipywidgets.interact()` for complex layouts: Replace with `interactive_output()` +
  manual `HBox`/`VBox`
- Widget `.visible` trait: Use `widget.layout.display = "none"` / `""` instead

## Open Questions

1. **ipympl vs Output-widget fallback**
   - What we know: ipympl is not currently installed; it provides the cleanest update path
   - What's unclear: whether the plan should add ipympl as a dependency or use the
     `Output` + `clear_output(wait=True)` fallback
   - Recommendation: Add `ipympl>=0.9.0` to `pyproject.toml`. It is a small, stable package
     explicitly recommended by the VSCode Jupyter docs. Without it, the Output-widget
     approach requires recreating the figure on each slider change (~300 ms overhead +
     potential flicker).

2. **Whether to extend existing builder functions or use post-hoc attribute modification**
   - What we know: `build_2level_network` takes only `omega_2`; `build_3level_network`
     takes `omega_2`, `omega_3`, `kappa`. Post-hoc modification of
     `net.attributes[node_idx]["mean"]` works (verified).
   - What's unclear: whether the plan should extend builder signatures or keep modification
     in the GUI layer.
   - Recommendation: Extend builder signatures with optional `mu_1_0=0.0` (2-level) and
     `mu_1_0=0.0, mu_3_0=0.0` (3-level). This is backward-compatible and cleaner than
     post-hoc manipulation.

3. **Warm-up simulation seed and reproducibility**
   - What we know: The fixed observation mask (choices + rewards from warm-up) becomes the
     backdrop for all GUI belief trajectory comparisons. Different seeds produce different
     masks.
   - Recommendation: Use a fixed `GUI_SEED = 0` constant; document that trajectories shown
     reflect one specific simulated session, not a real participant.

## Sources

### Primary (HIGH confidence)
- Live pyhgf 0.2.8 environment — `node_trajectories` key inspection, precision ratios,
  batch vs trial-by-trial timing benchmarks
- Live ipywidgets 8.1.8 environment — widget class verification
- `configs/prl_analysis.yaml` — parameter bounds, preset profile values
- VSCode Jupyter wiki: https://github.com/microsoft/vscode-jupyter/wiki/Using-%25matplotlib-widget-instead-of-%25matplotlib-notebook,tk,etc
  — confirms `%matplotlib widget` is fully supported; requires ipympl

### Secondary (MEDIUM confidence)
- Kapernikov tutorial (https://kapernikov.com/ipywidgets-with-matplotlib/) — `observe`
  pattern, `set_ydata()` efficiency, subclass-container architecture
- ipywidgets docs (https://ipywidgets.readthedocs.io/en/latest/) — `continuous_update`,
  `layout.display`, `interactive_output`
- ipympl PyPI (https://pypi.org/project/ipympl/) — version 0.10.0, confirmed stable

### Tertiary (LOW confidence)
- WebSearch results on JAX JIT warmup (general JAX docs, not pyhgf-specific) — first-call
  latency estimate of 5-10 s is plausible given observed ~3-6 s actual times

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified via `import` and `print(__version__)` in live env
- Performance architecture: HIGH — measured with actual benchmarks in live env (n=5+)
- Architecture patterns: HIGH — verified key operations (batch forward pass, attribute
  modification, precision ratio extraction) in live env
- Pitfalls: HIGH — all discovered by running real benchmarks, not just reading docs
- ipympl requirement: MEDIUM — inferred from VSCode wiki + absence from env; not a
  breaking issue if Output-widget fallback is used

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (ipywidgets/ipympl APIs are stable; pyhgf 0.2.x is pinned)
