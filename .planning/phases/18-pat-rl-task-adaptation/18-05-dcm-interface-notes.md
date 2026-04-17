# DCM-PyTorch Consumer Interface Audit — Plan 18-05

**Date:** 2026-04-17
**Purpose:** Ground the PAT-RL trajectory CSV schema in the actual dcm_pytorch
consumer contract before writing any export code.

---

## 1. Stimulus argument contract

**File:** `dcm_pytorch/src/pyro_dcm/simulators/task_simulator.py`

### Function signature (lines 34-47)

```python
def simulate_task_dcm(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: dict[str, torch.Tensor] | PiecewiseConstantInput,
    hemo_params: dict[str, float] | None = None,
    duration: float = 300.0,
    dt: float = 0.01,
    TR: float = 2.0,
    SNR: float = 5.0,
    solver: str = "dopri5",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> dict:
```

**`stimulus` parameter (lines 64-68, task_simulator.py):**

> Experimental stimulus. If dict, must have keys `'times'` (shape `(K,)`)
> and `'values'` (shape `(K, M)`) for constructing a
> `PiecewiseConstantInput`. If already a `PiecewiseConstantInput` instance,
> used directly.

**Conversion path (lines 142-147, task_simulator.py):**

```python
if isinstance(stimulus, PiecewiseConstantInput):
    input_fn = stimulus
else:
    times = stimulus["times"].to(device=device, dtype=dtype)
    values = stimulus["values"].to(device=device, dtype=dtype)
    input_fn = PiecewiseConstantInput(times, values)
```

**Key observation:** `times` is cast to `dtype=torch.float64` (line 45 shows
`dtype: torch.dtype = torch.float64` default). Both tensors go through `.to(device, dtype)`.

---

## 2. Time-axis convention

**File:** `dcm_pytorch/src/pyro_dcm/simulators/task_simulator.py`, lines 74-79

> `duration`: Simulation duration **in seconds**. Default 300.0.
> `TR`: Repetition time for BOLD downsampling **in seconds**. Default 2.0.

**File:** `dcm_pytorch/src/pyro_dcm/utils/ode_integrator.py`, lines 35-39

> `times`: Onset times, shape `(K,)`, sorted **ascending**.
> `values[i]` is active for `times[i] <= t < times[i+1]`.

**Confirmed:** `stimulus["times"]` is an **absolute time in seconds** from the
start of the session, sorted ascending. This matches PAT-RL `outcome_time_s`
which is already accumulated absolute seconds from session start (computed in
`compute_outcome_times_s` in `pat_rl_sequence.py`).

---

## 3. Bilinear B-matrix status — v0.3.0 present, not future

**Correction vs. pre-plan expectations:** The plan sketch stated "B matrix
deferred to v0.4+". This is **incorrect** based on reading the actual source.

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/neural_state.py`, lines 1-15

> The full bilinear form `dx/dt = (A + Sigma_j u_j * B_j) * x + Cu` — which
> gives this module its historical name — is supported as an opt-in path via
> the `B` / `u_mod` arguments of `NeuralStateEquation.derivatives` (added in
> v0.3.0) and the `parameterize_B` / `compute_effective_A` utilities in this
> module.

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/neural_state.py`, lines 62-90
— `parameterize_B(B_free, b_mask)` accepts `B_free: (J, N, N)`, applies a
binary mask, returns masked modulatory matrices.

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/neural_state.py`, lines 156-224
— `compute_effective_A(A, B, u_mod)` computes `A_eff = A + einsum("j,jnm->nm", u_mod, B)`.

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/coupled_system.py`, lines 20-22

> The module now supports the full Friston 2003 bilinear neural state
> equation `dx/dt = (A + sum_j u_mod[j] * B[j]) @ x + C @ u_drive`

**Input-splitting convention (coupled_system.py, lines 68-73):**

> Set `B: (J, N, N)` and `n_driving_inputs: int` to enable the bilinear
> path. `input_fn(t)` returns a `(n_driving_inputs + J,)` vector: the first
> `n_driving_inputs` columns are driving inputs (consumed by `C @ u_drive`);
> the remaining `J` columns are modulators (consumed by
> `compute_effective_A(A, B, u_mod)`).

**Summary:** Bilinear modulation is **live in v0.3.0**, not deferred. The
dcm_pytorch PROJECT.md note about v0.4+ (cited in the plan sketch) likely
referred to the the consumer study 4-node circuit and PEB-lite, not the bilinear
mechanism itself.

---

## 4. Modulator channel value conventions

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/neural_state.py`, lines 104-113

> Off-diagonal elements pass through via pure mask multiplication. No
> `-exp` transform and no `tanh` bounding: the `N(0, 1.0)` prior (D1)
> performs regularization, not the factory.

**File:** `dcm_pytorch/src/pyro_dcm/forward_models/coupled_system.py`, lines 225-233

> `n_driving_inputs` is required when B is non-empty; explicit-split policy
> prevents ambiguity between driving vs modulator columns.

**Conclusion:** dcm_pytorch imposes **no bounding or normalization** on
modulator channel values. The consumer accepts raw float64 values. For PAT-RL,
the modulator channel(s) will carry HGF belief trajectories (`mu2`, `psi2`,
`epsilon2` etc.) **as raw float64** — these should not be sigmoid-squeezed or
otherwise normalized before export. The caller (dcm_pytorch v0.4+ the consumer study
integration) decides whether to apply a transform when constructing `B_free`.

---

## 5. Final per-trial CSV schema

Derived from the above audit. The `times` column maps to
`stimulus["times"]` (absolute seconds, sorted ascending). The modulator
`values` columns map to the `stimulus["values"]` `(K, M)` tensor where each
column is one input channel.

**No schema drift from the plan's proposed table** — the plan's column set
was correct. The `outcome_time_s` column directly feeds `stimulus["times"]`;
HGF belief columns feed `stimulus["values"][:, j]`. Confirmed dtypes: float64
for belief/time columns, int32 for index/state/choice columns.

| column | dtype | source / notes |
|---|---|---|
| participant_id | str | subject identifier |
| trial_idx | int32 | 0..n_trials-1, session-level |
| run_idx | int32 | 0..n_runs-1 |
| trial_in_run | int32 | 0..trials_per_run-1 |
| regime | str | "stable" or "volatile" |
| outcome_time_s | float64 | cumulative seconds from session start → feeds `stimulus["times"]` |
| state | int32 | 0=safe, 1=dangerous |
| choice | int32 | 0=avoid, 1=approach (recorded behavior) |
| reward_mag | float64 | reward level if chosen |
| shock_mag | float64 | shock level if chosen |
| delta_hr | float64 | anticipatory Delta-HR in bpm |
| mu2 | float64 | HGF level-2 posterior mean (log-odds) → modulator channel |
| sigma2 | float64 | 1/precision at level 2 |
| mu3 | float64 | 3-level only; NaN for 2-level |
| sigma3 | float64 | 3-level only; NaN for 2-level |
| delta1 | float64 | input-level PE from node 0 temp dict |
| epsilon2 | float64 | level-2 precision-weighted PE (value_prediction_error) |
| epsilon3 | float64 | 3-level only; NaN for 2-level |
| psi2 | float64 | effective precision at level 2 (effective_precision) |

**Note on `delta1`:** pyhgf's binary-state node (node 0) stores the input-level
PE in `temp["value_prediction_error"]`. This was verified in the 18-03 runtime
inspection (SUMMARY 18-03). The `_safe_temp` helper returns NaN if the key is
absent, so the export is robust to pyhgf version drift.

---

## 6. Summary of surprises vs. plan sketch

| Claim in plan | Actual (post-audit) |
|---|---|
| "bilinear DCM deferred to v0.4+" | **Incorrect** — bilinear `NeuralStateEquation.derivatives(B=, u_mod=)` and `parameterize_B`/`compute_effective_A` are live in v0.3.0 (`neural_state.py` lines 62-224) |
| B-matrix modulator interface not in current tree | **Incorrect** — `CoupledDCMSystem.__init__` accepts `B` and `n_driving_inputs` (coupled_system.py); bilinear path is the default production path |
| Modulator values "bounded/normalized?" | **Raw float64** — no transform in `parameterize_B`; prior N(0,1) does regularization |
| Time in seconds, absolute from session start | **Confirmed** — `stimulus["times"]` in seconds (task_simulator.py lines 74, 145-146) |
| `stimulus` dict schema: keys `"times"` (K,) and `"values"` (K,M) | **Confirmed** — task_simulator.py lines 65-67 |
| `dtype=torch.float64` for stimulus tensors | **Confirmed** — default `dtype` arg is `torch.float64` (line 45) |
