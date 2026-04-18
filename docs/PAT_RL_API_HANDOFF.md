# PAT-RL Public API Handoff for dcm_hgf_mixed_models (heart2adapt-sim) v2

**Source repo:** `psilocybin_prl_analyses` (this repo)
**Consumer repo:** `dcm_hgf_mixed_models` (heart2adapt-sim) — v2 Sister-Toolbox Integration
**Supersedes:** `dcm_hgf_mixed_models/.planning/research/SISTER_API_PRL_HGF.md` (written before Phases 18-19; documents the older pick_best_cue surface)
**As of:** 2026-04-18 (Phase 18 cluster-smoke-validated; Phase 19 code-complete)

---

## 1. What changed since `SISTER_API_PRL_HGF.md`

That document (in the consumer repo) describes the **pick_best_cue** pipeline:
3-cue partial-feedback PRL, `generate_session`, `build_3level_network`,
`softmax_stickiness_surprise`, `fit_batch_hierarchical`. That surface is
correct and still available, but it is the **WRONG surface for PAT-RL /
HEART2ADAPT**.

Phases 18-19 added a **parallel `*_patrl` stack** for the binary-state
approach/avoid task (HEART2ADAPT's actual paradigm). Every pick_best_cue
module has a PAT-RL sibling:

| pick_best_cue (old) | PAT-RL (use this for HEART2ADAPT) |
|---------------------|-----------------------------------|
| `configs/prl_analysis.yaml` | `configs/pat_rl.yaml` |
| `env/task_config.py::load_config()` | `env/pat_rl_config.py::load_pat_rl_config()` |
| `env/simulator.py::generate_session()` | `env/pat_rl_sequence.py::generate_session_patrl()` |
| [N/A — no cohort helper] | `env/pat_rl_simulator.py::simulate_patrl_cohort()` |
| `models/hgf_2level.py::build_2level_network()` | `models/hgf_2level_patrl.py::build_2level_network_patrl()` |
| `models/hgf_3level.py::build_3level_network()` | `models/hgf_3level_patrl.py::build_3level_network_patrl()` |
| `models/response.py::softmax_stickiness_surprise()` | `models/response_patrl.py::model_a_logp()` |
| `fitting/hierarchical.py::fit_batch_hierarchical()` | `fitting/hierarchical_patrl.py::fit_batch_hierarchical_patrl()` |
| [N/A — MCMC only] | `fitting/fit_vb_laplace_patrl.py::fit_vb_laplace_patrl()` |
| [N/A — no trajectory export] | `analysis/export_trajectories.py::export_subject_trajectories()` |

**Rule for HEART2ADAPT:** never import a non-`_patrl` module from prl_hgf
unless you're deliberately using the pick_best_cue task. The two stacks share
zero runtime code to preserve pick_best_cue test stability.

---

## 2. Public API by subpackage (PAT-RL surface)

### 2.1 `prl_hgf.env` — task configuration + trial generation

```python
from prl_hgf.env.pat_rl_config import load_pat_rl_config, PATRLConfig
from prl_hgf.env.pat_rl_sequence import generate_session_patrl, PATRLTrial
from prl_hgf.env.pat_rl_simulator import (
    simulate_patrl_cohort, run_hgf_forward_patrl,
)
```

**`load_pat_rl_config(path: Path | None = None) -> PATRLConfig`**

Loads `configs/pat_rl.yaml` (or a user-provided path) into a frozen dataclass
tree. Completely isolated from `task_config.py` — no shared `AnalysisConfig`
inheritance. Validated via `__post_init__` for: hazard bounds, 2x2 magnitude
positivity, phenotype key completeness, MCMC hyperparam positivity.

`PATRLConfig` top-level fields:

- `.task: PATRLTaskConfig` — 192-trial / 4-run structure, hazard rates
  (stable=0.03, volatile=0.10), 2x2 reward/shock magnitudes, timing
  (cue=1.5s, anticipation=5.5s, outcome=2.0s, iti=2.0s), ΔHR stub
  distribution.
- `.simulation: PATRLSimulationConfig` — phenotype 2x2 grid
  (`healthy / anxious / reward_sensitive / anxious_reward_sensitive`) with
  omega_2 / beta / kappa / mu3_0 priors per phenotype. Literature-grounded:
  omega_2 (-6.0 healthy vs -3.5 anxious, Browning 2015 direction); beta
  (2.0 low vs 8.0 high reward, Daw 2006). kappa fixed at 1.0 across
  phenotypes. Master seed 5678.
- `.fitting: PATRLFittingConfig` — MCMC defaults (n_chains=2, n_tune=500,
  n_draws=500, target_accept=0.9, seed=42) and fitting priors (truncated
  normals for kappa/beta, normals for omega_2/omega_3/mu3_0).

**`generate_session_patrl(config: PATRLConfig, seed: int, delta_hr_override: np.ndarray | None = None) -> list[PATRLTrial]`**

Produces a 192-trial session with hazard-driven binary state reversals
(`stable` regime ~0.03 flips/trial; `volatile` ~0.10), uniform-random 2x2
magnitude draws, state-conditioned ΔHR stub (`N(-3,3)` on dangerous,
`N(0,3)` on safe, clipped to `[-15, 10]` bpm). Deterministic under `seed`.
Pass `delta_hr_override` for real-subject ΔHR data; values are still clipped.

`PATRLTrial` fields:

```python
trial_idx: int          # 0..191
run_idx: int            # 0..3
trial_in_run: int       # 0..47
regime: str             # "stable" or "volatile"
state: int              # 0=safe, 1=dangerous
reward_mag: float       # 1.0 or 5.0
shock_mag: float        # 1.0 or 5.0
delta_hr: float         # bpm, anticipatory bradycardia
outcome_time_s: float   # cumulative seconds from session start
```

**`simulate_patrl_cohort(n_participants: int, level: int, master_seed: int, config: PATRLConfig | None = None) -> tuple[pd.DataFrame, dict, dict]`**

Generates a multi-agent PAT-RL cohort by sampling phenotype-drawn parameters
from `config.simulation.phenotypes["healthy"]` (by default), producing per-
agent trials, running a forward HGF pass at the true parameters to derive
belief-driven choices, and rolling up into a `sim_df` suitable for
`fit_batch_hierarchical_patrl` / `fit_vb_laplace_patrl`.

Returns:
- `sim_df: pd.DataFrame` — columns: `participant_id` (str, "P000"-style),
  `trial_idx`, `run_idx`, `state`, `choice`, `reward_mag`, `shock_mag`,
  `delta_hr`, `outcome_time_s` — P rows x 192 trials each, tidy format.
- `true_params_by_participant: dict[str, dict[str, float]]` — ground-truth
  parameters per agent (omega_2, beta, kappa, mu3_0 as applicable). Used
  for recovery diagnostics only; NOT passed to the fit.
- `trials_by_participant: dict[str, list[PATRLTrial]]` — per-agent raw Trial
  objects, for trajectory export later.

### 2.2 `prl_hgf.models` — HGF topologies + Model A response

```python
from prl_hgf.models.hgf_2level_patrl import (
    build_2level_network_patrl, extract_beliefs_patrl,
    INPUT_NODE, BELIEF_NODE,
)
from prl_hgf.models.hgf_3level_patrl import (
    build_3level_network_patrl, extract_beliefs_patrl_3level,
)
from prl_hgf.models.response_patrl import model_a_logp, expected_value
```

**Binary-state HGF topology:** `INPUT_NODE=0, BELIEF_NODE=1` for 2-level;
add `VOLATILITY_NODE=2` for 3-level. Scalar single-input (NOT the pick_best_cue
3-branch `INPUT_NODES=(0,2,4)` tuple). Built via pyhgf `Network()` API.

**`build_2level_network_patrl(omega_2: float = -4.0) -> pyhgf.model.Network`**

Single binary-state input node (idx 0); single continuous-state value parent
(idx 1) with `tonic_volatility = omega_2`. Returns Network with
`input_idxs = (0,)` ready for `net.input_data(input_data=u[:, None],
time_steps=np.ones(n_trials))`.

**Critical pyhgf 0.2.8 API notes (learned the hard way in 18-03 / 18-04):**

- `time_steps` MUST be **1D** `np.ones(n_trials)`, not 2D `np.ones((n_trials, 1))`.
- `volatility_children=([BELIEF_NODE], [kappa])` tuple form for 3-level
  coupling; NOT `node_parameters={"volatility_coupling_children": (kappa,)}`.
- `net.node_trajectories` may include a key `-1` (time tracker); skip it when
  enumerating.
- Attribute reads: `attributes[node]["mean"]`, `["precision"]`,
  `["temp"]["value_prediction_error"]`,
  `["temp"]["volatility_prediction_error"]`,
  `["temp"]["effective_precision"]` — verified at runtime in
  `tests/test_export_trajectories.py::test_pyhgf_temp_keys_extracted`.

**`build_3level_network_patrl(omega_2=-4.0, omega_3=-6.0, kappa=1.0, mu3_0=1.0) -> Network`**

Adds a continuous volatility parent (idx 2) coupled to the value node via
`volatility_children=([1], [kappa])`.

**`extract_beliefs_patrl(net) -> dict[str, np.ndarray]`**

After `net.input_data(...)` has run, returns per-trial arrays:
`mu2` (level-2 posterior mean), `sigma2` (= 1/precision), `p_state` (=
sigmoid(mu2); **P(dangerous)** under the PAT-RL convention),
`expected_precision`. All shape `(n_trials,)`.

`extract_beliefs_patrl_3level(net)` adds `mu3`, `sigma3`, `epsilon3`.

**`model_a_logp(mu2, choices, reward_mag, shock_mag, beta) -> jnp.ndarray`**

Per-trial binary choice log-likelihood under Model A. EV direction
(Phase 18-03 decision #114):

```
P(dangerous) = sigmoid(mu2)
EV_approach  = (1 - P_danger) * V_rew - P_danger * V_shk
P(approach)  = softmax([0, beta * EV_approach])
```

`mu2` is clipped to `[-30, 30]` before sigmoid for numerical safety
(outer envelope; HGF-level clipping at `|mu2| < 14` is the inner guard).

Models B (ΔHR bias `gamma`), C (ΔHR x value `alpha, gamma`), D (trial-varying
omega via `lambda * ΔHR`) are **deferred to Phase 20+** — attempting to call
them raises `NotImplementedError`.

### 2.3 `prl_hgf.fitting` — batched NUTS + VB-Laplace

```python
from prl_hgf.fitting.hierarchical_patrl import fit_batch_hierarchical_patrl
from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl
from prl_hgf.fitting.laplace_idata import build_idata_from_laplace
```

**`fit_batch_hierarchical_patrl(sim_df, model_name, response_model="model_a", config=None, n_chains=None, n_tune=None, n_draws=None, target_accept=None, random_seed=None) -> az.InferenceData`**

Full-cohort BlackJAX NUTS fit. Reuses `_run_blackjax_nuts`, `_samples_to_idata`,
`_extract_nuts_stats` from `fitting/hierarchical.py` (pick_best_cue backend)
via import; the PAT-RL-specific logp factory builds a batched log-posterior
via `jax.vmap` over participants with Layer-2 clamping (`|mu2| < 14`) and
fp64 scan inputs (mandatory — pyhgf `lax.cond` requires dtype consistency).

Output: `az.InferenceData` with `chain`, `draw`, `participant_id` dims on
every parameter. Parameter names: `omega_2`, `beta` (2-level);
`omega_2, omega_3, kappa, beta, mu3_0` (3-level). Note `beta = exp(log_beta)`
is emitted as a deterministic transform of the log-space sampling variable
(Phase 18-04 decision #121).

**`fit_vb_laplace_patrl(sim_df, model_name, response_model="model_a", config=None, n_pseudo_draws=1000, max_iter=200, tol=1e-5, n_restarts=1, random_seed=0) -> az.InferenceData`**

Non-MCMC fit: quasi-Newton MAP via `jaxopt.LBFGS` on the SAME
`_build_patrl_log_posterior` as the NUTS path, then `jax.hessian` at the
mode, eigh-clip PD regularization (`eps=1e-8`), sample `n_pseudo_draws`
pseudo-samples from `MultivariateNormal(mode, cov)`, package as
`az.InferenceData` via `build_idata_from_laplace`. Deterministic under
`random_seed`. Typical walltime: <60s for 5 agents, single-threaded CPU.

Shape parity with `fit_batch_hierarchical_patrl`: **both paths return
InferenceData with the SAME variable names, dims, and `participant_id`
coords.** Downstream exporters accept either without a rename shim.

**Diagnostics**: Laplace output carries custom `sample_stats` keys
`hessian_min_eigval`, `hessian_max_eigval`, `n_eigenvalues_clipped`,
`ridge_added`, `converged`, `n_iterations`, `logp_at_mode`. NUTS output
carries the standard `diverging`, `acceptance_rate`, `energy`.

**`build_idata_from_laplace(mode, cov, param_names, participant_ids, n_pseudo_draws=1000, rng_key=0, diagnostics=None) -> az.InferenceData`**

Lower-level packaging helper. Emits dim name `participant_id` natively.
`param_names` must be one of
`_PARAM_ORDER_2LEVEL = ("omega_2", "log_beta")` or
`_PARAM_ORDER_3LEVEL = ("omega_2", "log_beta", "omega_3", "kappa", "mu3_0")`.
Mode dict values must be shape `(P,)` per key; validated at the top of the
factory.

### 2.4 `prl_hgf.analysis` — trajectory export

```python
from prl_hgf.analysis.export_trajectories import (
    export_subject_trajectories, export_subject_parameters,
)
```

**`export_subject_trajectories(participant_id, idata, trials, choices, model_name, output_dir) -> Path`**

Post-hoc forward pass at `idata.posterior` participant-mean parameters.
Writes one CSV per subject at `{output_dir}/{participant_id}_trajectories.csv`
with **exactly** these 19 columns (dcm_pytorch-consumable; see §5):

```
participant_id, trial_idx, run_idx, trial_in_run, regime,
outcome_time_s, state, choice, reward_mag, shock_mag, delta_hr,
mu2, sigma2, mu3, sigma3, delta1, epsilon2, epsilon3, psi2
```

2-level exports fill `mu3`, `sigma3`, `epsilon3` with `NaN` (schema
stability). `outcome_time_s` is cumulative absolute seconds from session
start — directly compatible with `pyro_dcm` `task_simulator.stimulus["times"]`
(which is also absolute seconds; confirmed in `18-05-dcm-interface-notes.md`).

**`export_subject_parameters(idata, model_name, output_dir, filename="parameter_summary.csv") -> Path`**

Single CSV with one row per `(participant_id, parameter)` pair. Columns:
`participant_id, parameter, posterior_mean, hdi_low, hdi_high`. Uses
`az.hdi(..., hdi_prob=0.94)` — **ArviZ 0.22 returns `"lower"/"higher"`
coord labels**, not `"low"/"high"` (decision #125).

Consumes **either** NUTS or Laplace InferenceData unchanged.

---

## 3. Integration-point map for dcm_hgf_mixed_models v2

The consumer's v2 milestone wires five subpackage stubs. Here is where each
should call prl_hgf PAT-RL modules:

### 3.1 `dcm_hgf_mixed_models.task/` — task environment

- **`task/trial_sequence.py`** — wrap `env.pat_rl_sequence.generate_session_patrl`
  behind a thin facade that accepts a `dcm_hgf_mixed_models` YAML (different
  schema from `pat_rl.yaml`). Do NOT re-implement trial generation.
- **`task/environment.py`** — wrap `env.pat_rl_simulator.simulate_patrl_cohort`
  for cohort-level synthesis. Provide adapters for HEART2ADAPT phenotype
  specifications (anxiety x reward sensitivity 2x2) if the consumer's
  phenotype schema diverges from `PATRLConfig.simulation.phenotypes`.

### 3.2 `dcm_hgf_mixed_models.agents/` — phenotype + HGF fitting

- **`agents/phenotypes.py`** — read `PATRLConfig.simulation.phenotypes` and
  map to the consumer's phenotype data classes. The 2x2 grid
  (`healthy / anxious / reward_sensitive / anxious_reward_sensitive`) is
  already defined with literature-grounded priors in `configs/pat_rl.yaml`.
- **`agents/simulate.py`** — call `simulate_patrl_cohort(...)` to generate
  the training data. The returned `(sim_df, true_params, trials_by_pid)`
  tuple is the primary handoff.
- **[IMPLICIT] agent-fit step** — use **either**
  `fit_batch_hierarchical_patrl` (NUTS, use for ground truth / cluster
  production runs) OR `fit_vb_laplace_patrl` (Laplace, use for rapid
  iteration + CI). Both return shape-identical `az.InferenceData`. For v2's
  "1 simulated agent" scope, Laplace is 20-50x faster and sufficient.

### 3.3 `dcm_hgf_mixed_models.bridge/` — HGF trajectory → DCM modulator

- **`bridge/hgf_to_dcm.py`** — call
  `export_subject_trajectories(participant_id, idata, trials, choices,
  model_name, output_dir)` to produce the per-trial CSV. Then extract the
  modulator columns the DCM bridge needs. For the current dcm_pytorch v0.3.0
  bilinear path (confirmed live in `neural_state.py:4-15`, §18-05 audit), the
  preferred modulator channels are:
  - `epsilon2` — level-2 precision-weighted prediction error (the primary
    HGF-derived "surprise" signal; see HEART2ADAPT proposal §2.3)
  - `epsilon3` — level-3 volatility PE (3-level only; exploratory)
  - `psi2` — effective precision / trial-by-trial learning rate
  - `delta_hr` — trial-level bradycardia, pass-through from input
- **`bridge/timing.py`** — use `outcome_time_s` from the trajectory CSV
  directly. Both axes are absolute seconds; no transform needed.
- **Modulator amplitude convention** (decision #123, from 18-05 audit):
  pass `epsilon2` / `epsilon3` / `psi2` / `delta_hr` as **raw float64 values,
  no bounding or normalization** — `pyro_dcm.neural_state.py:104-113` handles
  regularization via the `N(0, 1)` B-matrix prior.

### 3.4 `dcm_hgf_mixed_models.analysis/` — recovery + group comparison

- **`analysis/recovery.py`** — consume `export_subject_parameters(...)` CSV +
  `true_params_by_participant` dict to compute recovery correlations. Schema
  of the parameter summary CSV: `participant_id, parameter, posterior_mean,
  hdi_low, hdi_high`. Join on `participant_id`.
- **`analysis/group_comparison.py`** — phenotype-stratified contrasts live
  here. **prl_hgf has `analysis/bms.py` for random-effects BMS** (posterior
  model probabilities + exceedance probabilities) but does NOT currently
  emit per-subject Δ-evidence CSVs for PEB covariates. The consumer's
  `group_comparison` should either (a) call `bms.py` helpers directly and do
  its own PEB export, OR (b) request a Phase 20 feature in prl_hgf for
  `export_peb_covariates`. We recommend (a) for v2, (b) for v3+.

### 3.5 `dcm_hgf_mixed_models.plotting/` — visualization

- **`plotting/hgf_trajectories.py`** — pandas-read the per-subject
  trajectory CSV, plot `mu2`, `epsilon2`, `psi2` over `trial_idx`. For a
  reference visualization, see `notebooks/` in prl_hgf (not a hard handoff
  — rebuild from the trajectory CSV schema documented above).

---

## 4. Minimal end-to-end usage example (for v2 integration smoke)

Dropped into `dcm_hgf_mixed_models/scripts/00_run_full_pipeline.py` or a
notebook:

```python
from pathlib import Path

import arviz as az
import pandas as pd

# prl_hgf PAT-RL surface
from prl_hgf.env.pat_rl_config import load_pat_rl_config
from prl_hgf.env.pat_rl_simulator import simulate_patrl_cohort
from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl
from prl_hgf.analysis.export_trajectories import (
    export_subject_trajectories, export_subject_parameters,
)

# 1. Load PAT-RL config (or point at dcm_hgf_mixed_models's own YAML).
cfg = load_pat_rl_config()

# 2. Simulate a 3-agent cohort (fast iteration).
sim_df, true_params, trials_by_pid = simulate_patrl_cohort(
    n_participants=3, level=2, master_seed=42, config=cfg,
)

# 3. Fit via VB-Laplace (~15s on CPU for 3 agents).
idata = fit_vb_laplace_patrl(
    sim_df, model_name="hgf_2level_patrl", config=cfg,
)
# -> az.InferenceData with dims (chain=1, draw=1000, participant_id=3)

# 4. Export per-subject trajectories (CSV per agent, matches dcm_pytorch
#    modulator-input schema).
output_dir = Path("output/demo")
output_dir.mkdir(parents=True, exist_ok=True)

for pid in sim_df["participant_id"].unique():
    export_subject_trajectories(
        participant_id=pid,
        idata=idata,
        trials=trials_by_pid[pid],
        choices=sim_df[sim_df.participant_id == pid]["choice"].values,
        model_name="hgf_2level_patrl",
        output_dir=output_dir,
    )

# 5. Export parameter summary (for recovery / PEB).
export_subject_parameters(
    idata, model_name="hgf_2level_patrl", output_dir=output_dir,
)

# 6. Feed the trajectory CSV columns into pyro_dcm's bilinear DCM as
#    modulatory inputs. See dcm_pytorch/src/pyro_dcm/forward_models/
#    neural_state.py::NeuralStateEquation.derivatives(B=..., u_mod=...).
traj = pd.read_csv(output_dir / "P000_trajectories.csv")
# Example modulator tensor for DCM:
#   u_mod(t) = epsilon2(t), feeds into B_epsilon2 matrix.
```

Expected runtime: <60 seconds end-to-end on a laptop CPU, no MCMC.

---

## 5. Data contracts (authoritative)

### 5.1 Per-subject trajectory CSV

Produced by `export_subject_trajectories`. **19 columns, strict order,
float64 numerical / int64 indices / pd.StringDtype str**:

```
participant_id (str), trial_idx (int32), run_idx (int32),
trial_in_run (int32), regime (str — "stable"|"volatile"),
outcome_time_s (float64), state (int32 — 0=safe,1=dangerous),
choice (int32 — 0=avoid,1=approach), reward_mag (float64),
shock_mag (float64), delta_hr (float64 — bpm),
mu2 (float64), sigma2 (float64),
mu3 (float64, NaN if 2-level), sigma3 (float64, NaN if 2-level),
delta1 (float64), epsilon2 (float64),
epsilon3 (float64, NaN if 2-level), psi2 (float64)
```

Rows: 192 per agent (one per trial). Authoritative schema source:
`src/prl_hgf/analysis/export_trajectories.py` + the TSV captured in
`.planning/phases/18-pat-rl-task-adaptation/18-05-dcm-interface-notes.md`.

### 5.2 Per-subject parameter summary CSV

Produced by `export_subject_parameters`:

```
participant_id (str), parameter (str — omega_2|beta|omega_3|kappa|mu3_0),
posterior_mean (float64), hdi_low (float64), hdi_high (float64)
```

Rows: P agents x K parameters (K=2 for 2-level, K=5 for 3-level).
`beta` is reported in natural space (not log_beta) — already transformed
by `_samples_to_idata` / `build_idata_from_laplace`.

### 5.3 InferenceData shape

```
posterior:
  chain:          n_chains (NUTS) or 1 (Laplace)
  draw:           n_draws (NUTS) or n_pseudo_draws (Laplace, default 1000)
  participant_id: P  (always this dim name; NOT "participant")

variables (2-level): omega_2, beta, log_beta
variables (3-level): omega_2, omega_3, kappa, beta, log_beta, mu3_0

sample_stats (NUTS):    diverging, acceptance_rate, energy
sample_stats (Laplace): converged, n_iterations, logp_at_mode,
                        hessian_min_eigval, hessian_max_eigval,
                        n_eigenvalues_clipped, ridge_added
```

---

## 6. Known gaps / TODOs for HEART2ADAPT integration

- **Models B / C / D are not implemented yet.** `fit_vb_laplace_patrl` and
  `fit_batch_hierarchical_patrl` both raise `NotImplementedError` for
  `response_model != "model_a"`. If the consumer's v3 model comparison needs
  ΔHR-modulated response models, that's a Phase 20 feature request upstream.
- **No PEB covariate export helper.** `analysis/bms.py` produces model
  probabilities + exceedance; per-subject Δ-evidence / Δ-WAIC CSV for PEB
  regression is **not shipped**. v2 consumer should roll its own using
  `bms.py` primitives (see §3.4).
- **kappa is fixed at 1.0** in the 2x2 phenotype grid (`configs/pat_rl.yaml`).
  If HEART2ADAPT hypotheses need kappa variation across phenotypes, adjust
  the consumer-side phenotype spec and override via kwarg on the generative
  side.
- **Laplace-vs-NUTS hard gate still pending first cluster dual-fit run.**
  Preliminary agreement holds for 4 of 5 agents (see `VB_LAPLACE_CLOSURE_MEMO.md`);
  P004 is a known hard case for both methods. When the next
  `sbatch cluster/18_smoke_patrl_cpu.slurm` lands, it will produce
  `results/patrl_smoke/<job>/laplace_vs_nuts_diff.csv` with the formal
  Gate-#5 verdict.
- **`SISTER_API_PRL_HGF.md` in the consumer repo is now stale.** Replace /
  augment with a pointer to this document, or port the relevant sections
  in-tree.
- **pyhgf version pin.** CLAUDE.md says >=0.2.8; installed on the cluster
  at 0.2.8. Research for Phase 19 noted 0.2.10 in local dev env. The API
  paths documented in §2.2 work on 0.2.8 and 0.2.10. If the consumer
  upgrades pyhgf, re-run `tests/test_export_trajectories.py::test_pyhgf_temp_keys_extracted`
  canary first.
- **blackjax must be installed** for the NUTS path; `fit_vb_laplace_patrl`
  has zero blackjax dependency and works without it.

---

## 7. Quick pointers for the consumer

| Need | File |
|------|------|
| Full PAT-RL task spec (trial counts, magnitudes, timing, priors) | `configs/pat_rl.yaml` |
| Binary-state trial generator implementation | `src/prl_hgf/env/pat_rl_sequence.py` |
| Cohort simulation with phenotype sampling | `src/prl_hgf/env/pat_rl_simulator.py` |
| Binary HGF network builders (2L + 3L) | `src/prl_hgf/models/hgf_*_patrl.py` |
| Model A binary softmax response | `src/prl_hgf/models/response_patrl.py` |
| NUTS fitting (BlackJAX, cluster production path) | `src/prl_hgf/fitting/hierarchical_patrl.py` |
| VB-Laplace fitting (fast local CI path) | `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` |
| Trajectory + parameter CSV export for DCM bridge | `src/prl_hgf/analysis/export_trajectories.py` |
| dcm_pytorch consumer-interface audit | `.planning/phases/18-pat-rl-task-adaptation/18-05-dcm-interface-notes.md` |
| VB-Laplace design memo + tolerance gates | `.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md` |
| VB-Laplace current verdict | `.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_CLOSURE_MEMO.md` |
| Smoke entry point | `scripts/12_smoke_patrl_foundation.py` |
| Cluster entry point (CPU, dual-fit default) | `cluster/18_smoke_patrl_cpu.slurm` |
| End-to-end integration test | `tests/test_smoke_patrl_foundation.py` |

**For questions that reveal a gap in this document, update it in this repo
and send the diff over. This is the single source of truth for the PAT-RL
public API going forward; the consumer repo's `SISTER_API_PRL_HGF.md` should
be treated as deprecated stub documentation for the older pick_best_cue
surface only.**
