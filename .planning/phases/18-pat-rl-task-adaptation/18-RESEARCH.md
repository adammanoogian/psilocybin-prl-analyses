# Phase 18: PAT-RL Task Adaptation (HEART2ADAPT) — Research

**Researched:** 2026-04-17
**Domain:** Parallel task configuration + binary-state HGF + custom response models + trajectory export
**Confidence:** HIGH on repo architecture (first-hand source reads); MEDIUM on some pyhgf internals (verified via pyhgf 0.2.10 source in site-packages).
**User directive respected:** Repo conventions take precedence over YAML spec when they conflict.

---

## Summary

Phase 18 grafts the PAT-RL (approach/avoid, binary safe/dangerous state, 192 trials, hazard reversals, Delta-HR autonomic covariate) task onto the existing psilocybin PRL HGF pipeline, which is tightly specialized for **3 cues, 4 criterion-based phases, 420 trials, partial feedback**. Five pipeline touchpoints fight against that specialization: (1) `task_config.load_config` hardcodes `prl_analysis.yaml` and an `AnalysisConfig` dataclass tree built for 3-cue phases with criterion-driven reversals; (2) trial generation lives in `env/simulator.py` (not a non-existent `trial_sequence.py`) and emits a `Trial` dataclass that is intrinsically multi-cue; (3) both HGF builders (`hgf_2level.py`, `hgf_3level.py`) bake in `INPUT_NODES = (0, 2, 4)` and 3 parallel branches; (4) every fitting entry point (`hierarchical.py::build_logp_fn_batched`, `fit_batch_hierarchical`, `ops.py`, `jax_session.py`) passes arrays shaped `(P, n_trials, 3)` and `(P, n_trials)` keyed to exactly those 3 cues; (5) response math is hard-coded to a **3-way softmax** over `INPUT_NODES` means with `(beta, zeta)`, whereas PAT-RL Models A/B/C/D are **binary logistic** with `(beta, b, gamma, alpha, lambda)`.

The cleanest integration strategy — which the user's directive explicitly asks for — is to add PAT-RL **as a fully parallel vertical slice** rather than refactor the pick_best_cue stack. That means (a) a new YAML `configs/pat_rl.yaml` with its own dataclass tree parsed by a dedicated `env/pat_rl_config.py` loader; (b) a new trial-sequence module `env/pat_rl_sequence.py`; (c) new HGF builders `models/hgf_2level_patrl.py` and `models/hgf_3level_patrl.py` with single-input-node topology; (d) new response module `models/response_patrl.py` housing Models A/B/C/D; (e) a new batched JAX logp factory `fitting/hierarchical_patrl.py` that **does not** perturb the existing 3-cue logp; (f) a new trajectory-export module `analysis/export_trajectories.py`; and (g) a new BMS covariate helper on top of existing `analysis/bms.py`. The pick_best_cue modules are **untouched** — every existing callsite, test, and validation script keeps running byte-for-byte.

**Primary recommendation:** Build a parallel PAT-RL stack rooted at `configs/pat_rl.yaml` and `src/prl_hgf/env/pat_rl_config.py`; leave the existing pick_best_cue pipeline (and all its tests, VALID-01/02/03) completely untouched. Defer this work out of v1.2 into a new **v1.3 HEART2ADAPT milestone** spanning 4–5 integer phases; Phase 18 as currently scoped is too large for a single phase under the repo's observed granularity.

---

## Findings Keyed to the 11 Research Questions

### Q1 — Config loader dispatch strategy

**Recommendation: Option (c), dedicated `env/pat_rl_config.py` module with its own `PATRLConfig` dataclass tree and its own `load_pat_rl_config()` entry point. Do NOT modify `task_config.py`.**

**Evidence:**
- `src/prl_hgf/env/task_config.py:27` hardcodes `_DEFAULT_CONFIG_PATH = CONFIGS_DIR / "prl_analysis.yaml"`.
- `AnalysisConfig` (`task_config.py:428-446`) holds only `task: TaskConfig`, `simulation: SimulationConfig`, `fitting: FittingConfig`. `TaskConfig` (`task_config.py:140-226`) requires `n_cues`, `cue_labels`, `phases` (list of `PhaseConfig` whose `cue_probs` must match `n_cues`), and `transfer: TransferConfig`. None of that is PAT-RL compatible (no cues at all; state sequence instead; hazard rate not criterion).
- `load_config()` (`task_config.py:589-632`) is called from **21 distinct file locations** including `scripts/03_simulate_participants.py:39`, `scripts/04_fit_participants.py:49`, `scripts/05_run_validation.py:537`, `scripts/07b_run_recoverability_surface.py:23`, `scripts/08_run_power_iteration.py:50`, `scripts/09_run_prechecks.py:66`, `scripts/smoke_local_cpu.py:43`, `scripts/smoke_test_pipeline.py:32`, `src/prl_hgf/gui/explorer.py:38`, `src/prl_hgf/power/precheck.py:161`, `src/prl_hgf/power/config.py:192`, `src/prl_hgf/simulation/agent.py:123`, `src/prl_hgf/simulation/batch.py:98`, plus **~9 test files**. Polymorphism or dispatch would ripple through every callsite.
- Subclassing `TaskConfig` is blocked by `__post_init__` validation — `TaskConfig.__post_init__` at `task_config.py:176-204` will raise on `n_cues < 2` and on any phase whose `cue_probs` length differs from `n_cues`. PAT-RL state sequencing does not fit.
- **No shared `load_yaml` helper currently exists** (`task_config.py` uses `yaml.safe_load` inline at line 626). The planner should NOT refactor a shared helper now — that is a separate hygiene task. For this phase, `env/pat_rl_config.py` duplicates the ~12-line yaml-open-and-validate pattern.

**Concrete API:** `load_pat_rl_config(path: Path | None = None) -> PATRLConfig` where `PATRLConfig` is the PAT-RL-specific root dataclass. Scripts that run PAT-RL analyses import from the PAT-RL loader explicitly; this keeps test failures orthogonal to the pick_best_cue tests.

---

### Q2 — Trial generator module location

**Recommendation: New module `src/prl_hgf/env/pat_rl_sequence.py` with a new `PATRLTrial` dataclass. Do NOT extend `simulator.py` or subclass `Trial`.**

**Evidence:**
- The `Trial` dataclass at `env/simulator.py:29-57` has fields that are intrinsically 3-cue: `cue_probs: tuple[float, ...]`, `best_cue: int`, `phase_name`, `phase_label`. There is no `cue_chosen`/`state` concept at the Trial level. PAT-RL Trials need `state` (safe=0, dangerous=1), `run_idx`, `trial_in_run`, `reward_mag`, `shock_mag`, `delta_hr` (per-trial autonomic covariate), `outcome_time_s` (for DCM bridge export, per YAML PRL.4).
- `generate_session` (`simulator.py:83-165`) reads `config.task.n_sets`, `config.task.phases`, `config.task.transfer` — structures that do not apply to PAT-RL (hazard-based reversal, 4 runs of 48 trials).
- The `Trial` class is `frozen=True` so subclassing is permitted but provides no benefit because the required fields disjoin almost entirely. Dataclass composition is cleaner than inheritance here.
- Precedent: `src/prl_hgf/simulation/jax_session.py` was added as a new module parallel to `simulation/agent.py` rather than extending it (same pattern).

**Concrete API:** `PATRLTrial` dataclass + `generate_state_sequence(n_trials, hazard_rate, seed) -> np.ndarray`, `generate_magnitudes(...) -> (np.ndarray, np.ndarray)`, `generate_outcome(state, choice, contingencies, rng) -> tuple[float, float, float]` (reward_mag_received, shock_mag_received, nothing_flag), `generate_full_run(config, run_idx, seed) -> list[PATRLTrial]`, `generate_session_patrl(config, seed) -> list[PATRLTrial]`.

**Note on Delta-HR:** The YAML describes Delta-HR as a per-trial covariate but does NOT specify a generative model for it in the simulation path. The planner must either (a) treat Delta-HR as an exogenous input passed in from heart2adapt-sim, (b) add a simple generative model (e.g., Delta-HR correlated with anticipation of shock), or (c) punt — accept a Delta-HR array from the caller. **Recommendation: (c).** `generate_session_patrl` returns Trial objects WITHOUT Delta-HR; a separate helper `attach_delta_hr(trials, delta_hr_arr)` (or just a caller-passed covariate array) supplies it. This matches how real data will flow: the task provides structure, the physiology comes from the subject.

---

### Q3 — HGF Network topology for binary state

**Recommendation: Two new modules — `src/prl_hgf/models/hgf_2level_patrl.py` and `src/prl_hgf/models/hgf_3level_patrl.py` — each building a single-input-node binary HGF. Do NOT try to parameterize the existing builders by `n_inputs`.**

**Evidence:**
- `models/hgf_2level.py:41-47` defines `N_CUES: int = 3`, `INPUT_NODES = (0, 2, 4)`, `BELIEF_NODES = (1, 3, 5)` as module-level constants imported by **11 downstream modules** (fitting/hierarchical.py:69-72, fitting/ops.py, models/response.py:28, models/hgf_3level.py:55, simulation/agent.py:29, simulation/jax_session.py:215-221, gui/explorer.py, and test modules). Changing these to runtime-configurable would force a coordinated rewrite across the entire stack.
- `build_2level_network` (`hgf_2level.py:55-112`) and `build_3level_network` (`hgf_3level.py:84-158`) use three explicit `net.add_nodes(kind="binary-state")` / `net.add_nodes(kind="continuous-state", value_children=k)` calls — not a loop. Parameterizing by `n_inputs` would require a significant refactor.
- pyhgf 0.2.10 (verified from `C:\...\site-packages\pyhgf\model\network.py`) lets you build any graph topology — a single `kind="binary-state"` input node with one continuous-state value parent is 2 lines of `add_nodes` calls. 3-level adds a single volatility parent with one volatility-coupled child. Total new code per module: ~40 lines mirroring the existing builders but with scalar structures instead of 3-branch loops.

**Concrete API (2-level):**
```python
# hgf_2level_patrl.py — single-input binary HGF for PAT-RL
INPUT_NODE: int = 0
BELIEF_NODE: int = 1

def build_2level_network_patrl(omega_2: float = -4.0) -> Network:
    net = Network()
    net.add_nodes(kind="binary-state")                          # node 0
    net.add_nodes(kind="continuous-state",
                  value_children=0,
                  node_parameters={"tonic_volatility": omega_2})  # node 1
    net.input_idxs = (INPUT_NODE,)
    return net
```

**Note on extract helpers:** Provide `extract_beliefs_patrl(net)` returning `{"mu1", "sigma1", "mu2", "sigma2", "p_state", "expected_precision"}` — the PAT-RL response models need the **continuous-state posterior** `mu_2` (not `expected_mean` of the binary input), since per the YAML (PRL.3) `EV = sigmoid(mu_2) * V_rew - (1 - sigmoid(mu_2)) * V_shk`. That's a different quantity than the pick_best_cue response uses.

**3-level PAT-RL:** Single binary input + continuous value parent + continuous volatility parent (nodes 0, 1, 2; coupling 2→1 with kappa). Same simple mirror of `hgf_3level.py:84-158` but with single branch.

---

### Q4 — Response model scope

**Recommendation: New module `src/prl_hgf/models/response_patrl.py` containing Models A/B/C/D. Do NOT add to `response.py`.**

**Evidence:**
- `response.py::softmax_stickiness_surprise` (`response.py:31-115`) is a 3-way softmax indexed against `INPUT_NODES = (0, 2, 4)`: it stacks `hgf.node_trajectories[0]["expected_mean"]`, `[2]`, `[4]` (lines 90-92). That indexing is structurally wrong for a single-input-node network — the PAT-RL net has nodes 0 (binary input) and 1 (continuous value). Sharing a module invites dangerous misuse.
- The parameter vector contract differs: pick_best_cue is `[beta, zeta]` (2-vector); PAT-RL Models are `[beta, b]` (A), `[beta, b, gamma]` (B), `[beta, b, gamma, alpha]` (C), `[beta, b, lambda]` (D with lambda flowing to the perceptual model). Passing `response_function_parameters` of the wrong length produces silent index errors, not exceptions.
- Per repo convention (CLAUDE.md: "three-layer naming: math symbols inside class internals, descriptive names at API boundaries"), the math-symbol names used here (`beta`, `b`, `gamma`, `alpha`, `lambda`) are acceptable **inside** `response_patrl.py` internals, but the public API should also expose the log-likelihood shape contract clearly. Models A–D should each have a separate top-level function: `model_a_logp(...)`, `model_b_logp(...)`, etc., rather than packing them behind a `model_name` dispatcher inside one function (easier to unit test; easier to add Model E later).

**Concrete API:**
```python
# response_patrl.py — binary approach/avoid response models for PAT-RL
def model_a_logp(node_traj, choices, reward_mag, shock_mag,
                 beta, b) -> jnp.ndarray: ...   # softmax on EV
def model_b_logp(node_traj, choices, reward_mag, shock_mag, delta_hr,
                 beta, b, gamma) -> jnp.ndarray: ...   # Delta-HR bias
def model_c_logp(node_traj, choices, reward_mag, shock_mag, delta_hr,
                 beta, b, gamma, alpha) -> jnp.ndarray: ...  # x value sens.
# Model D's logp uses model_a's form; lambda enters perceptual scan, not here.
```

EV formula per YAML PRL.3: `EV = sigmoid(mu_2) * V_rew - (1 - sigmoid(mu_2)) * V_shk`. Use `jax.nn.log_sigmoid` / `log_softmax` for binary choice (2-way) to stay numerically stable.

---

### Q5 — Trial-varying omega for Model D (hardest item)

**Recommendation: Option (a) — extend the per-participant logp scan body to carry a **trial-axis `omega_2_arr` input**. pyhgf already reads `tonic_volatility` from `attributes[node]["tonic_volatility"]` at every step, so the clean implementation is: inside the scan step, mutate `attrs[belief_node]["tonic_volatility"]` to `omega_2_t` BEFORE calling `scan_fn`. This is a ~10-line addition to a new PAT-RL scan builder; it does NOT touch the pick_best_cue `_single_logp_2level` / `_single_logp_3level`.**

**Evidence:**
- pyhgf 0.2.10 implements `tonic_volatility` as a per-node attribute, not a closure constant. See `pyhgf.updates.prediction.continuous.py:70` and `:167` (fetched via Grep): the prediction step reads `time_step = attributes[-1]["time_step"]` on every step, and elsewhere the continuous-state node reads its own `tonic_volatility` from `attributes[node_idx]`. This means injecting a **trial-specific omega** simply requires overwriting that attribute each scan step. Precedent: the existing PyMC path injects omega once before the scan (`hierarchical.py:402-405`), but nothing in pyhgf's update sequence prevents doing it per-step.
- The current scan body in `hierarchical.py::_clamped_scan` (`hierarchical.py:116-195`) has the signature `_clamped_step(carry=attrs, x=scan_input_tuple)`. To add trial-varying omega, change the scan input to include omega per trial: `x = (values_t, observed_t, time_step_t, rng_key_t, omega_2_t)`. Inside `_clamped_step`, before calling `scan_fn`, do `attrs[belief_node] = {**attrs[belief_node], "tonic_volatility": omega_2_t}`.
- **Why not option (b) "callable omega"**: pyhgf's `scan_fn` is a partial'd JIT'd function (`utils/beliefs_propagation.py:24-160`, closed-over `update_sequence`, `edges`, `input_idxs`). Injecting a callable there means rewriting pyhgf internals — rejected by repo principle ("pyhgf.scan_fn is the source of truth; wrapping it in our own lax.scan gets the same benefit without maintenance burden" — REQUIREMENTS.md:64).
- The new PAT-RL scan signature effectively becomes: `scan_inputs = (values, observed, time_steps, None, omega_2_arr)` where `omega_2_arr = omega_2_base + lambda * delta_hr_arr` is precomputed per-trial outside the scan and threaded through. This keeps the scan math inside pyhgf.
- **Shape contract change is localized to the PAT-RL logp factory** — the pick_best_cue path ignores omega_2_arr and never pays the cost.

**Concrete API (inside `fitting/hierarchical_patrl.py`):**
```python
def _clamped_step_patrl(carry, x):
    values_t, observed_t, time_step_t, rng_key_t, omega_2_t = x
    attrs = carry
    # Mutate tonic_volatility for THIS trial (Model D trial-varying omega)
    belief_node_attrs = dict(attrs[BELIEF_NODE])
    belief_node_attrs["tonic_volatility"] = omega_2_t
    attrs = {**attrs, BELIEF_NODE: belief_node_attrs}
    new_attrs, new_node = scan_fn(attrs, (values_t, observed_t, time_step_t, rng_key_t))
    # ... same tapas-style clamping as hierarchical.py:155-195
```

Where for Models A/B/C: `omega_2_arr = jnp.full((n_trials,), omega_2_scalar)` (trial-invariant — numerically identical to the current path). For Model D: `omega_2_arr = omega_2 + lambda * delta_hr_arr`. The same scan body serves both; model_name just selects how to build the array.

**Open question flagged to planner:** pyhgf's `tapas_ehgf_binary.m`-style `_MU_2_BOUND = 14.0` clamp (`hierarchical.py:63`) applies to level-2 means. When omega varies per trial, the stability profile may change — high positive excursions of `omega` can destabilize the scan. **The planner should include a stability test case** that fits Model D on synthetic data with aggressive `lambda * delta_hr` excursions and verifies the clamp still prevents divergence.

---

### Q6 — Delta-HR input channel

**Recommendation: Option (b) + extend. Accept a SECOND input array `covariates_arr` shaped `(P, n_trials, K)` in the PAT-RL logp factory. Bit-exactness of VALID-01/02 for pick_best_cue is guaranteed because the PAT-RL factory is a SEPARATE function. Do NOT add a covariate axis to `fitting/hierarchical.py::build_logp_fn_batched`.**

**Evidence:**
- `build_logp_fn_batched` (`hierarchical.py:572-726`) signature is baked for `(P, n_trials, 3)` reward arrays. Every downstream user — `fit_batch_hierarchical` (line 1916), the two numpyro model functions (lines 1546, 1632), the BlackJAX `_build_log_posterior` (line 734), the traced-arg sampling loop `_build_sample_loop` (line 1229) — passes exactly `(input_data, observed, choices, trial_mask)` as a fixed tuple. Adding an optional covariate array means changing every one of those signatures and retesting VALID-01/02/03.
- The existing stack is post-Phase 17 BlackJAX and still has Phase 14 integration + Phase 15 production run pending. Perturbing shape contracts now creates regression risk for work already queued.
- A **parallel `build_logp_fn_batched_patrl`** (in a new `fitting/hierarchical_patrl.py`) can take whatever signature PAT-RL needs: `(state_arr, observed_arr, choices_arr, reward_mag_arr, shock_mag_arr, delta_hr_arr, trial_mask)` — 7 arrays. No ambiguity, no shared-code landmine, no pick_best_cue test perturbation.
- The batch DataFrame loader `_build_arrays_single` (`hierarchical.py:1877-1913`) is also 3-cue-specific (partial-feedback cue-masking on 3 columns). PAT-RL needs its own DataFrame loader that reads `state`, `choice` (binary 0/1), `reward_mag`, `shock_mag`, `delta_hr` columns.

**Concrete decision:** The PAT-RL fitting path is a full parallel stack from DataFrame → JAX arrays → logp → NUTS. Share the **BlackJAX NUTS runner** (`_run_blackjax_nuts`, `_build_sample_loop`) if feasible — it is generic over logdensity shape. But build separate logp/logdensity factories.

---

### Q7 — Trajectory export (PRL.4)

**Recommendation: Option (c) (with a modification). Use the same `_build_session_scanner` factory pattern already established in `simulation/jax_session.py:66-92`, but build a PAT-RL-specific scanner + a posterior-mean evaluator in a new module `analysis/export_trajectories.py`. Do NOT collect trajectories inside MCMC (option b — far too expensive at 4×1000 draws × 192 trials × 32 subjects).**

**Evidence:**
- `simulation/jax_session.py:66-92` already shows the factory pattern: build pyhgf Network once (outside JAX), seed `scan_fn` via one dummy `input_data` call, capture `base_attrs` and `scan_fn`. Then a pure-JAX `_run_session` function takes params + trial inputs + rng_key and runs `lax.scan`.
- pyhgf's node `temp` attributes carry exactly the quantities the DCM bridge needs:
  - `attributes[node]["temp"]["value_prediction_error"]` = delta_1 (raw PE level 1) and delta_2 (continuous PE level 2) — see `pyhgf/updates/prediction_error/binary.py:43` and `pyhgf/updates/prediction_error/continuous.py:70`.
  - `attributes[node]["temp"]["volatility_prediction_error"]` — see `pyhgf/updates/prediction_error/volatile.py:54` and `pyhgf/updates/prediction_error/continuous.py:128`.
  - `attributes[node]["temp"]["effective_precision"]` = psi_2 (precision weight / effective learning rate) — see `pyhgf/updates/prediction/volatile.py:158` and `pyhgf/updates/prediction/continuous.py:273`.
  - `attributes[node]["mean"]` / `["precision"]` give mu and sigma = 1/precision. sigma_2 = 1/precision_2. (`hgf_3level.py:207-211` already does this.)
- So **all eight per-trial quantities listed in YAML PRL.4** (mu2, sigma2, mu3, sigma3, epsilon2, epsilon3, delta1, psi2) are available as outputs of `scan_fn`'s accumulated trajectory — the full `node_trajectories` pytree already carries them. No pyhgf modifications required.
- Cost estimate: one forward pass per subject at posterior means = 192 trials × ~50us/step × 32 subjects ≈ 0.3 seconds. Option (c) is ~1000x cheaper than option (b) and loses nothing since DCM consumes posterior means, not full posteriors.

**Concrete API:**
```python
# analysis/export_trajectories.py
def export_subject_trajectories(
    subject_id: str,
    idata: az.InferenceData,
    trial_inputs: PATRLTrialInputs,  # state, choice, reward_mag, shock_mag, delta_hr
    model_name: str,   # "hgf_2level_patrl" or "hgf_3level_patrl"
    output_dir: Path,
) -> Path:
    posterior_means = _posterior_means_for_subject(idata, subject_id)
    traj = _run_forward_pass_at_means(posterior_means, trial_inputs, model_name)
    df = _build_trajectory_df(trial_inputs, traj)  # adds outcome_time_s, run_idx
    out_path = output_dir / f"{subject_id}_trajectories.csv"
    df.to_csv(out_path, index=False)
    return out_path

def export_subject_parameters(idata, output_dir) -> Path:
    # Per-subject posterior means as flat CSV for PEB covariates
```

**Per YAML PRL.4:** Also emit a per-subject parameter summary CSV (not per-trial) — posterior means of omega_2, omega_3, kappa, beta, b, gamma, alpha, lambda where applicable. This is just `idata.posterior.mean(dim=["chain","draw"]).to_dataframe()` with participant_id rows, ~15 lines.

---

### Q8 — Phenotype 2x2 generative framework (PRL.V2)

**Recommendation: New YAML key `simulation.phenotypes` inside `configs/pat_rl.yaml` (NOT in `prl_analysis.yaml`), with its own dataclass `PhenotypeConfig`. Parameter sampling is orthogonal to the psilocybin/placebo groups structure — don't try to unify them.**

**Evidence:**
- `configs/prl_analysis.yaml:152-213` has `simulation.groups` keyed by `psilocybin` / `placebo` with session deltas for pharmacological intervention (`omega_2_deltas` per session). PAT-RL's phenotype 2x2 (anxiety × reward sensitivity) describes **static individual differences**, not session-level drug effects — so the structure is different enough that forcing it into `groups` + `session_deltas` is awkward.
- `simulation/agent.py::sample_participant_params` (`agent.py:88-150`) is hard-coded to `GroupConfig` + `SessionConfig` + `session_idx` ordering. PAT-RL phenotype sampling is `phenotype_idx` (one of 4) with no session delta concept.
- The YAML PRL.V2 acceptance criterion "omega separates anxiety (d >= 0.5), beta separates reward sensitivity (d >= 0.5)" is measurable via a dedicated phenotype dataclass holding four `{mean, sd}` pairs per parameter per phenotype. The 2x2 can be represented cleanly as `phenotypes: { healthy: {...}, anxious: {...}, reward_sensitive: {...}, anxious_reward_sensitive: {...} }`.

**Concrete YAML shape:**
```yaml
# configs/pat_rl.yaml
simulation:
  n_participants_per_phenotype: 8     # 4 × 8 = 32 total for V1/V2
  master_seed: 5678
  phenotypes:
    healthy:
      omega_2: {mean: -3.5, sd: 0.5}
      kappa:   {mean: 1.0, sd: 0.3}
      beta:    {mean: 2.0, sd: 0.5}
      # ... plus response-model-specific params (b, gamma, alpha, lambda)
    anxious:
      omega_2: {mean: -2.5, sd: 0.5}   # HIGHER (less negative) = faster learning = anxiety
      # ...
    reward_sensitive:
      beta:    {mean: 4.0, sd: 0.5}    # HIGHER = more reward-driven
      # ...
    anxious_reward_sensitive:
      # ...
```

**Note:** The YAML spec does NOT provide numeric mean/sd targets for phenotype separability. The planner will need to choose values that satisfy the d >= 0.5 identifiability criterion — expect 1–2 iterations of tuning (this is part of PRL.V2's "if confounded, rethink definitions" loop).

---

### Q9 — BMS + PEB covariate export (PRL.5)

**Recommendation: Add a thin extension to `analysis/bms.py` (one new function, `compute_peb_covariates`) that emits per-subject Delta-elpd-WAIC (2-level minus 3-level) as a CSV. Existing `compute_subject_waic` / `compute_batch_waic` / `run_stratified_bms` need no modification.**

**Evidence:**
- `analysis/bms.py::compute_subject_waic` (`bms.py:64-167`) already computes per-subject `elpd_waic` for a given model. `compute_batch_waic` (`bms.py:170-277`) emits a DataFrame with columns `participant_id, group, session, model, elpd_waic` — **exactly the structure needed to pivot into per-subject ΔWAIC**.
- `run_stratified_bms` (`bms.py:367-447`) already averages across sessions and pivots into (n_subjects, n_models). Extracting Delta-elpd as (n_subjects,) is trivial: `delta = pivoted["hgf_2level_patrl"] - pivoted["hgf_3level_patrl"]`.
- The only new piece: write a helper that takes the `waic_df` + model-name pair and writes `{participant_id, delta_elpd_waic, delta_f_approx}` to CSV for heart2adapt-sim DCM / PEB.
- WAIC-based Delta-F (for BMS free energy) is approximated via Akaike-form correction; alternatively just export Delta-elpd-WAIC and let downstream PEB scripts decide the transformation. **Planner should confirm with user which quantity the DCM/PEB bridge expects.**

**Concrete API:**
```python
# New function in analysis/bms.py (or analysis/peb_covariates.py for separation)
def compute_peb_covariates(
    waic_df: pd.DataFrame,
    model_names: tuple[str, str] = ("hgf_2level_patrl", "hgf_3level_patrl"),
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Emit per-subject Delta-elpd-WAIC for PEB covariate use."""
    # pivot, difference, optional write
```

Stratified BMS per phenotype works **unchanged** — the existing `run_stratified_bms` groups by the `group` column. For PAT-RL, repurpose `group` to hold phenotype labels.

---

### Q10 — Fit pipeline entry point dispatcher

**Recommendation: Do NOT extend `fit_batch_hierarchical`. Create a new entry point `fit_batch_hierarchical_patrl` in `fitting/hierarchical_patrl.py` that calls the same BlackJAX NUTS runner but with PAT-RL-specific log-posterior and array preparation.**

**Evidence:**
- `fit_batch_hierarchical` signature at `hierarchical.py:1916-1927`: accepts `model_name: str = "hgf_3level"` where `_MODEL_NAMES = ("hgf_2level", "hgf_3level")` (line 66). Adding `"hgf_2level_patrl"`, `"hgf_3level_patrl"`, `"patrl_model_a"`, `"patrl_model_b"`, `"patrl_model_c"`, `"patrl_model_d"` (6 new dispatch targets) inflates the function into a god-object.
- The `sim_df` column contract is also pick_best_cue-specific: `_build_arrays_single` (line 1877-1913) requires `cue_chosen` and `reward`. PAT-RL needs `state`, `choice` (binary), `reward_mag`, `shock_mag`, `delta_hr`, plus potentially `run_idx`. Different DataFrame schema → different loader → different entry point.
- The PAT-RL entry point can **reuse** `_run_blackjax_nuts` (`hierarchical.py:902-1092`), `_build_sample_loop` (`hierarchical.py:1229-1466`), `_samples_to_idata` (`hierarchical.py:1469-1538`), and `_extract_nuts_stats` (`hierarchical.py:863-899`) — these are generic over logdensity-shape. This is the correct level of code sharing.

**Concrete API:**
```python
# fitting/hierarchical_patrl.py
def fit_batch_hierarchical_patrl(
    sim_df: pd.DataFrame,
    model_name: str = "hgf_3level_patrl",    # "hgf_2level_patrl" | "hgf_3level_patrl"
    response_model: str = "model_a",          # "model_a" | "model_b" | "model_c" | "model_d"
    n_chains: int = 4, n_draws: int = 1000, n_tune: int = 1000,
    target_accept: float = 0.95, random_seed: int = 42,
    warmup_params: dict | None = None,
) -> az.InferenceData | tuple[az.InferenceData, dict]:
```

That's 2 × 4 = 8 valid combinations; the response_model controls the logp form (A/B/C/D), and model_name controls perceptual model (2-level vs 3-level HGF). Model D is only valid with `response_model="model_d"` + either HGF variant.

---

### Q11 — Milestone placement

**Recommendation: MOVE Phase 18 out of v1.2 and into a new v1.3 HEART2ADAPT milestone. Phase 18 as currently scoped is too large for one integer phase (would be 5–7 plans); the repo's observed phase granularity (Phases 12–17 average 2–3 plans each, each with a single cohesive goal) suggests splitting PAT-RL into ~4 integer phases under v1.3.**

**Evidence:**
- v1.2's `REQUIREMENTS.md:1-78` is strictly "Hierarchical GPU Fitting" — BATCH-*, JSIM-*, VALID-*, BENCH-*, PROD-*, NPRO-*, BJAX-*. PAT-RL is an entirely different project (HEART2ADAPT, per the YAML) and introduces a new task, new config, new HGF topology, new response model family, new trajectory export. Zero requirements overlap.
- v1.2 has **two phases still pending** (Phase 14 Integration + GPU Benchmark, Phase 15 Production Run) that must close the original power-analysis mission. Phase 18 does not block those; those do not block Phase 18.
- Phase 17 (BlackJAX NUTS sampler) shipped 2026-04-15. Phase 18 depends on Phase 17's `fit_batch_hierarchical` BlackJAX path. But a new milestone v1.3 can still declare "depends on v1.2 Phase 17 complete" and sequence cleanly.
- Repo phase granularity evidence: Phases 12–17 each have a single cohesive deliverable and 2–3 plans (from ROADMAP.md:113-218). Phase 18's current success criteria (ROADMAP.md:225-232) list **7 distinct deliverables**: new YAML loader, trial-sequence module, four response models + trial-varying omega scan change, trajectory export, stratified BMS + PEB, parameter recovery validation at 192 trials, phenotype 2x2 validation. Collapsing those into one phase violates the observed cadence and creates a massive single-PR diff risk.

**Suggested v1.3 phase split:**
| v1.3 Phase | Goal | Rough plans |
|------------|------|-------------|
| 19 — PAT-RL env foundation | configs/pat_rl.yaml, `env/pat_rl_config.py`, `env/pat_rl_sequence.py`, PAT-RL HGF builders | 2 plans |
| 20 — PAT-RL response models | Models A/B/C + Model D trial-varying omega scan + `response_patrl.py` | 2 plans |
| 21 — PAT-RL fitting + simulation | `fitting/hierarchical_patrl.py` BlackJAX entry, PAT-RL simulation path + agent sampling | 2 plans |
| 22 — PAT-RL validation + export | PRL-V1 recovery at 192 trials, PRL-V2 phenotype identifiability, trajectory export, BMS covariate CSV | 2 plans |

(Planner can collapse to 3 phases if the response-models + fitting phases can land atomically.)

**If the user insists on keeping Phase 18 as one integer phase inside v1.2**, scope it down to the **bare minimum parallel infrastructure**: config loader + trial generator + PAT-RL HGF builders + one Model (A only). Defer Models B/C/D, trajectory export, BMS covariates, and validation to later phases. That still makes Phase 18 ~3 plans (the smallest phase would then be ~2 plans), which is acceptable.

---

## Scope and Milestone Question (Separate Detailed Recommendation)

The v1.2 REQUIREMENTS.md is about GPU batched hierarchical fitting to ship the v1.1 power-analysis deliverables. Phase 18 as described ships an entirely different feature area — it adds a whole new task type to the toolbox. It does not advance v1.1's production deliverables; it does not close any v1.2 requirement (BATCH-*, JSIM-*, VALID-*, BENCH-*, PROD-*, NPRO-*, BJAX-*). It is a thematic non-sequitur inside v1.2.

### Option A: Move to v1.3 HEART2ADAPT (recommended)
- Create new milestone `v1.3 HEART2ADAPT` with its own REQUIREMENTS.md carrying PRL.1–PRL.5 and PRL.V1–V2 as numbered requirements.
- v1.3 depends on v1.2 Phase 17 complete (BlackJAX path).
- v1.2 stays focused on finishing Phase 14/15 (the pending production run).
- v1.3 splits into ~4 integer phases (19–22) with 2 plans each. Typical phase size. Clear acceptance per phase.

### Option B: Keep Phase 18 in v1.2, but MINIMIZE scope
- Phase 18 ships ONLY: YAML + config loader + trial generator + single-input-node HGF builders (2-level + 3-level) + Model A response + one smoke test.
- Everything else (Models B/C/D, trial-varying omega, trajectory export, BMS covariates, recovery validation, phenotype validation) moves to Phases 19+ under v1.3 or v1.4.
- Phase 18 becomes "PAT-RL foundation" rather than "PAT-RL everything". ~3 plans.

### Option C (NOT recommended): Keep Phase 18 as a mega-phase
- 6–8 plans, unusual cadence, large single-PR risk.
- Only pursue if user explicitly accepts the cadence deviation.

**Planner should present these three options to the user before beginning Phase 18 planning.** The user's architectural directive (repo needs > YAML) already prefigures Option A — the YAML was authored without knowing the existing milestone structure.

---

## Planner Handoff

### Concrete plans to create (under Option A, v1.3)

**Phase 19 — PAT-RL environment foundation**
- `19-01-PLAN.md` — `configs/pat_rl.yaml` spec + `src/prl_hgf/env/pat_rl_config.py` (`PATRLConfig`, `load_pat_rl_config`) + 6 unit tests (YAML round-trip, validation errors)
- `19-02-PLAN.md` — `src/prl_hgf/env/pat_rl_sequence.py` (state sequence, magnitudes, outcome, PATRLTrial) + `src/prl_hgf/models/hgf_2level_patrl.py` + `src/prl_hgf/models/hgf_3level_patrl.py` + 8 unit tests

**Phase 20 — PAT-RL response models + trial-varying omega**
- `20-01-PLAN.md` — `src/prl_hgf/models/response_patrl.py` with Models A, B, C + 8 tests (per-model log-prob shape tests, numerical stability)
- `20-02-PLAN.md` — Trial-varying omega scan body in `fitting/hierarchical_patrl.py` (Model D perceptual support); 5 tests (constant-omega branch matches Model A bit-exactly; varying-omega produces different posterior trajectories; Layer 2 clamp holds under extreme lambda)

**Phase 21 — PAT-RL batched fitting + simulation**
- `21-01-PLAN.md` — `fitting/hierarchical_patrl.py::fit_batch_hierarchical_patrl` (BlackJAX path, reuses `_run_blackjax_nuts` + `_build_sample_loop`) + `_build_arrays_single_patrl` + 4 tests (5-participant CPU smoke for Models A/B/C/D × 2 HGFs)
- `21-02-PLAN.md` — PAT-RL JAX simulation path parallel to `simulation/jax_session.py` + PAT-RL agent sampling (`simulate_patrl_session_jax`, `sample_participant_params_patrl` with phenotype support) + 5 tests

**Phase 22 — PAT-RL validation, trajectory export, BMS covariates**
- `22-01-PLAN.md` — `analysis/export_trajectories.py` (per-subject trajectory CSV + per-subject parameter CSV) + extend `analysis/bms.py` with `compute_peb_covariates` + 6 tests
- `22-02-PLAN.md` — PRL-V1 recovery validation at 192 trials + PRL-V2 phenotype identifiability + scripts/`20_run_patrl_validation.py` + results figures

### Parallelization waves

- **Wave 1:** 19-01 and 19-02 are sequential (02 depends on 01's `PATRLConfig` dataclass shape). Plan 19-02's three sub-artifacts (sequence + 2 HGF builders) could split into 3 parallel sub-plans if the planner wants finer granularity.
- **Wave 2:** After Phase 19 lands, 20-01 and 20-02 can run in parallel — response models (A/B/C) and scan-body extension (D) touch different files (`response_patrl.py` vs. `fitting/hierarchical_patrl.py`). Merge order: 20-02 first, then 20-01 (so Model D logp in 21-01 can import from both).
- **Wave 3:** 21-01 and 21-02 are parallelizable — fitting entry point uses live HGFs but no simulation code, and simulation path uses HGFs but no fitting code.
- **Wave 4:** 22-01 and 22-02 are sequential (22-02 runs the pipeline end-to-end using 22-01's export helpers).

### Shared infrastructure to keep stable (do NOT edit during Phase 18+)

The planner should declare these files untouched by PAT-RL work:
- `src/prl_hgf/env/task_config.py` (pick_best_cue only)
- `src/prl_hgf/env/simulator.py` (pick_best_cue only)
- `src/prl_hgf/models/hgf_2level.py`, `hgf_3level.py`, `response.py` (pick_best_cue only)
- `src/prl_hgf/fitting/hierarchical.py` (including `build_logp_fn_batched`, `fit_batch_hierarchical`, `_build_log_posterior`, `_run_blackjax_nuts`, `_build_sample_loop`, `_samples_to_idata`, `_extract_nuts_stats`) — all preserved for pick_best_cue
- `src/prl_hgf/simulation/agent.py`, `simulation/jax_session.py`, `simulation/batch.py` — pick_best_cue only
- `configs/prl_analysis.yaml`
- All existing tests in `tests/` — they continue to pass without modification

PAT-RL **reuses** (imports from) the BlackJAX generic runners but does not modify them.

---

## Open Questions the Planner Should NOT Assume Away

1. **Delta-HR generative model unspecified.** The YAML describes Delta-HR as a per-trial covariate used in Models B/C/D but does NOT specify how simulated Delta-HR is produced during parameter recovery. Options: (a) sample from a simple Gaussian, (b) correlate with anticipated shock magnitude, (c) accept caller-supplied arrays. The planner must surface this to the user — recommended choice (c) above but this is a user decision.

2. **PEB covariate format.** PRL.5 says "ΔWAIC or ΔF". These are different quantities. The planner should ask the user whether the heart2adapt-sim DCM bridge expects Delta-elpd-WAIC (continuous, per-subject), log Bayes factor (requires free-energy computation), or both. Default to Delta-elpd-WAIC unless user specifies.

3. **2x2 phenotype numeric parameters.** The YAML lists the design axes (anxiety, reward sensitivity) but does NOT give mean/sd for phenotype parameter distributions. The planner must decide whether (a) the user supplies these, or (b) Phase 22 iterates on values until PRL-V2 identifiability (d >= 0.5, r < 0.5) is achieved. This may require a plan-within-a-plan tuning loop.

4. **Trial-varying omega stability.** `_MU_2_BOUND = 14.0` clamp (per tapas) is sized for static omega. With Model D's trial-varying omega, aggressive lambda can push the scan into repeated clamping events, degrading the logp. Recommend: Phase 20 plan 20-02 include a stability sweep (`lambda in [-0.5, 0.5]`, `delta_hr ~ N(0, 1)`) and report clamp rate.

5. **Model D response form.** YAML PRL.3 says "Model D: learning rate by Delta-HR, response model same as A". So Model D uses Model A's response logp but Model D's perceptual scan. Confirm: Models B and C use their own response (Delta-HR in the choice logit); Model D uses plain softmax-EV (Model A's response) but with trial-varying omega in the perceptual model. **Planner verifies this is the correct interpretation against YAML lines 124-128 and 130-136.**

6. **pyhgf 0.2.10 vs 0.2.8.** The repo CLAUDE.md lists pyhgf 0.2.8 but the installed version is 0.2.10 (per `pip show`). Attribute keys (`temp`, `expected_mean`, `mean`, `precision`) match between minor versions — verified from source — but the planner should run an early smoke test to confirm PAT-RL trajectory extraction works on whichever version is pinned in environment.yml.

7. **Delta-HR array shape on Model D.** When omega becomes trial-varying, the scan body must receive `(omega_t,)` along with the existing scan inputs. The planner should confirm whether Delta-HR is single-session (192 values) or multi-run (4 × 48 values with ITI gaps). If multi-run, `time_step` varies per trial — existing pick_best_cue uses `time_steps = np.ones(n_trials)`, which is fine when samples are equally spaced. For PAT-RL, if ITIs vary, use actual `time_step_s` from the trial schedule so pyhgf's volatile prediction integrates correctly.

8. **Milestone vs phase decision.** Research recommends Option A (v1.3 split). Planner must escalate to the user before planning proceeds — this is not a planner-level decision.

---

## Sources

### Primary (HIGH confidence) — repo source code, directly read

- `src/prl_hgf/env/task_config.py` (lines 27, 140–226, 428–446, 589–632)
- `src/prl_hgf/env/simulator.py` (lines 29–57, 83–165)
- `src/prl_hgf/env/__init__.py` (lines 1–19)
- `src/prl_hgf/models/hgf_2level.py` (lines 41–47, 55–112, 120–176, 184–234)
- `src/prl_hgf/models/hgf_3level.py` (lines 55–76, 84–158, 166–213)
- `src/prl_hgf/models/response.py` (lines 28–115)
- `src/prl_hgf/models/__init__.py` (lines 37–63)
- `src/prl_hgf/fitting/hierarchical.py` (lines 60–72, 80–260, 268–564, 572–726, 734–860, 863–1226, 1229–1466, 1469–1538, 1546–1629, 1632–1694, 1702–1869, 1877–1913, 1916–2220)
- `src/prl_hgf/fitting/__init__.py` (lines 1–63)
- `src/prl_hgf/fitting/ops.py` (lines 54–100)
- `src/prl_hgf/analysis/bms.py` (lines 44–277, 285–447)
- `src/prl_hgf/simulation/agent.py` (lines 29, 36–150)
- `src/prl_hgf/simulation/jax_session.py` (lines 43, 66–92, 100–296, 303–441)
- `configs/prl_analysis.yaml` (lines 1–259)
- `config.py` (lines 1–26)
- `.planning/ROADMAP.md` (Phase 18 entry at lines 220–247)
- `.planning/REQUIREMENTS.md` (v1.2 requirements, lines 1–78)
- `.planning/MILESTONES.md`

### Primary (HIGH confidence) — pyhgf 0.2.10 installed source

- `pyhgf/utils/beliefs_propagation.py` (lines 1–161) — confirms scan_fn reads `attributes[idx]["tonic_volatility"]` every step
- `pyhgf/updates/prediction/continuous.py` (lines 70, 167, 247, 273) — `tonic_volatility`, `temp["effective_precision"]`, `temp["current_variance"]`
- `pyhgf/updates/prediction/volatile.py` (lines 15, 46, 89, 134, 145, 158) — volatility prediction internals
- `pyhgf/updates/prediction_error/binary.py` (lines 43, 86–87) — `temp["value_prediction_error"]`
- `pyhgf/updates/prediction_error/continuous.py` (lines 70, 128) — `temp["value_prediction_error"]`, `temp["volatility_prediction_error"]`
- `pyhgf/updates/prediction_error/volatile.py` (lines 28, 54) — volatility PE in temp
- `pyhgf/updates/posterior/continuous/posterior_update_mean_continuous_node.py` (lines 131, 164, 170) — confirms temp reads are the canonical PE source
- `pyhgf/model/add_nodes.py` (lines 45, 119, 207) — `temp` initialization, `effective_precision` default
- `pyhgf/model/network.py` (lines 98, 109–127, 468, 574) — Network, input_idxs, input_data, scan_fn setters

### Secondary (MEDIUM confidence) — YAML spec

- `C:/Users/aman0087/Downloads/GSD_prl_hgf.yaml` (full file, PRL.1–V2)

### Source hierarchy compliance

No WebSearch was used; all claims are grounded in direct source reads of either the repo or the installed pyhgf package. Every file and line citation was verified before writing.

---

## Metadata

**Confidence breakdown:**
- Repo architecture and callsite mapping: HIGH — first-hand code reads from all affected files
- pyhgf integration strategy (trial-varying omega, temp extraction): HIGH — verified against pyhgf 0.2.10 source in site-packages
- Milestone/phase scoping recommendation: MEDIUM — based on observed repo cadence (Phases 12–17) and requirements analysis; user judgment still required for final decision
- Delta-HR generative model and PEB covariate format: LOW — YAML underspecifies; flagged to planner
- pyhgf version pin (0.2.8 in CLAUDE.md vs 0.2.10 installed): MEDIUM — attribute keys match, but planner should confirm at smoke-test time

**Research date:** 2026-04-17
**Valid until:** 2026-05-17 (30 days; pyhgf minor bumps may change `temp` key names)

---

## Addendum: User Decisions (2026-04-17)

After presenting A/B/C scope options, the user selected **Option A (Minimum Viable Phase 18)** with these clarifications:

### Scope confirmation

> "option A, but the HGF is already made with another repo as well as the DCM. refer to `C:\Users\aman0087\Documents\Github\psilocybin_prl_analyses` and `C:\Users\aman0087\Documents\Github\dcm_pytorch` for their references. it's just about integrating them."

Interpretation: the HGF side lives in THIS repo (psilocybin_prl_analyses); the DCM side lives in `dcm_pytorch`. Phase 18 is an **integration phase** where this repo produces the per-trial HGF trajectory artifact that dcm_pytorch will eventually consume as bilinear-DCM modulatory inputs. This bumps PRL.4 (trajectory export) INTO scope for the minimum viable phase, since it IS the integration surface. The export format should be chosen to plug directly into dcm_pytorch's bilinear DCM modulator interface.

### dcm_pytorch context (cross-referenced)

- Current milestone v0.3.0 = bilinear DCM: `dx/dt = Ax + Σ_j u_j·B_j·x + Cu`
- HEART2ADAPT-specific work (PEB-lite group GLM / DCM.5, 4-node AMY↔dACC↔vmPFC↔Insula circuit / DCM.V3) is **explicitly deferred** from dcm_pytorch v0.3.0 to v0.4+
- `src/pyro_dcm/simulators/task_simulator.py` and `src/pyro_dcm/forward_models/neural_state.py` are the consumer interfaces our trajectory CSV must align with
- dcm_pytorch PROJECT.md line 67: "Group-level PEB-lite GLM (HEART2ADAPT-specific; not scoped to this single-subject toolbox)"

**Integration contract for Phase 18 producer side**: one CSV per subject containing per-trial columns suitable for use as `u_j(t)` modulatory-input traces in dcm_pytorch's bilinear DCM. Specifically: at minimum `trial_idx`, `run_idx`, `outcome_time_s`, and the HGF-derived precision-weighted prediction errors (ε₂, ε₃), expected value, and posterior means (μ₂, μ₃). The ΔHR covariate is passed through as a column (caller-supplied input). Choice and outcome are included for downstream reference.

### ΔHR clarification (user asked)

User question: "what is delta HR (heart rate?)"

Yes — ΔHR = per-trial change in heart rate, specifically **anticipatory cardiac deceleration (bradycardia)** measured during the ~5.5s anticipation window after a context cue in the PAT-RL task. This is the classic "fear bradycardia" / defensive-cascade orienting response (Lang, Bradley & Cuthbert 1997; Löw, Lang, Smith & Bradley 2008; Hamm & Weike 2005). Computed as mean HR during the cue window minus a pre-cue baseline (1–3s), yielding a negative ΔHR under threat relative to safe.

**Typical magnitude:** −2 to −6 bpm (threat minus safe) in healthy adults, within-subject SD ~3–5 bpm. Reasonable prior for simulation: `delta_hr ~ N(−3, 3)` on threat trials, `N(0, 3)` on safe trials. Hard bounds [−15, +10] bpm for outliers.

**Anxiety modulation:** high-anxiety agents typically show ~1.5–2× greater deceleration under cued predictable threat (Hamm & Weike 2005; McTeague & Lang 2012). Under unpredictable/sustained threat the pattern can reverse — PAT-RL uses cued anticipation, so the cued-threat prior applies.

**Phase 18 treatment:** per the user's guidance "rely on the other toolbox to handle pyHGF" and the minimum-viable scope, ΔHR is a **caller-supplied per-trial input column** in Phase 18, not a generative quantity. The trial generator should accept a `delta_hr` array or produce one from a simple literature-grounded stub (N(−3, 3) threat / N(0, 3) safe). A proper anxiety-modulated autonomic generative model is out of scope for Phase 18 and deferred to the follow-up phase that introduces Models B/C/D.

### Phenotype 2×2 parameter values (literature-grounded)

User asked me to "do research to inform the parameters." Literature findings:

**Direction of anxiety effect on ω (ω₂):** higher ω (less negative) = faster updating = anxiety. Browning, Behrens, Jocham, O'Reilly & Bishop (2015, *Nat Neurosci*) shows trait-anxious subjects fail to reduce learning rate in stable blocks, effectively inferring higher volatility. Replicated by Aylward et al. (2019, *Nat Hum Behav*) for GAD patients, and Huang, Thompson & Paulus (2017).

**Suggested simulation priors for the 2×2 phenotype grid (PRL.V2):**

| Phenotype dimension | Low | High | Source |
|---|---|---|---|
| Anxiety — ω₂ | −6.0 | −4.0 to −3.0 | Browning 2015; Aylward 2019 |
| Reward sensitivity — β | 2.0 | 6.0 to 10.0 | Daw 2006; Schönberg 2007; Gershman 2016 |
| κ (3-level coupling) | fix at 1.0 | fix at 1.0 | Weak evidence in anxiety; fix across phenotypes (de Berker 2016 insufficient) |

**Known confound to watch:** ω × β are modestly negatively correlated (r ≈ −0.2 to −0.4) in published fits (Wise & Dolan 2020; Reiter et al. 2021). To keep the 2×2 grid cleanly separable in simulation, use a wide β gap (β_low=2, β_high=8) and moderate ω gap (Δω ≈ 2–3).

**Scope warning:** Browning's anxiety-by-volatility effect REQUIRES multiple volatility regimes in the task. PAT-RL has stable (hazard=0.03) and volatile (hazard=0.10) runs — this provides the regime contrast the signature depends on. Do NOT claim anxiety will separate on ω without the counterbalanced stable/volatile run structure actually being simulated in the generator.

### Open questions RESOLVED by this addendum

1. ΔHR generative model → caller-supplied; literature stub `N(−3, 3)` threat / `N(0, 3)` safe for Phase 18 smoke only
2. Phenotype 2×2 numeric values → table above (Browning/Daw/Schönberg grounded)
3. pyhgf version concerns → "rely on the other toolbox"; use installed 0.2.10 as-is, no version-bump work in Phase 18

### Open questions REMAINING (planner should mark as deferred)

1. **PEB covariate format (ΔWAIC vs ΔF)** — user explicitly deferred ("i'm not sure about peb covariate format"). Mark as a Phase 19+ decision. Phase 18 should NOT implement per-subject evidence-difference export.
2. **Anxiety-modulated ΔHR generative model** — out of scope for Phase 18; belongs with Models B/C/D in the follow-up phase
3. **Model D trial-varying ω stability / λ clamp ranges** — out of scope
4. **Exact column set + units for trajectory CSV** — planner must check dcm_pytorch's `task_simulator.py` / bilinear modulator interface for the exact schema

### Refined Option A scope (committed)

Foundation + integration surface:

1. **PRL.1**: `configs/pat_rl.yaml` + `src/prl_hgf/env/pat_rl_config.py` (parallel dataclass tree; zero edits to `task_config.py`)
2. **PRL.2**: `src/prl_hgf/env/pat_rl_sequence.py` (binary-state hazard generator + 2×2 magnitudes + ΔHR stub + 192-trial structure)
3. **PRL.3 Model A only**: binary-state 2-level and 3-level HGF builders (`models/hgf_2level_patrl.py`, `models/hgf_3level_patrl.py`) + Model A softmax-on-EV response (`models/response_patrl.py`); Models B/C/D deferred
4. **PRL.4 (integration surface)**: `src/prl_hgf/analysis/export_trajectories.py` — per-subject CSV with HGF-derived quantities (μ₂, σ₂, μ₃, σ₃, ε₂, ε₃, δ₁, ψ₂, choice, outcome, ΔHR pass-through) using post-hoc forward pass at posterior means via the existing scanner factory pattern
5. **Recovery smoke test** (lightweight, NOT full PRL.V1 gate): 5-participant simulate→fit→recover cycle at 192 trials on CPU, confirming ω₂/β/κ are all finite and roughly on the right side of chance. Full r≥0.7 acceptance gate deferred to follow-up phase.

**Deferred to follow-up phase (tentative Phase 19+):**
- Models B (ΔHR bias), C (ΔHR × value sensitivity), D (trial-varying ω)
- Full PRL.V1 recovery acceptance gate at r ≥ 0.7
- PRL.V2 phenotype 2×2 identifiability validation
- PRL.5 stratified BMS with ΔWAIC PEB covariate export

**Milestone placement:** Phase 18 stays appended to v1.2 per user instruction; the follow-up integer phases will naturally open v1.3 HEART2ADAPT if the user later elects to split.
