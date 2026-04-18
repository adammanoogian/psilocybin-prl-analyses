# Roadmap: PRL HGF Analysis Pipeline

## Milestones

- **v1.0 Simulation-to-Inference Pipeline** — Phases 1-7 (shipped 2026-04-07)
- **v1.1 BFDA Power Analysis** — Phases 8-11 (code-complete 2026-04-07)
- **v1.2 Hierarchical GPU Fitting** — Phases 12-20 (in progress; Phase 18 HEART2ADAPT adaptation + Phase 19 VB-Laplace parity fit path + Phase 20 HEART2ADAPT scientific completion appended — see notes)

---

<details>
<summary>v1.0 Simulation-to-Inference Pipeline (Phases 1-7) — SHIPPED 2026-04-07</summary>

See `.planning/milestones/v1.0-ROADMAP.md` for full detail.

| Phase | Goal | Status |
|-------|------|--------|
| 1 - Foundation | Project skeleton, config, task environment | Complete 2026-04-04 |
| 2 - Models | 2-level and 3-level HGF model definitions | Complete 2026-04-05 |
| 3 - Simulation | Batch synthetic participant generation | Complete 2026-04-05 |
| 4 - Fitting | PyMC MCMC fitting pipeline | Complete 2026-04-05 |
| 5 - Validation | Parameter recovery + model comparison | Complete 2026-04-06 |
| 6 - Group Analysis | Mixed-effects hypothesis testing + manuscript | Complete 2026-04-06 |
| 7 - GUI | Interactive parameter explorer | Complete 2026-04-07 |

</details>

---

## v1.1 BFDA Power Analysis (Phases 8-11)

**Milestone Goal:** Determine required sample size and trial count for detecting psilocybin x session interactions on HGF parameters via simulation-based Bayes Factor Design Analysis. Produces a publication-ready 4-panel figure and a concrete N/group recommendation for grant submission.

### Phase 8: Config and Infrastructure

**Goal**: The power analysis infrastructure exists — a config factory builds frozen per-job configs without touching YAML, SeedSequence guarantees independent RNG streams across SLURM array tasks, and every iteration writes a tidy parquet row with a validated schema.
**Depends on**: Phase 7 (existing pipeline functions: simulate_batch, fit_batch, build_estimates_wide)
**Requirements**: PWR-01, PWR-09, PWR-10, SEED-01
**Success Criteria** (what must be TRUE):
  1. `make_power_config(base_config, n_per_group, effect_size_delta, master_seed)` returns a frozen config with correct `n_participants_per_group` and `session_deltas` overrides without writing to or reading from the YAML file
  2. A 10-job SLURM array smoke test runs with `--array=0-9%50`, each task writes its own parquet file with no naming collisions and no import-storm failures
  3. Each parquet output contains the full required schema: sweep_type, effect_size, n_per_group, trial_count, iteration, parameter, bf_value, bf_exceeds, bms_xp, bms_correct, recovery_r, n_divergences, mean_rhat
  4. Child seeds from `SeedSequence(seed_base).spawn(n_jobs)` produce uncorrelated RNG streams — verified by asserting no two child seeds share a state vector
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md — power/config.py: make_power_config factory, PowerConfig dataclass, YAML power section, unit tests
- [x] 08-02-PLAN.md — power/schema.py + power/seeds.py + SLURM array template + entry point script + infrastructure tests

### Phase 9: Prechecks

**Goal**: The set of power-eligible parameters is established before any sweep runs — recovery r >= 0.7 is confirmed for at least omega_2 and beta, the minimum adequate trial count is identified, and no MCMC convergence-failing fits contaminate downstream power estimates.
**Depends on**: Phase 8 (make_power_config needed to vary trial counts)
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04, PRE-05, PRE-06
**Success Criteria** (what must be TRUE):
  1. A precheck run on 50 simulated participants produces a recovery_r table per parameter; omega_2 and beta both show r >= 0.7 and advance to power-eligible status
  2. omega_3 results appear in the output labeled "exploratory — upper bound" rather than being silently excluded or promoted to primary
  3. The trial count sweep figure (r vs trial count, one line per parameter, reference line at r = 0.7) identifies the minimum trial count where all power-eligible parameters exceed threshold
  4. A confound matrix flagging |r| > 0.8 pairwise correlations is produced alongside the eligibility table
  5. Fits with R-hat > 1.05 or ESS < 400 are excluded from recovery calculations with the exclusion count reported per condition
**Plans**: 2 plans

Plans:
- [x] 09-01-PLAN.md — power/precheck.py: make_trial_config, run_recovery_precheck, build_eligibility_table, convergence gating, pipeline script + 9 unit tests
- [x] 09-02-PLAN.md — Trial count sweep (run_trial_sweep, plot_trial_sweep, find_minimum_trial_count), VIZ-01 precheck figure, --sweep CLI flag + 7 unit tests

### Phase 10: Core Power Modules and Sweep

**Goal**: The full N x effect_size power sweep runs on the cluster — JZS Bayes Factors are computed on posterior interaction contrasts, BMS discriminability is measured across N levels, and all per-job results land in data/power/results/ as parquet files.
**Depends on**: Phase 9 (power-eligible parameter list and minimum trial count)
**Requirements**: PWR-02, PWR-03, PWR-04, PWR-05, PWR-06, PWR-07, PWR-08
**Success Criteria** (what must be TRUE):
  1. `compute_jzs_bf(posterior_draws)` using `pingouin.bayesfactor_ttest` with r = sqrt(2)/2 returns a finite BF for synthetic contrast draws and matches a reference JASP calculation to within 1%
  2. The primary contrast delta_i = (theta_psi_post - theta_psi_baseline) - (theta_plc_post - theta_plc_baseline) is computed per iteration alongside baseline->followup and linear trend contrasts, with the primary contrast documented in config
  3. The full N x effect_size grid (n_per_group in [10,15,20,25,30,40,50], d in [0.3,0.5,0.7], K=200 iterations) produces parquet files for Power Analysis A with fewer than 5% of cells having exclusion rate > 5%
  4. Power Analysis B generates data from the 3-level model, fits both models, runs random-effects BMS, and records P(XP_true > 0.75) at each N level
  5. Group-stratified BMS (PWR-08) results are present in the output and clearly labeled as optional/exploratory
**Plans**: 3 plans

Plans:
- [x] 10-01-PLAN.md — power/contrasts.py: JZS BF via pingouin, DiD and linear trend contrasts, 10 unit tests (PWR-03, PWR-05)
- [x] 10-02-PLAN.md — YAML power grid update (d={0.3,0.5,0.7}, K=200), power/grid.py decode_task_id, SLURM update, reduced MCMC CLI flags (PWR-04, PWR-06)
- [x] 10-03-PLAN.md — power/iteration.py: run_power_iteration full pipeline, entry point wiring, BMS power path, 5 unit tests (PWR-02, PWR-07, PWR-08)

### Phase 11: Aggregation, Figures, and Recommendation

**Goal**: All sweep results are aggregated into a single master parquet, publication-quality figures are produced, and a concrete N/group recommendation with supporting evidence is written in results/power/.
**Depends on**: Phase 10 (per-job parquet files in data/power/results/)
**Requirements**: VIZ-01, VIZ-02, VIZ-03, VIZ-04, REC-01
**Success Criteria** (what must be TRUE):
  1. `scripts/09_aggregate_power.py` concatenates all per-job parquets into power_master.csv and warns on any missing iteration cells
  2. The Power A figure shows P(BF > 6) vs N with one curve per effect size, reference lines at 80% and 90%, and an annotation marking the N where d = 0.5 crosses 80%
  3. The Power B figure shows P(correct BMS) vs N with a reference line at 75%
  4. The 4-panel publication figure (precheck recovery + Power A + Power B + sensitivity heatmap) is saved as both PDF and PNG and is self-contained for grant/preregistration use
  5. results/power/recommendation.md states a concrete recommended N/group and trial count with supporting BF evidence, exclusion rate summary, and the omega_3 upper-bound caveat
**Plans**: 3 plans

Plans:
- [x] 11-01-PLAN.md — power/curves.py + scripts/09_aggregate_power.py: aggregation, P(BF>threshold) & P(BMS correct) computation, 8+ unit tests
- [x] 11-02-PLAN.md — scripts/10_plot_power_curves.py: Power A, Power B, sensitivity heatmap, 4-panel publication figure (PDF+PNG), 6+ tests
- [x] 11-03-PLAN.md — scripts/11_write_recommendation.py: recommendation.md with N/group, trial count, evidence table, caveats, 7+ tests

---

## v1.2 Hierarchical GPU Fitting (Phases 12-15)

**Milestone Goal:** Refactor the v1.1 power analysis fitting pipeline into a batched hierarchical architecture so GPU acceleration is usable (amortizing NUTS launch overhead across all participants in one `sample_numpyro_nuts` call), then execute the full v1.1 production sweep on real compute.

**Why this exists:** The v1.1 L40S benchmark revealed ~1.5s per NUTS sample due to CPU↔GPU dispatch dominating the 420-trial sequential scan. Per-participant sequential fitting is the wrong architecture for GPU. The compute projection is ~18,000 GPU-hours for the full sweep — infeasible. v1.2 fixes this architecturally and completes v1.1's production deliverables.

### Phase 12: Batched Hierarchical JAX Logp

**Goal**: A batched JAX logp function exists that accepts `(n_participants, ...)` shaped parameters and data, returns a scalar summed logp via `jax.vmap`, includes tapas-style Layer 2 per-trial NaN clamping, and is wrapped in a PyMC model that runs one `pmjax.sample_numpyro_nuts` call for the entire cohort. Mathematically equivalent (bit-exact at `n_participants=1`, within-MCSE at `n_participants=5`) to the legacy per-participant path, verified on CPU.
**Depends on**: Phase 11 (existing pyhgf Network + scan_fn extraction pattern in `ops.py`)
**Requirements**: BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05, BATCH-06, BATCH-07, VALID-01, VALID-02
**Success Criteria** (what must be TRUE):
  1. `build_logp_ops_batched(input_data_arr, observed_arr, choices_arr)` accepts arrays with a leading participant dimension `(P, n_trials, 3)` and returns a PyTensor Op whose forward pass runs `jax.vmap(single_participant_logp)(params_batch, data_batch)` and sums across P
  2. Inside the batched logp, `lax.scan` steps include a Layer 2 check: `is_stable = jnp.all(jnp.isfinite(new_attrs_flat))`, then `safe_attrs = jax.tree_util.tree_map(lambda n, o: jnp.where(is_stable, n, o), new, old)` — unstable trials revert to previous state and contribute 0 to logp via a per-trial mask
  3. `fit_batch_hierarchical(sim_df, model_name, ...)` builds a hierarchical `pm.Model()` with `shape=(n_participants,)` priors, runs one `pmjax.sample_numpyro_nuts` call, returns a single `InferenceData` with a participant dim on every parameter
  4. VALID-01: batched logp with `P=1` returns float64-identical value to the existing per-participant logp for matched inputs (regression test)
  5. VALID-02: fit 5 participants sequentially (legacy) and 5 batched (new), both on CPU with matched seeds; per-parameter posterior means agree within 3× MCSE
**Plans**: 4 plans

Plans:
- [x] 12-01-legacy-migration-PLAN.md — Move fitting/single.py + fitting/batch.py into legacy/, add backward-compat shims, leave ops.py + models.py in place
- [x] 12-02-batched-jax-logp-PLAN.md — Create hierarchical.py with build_logp_ops_batched (jax.vmap over participants, Layer 2 clamping, trial_mask, two-Op split, jax_funcify)
- [x] 12-03-hierarchical-pymc-orchestrator-PLAN.md — Add build_pymc_model_batched + fit_batch_hierarchical (one pmjax.sample_numpyro_nuts call), wire __init__.py exports
- [x] 12-04-validation-tests-PLAN.md — tests/test_hierarchical_logp.py: VALID-01 bit-exact P=1, P=2 doubling, Layer 2 clamping smoke, VALID-02 5-participant within-MCSE

### Phase 13: JAX-Native Cohort Simulation

**Goal**: A JAX-native simulation path exists that runs a full 420-trial session via `lax.scan` using pyhgf's `net.scan_fn` (no HGF math rewrite), applies the same tapas-style Layer 2 clamping, and `jax.vmap`s across participants to simulate a full cohort in one compiled kernel. Produces statistically equivalent output to the legacy NumPy `simulate_agent` loop.
**Depends on**: Phase 12 (tapas-Layer-2 clamping pattern established)
**Requirements**: JSIM-01, JSIM-02, JSIM-03, JSIM-04, JSIM-05, JSIM-06, VALID-04
**Success Criteria** (what must be TRUE):
  1. `simulate_session_jax(params, trial_inputs, rng_key)` runs one session entirely inside `lax.scan` with pyhgf's `scan_fn` for HGF updates and `jax.random` for choice + reward sampling
  2. `simulate_cohort_jax(params_batch, trial_inputs, rng_keys_batch)` `jax.vmap`s the session function across participants
  3. `simulate_batch` uses the new path internally, preserves the DataFrame schema (including the `diverged` column from JSIM-05)
  4. RNG determinism: fixing the master seed reproduces the same cohort across runs (and across devices, up to floating-point noise)
  5. VALID-04: over 100 replicates with matched master seeds, `simulate_agent` and `simulate_session_jax` produce statistically equivalent choice frequency distributions per cue per phase (KS test p > 0.05 or equivalent)
**Plans**: 3 plans

Plans:
- [x] 13-01-PLAN.md — simulate_session_jax: lax.scan session simulator with factory pattern, Layer 2 clamping, PRNG key threading, 5 unit tests
- [x] 13-02-PLAN.md — simulate_cohort_jax vmap wrapper, simulate_batch JAX rewrite, __init__.py exports, 4 cohort tests
- [x] 13-03-PLAN.md — VALID-04 statistical equivalence: 100 replicates KS test (legacy vs JAX), 2 tests

### Phase 14: Integration + GPU Benchmark + Decision Gate

**Goal**: The new batched fit and JAX simulation are wired into `run_sbf_iteration`. A GPU benchmark job runs one full iteration (300 participant-sessions × 2 models) on a real GPU node, measures per-iteration wall time + VRAM + utilization, and drives a decision gate: if projected total GPU-hours per chunk exceed the walltime budget, fall back to the CPU `comp` partition (the new batched code still wins over v1.1 sequential on CPU). Cross-platform consistency validated.
**Depends on**: Phases 12 and 13
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, VALID-03, VALID-05
**Success Criteria** (what must be TRUE):
  1. `run_sbf_iteration` uses `fit_batch_hierarchical` + `simulate_cohort_jax` by default; `--legacy` flag preserved for reproducibility (VALID-05)
  2. `08_run_power_iteration.py --benchmark` produces `results/power/benchmark_batched.json` with per-iteration time, peak VRAM, and mean GPU utilization during fitting
  3. Decision rule applied: if `per_iter_seconds × 600 / 3600 > 50`, recommendation = CPU `comp` partition; else = GPU. Decision recorded in benchmark JSON and STATE.md
  4. JAX compilation cache verified: a second chunk started after a first chunk completes shows < 5 s of JIT time (vs ~60 s cold), confirming `JAX_COMPILATION_CACHE_DIR` persistence
  5. VALID-03: same small fit run on CPU and GPU — posterior means agree within 1 % relative error
**Plans**: 3 plans + 6 gap-closure plans (14.1)

Plans:
- [x] 14-01-PLAN.md — Wire run_sbf_iteration to use fit_batch_hierarchical by default, _idata_to_fit_df + _split_idata helpers, --legacy flag (BENCH-03, VALID-05)
- [x] 14-02-PLAN.md — Rewrite _run_benchmark for full batched iteration, _GpuMonitor, decision gate, JAX cache test (BENCH-01, BENCH-02, BENCH-04, BENCH-05)
- [x] 14-03-PLAN.md — VALID-03 cross-platform consistency validation script + comparison tests

Gap-closure (14.1): code complete post-verification, operational gaps remain (cluster runs never executed; sampler drift post-Phase 17)
- [ ] 14.1-01-PLAN.md — SLURM auto-push + SAMPLER env var on cluster/14_benchmark_gpu.slurm
- [ ] 14.1-02-PLAN.md — Triage in-flight numpyro benchmark, backfill STATE.md decision-gate row if needed
- [ ] 14.1-03-PLAN.md — BlackJAX benchmark re-run for Phase 15 production sampler gate
- [ ] 14.1-04-PLAN.md — VALID-03 CPU vs GPU run (auto-push patch + submit + compare verdict)
- [ ] 14.1-05-PLAN.md — Cross-chunk JIT cache persistence test (BENCH-05 human-verify #2)
- [ ] 14.1-06-PLAN.md — Phase 14 re-verification; flip VERIFICATION.md status; unblock Phase 15

### Phase 15: Production Run + Results

**Goal**: The full SBF power sweep runs on the platform chosen in Phase 14. All 600 tasks complete, aggregate into `power_master.csv` without missing cells, and produce the real v1.1 deliverables: the 4-panel publication figure, `recommendation.md` with concrete N/group backed by real BF evidence, and an auto-pushed results branch.
**Depends on**: Phase 14 (integration complete, platform decision made)
**Requirements**: PROD-01, PROD-02, PROD-03, PROD-04, PROD-05
**Success Criteria** (what must be TRUE):
  1. Full sweep (3 chunks × 200 iterations) completes on the chosen platform; SLURM logs show no divergent-chain warnings above ~5 % exclusion rate per cell
  2. `scripts/09_aggregate_power.py` produces `power_master.csv` with 12,600 rows (600 iterations × 3 sweep_types × 7 N-levels) and no missing-cell warnings
  3. `scripts/10_plot_power_curves.py` regenerates the 4-panel figure as PDF + PNG with real data, no placeholders
  4. `scripts/11_write_recommendation.py` writes `recommendation.md` containing a concrete recommended N per group, chosen trial count, 80 %-power crossing for d = 0.5, exclusion rate summary, and the omega_3 upper-bound caveat
  5. Wave 3 push (`99_push_results.slurm`) triggered via `afterany:${POSTPROC_JOBID}` and commits `results/power/` to git on completion
**Plans**: 2 plans

Plans:
- [ ] 15-01-PLAN.md -- Pre-flight fixes: dynamic date, push script staging, platform selection
- [ ] 15-02-PLAN.md -- Production sweep submission, monitoring, and success criteria verification

### Phase 16: NumPyro Direct Sampling + CUDA Fix

**Goal**: Replace PyMC's `pmjax.sample_numpyro_nuts()` with direct numpyro MCMC throughout the fitting pipeline, enabling JIT cache reuse across iterations (data as traced arguments, not closure constants). Fix the CUDA driver/PTX mismatch that disables XLA parallel compilation. Validate with JIT timing baselines and a CUDA environment check.
**Depends on**: Phase 15 (production run complete with current architecture; Phase 16 is a performance refactor)
**Requirements**: NPRO-01, NPRO-02, NPRO-03, NPRO-04, NPRO-05, NPRO-06
**Success Criteria** (what must be TRUE):
  1. `fit_batch_hierarchical` uses numpyro MCMC directly (no PyMC model construction, no `pmjax.sample_numpyro_nuts`); priors defined in numpyro, data passed as argument to `MCMC.run()` so JIT cache reuses across iterations with different data
  2. JIT compilation cache verified: second call to `fit_batch_hierarchical` with same shapes but different data shows <5s JIT overhead (vs ~800s cold compile currently)
  3. CUDA environment check: a startup diagnostic in SLURM scripts verifies PTX compiler version <= driver CUDA version; warns and suggests fix if mismatched
  4. `nvidia-cuda-nvcc-cu12` pinned to 12.8.x in cluster environment, XLA parallel compilation re-enabled (no "disabling parallel compilation" warning in logs)
  5. Existing VALID-01/02/03 tests pass with the numpyro-direct path (posterior equivalence within MCSE)
  6. Benchmark smoke test logs JIT cold/warm times and reports cache hit/miss status
**Plans**: 2 plans

Plans:
- [x] 16-01-PLAN.md -- Core numpyro refactor: build_logp_fn_batched, numpyro model functions, rewrite fit_batch_hierarchical (NPRO-01, NPRO-02)
- [x] 16-02-PLAN.md -- Caller updates, CUDA environment check in SLURM scripts, validation tests (NPRO-03, NPRO-04, NPRO-05, NPRO-06)

### Phase 17: BlackJAX NUTS Sampler

**Goal**: Replace NumPyro MCMC with BlackJAX NUTS for the fitting pipeline, eliminating the ~1800s per-call JIT recompilation caused by NumPyro's internal function-object recreation. BlackJAX compiles the NUTS kernel once via `jax.jit`, then reuses it across all MCMC steps and power-sweep iterations. Pure JAX log-posterior (priors + batched HGF logp) replaces the numpyro.sample/factor pattern. pyhgf `scan_fn` usage unchanged. Multi-GPU chain parallelism via `jax.pmap` restored (safe with BlackJAX since it avoids the NumPyro pmap/psum codepath that triggers the L40S bug JAX #31626).
**Depends on**: Phase 16 (numpyro-direct path established; batched logp function exists)
**Requirements**: BJAX-01, BJAX-02, BJAX-03, BJAX-04, BJAX-05, BJAX-06, BJAX-07
**Success Criteria** (what must be TRUE):
  1. `fit_batch_hierarchical` uses `blackjax.nuts` with `blackjax.window_adaptation` for warmup; compiles the NUTS step function exactly once per shape via `jax.jit`
  2. Cold JIT < 120s on L40S GPU (vs ~1800s with NumPyro); warm JIT < 5s (compiled kernel reuse across power-sweep iterations with same shapes)
  3. Pure JAX log-posterior function combines truncated-normal/normal priors with `batched_logp_fn`; no NumPyro MCMC dependency in the fitting path (numpyro.distributions used standalone for prior log-probs)
  4. Multi-GPU support: `jax.pmap` across available GPUs for chain parallelism (1 chain per device); falls back to `jax.vmap` on single GPU
  5. ArviZ `InferenceData` output with participant coords preserved; downstream analysis scripts unchanged
  6. VALID-01/02 pass with BlackJAX path (posterior equivalence within MCSE vs numpyro-direct baseline); VALID-03 deferred to Phase 14
  7. SLURM scripts updated; smoke test passes all 3 gates (cold JIT < 600s, cache speedup > 3x, warm JIT < 120s)
**Plans**: 2 plans

Plans:
- [x] 17-01-PLAN.md -- Core BlackJAX implementation: _build_log_posterior, _run_blackjax_nuts, _samples_to_idata, rewrite fit_batch_hierarchical (BJAX-01, BJAX-02, BJAX-03, BJAX-04, BJAX-05)
- [x] 17-02-PLAN.md -- Validation tests for BlackJAX path, SLURM script updates for multi-GPU pmap (BJAX-06, BJAX-07)

### Phase 18: PAT-RL Task Adaptation (HEART2ADAPT)

**Goal**: The PRL HGF toolbox supports the PAT-RL task (binary safe/dangerous state, approach/avoid decisions, 2x2 reward/shock magnitude design, trial-level Delta-HR autonomic covariate, hazard-based reversals, 192 trials across 4 runs) as a **parallel** task configuration alongside the existing pick_best_cue pipeline. Delivers: (a) new YAML + config loader path, (b) binary-state trial sequence generator, (c) extended response models A/B/C/D (including trial-varying omega for Model D), (d) trial-by-trial posterior trajectory export for the heart2adapt-sim DCM bridge, (e) 2-level vs 3-level BMS stratified by phenotype with Delta-evidence as a PEB covariate.
**Depends on**: Phase 17 (BlackJAX fitting pipeline is the target fit path for new response models)
**Requirements**: PRL-01 (new config), PRL-02 (trial generator), PRL-03 (response models A-D), PRL-04 (trajectory export), PRL-05 (stratified BMS), PRL-V1 (recovery at 192 trials), PRL-V2 (phenotype separability)
**Success Criteria** (what must be TRUE):
  1. `configs/pat_rl.yaml` exists and `load_config("pat_rl")` (or equivalent dispatch) returns a validated dataclass tree for the binary-state schema without mutating or re-parsing `prl_analysis.yaml`; existing pick_best_cue callers are unaffected (regression test passes)
  2. A PAT-RL trial sequence module generates a 192-trial run structure with hazard-rate-driven state transitions (stable=0.03, volatile=0.10), 2x2 reward/shock magnitude assignment, and Delta-HR input columns wired alongside the HGF input `u` and observed choice `y`
  3. Response models A (softmax on EV), B (Delta-HR bias `gamma`), C (Delta-HR x value sensitivity `alpha`), and D (trial-varying `omega_eff(t) = omega + lambda * Delta-HR(t)`) each fit cleanly through `fit_batch_hierarchical` on a 5-participant CPU smoke; Model D requires a trial-varying-omega code path in both the JAX logp and the pyhgf Network (new work, not drop-in)
  4. `analysis/export_trajectories.py` emits a per-subject CSV with the full columns listed in the YAML (`mu2, sigma2, mu3, sigma3, epsilon2, epsilon3, delta1, psi2, delta_hr, outcome_time_s`, etc.) plus a per-subject parameter summary CSV for PEB covariates
  5. Random-effects BMS (using existing `analysis/bms.py`) comparing 2-level vs 3-level HGF on simulated PAT-RL data returns a posterior model probability + exceedance probability, stratified by phenotype group; Delta-evidence per subject is written to the PEB covariate export
  6. **PRL-V1**: parameter recovery at 192 trials meets r >= 0.7 for `omega_2`, `kappa`, `beta`; `mu3_0` and `omega_3` remain exploratory
  7. **PRL-V2**: the 2x2 phenotype grid (anxiety x reward sensitivity) is identifiable — `omega` separates anxiety (d >= 0.5), `beta` separates reward sensitivity (d >= 0.5), and `cor(omega, beta) < 0.5` across simulated agents
**Plans**: 6 plans

**Option A Minimum Viable scope** (user-confirmed 2026-04-17; see `.planning/phases/18-pat-rl-task-adaptation/18-RESEARCH.md` Addendum). Models B/C/D, full PRL-V1 r>=0.7 gate, PRL-V2 phenotype identifiability, and PRL.5 stratified BMS / PEB covariate export are explicitly deferred to Phase 19+. Phase 18 ships the producer side of the dcm_pytorch integration: config + trial generator + binary-state HGF builders + Model A response + BlackJAX fit + trajectory export + 5-agent CPU smoke.

Plans:
- [x] 18-01-PLAN.md — configs/pat_rl.yaml + env/pat_rl_config.py (parallel PATRLConfig dataclass tree + loader + unit tests)
- [x] 18-02-PLAN.md — env/pat_rl_sequence.py (binary-state hazard generator, 2x2 magnitudes, Delta-HR stub, 192-trial structure) + tests
- [x] 18-03-PLAN.md — models/hgf_2level_patrl.py + hgf_3level_patrl.py (single-input-node HGF) + models/response_patrl.py (Model A softmax on EV) + tests
- [x] 18-04-PLAN.md — fitting/hierarchical_patrl.py (batched logp + BlackJAX orchestrator reusing generic helpers from hierarchical.py without modifying them) + 5-participant CPU smoke tests
- [x] 18-05-PLAN.md — analysis/export_trajectories.py (post-hoc forward pass at posterior means, per-trial CSV + parameter summary) with dcm_pytorch consumer-interface verification
- [x] 18-06-PLAN.md — scripts/12_smoke_patrl_foundation.py end-to-end simulate -> fit -> export + integration test

**Integration notes (YAML assumptions flagged against existing code)**:
- **Config loader is NOT task-agnostic today.** `src/prl_hgf/env/task_config.py` hardcodes `_DEFAULT_CONFIG_PATH = CONFIGS_DIR / "prl_analysis.yaml"` and `AnalysisConfig`/`PhaseConfig` are shaped for 3 cues + 4 criterion-based phases. Adding `configs/pat_rl.yaml` alongside requires either (a) a task-dispatch layer on `load_config()` plus a parallel `PATRLConfig` dataclass tree, or (b) a subclass hierarchy. The YAML does not address this.
- **No `trial_sequence.py` exists.** The YAML's "adapts the existing `trial_sequence.py` pattern" is inaccurate — trial generation lives in `src/prl_hgf/env/simulator.py` (`Trial`, `generate_reward`, `generate_session`) and is tightly bound to the 3-cue partial-feedback protocol. PAT-RL needs a new binary-state generator module and a new `Trial`-like dataclass that carries `reward_mag`, `shock_mag`, `delta_hr`.
- **"Core HGF engine is unchanged" is only half-true.** `src/prl_hgf/models/hgf_2level.py` and `hgf_3level.py` define `INPUT_NODES = 3` (one binary node per cue for partial feedback). PAT-RL's binary-state HGF needs a single-input-node topology — that is a new Network build, even if pyhgf primitives are reused.
- **Response model signature mismatch.** Current `src/prl_hgf/models/response.py::softmax_stickiness_surprise` is a 3-way softmax with `(beta, zeta)`. Models A/B/C/D propose binary approach/avoid logits with `(beta, b, gamma, alpha, lambda)`. This is a new module (`response_models.py` or extension of `response.py`) — not a minor tweak. Model D in particular requires trial-varying `omega` wired into the scan body of `fitting/hierarchical.py::build_logp_fn_batched` AND into the JAX session simulator in `env/simulator.py`'s JAX path (Phase 13 work). That is a non-trivial extension of the logp contract.
- **Delta-HR as additional input has no current plumbing.** The batched logp in `fitting/hierarchical.py` takes `input_data_arr` shaped `(P, n_trials, 3)` (per-cue reward signals). PAT-RL adds a Delta-HR per-trial covariate. The shape contract, the scan signature, and `build_logp_ops_batched` all need an additional input axis; this cascades through VALID-01/02 regression tests.
- **Trajectory export (PRL.4) is wholly new.** No existing module evaluates the HGF forward pass at posterior means to produce per-trial `mu2, sigma2, mu3, sigma3, epsilon2, epsilon3, psi2`. The existing pipeline stores `InferenceData` from MCMC but does not re-run the perceptual model on each posterior draw. This is a substantive implementation task, not a data-dump.
- **Phenotype 2x2 design (PRL-V2) has no current home.** `configs/prl_analysis.yaml::simulation.groups` is psilocybin vs placebo (post-concussion). The YAML's anxiety x reward sensitivity 2x2 phenotype grid is an entirely new generative-parameter specification that requires either a new YAML key (`simulation.phenotypes`) or a separate `configs/pat_rl.yaml::simulation` block.
- **PEB covariate export (in PRL.5) has no current pipeline.** `analysis/bms.py` computes exceedance probabilities but does not emit a per-subject `Delta-WAIC` / `Delta-F` CSV for downstream PEB. New analysis step.
- **Milestone fit is ambiguous.** This phase introduces a **new task** for a **separate project (HEART2ADAPT)**, not a continuation of v1.2's "Hierarchical GPU Fitting" goal. Recommend considering whether Phase 18 should instead open a new milestone **v1.3 HEART2ADAPT** — particularly given the scope (config loader refactor + new env + new response models + trial-varying omega + trajectory export + phenotype framework) is comparable to several v1.1 phases combined. Added to v1.2 as instructed; flag for user review before planning.

### Phase 19: VB-Laplace Fit Path for PAT-RL (Tapas-Parity Validation)

**Goal**: A second, non-MCMC fit path exists for PAT-RL alongside BlackJAX NUTS: variational Bayes with Laplace approximation at the MAP. Mirrors the matlab tapas HGF toolbox convention (quasi-Newton optimization → numerical Hessian at the mode → Laplace posterior covariance). Lives in a new `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` that reuses the existing `_build_patrl_log_posterior` pure-JAX logp surface from `hierarchical_patrl.py` without modifying it. Returns an ArviZ `InferenceData` shape-compatible with the NUTS path so the existing `export_subject_trajectories` + `export_subject_parameters` from 18-05 accept both fit types unchanged. Unblocks downstream PEB development while cluster NUTS numbers come in; provides a deterministic reference fit to validate NUTS posteriors against.
**Depends on**: Phase 18 (consumes `_build_patrl_log_posterior` + PAT-RL HGF builders + config + trajectory export)
**Requirements**: VBL-01 (MAP optimizer), VBL-02 (Laplace covariance + PD regularization), VBL-03 (ArviZ InferenceData shape parity), VBL-04 (export-path compatibility), VBL-05 (parameter recovery smoke at 5 agents), VBL-06 (Laplace-vs-NUTS posterior comparison harness)
**Success Criteria** (what must be TRUE):
  1. `fit_vb_laplace_patrl(sim_df, model_name="hgf_{2,3}level_patrl", response_model="model_a", config=None)` runs a quasi-Newton MAP (jaxopt.LBFGS or scipy L-BFGS-B with jaxified gradient) on the existing `_build_patrl_log_posterior` logp, computes the Hessian at the mode via `jax.hessian`, and returns an `az.InferenceData` whose `posterior` group has the same parameter names, dims, and coords as `fit_batch_hierarchical_patrl` output (chain dim = 1, draw dim = K pseudo-draws from the Laplace Gaussian)
  2. When the Hessian at the mode is indefinite or ill-conditioned, a PD regularization fallback (add `lambda*I` until Cholesky succeeds, log the ridge added) produces a valid covariance matrix and does NOT silently return garbage
  3. `export_subject_trajectories` + `export_subject_parameters` from Plan 18-05 consume a Laplace-produced `InferenceData` unchanged and produce the same CSV schema (columns, dtypes, shapes) as for NUTS-produced `InferenceData`
  4. 5-agent CPU smoke completes in <60 seconds total (not per subject) on a dev laptop; parameter recovery sanity: posterior-mean omega_2 is within 0.5 of the generative truth for at least 4 of 5 agents
  5. Laplace-vs-NUTS comparison harness produces a diff table per subject per parameter: `|Δ posterior_mean| < 0.3` for omega_2 and `|Δ log_sd| < 0.5` across 5-agent smoke when both fit paths run on the identical sim_df
  6. Parallel-stack invariant preserved: `git diff` is empty for `hierarchical.py`, `hierarchical_patrl.py`, `task_config.py`, `simulator.py`, `hgf_{2,3}level.py`, `response.py`, `configs/prl_analysis.yaml`
**Plans**: 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 19 to break down)

**Sources of record / reference implementations**:
- `.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md` — Option C decision memo (dual NUTS + Laplace) with concrete tolerance gates and downgrade triggers
- matlab tapas HGF toolbox: `tapas_fitModel.m` (orchestrator), `tapas_quasinewton_optim.m` (MAP), `tapas_riddersmatrix.m` (numerical Hessian via Ridders' method)
- Target logp surface: `src/prl_hgf/fitting/hierarchical_patrl.py::_build_patrl_log_posterior` (PAT-RL pure-JAX log-posterior) — reused, not modified
- jaxopt.LBFGS docs + `jax.hessian` / `jax.hessian_on_rev` API for the JAX-side implementation

**Scope notes**:
- This phase IMPLEMENTS; it does NOT re-explore feasibility (that was quick-004). The feasibility memo already chose Option C and specified tolerance gates.
- Downgrade triggers live in STATE.md blockers; this phase executes under Option C unless the cluster smoke (Phase 18 validation) returns numbers that force a downgrade. If downgrade happens, the planner revisits but the parallel-stack module boundaries stay valid.
- A decision memo at phase close reports Laplace-vs-NUTS agreement on real cluster data and recommends whether to keep both paths or consolidate on one.

### Phase 20: HEART2ADAPT Scientific Completion — Models B/C/D + Cohort Scale + Config-Correctness

**Goal**: Close every remaining gap between `prl_hgf`'s current PAT-RL surface and the HEART2ADAPT study hypotheses documented in `dcm_hgf_mixed_models/docs/files/GSD_heart2adapt_sim.yaml`. Specifically: (a) correct `configs/pat_rl.yaml` to match the HEART2ADAPT spec (contingencies safe 70/10/20, dangerous 10/70/20, avoid 10/10/80; run order SVVS; magnitudes [1, 3]; phenotype priors ω=-3/-2, β=2/3.5, b=0/±0.3, ϑ=0.005/0.01); (b) add response-bias parameter `b` to Model A and implement Models B (ΔHR bias γ), C (ΔHR × value sensitivity α + γ), D (trial-varying ω_eff = ω + λ·ΔHR); (c) implement phenotype-specific, ε₂-coupled ΔHR generative model (healthy N(-2, 0.5), high-anxiety N(-0.5, 0.8), ε₂-modulated freezing); (d) scale cohort simulation to 40 agents × 4 phenotypes = 160 with deterministic per-phenotype RNG; (e) phenotype-stratified random-effects BMS in `analysis/bms.py` with per-subject Δ-evidence PEB covariate export; (f) formal PRL-V1 (ω/κ/β recovery r ≥ 0.7 at 192 trials) and PRL-V2 (phenotype 2x2 Cohen's d ≥ 0.5, cor(ω, β) < 0.5) gates. Unblocks `dcm_hgf_mixed_models` v2 bridge wiring (H2A.1.4, H2A.1.5, H2A.1.6 + H2A.2.4 PEB).
**Depends on**: Phase 19 (consumes `fit_vb_laplace_patrl`, `fit_batch_hierarchical_patrl`, trajectory export, Laplace InferenceData factory)
**Requirements**: PRL-02.1 (config correctness), PRL-03.1 (Model A + `b`), PRL-03.2 (Models B, C, D including trial-varying ω scan body for D), PRL-V1 (formal r ≥ 0.7 recovery at 192 trials), PRL-V2 (phenotype 2x2 identifiability), PRL-05 (phenotype-stratified BMS + PEB Δ-evidence export), PRL-06 (phenotype-specific ε₂-coupled ΔHR generative model), PRL-07 (cohort scale 40×4=160), PRL-08 (config-driven adaptation of existing scripts)
**Success Criteria** (what must be TRUE):
  1. `configs/pat_rl.yaml` contingencies match HEART2ADAPT spec **exactly**: `safe.p_reward_approach=0.70, p_shock_approach=0.10, p_nothing_approach=0.20`; `dangerous.10/0.70/0.20`; `avoid.0.10/0.10/0.80` (non-zero baseline avoid outcomes). Run order = `[stable, volatile, volatile, stable]` (SVVS counterbalance). Magnitudes `reward_levels=[1, 3], shock_levels=[1, 3]`. Phenotype priors: HEALTHY ω=-3.0, β=2.0, b=0.0, ϑ=0.005, μ₃⁰=1.0; REWARD-SUSCEPTIBLE ω=-3.0, β=3.5, b=+0.3, ϑ=0.005, μ₃⁰=1.0; HIGH-ANXIETY ω=-2.0, β=2.0, b=-0.3, ϑ=0.01, μ₃⁰=2.0; ANXIOUS+REWARD ω=-2.0, β=3.5, b=0.0, ϑ=0.01, μ₃⁰=2.0
  2. Response model A extended with bias `b`: `p(approach) = σ(β·EV + b)`. Model B adds ΔHR bias: `σ(β·EV + b + γ·ΔHR)`. Model C adds ΔHR-modulated value sensitivity: `σ((β + α·ΔHR)·EV + b + γ·ΔHR)`. Model D leaves response form as A but swaps perceptual ω for `ω_eff(t) = ω + λ·ΔHR(t)`. All four fit cleanly through BOTH `fit_batch_hierarchical_patrl` and `fit_vb_laplace_patrl` on the 5-agent CPU smoke
  3. Model D's trial-varying ω is implemented by injecting per-trial `ω_eff(t)` into the scan body of the batched logp (mirror the Phase 18-04 kappa-via-attrs pattern: `attrs[1]["tonic_volatility"] = ω + λ·ΔHR[t]` inside `_clamped_step`). Layer-2 clamp (`|μ₂| < 14`) and fp64 dtype invariants preserved. Recovery smoke of λ at 5 agents shows posterior-mean within 0.3 of truth when data simulated under Model D
  4. Phenotype-specific ΔHR generative model lives in `env/pat_rl_simulator.py`: `simulate_patrl_cohort` generates trial-level ΔHR as `base ~ N(phenotype.dhr_mean, phenotype.dhr_sd) + ε₂_coupling_coef · ε₂(t)` where `base` comes from the phenotype (healthy N(-2, 0.5); high-anxiety N(-0.5, 0.8); literature-calibrated — see citation gate in SC10). ε₂ computed inline from the forward HGF pass at true parameters
  5. Cohort scale: `simulate_patrl_cohort(n_per_phenotype=40, config, master_seed)` produces 40 × 4 = 160 agents with deterministic per-phenotype RNG (SeedSequence spawn-per-phenotype pattern). All 4 phenotype groups populated from `config.simulation.phenotypes`. Cluster SLURM default updated to `PRL_PATRL_SMOKE_N=40 PRL_PATRL_SMOKE_PHENOTYPES=all`
  6. Phenotype-stratified BMS: `analysis/bms.py::compute_stratified_bms(fit_df, phenotype_col="phenotype")` returns per-phenotype posterior model probabilities + exceedance probabilities for the 2-level-vs-3-level comparison. Per-subject ΔWAIC + ΔF exported via new `bms.py::export_peb_covariates(fit_df_2level, fit_df_3level, output_path)` — single CSV with columns `participant_id, phenotype, delta_waic, delta_f`
  7. Formal PRL-V1 recovery gate: `scripts/05_run_validation.py` extended (NOT a new script) with a `--task=patrl` config toggle that runs the PAT-RL recovery loop. Gate criterion: r(ω_2) ≥ 0.7, r(κ) ≥ 0.7, r(β) ≥ 0.7 across the 160-agent cohort; μ₃⁰ and ω₃ labeled "exploratory — upper bound" per 09-01 precedent
  8. Formal PRL-V2 phenotype-separability gate: `scripts/06_group_analysis.py` extended (NOT a new script) with `--task=patrl --analysis=phenotype_separability` that computes Cohen's d(ω | anxiety_high vs anxiety_low) ≥ 0.5, Cohen's d(β | reward_high vs reward_low) ≥ 0.5, and |cor(ω, β)| < 0.5 across the 160-agent cohort. Output: publication-quality figure + summary CSV
  9. **No new scripts created**: every analysis lives in an existing numbered script (`03_simulate_participants.py`, `04_fit_participants.py`, `05_run_validation.py`, `06_group_analysis.py`, `scripts/12_smoke_patrl_foundation.py`) or in the corresponding `src/prl_hgf/` module, made config-driven via `configs/pat_rl.yaml` flags. Verify via `git diff --stat scripts/` showing ONLY modifications (zero additions) to scripts/
  10. **Citation hygiene enforced**: every literature citation in config files, docstrings, and planning docs is dated 2020 or later. Prefer **Karin Roelofs' group** (Nijmegen Donders/Behavioural Science Institute — fear bradycardia, approach-avoidance conflict, threat anticipation cardiac deceleration): e.g. Klaassen et al. 2021 *Neuroimage* / 2024 *Biol Psychiatry*, Terburg et al. 2020+, Hulsman et al. 2020+, Ly et al. 2022+, Roelofs 2017/2020 reviews. Explicitly retire the Browning 2015 / Daw 2006 / Schönberg 2007 defaults from Phase 18's config where they conflict with HEART2ADAPT spec numbers (which supersede literature)
  11. Parallel-stack invariant preserved where possible; where config-driven adaptation necessarily extends existing modules (e.g. `models/response_patrl.py` adding `b`, B, C, D; `env/pat_rl_simulator.py` adding ε₂-coupled ΔHR; `analysis/bms.py` adding stratified variant), the extensions are ADDITIVE (new kwargs with safe defaults; existing Phase 18/19 tests remain green)
**Plans**: 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 20 to break down)

**Sources of record**:
- `C:/Users/aman0087/Documents/Github/dcm_hgf_mixed_models/docs/files/GSD_heart2adapt_sim.yaml` — HEART2ADAPT study-level hypotheses (§H2A.1.1–H2A.1.6 + §H2A.2.4). The phenotype table in H2A.1.2 is the source-of-truth for phenotype priors; the contingency block in H2A.1.1 is the source-of-truth for `configs/pat_rl.yaml`
- `C:/Users/aman0087/Documents/Github/dcm_hgf_mixed_models/docs/files/GSD_prl_hgf.yaml` — PRL implementation spec (PRL.1-5, V1-V2); response-model signatures for B/C/D come from here
- `docs/PAT_RL_API_HANDOFF.md` (quick-005) — current public API surface; Phase 20 extends this, does NOT break it
- Phase 18 gap analysis — see STATE.md decision 114 (EV direction), 121 (log_beta parameterisation), 114-128 (PAT-RL runtime lessons)

**Architectural directives (USER-SPECIFIED, non-negotiable)**:
- **No new scripts.** All new behaviour lives in existing numbered scripts (`scripts/{03,04,05,06}_*.py`, `scripts/12_smoke_patrl_foundation.py`) made config-driven via `configs/pat_rl.yaml` toggles. If a new concern genuinely doesn't fit any existing script, the planner must surface it as a checkpoint before proceeding. Verify via `git diff --stat scripts/` at phase close
- **Citations must be 2020 or later, Roelofs-group-first.** Retire Browning 2015 / Daw 2006 / Schönberg 2007 defaults from the Phase 18 config. Substitute with Klaassen 2021/2024, Terburg 2020+, Hulsman 2020+, Ly 2022+, Roelofs 2020+ reviews. The HEART2ADAPT spec is the primary source of truth for parameter values; literature is secondary grounding for direction/magnitude
- **Config-driven adaptation.** Every behavioural variant (Model choice, phenotype group, ε₂-coupling coefficient, cohort size, stratified-BMS toggle) must be reachable via a `configs/pat_rl.yaml` key. No hardcoded magic numbers. No new CLI flags for things that belong in YAML

**Scope notes**:
- This phase supersedes the Phase 18 "Option A Minimum Viable" deferrals (Models B/C/D, full PRL-V1 gate, PRL-V2 gate, stratified BMS, PEB export)
- This phase DOES NOT open v1.3 HEART2ADAPT milestone — it completes v1.2 per user's explicit "Option B" choice
- Post-Phase-20 the `dcm_hgf_mixed_models` v2 bridge layer has everything it needs to fit H2A.1.4-1.6 end-to-end

---

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1 - Foundation | v1.0 | 2/2 | Complete | 2026-04-04 |
| 2 - Models | v1.0 | 2/2 | Complete | 2026-04-05 |
| 3 - Simulation | v1.0 | 2/2 | Complete | 2026-04-05 |
| 4 - Fitting | v1.0 | 2/2 | Complete | 2026-04-05 |
| 5 - Validation | v1.0 | 3/3 | Complete | 2026-04-06 |
| 6 - Group Analysis | v1.0 | 5/5 | Complete | 2026-04-06 |
| 7 - GUI | v1.0 | 2/2 | Complete | 2026-04-07 |
| 8 - Config + Infrastructure | v1.1 | 2/2 | Complete | 2026-04-07 |
| 9 - Prechecks | v1.1 | 2/2 | Complete | 2026-04-07 |
| 10 - Core Power Modules + Sweep | v1.1 | 3/3 | Complete | 2026-04-07 |
| 11 - Aggregation + Publication | v1.1 | 3/3 | Complete | 2026-04-07 |
| 12 - Batched Hierarchical JAX Logp | v1.2 | 4/4 | Complete | 2026-04-12 |
| 13 - JAX-Native Cohort Simulation | v1.2 | 3/3 | Complete | 2026-04-12 |
| 14 - Integration + GPU Benchmark | v1.2 | 0/3 | Pending | -- |
| 15 - Production Run + Results | v1.2 | 0/2 | Pending | -- |
| 16 - NumPyro Direct + CUDA Fix | v1.2 | 2/2 | Complete | 2026-04-13 |
| 17 - BlackJAX NUTS Sampler | v1.2 | 2/2 | Complete | 2026-04-15 |
| 18 - PAT-RL Task Adaptation (HEART2ADAPT) | v1.2 | 6/6 | Complete (Option A scope) | 2026-04-18 |
| 19 - VB-Laplace Fit Path (Tapas-Parity) | v1.2 | 5/5 | Complete | 2026-04-18 |
| 20 - HEART2ADAPT Scientific Completion | v1.2 | 0/0 | Not planned | -- |
