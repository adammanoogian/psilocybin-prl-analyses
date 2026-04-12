# Roadmap: PRL HGF Analysis Pipeline

## Milestones

- **v1.0 Simulation-to-Inference Pipeline** — Phases 1-7 (shipped 2026-04-07)
- **v1.1 BFDA Power Analysis** — Phases 8-11 (code-complete 2026-04-07)
- **v1.2 Hierarchical GPU Fitting** — Phases 12-15 (in progress)

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
**Plans**: ~3 plans (iteration wiring + legacy flag, benchmark harness, decision-gate recording + validation)

### Phase 15: Production Run + Results

**Goal**: The full SBF power sweep runs on the platform chosen in Phase 14. All 600 tasks complete, aggregate into `power_master.csv` without missing cells, and produce the real v1.1 deliverables: the 4-panel publication figure, `recommendation.md` with concrete N/group backed by real BF evidence, and an auto-pushed results branch.
**Depends on**: Phase 14 (integration complete, platform decision made)
**Requirements**: PROD-01, PROD-02, PROD-03, PROD-04, PROD-05
**Success Criteria** (what must be TRUE):
  1. Full sweep (3 chunks × 200 iterations) completes on the chosen platform; SLURM logs show no divergent-chain warnings above ~5 % exclusion rate per cell
  2. `scripts/09_aggregate_power.py` produces `power_master.csv` with 360,000 rows (600 iterations × 300 fits × ... confirm schema) and no missing-cell warnings
  3. `scripts/10_plot_power_curves.py` regenerates the 4-panel figure as PDF + PNG with real data, no placeholders
  4. `scripts/11_write_recommendation.py` writes `recommendation.md` containing a concrete recommended N per group, chosen trial count, 80 %-power crossing for d = 0.5, exclusion rate summary, and the omega_3 upper-bound caveat
  5. Wave 3 push (`99_push_results.slurm`) triggered via `afterany:${POSTPROC_JOBID}` and commits `results/power/` to git on completion
**Plans**: ~2-3 plans (cluster run orchestration, aggregation + figure regeneration, recommendation)

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
| 14 - Integration + GPU Benchmark | v1.2 | 0/3 | Pending | — |
| 15 - Production Run + Results | v1.2 | 0/3 | Pending | — |
