# Roadmap: PRL HGF Analysis Pipeline

## Milestones

- **v1.0 Simulation-to-Inference Pipeline** — Phases 1-7 (shipped 2026-04-07)
- **v1.1 BFDA Power Analysis** — Phases 8-11 (in progress)

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
- [ ] 11-01-PLAN.md — power/curves.py + scripts/09_aggregate_power.py: aggregation, P(BF>threshold) & P(BMS correct) computation, 8+ unit tests
- [ ] 11-02-PLAN.md — scripts/10_plot_power_curves.py: Power A, Power B, sensitivity heatmap, 4-panel publication figure (PDF+PNG), 6+ tests
- [ ] 11-03-PLAN.md — scripts/11_write_recommendation.py: recommendation.md with N/group, trial count, evidence table, caveats, 7+ tests

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
| 11 - Aggregation + Publication | v1.1 | 0/3 | Not started | - |
