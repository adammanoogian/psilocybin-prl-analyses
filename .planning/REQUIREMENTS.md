# Requirements: v1.1 Power Analysis

**Defined:** 2026-04-07
**Core Value:** Determine required sample size and trial count for detecting psilocybin x session interactions on HGF parameters via simulation-based BFDA.
**Design:** Post-concussion participants, psilocybin vs placebo (between), 3 sessions (within)

## v1.1 Requirements

### Config & Infrastructure (PWR)

- [ ] **PWR-01**: Power analysis config section in prl_analysis.yaml: effect_sizes [0.3, 0.5, 0.7], n_per_group_levels [10, 15, 20, 25, 30, 40, 50], trial_count_levels [50, 100, 150, 200, 250], n_iterations (K=200), bf_threshold (6), recovery_r_threshold (0.7), bms_exceedance_threshold (0.75), jzs_scale (sqrt(2)/2), seed_base
- [ ] **PWR-09**: Embarrassingly parallel iterations via SLURM array jobs (--array=0-K%50 throttle). Each iteration is an independent process with its own JAX initialization
- [ ] **PWR-10**: Tidy results DataFrame per iteration saved as parquet: sweep_type, effect_size, n_per_group, trial_count, iteration, parameter, bf_value, bf_exceeds, bms_xp, bms_correct, recovery_r, n_divergences, mean_rhat

### Prechecks (PRE) — Gate the Power Analysis

- [ ] **PRE-01**: Parameter recovery at current trial count: simulate 50 participants, fit MCMC, compute Pearson r per parameter. r < 0.7 -> excluded from power analysis. omega_3 results labeled "exploratory — upper bound"
- [ ] **PRE-02**: Confound matrix: pairwise correlations between recovered parameters. |r| > 0.8 -> flag
- [ ] **PRE-03**: Output list of power-eligible parameters with exclusion reasons documented
- [ ] **PRE-04**: Trial count sweep: fix N=30/group, vary trials [50-250], compute recovery r per parameter per level. Preserve stable/volatile trial ratio when adjusting total count
- [ ] **PRE-05**: Identify minimum trial count where all power-eligible parameters exceed recovery threshold
- [ ] **PRE-06**: MCMC convergence gating: R-hat < 1.05 and ESS > 400 required per fit; fits failing these thresholds are flagged and excluded from power calculation

### Power Analysis A: Psilocybin x Session Interaction (PWR-A)

- [ ] **PWR-02**: Sample size sweep: fix trial count (from PRE-05), iterate N levels. Each iteration: generate N x 2 groups (psilocybin vs placebo) x 3 sessions with interaction of magnitude d on target parameter via session_deltas. Fit all MCMC. Compute interaction contrast. Compute JZS BF. Record BF > threshold
- [ ] **PWR-03**: JZS BF via pingouin.bayesfactor_ttest (Rouder et al. 2009 one-sample t-test) on the difference-in-differences contrast. Prior scale r = sqrt(2)/2 (default Cauchy)
- [ ] **PWR-04**: Sensitivity sweep at d = {0.3, 0.5, 0.7}, producing family of power curves
- [ ] **PWR-05**: Primary contrast: delta_i = (theta_psi,post - theta_psi,baseline) - (theta_plc,post - theta_plc,baseline). Also test baseline->followup and linear trend. Document which is primary
- [ ] **PWR-06**: Effect size parameterized via session_deltas: delta = d x pooled_sd(parameter) from config group distributions. make_power_config() factory constructs frozen config instances from (N, d, seed) without modifying YAML

### Power Analysis B: Model Discriminability (PWR-B)

- [ ] **PWR-07**: Generate from 3-level model, fit both models MCMC, compute WAIC/LOO, run random-effects BMS (Rigoux et al. 2014). Record P(XP_true > 0.75) at each N
- [ ] **PWR-08**: Optional group-stratified BMS: does psilocybin shift model preference?

### Seed & Reproducibility (SEED)

- [ ] **SEED-01**: numpy.random.SeedSequence for parallel seed independence across SLURM array tasks. Each iteration gets a child seed from the base SeedSequence, not bare integer arithmetic

### Visualizations (VIZ)

- [ ] **VIZ-01**: Precheck figure: recovery r vs trial count, one line per parameter, reference at r=0.7
- [ ] **VIZ-02**: Power A figure: P(BF>6) vs N, one line per effect size, references at 80%/90%, annotate N where d=0.5 crosses 80%
- [ ] **VIZ-03**: Power B figure: P(correct BMS) vs N, reference at 75%
- [ ] **VIZ-04**: Combined 4-panel publication figure (precheck + A + B + sensitivity heatmap). PDF + PNG

### Recommendation (REC)

- [ ] **REC-01**: Summary table + text: recommended N/group and trial count with evidence. Markdown report in results/power/

## v2 Requirements (Deferred)

- **V2-PWR-01**: MAP fitting as fast proxy for power loop (validate against MCMC first)
- **V2-PWR-02**: Sequential BFDA with optional stopping (not applicable to fixed clinical N)
- **V2-PWR-03**: Full posterior propagation to Level 2 (hierarchical model; current pipeline uses summary statistics)
- **V2-PWR-04**: Interactive power analysis GUI

## Out of Scope

| Feature | Reason |
|---------|--------|
| Sequential BFDA | Clinical trial has fixed N; sequential stopping not applicable |
| rpy2 bridge to R BayesFactor | pingouin handles JZS BF; anovaBF has documented misspecification for RM designs (Van den Bergh et al. 2023) |
| Full posterior propagation | Doesn't match the actual two-stage analysis pipeline |
| Power analysis GUI | Batch HPC computation, not interactive |
| omega_3 as primary power target | Recovery r < 0.7 expected; labeled exploratory only |

## Success Criteria

1. At least omega_2 and beta pass recovery precheck (r > 0.7)
2. Power A at d=0.5 crosses 80% at a concrete N
3. Power B crosses 75% at a concrete N
4. 4-panel figure is self-contained for grant/preregistration
5. Full analysis < 200 cluster-hours

## Key References

- Schreiber et al. (2024) — BFDA for multi-armed bandit tasks
- Hess et al. (2025) — Bayesian workflow for computational psychiatry
- Schonbrodt & Wagenmakers (2018) — Bayes factor design analysis
- Rouder et al. (2009) — JZS Bayesian t tests
- Rigoux et al. (2014) — Random-effects BMS
- Wilson & Collins (2019) — Ten simple rules for computational modeling
- Van den Bergh et al. (2023) — anovaBF misspecification for RM designs

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PWR-01 | TBD | Pending |
| PWR-09 | TBD | Pending |
| PWR-10 | TBD | Pending |
| PRE-01 | TBD | Pending |
| PRE-02 | TBD | Pending |
| PRE-03 | TBD | Pending |
| PRE-04 | TBD | Pending |
| PRE-05 | TBD | Pending |
| PRE-06 | TBD | Pending |
| PWR-02 | TBD | Pending |
| PWR-03 | TBD | Pending |
| PWR-04 | TBD | Pending |
| PWR-05 | TBD | Pending |
| PWR-06 | TBD | Pending |
| PWR-07 | TBD | Pending |
| PWR-08 | TBD | Pending |
| SEED-01 | TBD | Pending |
| VIZ-01 | TBD | Pending |
| VIZ-02 | TBD | Pending |
| VIZ-03 | TBD | Pending |
| VIZ-04 | TBD | Pending |
| REC-01 | TBD | Pending |

**Coverage:** 22 v1.1 requirements across 7 categories.

---
*Requirements defined: 2026-04-07*
