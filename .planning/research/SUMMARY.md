# Project Research Summary

**Project:** PRL HGF Analysis — v1.1 BFDA Power Analysis
**Domain:** Simulation-based Bayes Factor Design Analysis for a two-level computational psychiatry pipeline
**Researched:** 2026-04-07
**Confidence:** HIGH (stack and architecture from direct source inspection; features and pitfalls from peer-reviewed literature)

## Executive Summary

This milestone adds a Bayesian power analysis layer (BFDA) on top of the existing HGF simulation-fit pipeline. The design is straightforward Monte Carlo: for each (N, effect_size, iteration) cell, simulate a synthetic cohort, run the existing MCMC fitting pipeline, extract parameter posterior means, compute a JZS Bayes factor on the group-level contrast, and record the BF. Power is P(BF > threshold) across iterations. No new Python packages are required — `pingouin.bayesfactor_ttest` (already declared) handles JZS BF computation, SLURM array jobs handle parallelism, and `matplotlib` handles visualization. The entire BFDA layer lives in a new `src/prl_hgf/power/` package that wraps the existing pipeline without modifying any existing module.

The primary research conflict: FEATURES recommended `rpy2` + R's `anovaBF()`. STACK found (a) `pingouin.bayesfactor_ttest` already implements the identical JZS integral (Rouder et al. 2009, Eq. 2) and is already installed, and (b) `anovaBF()` is misspecified for repeated-measures designs (van den Bergh et al. 2023). Resolution: use `pingouin.bayesfactor_ttest` on posterior contrast draws, use `bambi` with explicit random slopes, do not introduce `rpy2`.

The two primary scientific risks are (1) omega_3 recovery is poor with binary-only data (r = 0.67, Hess et al. 2025), so BFDA results are an upper bound and must be labeled as such, and (2) seed independence across 4,000+ SLURM array tasks requires `numpy.random.SeedSequence` rather than task-ID seeding. Both are preventable with explicit design choices before the first cluster submission.

## Key Findings

### Recommended Stack

Zero new packages. `pingouin >= 0.5.5` (currently 0.6.1) provides a validated JZS BF implementation verified from source: it uses `scipy.integrate.quad` over the exact Rouder et al. (2009) Eq. 2 integrand and explicitly handles the one-sample case (`ny is None`). The 1D integral completes in microseconds -- not the compute bottleneck (JAX has no production-ready `quad`, confirmed via issue #27493). SLURM array jobs are the correct parallelism mechanism because JAX's documented `multiprocessing.fork` incompatibility (issues #1805, #7620) makes in-process joblib parallelism unsafe. The power loop should use CPU MCMC (`cores=1, chains=2`) to avoid GPU OOM failures.

**Core technologies:**
- `pingouin.bayesfactor_ttest`: JZS BF for one-sample/paired contrast — validated against JASP and R BayesFactor; already installed; source-verified
- SLURM array jobs with `--array=0-N%50`: embarrassingly parallel dispatch — %50 throttle prevents conda import storm on shared filesystem
- `bambi` (existing): group-level mixed-effects model with explicit random slopes — replaces `anovaBF` and avoids repeated-measures misspecification
- `matplotlib` / `seaborn` (existing): power curve visualization — `statsmodels` power functions compute frequentist (1-beta), not Bayesian P(BF > threshold)
- `numpy.random.SeedSequence`: guaranteed independent RNG streams per job — task-ID seeding gives unique but correlated seeds

**Explicitly rejected:**
- `rpy2` + `anovaBF()`: heavy dependency AND misspecified for this design (omits random slopes; van den Bergh et al. 2023)
- Custom JZS `scipy.integrate.quad` implementation: pingouin already provides this, tested and validated
- `statsmodels` power functions: frequentist power is the wrong abstraction here
- `torchquad` GPU quadrature: 1D integration is not the bottleneck

### Expected Features

BFDA scope is strictly the two primary hypotheses: omega_2 and kappa group x session interaction. omega_3 BFDA is secondary and must be labeled as an upper bound. No GUI, no sequential stopping rule (fixed IRB-approved N), no per-participant BF.

**Must have (table stakes):**
- Parameter recovery precheck at current trial counts — gate: do not proceed if R² < 0.6 for omega_2, kappa
- Simulation loop with MCMC convergence gate — R-hat > 1.05 or divergences > 0 flags a fit for exclusion
- P(BF > threshold) swept across N x effect_size grid for omega_2 and kappa separately
- Effect size sweep: at minimum d = 0.2, 0.4, 0.6, 0.8, 1.0 (not a single literature point estimate)
- P(correct model wins BMS) swept across N — required by Nature Human Behaviour 2025
- Seeded reproducibility via `numpy.random.SeedSequence`
- One CSV per job (implicit checkpoint/resume; no shared writer)
- Summary statistics per (N, d) cell: P(BF10 > 10), P(BF10 > 6), P(BF10 > 3), median BF, false-positive rate analog

**Should have (differentiators for publication):**
- Separate power curves for omega_2 and kappa
- 4-panel publication figure synthesizing all BFDA outputs
- Monte Carlo error bands on power curves (MCE ≤ 0.005 requires ≥ 2,000 iterations per cell)
- Recovery-penalty simulation variant for omega_3 (inject Gaussian noise with SD = empirical RMSE)
- BF threshold sensitivity (P(BF>10) vs P(BF>6) vs P(BF>3)) computed post-hoc

**Defer to post-launch:**
- Trial count sweep (unless recovery precheck reveals marginal adequacy)
- MAP/VI vs NUTS pilot comparison
- Informed effect sizes from pilot data
- BF threshold sensitivity analysis at revision stage

### Architecture Approach

The BFDA layer is a new package `src/prl_hgf/power/` that wraps the existing pipeline without modifying any existing module. A config factory (`make_power_config`) builds modified `AnalysisConfig` instances at runtime — YAML never mutated (frozen dataclass; 4,000 concurrent jobs cannot share a mutable config file). All existing functions (`simulate_batch`, `fit_batch`, `build_estimates_wide`, `fit_group_model`, `extract_posterior_contrasts`) called unchanged. Scripts extend the `03_`-`06_` numbering at `07_`-`10_`.

**Major components:**
1. `power/config_override.py` — `make_power_config(base_config, n_per_group, effect_size_delta, master_seed)` factory; no YAML mutation; unblocks all other components
2. `power/bayes_factor.py` — `compute_jzs_bf(posterior_draws)` using `pingouin.bayesfactor_ttest` on draws from `extract_posterior_contrasts`; unit-testable before any MCMC run
3. `power/stopping_rule.py` — evaluates BF against H1/H0 thresholds; returns "H1" | "H0" | "inconclusive"
4. `power/iteration.py` — orchestrates one (N, d, k) simulation+fit+BF cycle; writes single-row CSV
5. `power/curves.py` — aggregates per-job CSVs, computes power per (N, d) cell, plots curves

**New scripts:**
- `scripts/07a_generate_job_params.py` — writes `data/power/job_params.csv` (N x d x k grid)
- `scripts/07_power_single_iteration.py` — CLI entry point for one SLURM task
- `scripts/08_power_slurm_array.sh` — SLURM array submitter with %50 throttle
- `scripts/09_aggregate_power.py` — concatenates per-job CSVs; warns on missing jobs
- `scripts/10_plot_power_curves.py` — generates publication figures

### Critical Pitfalls

1. **anovaBF misspecification — resolved: use pingouin instead** — `anovaBF()` omits random slopes for within-subject factors (van den Bergh et al. 2023), inflating BF for interaction terms. Prevention: use `pingouin.bayesfactor_ttest` on posterior contrast draws; use `bambi` with explicit random slopes. Do not introduce `rpy2`.

2. **omega_3 recovery caveat** — omega_3 recovery is r = 0.67 with binary-only data (Hess et al. 2025; also flagged in project CLAUDE.md). Naive BFDA inflates estimated power by 20-40 percentage points. Prevention: label all omega_3 BFDA outputs as upper bounds; recovery-penalty variant (inject RMSE noise before BF); pre-register omega_3 as exploratory. Primary focus: omega_2 and kappa (r > 0.85).

3. **Non-independent seeds across SLURM jobs** — `np.random.seed(SLURM_ARRAY_TASK_ID)` gives correlated streams; bias is undetectable post-hoc. Prevention: `ss = np.random.SeedSequence(base_seed); rng = np.random.default_rng(ss.spawn(n_jobs)[task_id])`; for JAX use `jax.random.split(base_key, n_jobs)[task_id]`.

4. **SLURM import storm** — 4,000 concurrent Python jobs saturate the Lustre metadata server (documented M3 MASSIVE issue); empty output files silently treated as missing cells. Prevention: `#SBATCH --array=0-N%50` throttle; or bundle 20 iterations per task to reduce job count 20x.

5. **MCMC convergence failures silently contaminating power estimates** — divergent transitions and high R-hat fits bias the BF distribution. Prevention: save R-hat, ESS-bulk, divergence count in every fit output CSV; filter R-hat > 1.05 or divergences > 0; report exclusion rate per (N, d) cell; flag cells > 5% exclusion.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Infrastructure and Prechecks
**Rationale:** Recovery validation gates everything. If omega_2 or kappa cannot be recovered (R² < 0.6), BFDA is invalid regardless of N. The recovery module already exists -- re-run with pass/fail criteria. `power/config_override.py` built here as lowest-risk new code, unblocking all other phases.
**Delivers:** Recovery pass/fail gate; `make_power_config` factory with unit tests on `n_participants_per_group` and `omega_2_deltas` overrides
**Addresses:** Recovery precheck; seeded reproducibility from the start
**Avoids:** Pitfall 3 (omega_3 overconfidence), Pitfall 1 (MCMC convergence failures)

### Phase 2: Core BF and Single-Iteration Pipeline
**Rationale:** BF computation and single-iteration orchestrator can be unit-tested with synthetic data before any cluster work. Validates the pingouin BF implementation with known inputs.
**Delivers:** `power/bayes_factor.py`, `power/stopping_rule.py`, `power/iteration.py`, `scripts/07_power_single_iteration.py`; smoke test: N=5, --n-draws 50 --n-tune 50
**Uses:** `pingouin.bayesfactor_ttest`, `analysis.group.extract_posterior_contrasts` (signature verified from source)
**Implements:** Architecture components 2, 3, 4
**Avoids:** Pitfall 5 (anovaBF not used), Pitfall 9 (SeedSequence from the start)
**Research flag:** Standard patterns — no deeper research needed

### Phase 3: SLURM Array Orchestration
**Rationale:** Cluster-ready array job requires %50 throttle, one-file-per-job output, and empirical validation on M3 MASSIVE. Start with 10-job smoke test.
**Delivers:** `scripts/07a_generate_job_params.py`, `scripts/08_power_slurm_array.sh`; 10-job smoke test confirming naming, schema, and no import-storm symptoms
**Avoids:** Pitfall 9 (seeds), Pitfall 10 (import storm via %50 throttle), Pitfall 11 (GPU OOM -- CPU MCMC in power loop), shared-writer corruption
**Research flag:** Needs empirical validation on M3 MASSIVE -- tune %50 based on 10-job test

### Phase 4: Full N x Effect-Size Sweep
**Rationale:** Full grid submission after SLURM validation. omega_2 and kappa swept separately. BMS sweep parallel workstream via groupBMC. Run MAP vs NUTS pilot comparison first.
**Delivers:** Full `data/power/results/` (per-job CSVs); BMS sweep results; all primary power estimates
**Addresses:** P(BF > threshold) over N x d, P(correct BMS) over N, separate curves per parameter
**Avoids:** Pitfall 6 (2,000 iterations per cell; 500 first to find inflection zone), Pitfall 13 (BF and BMS as separate outputs)
**Research flag:** MAP vs NUTS pilot (100 iterations each) before full NUTS; BMS pilot (10 iter) to estimate compute

### Phase 5: Aggregation, Power Curves, and Publication Figure
**Rationale:** Aggregation developed against Phase 3 10-job test output so it is ready before Phase 4 completes.
**Delivers:** `scripts/09_aggregate_power.py`, `power/curves.py`, `scripts/10_plot_power_curves.py`; `data/power/power_master.csv`; `figures/power_curves.png` (4-panel with MCE error bands)
**Addresses:** 4-panel figure, BF threshold sensitivity post-hoc, recovery-penalty omega_3 variant
**Avoids:** Pitfall 2 (attenuation bias documented in figure caption), Pitfall 12 (design/analysis prior distinction labeled in legend)
**Research flag:** Standard patterns -- no deeper research needed

### Phase Ordering Rationale

- Phase 1 first: recovery precheck is cheap and gates everything; `config_override.py` is the dependency root.
- Phase 2 independent of cluster: synthetic data unit tests validate BF before any MCMC run.
- Phase 3 gates Phase 4; BMS and BF sweeps can run in parallel once SLURM is confirmed.
- Phase 5 starts developing against Phase 3 outputs and finalizes after Phase 4.
- No-modification-to-existing-modules principle: no regression risk in the existing pipeline.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (SLURM throttle on M3 MASSIVE):** %50 cap from general HPC guidance; tune empirically with 10-job test.
- **Phase 4 (MAP vs NUTS):** 100-iteration pilot comparison before committing to full NUTS budget.
- **Phase 4 (BMS compute budget):** 10-iteration BMS pilot to estimate per-cell wall time.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Recovery module exists; pass/fail gate only.
- **Phase 2:** pingouin API source-verified; `extract_posterior_contrasts` signature confirmed.
- **Phase 5:** pd.concat and matplotlib are standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | pingouin bayesian.py source-inspected; JAX fork issues from tracker; SLURM from existing repo scripts |
| Features | MEDIUM | Triangulated from Schonbrodt 2018, Stefan 2019, Hess 2025, Nature Human Behaviour 2025; HGF+JZS combination not in single canonical source |
| Architecture | HIGH | All integration points confirmed from source code: function signatures, frozen dataclass fields, DataFrame output schemas |
| Pitfalls | HIGH | van den Bergh 2023 (anovaBF), Hess 2025 (omega_3), NumPy SeedSequence docs, M3 MASSIVE cluster docs |

**Overall confidence:** HIGH

### Resolved Conflicts

**FEATURES vs STACK: BF computation method**

FEATURES.md recommended `rpy2` + `BayesFactor::anovaBF()`. STACK.md identified two independent reasons to reject this:
1. `pingouin.bayesfactor_ttest` already implements the identical Rouder et al. (2009) JZS integral, is already installed, and has been validated against JASP and R BayesFactor.
2. `anovaBF()` is misspecified for repeated-measures designs with two or more within-subject factors (van den Bergh et al. 2023); it omits random slopes and inflates BF for interaction terms.

**Decision: pingouin wins. Do not introduce rpy2. Use bambi with explicit random slopes for the full mixed-effects group model.**

### Gaps to Address

- **MAP vs NUTS for power loop:** 100-iteration pilot in Phase 4 before committing to full NUTS.
- **Effect size parameterization for kappa:** omega_2 maps cleanly onto `session_deltas`; kappa entry point in `GroupConfig` vs `SessionConfig` needs verification in `make_power_config` unit tests before cluster submission.
- **Optimal SLURM concurrency on M3 MASSIVE:** %50 from general guidance; verify empirically in Phase 3.

## Sources

### Primary (HIGH confidence — direct source inspection or peer-reviewed with code)
- pingouin 0.6.1 bayesian.py source (GitHub) — JZS BF implementation confirmed, one-sample case confirmed
- Existing codebase: simulate_batch, fit_batch, build_estimates_wide, fit_group_model, extract_posterior_contrasts, AnalysisConfig -- all verified from source
- Van den Bergh et al. (2023). Bayesian Repeated-Measures ANOVA: Updated Methodology. Psychological Methods
- Hess et al. (2025). Bayesian Workflow for Generative Modeling in Computational Psychiatry. PMC11951975
- NumPy SeedSequence documentation
- JAX multiprocessing fork issues #1805, #7620

### Secondary (MEDIUM confidence — peer-reviewed methodology)
- Schonbrodt & Wagenmakers (2018). Bayes factor design analysis. Psychonomic Bulletin & Review
- Stefan et al. (2019). Tutorial on BFDA using an informed prior. Behavior Research Methods. PMC6538819
- Nature Human Behaviour (2025). Addressing low statistical power in computational modelling
- Rigoux et al. (2014). Bayesian model selection for group studies -- Revisited. NeuroImage
- Schreiber et al. (2024). BFDA for multi-armed bandit tasks. Wellcome Open Research
- Wilson & Collins (2019). Ten simple rules for computational modeling. PMC6879303
- LMU Power Simulation tutorial -- MCE target ≤ 0.005

### Tertiary (MEDIUM confidence — cluster-specific, needs empirical validation)
- Monash M3 MASSIVE array jobs documentation — import storm and throttling
- PyMC discourse GPU memory thread — CPU MCMC for power loops

---
*Research completed: 2026-04-07*
*Ready for roadmap: yes*
