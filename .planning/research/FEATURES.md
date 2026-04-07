# Feature Landscape: BFDA Power Analysis Pipeline

**Domain:** Simulation-based Bayes Factor Design Analysis for a two-level computational psychiatry pipeline (HGF + group-level mixed-effects)
**Researched:** 2026-04-07
**Study:** Post-concussion syndrome, psilocybin vs placebo, 3 sessions, primary outcomes omega_2 and kappa

---

## Context: What Makes This Domain Unusual

Standard BFDA (Schonbrodt & Wagenmakers 2018, Stefan et al. 2019) targets simple t-tests and ANOVA
designs with known analytic Bayes factors. This project's inference chain is non-standard:

1. Individual MCMC fitting of a JAX-backed HGF model (the expensive part)
2. Point estimates extracted and fed into group-level mixed-effects models (bambi)
3. BF computed on the group-level outcome (omega_2, kappa), not directly on trial-by-trial data

This means there is no off-the-shelf BFDA package. The pipeline is entirely simulation-based Monte Carlo.
Every feature below must be evaluated against whether it fits this two-stage inference chain.

**Confidence on feature classification:** MEDIUM. The canonical BFDA literature (Schonbrodt, Stefan)
covers simpler designs. The Hess et al. 2025 Bayesian workflow and the Nature Human Behaviour 2025
paper on low power in computational modelling provide the computational psychiatry context. The
specific combination of MCMC-extracted point estimates + anovaBF for a group x session design is
not documented in a single canonical source; classification draws on triangulation across these sources.

---

## Table Stakes

Features that any credible BFDA for this study must have. Missing one makes the analysis
unpublishable or methodologically indefensible.

| Feature | Why Expected | Complexity | v1.0 Dependency |
|---------|--------------|------------|-----------------|
| **Parameter recovery check at current trial count** | Must confirm inference is valid at the actual trial counts before sweeping N. Failure here invalidates all downstream analysis. Hess et al. 2025 mandates this as phase 1 of Bayesian workflow. | Low (already have recovery module; just re-run at fixed N) | Phase 5 recovery module (05-01-PLAN) |
| **Simulation loop: generate synthetic data -> fit -> extract point estimate** | The atomic unit of the entire pipeline. Each iteration is one simulated "study". Cannot do BFDA without it. | High (MCMC per iteration is expensive; the bottleneck) | Phases 3, 4 (simulation + fitting) |
| **P(BF > threshold) swept across N** | The primary output of any fixed-N BFDA (Schonbrodt & Wagenmakers 2018). Answers "how many participants do we need?" | Medium (orchestration + BF computation; iterative over N grid) | Simulation loop above |
| **Effect size grid** | Power depends jointly on N and delta. A fixed-effect-size power curve is only useful if effect size is known with high confidence; for psilocybin/PCS it is not. Must sweep at minimum 3 effect sizes (small/medium/large or literature-informed range). | Medium (outer loop over effect size; statistical overhead is low once simulation loop exists) | None; define effect-size parameterization |
| **JZS Bayes factor for group x session interaction** | Explicitly required by the user. JZS is the default in BayesFactor::anovaBF (Rouder et al.). It is the most commonly reported BF for ANOVA-type hypotheses in psychology. | Medium (requires R via rpy2 bridge OR reimplementation; rpy2 is the pragmatic path) | None |
| **P(correct model wins BMS) swept across N** | Mandated by the 2025 Nature Human Behaviour paper: 41 of 52 reviewed computational psychiatry studies had <80% model recovery probability. Must verify both 2-level and 3-level models are recoverable at planned N. | Medium (groupBMC already integrated in Phase 5; need outer N-sweep loop) | Phase 5 BMS module (05-02-PLAN) |
| **MCMC convergence validation across simulated participants** | Rhat >= 1.01, ESS < 400, divergence counts must be acceptable before including a simulated participant in power estimates. Contaminating power analysis with unconverged chains gives false results. | Low (ArviZ diagnostics already in pipeline; add gate logic) | Phase 4 fitting pipeline |
| **Seeded reproducibility** | All simulations must be reproducible by seed. Required for replication and publication. v1.0 already enforces this; it must extend to the power pipeline. | Low | Phase 3 simulation (existing `np.random.seed` pattern) |
| **Results serialization to disk** | Each simulation is expensive (minutes per MCMC). Results must be saved incrementally so that cluster preemption does not lose work. Minimum: save per-N-per-effectsize BF array. | Medium (I/O orchestration; xarray or numpy structured saves are sufficient) | None |
| **Summary statistics per cell (N, delta)** | At minimum: P(BF10 > 10), P(BF10 > 3), mean(BF10), median(BF10), P(BF01 > 10) (false-positive rate analog). These are the standard BFDA outputs (Stefan et al. 2019 Table 1). | Low (numpy reductions over simulation array) | Simulation loop |

---

## Differentiators

Features that go beyond what a standard BFDA pipeline would include. Valuable for the
publication, for robustness, or for the specific computational psychiatry context.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Trial count sweep (prechecks)** | Not standard BFDA. Answers a different question: "does the task have enough trials for this model to be identifiable at all, independent of group N?" Directly addresses the known omega_3 recovery problem. Needed here because PRL phase structure is criterion-based (variable trial counts). | Medium (re-run fitting + recovery module at trial counts bracketing observed range; likely 3-5 counts) | Depends on Phase 5 recovery; adds ~1 day of compute per count on cluster |
| **4-panel publication figure** | Synthesizes all BFDA outputs into a single figure for a methods/results section. Standard format in BFDA papers (Schonbrodt 2018 Fig 3; Stefan 2019 Fig 5). Panel structure: (1) P(BF>10) heatmap over N x delta, (2) P(correct BMS) curve over N, (3) BF distribution violin at recommended N, (4) parameter recovery R^2 vs trial count. | Medium (matplotlib figure assembly; no new computation) | All upstream results must be complete |
| **Separate power curves for omega_2 and kappa** | The two primary hypotheses target different parameters with different expected effect sizes and different recovery quality. Treating them identically conflates distinct power profiles. | Low (run the BF sweep twice with different effect size parameterizations for each parameter) | omega_3 should be explicitly flagged as secondary with a note about known poor recovery |
| **SLURM array job parallelization** | Each (N, delta) cell is embarrassingly parallel. Without cluster parallelization, the full sweep at realistic MCMC cost (~3 min/participant, 1000 simulations, 6 N values, 4 delta values) would take ~5,000 CPU-hours sequential. This is not optional for a complete sweep; it is a compute necessity. | High (SLURM script templates, task-ID-to-parameter mapping, result aggregation step) | Requires HPC access; the quick task 001-cluster-gpu-setup is already in planning |
| **Checkpoint/resume for interrupted runs** | SLURM jobs can be preempted. Without checkpointing, a preempted array job wastes all work done. Minimum viable: each array task writes its own output file; aggregation step skips already-completed cells. | Medium (file-existence check before running; trivial if output files are per-task) | Natural extension of results serialization |
| **Effect size informed by pilot data or literature** | Rather than arbitrary Cohen's d small/medium/large, parameterize effect sizes from published psilocybin cognitive studies or from pilot simulations using the omega_2 prior range. This is Stefan et al. 2019's "informed BFDA" approach. | Medium (literature search + prior predictive simulation to derive plausible delta range) | Requires design decisions outside code |
| **BF threshold sensitivity** | Report P(BF > 10) and P(BF > 6) and P(BF > 3). Reviewers often ask about threshold choice. Adds minimal cost (same simulations, different threshold applied at summary time). | Low | Post-hoc on simulation outputs |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain that would waste time or
produce misleading results.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Sequential BFDA (optional stopping)** | The study has a fixed, IRB-approved N per arm. Sequential designs (SBF) where data collection stops when BF > threshold are ethically and practically impossible in a clinical trial. Building SBF infrastructure is waste. | Use fixed-N BFDA only. Document this design constraint explicitly in the paper. |
| **Analytic BF formulas (t-test BF)** | The outcome is a mixed-effects model on extracted point estimates, not a simple two-sample t-test. Using BayesFactor::ttest.bf on a within-between design ignores the repeated-measures correlation and gives wrong BF values. | Use anovaBF (which handles the group x session x subject structure) or lmBF for more complex covariate structures. |
| **Simulating at full posterior (propagating parameter uncertainty into group level)** | The actual analysis pipeline uses point estimates (MAP or posterior mean) from individual fits and passes them to the group-level model. Power analysis must match the actual analysis. If power analysis uses the "correct" Bayesian propagation of uncertainty but the paper uses point estimates, the power estimate will be optimistic. | Simulate the pipeline as it is actually run: extract point estimates, run anovaBF on those estimates. Flag this as a known approximation in the methods. |
| **Power for omega_3 as a primary outcome** | omega_3 recovery is documented as poor in the literature (the project CLAUDE.md flags this explicitly). Running power analysis for a parameter that cannot be recovered gives a misleading impression of the study's capacity to detect effects on that parameter. | Run omega_3 power analysis only as a supplementary negative result (expected: low power regardless of N, because the parameter is not identifiable from this task). |
| **Per-participant BF (treating subject as observation)** | It is tempting to compute BF per simulated participant as a within-person measure. This confuses the level of inference. The BF belongs to the group-level hypothesis test, not to individual fits. | Compute BF once per simulated "study" using the group-level anovaBF on all participants in that simulated study. |
| **GUI for the power analysis pipeline** | A GUI was appropriate for the parameter explorer (Phase 7, interactive exploration). Power analysis is a batch HPC computation producing static publication figures. An interactive GUI adds engineering cost with no scientific benefit. | Static matplotlib figures from Python scripts. Jupyter notebook for result inspection only. |
| **Full MCMC at every power simulation iteration if faster approximations are sufficient** | Running 4-chain NUTS MCMC for every simulated participant across thousands of power simulation iterations is the bottleneck. This cost should be justified: confirm that MAP/VI approximations do not give the same power conclusion before committing to full MCMC. | Run a pilot comparison: 100 iterations with NUTS vs 100 with MAP. If power estimates agree within 5%, use MAP. If they diverge, use NUTS and document why. |
| **Power for more than the two primary hypotheses** | The study has two primary outcomes (omega_2 group x session, kappa group x session) and several secondary/exploratory outcomes (phase effects, win-stay/lose-shift). Computing BFDA for all outcomes inflates the analysis and buries the primary message. | Strictly scope BFDA to omega_2 and kappa. Note that secondary outcomes are exploratory and not powered. |

---

## Feature Dependencies

```
Trial count sweep (precheck)
  -> requires: Phase 5 recovery module (05-01-PLAN)
  -> must pass before: Power Analysis A or B

MCMC convergence gate
  -> requires: Phase 4 fitting pipeline
  -> embedded in: Simulation loop (every iteration)

Simulation loop (generate -> fit -> extract)
  -> requires: Phase 3 simulation, Phase 4 fitting
  -> feeds: Power Analysis A (BF sweep), Power Analysis B (BMS sweep)

JZS BF via anovaBF (rpy2 bridge)
  -> requires: R installation + BayesFactor package on compute nodes
  -> feeds: Power Analysis A

groupBMC N-sweep
  -> requires: Phase 5 BMS module (05-02-PLAN)
  -> feeds: Power Analysis B

Power Analysis A outputs
Power Analysis B outputs
  -> both required before: 4-panel publication figure

SLURM parallelization
  -> wraps: Simulation loop
  -> requires: Results serialization + checkpoint/resume
  -> enables: Full N x delta grid at practical compute cost
```

---

## MVP Recommendation

For a minimum viable BFDA that supports a grant application or preregistration:

1. **Precheck: parameter recovery at current trial counts** — gate; do not proceed if recovery
   is inadequate (R^2 < 0.6 for omega_2, kappa)
2. **Simulation loop with convergence gate** — the engine; acceptable with MAP instead of
   full MCMC in early passes
3. **Power Analysis A: P(BF > 10) over N x delta for omega_2 and kappa** — the primary deliverable
4. **Power Analysis B: P(correct model wins BMS) over N** — required by the 2025 Nature paper critique
5. **Summary figure** — minimum 2 panels (heatmap + BMS curve); expand to 4 panels for submission

Defer to post-MVP:
- **Trial count sweep**: defer unless recovery precheck reveals marginal adequacy; then
  it becomes urgent
- **SLURM parallelization**: implement from the start if HPC is available; defer to
  sequential execution only if N x delta grid is small enough (e.g., 3 N values x 3 delta
  values x 200 iterations = 1800 MCMC runs, borderline feasible overnight on CPU)
- **Informed effect sizes**: defer to after seeing pilot recovery results; use
  literature-range defaults initially
- **BF threshold sensitivity analysis**: defer to revision stage

---

## Sources

- Schonbrodt, F. D., & Wagenmakers, E. J. (2018). Bayes factor design analysis: Planning for
  compelling evidence. *Psychonomic Bulletin & Review*, 25(1), 128-142.
  https://link.springer.com/article/10.3758/s13423-017-1230-y

- Stefan, A. M., et al. (2019). A tutorial on Bayes Factor Design Analysis using an informed
  prior. *Behavior Research Methods*, 51, 1042-1058.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6538819/

- Hess, A. J., et al. (2025). Bayesian Workflow for Generative Modeling in Computational
  Psychiatry. *Computational Psychiatry*.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11951975/

- Nature Human Behaviour (2025). Addressing low statistical power in computational modelling
  studies in psychology and neuroscience.
  https://www.nature.com/articles/s41562-025-02348-6

- Rigoux, L., et al. (2014). Bayesian model selection for group studies — Revisited.
  *NeuroImage*. (groupBMC power; PEP framework)
  https://www.tnu.ethz.ch/fileadmin/user_upload/documents/Publications/2014/2014_Rigoux_Stephan_Friston_Daunizeau.pdf

- BFDA R package (nicebread/BFDA): workflow and Monte Carlo procedure reference.
  https://github.com/nicebread/BFDA

- BayesFactor R package (Morey, Rouder): anovaBF JZS implementation.
  https://richarddmorey.github.io/BayesFactor/

- ArviZ / PyMC convergence diagnostics: Rhat, ESS, divergence standards.
  https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html

- Vehtari et al. (2020). Improved R-hat for assessing convergence of MCMC.
  https://sites.stat.columbia.edu/gelman/research/published/Vehtari_etal_2020_rhat_ess.pdf
