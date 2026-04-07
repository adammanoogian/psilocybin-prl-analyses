# Domain Pitfalls: Simulation-Based BFDA for Computational Psychiatry

**Domain:** Bayes Factor Design Analysis for two-level HGF inference pipeline
**Researched:** 2026-04-07
**Milestone context:** v1.1 Power Analysis (BFDA) added to existing HGF pipeline

---

## Critical Pitfalls

Mistakes that cause invalid power estimates, require reruns, or produce misleading N recommendations.

---

### Pitfall 1: Ignoring MCMC Convergence Failures Inside the Power Loop

**What goes wrong:** Each power simulation iteration runs MCMC fitting. When a fit diverges (too many divergent transitions, R-hat > 1.05, ESS too low), the resulting posterior summary is biased. If the power loop silently accepts these fits, the recovered parameter values fed into the group-level Bayes factor test are wrong. This inflates or deflates the BF distribution in ways that corrupt the power curve entirely.

**Why it happens:** In a batch of 4,000+ parallel jobs each fitting a single subject, convergence warnings are printed to stderr and easy to miss. Post-hoc aggregation scripts typically just read the summary CSV; any warning that occurred during fitting is already gone.

**Consequences:** Power estimates based on biased posteriors are unreliable. The direction of corruption is unpredictable — divergences tend to cluster in specific parameter regimes (e.g., omega_3 at its prior boundary), so power may be systematically wrong for the exact parameter values you care about.

**Prevention:**
- Save R-hat, ESS-bulk, and divergence count as columns in every fit output CSV.
- After aggregation, filter out fits where any R-hat > 1.05 or divergences > 0. Report the exclusion rate per simulation cell.
- If exclusion rate exceeds ~5% in any cell, flag that cell as unreliable and investigate before concluding anything about power at those parameter values.
- Pre-test the fitting pipeline on 10 representative simulated datasets before launching the full cluster job.

**Detection:** Exclusion rate histogram across simulation cells; scatter of R-hat values versus recovered parameters.

**Phase:** Prechecks phase (MCMC validation step before main sweep).

**Sources:** Stan convergence documentation; PyMC discourse on divergent transitions.

---

### Pitfall 2: Using the Posterior Mean as a Noise-Free Point Estimate at Level 2

**What goes wrong:** The standard two-level pipeline extracts posterior means for each parameter per participant, then feeds those means into a group-level t-test or mixed ANOVA. The posterior mean is not the true parameter value — it carries MCMC estimation uncertainty. Treating it as if it were a clean observation attenuates the group-level effect size exactly as classical measurement error does (errors-in-variables attenuation bias). For the power analysis, this means the BF distribution under H1 will be shifted toward H0, and estimated power will be pessimistic relative to what a properly propagated model would show.

**Why it happens:** It is the default workflow (Wilson & Collins 2019 Rule 8; Hess et al. 2025). It is fast and workable for large N, but systematically underestimates power when parameter-level uncertainty is high relative to the true group effect.

**Consequences:** Power estimates are conservative — you may recommend a larger N than is actually needed. The bias is worst for poorly-recovered parameters (omega_3, kappa) and shrinks as N_trials and N_subjects increase. In this study, given poor omega_3 recovery (r ≈ 0.67 with binary-only data; Hess et al. 2025), attenuation for omega_3 may be severe enough to make any JZS BF test underpowered even at realistic N.

**Prevention:**
- Quantify the expected attenuation per parameter before the main sweep: compute ICC between true simulated values and recovered posterior means across the recovery sample. Attenuation factor ≈ ICC. Use this to calibrate expectations.
- Consider whether the BFDA power estimate should be labeled as a lower bound, with a note that fully propagated (single-step hierarchical) inference would perform better.
- For omega_3 specifically: treat BFDA results as an exploratory estimate only, not a design-determinative one.

**Detection:** Compare power curves from posterior-mean pipeline vs. a small trial using posterior samples directly (jackknife or bootstrap the posterior distribution). Large divergence confirms attenuation is material.

**Phase:** Two-level inference design decision (before main sweep).

**Sources:** Reliability paper (Friston group, PMC11104400); Hess et al. 2025 (PMC11951975); attenuation bias literature (Psychometrika 2024).

---

### Pitfall 3: Simulating Power for omega_3 as if It Were Identifiable

**What goes wrong:** omega_3 (meta-volatility) is the primary 3-level parameter that distinguishes the two models. In the HGF literature, omega_3 is known to have poor recovery with binary-only data. Hess et al. (2025) report r = 0.67 for omega_3 recovery in a 3-level eHGF with binary inputs — the weakest of all free parameters. The test-retest reliability study (PMC11573178) found that the phasic learning rate (kappa) had ICC = 0.04 before outlier removal. Running BFDA for group differences on these parameters as if recovery were adequate will produce power curves that are optimistic in a misleading way: the simulation will show apparent power because parameters were set by the simulation truth, but real-data power is much lower because those parameters cannot be reliably extracted.

**Why it happens:** Researchers design power analyses using simulation, where the generating truth is known. The power analysis loop treats recovery as perfect by construction, unless recovery error is explicitly injected as noise in the simulation.

**Consequences:** The study is designed around power targets that do not apply to the actual measurable quantity. For omega_3, the effective power curve may be 20-40 percentage points lower than the naive simulation predicts.

**Prevention:**
- Include a "recovery penalty" simulation variant: after fitting, inject Gaussian noise with SD equal to the observed parameter recovery RMSE before computing the group-level BF. Compare the resulting power curve against the noise-free version. The difference is the expected inflation.
- Explicitly document in all outputs: "Power for omega_3 is an upper bound assuming recovery noise equal to empirical RMSE; effective power may be substantially lower."
- Pre-register that omega_3 group effects are exploratory, not confirmatory. Focus BFDA on omega_2, beta, and zeta where recovery r > 0.85.

**Detection:** Parameter recovery scatter plots (already in v1.0 pipeline at REC-01/02); ensure REC runs on full trial count before BFDA launches.

**Phase:** Prechecks phase (parameter recovery validation); BFDA output labeling.

**Sources:** Hess et al. 2025 (r = 0.67 for omega_3 with binary data); PMC11573178 (kappa ICC = 0.04); project CLAUDE.md known limitation note.

---

### Pitfall 4: Specifying the Effect Size Prior for BFDA from a Single Literature Source

**What goes wrong:** BFDA requires specifying an expected effect size distribution (the H1 prior). Researchers commonly pick a Cohen's d from the closest published study and treat it as ground truth. For psilocybin-on-computational-parameters studies: (a) the literature is thin, (b) effect sizes from published studies are inflated by publication bias and winner's curse, and (c) the effect on HGF parameters specifically may differ substantially from the effect on simpler behavioral measures.

**Why it happens:** There is no established convention for HGF parameter effect sizes in psilocybin-PCS studies. The nearest comparators (psychedelic reversal learning studies; post-concussion computational studies) are methodologically different enough that naive effect size transfer is unreliable.

**Consequences:** If the assumed effect size is too large, the power analysis recommends a sample that is too small and the real study is underpowered. If it is too small, the power analysis recommends an unnecessarily large N. Schönbrodt et al. (2018) show that when the true effect size diverges greatly from the informed prior location, the efficiency benefit of the informed prior collapses, but the false-positive rate remains elevated.

**Prevention:**
- Run BFDA as a sweep over a range of effect sizes (e.g., d = 0.2, 0.4, 0.6, 0.8, 1.0) rather than a point estimate. Report the N required for each. Present the full power-by-effect-size surface.
- Treat the literature-derived estimate as the midpoint, not the only scenario.
- Clearly separate the "design prior" (what effect size we expect) from the "analysis prior" (the JZS prior used in computing the BF). These are different and must not be conflated.
- Flag that publication bias inflates literature effect sizes; use half the published estimate as a conservative scenario.

**Detection:** Sensitivity analysis: does the N recommendation change substantially if the assumed effect size changes by ±0.2? If yes, report the full range.

**Phase:** Effect size specification step (before sweep configuration).

**Sources:** Schönbrodt & Wagenmakers (2018); Stefan et al. (2019 tutorial in PMC6538819); Schreiber et al. (2024) bandit BFDA paper.

---

### Pitfall 5: anovaBF Random-Slope Misspecification for Repeated-Measures Interaction

**What goes wrong:** The standard BayesFactor R package function `anovaBF()` is misspecified for designs with two or more repeated-measures factors. It omits random slopes for within-subject factors, creating an RIO (random intercepts only) model that can inflate Bayes factors for interaction terms and produce false-positive evidence for interactions that do not exist. This study has at least two repeated-measures factors (session × phase) with a between-subjects factor (group), making it susceptible.

**Why it happens:** The `anovaBF()` function has been the default recommendation in the field for years. The misspecification was documented and corrected only in the 2023 update (van den Bergh et al. 2023), and many researchers continue using the old approach.

**Consequences:** The group × session interaction Bayes factor — the primary target of the power analysis — may be inflated. Power analysis built on an inflated BF test will predict correct power for the wrong test. When JASP or corrected BF functions are used in the actual study analysis, results may differ from the power analysis predictions.

**Prevention:**
- Do not use `anovaBF()` directly. Use `lmBF()` or `generalTestBF()` with random slopes manually specified.
- Alternatively, use JASP version 0.16.3+ which implements MRE (maximal random effects) by default.
- In Python: use bambi with mixed-effects specification and extract the Bayes factor via bridge sampling on competing models.
- Validate on simulated data: run the intended BF computation on a known null dataset and confirm the BF does not systematically exceed 3.

**Detection:** Cross-check any BF > 5 for interaction terms by computing with both RIO and MRE specifications. Large discrepancy indicates the misspecification is material for your data structure.

**Phase:** Group-level BF test implementation; BFDA simulation kernel.

**Sources:** Van den Bergh et al. 2023 (Psychological Methods); JASP blog 2022 (jasp-stats.org); BayesFactor package anovaBF documentation.

---

## Moderate Pitfalls

Mistakes that corrupt individual simulation cells or introduce systematic bias in subsets of the power curve.

---

### Pitfall 6: Using Too Few Simulations per Power Cell

**What goes wrong:** Each cell in the BFDA grid (N_subjects × effect_size × trial_count) needs enough simulation repetitions for the power estimate to be stable. With too few repetitions, Monte Carlo error inflates the variance of the power estimate, and the resulting power curve is jagged and untrustworthy.

**Prevention:**
- Use at least 1,000 repetitions per cell for exploratory sweeps, 2,000-3,000 for the final reportable power curve (MCE target ≤ 0.005, per LMU Power Simulation tutorial).
- During sweep design, start with 500 repetitions to locate the inflection zone, then re-run at 2,000 in that zone.
- Report Monte Carlo error as error bands on all power curves.

**Phase:** Sweep configuration; final power figure generation.

**Sources:** LMU power simulation tutorial (shiny.psy.lmu.de/r-tutorials/powersim/); Schönbrodt BFDA package documentation (nicebread/BFDA on GitHub).

---

### Pitfall 7: Conflating the Trial Count Sweep with a Sample Size Sweep

**What goes wrong:** Both N_subjects and N_trials affect power in a computational model pipeline, but through different mechanisms. N_subjects controls the group-level statistical power (the second-level BF test). N_trials controls how well MCMC can recover parameters (the first-level noise). These are not interchangeable. A common mistake is to run only the N_subjects sweep and assume trial count is fixed, then recommend an N that only works at the existing task length. Alternatively, researchers sweep N_trials and treat it as equivalent to sweeping power when in fact more trials helps recovery but does not replace more subjects.

**Why it happens:** Standard power analyses in non-computational designs only involve sample size. The additional dimension of trial count is unique to computational models and easy to overlook.

**Consequences:** The N recommendation is valid only for one specific trial count. If the actual study uses a different task length (due to participant fatigue, IRB constraints, or phase-criterion variability), the power estimate is wrong.

**Prevention:**
- Run a 2D sweep: N_subjects × N_trials. Present as a heatmap.
- Document the assumed trial count range explicitly and tie it to the config YAML trial structure.
- Flag that criterion-based reversals create variable effective trial counts per participant. Use a distribution of trial counts in simulation, not a fixed value.

**Phase:** Sweep design; trial count sweep implementation.

**Sources:** Schreiber et al. 2024 (Wellcome Open Research) — explicitly examined "number of games per participant" as a separate dimension from sample size in bandit BFDA.

---

### Pitfall 8: Partial Feedback Confound in Trial Count Assumptions

**What goes wrong:** In the PRL pick_best_cue task, only the chosen cue receives a reward signal. Unchosen cues are not updated. This means the effective information per trial depends on the choice distribution: if the participant is highly consistent (high stickiness), some cues accumulate very few observations and their belief nodes are essentially unconstrained by data. A power analysis that treats all 3 cues as equally observed overestimates the per-trial information content.

**Why it happens:** Partial observability is specific to this task type and rarely addressed in generic power analysis frameworks. HGF partial-feedback handling is documented in the project CLAUDE.md but may be forgotten when designing simulation parameters.

**Consequences:** Recovery of parameters for the least-chosen cue is worse than expected. This propagates to the group-level test because that cue's belief trajectory is noisier, adding variance to the softmax choice probabilities and inflating estimation uncertainty.

**Prevention:**
- When simulating participants for power analysis, use realistic choice distributions informed by the existing recovery analyses. Do not assume uniform exploration across cues.
- Check that the trial count sweep reflects variation in "effective observations per cue branch," not just total trial count.
- Document the partial-feedback assumption explicitly in power analysis outputs.

**Phase:** Simulation parameter design for BFDA; trial count sweep.

**Sources:** Project CLAUDE.md task structure; Wilson & Collins 2019 (eLife) — warns that tasks not sufficiently engaging target processes cause computational modeling to fail.

---

### Pitfall 9: Non-Independent Random Streams Across Parallel Jobs

**What goes wrong:** When 4,000+ SLURM job array tasks all seed numpy or JAX's PRNG from `np.random.seed(SLURM_ARRAY_TASK_ID)`, the seeds are sequentially correlated. If the PRNG's internal state from seed N quickly overlaps with the stream from seed N+1 (depends on PRNG period and advance rate), the simulated datasets across jobs are not statistically independent. This introduces subtle correlations in the BF distribution.

**Why it happens:** Using the SLURM task ID as the seed is a natural shortcut and looks like it guarantees uniqueness. It guarantees unique seeds but not independent streams.

**Consequences:** The Monte Carlo estimate of power is biased by inter-job correlation. The effect is usually small for modern PRNGs with large periods, but it is never zero and is not diagnosable post-hoc without re-running.

**Prevention:**
- Use numpy's `SeedSequence` API: `ss = np.random.SeedSequence(base_seed); child_seeds = ss.spawn(n_jobs); rng = np.random.default_rng(child_seeds[task_id])`. This guarantees statistically independent streams.
- For JAX (pyhgf): use `jax.random.PRNGKey(task_id)` with a fixed base key split — `jax.random.split(base_key, n_jobs)[task_id]`.
- Store the seed used in each job's output CSV for auditability.

**Phase:** SLURM job array implementation.

**Sources:** NumPy SeedSequence documentation (blog.scientific-python.org/numpy/numpy-rng/); reproducibility/PRNG tutorial (r-ega.net/articles/reproducibility-prng.html).

---

### Pitfall 10: SLURM Filesystem Saturation from Python Conda Import Storm

**What goes wrong:** A job array with 4,000 tasks launching simultaneously all import Python packages from a shared conda environment on the cluster filesystem (Lustre/GPFS). Each import triggers thousands of filesystem calls. With 4,000 concurrent jobs, this creates an "import storm" that saturates the metadata server and slows or hangs all jobs, often causing cascade failures with cryptic error messages unrelated to the actual code.

**Why it happens:** Standard conda environments are not optimized for high-concurrency import. The M3 MASSIVE cluster has documented this as a known issue for array jobs using Python environments stored on work disk (Monash eResearch docs; general HPC guidance).

**Consequences:** Jobs time out waiting on filesystem metadata, SLURM kills them, and the failed jobs produce empty output files. The aggregation step silently treats missing files as non-existent cells, leading to a power curve with holes that look like valid zeros.

**Prevention:**
- Throttle concurrent jobs: use `#SBATCH --array=0-3999%50` (max 50 concurrent) to stagger startup.
- Build a conda-pack archive of the environment and extract it to each node's local scratch (`$TMPDIR`) before import. This distributes filesystem load.
- Alternatively, run multiple simulations per job (e.g., 20 repetitions per task) rather than 1 per task, reducing total job count by 20× while maintaining the same total simulation count.
- After the run, check for missing output files before aggregating. Any missing file should cause a visible warning, not silent omission.

**Phase:** SLURM job array configuration; output aggregation.

**Sources:** Monash M3 array jobs documentation; general HPC Python filesystem guidance (docs.hpc.shef.ac.uk EmbarrassinglyParallel); SLURM job failure investigation (doc.hpc.iter.es).

---

### Pitfall 11: GPU Memory Exhaustion When PyMC JAX Sampling Accumulates Chains

**What goes wrong:** PyMC's JAX/NumPyro backend does not offload posterior samples from GPU memory until sampling completes. In a power loop running many short chains, if all chains are held in GPU memory simultaneously, the GPU OOM-kills the process. This is distinct from MCMC divergence — the job appears to hang or exit without useful diagnostics.

**Why it happens:** GPU sampling is fast for individual fits, but the memory model differs from CPU sampling. This is documented in PyMC discourse threads on `pm.sampling_jax.sample_numpyro_nuts()`.

**Consequences:** Silent job failure, especially for the 3-level model with longer chain requirements. Jobs that OOM-die produce no output, which aggregation treats as missing cells.

**Prevention:**
- For power loop fits on CPU (more robust for parallel jobs), use `pm.sample()` with `cores=1`, `chains=2`, and short runs (e.g., 500 tune + 500 draw).
- Reserve GPU sampling for the production single-subject fitting pipeline, not the power simulation loop where reliability matters more than speed.
- Test memory usage on a single node before launching the full array: `sacct -j <jobid> --format=MaxRSS`.

**Phase:** SLURM job configuration; MCMC backend selection for power loop.

**Sources:** PyMC discourse (discourse.pymc.io/t/reduce-memory-requirements-on-the-gpu); PyMC Labs benchmark (pymc-labs.com/blog-posts/pymc-stan-benchmark).

---

## Minor Pitfalls

Mistakes that are fixable post-hoc but waste time or require partial reruns.

---

### Pitfall 12: Confounding the Design Prior with the Analysis Prior

**What goes wrong:** BFDA uses two distinct priors: (a) the design prior on effect size, which specifies what effects are assumed when computing power; and (b) the JZS prior on effect size used in the Bayes factor computation itself. These are conceptually separate. A common mistake is to use the same prior for both (e.g., default Cauchy r = 0.707 for both), which makes the design prior a match for the analysis prior by construction and produces optimistic power estimates.

**Prevention:** Explicitly document which prior is used for which purpose. Run a sensitivity check where the design prior is more diffuse than the analysis prior to see whether power estimates hold.

**Phase:** BFDA configuration; analysis plan documentation.

**Sources:** Schönbrodt & Wagenmakers (2018); Stefan et al. (2019 tutorial).

---

### Pitfall 13: Mixing Model Recovery Power with Parameter Power

**What goes wrong:** The v1.1 milestone includes two distinct power analyses: (A) N × effect_size sweep for JZS BF on parameter group differences, and (B) model discriminability via BMS at varying N. These answer different questions and should not be merged into a single "power" number. Combining them leads to N recommendations that are simultaneously too conservative for one question and too liberal for the other.

**Prevention:** Keep the two power analyses in separate scripts with separate outputs and clearly labeled figures. The final N recommendation should specify which analysis drives it and for which parameter/model.

**Phase:** Output reporting; N recommendation synthesis.

**Sources:** Wilson & Collins 2019 (model recovery as distinct from parameter recovery); Rigoux et al. 2014 (BMS power).

---

### Pitfall 14: Ignoring Phase-Criterion Variability in Trial Count Assumptions

**What goes wrong:** The PRL task uses criterion-based reversal triggers, so the number of trials in each phase varies between simulated participants and will vary even more between real participants. A power analysis using a fixed trial count per participant will not match the distribution of trial counts in real data. The power estimate is implicitly conditional on a specific task experience that participants may not have.

**Prevention:** Simulate trial counts by sampling from a distribution (uniform or beta-distributed between the minimum and maximum expected trials per phase) rather than using a single fixed value. Document the assumed distribution in outputs.

**Phase:** Simulation parameter design; trial count sweep.

**Sources:** Project CLAUDE.md task structure (criterion-based reversal phases).

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|----------------|------------|
| Prechecks: parameter recovery | omega_3 appears recoverable in simulation but is not in real data (Pitfall 3) | Report recovery r with 95% CI; add recovery penalty variant |
| Prechecks: MCMC validation | Convergence failures silently accepted (Pitfall 1) | Mandatory R-hat/ESS column in every output; filter before aggregation |
| Trial count sweep | Sweep conflates trial count and sample size (Pitfall 7); partial feedback ignored (Pitfall 8) | 2D sweep; realistic choice distribution in simulation |
| SLURM job array setup | Import storm (Pitfall 10); non-independent seeds (Pitfall 9); GPU OOM (Pitfall 11) | Throttle concurrency; SeedSequence; CPU MCMC for power loop |
| Effect size specification | Single-point estimate from literature (Pitfall 4) | Sweep effect sizes d = 0.2 to 1.0 |
| Group × session BF computation | anovaBF misspecification for repeated measures (Pitfall 5) | Use lmBF with manual random slopes or JASP 0.16.3+ |
| N recommendation synthesis | Posterior mean attenuation bias (Pitfall 2); design vs. analysis prior confusion (Pitfall 12) | Report attenuation-corrected and -uncorrected curves; separate priors explicitly |
| Final figure and reporting | Model recovery power merged with parameter power (Pitfall 13) | Two separate figures, two separate N recommendations |

---

## Sources

- Wilson & Collins (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*. [PMC6879303](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879303/)
- Schönbrodt & Wagenmakers (2018). Bayes factor design analysis: Planning for compelling evidence. *Psychonomic Bulletin & Review*. [doi:10.3758/s13423-017-1230-y](https://link.springer.com/article/10.3758/s13423-017-1230-y)
- Stefan et al. (2019). A tutorial on Bayes Factor Design Analysis using an informed prior. *Behavior Research Methods*. [PMC6538819](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538819/)
- Hess et al. (2025). Bayesian Workflow for Generative Modeling in Computational Psychiatry. *Computational Psychiatry*. [PMC11951975](https://pmc.ncbi.nlm.nih.gov/articles/PMC11951975/)
- Schreiber et al. (2024). Enhancing experimental design through BFDA: insights from multi-armed bandit tasks. *Wellcome Open Research*. [doi:10.12688/wellcomeopenres.20041.2](https://wellcomeopenresearch.org/articles/9-423)
- Test-retest reliability of HGF parameters. [PMC11573178](https://pmc.ncbi.nlm.nih.gov/articles/PMC11573178/)
- Reliability of RL computational parameters. [PMC11104400](https://pmc.ncbi.nlm.nih.gov/articles/PMC11104400/)
- Van den Bergh et al. (2023). Bayesian Repeated-Measures ANOVA: Updated Methodology. *Psychological Methods*. [doi:10.1177/25152459231168024](https://journals.sagepub.com/doi/10.1177/25152459231168024)
- JASP blog: Bayesian RM-ANOVA random slope misspecification (2022). [jasp-stats.org](https://jasp-stats.org/2022/07/29/bayesian-repeated-measures-anova-an-updated-methodology-implemented-in-jasp/)
- LMU Power Simulation: how many Monte Carlo iterations? [shiny.psy.lmu.de/r-tutorials/powersim/](https://shiny.psy.lmu.de/r-tutorials/powersim/how_many_iterations.html)
- PyMC GPU memory issue: [discourse.pymc.io](https://discourse.pymc.io/t/reduce-memory-requirements-on-the-gpu-when-sampling-with-pm-sampling-jax-sample-numpyro-nuts/11596)
- NumPy best practices for RNG in parallel: [blog.scientific-python.org](https://blog.scientific-python.org/numpy/numpy-rng/)
- Monash M3 MASSIVE array jobs: [docs.erc.monash.edu](https://docs.erc.monash.edu/old-M3/M3/slurm/array-jobs/)
- Sheffield HPC embarrassingly parallel guide: [docs.hpc.shef.ac.uk](https://docs.hpc.shef.ac.uk/en/latest/parallel/EmbarrassinglyParallel.html)
