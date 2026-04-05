# Roadmap: PRL HGF Analysis Pipeline

## Overview

Build a validated HGF-based analysis pipeline for the PRL pick_best_cue task. The project progresses from task environment reproduction through model definition, simulation, fitting, recovery validation, group-level hypothesis testing, and an interactive parameter exploration GUI. Each phase produces independently verifiable outputs.

## Phases

- [x] **Phase 1: Foundation** — Repo scaffold, dependencies, unified config, task environment simulator
- [x] **Phase 2: Models** — 2-level and 3-level binary HGF networks in pyhgf, custom softmax+stickiness response function
- [x] **Phase 3: Simulation** — Agent simulator, batch synthetic data generation with group/session structure
- [ ] **Phase 4: Fitting** — Single-subject MCMC fitting via PyMC, batch fitting pipeline, diagnostics
- [ ] **Phase 5: Validation & Comparison** — Parameter recovery, model comparison (random-effects BMS)
- [ ] **Phase 6: Group Analysis** — Second-level statistics, group x session x phase effects, visualizations
- [ ] **Phase 7: GUI** — Interactive Jupyter widget for parameter exploration and belief trajectory visualization

## Phase Details

### Phase 1: Foundation
**Goal**: The project skeleton exists — dependencies install cleanly, the unified config loads and validates, and the task environment simulator generates correct trial sequences matching the PRL pick_best_cue structure.
**Depends on**: PRL task repo (for pick_best_cue.json config reference)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, INF-01, INF-02, INF-03, INF-05
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Project scaffold, pyproject.toml, unified YAML config, config loader with validation
- [x] 01-02-PLAN.md — Task environment simulator and comprehensive unit tests
**Success Criteria**:
  1. `pip install -e .` succeeds with all dependencies (pyhgf, jax, pymc, arviz, ipywidgets)
  2. The analysis config loads the PRL pick_best_cue task structure and extends it with analysis-specific fields (group definitions, simulation parameters)
  3. The environment simulator generates a trial sequence for one session: correct number of trials per phase, correct cue reward probabilities per phase, correct reversal schedule, each trial labeled stable/volatile
  4. Running with the same seed produces byte-identical output

### Phase 2: Models
**Goal**: Both HGF model variants are defined as pyhgf Network objects, forward-pass correctly on a synthetic input sequence, and the custom response function computes log-likelihood given beliefs and observed choices.
**Depends on**: Phase 1 (environment simulator for test inputs)
**Requirements**: MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, RSP-01, RSP-02, RSP-03, RSP-04
**Plans:** 2 plans
Plans:
- [x] 02-01-PLAN.md — 2-level and 3-level HGF model builders with forward pass and belief extraction
- [x] 02-02-PLAN.md — Softmax + stickiness response function with end-to-end integration tests
**Success Criteria**:
  1. The 2-level model creates 3 parallel binary HGF branches (3 input nodes, 3 continuous-state nodes), accepts a trial sequence, and produces belief trajectories mu_1 for each cue
  2. The 3-level model adds a shared volatility parent node and produces both mu_1 (per cue) and mu_2 (volatility) trajectories; the volatility trajectory responds visibly to reversal points
  3. The custom response function returns finite log-likelihood for a sequence of observed choices given HGF belief states and (beta=2, zeta=0.5) response parameters
  4. Parameter names map clearly to documented HGF parameters (omega_2, omega_3, kappa, beta, zeta)

### Phase 3: Simulation
**Goal**: Synthetic participants with known parameters produce realistic choice data, and batch simulation generates a complete group x session dataset ready for fitting.
**Depends on**: Phase 2 (model definitions + response function)
**Requirements**: SIM-01, SIM-02, SIM-03, SIM-04, SIM-05, SIM-06
**Plans:** 2 plans
Plans:
- [x] 03-01-PLAN.md — Single-agent simulator with trial-by-trial HGF loop and parameter sampling
- [x] 03-02-PLAN.md — Batch simulation orchestration, group x session structure, pipeline script
**Success Criteria**:
  1. A single simulated agent with high beta (=5) and correct omega_2 chooses the best cue >80% of the time during stable phases and shows transient accuracy drops after reversals
  2. Batch simulation generates 30 participants/group x 2 groups x 3 sessions = 180 synthetic datasets, each with full trial-level data and ground-truth parameters
  3. The two groups show visibly different parameter distributions (e.g., post-concussion group has lower kappa at baseline)
  4. Session 2 parameters shift from session 1 by the configured deltas, and the shift magnitude differs by group

### Phase 4: Fitting
**Goal**: The PyMC fitting pipeline recovers posterior distributions for each free parameter on individual simulated participants, with clean MCMC diagnostics.
**Depends on**: Phase 3 (simulated data to fit)
**Requirements**: FIT-01, FIT-02, FIT-03, FIT-04, FIT-05, INF-04
**Success Criteria**:
  1. Fitting a single simulated participant (2-level model, 4 chains x 1000 draws) converges: all R-hat < 1.05, ESS > 400 for omega_2, beta, zeta
  2. Posterior means are within 1 SD of the true generating parameters for omega_2 and beta on at least 80% of test fits
  3. Batch fitting pipeline processes all 180 synthetic datasets (both model variants) and saves results to a structured DataFrame with columns: participant_id, group, session, model, parameter, mean, sd, hdi_3%, hdi_97%, r_hat, ess
  4. Unit tests verify response function gradient is finite and MCMC sampling does not diverge on edge-case parameter values

### Phase 5: Validation & Comparison
**Goal**: Parameter recovery is verified (or limitations documented), and formal model comparison identifies whether the 3-level model is justified by the data.
**Depends on**: Phase 4 (fitted posteriors for all participants x both models)
**Requirements**: REC-01, REC-02, REC-03, REC-04, CMP-01, CMP-02, CMP-03, CMP-04
**Success Criteria**:
  1. Parameter recovery plots show r > 0.7 for omega_2, beta, zeta; omega_3 and kappa recovery quality is explicitly documented (expected: potentially r < 0.7 for omega_3)
  2. Parameter correlation matrix reveals no severe confounds (|r| > 0.8 between any two parameters triggers a documented concern)
  3. Random-effects BMS produces exceedance probabilities for 2-level vs. 3-level across the full sample and per-group; results match expectations (3-level should win when data was generated from a 3-level process)
  4. Model comparison summary table and exceedance probability bar plot are generated

### Phase 6: Group Analysis
**Goal**: The pipeline tests the primary hypotheses: group x session interactions on HGF parameters, phase-stratified learning rate effects, and produces publication-quality figures.
**Depends on**: Phase 5 (validated parameter estimates + model selection)
**Requirements**: GRP-01, GRP-02, GRP-03, GRP-04, GRP-05
**Success Criteria**:
  1. Mixed-effects model (or repeated-measures ANOVA) detects the simulated group x session interaction on omega_2 with p < 0.05 (since the effect was simulated, this validates power)
  2. Phase-stratified analysis (stable vs. volatile) shows different effective learning rates, and the group x phase interaction is testable
  3. Raincloud plots show parameter distributions by group x session for all parameters of interest
  4. Effect sizes are computed and reported for all primary comparisons

### Phase 7: GUI
**Goal**: An interactive Jupyter widget allows real-time exploration of how HGF parameters affect belief trajectories, learning rates, and choice probabilities on the PRL task environment.
**Depends on**: Phase 2 (models) + Phase 1 (environment)
**Requirements**: GUI-01, GUI-02, GUI-03, GUI-04, GUI-05, GUI-06
**Success Criteria**:
  1. Opening the GUI notebook in VSCode shows sliders for omega_2, omega_3, kappa, beta, zeta and a multi-panel plot
  2. Moving any slider updates the plot within <2 seconds (fast enough for interactive exploration)
  3. Toggling between 2-level and 3-level model shows/hides the volatility trajectory panel
  4. Pre-set parameter profiles ("healthy baseline", "post-concussion", "post-psilocybin") load instantly and produce visibly different belief/choice patterns
  5. Ground-truth reward probabilities are overlaid on belief trajectories so the user can see tracking accuracy

## Proposed Project Structure

```
prl-hgf-analysis/
├── .planning/              # Planning docs (this file, PROJECT.md, REQUIREMENTS.md)
├── configs/
│   ├── task_config.json    # Imported/adapted from PRL task pick_best_cue.json
│   └── analysis_config.yaml # Analysis-specific: groups, simulation params, priors, fitting settings
├── src/
│   ├── environment/        # Task environment simulator
│   │   ├── __init__.py
│   │   └── task_env.py     # Trial sequence generator from config
│   ├── models/             # HGF model definitions
│   │   ├── __init__.py
│   │   ├── hgf_2level.py   # 2-level binary HGF (3 parallel branches)
│   │   ├── hgf_3level.py   # 3-level binary HGF (shared volatility parent)
│   │   └── response.py     # Custom softmax + stickiness response function
│   ├── simulation/         # Synthetic data generation
│   │   ├── __init__.py
│   │   ├── agent.py        # Single-agent simulator
│   │   └── batch.py        # Batch simulation with group/session structure
│   ├── fitting/            # Model fitting pipeline
│   │   ├── __init__.py
│   │   ├── fit_single.py   # Single-subject MCMC fitting
│   │   ├── fit_batch.py    # Batch fitting loop
│   │   └── diagnostics.py  # MCMC diagnostic checks
│   ├── analysis/           # Recovery, comparison, group stats
│   │   ├── __init__.py
│   │   ├── recovery.py     # Parameter recovery analysis
│   │   ├── model_comparison.py  # BMS (random-effects)
│   │   └── group_stats.py  # Second-level mixed-effects
│   └── gui/                # Interactive widget
│       ├── __init__.py
│       └── explorer.py     # Jupyter widget for parameter exploration
├── notebooks/
│   ├── 01_environment_demo.ipynb
│   ├── 02_model_demo.ipynb
│   ├── 03_simulation.ipynb
│   ├── 04_fitting.ipynb
│   ├── 05_recovery_and_comparison.ipynb
│   ├── 06_group_analysis.ipynb
│   └── 07_parameter_explorer.ipynb  # GUI notebook
├── tests/
│   ├── test_environment.py
│   ├── test_models.py
│   ├── test_response.py
│   └── test_fitting.py
├── results/                # Generated outputs (gitignored except summaries)
├── pyproject.toml
└── README.md
```

## Progress

**Execution Order:** Phases 1 -> 2 -> 3 -> 4 -> 5 -> 6; Phase 7 can start after Phase 2.

| Phase | Status | Completed |
|-------|--------|-----------|
| 1. Foundation | Complete | 2026-04-04 |
| 2. Models | Complete | 2026-04-05 |
| 3. Simulation | Complete | 2026-04-05 |
| 4. Fitting | Not started | -- |
| 5. Validation & Comparison | Not started | -- |
| 6. Group Analysis | Not started | -- |
| 7. GUI | Not started | -- |

## Key Risks

| Risk | Mitigation |
|------|-----------|
| omega_3 parameter recovery is poor | Document explicitly; focus primary hypotheses on omega_2 and kappa; consider fixing omega_3 and comparing models with/without it |
| pyhgf Network API doesn't support 3 parallel binary inputs with shared volatility parent cleanly | Verify in Phase 2; fallback: build 3 separate HGFs and manually couple them |
| PyMC + pyhgf integration breaks with JAX version mismatch | Pin exact JAX version in pyproject.toml; test in CI before proceeding |
| MCMC fitting too slow for 180 participants x 2 models | Profile in Phase 4; consider MAP estimation as fast fallback, use MCMC for subset |
| Partial feedback (only chosen cue updated) creates identifiability issues | Standard approach in literature; verify via recovery analysis in Phase 5 |

---
*Roadmap created: 2026-04-04*
