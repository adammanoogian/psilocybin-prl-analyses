# Requirements: PRL HGF Analysis Pipeline

**Defined:** 2026-04-04
**Core Value:** A validated simulation-to-inference pipeline that can recover known parameters and test group × session hypotheses before real data arrives.

## v1 Requirements

### Config & Environment (ENV)

- [ ] **ENV-01**: Unified analysis config (YAML or JSON) that imports/extends the PRL pick_best_cue task config — must define: cue reward probabilities per phase, reversal schedule, trial counts, timing of stable vs. volatile phases
- [ ] **ENV-02**: Task environment simulator that generates trial-by-trial input sequences: for each trial, which cue was "best" (ground truth reward probabilities), what feedback the agent would receive given a choice — deterministic or probabilistic per config
- [ ] **ENV-03**: Environment correctly reproduces PRL task structure: 3 sets × (acquisition + N reversals) + transfer phase, with criterion-based phase transitions replaced by fixed trial counts (simulation doesn't have a sliding-window participant)
- [ ] **ENV-04**: Stable vs. volatile phase labeling: each trial tagged as 'stable' (acquisition, post-final-reversal) or 'volatile' (reversal phases) for phase-stratified parameter analysis
- [ ] **ENV-05**: RNG seeding: environment generation reproducible from a single integer seed

### Models (MOD)

- [ ] **MOD-01**: 2-level binary HGF with 3 parallel input nodes (one per cue) using pyhgf Network API — each branch has binary-state → continuous-state architecture
- [ ] **MOD-02**: 3-level binary HGF with 3 parallel input nodes sharing a single volatility parent at level 2 — extends 2-level with volatility coupling
- [ ] **MOD-03**: Both models accept trial-by-trial binary input: for the chosen cue, input=1 if rewarded, input=0 if not; unchosen cues receive no update (partial feedback)
- [ ] **MOD-04**: Forward simulation (filtering): given parameters + input sequence, produce belief trajectories (μ₁, σ₁ for each cue; μ₂, σ₂ for volatility in 3-level)
- [ ] **MOD-05**: Model parameters exposed as named dict with clear mapping to pyhgf node parameters: ω₂ → tonic_volatility on level-1 nodes, ω₃ → tonic_volatility on level-2 node, κ → volatility_coupling on level-1→level-2 edge

### Response Model (RSP)

- [ ] **RSP-01**: Custom softmax + stickiness response function compatible with pyhgf's surprise() API — takes HGF network state + response parameters + observed choices, returns per-trial log-likelihood
- [ ] **RSP-02**: Softmax over 3 cues: P(choice=k) ∝ exp(β * μ₁ₖ + ζ * 𝟙[prev_choice=k]) where μ₁ₖ is the sigmoid-transformed level-1 belief for cue k
- [ ] **RSP-03**: Response function parameters: β (inverse temperature, >0) and ζ (stickiness, can be positive or negative)
- [ ] **RSP-04**: Response function handles first trial (no previous choice) gracefully — stickiness term is 0

### Simulation (SIM)

- [ ] **SIM-01**: Agent simulator: given a model (2-level or 3-level), parameter set, and environment trial sequence, generate a complete simulated dataset — choices, rewards, belief trajectories per trial
- [ ] **SIM-02**: Batch simulation: generate N_per_group × N_groups × N_sessions synthetic datasets with group-specific parameter distributions
- [ ] **SIM-03**: Group parameter specification: each group defined by (mean, sd) for each free parameter; individual parameters drawn from Gaussian (or transformed Gaussian for constrained params)
- [ ] **SIM-04**: Session parameter specification: parameters can shift across sessions via additive deltas (Δω₂_session2, Δω₂_session3, etc.) to simulate treatment effects
- [ ] **SIM-05**: Group × session interaction: the deltas can differ by group (e.g., post-concussion group shows larger Δκ after psilocybin)
- [ ] **SIM-06**: Output format: tidy CSV/DataFrame with columns: participant_id, group, session, trial, cue_chosen, reward, cue_0_prob, cue_1_prob, cue_2_prob, phase_label, true_params (as JSON or separate cols)

### Fitting (FIT)

- [ ] **FIT-01**: Single-subject fitting via PyMC + pyhgf HGFDistribution — MCMC sampling of perceptual model parameters (ω₂, and optionally ω₃, κ) and response parameters (β, ζ)
- [ ] **FIT-02**: Priors: documented, justified priors for each free parameter with references to literature defaults (e.g., ω₂ ~ Normal(-3, 2), β ~ HalfNormal(0, 5))
- [ ] **FIT-03**: MCMC diagnostics: R-hat, ESS, trace plots saved per fit; fits flagged if R-hat > 1.05 or ESS < 400
- [ ] **FIT-04**: Batch fitting: loop over all simulated participants, save posterior summaries (mean, sd, HDI) to a results DataFrame
- [ ] **FIT-05**: Fit both 2-level and 3-level models to each participant's data for model comparison

### Parameter Recovery (REC)

- [ ] **REC-01**: Recovery analysis: scatter plots of true vs. recovered parameter values (posterior mean) for each free parameter, with Pearson r and bias metrics
- [ ] **REC-02**: Recovery must demonstrate r > 0.7 for ω₂, β, ζ to consider them interpretable; flag ω₃ and κ recovery quality explicitly (known difficulty)
- [ ] **REC-03**: Confusion/correlation matrix between parameters to check for trade-offs (e.g., β-ζ correlation, ω₂-κ correlation)
- [ ] **REC-04**: Recovery run with at least N=30 simulated participants per parameter regime

### Model Comparison (CMP)

- [ ] **CMP-01**: Per-subject model evidence: WAIC or LOO-CV computed via ArviZ for each model (2-level, 3-level) on each participant
- [ ] **CMP-02**: Random-effects Bayesian model selection (Rigoux et al. 2014): expected posterior probability and exceedance probability for each model at the group level
- [ ] **CMP-03**: Model comparison stratified by group: does the 3-level model fit better for one group? Report protected exceedance probability
- [ ] **CMP-04**: Model comparison results summary table and visualization

### Group Analysis (GRP)

- [ ] **GRP-01**: Extract point estimates (posterior mean or MAP) of each fitted parameter per participant
- [ ] **GRP-02**: Second-level mixed-effects model: group (between) × session (within, 3 levels) on each parameter of interest (ω₂, κ, β, ζ, and ω₃ if recovery is adequate)
- [ ] **GRP-03**: Phase-stratified analysis: fit separate models (or extract phase-specific trajectories) for stable vs. volatile trials; test group × phase interaction on effective learning rate
- [ ] **GRP-04**: Visualization: raincloud plots / violin plots of parameter distributions by group × session; interaction plots for key effects
- [ ] **GRP-05**: Effect sizes (Cohen's d or partial η²) for all group comparisons

### Interactive GUI (GUI)

- [ ] **GUI-01**: Jupyter widget (ipywidgets) — runs in VSCode Jupyter notebook
- [ ] **GUI-02**: Parameter sliders for: ω₂, ω₃, κ, β, ζ, μ₁⁰ (initial belief), μ₃⁰ (initial volatility prior)
- [ ] **GUI-03**: Real-time plot updates showing: (a) belief trajectories μ₁ for each cue across trials, (b) volatility estimate μ₂ (3-level only), (c) trial-by-trial choice probabilities from softmax, (d) effective learning rate (precision ratio) over time
- [ ] **GUI-04**: Task environment visualization: ground-truth reward probabilities overlaid on belief trajectories
- [ ] **GUI-05**: Toggle between 2-level and 3-level model
- [ ] **GUI-06**: Pre-set parameter profiles: "healthy baseline", "post-concussion", "post-psilocybin" for quick comparison

### Infrastructure (INF)

- [ ] **INF-01**: pyproject.toml with pinned dependencies: pyhgf>=0.2.8, jax, pymc>=5.0, arviz, numpy, pandas, matplotlib, ipywidgets, scipy
- [ ] **INF-02**: Reproducible environment: all random seeds propagated through config; MCMC chains serializable
- [ ] **INF-03**: Project structure: src/ for library code, notebooks/ for analysis workflows, configs/ for experiment + analysis configs, tests/ for unit tests
- [ ] **INF-04**: Unit tests for: environment simulator output shape/values, response function log-likelihood computation, parameter recovery on a 3-participant mini-batch
- [ ] **INF-05**: README with setup instructions, quick-start notebook link, and architecture diagram

## v2 Requirements (Deferred)

- **V2-01**: Real data ingestion: parse PRL task CSV output (29-column spec) into HGF input format
- **V2-02**: Active inference comparator model (as in Mason et al. 2024)
- **V2-03**: Reduced Bayesian Observer (κ=0) as additional null model
- **V2-04**: Independent-volatility variant (each cue has own volatility node)
- **V2-05**: Neuroimaging regressors: extract trial-by-trial prediction error and uncertainty trajectories for fMRI GLM
- **V2-06**: Hierarchical/multilevel Bayesian fitting (all participants fit simultaneously with group-level hyperpriors)
- **V2-07**: Simulate and test active inference model with engagement parameter (forgetting rate, loss aversion — per Mason et al.)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | 1 | — |
| ENV-02 | 1 | — |
| ENV-03 | 1 | — |
| ENV-04 | 1 | — |
| ENV-05 | 1 | — |
| MOD-01 | 2 | — |
| MOD-02 | 2 | — |
| MOD-03 | 2 | — |
| MOD-04 | 2 | — |
| MOD-05 | 2 | — |
| RSP-01 | 2 | — |
| RSP-02 | 2 | — |
| RSP-03 | 2 | — |
| RSP-04 | 2 | — |
| SIM-01 | 3 | — |
| SIM-02 | 3 | — |
| SIM-03 | 3 | — |
| SIM-04 | 3 | — |
| SIM-05 | 3 | — |
| SIM-06 | 3 | — |
| FIT-01 | 4 | — |
| FIT-02 | 4 | — |
| FIT-03 | 4 | — |
| FIT-04 | 4 | — |
| FIT-05 | 4 | — |
| REC-01 | 5 | — |
| REC-02 | 5 | — |
| REC-03 | 5 | — |
| REC-04 | 5 | — |
| CMP-01 | 5 | — |
| CMP-02 | 5 | — |
| CMP-03 | 5 | — |
| CMP-04 | 5 | — |
| GRP-01 | 6 | — |
| GRP-02 | 6 | — |
| GRP-03 | 6 | — |
| GRP-04 | 6 | — |
| GRP-05 | 6 | — |
| GUI-01 | 7 | — |
| GUI-02 | 7 | — |
| GUI-03 | 7 | — |
| GUI-04 | 7 | — |
| GUI-05 | 7 | — |
| GUI-06 | 7 | — |
| INF-01 | 1 | — |
| INF-02 | 1 | — |
| INF-03 | 1 | — |
| INF-04 | 4 | — |
| INF-05 | 1 | — |

**Coverage:** 49 v1 requirements across 7 phases.

---
*Requirements defined: 2026-04-04*
