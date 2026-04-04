# PRL HGF Analysis Pipeline

## What This Is

A Python pipeline for simulating, fitting, and comparing Hierarchical Gaussian Filter (HGF) models on data from the PRL (Probabilistic Reversal Learning) pick_best_cue task. Two model variants — 2-level and 3-level binary HGF — are applied to a 2-group × 3-session longitudinal design studying psilocybin effects on belief updating in post-concussion syndrome.

## Core Value

A validated simulation-to-inference pipeline: simulate synthetic participants with known parameters, recover those parameters via Bayesian fitting, compare models formally, and test group × session hypotheses — all before real data arrives. Every step is reproducible from a single shared config that mirrors the PRL task structure.

## Study Design

- **Groups**: (1) post-concussion + psilocybin, (2) healthy controls + psilocybin
- **Sessions**: 3 timepoints — baseline (pre-dose), post-dose, follow-up
- **Task**: pick_best_cue mode — 3 cues with different reward values, participant selects highest-value cue; criterion-based reversals create volatile phases within sets
- **Hypothesis direction**: Two-tailed. Post-concussion group may show altered belief updating (e.g., reduced flexibility, different volatility priors); psilocybin may modulate these parameters differentially across groups

## Model Architecture

### Perceptual Model

Three parallel binary HGF branches — one per cue — each tracking the reward probability of selecting that cue.

**2-level HGF** (per cue branch):
- Level 0: Binary input node (reward = 1 / no-reward = 0 for the chosen cue)
- Level 1: Continuous state node — tracks log-odds of cue being rewarding
- Free parameter: ω₂ (tonic volatility / evolution rate at level 1)

**3-level HGF** (per cue branch):
- Levels 0–1: Same as 2-level
- Level 2: Volatility parent — estimates how much the cue's reward probability is changing
- Additional free parameters: ω₃ (tonic meta-volatility), κ (coupling between volatility estimate and level-1 learning rate)
- Optional: m₃ (mean-reversion equilibrium for volatility belief, prevents drift)

All three branches can optionally share a common volatility parent at level 2 (shared-volatility variant) or each have independent volatility nodes (independent-volatility variant). Literature default: shared volatility parent.

### Response Model

Softmax over the three cue beliefs with:
- β (inverse temperature): exploitation vs. exploration
- ζ (stickiness/perseveration): bias toward repeating previous choice

Choice probability for cue k:
```
P(choice=k) = softmax(β * μ₁ₖ + ζ * 𝟙[previous_choice=k])
```
where μ₁ₖ is the expected value belief about cue k from the HGF.

### Parameters of Interest

| Parameter | Model | Meaning | Hypothesis relevance |
|-----------|-------|---------|---------------------|
| ω₂ | Both | Tonic volatility (≈ log-space learning rate) | Δω₂ across sessions = learning rate change |
| ω₃ | 3-level | Meta-volatility (how quickly volatility beliefs update) | Sensitivity to environmental change |
| κ | 3-level | Coupling strength (volatility → learning rate) | How much perceived volatility drives flexible updating |
| β | Both | Inverse temperature (decision noise) | Confidence / exploitation |
| ζ | Both | Stickiness (choice perseveration) | Cognitive flexibility vs. rigidity |
| μ₃⁰ | 3-level | Initial volatility prior | Prior expectation of environmental change |

**Primary outcomes:**
1. Δω₂ across sessions (within-subject learning rate change)
2. Δω₂ across stable vs. volatile phases (acquisition vs. reversal)
3. Group × session interactions on ω₂, κ, ω₃
4. Model comparison: does the 3-level (volatility-tracking) model fit better in one group?

### Other HGF Variants (considered, not in v1)

| Variant | Description | Why deferred |
|---------|-------------|-------------|
| AR-HGF | Autoregressive drift at higher levels | Adds complexity; standard HGF is more comparable to literature |
| eHGF | Enhanced nonlinear coupling functions | Novel, less validated in reversal learning |
| Categorical HGF | Single categorical distribution over 3 cues | Elegant but non-standard; harder to compare with binary HGF literature |
| Reduced Bayesian Observer | 2-level with κ=0 (≡ Rescorla-Wagner) | Useful null model; can add in v2 if needed |
| Independent vs. shared volatility | Each cue has its own volatility node | Worth exploring; shared is default in literature |

## Requirements

### Active

- [ ] Unified config file that mirrors PRL task pick_best_cue structure (reads from or extends the same JSON)
- [ ] Task environment simulator: generates trial sequences from config (acquisition + reversals + transfer)
- [ ] 2-level binary HGF with 3 parallel inputs via pyhgf Network API
- [ ] 3-level binary HGF with shared volatility parent via pyhgf Network API
- [ ] Custom softmax + stickiness response function compatible with pyhgf surprise API
- [ ] Agent simulator: given model + parameters, generates synthetic choice data for a full session
- [ ] Batch simulation: generate N synthetic participants per group × session with parameterized group differences
- [ ] Single-subject model fitting via PyMC + HGFDistribution (MCMC sampling)
- [ ] Parameter recovery analysis: fit simulated data, compare recovered vs. true parameters
- [ ] Formal Bayesian model selection: random-effects BMS (Rigoux et al. 2014) comparing 2-level vs. 3-level
- [ ] Second-level group analysis: group × session mixed-effects on fitted parameters
- [ ] Interactive Jupyter widget GUI: adjust parameters, visualize belief trajectories + choice probabilities in real time
- [ ] Reproducible environment: pyproject.toml with pinned dependencies, seed-controlled RNG throughout

### Out of Scope

- Real data ingestion (no data yet; pipeline validates on synthetic data)
- RLWM pipeline integration (separate project)
- Neuroimaging regressors from HGF trajectories (v2)
- Active inference model (used in Mason et al. 2024; could be a v2 comparator)

## Constraints

- **pyhgf version**: Pin to 0.2.8 (latest stable with PyMC integration)
- **Python**: 3.10+ (JAX compatibility)
- **Config-driven**: Task structure parameters read from the PRL config JSON — no hardcoded task sequences
- **Reproducibility**: All simulations seeded; all MCMC chains saved; all figures regenerable from saved posteriors
- **Parameter recovery caveat**: ω₃ recovery is known to be poor (Diaconescu et al., review literature). Must verify before interpreting group effects on this parameter.

## Key References

- Mathys et al. (2011, 2014) — HGF theory and binary HGF
- Weber et al. (2024) — Generalized HGF (gHGF)
- Legrand et al. (2024) — pyhgf paper (arXiv:2410.09206)
- Mason et al. (2024) — Psilocybin + reversal learning + computational modeling (Translational Psychiatry)
- Diaconescu et al. — HGF applied to reversal learning in clinical populations
- Rigoux et al. (2014) — Random-effects Bayesian model selection
- Stephan et al. (2009) — Bayesian model selection for group studies

---
*Last updated: 2026-04-04 after planning discussion*
