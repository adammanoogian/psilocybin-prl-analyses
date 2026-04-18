# HGF Analysis Toolbox — AI Assistant Guidelines

## Project Overview

General-purpose Hierarchical Gaussian Filter (HGF) analysis toolbox for
reversal-learning paradigms. Two HGF variants (2-level and 3-level binary
HGF) paired with task-specific response models, batched JAX/BlackJAX NUTS
fitting, VB-Laplace fast-path fitting, and config-driven task definitions.
Validated via simulation-to-inference.

**Supported task configurations** (add more via new YAML + loader module):

- **`pick_best_cue`** (`configs/prl_analysis.yaml`): 3-cue partial-feedback
  PRL with criterion-based reversals. Original use case: longitudinal
  psilocybin vs placebo × 3 sessions study on post-concussion participants.
- **`pat_rl`** (`configs/pat_rl.yaml`): binary-state safe/dangerous
  approach/avoid reversal learning with 2x2 reward/shock magnitudes,
  hazard-driven reversals, trial-level ΔHR autonomic covariate.

Tasks plug in by adding a `configs/<task>.yaml` + matching loader in
`src/prl_hgf/env/`. The fitting + analysis stack is task-agnostic.

## Key Paths

```
config.py          # Root path constants (PROJECT_ROOT, DATA_DIR, CONFIGS_DIR, …)
configs/
  prl_analysis.yaml  # pick_best_cue task (psilocybin use case)
  pat_rl.yaml        # PAT-RL approach-avoidance task
src/prl_hgf/
  env/             # Task configs + trial sequence generators (one per task)
  models/          # HGF builders + response models (one set per task)
  fitting/         # Batched NUTS (BlackJAX) + VB-Laplace orchestrators
  analysis/        # Group-level analysis, BMS, trajectory export
  power/           # BFDA power analysis (pick_best_cue)
  simulation/      # JAX-native cohort simulation
scripts/           # Numbered pipeline: 01_*, 02_*, …
tests/             # Unit + integration tests
validation/        # Scientific validation (parameter recovery)
cluster/           # SLURM scripts for M3/MASSIVE-style clusters
```

## Task Structures

### pick_best_cue (3-cue PRL)

- **3 cues** with distinct reward probabilities
- **4 phases**: 2 acquisition phases + 2 reversal phases (criterion-based)
- **Partial feedback**: only the chosen cue gets a reward signal — unchosen
  cues are NOT updated
- All task parameters come from `configs/prl_analysis.yaml` — never hardcode
  trial counts, reward probabilities, or phase structure

### PAT-RL (binary-state approach-avoid)

- **Binary state**: safe (0) / dangerous (1), hazard-rate-driven reversals
- **192 trials** across 4 runs (48 trials each)
- **2x2 magnitudes**: reward_mag ∈ {low, high} × shock_mag ∈ {low, high}
- **ΔHR covariate**: trial-level anticipatory bradycardia (caller-supplied)
- All task parameters come from `configs/pat_rl.yaml`

## Model Parameters

| Parameter | Model   | Meaning                             |
|-----------|---------|-------------------------------------|
| ω₂        | Both    | Tonic volatility (log-space LR)     |
| ω₃        | 3-level | Meta-volatility                     |
| κ         | 3-level | Coupling: volatility → LR           |
| β         | Both    | Inverse temperature (decision noise)|
| ζ         | Both    | Stickiness / choice perseveration   |
| μ₃⁰       | 3-level | Initial volatility prior            |

**Caveat**: ω₃ recovery is known to be poor in the literature. Primary
hypotheses focus on ω₂ and κ. Verify recovery before interpreting group
effects on ω₃.

## Python Conventions

- `from __future__ import annotations` at top of every module
- NumPy-style docstrings (enforced by ruff pydocstyle)
- Type hints: Python 3.10+ native syntax (`list[float]`, `str | None`)
- Line length 88, formatter/linter: ruff, type checker: mypy
- Three-layer naming: math symbols inside class internals, descriptive names
  at API boundaries, domain English in scripts
- Fitted/computed attributes use trailing underscore: `K_`, `x_post_`
- No wildcard imports. Absolute imports only.
- Config via `config.py` Path constants — no scattered hardcoded paths
- Tests in `tests/` (unit/integration), `validation/` (scientific)

## Architecture Decisions

- **pyhgf 0.2.8**: JAX-backed HGF with PyMC integration, Network API for
  custom graph topologies
- **3 parallel binary branches** with shared volatility parent (literature
  default)
- **Softmax + stickiness**: `P(k) = softmax(β·μ₁ₖ + ζ·𝟙[prev=k])`
- **Random-effects BMS** (Rigoux et al. 2014) for model comparison
- **Mixed-effects second level**: group × session × phase
- **Jupyter ipywidgets** for GUI (VSCode compatible)
- All simulations seeded; all MCMC chains saved

## Workflow

- **Always push after committing** — this repo is worked on from both a local
  machine and the M3 cluster. Every commit must be pushed so the other side
  can `git pull` before running.

## Development Phases

1. Foundation (this plan): package skeleton, config system
2. Task environment: trial sequence generator from config
3. HGF models: 2-level and 3-level via pyhgf
4. Simulation: synthetic participants with known parameters
5. Fitting: single-subject MCMC via PyMC
6. Parameter recovery: validate before real data
7. Group analysis + GUI
