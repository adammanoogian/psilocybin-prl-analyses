# PRL HGF Analysis — AI Assistant Guidelines

## Project Overview

HGF-based analysis pipeline for the PRL pick_best_cue task studying psilocybin
effects on belief updating in post-concussion syndrome (psilocybin vs placebo × 3 sessions).

Two model variants (2-level and 3-level binary HGF) with three parallel cue
branches and a shared softmax + stickiness response model. Validated via
simulation-to-inference before real data arrives.

## Key Paths

```
config.py          # Root path constants (PROJECT_ROOT, DATA_DIR, CONFIGS_DIR, …)
configs/
  prl_analysis.yaml  # Single source of truth for task + analysis parameters
src/prl_hgf/
  env/             # Task environment: config loading + trial sequence generation
  models/          # HGF model definitions (pyhgf Network API)
  fitting/         # Bayesian fitting (PyMC + HGFDistribution)
  analysis/        # Group-level analysis + BMS
scripts/           # Numbered pipeline: 01_*, 02_*, …
tests/             # Unit + integration tests
validation/        # Scientific validation (parameter recovery)
```

## Task Structure (PRL pick_best_cue)

- **3 cues** with distinct reward probabilities
- **4 phases**: 2 acquisition phases + 2 reversal phases (criterion-based)
- **Partial feedback**: only the chosen cue gets a reward signal — unchosen cues
  are NOT updated
- All task parameters come from `configs/prl_analysis.yaml` — never hardcode
  trial counts, reward probabilities, or phase structure

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
