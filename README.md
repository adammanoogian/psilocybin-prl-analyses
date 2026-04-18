# HGF Analysis Toolbox

General-purpose Hierarchical Gaussian Filter (HGF) analysis pipeline for
reversal-learning paradigms. Batched JAX/BlackJAX NUTS fitting, VB-Laplace
fast-path fitting, config-driven task definitions, and parameter-recovery
validation.

## Overview

Two HGF variants (2-level and 3-level binary Hierarchical Gaussian Filter)
paired with task-specific response models. Tasks are described entirely in
YAML so the same fitting/analysis pipeline runs across paradigms.

Validated via simulation-to-inference: simulate synthetic participants with
known parameters, recover them via Bayesian fitting, compare models formally
via random-effects BMS and Bayes Factor Design Analysis (BFDA).

### Supported task configurations

| Task | Config | Paradigm |
|------|--------|----------|
| `pick_best_cue` (default) | `configs/prl_analysis.yaml` | 3-cue partial-feedback PRL with criterion-based reversals; longitudinal psilocybin study use case (2 groups × 3 sessions) |
| `pat_rl` | `configs/pat_rl.yaml` | Binary-state (safe/dangerous) approach/avoid reversal-learning task with 2x2 reward/shock magnitudes, hazard-driven reversals, trial-level ΔHR autonomic covariate |

Additional tasks plug in by adding a `configs/<task>.yaml` + matching loader
module in `src/prl_hgf/env/`. The fitting + analysis stack is task-agnostic.

### Use cases

- **Psilocybin PRL study** (original): post-concussion psilocybin vs placebo
  × 3 sessions longitudinal design on pick_best_cue. See `configs/prl_analysis.yaml`
  and `notebooks/` for hypothesis-level details.
- **Approach-avoidance (PAT-RL) studies**: binary-state reversal with
  autonomic covariate modulation. See `configs/pat_rl.yaml` and
  `docs/PAT_RL_API_HANDOFF.md` for the PAT-RL public API.
- **Custom task**: author a new YAML + loader; reuse HGF builders, fitting
  path, and trajectory export without modification.

## Setup

### Prerequisites

- Python 3.10+
- Conda or venv recommended

### Install

```bash
git clone <repo-url>
cd <repo-dir>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Verify Installation

```python
import prl_hgf
print(prl_hgf.__version__)  # 0.1.0

# Default pick_best_cue config
from prl_hgf.env.task_config import load_config
cfg = load_config()
print(cfg.task.n_cues, len(cfg.task.phases))  # 3 4

# PAT-RL config
from prl_hgf.env.pat_rl_config import load_pat_rl_config
pcfg = load_pat_rl_config()
print(pcfg.task.n_trials, pcfg.task.n_runs)  # 192 4
```

### Run Tests

```bash
pytest                              # full suite
pytest -m "not slow"                # fast tests only
```

### Lint and Type Check

```bash
ruff check src/ config.py
mypy src/ config.py
```

## Project Structure

```
config.py              # Path constants (PROJECT_ROOT, DATA_DIR, …)
configs/
  prl_analysis.yaml    # pick_best_cue task
  pat_rl.yaml          # PAT-RL task
src/prl_hgf/
  env/                 # Task configs + trial sequence generators
  models/              # HGF builders + response models
  fitting/             # Batched NUTS (BlackJAX) + VB-Laplace
  analysis/            # Group analysis, BMS, trajectory export
  power/               # BFDA power analysis (pick_best_cue)
  simulation/          # JAX-native cohort simulation
scripts/               # Pipeline scripts (01_*, 02_*, …)
tests/                 # Unit + integration tests
validation/            # Scientific validation (parameter recovery)
cluster/               # SLURM scripts for M3/MASSIVE-style clusters
notebooks/             # Exploratory notebooks
docs/                  # Project + API documentation
results/               # Git-tracked small reference artifacts
output/                # Generated artifacts (gitignored)
figures/               # Generated plots (gitignored)
```

## Configuration

All task-specific parameters live in a YAML under `configs/`.

```python
# pick_best_cue (default)
from prl_hgf.env.task_config import load_config, AnalysisConfig
cfg: AnalysisConfig = load_config()

# PAT-RL
from prl_hgf.env.pat_rl_config import load_pat_rl_config, PATRLConfig
pcfg: PATRLConfig = load_pat_rl_config()
```

Never hardcode task structure (trial counts, reward probabilities, phases).
Every behavioural variant should be reachable via a config key.

## Key References

- Mathys et al. (2011, 2014) — HGF theory and binary HGF
- Legrand et al. (2024) — pyhgf (arXiv:2410.09206)
- Mason et al. (2024) — Psilocybin + reversal learning (Translational Psychiatry;
  psilocybin use case)
- Rigoux et al. (2014) — Random-effects Bayesian model selection
