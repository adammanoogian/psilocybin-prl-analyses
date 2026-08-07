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
  psilocybin vs placebo × 3 sessions study on post-concussion participants —
  the study itself (pipeline, Snakemake/SLURM infra, manuscript) now lives in
  the `psilocybin_decision_making` repo, which consumes this toolbox as a
  package dependency. The yaml remains here as the library default.
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
scripts/           # Toolbox utilities: demo_quickstart.py, ci/, _maintenance/
tests/             # Test suite (tiered)
  unit/            # Fast isolated tests (< 1s, no I/O)
  integration/     # Cross-module tests (< 60s, structure guards)
  scientific/      # Slow validation (> 60s, parameter recovery)
```

The psilocybin study pipeline (numbered scripts 01–06, `cluster/` SLURM jobs,
`workflow/` Snakemake DAG, `manuscript/`) moved to the
`psilocybin_decision_making` repo (2026-08-07) under `analysis/`.

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
- Tests in `tests/unit/` (fast), `tests/integration/` (cross-module), `tests/scientific/` (slow/validation)

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

- **Local is source of truth; Mutagen syncs to M3.** Edit files locally,
  changes propagate to `m3:/home/aman0087/fc37/adam/projects/hgf-analysis`
  in ~1s. Never edit files directly on M3 (no `ssh m3 vim ...`, no
  `sed -i` over SSH on synced paths) — that corrupts the sync direction.
- **Code does not need to be committed-and-pushed before running on M3.**
  Mutagen has already synced. Just commit when you're at a logical
  checkpoint, push when you want a backup or to share.
- **Line endings:** `.gitattributes` enforces LF for all text files +
  per-repo `core.autocrlf=input`. Required for Mutagen Windows↔Linux sync.
- **Study pipelines live downstream:** this repo carries no study pipeline of
  its own. Snakemake/SLURM orchestration for the psilocybin study is in
  `psilocybin_decision_making/analysis/`; run heavy toolbox tests
  (`tests/scientific/`) on M3 per the compute-routing rule.

## M3 / MASSIVE settings

- **Slurm account:** `fc37`
- **Default GPU partition:** `gpu` (request `--gres=gpu:T4:1` for quick
  probes, `--gres=gpu:A40:1` for 48GB-memory runs, `--partition=m3g
  --gres=gpu:V100:1` for V100)
- **Default CPU partition:** `comp`
- **Default walltime:** 02:00:00 (sbatch); 00:15:00 (srun debug)
- **Local path:** `C:/Users/aman0087/Documents/Github/hgf-analysis/`
- **M3 path:** `/home/aman0087/fc37/adam/projects/hgf-analysis/`
  (symlinks via `~/fc37/adam/projects/hgf-analysis`)
- **Logs:** `<project>/logs/<jobname>_<jobid>.{out,err}` (project-internal,
  on `/fs04` Lustre — `/home` has 15G quota, too small)
- **Conda env:** `ds_env` (Phase 27 target: `ds_env_v10` with Python 3.11 +
  JAX 0.9 + BlackJAX 1.5; build per `environment-v10.yml`)
- **Mutagen session name:** `hgf-analysis`
  (`mutagen sync list` to verify; create per setup section below)

## SSH / Mutagen control

- `m3-unlock <hours>` — unlock SSH access (passphrase-prompted, time-limited)
- `m3-lock` — kill switch; revoke SSH and ssh-agent key
- `mutagen sync list` — verify sync status; should be "Watching for changes"
  with no conflicts
- On `Permission denied (publickey)`: tell user to run `m3-unlock`. Do NOT
  retry, regenerate keys, or modify `~/.ssh/config`.

## Development Phases

1. Foundation (this plan): package skeleton, config system
2. Task environment: trial sequence generator from config
3. HGF models: 2-level and 3-level via pyhgf
4. Simulation: synthetic participants with known parameters
5. Fitting: single-subject MCMC via PyMC
6. Parameter recovery: validate before real data
7. Group analysis + GUI

## Research track

This repo serves two research tracks:

- **HGF at GPU Scale** (`hgf-gpu`) -- `Tracks/Track - HGF at GPU Scale.md`
- **Joint DCM-Behavioural Modelling** (`joint-dcm-behaviour`) -- `Tracks/Track - Joint DCM-Behavioural Modelling.md`

The GPU HGF toolbox itself, and the behavioural half of the joint brain-behaviour fits. Its public interface is load-bearing for another track -- check before changing it.

A *track* is a research programme -- one scientific question pursued across many
repos, over years. The roster, the full repo-to-track map, and the cross-track
couplings live in the **`research-tracks` skill**
(`~/.claude/skills/research-tracks/SKILL.md`); the authoritative notes live in
the Obsidian vault at `C:\Users\aman0087\Documents\Obsidian Vault\Tracks\`.

Read the track note before substantive work here -- it carries the current
state, the open questions, and the decision log, none of which are duplicated in
this repo. When work produces something durable (a derivation, a protocol, a
transferable failure), file it to the vault layer that matches its lifetime
rather than leaving it in this repo's docs.
