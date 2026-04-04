# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Phase 1 complete — both plans done. Phase 2 (HGF models) next.

## Current Position

Phase: 1 of 7 (Foundation)
Plan: 2 of 2 in phase (01-01 + 01-02 complete)
Status: Phase 1 complete
Last activity: 2026-04-04 — Completed 01-02-PLAN.md (task environment simulator)

Progress: [██░░░░░░░░] ~14% (2 of ~14 plans complete)

## Accumulated Context

### Decisions

| Decision | Rationale | Phase |
|----------|-----------|-------|
| pyhgf 0.2.8 chosen as HGF library | JAX-backed, PyMC integration, Network API supports custom topologies | Planning |
| 3 parallel binary HGF branches (one per cue) with shared volatility parent | Literature default; independent-volatility is a v2 variant | Planning |
| Softmax + stickiness response model (β, ζ) | Standard in computational psychiatry literature | Planning |
| Partial feedback: only chosen cue receives reward input | Standard PRL protocol | Planning |
| Random-effects BMS (Rigoux et al. 2014) for model comparison | Formal Bayesian model comparison at group level | Planning |
| Jupyter ipywidgets for GUI | VSCode compatible, no web server needed | Planning |
| ω₃ recovery caveat acknowledged upfront | Known issue in literature; primary hypotheses focus on ω₂ and κ | Planning |
| Task environment reads from PRL pick_best_cue config | No hardcoded task structure anywhere | Planning |
| Mixed-effects model for second-level group analysis | group × session × phase | Planning |
| ds_env (Python 3.10 conda env) as installation environment | pyhgf 0.2.8 Requires-Python <=3.13 excludes system Python 3.13 | 01-01 |
| Single merged prl_analysis.yaml | Task + analysis params in one file for single source of truth | 01-01 |
| Frozen dataclasses for all config types | Immutability prevents accidental mutation in downstream pipeline | 01-01 |
| /env/ (root-anchored) in .gitignore | Prevents ignoring src/prl_hgf/env/ subpackage | 01-01 |
| Phase n_trials 40 → 30 per phase | Plan spec: 3 sets × 4 phases × 30 + 3 × 20 transfer = 420 trials total | 01-02 |
| TransferConfig as separate dataclass from PhaseConfig | Transfer has no name field; avoids unused required field | 01-02 |
| pytest pythonpath includes "." (project root) | Root config.py must be importable during test collection | 01-02 |

### Pending Todos

- Phase 2: HGF model implementation (2-level and 3-level binary HGF with softmax response)
- Consider creating project-specific .venv with Python 3.10 (deferred from Phase 1)

### Blockers/Concerns

- pyhgf Network API for 3 parallel binary inputs with shared volatility parent needs verification (Phase 2)
- JAX / PyMC version compatibility needs testing — install succeeded but no JAX code executed yet
- ω₃ parameter recovery expected to be challenging (known issue in literature)
- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env or a Python 3.10 venv

## Session Continuity

Last session: 2026-04-04T19:13:00Z
Stopped at: Completed 01-02-PLAN.md — task environment simulator (Phase 1 complete)
Resume file: None — begin Phase 2 (HGF models)
