# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Phase 2 (HGF models) in progress — plan 02-01 complete.

## Current Position

Phase: 2 of 7 (Models)
Plan: 1 of ~2 in phase (02-01 complete)
Status: In progress
Last activity: 2026-04-05 — Completed 02-01-PLAN.md (HGF model builders)

Progress: [███░░░░░░░] ~21% (3 of ~14 plans complete)

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
| net.edges is a tuple (positional), not a dict | pyhgf 0.2.8: node N accessed as net.edges[N] by position index | 02-01 |
| extract_beliefs uses "mean" for continuous nodes and "expected_mean" for binary nodes | "mean" is log-odds posterior; "expected_mean" is sigmoid probability in [0,1] | 02-01 |
| observed mask must use int dtype (not bool) | JAX tracing multiplies observed against PE; int required for correctness | 02-01 |

### Pending Todos

- Phase 2 plan 02-02: softmax + stickiness response function (if planned)
- Phase 3: HGF simulation of synthetic participants with known parameters
- Phase 4: Single-subject MCMC fitting via PyMC (custom Op needed — HGFDistribution incompatible with multi-branch Network)
- Consider creating project-specific .venv with Python 3.10 (deferred from Phase 1)

### Blockers/Concerns

- Phase 4 (fitting) will need a custom PyMC Op wrapping the multi-branch Network's logp — HGFDistribution cannot be used with custom Network (confirmed empirically)
- ω₃ parameter recovery expected to be challenging (known issue in literature)
- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env or a Python 3.10 venv
- JAX forward pass takes ~1s per call due to JIT compilation (first call per session); acceptable for simulation but may slow fitting iteration

## Session Continuity

Last session: 2026-04-05T11:33:44Z
Stopped at: Completed 02-01-PLAN.md — HGF model builders (Phase 2, plan 1)
Resume file: None — continue Phase 2 with next plan (response function or simulation)
