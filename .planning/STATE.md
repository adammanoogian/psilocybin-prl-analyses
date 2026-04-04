# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Planning complete, ready to begin Phase 1.

## Current Position

Phase: 0 of 7 (Planning complete)
Plan: None yet
Status: Planning documents finalized
Last activity: 2026-04-04 — PROJECT.md, REQUIREMENTS.md, ROADMAP.md created

Progress: [░░░░░░░░░░] 0% (planning only)

## Accumulated Context

### Decisions

- pyhgf 0.2.8 chosen as HGF library (JAX-backed, PyMC integration, Network API supports custom topologies)
- 3 parallel binary HGF branches (one per cue) with shared volatility parent for 3-level variant
- Softmax + stickiness response model (β, ζ) — standard in computational psychiatry literature
- Partial feedback: only the chosen cue receives reward input; unchosen cues are not updated
- Random-effects BMS (Rigoux et al. 2014) for formal model comparison
- Jupyter ipywidgets for GUI (VSCode compatible, no web server needed)
- ω₃ recovery caveat acknowledged upfront — primary hypotheses focus on ω₂ and κ
- Task environment reads from PRL pick_best_cue config — no hardcoded task structure
- Mixed-effects model for second-level group analysis (group × session × phase)

### Pending Todos

- Begin Phase 1 implementation

### Blockers/Concerns

- pyhgf Network API for 3 parallel binary inputs with shared volatility parent needs verification (Phase 2)
- JAX / PyMC version compatibility needs testing during Phase 1 environment setup
- ω₃ parameter recovery expected to be challenging (known issue in literature)

## Session Continuity

Last session: 2026-04-04
Stopped at: Planning documents complete
Resume file: None — begin Phase 1
