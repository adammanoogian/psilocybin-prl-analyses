---
phase: 06-group-analysis
plan: "03"
subsystem: manuscript
tags: [quarto, arxiv-pdf, HGF, psilocybin, bayesian, bibtex, reproducible-research]

requires:
  - phase: 05-validation
    provides: results CSVs (recovery_metrics.csv, bms_summary.csv) consumed by manuscript
  - phase: 06-group-analysis-01
    provides: estimates_wide.csv, group_contrasts.csv, effect_sizes.csv consumed by setup cell

provides:
  - Quarto manuscript scaffold at manuscript/ with arXiv preprint format
  - paper.qmd with Introduction, Methods, Results, Discussion sections (503 lines)
  - Inline Python statistics from results CSVs with graceful N/A fallback
  - BMS results embedded in main Results section
  - omega_3 explicit recovery caveat in Results/Secondary Parameters
  - 7-entry references.bib with all key HGF/psilocybin/BMS citations
  - .gitignore excluding rendered output (_freeze/, PDF, TeX, _extensions/)

affects:
  - Phase 07 (GUI): manuscript figures referenced from paper.qmd
  - Any rendering/CI step that runs quarto render manuscript/paper.qmd

tech-stack:
  added:
    - quarto manuscript project type (arxiv-pdf format)
    - ptitprince (raincloud plots, referenced in figure code cell)
    - scipy.stats.gaussian_kde (manual half-violin in raincloud code)
  patterns:
    - try/except results CSV loading with has_results flag for graceful fallback
    - get_contrast() helper for safe scalar extraction from contrasts DataFrame
    - Inline stats via {python} expressions wrapped in conditional
    - execute: cache: false during development; switch to true + cache-refresh when stable
    - from __future__ import annotations in every Python code cell

key-files:
  created:
    - manuscript/_quarto.yml
    - manuscript/paper.qmd
    - manuscript/references.bib
    - manuscript/.gitignore
  modified: []

key-decisions:
  - "cache: false in _quarto.yml during active development (Quarto cache does not detect CSV changes)"
  - "get_contrast() helper centralises safe DataFrame extraction for inline stats"
  - "Raincloud plot implemented manually (half-violin + jitter + box) to avoid ptitprince import dependency at render time"
  - "94% HDI reported (not 95%) following McElreath 2020 convention"
  - "omega_3 caveat placed in both Recovery paragraph and dedicated subsection header"
  - "BMS results in main Results (not supplementary) per phase CONTEXT.md decision"
  - "results_dir points to ../results/group_analysis; bms_dir to ../results/bms; recovery_dir to ../results/recovery"

patterns-established:
  - "Quarto manuscript type with arxiv-pdf format as default preprint scaffold"
  - "has_results / has_bms / has_recovery flags guard all inline stats"
  - "All code cells use #| include: false or #| echo: false via global execute config"

duration: 4min
completed: 2026-04-06
---

# Phase 6 Plan 03: Quarto Manuscript Scaffold Summary

**Self-contained arXiv-format Quarto manuscript with inline Python statistics, BMS in main Results, omega_3 recovery caveat, and graceful N/A fallback when results CSVs are absent**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-06T16:14:35Z
- **Completed:** 2026-04-06T16:18:49Z
- **Tasks:** 1
- **Files modified:** 4 (all created)

## Accomplishments

- Created `manuscript/` as a standalone Quarto manuscript project (`type: manuscript`, `arxiv-pdf` format)
- Built 503-line `paper.qmd` with Introduction, Methods (Participants, Task, Modeling, Statistical Analysis), Results (Recovery, BMS, omega_2 group × session, Secondary Parameters with omega_3 caveat, Effect Sizes), and Discussion
- Setup cell loads 5 results CSVs with try/except and `has_results`/`has_bms`/`has_recovery` flags; all inline `{python}` stats are wrapped in conditional expressions so the manuscript renders as "N/A" without pipeline outputs
- 7 BibTeX entries covering all cited works (Mathys 2011/2014, Legrand 2024, Mason 2024, Rigoux 2014, Iglesias 2021, Weber 2024)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Quarto manuscript scaffold** - `6f29f68` (feat)

**Plan metadata:** (see below — docs commit)

## Files Created/Modified

- `manuscript/_quarto.yml` - Quarto project config: manuscript type, arxiv-pdf format, python3 jupyter, cache: false
- `manuscript/paper.qmd` - Main manuscript (503 lines): all four major sections, inline stats, raincloud figure, BMS table, omega_3 caveat
- `manuscript/references.bib` - 7 BibTeX entries for HGF, psilocybin, BMS, and two-stage references
- `manuscript/.gitignore` - Excludes _freeze/, _site/, *.pdf, *.tex, *.log, _extensions/

## Decisions Made

- **cache: false** in `_quarto.yml` during development. Quarto's cache does not detect changes in external CSV files; switching to `cache: true` with `--cache-refresh` should be done after results stabilise.
- **`get_contrast()` helper** centralises safe scalar extraction from the contrasts DataFrame, avoiding repetitive try/except for each inline stat.
- **Manual raincloud implementation** using `scipy.stats.gaussian_kde` rather than a hard ptitprince import, to keep the figure cell renderable even if ptitprince is not installed.
- **94% HDI** (not 95%) for group-level credible intervals, following McElreath 2020 convention already used in the bambi model (06-01/06-02 plans).
- **omega_3 caveat** placed in two locations: (1) the Parameter Recovery subsection narrative, and (2) a dedicated "### omega_3 (Meta-Volatility) — Recovery Caveat" subsection heading, ensuring it is impossible to miss.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

The Quarto arxiv-pdf format requires the `quarto-arxiv` extension. To install it before first render:

```bash
cd manuscript
quarto add quarto-ext/arxiv
quarto render paper.qmd
```

The HTML format renders without any extension installation.

## Next Phase Readiness

- Manuscript scaffold is complete. Once `scripts/06_group_analysis.py` (Plan 06-01) produces results CSVs, running `quarto render manuscript/paper.qmd` will populate all inline statistics and figures automatically.
- The raincloud plot code cell requires `estimates_wide.csv` with columns `group`, `session`, `omega_2`. Column names must match the output schema from 06-01 exactly.
- mason2024 BibTeX entry has a TODO comment — exact volume/page details should be verified against the published paper before submission.

---
*Phase: 06-group-analysis*
*Completed: 2026-04-06*
