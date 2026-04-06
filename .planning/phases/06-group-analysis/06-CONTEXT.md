# Phase 6: Group Analysis - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Test the primary hypotheses: group x session interactions on HGF parameters,
phase-stratified learning rate effects, and produce publication-quality figures.
Includes setting up a Quarto scientific manuscript folder for hypothesis writeup
with inline computed statistics.

</domain>

<decisions>
## Implementation Decisions

### Manuscript structure
- Quarto manuscript folder at `manuscript/` in project root (self-contained Quarto project)
- Scaffold from the quarto-scientific skill template: `paper.qmd`, `_quarto.yml`, `arxiv_template.tex`, `references.bib`
- arXiv preprint style format (single-column, easy to adapt to journal later)
- Model comparison (2-level vs 3-level BMS) included in main Results section, not supplementary
- All statistics rendered inline via Python code cells (`{python} f"{beta:.3f}"`) — fully reproducible, updates when pipeline reruns
- Manuscript imports results CSVs from `results/` and renders figures dynamically

### Statistical modeling
- Bayesian mixed-effects model via bambi (formula interface on PyMC) for group x session analysis
- Random intercepts only: `(1 | participant_id)` — simplest, most stable for the expected sample size
- Full posterior propagation: participant-level uncertainty feeds into group model (not just posterior means)
- Claude's Discretion: specific mechanism for posterior propagation (measurement error model vs sampling loop — pick most practical approach given the pipeline)

### Primary hypothesis
- Group x Session interaction on omega_2 (tonic volatility / learning rate)
- Post-concussion group shows different omega_2 trajectory across sessions (baseline → psilocybin → follow-up) compared to controls
- kappa, beta, zeta as secondary parameters
- omega_3 reported with caveat (known poor recovery)

### Visualization style
- Raincloud plots for parameter distributions by group x session (half-violin + strip + boxplot)
- Colorblind-safe palette (seaborn's colorblind palette — blue/orange)
- Figures rendered inside Quarto code cells in `paper.qmd` (not pre-generated PNGs)
- Publication-quality: serif fonts, 150 DPI, consistent sizing

### Phase stratification
- Claude's Discretion: approach for analyzing stable vs volatile phases (aggregate by phase label vs trial-level trajectories — pick what HGF belief structure best supports)
- Group x phase (stable/volatile) interaction is exploratory, not primary
- Report if significant, but main story is group x session on omega_2

### Effect sizes
- Cohen's d or partial eta-squared for all primary comparisons
- Bayesian effect sizes from bambi posteriors (credible intervals on group differences)

</decisions>

<specifics>
## Specific Ideas

- Quarto manuscript follows the quarto-scientific.skill template from project_utils repo (Adam's standard setup: Monash affiliations, arXiv template, seaborn-whitegrid matplotlib style)
- Hypothesis framing: psilocybin differentially modulates belief updating (omega_2) in post-concussion vs control participants across treatment sessions
- The pipeline script (06_group_analysis.py) produces results CSVs; the Quarto manuscript consumes them for inline stats and figure rendering

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-group-analysis*
*Context gathered: 2026-04-06*
