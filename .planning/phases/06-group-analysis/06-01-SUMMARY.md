---
phase: 06-group-analysis
plan: "01"
subsystem: analysis
tags: [bambi, pingouin, mixed-effects, cohen-d, partial-eta-sq, bayesian-contrasts, group-analysis]

requires:
  - phase: 04-fitting
    provides: fit_batch output with posterior means per participant-session-parameter
  - phase: 05-validation
    provides: validated pipeline with flagged participant handling

provides:
  - Wide-form estimates pivot (build_estimates_wide) for group analysis
  - Bambi mixed-effects model factory (fit_group_model) with group x session interaction
  - Posterior contrast extraction with 94% HDI (extract_posterior_contrasts)
  - Multi-parameter batch summarization (summarize_group_models)
  - Cohen's d computation via pingouin (compute_cohens_d)
  - Effect size table with partial eta-squared (compute_effect_sizes_table)

affects:
  - 06-02 (visualization will consume contrasts and effect size table)
  - 06-03 (GUI/reporting will call summarize_group_models and compute_effect_sizes_table)

tech-stack:
  added:
    - bambi 0.15.0 (Bayesian mixed-effects models via PyMC backend)
    - pingouin 0.6.1 (frequentist effect sizes)
    - ptitprince 0.3.1 (raincloud plots, added per plan spec)
  patterns:
    - Wide-form pivot pattern for posterior means (one row per participant-session)
    - Dynamic variable name discovery for posterior contrasts (no hardcoded bambi names)
    - Reference vs non-reference session contrast decomposition (main + interaction terms)
    - Partial eta-squared from Cohen's d via d^2 / (d^2 + 4) two-group formula

key-files:
  created:
    - src/prl_hgf/analysis/group.py
    - src/prl_hgf/analysis/effect_sizes.py
  modified:
    - pyproject.toml
    - src/prl_hgf/analysis/__init__.py

key-decisions:
  - "bambi>=0.13.0 used instead of >=0.17.2 — bambi 0.17.2 requires Python >=3.11; ds_env uses Python 3.10; installed 0.15.0 which provides identical formula/fit API"
  - "Dynamic posterior variable discovery via regex — bambi variable names embed category levels (e.g. C(group)[T.post_concussion]:C(session)[T.session2]); hardcoding would break on different category orderings"
  - "94% HDI throughout — matches Kruschke 2018 conventions used in computational psychiatry literature"
  - "Partial eta-squared from Cohen's d formula (not from ANOVA table) — appropriate for two-group comparisons without full ANOVA structure"
  - "Groups discovered dynamically in compute_effect_sizes_table — alphabetically sorted; first = control (reference), second = post_concussion (treatment)"

patterns-established:
  - "Bayesian + frequentist dual reporting: posterior contrasts from bambi (Bayesian) paired with Cohen's d and partial eta-squared (frequentist)"
  - "Flagged participant exclusion before pivot: exclude_flagged=True default in build_estimates_wide matches fit_batch NaN-fill convention"

duration: 6min
completed: 2026-04-06
---

# Phase 06 Plan 01: Group Analysis Engine Summary

**Bambi mixed-effects group x session model with dynamic posterior contrast extraction and pingouin effect sizes for HGF parameter comparisons.**

## Performance

- **Duration:** ~6 minutes
- **Started:** 2026-04-06T16:27:21Z
- **Completed:** 2026-04-06T16:33:16Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Created `src/prl_hgf/analysis/group.py` with the full group analysis engine:
  `build_estimates_wide` pivots fit_batch output to wide form; `fit_group_model`
  constructs and fits a `group * session + (1|participant_id)` bambi model;
  `extract_posterior_contrasts` discovers variable names dynamically and
  decomposes contrasts for each session (main effect for reference, main + interaction
  for others); `summarize_group_models` loops over multiple parameters.
- Created `src/prl_hgf/analysis/effect_sizes.py` with `compute_cohens_d` (via
  `pingouin.compute_effsize`) and `compute_effect_sizes_table` with dynamic group
  discovery and partial eta-squared from the two-group formula.
- Added bambi 0.15.0, pingouin 0.6.1, ptitprince 0.3.1 to `pyproject.toml`
  dependencies and updated mypy overrides for all three.
- Updated `src/prl_hgf/analysis/__init__.py` to export all 6 new public functions
  alongside existing recovery and BMS exports.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| bambi 0.15.0 (not 0.17.2) | Python 3.10 ds_env incompatible with 0.17.2's Python >=3.11 requirement |
| Dynamic posterior variable discovery | Bambi embeds category levels in variable names; regex parsing avoids fragility |
| 94% HDI | Kruschke 2018 standard used across computational psychiatry reporting |
| Partial eta-squared from d^2/(d^2+4) | Two-group formula; doesn't require full ANOVA residual structure |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] bambi>=0.17.2 unavailable for Python 3.10**

- **Found during:** Task 1 dependency installation
- **Issue:** bambi 0.17.2+ requires Python >=3.11; ds_env uses Python 3.10 (required by pyhgf 0.2.8)
- **Fix:** Changed version constraint to `bambi>=0.13.0`; pip installed 0.15.0 which has identical Model/fit API
- **Files modified:** pyproject.toml
- **Commit:** 6f1aeba

## Next Phase Readiness

Phase 06-02 (visualization) can proceed immediately. All analysis functions are importable from `prl_hgf.analysis`. The contrasts DataFrame schema (session, mean, sd, hdi_3, hdi_97) and the effect sizes table schema (parameter, session, cohen_d, partial_eta_sq, n_control, n_post_concussion) are stable interfaces for downstream plotting.

Dependency note: `fit_group_model` runs MCMC which is slow without real data. Downstream scripts should call `summarize_group_models` only after real fit_batch output is available.
