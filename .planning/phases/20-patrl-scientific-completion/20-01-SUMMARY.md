---
phase: 20-patrl-scientific-completion
plan: "01"
subsystem: config-loader
tags: [pat-rl, config, yaml, loader, consumer-spec, phenotype, avoid-contingency]
one-liner: "Consumer-spec PAT-RL YAML delta (SC1 contingencies + SVVS run order + magnitudes [1,3] + phenotype priors) with loader extension for avoid block, b/dhr/epsilon2 phenotype fields, and b/gamma/alpha/lam fitting priors"
requires: []
provides:
  - "configs/pat_rl.yaml corrected to consumer spec (SC1)"
  - "pat_rl_config.py loader parses avoid, b, dhr_mean, dhr_sd, epsilon2_coupling_coef, theta, omega_3 phenotype fields"
  - "pat_rl_config.py loader parses b, gamma, alpha, lam fitting priors"
  - "PHENOTYPE_COLUMN_NAME module constant"
affects:
  - "Plan 20-02 (Model A+b, Models B/C): can now read b/gamma/alpha/lam from FittingPriorConfig"
  - "Plan 20-03 (Model D): can read lam prior"
  - "Plan 20-04 (epsilon2 simulator): can read dhr_mean/dhr_sd/epsilon2_coupling_coef from PhenotypeParams"
  - "Plan 20-05 (cohort scale): can use PHENOTYPE_COLUMN_NAME for sim_df column"
  - "Plan 20-06 (stratified BMS): can use PHENOTYPE_COLUMN_NAME constant"
tech-stack:
  added: []
  patterns:
    - "Optional-key parsing with dataclass defaults for backward compat (Phase 18/19 configs still load)"
    - "frozen=True dataclass default field values using immutable nested dataclasses"
    - "Module-level string constant for column-name contract across DataFrames"
key-files:
  created: []
  modified:
    - "configs/pat_rl.yaml (+82 / -40 lines)"
    - "src/prl_hgf/env/pat_rl_config.py (+147 / -17 lines)"
    - "tests/test_env_pat_rl_config.py (+211 / -1 lines)"
decisions:
  - id: D-20-01-A
    decision: "Stochastic avoid contingency (Option B) — P(reward|avoid)=0.10, P(shock|avoid)=0.10, P(nothing|avoid)=0.80"
    rationale: "Authoritative consumer spec (sister repo H2A.1.1) says verbatim: Avoid always = 10/10/80. Config surface wired in Plan 20-01; stochastic logp/scan wiring deferred to Plans 20-02/20-03."
    source: "Sister repo docs/files/GSD_heart2adapt_sim.yaml tasks H2A.1.1, H2A.1.2, H2A.1.4"
  - id: D-20-01-B
    decision: "Phenotype priors use H2A.1.2+H2A.1.4 consumer spec values; placeholders marked TODO:consumer-spec"
    rationale: "Authoritative values from consumer repo provided before plan execution. b.sd=0.5 and epsilon2_coupling_coef=0.3 are placeholders pending downstream confirmation."
    source: "Sister repo GSD_heart2adapt_sim.yaml"
metrics:
  duration: "~12 minutes"
  completed: "2026-04-18"
---

# Phase 20 Plan 01: Config Delta and Loader Summary

**One-liner:** Consumer-spec PAT-RL YAML delta (SC1 contingencies + SVVS run order + magnitudes [1,3] + phenotype priors) with loader extension for avoid block, b/dhr/epsilon2 phenotype fields, and b/gamma/alpha/lam fitting priors.

---

## What Was Done

### Task 1 (pre-resolved as checkpoint:decision)
Decisions resolved by orchestrator reading the authoritative downstream consumer spec at `../dcm_hgf_mixed_models/docs/files/GSD_heart2adapt_sim.yaml`:
- **Avoid contingency**: STOCHASTIC (Option B) — `P(reward|avoid)=0.10, P(shock|avoid)=0.10, P(nothing|avoid)=0.80` per H2A.1.1.
- **Phenotype priors**: All 4 phenotypes confirmed from H2A.1.2 + H2A.1.4. Placeholders (b.sd, epsilon2_coupling_coef for some phenotypes) marked `TODO:consumer-spec`.

### Task 2 — configs/pat_rl.yaml (commit `2475591`)

**Files modified:** `configs/pat_rl.yaml` (82 insertions, 40 deletions)

Changes:
| Field | Before | After |
|-------|--------|-------|
| `contingencies.safe` | reward=0.75, shock=0.00, nothing=0.25 | reward=0.70, shock=0.10, nothing=0.20 |
| `contingencies.dangerous` | reward=0.25, shock=0.50, nothing=0.25 | reward=0.10, shock=0.70, nothing=0.20 |
| `contingencies.avoid` | (absent) | reward=0.10, shock=0.10, nothing=0.80 |
| `run_order` | SVSV | SVVS |
| `reward_levels` / `shock_levels` | [1.0, 5.0] | [1, 3] |
| Phenotype `omega_2.mean` healthy | -6.0 | -3.0 |
| Phenotype `omega_2.mean` anxious | -3.5 | -2.0 |
| Phenotype `beta.mean` reward_sensitive | 8.0 | 3.5 |
| New phenotype fields | (absent) | omega_3, theta, b, dhr_mean, dhr_sd, epsilon2_coupling_coef |
| New fitting priors | (absent) | b, gamma, alpha, lam |

Citations updated: "Browning 2015 / Daw 2006 / Schönberg 2007" → Klaassen et al. 2024 (Communications Biology, doi:10.1038/s42003-024-06267-6).

### Task 3 — src/prl_hgf/env/pat_rl_config.py (commit `6320b43`)

**Files modified:** `src/prl_hgf/env/pat_rl_config.py` (147 insertions, 17 deletions)

New public names added to module:
- `PHENOTYPE_COLUMN_NAME: str = "phenotype"` — canonical column name for phenotype in sim/fit DataFrames
- `ContingencyConfig.avoid: OutcomeProbs` — new field with default `OutcomeProbs(0.0, 0.0, 1.0)` for legacy compat
- `PhenotypeParams.omega_3: PriorGaussian` — default `PriorGaussian(-4.0, 0.5)`
- `PhenotypeParams.theta: PriorGaussian` — default `PriorGaussian(0.005, 0.002)`
- `PhenotypeParams.b: PriorGaussian` — default `PriorGaussian(0.0, 0.5)`
- `PhenotypeParams.dhr_mean: float` — default `-2.0`
- `PhenotypeParams.dhr_sd: float` — default `0.5`
- `PhenotypeParams.epsilon2_coupling_coef: float` — default `0.3`
- `FittingPriorConfig.b: PriorGaussian` — default `PriorGaussian(0.0, 1.0)`
- `FittingPriorConfig.gamma: PriorGaussian` — default `PriorGaussian(0.0, 0.5)`
- `FittingPriorConfig.alpha: PriorGaussian` — default `PriorGaussian(0.0, 0.5)`
- `FittingPriorConfig.lam: PriorGaussian` — default `PriorGaussian(0.0, 0.1)`

Validation added:
- `PhenotypeParams.__post_init__`: `dhr_sd > 0`, `epsilon2_coupling_coef >= 0`
- `_parse_contingencies`: optional-key parsing for `avoid` with legacy fallback

### Task 4 — tests/test_env_pat_rl_config.py (commit `dc3ffe7`)

**Files modified:** `tests/test_env_pat_rl_config.py` (211 insertions, 1 deletion)

8 new tests in `TestPhase20ConfigExtensions`:
1. `test_avoid_contingency_loaded` — reward=0.10, shock=0.10, nothing=0.80, sum=1.0
2. `test_avoid_contingency_sum_validation` — bad sum (1.1) raises ValueError
3. `test_run_order_svvs` — SVVS order
4. `test_magnitudes_1_3` — (1.0, 3.0) for reward and shock
5. `test_phenotype_has_new_fields` — all 4 phenotypes have valid b/dhr_mean/dhr_sd/epsilon2_coupling_coef
6. `test_phenotype_b_means_match_sc1` — SC1 values: 0.0 / +0.3 / -0.3 / 0.0
7. `test_fitting_priors_have_bias_and_hr_terms` — b/gamma/alpha/lam are PriorGaussian; lam.sd=0.1
8. `test_phenotype_column_name_constant` — PHENOTYPE_COLUMN_NAME == 'phenotype'

---

## Test Results

```
pytest tests/test_env_pat_rl_config.py tests/test_models_patrl.py tests/test_pat_rl_simulator.py tests/test_hierarchical_patrl.py -v
37 passed, 2 deselected in 52.23s
```

- `test_env_pat_rl_config.py`: 16/16 (8 original + 8 new)
- `test_models_patrl.py`: 9/9
- `test_pat_rl_simulator.py`: 6/6
- `test_hierarchical_patrl.py`: 6/6
- 2 deselected = blackjax tests (known blocker: blackjax not installed in ds_env, per STATE.md line 165)

Zero regressions from Phase 18/19.

---

## Deviations from Plan

### Auto-extensions (not strictly in task spec but required for correct YAML round-trip)

**1. [Rule 2 - Missing Critical] Added omega_3 and theta to PhenotypeParams**

- **Found during:** Task 3 (when YAML Task 2 wrote `omega_3` and `theta` keys into each phenotype block per consumer spec values)
- **Issue:** The new YAML includes `omega_3` and `theta` per phenotype, but `_parse_phenotype_params` would raise `KeyError` if it tried to parse them, or silently drop them if it ignored them. Plan Task 3 only listed b/dhr_mean/dhr_sd/epsilon2_coupling_coef as new fields.
- **Fix:** Added `omega_3: PriorGaussian` and `theta: PriorGaussian` to `PhenotypeParams` with defaults matching consumer spec. Both use optional-key parsing (fall back to defaults if absent in legacy configs). No existing test broken.
- **Files modified:** `src/prl_hgf/env/pat_rl_config.py`
- **Commits:** `6320b43`

---

## Placeholder TODOs for Downstream Plans

The following `TODO:consumer-spec` markers are in `configs/pat_rl.yaml` and must be resolved before scientific publication or V2 validation:

| Field | All Phenotypes | Note |
|-------|---------------|------|
| `b.sd` | 0.5 (placeholder) | Consumer spec gives b.mean only. Plan 20-08 citation pass or researcher confirmation needed. |
| `epsilon2_coupling_coef` | 0.3 (placeholder) | Consumer spec gives direction (larger PE → bradycardia) but no magnitude. Klaassen 2024 Comms Bio range ~0.2–0.4. |
| `dhr_mean` / `dhr_sd` for `reward_sensitive` | Inherited from `healthy` (-2.0 / 0.5) | Consumer spec (H2A.1.4) only names healthy and high-anxiety; reward-susceptible and anxious+reward inherit from their baseline phenotype. |
| `dhr_mean` / `dhr_sd` for `anxious_reward_sensitive` | Inherited from `anxious` (-0.5 / 0.8) | Same as above. |
| Fitting priors: `gamma.sd`, `alpha.sd`, `lam.sd` | 0.5 / 0.5 / 0.1 (placeholders) | Consumer spec does not specify these priors; plans 20-02/20-03 will confirm or adjust. |

**Stochastic avoid in logp/scan body:** The `avoid` block is wired into the YAML and loader config surface in this plan. The actual stochastic avoid outcome draw in `pat_rl_sequence.py` and the `expected_value` function update are **deferred to Plans 20-02/20-03** as stated in the note added to `_parse_contingencies`.

---

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 2 (YAML) | `2475591` | `configs/pat_rl.yaml` |
| 3 (loader) | `6320b43` | `src/prl_hgf/env/pat_rl_config.py` |
| 4 (tests) | `dc3ffe7` | `tests/test_env_pat_rl_config.py` |

All commits pushed to `origin/main` immediately after creation (laptop ↔ M3 cluster sync invariant).

---

## Final YAML Values (audit trail for Plan 20-08 citation pass)

```yaml
# Phenotype priors as written in configs/pat_rl.yaml after Plan 20-01
# Source: consumer repo GSD_heart2adapt_sim.yaml H2A.1.2 + H2A.1.4
healthy:    omega_2 -3.0±0.5, beta 2.0±0.5, b 0.0±0.5, dhr_mean -2.0, dhr_sd 0.5
reward_sensitive:  omega_2 -3.0±0.5, beta 3.5±0.5, b 0.3±0.5, dhr_mean -2.0 (inherited), dhr_sd 0.5 (inherited)
anxious:    omega_2 -2.0±0.5, beta 2.0±0.5, b -0.3±0.5, dhr_mean -0.5 (SC1), dhr_sd 0.8 (SC1)
anxious_reward_sensitive: omega_2 -2.0±0.5, beta 3.5±0.5, b 0.0±0.5, dhr_mean -0.5 (inherited), dhr_sd 0.8 (inherited)
```
