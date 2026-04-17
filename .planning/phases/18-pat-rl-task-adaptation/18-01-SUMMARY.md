---
phase: 18-pat-rl-task-adaptation
plan: 01
subsystem: env
tags: [yaml, dataclass, config, pat_rl, consumer-study, parallel-stack, phenotype]

# Dependency graph
requires:
  - phase: config.py
    provides: CONFIGS_DIR path constant for YAML resolution
provides:
  - configs/pat_rl.yaml — single source of truth for PAT-RL task/simulation/fitting
  - src/prl_hgf/env/pat_rl_config.py — PATRLConfig dataclass tree + load_pat_rl_config()
  - tests/test_env_pat_rl_config.py — 8-test round-trip + validator coverage
affects:
  - 18-02 (pat_rl_sequence.py will import PATRLConfig from here)
  - 18-03 (HGF builders will use hazards, run_order, n_trials from task config)
  - 18-04 (fitting orchestrator imports PATRLFittingConfig.priors)

# Tech tracking
tech-stack:
  added: [types-PyYAML (mypy stubs for yaml module)]
  patterns:
    - Parallel-stack isolation — PAT-RL env stack is fully independent of pick_best_cue
      (no imports from task_config.py, no modifications to env/__init__.py)
    - Frozen dataclass hierarchy with __post_init__ validation and expected-vs-actual
      error messages at every boundary

key-files:
  created:
    - configs/pat_rl.yaml
    - src/prl_hgf/env/pat_rl_config.py
    - tests/test_env_pat_rl_config.py
  modified: []

key-decisions:
  - "Parallel loader pattern: pat_rl_config.py does not import from task_config.py,
     does not subclass AnalysisConfig/TaskConfig — keeps pick_best_cue tests isolated"
  - "PhenotypeParams uses PriorGaussian for all fields (including kappa, mu3_0 with sd=0)
     rather than a separate FixedParam type — simpler and sufficient for sd=0 fixed case"
  - "env/__init__.py deliberately NOT updated — adding PAT-RL exports risks side-effects
     on pick_best_cue imports; PAT-RL callers use direct module import"
  - "types-PyYAML installed to resolve mypy import-untyped error for yaml — pre-existing
     issue that existed in task_config.py too; installed once for the env"

patterns-established:
  - "PAT-RL parallel stack root: all downstream 18-0X modules import PATRLConfig here"
  - "_parse_task / _parse_simulation / _parse_fitting private helpers separate YAML
     parsing from dataclass construction"
  - "Validator pattern: __post_init__ raises ValueError with f-strings including both
     expected and actual values (e.g., 'got {val}', 'expected {N}')"

# Metrics
duration: ~90min (including JAX-heavy regression suite execution)
completed: 2026-04-17
---

# Phase 18 Plan 01: PAT-RL Config Surface Summary

**Frozen PATRLConfig dataclass tree (13 classes) loaded from configs/pat_rl.yaml via load_pat_rl_config(), completely parallel to pick_best_cue stack with 8-test coverage and zero regressions**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-04-17T00:00:00Z
- **Completed:** 2026-04-17
- **Tasks:** 3/3
- **Files created:** 3

## Accomplishments

- Created `configs/pat_rl.yaml` as single source of truth: 192/4/4-run structure,
  hazard-driven reversals (stable=0.03, volatile=0.10), 2x2 reward/shock magnitudes,
  outcome contingencies, Delta-HR stub, phenotype 2x2 grid, fitting priors
- Implemented 13-class frozen dataclass hierarchy in `pat_rl_config.py` with 7
  validators that all emit expected-vs-actual error messages
- 8 pytest tests covering round-trip load, computed property (trial_duration_s=11.0),
  and all 7 validator error paths; pick_best_cue regression confirms n_cues=3 unchanged
- parallel-stack constraint verified: `src/prl_hgf/env/__init__.py` unchanged, zero
  imports from task_config.py

## Task Commits

1. **Task 1: Write configs/pat_rl.yaml with full PAT-RL spec** — `98923f3` (feat)
2. **Task 2: Implement PATRLConfig dataclass tree + load_pat_rl_config** — `64ac286` (feat)
3. **Task 3: Unit tests for pat_rl_config loader + pick_best_cue regression** — `34d4a9a` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `configs/pat_rl.yaml` — PAT-RL task/simulation/fitting YAML, 192 trials/4 runs,
  hazard reversals, 2x2 magnitudes, phenotype grid, Delta-HR stub
- `src/prl_hgf/env/pat_rl_config.py` — 13-class frozen dataclass tree:
  HazardConfig, OutcomeProbs, ContingencyConfig, MagnitudeConfig, TimingConfig,
  DeltaHRDistribution, DeltaHRStubConfig, PATRLTaskConfig, PriorGaussian,
  PriorTruncated, PhenotypeParams, PATRLSimulationConfig, FittingPriorConfig,
  PATRLFittingConfig, PATRLConfig + load_pat_rl_config() entry point
- `tests/test_env_pat_rl_config.py` — 8 tests; _write_config tmp-YAML helper

## Decisions Made

1. **Parallel loader pattern**: `pat_rl_config.py` has zero imports from `task_config.py`
   and zero modifications to `env/__init__.py`. PAT-RL callers use
   `from prl_hgf.env.pat_rl_config import load_pat_rl_config` directly. Rationale:
   keeps pick_best_cue test isolation absolute; no regression risk as more PAT-RL
   modules are added.

2. **PhenotypeParams uses PriorGaussian for kappa and mu3_0** (with sd=0 for "fixed"
   parameters) rather than a separate `FixedParam` type. sd>=0 is validated (not sd>0)
   to allow the degenerate case. This is sufficient for Phase 18 and avoids over-
   engineering before Models B/C/D clarify what param variation is needed.

3. **types-PyYAML installed**: `mypy src/prl_hgf/env/pat_rl_config.py` produced an
   `import-untyped` error for `yaml`. Installed `types-PyYAML` to resolve — the same
   latent error existed in `task_config.py` before this phase. This is an env-level
   fix, not a code change.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed types-PyYAML to resolve mypy import-untyped error**

- **Found during:** Task 2 verification (mypy check)
- **Issue:** `mypy src/prl_hgf/env/pat_rl_config.py` exited 1 with
  `Library stubs not installed for "yaml"  [import-untyped]`. Same error also existed
  pre-Phase-18 in `task_config.py`.
- **Fix:** `pip install types-PyYAML` in ds_env. Zero source-code changes.
- **Files modified:** None (pip package only)
- **Verification:** `mypy src/prl_hgf/env/pat_rl_config.py` exits 0, "Success"
- **Committed in:** Not committed (pip env change, not tracked in repo)

**2. [Rule 1 - Bug] Fixed ruff I001 isort ordering in test file**

- **Found during:** Task 3 verification (ruff check)
- **Issue:** `ruff check tests/test_env_pat_rl_config.py` flagged I001 (import block
  unsorted). The blank line between the `from typing import Any` stdlib import and the
  `import pytest` third-party group triggered the violation.
- **Fix:** `ruff check --fix` removed the spurious blank line in the import block.
- **Files modified:** `tests/test_env_pat_rl_config.py`
- **Verification:** `ruff check tests/test_env_pat_rl_config.py` passes; 8 tests still pass
- **Committed in:** `34d4a9a` (Task 3 commit, single atomic commit)

---

**Total deviations:** 2 auto-fixed (1 blocking env install, 1 linter fix)
**Impact on plan:** Both auto-fixes trivial. No scope creep. Plan executed as specified.

## Issues Encountered

**Pre-existing test failure (NOT caused by Phase 18):**
`test_hierarchical_logp.py::test_valid_02_batched_blackjax_convergence` fails with
`ModuleNotFoundError: No module named 'blackjax'` in ds_env. This failure predates
Phase 18 — confirmed by running the full 4-module regression suite before any Phase 18
commits. 56/57 tests pass; 1 pre-existing failure is an environment issue (blackjax
not installed in ds_env) unrelated to the PAT-RL parallel stack. Pick_best_cue code
was not modified.

## Next Phase Readiness

Ready for Phase 18 Plan 02 (`pat_rl_sequence.py`):
- `PATRLConfig` / `PATRLTaskConfig` exports stable and importable
- `load_pat_rl_config()` returns validated config with n_trials=192, n_runs=4,
  trials_per_run=48, hazards, run_order, contingencies, magnitudes, timing
- All downstream modules import `from prl_hgf.env.pat_rl_config import load_pat_rl_config`
- No blockers for 18-02 (trial sequence generator) or 18-03 (HGF builders)

---
*Phase: 18-pat-rl-task-adaptation*
*Completed: 2026-04-17*
