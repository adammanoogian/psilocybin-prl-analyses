---
phase: 01-foundation
plan: "02"
subsystem: infra
tags: [python, numpy, dataclasses, pytest, simulator, prl, trial-sequence]

# Dependency graph
requires:
  - phase: 01-01
    provides: Installable prl-hgf package, load_config(), AnalysisConfig frozen dataclasses, configs/prl_analysis.yaml
provides:
  - Trial frozen dataclass (trial_idx, set_idx, phase_name, phase_label, cue_probs, best_cue)
  - generate_session(config, seed) -> list[Trial] — deterministic trial sequence from config
  - generate_reward(cue_chosen, cue_probs, rng) -> int — stochastic reward sampler
  - TransferConfig dataclass for per-set transfer phase
  - PhaseConfig.phase_label property (mirrors phase_type)
  - TaskConfig.n_sets and TaskConfig.transfer fields
  - TaskConfig.n_trials_per_set and n_trials_total properties updated for sets + transfer
  - 14 unit tests verifying trial count, phase labels, cue probs, best_cue, reproducibility
affects:
  - 02-models (consumes list[Trial] as input to HGF model fitting)
  - 03-simulation (uses generate_session + generate_reward to produce synthetic agent data)
  - All downstream phases that need correct 3-set, 420-trial session structure

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Simulator reads everything from config — no hardcoded trial counts or probabilities
    - np.random.default_rng(seed) for reproducible, seed-controlled stochastic sampling
    - Frozen dataclass Trial for immutable, hashable trial records
    - TransferConfig mirrors PhaseConfig pattern with phase_label property

key-files:
  created:
    - src/prl_hgf/env/simulator.py
    - tests/test_env_simulator.py
  modified:
    - src/prl_hgf/env/task_config.py
    - src/prl_hgf/env/__init__.py
    - configs/prl_analysis.yaml
    - pyproject.toml

key-decisions:
  - "Phase n_trials reduced from 40 to 30 to match plan spec (3 sets x 4 phases x 30 + 3 x 20 transfer = 420)"
  - "TransferConfig is a separate dataclass from PhaseConfig (no name field needed; unique per session)"
  - "phase_label property added to both PhaseConfig and TransferConfig — mirrors phase_type for API clarity"
  - "pytest pythonpath extended to include '.' (project root) so config.py is importable in tests"

patterns-established:
  - "Trial sequence pattern: generate_session returns list[Trial] where structure is config-driven and deterministic"
  - "Reward sampling pattern: generate_reward(cue_chosen, cue_probs, rng) — caller owns the RNG"
  - "Test fixture pattern: @pytest.fixture(scope='session') for config and trial list to avoid repeated load"

# Metrics
duration: 20min
completed: 2026-04-04
---

# Phase 1 Plan 02: Task Environment Simulator Summary

**Frozen Trial dataclass + generate_session() producing 420-trial sequences (3 sets x 140) from config, with 14 unit tests covering trial counts, phase labels, cue probabilities, best-cue reversals, and seed reproducibility**

## Performance

- **Duration:** 20 min
- **Started:** 2026-04-04T18:53:22Z
- **Completed:** 2026-04-04T19:13:00Z
- **Tasks:** 2
- **Files modified:** 4 modified, 2 created

## Accomplishments

- `generate_session(config, seed)` returns a `list[Trial]` of 420 trials — 3 sets of 4 phases (30 trials each) followed by a 20-trial transfer phase, all values read from config
- `generate_reward(cue_chosen, cue_probs, rng)` provides seeded stochastic reward sampling for Phase 3 synthetic agent generation
- `configs/prl_analysis.yaml` extended with `n_sets: 3` and a `transfer` block (20 trials, equal cue probs); phase `n_trials` corrected from 40 to 30
- `task_config.py` extended with `TransferConfig` dataclass, `PhaseConfig.phase_label` property, and `TaskConfig.n_sets`/`transfer` fields with validation
- 14 pytest unit tests pass: trial count (420), set structure (140/set), transfer count (60), phase labels (stable/volatile), cue_probs correctness, best_cue at each reversal, seed reproducibility (ENV-05), sequential trial_idx, deterministic reward generation

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the task environment simulator** - `0c81322` (feat)
2. **Task 2: Write comprehensive unit tests for the environment simulator** - `eb6aae2` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/prl_hgf/env/simulator.py` - Trial dataclass, generate_session(), generate_reward()
- `tests/test_env_simulator.py` - 14 unit tests for simulator correctness and reproducibility
- `src/prl_hgf/env/task_config.py` - Added TransferConfig, PhaseConfig.phase_label, TaskConfig.n_sets + transfer fields
- `src/prl_hgf/env/__init__.py` - Re-exports generate_session, generate_reward, Trial, load_config, AnalysisConfig
- `configs/prl_analysis.yaml` - n_sets=3, phase n_trials 40→30, transfer block added
- `pyproject.toml` - pytest pythonpath extended from ["src"] to ["src", "."]

## Decisions Made

- **Phase trial count 40 → 30:** Plan spec requires 3 sets × 4 phases × 30 = 360 + 3 × 20 transfer = 420 total. The original YAML used 40 trials/phase (yielding 480 + transfer), which contradicted the plan. Corrected to 30.
- **TransferConfig as separate dataclass:** Transfer phase has no `name` field (it's always "transfer") unlike `PhaseConfig`. A separate `TransferConfig` is cleaner and avoids an unused required field.
- **phase_label property:** Added to both `PhaseConfig` and `TransferConfig` to provide a consistent `.phase_label` API in the simulator without duplicating the `phase_type` string in YAML or dataclass fields.
- **pytest pythonpath fix:** Root `config.py` was not on the pytest import path (only `src/` was listed). Extended to include `"."` so `from config import CONFIGS_DIR` resolves during test collection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated YAML phase n_trials from 40 to 30 to match plan spec**

- **Found during:** Task 1 analysis (pre-implementation)
- **Issue:** configs/prl_analysis.yaml had n_trials=40 per phase, but plan spec requires 30 (3×4×30 + 3×20 = 420). Using 40 would produce 3×4×40 + 3×20 = 540 trials, breaking all downstream trial count assertions.
- **Fix:** Changed all four phases from n_trials=40 to n_trials=30; added transfer block and n_sets=3 to YAML
- **Files modified:** configs/prl_analysis.yaml
- **Verification:** `config.task.n_trials_total == 420` confirmed
- **Committed in:** 0c81322 (Task 1 commit)

**2. [Rule 2 - Missing Critical] Added TransferConfig and n_sets to task_config.py**

- **Found during:** Task 1 (simulator references config.task.transfer and config.task.n_sets)
- **Issue:** AnalysisConfig/TaskConfig had no transfer field or n_sets field; simulator.py code from plan spec required both
- **Fix:** Added TransferConfig frozen dataclass, phase_label property on PhaseConfig and TransferConfig, n_sets + transfer fields to TaskConfig with validation, updated _parse_task_config
- **Files modified:** src/prl_hgf/env/task_config.py
- **Verification:** load_config() returns n_sets=3, transfer.n_trials=20; ruff passes
- **Committed in:** 0c81322 (Task 1 commit)

**3. [Rule 3 - Blocking] Added "." to pytest pythonpath in pyproject.toml**

- **Found during:** Task 2 (first pytest run)
- **Issue:** pytest could not import `prl_hgf.env.task_config` because `task_config.py` does `from config import CONFIGS_DIR` and root `config.py` was not on the Python path (only `src/` was listed)
- **Fix:** Changed `pythonpath = ["src"]` to `pythonpath = ["src", "."]` in `[tool.pytest.ini_options]`
- **Files modified:** pyproject.toml
- **Verification:** `pytest tests/test_env_simulator.py -v` — all 14 tests collected and passed
- **Committed in:** eb6aae2 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 Rule 1 - Bug, 1 Rule 2 - Missing Critical, 1 Rule 3 - Blocking)
**Impact on plan:** All three were necessary to make the simulator correct and testable. No scope creep.

## Issues Encountered

- `conda run -n ds_env python -c "..."` cannot find `config` module unless the working directory is the project root (same issue as 01-01). Continued using temp `.py` files written to project root and deleted before commit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Environment simulator complete and tested: `generate_session(config, seed)` produces 420-trial sessions
- Phase 2 models can directly consume `list[Trial]` output — phase_name, phase_label, cue_probs, best_cue all present
- Phase 3 simulation can use `generate_reward(cue_chosen, cue_probs, rng)` for stochastic outcome generation
- All 14 simulator unit tests pass; pytest infrastructure functional with correct pythonpath

---
*Phase: 01-foundation*
*Completed: 2026-04-04*
