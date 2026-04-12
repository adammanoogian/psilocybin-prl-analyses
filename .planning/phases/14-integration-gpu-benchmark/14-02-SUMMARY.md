---
phase: 14-integration-gpu-benchmark
plan: 02
subsystem: benchmarking
tags: [nvidia-smi, threading, decision-gate, gpu-monitoring, JAX, benchmarking]

# Dependency graph
requires:
  - phase: 14-01
    provides: run_sbf_iteration with use_legacy=False batched path + helpers
  - phase: 12
    provides: fit_batch_hierarchical returning joint InferenceData
provides:
  - apply_decision_gate public function (BENCH-02 gate formula)
  - _GpuMonitor background-threaded nvidia-smi polling class
  - Rewritten _run_benchmark: full batched iteration timing + GPU monitoring + decision gate + JAX cache test
  - benchmark_batched.json schema and write path
  - _update_state_md helper for STATE.md decision row append
  - 5 new unit tests (decision gate boundary/schema + GPU monitor graceful degradation)
affects:
  - phase 15 (production sweep): decision gate result determines mgpu vs comp partition selection

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Background-threaded nvidia-smi polling (2s interval) via threading.Thread + subprocess.run
    - Decision gate formula: per_iter_s * 600 / 3600 > 50 -> cpu_comp else gpu
    - JAX compilation cache test: two back-to-back fit_batch_hierarchical calls record cold vs warm JIT time

key-files:
  created: []
  modified:
    - src/prl_hgf/power/iteration.py
    - scripts/08_run_power_iteration.py
    - tests/test_power_iteration.py

key-decisions:
  - "_update_state_md uses string search for table header then appends row; handles missing STATE.md gracefully"
  - "BLE001 noqa on broad except in _GpuMonitor._run: intentional swallow for nvidia-smi transient failures"
  - "patch target uses module-qualified name (08_run_power_iteration.subprocess.run) for test isolation"
  - "import threading and import subprocess promoted to module level to satisfy linting"

patterns-established:
  - "Decision gate is a pure function in iteration.py (apply_decision_gate) testable independently of the script"
  - "_GpuMonitor is a self-contained class in the script; daemon thread ensures it doesn't block exit"

# Metrics
duration: 10min
completed: 2026-04-12
---

# Phase 14 Plan 02: GPU Benchmark and Decision Gate Summary

**apply_decision_gate public function + _GpuMonitor background nvidia-smi class + fully rewritten _run_benchmark benchmarking one full 300-session x 2-model batched iteration with JAX cache timing and benchmark_batched.json output**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-12T16:15:13Z
- **Completed:** 2026-04-12T16:25:07Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `apply_decision_gate` public function to `iteration.py`, exported in `__all__`, implementing BENCH-02 formula (`per_iter_s * 600 / 3600 > 50`)
- Added `_GpuMonitor` class to benchmark script with background-threaded nvidia-smi polling (2s intervals), peak VRAM, mean GPU utilization, and total VRAM properties
- Rewrote `_run_benchmark` to: (1) test JAX compilation cache with two back-to-back small fits (BENCH-05), (2) time one full `run_sbf_iteration` call at max N with GPU monitor running (BENCH-01 + BENCH-04), (3) apply decision gate and write `benchmark_batched.json` (BENCH-02), (4) append gate decision row to `STATE.md`
- Added 5 unit tests: decision gate at GPU boundary (100s), CPU boundary (500s), exact 50-hour boundary (300s), schema completeness, and GPU monitor graceful degradation when nvidia-smi is absent

## Task Commits

1. **Task 1: Add apply_decision_gate + _GpuMonitor** - `3858b07` (feat)
2. **Task 2: Rewrite _run_benchmark** - `aba749b` (feat)
3. **Task 3: Add unit tests** - `c54511e` (test)

## Files Created/Modified

- `src/prl_hgf/power/iteration.py` - Added `apply_decision_gate` function and updated `__all__`
- `scripts/08_run_power_iteration.py` - Added `_GpuMonitor` class, `_update_state_md` helper, rewrote `_run_benchmark`; promoted `import subprocess` and `import threading` to module level
- `tests/test_power_iteration.py` - Added 5 new tests (tests 10-14) for decision gate logic and GPU monitor

## Decisions Made

- `_update_state_md` uses string search for Key Decisions table header then appends row after last `| ` line; handles missing STATE.md gracefully with a warning print rather than raising.
- `BLE001` noqa suppressed on broad `except Exception` in `_GpuMonitor._run`: intentional swallow for nvidia-smi transient failures (missing binary, timeout, parse error).
- Patch target uses module-qualified name (`08_run_power_iteration.subprocess.run`) in the GPU monitor test rather than `subprocess.run` globally, to isolate only the script's subprocess calls.
- `import subprocess` and `import threading` promoted to module level (were inside old `_run_benchmark` body) to satisfy ruff linting and make `_GpuMonitor` functional at class-definition time.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] f-string without placeholder in _run_benchmark print statement**
- **Found during:** Task 2 linting
- **Issue:** `print(f"  Formula: per_iter_s * 600 / 3600 > 50")` used `f` prefix with no interpolation; ruff F541
- **Fix:** Removed `f` prefix
- **Files modified:** `scripts/08_run_power_iteration.py`
- **Verification:** `ruff check` passes
- **Committed in:** aba749b (Task 2 commit)

**2. [Rule 1 - Bug] Import ordering violation in test file**
- **Found during:** Task 3 final lint
- **Issue:** Added `import importlib`, `import time`, `from unittest.mock import patch` as a second import block after third-party imports; ruff I001
- **Fix:** Merged all stdlib imports into single sorted block at top of file
- **Files modified:** `tests/test_power_iteration.py`
- **Verification:** `ruff check` passes
- **Committed in:** c54511e (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - linting bugs caught during verification)
**Impact on plan:** Both trivial linting fixes. No scope creep.

## Issues Encountered

None beyond the two linting fixes above.

## Next Phase Readiness

- All BENCH-01/02/04/05 code paths implemented and tested
- `--benchmark` flag is ready to run on a GPU node; outputs `benchmark_batched.json` with all required fields
- Decision gate result will be appended to STATE.md when the benchmark runs on GPU hardware
- Phase 15 (production sweep) can proceed; the gate decision (gpu vs cpu_comp) determines which SLURM partition to target

---
*Phase: 14-integration-gpu-benchmark*
*Completed: 2026-04-12*
