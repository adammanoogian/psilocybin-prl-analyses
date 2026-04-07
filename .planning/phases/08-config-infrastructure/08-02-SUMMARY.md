---
phase: 08-config-infrastructure
plan: "02"
subsystem: infra
tags: [numpy, seedsequence, parquet, pyarrow, slurm, pandas, power-analysis]

# Dependency graph
requires:
  - phase: 08-01
    provides: PowerConfig, load_power_config, make_power_config in power/ subpackage
provides:
  - make_child_rng: SeedSequence-based independent RNG per SLURM task
  - POWER_SCHEMA: 13-column parquet schema constant
  - write_parquet_row: schema-enforced parquet writer
  - cluster/08_power_sweep.slurm: SLURM array template with %50 throttle
  - scripts/08_run_power_iteration.py: entry point with --dry-run and --output-dir
  - tests/test_power_infra.py: 12 unit/integration tests
affects: [phase-09, phase-10, cluster-submission, power-sweep-orchestration]

# Tech tracking
tech-stack:
  added: [pyarrow>=23 (installed in ds_env for parquet engine)]
  patterns:
    - SeedSequence.spawn for independent parallel RNG streams (guarantees distinct state vectors)
    - Schema-enforced parquet writes via POWER_SCHEMA dict + write_parquet_row
    - SLURM array task entry points accept --task-id and --output-dir for testability
    - sys.path.insert(0, project_root) at top of cluster scripts for editable-install-free operation

key-files:
  created:
    - src/prl_hgf/power/seeds.py
    - src/prl_hgf/power/schema.py
    - cluster/08_power_sweep.slurm
    - scripts/08_run_power_iteration.py
    - tests/test_power_infra.py
  modified:
    - src/prl_hgf/power/__init__.py

key-decisions:
  - "write_parquet_row rejects both missing and extra columns to prevent silent schema drift"
  - "--output-dir flag on entry point enables integration tests to write to tmp_path without touching results/"
  - "np.True_ is-identity check replaced with == True in roundtrip test (Rule 1 bug)"

patterns-established:
  - "Entry point scripts: always add --output-dir for test isolation"
  - "RNG streams: always use SeedSequence.spawn, never integer seeding"
  - "Parquet writes: always go through write_parquet_row for schema enforcement"

# Metrics
duration: 12min
completed: 2026-04-07
---

# Phase 08 Plan 02: Seeds, Schema, SLURM, and Entry Point Summary

**SeedSequence-backed independent RNG streams, 13-column parquet schema enforcement, SLURM %50-throttled array template, and a --dry-run entry point that wires everything together end-to-end**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-07T15:25:27Z
- **Completed:** 2026-04-07T15:37:54Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- `make_child_rng` uses `SeedSequence(master_seed).spawn(n_jobs)` to guarantee distinct PCG64 state vectors for every SLURM array task — verified directly in `test_make_child_rng_distinct_state_vectors`
- `write_parquet_row` enforces the 13-column `POWER_SCHEMA` (missing and extra column detection) before writing via pyarrow; auto-creates parent dirs
- SLURM template (`08_power_sweep.slurm`) uses `%A_%a` log naming, default `--array=0-9%50` smoke-test directive with override instructions, JAX compilation cache, and GPU verification matching existing `04_fit_mcmc_gpu.slurm` conventions
- Entry point (`08_run_power_iteration.py`) reads `SLURM_ARRAY_TASK_ID`, builds child RNG, and in `--dry-run` mode writes a placeholder parquet row — all 13 schema columns confirmed correct
- 12 tests pass: 6 RNG tests (state vector, correlation, reproducibility, validation), 5 schema tests (count, roundtrip+dtype, missing col, extra col, dir creation), 1 subprocess integration test

## Task Commits

Each task was committed atomically:

1. **Task 1: Create seeds.py and schema.py** - `36b85b5` (feat)
2. **Task 2: SLURM template and entry point script** - `492e401` (feat)
3. **Task 3: Unit tests for seeds, schema, integration smoke test** - `13b2bc9` (test)

## Files Created/Modified

- `src/prl_hgf/power/seeds.py` - `make_child_rng(master_seed, n_jobs, job_index)` via SeedSequence
- `src/prl_hgf/power/schema.py` - `POWER_SCHEMA` dict and `write_parquet_row` with schema validation
- `src/prl_hgf/power/__init__.py` - Re-exports `make_child_rng`, `POWER_SCHEMA`, `write_parquet_row`
- `cluster/08_power_sweep.slurm` - SLURM array template, `%A_%a` logs, `--array=0-9%50` default
- `scripts/08_run_power_iteration.py` - Entry point: RNG + config + parquet; `--dry-run`, `--output-dir`
- `tests/test_power_infra.py` - 12 unit/integration tests (234 lines)

## Decisions Made

- `write_parquet_row` rejects both missing and extra columns: strict schema enforcement prevents silent drift as Phase 10 adds real pipeline results
- `--output-dir` flag added to entry point: allows tests to redirect to `tmp_path` without writing to `results/power/`, keeping the test suite clean
- `np.True_ is True` check changed to `== True` in roundtrip test: numpy boolean scalars are not Python singletons — Rule 1 bug fix applied in Task 3

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed pyarrow in ds_env**
- **Found during:** Task 2 (entry point dry-run verification)
- **Issue:** `pandas.to_parquet(engine="pyarrow")` raised `ImportError: Missing optional dependency 'pyarrow'`; pyarrow not installed in ds_env despite being in `pyproject.toml`
- **Fix:** `conda run -n ds_env pip install pyarrow` (installed 23.0.1)
- **Files modified:** ds_env conda environment (no source file changes)
- **Verification:** Dry-run produced valid parquet; 12 tests pass including roundtrip
- **Committed in:** `492e401` (Task 2 commit, noted in message)

**2. [Rule 1 - Bug] Fixed np.True_ identity check in roundtrip test**
- **Found during:** Task 3 (first test run)
- **Issue:** `assert df["bf_exceeds"].iloc[0] is True` failed with `np.True_ is True` — numpy boolean scalars do not satisfy Python `is` identity
- **Fix:** Changed to `assert df["bf_exceeds"].iloc[0] == True  # noqa: E712`
- **Files modified:** `tests/test_power_infra.py`
- **Verification:** All 12 tests pass after fix
- **Committed in:** `13b2bc9` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## User Setup Required

None - no external service configuration required. pyarrow installed automatically in ds_env.

## Next Phase Readiness

- Phase 8 infrastructure complete: RNG streams, parquet schema, SLURM template, and entry point all tested
- Phase 9 (simulation sweep orchestration) can import from `prl_hgf.power` and invoke `scripts/08_run_power_iteration.py`
- Full SLURM smoke test (`sbatch cluster/08_power_sweep.slurm` on cluster) is the next logical verification step
- Phase 10 (full simulate+fit pipeline): replace `--dry-run` placeholder in `main()` with actual HGF simulation + MCMC calls; no structural changes needed

---
*Phase: 08-config-infrastructure*
*Completed: 2026-04-07*
