---
phase: quick-004
plan: "004"
subsystem: patrl-smoke-infrastructure
tags: [slurm, smoke-test, vb-laplace, housekeeping, cluster]
completed: 2026-04-18

dependency-graph:
  requires: [18-05]
  provides:
    - cluster/18_smoke_patrl_cpu.slurm (cluster entry point for Phase 18 PAT-RL smoke)
    - scripts/12_smoke_patrl_foundation.py --dry-run (validate wiring without MCMC)
    - tests/test_smoke_patrl_foundation.py (5 structural tests, no MCMC invocation)
    - .planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md
  affects: [18-06, quick-005]

tech-stack:
  added: []
  patterns:
    - SLURM script mirroring 16_smoke_test_cpu.slurm (separate JAX cache path, auto-push, partition=comp)
    - --dry-run flag pattern for cluster-targeted scripts (skips MCMC, validates wiring, exits 0)
    - Structural test pattern (py_compile + subprocess argparse rejection + source text scan; no MCMC)
    - VB-Laplace + NUTS dual-path recommendation (Option C)

key-files:
  created:
    - cluster/18_smoke_patrl_cpu.slurm
    - .planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md
  modified:
    - scripts/12_smoke_patrl_foundation.py  (added --dry-run flag)
    - tests/test_smoke_patrl_foundation.py  (rewrite: 5 structural tests, no MCMC)
    - pyproject.toml                        (mypy ignore_missing_imports for prl_hgf.* + config)
    - .planning/ROADMAP.md                  (Phase 18 entry + plan-list)
    - .planning/phases/18-pat-rl-task-adaptation/18-RESEARCH.md  (user decision addendum)
  deleted:
    - _run_tests.py
    - _run_models_tests.py
    - _verify_loader.py
    - _verify_yaml.py
  tracked-untracked:
    - .planning/phases/18-pat-rl-task-adaptation/18-01-PLAN.md
    - .planning/phases/18-pat-rl-task-adaptation/18-02-PLAN.md
    - .planning/phases/18-pat-rl-task-adaptation/18-03-PLAN.md
    - .planning/phases/18-pat-rl-task-adaptation/18-04-PLAN.md
    - .planning/phases/18-pat-rl-task-adaptation/18-05-PLAN.md
    - .planning/phases/18-pat-rl-task-adaptation/18-06-PLAN.md

decisions:
  - id: D1
    decision: VB-Laplace + BlackJAX NUTS dual-path (Option C)
    rationale: Implementation cost ~380 LOC; unblocks PEB immediately; preserves NUTS as validation
    phase: quick-004

metrics:
  duration: ~30 min
  completed: 2026-04-18
---

# Phase quick-004: PAT-RL Smoke Infrastructure + VB-Laplace Exploration

**One-liner**: Cluster-targeted PAT-RL smoke SLURM + --dry-run flag + 5 structural tests + VB-Laplace Option C recommendation (dual NUTS + Laplace paths).

## What Shipped

### Task 1: Cluster smoke setup + Phase-18 housekeeping

**cluster/18_smoke_patrl_cpu.slurm** — new SLURM entry point mirroring
`cluster/16_smoke_test_cpu.slurm` exactly: partition=comp, 64G, 8 cpus,
08:00:00 walltime, `conda activate ds_env` with scratch fallback, separate
JAX compilation cache at `.jax_cache_cpu/patrl/${SLURMD_NODENAME}` (avoids
collision with pick_best_cue CPU cache), auto-push of logs and CSVs on
completion.  Five env-override variables (`PRL_PATRL_SMOKE_LEVEL`,
`PRL_PATRL_SMOKE_N`, `PRL_PATRL_SMOKE_TUNE`, `PRL_PATRL_SMOKE_DRAWS`,
`PRL_PATRL_SMOKE_SEED`) default to level=2, N=5, tune=500, draws=500,
seed=42.

**scripts/12_smoke_patrl_foundation.py** — retained as cluster entrypoint;
`--dry-run` flag added.  Dry-run skips steps 3-5 (MCMC fit, export,
sanity-check), runs steps 1-2 only (load config, simulate cohort + HGF
forward pass), prints participant/trial summary, exits 0.  Blackjax import
remains lazy inside `_fit` so dry-run is portable without blackjax.
Verified: `python scripts/12_smoke_patrl_foundation.py --dry-run --n-participants 2`
exits 0 in 1.0 s.

**tests/test_smoke_patrl_foundation.py** — complete rewrite.  Old body
(subprocess MCMC, RUN_SMOKE_TESTS env gate, blackjax dependency) removed.
Five structural tests: (1) py_compile syntax check, (2) argparse rejects
`--level 4`, (3) source-text scan for pick_best_cue import patterns,
(4) source-text check that blackjax is not imported at module top-level,
(5) parametrized py_compile canary for three pick_best_cue modules.
All 7 collected items (5 functions, 3 from parametrize) pass locally on
Windows without blackjax.

**Housekeeping**:
- Deleted 4 scratch probe files: `_run_tests.py`, `_run_models_tests.py`,
  `_verify_loader.py`, `_verify_yaml.py` (were untracked; removed from
  working tree).
- Committed `pyproject.toml` mypy `ignore_missing_imports` override for
  `prl_hgf.*` and `config` (left over from prior executor session).
- Added all 6 untracked `18-NN-PLAN.md` files to git history.
- Committed doc-only edits to `.planning/ROADMAP.md` (Phase 18 entry
  + plan list) and `18-RESEARCH.md` (user decision addendum from 2026-04-17).

### Task 2: VB-Laplace feasibility memo

`.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_FEASIBILITY.md`

Seven sections covering:

1. **Sampling friction diagnosis** — backed by actual cluster log numbers
   from `cluster/logs/smoke16cpu_54882395.out`: Cold JIT 11,688 s,
   warm JIT ~8,000 s, integration_steps=1023 (max_tree_depth saturated
   every step), 0% divergences, ~400 s/draw pure sampling cost.

2. **tapas VB-Laplace algorithm** — 6-step recap (reparametrise →
   quasi-Newton MAP → Ridders' Hessian → Laplace N(θ*,H^{-1}) →
   back-transform via delta rule).

3. **Mapping onto existing logp surface** — confirms
   `_build_patrl_log_posterior` is JAX-differentiable throughout; exact
   Hessian via `jax.hessian` (no Ridders' numerical error); `jaxopt.LBFGS`
   for MAP; proposed `fit_vb_laplace_patrl.py` module scoped.

4. **Implementation cost** — ~380 LOC total (module + tests + shim), one
   quick plan, no new dependencies beyond `jaxopt`.

5. **Tradeoffs** — structured comparison table; honest caveat that Laplace
   also fails on the κ × ω₃ ridge (neither path cleanly solves 3-level).

6. **Recommendation: Option C** — dual NUTS + Laplace paths; Laplace in
   quick-005 as second fit path; both export same CSV schema; user selects
   per downstream analysis.  Downgrade triggers to A or B documented.

7. **Non-goals** — no implementation, no timings beyond section 1, no PEB
   decision.

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Commit | Message | Key files |
|--------|---------|-----------|
| 468278d | feat(quick-004): cluster-targeted PAT-RL foundation smoke + structural tests | cluster/18_smoke_patrl_cpu.slurm, scripts/12_smoke_patrl_foundation.py, tests/test_smoke_patrl_foundation.py |
| b79e6df | chore: mypy ignore_missing_imports for prl_hgf.* and config | pyproject.toml |
| 82c421d | docs(18): track all 18-NN-PLAN.md files + roadmap/research updates | 6× 18-NN-PLAN.md, ROADMAP.md, 18-RESEARCH.md |
| 362808d | docs(quick-004): VB-Laplace feasibility memo + quick plan | VB_LAPLACE_FEASIBILITY.md, 004-PLAN.md |

## VB-Laplace Recommendation (section 6 verbatim summary)

> **Recommended: Option C** — implement VB-Laplace in quick-005 as a *second*
> fit path alongside BlackJAX NUTS. Export both; compare posteriors on shared
> smoke subjects. Both paths land in the pipeline; user selects per downstream
> analysis.
>
> Rationale: (1) ~380 LOC implementation cost, one quick plan, no new
> dependencies; (2) unblocks PEB development immediately via mean+cov moments;
> (3) mirrors matlab tapas default path for cross-validation; (4) existing
> `_build_patrl_log_posterior` is JAX-differentiable so exact Hessian via
> `jax.hessian` is essentially free — materially better than tapas Ridders'.

## Next Steps

**If memo recommendation stands (Option C):** open quick-005 to implement
`fit_vb_laplace_patrl.py`.  Estimated scope: 2 tasks, wave 1.
- Task 1: `fit_vb_laplace_patrl.py` (MAP via `jaxopt.LBFGS` + exact Hessian
  via `jax.hessian` + Laplace back-transform) + unit tests.
- Task 2: exporter shim (`export_subject_trajectories_vb_laplace`) producing
  same CSV schema as Phase 18-05 trajectories.

**Cluster smoke (independent of Option C):** `sbatch cluster/18_smoke_patrl_cpu.slurm`
to get first real PAT-RL NUTS numbers.  If it blows past 6 h or shows > 20%
divergences on 2-level, downgrade recommendation to Option A.
