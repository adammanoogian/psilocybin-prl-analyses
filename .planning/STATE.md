# Current State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-04)

**Core value:** Reliable, scalable hierarchical Bayesian HGF fitting that exposes proper posterior UQ at production cohort sizes.
**Current focus:** Phase 28 — FitConfig + HGFPriorSpec refactor (next; Phase 27 closed 2026-05-08).

## Current Position

Phase: 28 of 10 (FitConfig + HGFPriorSpec refactor)
Plan: 3 of 5 complete (28-01, 28-02, 28-03 closed)
Status: In progress
Last activity: 2026-05-17 — Completed 28-03-PLAN.md (HGFPriorSpec extraction)

Progress: [████░░░░░░] 16% (5 of 9 remaining v1.0 plans in Phase 28)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Phase 27 wall-clock: 4 plans across ~3 days (compute-heavy)
- Phase 28 wall-clock: 3 plans in ~1 day (code-only, fast)
- Effective work time across plans: ~5.8h

**By Phase:**

| Phase | Plans | Status | Completed |
|-------|-------|--------|-----------|
| 27-dependency-upgrade-chain | 4 of 4 | ✓ Complete | 2026-05-08 |
| 28-fitconfig-hgfpriorspec-refactor | 3 of 5 | In progress | — |

*Updated after each phase verification.*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- **Build order flipped vs original framing** (Phase 28 before benchmarks): per ARCHITECTURE.md, per-flag SLURM env-passthrough plumbing is wasted work without `FitConfig` first; benchmarks land LAST against fully-wired toolkit.
- **M2 + W4 fused into Phase 33**: per ARCHITECTURE.md, both touch prior construction in the same files; sequential implementation means rewriting `_make_logdensity_closure` twice.
- **Phase 35 (M3 Gibbs) is conditional**: fires only on Phase 34 evidence per M3-01 — research disagrees with building it speculatively given LOW confidence on production-scale convergence.
- **Pre-registered audit protocol (AUDIT-01) gates Phase 32**: must land before any audit run per Pitfall 10 (confounded sampler audit).
- **Per-parameter non-centering field shape lands in Phase 30**: `MitigationConfig.non_centered: tuple[str, ...]` is a list of parameter names, not a global boolean — even though the consumer ships in Phase 33 (Pitfall 3 prevention).
- **[27-01] Strategy C (pyhgf fork) mandatory**: `jaxlib.xla_extension.PjitFunction` removed in JAX 0.5+; all 6 matrix cells (pyhgf {0.2.8,0.2.11} x JAX {0.7.0,0.9.0,0.10.0}) fail at import time; `--no-deps` Strategy A is unviable.
- **[27-01] Fork base: pyhgf 0.2.8**: project-validated version; minimum fix is one line in `pyhgf/typing.py` replacing `PjitFunction` with `Callable`.
- **[27-01] JAX pin narrowed to >=0.9.0,<0.10**: BlackJAX 1.5 minimum; verifier flagged that 27-01's original DEPS-02 decision said `<0.11` but the live pyproject is `<0.10` due to `nvidia-cusparse-cu12>=12.9` PyPI gap hit on m3g107 — Phase 28 should consider whether to widen back to `<0.11`.
- **[27-02] Fork install: pip wheel + in-place patch**: pyhgf 0.2.8 uses maturin (Rust build backend); no Rust toolchain available. Install = `pip install --no-deps pyhgf==0.2.8` + `python scripts/ci/patch_pyhgf_typing.py`. vendor/pyhgf/ is reference archive only.
- **[27-02] vendor/ as plain directory**: no git submodule; .git removed from clone; Python source only tracked.
- **[27-02] requirements-v10.txt is platform-specific**: Windows/CPU snapshot; must regenerate on M3 for Linux/CUDA12 jaxlib wheel.
- **[27-03] BlackJAX 1.5 has NO low-rank API**: `window_adaptation` only exposes `is_mass_matrix_diagonal: bool`. Setting `False` triggers full dense (O(D²)/step), not low-rank. Phase 29 mass-matrix wiring will need to honor this — the Roadmap's `mass_matrix_kind: Literal["diagonal", "low_rank", "dense"]` enum has no current backend support for `low_rank` and the field shape may need to shrink to `Literal["diagonal", "dense"]` until a future BlackJAX release.
- **[27-03] Dense+P=30+1000-warmup is infeasible at 24h**: empirical evidence from job 55198489. Phase 29 pre-flight estimator should refuse this config by default; Phase 30 Laplace warmup is the most promising single mitigation.
- **[27-04] CONDA_ENV opt-in pattern**: `_CONDA_ENV="${CONDA_ENV:-ds_env}"` in 9 SLURM scripts; default unchanged. Cluster-wide promotion stays deferred since DEPS-05 came back NOT CLEARED.
- **[27-04] PRL_EXPLAIN_CACHE_MISSES env-gate**: 5 sites in `scripts/03_pre_analysis/03_run_power_iteration.py` gated against an upstream JAX 0.9.2 partial_eval bug that fires when `jax_explain_cache_misses=True` interacts with nested JIT'd scan bodies.

### Pending Todos

None tracked in `.planning/todos/pending/`.

### Blockers/Concerns

- **pyhgf 0.2.8 ↔ JAX > 0.4.31 compat** — RESOLVED by 27-02. Strategy C: pip wheel + in-place patch via `scripts/ci/patch_pyhgf_typing.py`. Probe confirmed PASS at JAX 0.10.
- **DEPS-05 cliff not cleared by BlackJAX 1.5 alone** — RESOLVED-AS-NOT-CLEARED by 27-03 (TIMEOUT @ 24h). Phase 29 (M1 wiring + pre-flight) and Phase 30 (Laplace warmup) inherit; cliff mitigation is the entire purpose of those phases.
- **cuSPARSE 12.5 / driver 13.0 GPU lottery on M3** — open. Some nodes (m3g112) fall back to CPU under `ds_env_v10` despite GPU allocation; m3g108 works. Phase 28 jobs need device-check + requeue OR `--nodelist` constraint OR a different cuSPARSE wheel pin. Memo'd in `memory/project_phase27_cusparse_node_lottery.md`.
- **Per-parameter vs shared σ_θ identifiability at P=200 with K=2-3 covariates** — research-flagged for Phase 33 pre-planning; sim-to-inference evidence required before committing to the more flexible variant.
- **M3 conditional-independence proof + per-block convergence diagnostics** — research-flagged for Phase 35 pre-planning if it fires; lowest-confidence area in v1.0 scope.

### v1.0 Carry-forward (from pre-v1.0 work)

- BlackJAX `max_num_doublings` duplicate-kwarg bug fixed in `b737fe6` (2026-05-04). Capability-map's "P=50 was last PASS" reflects this, not a hardware cliff.
- 3-level NUTS conditioning cliff is real even at n_per_group=5 (P=30): step-size collapses to 7.245e-10, depth-10 saturated, on both A100 80GB and L40S. Diagonal mass matrix can't precondition the κ × ω₂ × ω₃ banana. This is what the v1.0 mitigation ladder addresses. Phase 27 confirms BlackJAX 1.5 dense alone is also insufficient.
- Laplace warmup variant (commit `385e8c3`, Phase 14.2-05) is wired but never tested at scale due to the kwarg bug. First v1.0 phase to exercise it: Phase 30.
- `tests/integration/test_capability_map.py` closure-guard validates the map is well-formed. Don't break it; Phase 31 extends it.

## Session Continuity

Last session: 2026-05-17
Stopped at: Completed 28-03-PLAN.md (HGFPriorSpec extraction)
Resume: `/gsd:execute-phase` on 28-04-PLAN.md

---
*Last updated: 2026-05-17 after 28-03 completion*
