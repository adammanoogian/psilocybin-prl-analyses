# Current State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-04)

**Core value:** Reliable, scalable hierarchical Bayesian HGF fitting that exposes proper posterior UQ at production cohort sizes.
**Current focus:** Overnight autonomous code sweep (2026-05-21). Phase 36-01 (PPC), 36-02 (diagnostics CSV), 31-03 (closure guard), 36-03 (sampler recommendation) executed locally. Cluster benchmark debt accumulated for Phases 31, 32, 34.

## Current Position

Phase: 36 (polish) — code work in parallel with Phase 31/32/34 cluster debt
Plan: Multiple phases partially advanced (see phase table)
Status: Active — code deliverables landing, cluster benchmarks deferred
Last activity: 2026-05-21 — Overnight autonomous execution of Phase 36-01, 36-02, 31-03, 36-03, 34-02 partial

Progress: [████████████████████████] 88% (~29 of ~33 v1.0 plans touched)

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Phase 27 wall-clock: 4 plans across ~3 days (compute-heavy)
- Phase 28 wall-clock: 5 plans in ~1 day (code-only, fast)
- Effective work time across plans: ~6.5h

**By Phase:**

| Phase | Plans | Status | Completed |
|-------|-------|--------|-----------|
| 27-dependency-upgrade-chain | 4 of 4 | ✓ Complete | 2026-05-08 |
| 28-fitconfig-hgfpriorspec-refactor | 5 of 5 | Complete | 2026-05-17 |
| 29-m1-dense-lowrank-mass-matrix-wiring | 3 of 3 | Complete | 2026-05-17 |
| 30-laplace-warmup-fp64-multigpu-flags | 4 of 4 | ✓ Complete | 2026-05-17 |
| 31-benchmark-no-pooling-mode | 2.5 of 3 | 31-03 code done (xfail); 31-02 cluster DEBT | — |
| 32-sampler-audit-harness | 3.5 of 5 | 32-04 aggregation script done; cluster DEBT | — |
| 33-ts1-m2-w4-fused | 6 of 6 | Complete (cluster pending) | 2026-05-19 |
| 34-benchmark-hierarchical-mode | 1.5 of 3 | 34-02 smoke done, full grid DEBT | — |
| 36-polish | 2.5 of 3 | 36-01+36-02 code done; 36-03 partial | — |

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
- **[28-05] Legacy CLI args kept as deprecated**: --fit-chains/--fit-draws/--fit-tune/--max-tree-depth remain in 03_run_power_iteration.py as deprecated; --fit-config takes precedence when provided. Smoke test and laplace-only paths rely on the legacy attrs.
- **[28-05] FitConfig populates legacy args namespace**: Rather than threading FitConfig through all code paths, loading from YAML sets args.fit_chains/fit_draws/fit_tune/max_tree_depth so all downstream FitConfig construction continues unchanged.
- **[28-04] prior_spec stays separate from FitConfig**: Prior distributions are domain-specific (vary per experiment hypothesis), while FitConfig is infrastructure-level (sampler settings, chain count); mixing them would conflate concerns. `HGFPriorSpec` is passed as a separate optional kwarg.
- **[28-04] run_sbf_iteration legacy kwargs preserved**: `fit_config=None` triggers internal FitConfig construction from the legacy `n_chains/n_draws/n_tune` kwargs for backward compatibility with callers not yet migrated.
- **[29-02] 25% device memory threshold for dense refusal**: Pre-flight refuses dense mass matrix when D^2*8*n_chains*(4 if pmap) exceeds 25% of detected device memory. Error message points to low_rank and M3 cluster.
- **[30-01] GUARD-05 via prl_hgf.runtime.set_x64**: Centralized fp64 toggle sets env var + jax.config.update + post-call assertion; RuntimeError on silent-flip (PyTensor P6 prevention). All scripts must import from prl_hgf.runtime, not call jax.config.update directly.
- **[30-01] Phase 32 sampler-audit obligation**: `04_sampler_audit.py` does not exist yet; Phase 32 plan MUST include `from prl_hgf.runtime import set_x64` as its first JAX-config action.
- **[30-02] n_starts=4 not threaded through FitConfig**: roadmap specifies "n >= 4" as a fixed architectural minimum, not a user-configurable knob; `use_laplace_warmup: bool` in `MitigationConfig` is the user-facing toggle.
- **[30-02] Two-pass Hessian for basin comparison**: Hessian diagonal at best MAP provides the SE reference (se = sqrt(1/hess_diag_pd)); more principled than raw parameter spread vs perturbation scale.
- **[30-02] n_success < 2 returns None without basin check**: insufficient convergence to judge unimodality — conservative fallback to window_adaptation.
  - **[30-03] check_rep=False mandatory for NUTS+shard_map**: `lax.while_loop` (NUTS tree expansion) has no replication rule in shard_map; `check_rep=False` disables the check while `out_specs=P("chains")` still enforces output sharding.
  - **[30-03] vmap inside shard body for n_chains > n_devices**: When local_n > 1 (n_chains/n_devices > 1), shard body receives `(local_n, ...)` tensors; `jax.vmap` applies per-chain logic independently. Legacy uint32 key shape `(n, 2)` makes this pattern mandatory.
  - **[30-03] MitigationConfig.use_shard_map reserved for Phase 31**: Multi-device decision stays automatic (`n_devices >= n_chains`); Phase 31 will consume the flag for forced-sharding with vmap fallback.
  - **[30-04] GUARD-03 subprocess isolation**: JAX_LOG_COMPILES=1 + subprocess.run provides clean JIT compile-count baseline; in-process redirect_stderr is contaminated by session-level JIT state. Compile threshold of 12 (3x single-iter budget of 4) catches pathological per-iter recompile (20+) without false-positives from scan-body specialisation.
  - **[30-04] All Phase 30 MitigationConfig fields have explicit smoke-test coverage**: non_centered tuple, use_fp64, use_shard_map YAML round-trips plus hash stability test are CI-verified locally and gated for cluster.
  - **[32-01] AUDIT-01 pre-registration gates all Phase 32 work**: `.planning/AUDIT_PROTOCOL.md` committed before any code or run; locks hyperparameters (target_accept=0.95, n_warmup=1000, n_draws=2000, n_chains=4, max_tree_depth=10), cohort grid (26 SLURM tasks), metrics, and decision rules.
  - **[32-02] NumPyro config equivalence closed**: max_tree_depth now passed to NumPyro NUTS; extra_fields=(num_steps, mean_accept_prob) requested in mcmc.run(). AUDIT-02 logp parity test guards against future prior drift.
  - **[33-01] Mode B uses Normal (not TruncatedNormal) participant priors**: LocScaleReparam requires real support; hyperprior centering (mu_omega2 ~ Normal(-3, 1)) provides soft constraint.
  - **[33-01] Shared sigma_p per Boehm 2018**: Not per-group. Phase 33-05 simulation validates recovery; per-group sigma is COVAR-EXT-03.
  - **[33-01] Log-space sigma with Jacobian in BlackJAX closure**: log_sigma_* keys keep NUTS unconstrained; exp-transform + log-abs-det-Jacobian standard pattern.
  - **[33-01] jax.scipy.stats in Mode B closure (not numpyro)**: Pure JAX log_prob avoids numpyro import inside JIT-traced closure.
  - **[33-04] Collinearity threshold 0.7**: |r|=0.7 implies R^2=0.49 shared variance between covariate and group; beta_p and mu_g become practically unidentifiable. Group-mean-centering is default remediation.
  - **[34-01] Two-level seed strategy for Mode B**: cohort_seed = seed_base + n_idx*1000 + model_idx*100 is identical across mitigation combos; mcmc_seed = seed_base + cell_id varies per cell. Pitfall 2 prevention.
  - **[34-01] No fp64 in Mode B grid**: Mode B uses non-centering (M2) for numerical stability; fp64 was Mode A-specific.
  - **[34-01] test_mode_b_minimum_cells deferred with skipif**: activated in Phase 34-03 after grid sweep populates rows.

### Pending Todos

None tracked in `.planning/todos/pending/`.

### Cluster Benchmark DEBT (deferred >1h jobs)

| ID | Phase | Description | Command |
|----|-------|-------------|---------|
| DEBT-01 | 31-02 | Mode A grid resubmit — 48 cells (9 stale cuSPARSE, 32 NOT_RUN). Cuda13 env now fixed. | `sbatch cluster/31_grid_sweep_gpu.slurm` |
| DEBT-02 | 32-04 | Sampler audit head-to-head runs — 26 cells (BlackJAX vs NumPyro). Zero results exist. | `sbatch cluster/32_sampler_audit_gpu.slurm` |
| DEBT-03 | 34-02 | Mode B full grid sweep — 24 cells, production draws. Smoke test passed (4 cells, tiny draws). | `sbatch cluster/34_grid_sweep_gpu.slurm` |

**Blocking dependencies:**
- DEBT-01 → Phase 31-03 full completion (remove xfails on MODEA-08, coverage tests)
- DEBT-02 → Phase 32-05 (sampler recommendation from audit data)
- DEBT-03 → Phase 34-03 (Phase 35 trigger decision + MODEB-08 sigma decision)
- Phase 34-03 → Phase 35 (conditional Gibbs sharding, fire/defer)

### Blockers/Concerns

- **pyhgf 0.2.8 ↔ JAX > 0.4.31 compat** — RESOLVED by 27-02. Strategy C: pip wheel + in-place patch via `scripts/ci/patch_pyhgf_typing.py`. Probe confirmed PASS at JAX 0.10.
- **DEPS-05 cliff not cleared by BlackJAX 1.5 alone** — RESOLVED-AS-NOT-CLEARED by 27-03 (TIMEOUT @ 24h). Phase 29 (M1 wiring + pre-flight) and Phase 30 (Laplace warmup) inherit; cliff mitigation is the entire purpose of those phases.
- **cuSPARSE GPU lottery** — RESOLVED 2026-05-18. Switched `jax[cuda12]` → `jax[cuda13]` in ds_env_v10. GPU works on all M3 nodes. V100 (SM 7.0) dropped. Phase 31 cells 0-8 (pre-fix crashes) need resubmission.
- **Per-parameter vs shared σ_θ identifiability at P=200 with K=2-3 covariates** — RESOLVED in 33-RESEARCH: shared sigma_p per Boehm 2018; per-group sigma is future COVAR-EXT-03 extension, not Phase 33 blocker.
- **M3 conditional-independence proof + per-block convergence diagnostics** — research-flagged for Phase 35 pre-planning if it fires; lowest-confidence area in v1.0 scope.

### v1.0 Carry-forward (from pre-v1.0 work)

- BlackJAX `max_num_doublings` duplicate-kwarg bug fixed in `b737fe6` (2026-05-04). Capability-map's "P=50 was last PASS" reflects this, not a hardware cliff.
- 3-level NUTS conditioning cliff is real even at n_per_group=5 (P=30): step-size collapses to 7.245e-10, depth-10 saturated, on both A100 80GB and L40S. Diagonal mass matrix can't precondition the κ × ω₂ × ω₃ banana. This is what the v1.0 mitigation ladder addresses. Phase 27 confirms BlackJAX 1.5 dense alone is also insufficient.
- Laplace warmup variant (commit `385e8c3`, Phase 14.2-05) is wired but never tested at scale due to the kwarg bug. First v1.0 phase to exercise it: Phase 30.
- `tests/integration/test_capability_map.py` closure-guard validates the map is well-formed. Don't break it; Phase 31 extends it.

## Session Continuity

Last session: 2026-05-21T04:00:00Z
Stopped at: Overnight autonomous code sweep — Phases 36-01, 36-02, 31-03, 36-03, 32-04 partial, 34-02 partial
Resume: Submit cluster DEBT (DEBT-01 through DEBT-03), then execute blocked plans (31-03 full, 32-05, 34-03)

---
*Last updated: 2026-05-21 after overnight autonomous execution*
