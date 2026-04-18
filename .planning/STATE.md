# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-07)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Phase 19 VB-Laplace Fit Path (Tapas-Parity Validation) — Plan 2/5 in progress

## Current Position

Phase: 19 of 19 (VB-Laplace Fit Path)
Plan: 2/5 complete (19-02 Laplace InferenceData factory)
Status: In progress — Phase 19 Plans 1-2 complete (19-01 pat_rl_simulator, 19-02 laplace_idata); Plans 3-5 pending
Last activity: 2026-04-18 — Completed 19-02: build_idata_from_laplace with NUTS-parity schema; 11 tests; export_subject_parameters consumer contract verified

[===========█████████████]   v1.1 code-complete (Phases 1-11); Phases 12-14 verified; Phase 16 complete; Phase 17 complete; Phase 18 complete (6/6); Phase 19 in progress (2/5)

## Performance Metrics

| Metric | v1.0 Value |
|--------|------------|
| Phases shipped | 7 |
| Plans completed | 18 |
| Files created/modified | 105 |
| Lines of Python | ~11,016 |
| Days elapsed | 4 (2026-04-04 to 2026-04-07) |

## Accumulated Context

### Key Decisions

See `.planning/milestones/v1.0-ROADMAP.md` for v1.0 decision log.

| Decision | Rationale | Phase |
|----------|-----------|-------|
| MCMC throughout (no MAP proxy) | Simplicity; leverage cluster parallelism for power loop | v1.1 planning |
| Psilocybin vs placebo groups (both post-concussion) | Corrected study design | v1.1 planning |
| pingouin.bayesfactor_ttest over rpy2+anovaBF | pingouin already installed; anovaBF misspecified for RM designs (van den Bergh 2023) | v1.1 roadmap |
| SLURM array %50 throttle | Prevents Lustre metadata import storm on M3 MASSIVE | v1.1 roadmap |
| omega_3 BFDA labeled "exploratory — upper bound" | Recovery r ~ 0.67 with binary data; naive BFDA inflates power 20-40pp | v1.1 roadmap |
| SeedSequence for parallel RNG | task-ID integer seeding gives correlated streams; SeedSequence guarantees independence | v1.1 roadmap |
| power/ package wraps existing pipeline; no existing module modified | Eliminates regression risk; all existing functions called unchanged | v1.1 roadmap |
| make_power_config shifts psilocybin omega_2_deltas only; placebo unchanged | Study hypothesis: psilocybin increases learning rate; placebo is inert control | 08-01 |
| load_power_config reads only power: YAML key; does not re-parse task/simulation/fitting | Clean separation of concerns; existing load_config unaffected | 08-01 |
| write_parquet_row rejects missing AND extra columns | Strict schema enforcement prevents silent drift as Phase 10 adds real pipeline results | 08-02 |
| --output-dir flag on entry point for test isolation | Integration tests write to tmp_path, not results/; keeps test suite clean | 08-02 |
| make_trial_config scales only PhaseConfig.n_trials; TransferConfig.n_trials untouched | Transfer trials are fixed study design; only acquisition/reversal phases vary for trial count studies | 09-01 |
| run_recovery_precheck filters to baseline only before fit_batch | Avoids 3x compute cost; post_dose and followup sessions not needed for recovery gate | 09-01 |
| omega_3 eligibility always "exploratory -- upper bound" regardless of r | Locked decision: binary PRL data r~0.67 in literature; BFDA inflation 20-40pp | 09-01 |
| n_flagged uses groupby("participant_id")["flagged"].any().sum() | fit_df has per-parameter rows; participant is flagged if ANY parameter row is flagged | 09-01 |
| run_trial_sweep passes min_n=0 to build_recovery_df | Small trial counts may lose many participants to convergence failures; downstream callers apply own filter | 09-02 |
| find_minimum_trial_count excludes omega_3 from all-must-pass by default | Consistent with locked exploratory decision; omega_3 never gates the trial count requirement | 09-02 |
| seed+idx seeding per grid point in run_trial_sweep | Fresh independent participants at each trial count; no between-condition correlation in recovery estimates | 09-02 |
| compute_power_b deduplicates on (n_per_group, iteration) before mean | Each SLURM task writes 3 parquet rows (one per sweep_type) with identical BMS values; dedup prevents triple-counting | 11-01 |
| bf_threshold parameter on compute_power_a is documentary; uses pre-computed bf_exceeds bool column directly | No re-thresholding at aggregation time — threshold is baked into schema at write time | 11-01 |
| plot_power_a recomputes from raw bf_value (not bf_exceeds) so --bf-threshold CLI arg is live | Enables users to replot at different thresholds without re-running the sweep | 11-02 |
| plot_combined_figure saves both PDF and PNG from same stem | Grant submission needs PDF; PNG for quick inspection | 11-02 |
| _draw_*_panel helpers separate subplot rendering from file I/O | Enables test isolation and reuse in combined figure without recursive file saves | 11-02 |
| generate_recommendation re-applies bf_threshold to bf_value for N selection | Allows any --bf-threshold CLI arg to affect recommended N, not just pre-baked bf_exceeds bool | 11-03 |
| power_a_df and power_b_df recomputed from master if summary CSVs missing | Makes recommendation script self-sufficient when only power_master.csv exists | 11-03 |
| Chunk-based SLURM: 3 jobs instead of 4200 | JAX compiles once per chunk; reuses compiled model for ~1400 iterations; one combined parquet per chunk | 11 |
| legacy/batch.py imports from legacy/single.py (not shim) | Ensures frozen code calls frozen code; no circular dependency through shims | 12-01 |
| Shims use noqa: F401 for re-exports | Ruff would flag unused imports in shim modules; F401 suppression is standard for re-export patterns | 12-01 |
| Data as runtime args for vmap (not closure-over-data) | Clean vmap signature; XLA sees full data flow; no closure recreation on data change | 12-02 |
| Separate named functions for 2-level/3-level logp | Avoids mypy error from conditional redefinition with different signatures | 12-02 |
| Level-2 mean key is attrs[i]["mean"] | Confirmed via runtime inspection of pyhgf attribute pytree structure | 12-02 |
| Post-hoc dim rename for participant coords (not pm.Model coords=) | dims= parameter interacts unpredictably with pm.Potential in some PyMC versions; post-hoc rename is robust | 12-03 |
| TYPE_CHECKING guard for pd/az imports | Avoids heavy runtime imports; satisfies ruff F821 + UP037 for forward-ref type annotations | 12-03 |
| Both VALID-02 paths use numpyro sampler | PyMC sampler hits _init_jitter read-only array bug with JAX-backed Ops; numpyro bypasses PyTensor entirely | 12-04 |
| Simulated agent data for VALID-02 (not random) | Random choice/reward data produces poorly constrained posteriors, making MCSE-based comparison unreliable | 12-04 |
| Positional dim indexing for batched posterior | PyMC assigns per-variable dim names when shape= used without dims=; positional indexing is robust | 12-04 |
| Factory pattern for jax_session vmappability | _build_session_scanner builds pyhgf Network once outside JAX trace; _run_session is pure-JAX and vmappable for Plan 02 cohort path | 13-01 |
| jnp.int32(-1) sentinel for trial-0 stickiness | (prev_choice == jnp.arange(3)) evaluates all-False for -1 giving zero stickiness; verified in test_session_jax_stickiness_sentinel | 13-01 |
| values_t elements use .reshape(1) in sim path | Matches pyhgf scan_fn shape contract: logp path uses input_data[:, 0:1] sliced to (1,) per step; simulation must match exactly | 13-01 |
| simulate_cohort_jax captures shared cue_probs_arr in lambda closure | Shared trial sequence across participants; closure keeps vmap axes to 6 scalar params + 1 key; cleaner than in_axes=None | 13-02 |
| simulate_batch stacks per-participant cue_probs as axis=0 | Each session has distinct env_seed → different trial sequence; (P, n_trials, 3) stack with in_axes=0 handles variation across participants | 13-02 |
| Two-phase batch: Python collection then single vmap dispatch | Preserves deterministic seed derivation; eliminates _prewarm_jit; DataFrame assembly stays in Python after compiled kernel runs | 13-02 |
| Single fixed trial sequence for VALID-04 controlled-comparison design | Isolates agent RNG from environment RNG; holding env constant makes KS test sensitive only to JAX vs NumPy behavioral difference | 13-03 |
| KS test on choice frequency distributions (not per-trial match) for VALID-04 | NumPy PCG64 and JAX ThreeFry are incompatible RNG streams; per-trial exact match impossible; aggregate distributional equivalence is the correct scientific test | 13-03 |
| VALID-03 is two separate script invocations + JSON comparison (not a single pytest) | JAX platform (CPU/GPU) is set at import time; cannot be changed within a running process | 14-03 |
| compare_results denominator uses abs(mean_a) + 1e-8 | Prevents near-zero division false failures when parameter means (e.g. zeta) are ~ 0.001 | 14-03 |
| Deferred import of fit_batch_hierarchical inside else block of run_sbf_iteration | Keeps heavy JAX/PyMC/pyhgf imports out of import-time when using legacy path; consistent with existing deferred-import pattern | 14-01 |
| fit_df_2 not constructed in batched path of run_sbf_iteration | SBF subsampling loop only uses fit_df_3 for BF contrasts and diagnostics; same is true in legacy path — no structural need to construct it | 14-01 |
| az.rhat(da)[param].values for scalar extraction from az.rhat Dataset | az.rhat(DataArray) returns a Dataset not a scalar; [param].values extracts the 0-d NumPy scalar correctly | 14-01 |
| strict=True on all zip() calls in _idata_to_fit_df and _build_idata_dict | Catches participant metadata length misalignment at helper boundaries before silent posterior-to-participant mapping errors | 14-01 |
| apply_decision_gate is pure function in iteration.py (not script-local) | Testable independently of benchmark script; importable by any caller needing gate logic | 14-02 |
| _update_state_md uses string search + last-| -line insertion (not regex) | Simple and robust for the fixed table structure; handles missing STATE.md gracefully | 14-02 |
| BLE001 noqa on broad except in _GpuMonitor._run | Intentional swallow for nvidia-smi transient failures (missing binary, timeout, parse error) | 14-02 |
| Additive refactor: all PyTensor Op code kept for VALID-01/02 backward compat | Deprecated functions still importable; no test breakage risk | 16-01 |
| chain_method="vectorized" for numpyro MCMC | Single kernel for all 4 chains on one GPU; better throughput than sequential+jit_model_args | 16-01 |
| numpyro.sample() + numpyro.factor() pattern over raw potential_fn | Preserves named parameters for az.from_numpyro(); avoids manual Param:0 renaming | 16-01 |
| Data passed as kwargs to mcmc.run(), not captured in closures | XLA trace is shape-dependent but value-independent; enables JIT cache reuse across power-sweep iterations | 16-01 |
| sampler="pymc" raises DeprecationWarning; numpyro path always used | API backward compat preserved; old callers get warning but still work | 16-01 |
| sampler= kwarg removed from batched fit_batch_hierarchical calls; kept in signatures for backward compat | Batched path always uses numpyro-direct; no need to forward sampler to fit function | 16-02 |
| check_cuda_compat is non-fatal in SLURM scripts | MCMC still works without XLA parallel compilation, just slower; no reason to abort the job | 16-02 |
| GPU pip deps tracked in cluster/requirements-gpu.txt (separate from main requirements) | Cluster-specific CUDA pins should not pollute the main dev environment | 16-02 |
| BlackJAX as default sampler (sampler="blackjax"); NumPyro preserved as fallback | Eliminates ~1800s per-call JIT recompilation; BlackJAX compiles NUTS step once via jax.jit | 17-01 |
| Single warmup replicated across chains (not per-chain warmup) | Simpler implementation; posterior geometry similar across chains with IID priors | 17-01 |
| numpyro.distributions for standalone prior log_prob | Pure JAX, no model context; matches existing prior specs exactly; avoids jax.scipy.stats standardized-bounds pitfall | 17-01 |
| pmap when device_count >= n_chains; vmap fallback on single device | Multi-GPU utilization when available; no overhead from pmap on single device | 17-01 |
| sampler="pymc" deprecation falls through to numpyro (not blackjax) | PyMC users expect NumPyro-style behavior; blackjax path is new and should be explicitly chosen | 17-01 |
| Separate 2-level/3-level log-posterior smoke tests (not parameterized) | Clearer failure diagnostics when one model variant fails | 17-02 |
| SLURM default SAMPLER env var changed from numpyro to blackjax | Matches fit_batch_hierarchical default sampler; override with SAMPLER=numpyro if needed | 17-02 |
| JIT gate thresholds preserved at NumPyro levels with annotation | Thresholds conservative for BlackJAX; will tighten after cluster benchmarking | 17-02 |
| _build_sample_loop factory passes data as traced JIT args (not closure) | XLA persistent cache keys on HLO hash; closure data = HLO constants = cache miss; traced args = shape placeholders = cache hit | quick-003 |
| vmap path @jax.jit, pmap path lets pmap handle compilation | pmap inside JIT boundary is problematic; factory returns different function variants | quick-003 |
| Legacy fallback when batched_logp_fn is None | Backward compat for callers not providing traced-arg data | quick-003 |
| PAT-RL uses fully parallel loader (pat_rl_config.py) with zero imports from task_config.py | task_config.py has 21 callsites and TaskConfig.__post_init__ would reject PAT-RL structure; parallel stack keeps pick_best_cue tests isolated | 18-01 |
| env/__init__.py deliberately not updated with PAT-RL exports | Adding exports risks side-effects on pick_best_cue imports; PAT-RL callers use direct module import | 18-01 |
| SeedSequence 4-way spawn for PAT-RL trial generator (state/mag/dHR/reserved) | Independent child streams for each RNG role; reserved 4th stream lets Plan 18-04 add outcome draws without changing existing seeds | 18-02 |
| State carried forward across run boundaries in generate_session_patrl | Biological realism: context state is continuous; artificial reset at run boundaries would bias run-1 outcomes | 18-02 |
| PhenotypeParams uses PriorGaussian for kappa/mu3_0 (sd=0 allowed) | Avoids over-engineering separate FixedParam type before Models B/C/D clarify param variation needs | 18-01 |
| PAT-RL Model A EV direction: sigmoid(mu2)=P(dangerous); EV_approach=(1-P_danger)*V_rew - P_danger*V_shk | Consistent with state=1 meaning dangerous; at mu2=0,V_shk>>V_rew: P(approach)<<0.01 (validated in test 7) | 18-03 |
| pyhgf 0.2.8 kappa coupling via volatility_children=([child],[kappa]) not node_parameters | node_parameters dict does not accept coupling strength keys in pyhgf 0.2.8; tuple-of-lists is correct API | 18-03 |
| pyhgf 0.2.8 time_steps must be 1D np.ones(n_trials) not 2D matrix | 2D time_steps causes JAX carry-type shape mismatch in scan_fn | 18-03 |
| model_a_logp MU2_CLIP=30 (outer envelope); HGF-level clamping at |mu2|<14 (inner) | Conservative outer clip keeps sigmoid finite at export time; inner clamping handles fitting instability | 18-03 |
| float64 scan inputs in hierarchical_patrl._single_logp | pyhgf attrs are float64; jax.lax.cond in continuous_node_posterior_update requires dtype consistency between branches; float32 inputs cause TypeError when vmap'd | 18-04 |
| PAT-RL closure-based logdensity_fn (not traced-arg sample loop) | _build_sample_loop hardcodes pick_best_cue 7-arg logp signature; re-use would require modifying hierarchical.py (parallel-stack violation); closure path is correct for Phase 18 smoke | 18-04 |
| kappa injected via attrs[2]["volatility_coupling_children"] = jnp.asarray([kappa_i]) | Confirmed at runtime: kappa coupling strength is stored in attributes dict (not only edges), enabling per-participant dynamic injection inside lax.scan | 18-04 |
| log_beta parameterisation: beta sampled in log-space; prior N(log(beta_mean), beta_sd/beta_mean) | NUTS can freely explore without positivity boundary; delta-method approximation centres prior near config beta.mean | 18-04 |
| dcm_pytorch bilinear B-matrix path is LIVE in v0.3.0 (not deferred): parameterize_B + compute_effective_A in neural_state.py | Plan sketch was wrong; read neural_state.py:4-15 + coupled_system.py:20-73 in Task 1 audit | 18-05 |
| Modulator channel values for dcm_pytorch: raw float64, no bounding or normalization | dcm_pytorch neural_state.py:104-113: off-diagonal pass through via pure mask; N(0,1) prior does regularization | 18-05 |
| outcome_time_s feeds stimulus["times"] directly: both are absolute seconds from session start | Confirmed in task_simulator.py:74, ode_integrator.py:35-39; no time-axis transform needed | 18-05 |
| az.hdi returns "lower"/"higher" coordinate labels (not "low"/"high") in ArviZ 0.22+ | Verified at runtime; pivot in export_subject_parameters uses hdi="lower" and hdi="higher" selectors | 18-05 |
| pyhgf 0.2.8 temp keys confirmed: value_prediction_error, effective_precision, volatility_prediction_error all present | Runtime inspection in test_pyhgf_temp_keys_extracted canary test | 18-05 |
| 3-level-only cols (mu3/sigma3/epsilon3) present as NaN in 2-level trajectory CSV | Schema consistency: downstream concat/join across model variants works without column-presence check | 18-05 |
| _samples_to_idata() coord_name kwarg (default "participant"; PAT-RL passes "participant_id") | PRL pipeline and PAT-RL exporter (Plan 18-05) disagree on coord label; kwarg keeps PRL back-compat while making PAT-RL's contract with export_subject_trajectories explicit | 18-06 |
| Smoke script exit codes: 0 success / 1 runtime error / 2 blackjax missing | 3-state map lets CI/docs distinguish "real bug" from "environment not provisioned" without parsing stderr | 18-06 |
| Smoke pytest: 7 structural tests unconditional; end-to-end fit path exercised only by cluster SLURM | blackjax not in dev env; structural tests (compile, argparse, import-scan, lazy-blackjax) give meaningful local signal without MCMC cost | 18-06 |
| Cluster smoke runs CPU comp partition (not GPU) | Fit wall-clock 11.8s single-thread; GPU dispatch overhead dominates at this batch size; no justification for GPU hours | 18-06 |
| build_idata_from_laplace emits dim 'participant_id' natively (not 'participant') | NUTS path _samples_to_idata emits 'participant' (latent OQ1 bug); Laplace path sidesteps it by emitting consumer-correct dim without touching hierarchical.py (parallel-stack invariant) | 19-02 |
| cast(az.InferenceData, az.from_dict(...)) for mypy satisfaction | az.from_dict stub returns Any in arviz typeshed; cast is zero-overhead and makes return type explicit | 19-02 |

### Pending Todos

- manuscript/references.bib: mason2024 volume/page details need verification before submission
- quarto-arxiv extension must be installed before first arxiv-pdf render
- Phase 10 kappa effect size parameterization: verify kappa entry point in GroupConfig vs SessionConfig during Phase 10 planning (kappa delta lives in SessionConfig.kappa_deltas — confirmed in 08-01 tests, but grid parameterization for kappa needs review before Phase 10 sweep)
- Phase 10: run 100-iteration MAP vs NUTS pilot before committing to full NUTS budget

### Blockers/Concerns

- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env
- omega_3 parameter recovery expected to be challenging (known issue in literature)
- **v1.1 per-participant sequential MCMC is GPU-pessimal** — L40S benchmark showed ~1.5s/NUTS-sample vs ~5ms on CPU due to PCIe dispatch overhead. v1.2 refactor is mandatory for GPU feasibility.
- **Decision gate at Phase 14:** if batched hierarchical GPU benchmark is still > 50 GPU-hours per chunk, fall back to CPU `comp` partition (new batched code still wins over v1.1 sequential on CPU).
- pyhgf has no built-in NaN clamping — **RESOLVED in 12-02**: Layer 2 clamping implemented in hierarchical.py using jnp.where + tree_map (|mu_2| < 14 bound).
- `_init_jitter` PyTensor read-only-array bug means we can't use `pm.sample(...)` directly even with `nuts_sampler="numpyro"`; must call `pmjax.sample_numpyro_nuts()` directly. **RESOLVED in 16-01**: fit_batch_hierarchical now uses direct numpyro MCMC, bypassing PyMC/PyTensor entirely.
- **blackjax not installed in ds_env** — PAT-RL smoke tests (18-04 tests 5-6) skip via importorskip. Install blackjax on cluster before running smoke validation. Pre-existing: test_valid_02_batched_blackjax_convergence also fails for same reason.
- **VB-Laplace quick-005 decision pending**: quick-004 memo recommends Option C (dual NUTS + Laplace paths). If first `sbatch cluster/18_smoke_patrl_cpu.slurm` blows past 6h or shows >20% divergences on 2-level, downgrade to Option A (Laplace primary). If Laplace unit tests show >2× underestimation of ω₂ posterior width, downgrade to Option B (NUTS only). See `.planning/quick/004-.../VB_LAPLACE_FEASIBILITY.md` §6.

## Quick Tasks

| ID  | Name | Status | Summary |
|-----|------|--------|---------|
| 001 | Cluster GPU Setup & Smoke Test | Complete | M3 SLURM infrastructure + smoke test PASS |
| 002 | HGF Fitting Lessons Obsidian Doc | Complete | Comprehensive coding guide in Obsidian Vault covering math, JAX/NumPyro patterns, cluster pitfalls |
| 003 | JIT Cache: Data as Traced Args | Complete | BlackJAX sampling loop restructured so data arrays flow as traced JIT args for persistent XLA cache hits |
| 004 | PAT-RL Smoke + VB-Laplace Feasibility | Complete | Cluster SLURM for Phase 18 PAT-RL smoke; --dry-run flag; 5 structural tests; 4 scratch files deleted; 6 PLAN.md files tracked; VB-Laplace Option C recommendation | [004-SUMMARY](./quick/004-patrl-smoke-and-vb-laplace-feasibility/004-SUMMARY.md) |

### Roadmap Evolution

- Phase 16 added (2026-04-13): NumPyro direct sampling + CUDA fix — replace PyMC wrapper with direct numpyro MCMC to enable JIT cache reuse; fix CUDA PTX mismatch; add environment diagnostics
- Phase 17 added (2026-04-15): BlackJAX NUTS sampler — replace NumPyro MCMC with BlackJAX to eliminate ~1800s JIT recompilation per call; restore multi-GPU pmap for chain parallelism
- Phase 17-01 complete (2026-04-15): BlackJAX core + orchestrator — _build_log_posterior, _run_blackjax_nuts, _samples_to_idata, fit_batch_hierarchical rewrite
- Phase 17-02 complete (2026-04-15): BlackJAX smoke tests + SLURM updates — 4 new fast tests (logp, gradient, idata), VALID-02 blackjax convergence test, SLURM scripts updated for BlackJAX default
- Phase 18 added (2026-04-17): PAT-RL Task Adaptation (the consumer study) — new binary-state approach/avoid task config, trial generator, response models A-D (including trial-varying omega), trajectory export for DCM bridge, and phenotype-stratified BMS. Source: GSD_prl_hgf.yaml. **Caveat**: Multiple YAML assumptions conflict with repo (config loader is task-specific, no `trial_sequence.py` exists, response model signature differs, Delta-HR plumbing absent, no phenotype framework). Integration notes captured inline in ROADMAP.md Phase 18 entry. Consider promoting to v1.3 the consumer study milestone before planning.
- Phase 19 added (2026-04-18): VB-Laplace Fit Path (Tapas-Parity Validation) — implements `fit_vb_laplace_patrl.py` as a second, non-MCMC fit path alongside BlackJAX NUTS. Mirrors matlab tapas HGF toolbox (quasi-Newton MAP + Laplace covariance from numerical Hessian at the mode). Reuses `_build_patrl_log_posterior` from `hierarchical_patrl.py` without modification. ArviZ `InferenceData` output shape-compatible with the NUTS path so the existing Plan 18-05 exporters accept both. Unblocks PEB development immediately + gives a deterministic reference fit to validate cluster NUTS posteriors against. Source of record: `.planning/quick/004-.../VB_LAPLACE_FEASIBILITY.md` Option C. Explicitly implementation (not feasibility re-exploration).

## Session Continuity

Last session: 2026-04-18
Stopped at: Completed 19-02 — build_idata_from_laplace factory + 11 tests; consumer contract (export_subject_parameters) verified; participant_id dim name native (OQ1 sidestepped).
Resume file: None
Next action: Execute 19-03 (fit_vb_laplace_patrl: MAP optimizer + Hessian computation + fit function that calls build_idata_from_laplace).
