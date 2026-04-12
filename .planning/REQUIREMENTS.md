# Requirements: v1.2 Hierarchical GPU Fitting

**Defined:** 2026-04-11
**Core Value:** Refactor v1.1's per-participant sequential fitting into a batched hierarchical architecture so GPU acceleration is actually usable, and finish the v1.1 production run on real compute.

**Motivation:** v1.1 benchmark on L40S showed ~1.5s per NUTS sample (vs ~5ms on CPU) because each sample triggers a CPU↔GPU dispatch for a tiny 420-trial sequential scan. Projected total cost: ~18,000 GPU-hours. Infeasible. The fix is architectural: batch all participants through one vmapped logp so the 5000 NUTS launches per fit are amortized across the whole cohort.

## v1.2 Requirements

### Batched Hierarchical Fitting (BATCH)

- [ ] **BATCH-01**: Build a hierarchical PyMC model with `shape=(n_participants,)` parameters (omega_2, beta, zeta for 2-level; omega_2, omega_3, kappa, beta, zeta for 3-level). Independent priors per participant — no hyperpriors. This preserves v1.1 statistical semantics while exposing the participant dimension to numpyro's vmap.
- [ ] **BATCH-02**: New factory `build_logp_ops_batched(input_data_arr, observed_arr, choices_arr)` where arrays have a leading participant dimension (shape `(P, n_trials, 3)`). Returns a PyTensor Op whose `perform` and JAX dispatch accept batched parameters and return a scalar sum-over-participants log-likelihood.
- [ ] **BATCH-03**: Batched logp implemented via `jax.vmap` over the participant dimension, calling pyhgf's `scan_fn` inside the per-participant branch. Preserves the two-Op split (logp Op + grad Op) so PyMC's gradient machinery works unchanged.
- [ ] **BATCH-04**: tapas-style Layer 2 per-trial clamping inside the `lax.scan` step. On each update, check `jnp.isfinite` on all belief states; if unstable, `jnp.where` reverts to the previous trial's state and that trial contributes 0 to the logp (via a per-trial mask). Must be implemented with pure JAX ops (no Python `if`) to stay inside `jit`.
- [ ] **BATCH-05**: New orchestrator `fit_batch_hierarchical(sim_df, model_name, n_chains, n_draws, n_tune, sampler)` that builds the model, runs **one** `pmjax.sample_numpyro_nuts` call for the full cohort, and returns a single `InferenceData` with a participant dimension on every parameter. Replaces the current loop of per-participant `fit_participant` calls.
- [ ] **BATCH-06**: Preserve the `-jnp.inf` logp sentinel for completely broken parameter combinations (Layer 3 fallback, already present in `ops.py`). With Layer 2 clamping, this should rarely fire.
- [ ] **BATCH-07**: Padding-ready shape — accept an optional `trial_mask` argument (even if currently all-ones) so future variable-length cohorts can reuse the same compiled kernel without triggering XLA recompilation. Lesson from rlwm's fixed-shape pattern.

### JAX-Native Cohort Simulation (JSIM)

- [ ] **JSIM-01**: New `simulate_session_jax(params, trial_inputs, rng_key)` function that runs one full 420-trial session via `lax.scan`, using pyhgf's `net.scan_fn` for the HGF update (no reimplementation of HGF math — pyhgf remains the source of truth). All sampling via `jax.random` (categorical for choice, bernoulli for reward).
- [ ] **JSIM-02**: Same tapas-style Layer 2 clamping as BATCH-04 — on NaN belief, revert to the previous trial state and continue. The `diverged` flag is a `jnp.any` reduction over per-trial clamping events.
- [ ] **JSIM-03**: New `simulate_cohort_jax(params_batch, trial_inputs, rng_keys_batch)` that `jax.vmap`s `simulate_session_jax` across participants, running an entire cohort (up to 300 participant-sessions) in one compiled kernel.
- [ ] **JSIM-04**: `simulate_batch` updated to use the new path internally; produces the same DataFrame schema as before so `run_sbf_iteration` and all downstream code is unchanged.
- [ ] **JSIM-05**: Per-session `diverged` flag propagated through to the output DataFrame as a `diverged` column (matches current simulate_agent behavior after the NaN fix).
- [ ] **JSIM-06**: RNG key threading via `jax.random.split` ensures determinism — same master seed reproduces the same cohort, even across devices.

### CPU Validation Harness (VALID)

- [ ] **VALID-01**: Bit-exact numerical equivalence test — batched logp with `n_participants=1` returns the same float64 value as the existing per-participant `ops.py` logp for matched data and parameters. Any deviation is a bug.
- [ ] **VALID-02**: Small-batch statistical equivalence — fit 5 participants sequentially via the legacy path and 5 participants batched via the new path on CPU with matched seeds. Posterior means agree within 3× MCSE per parameter.
- [ ] **VALID-03**: Cross-platform consistency — run the same small fit on CPU (`ds_env`) and GPU (a single `srun` allocation), verify posterior means agree within 1% relative error. NUTS is stochastic so exact match is not expected.
- [ ] **VALID-04**: Simulation equivalence test — `simulate_agent` (NumPy loop, old path) and `simulate_session_jax` (new path) produce choice/reward distributions that agree on aggregate statistics (mean choice frequency per cue per phase) over 100 replicates with matched master seeds. Exact per-trial match is not required because RNG ordering changes.
- [ ] **VALID-05**: Legacy path preservation — the existing per-participant sequential fitting path (`fit_participant`, `run_power_iteration`) stays runnable via `--legacy` flag on `08_run_power_iteration.py`, so v1.1-era reproducibility is preserved for debugging.

### GPU Benchmark + SBF Integration (BENCH)

- [ ] **BENCH-01**: New benchmark mode fits one full iteration (300 participant-sessions × 2 models) via the batched path on a GPU node and reports per-iteration wall-clock time, VRAM peak, and GPU utilization. Writes `results/power/benchmark_batched.json`.
- [ ] **BENCH-02**: Decision gate — if `(per_iter_seconds × 600 / 3600) > 50` GPU-hours per chunk, recommend falling back to the CPU `comp` partition for production. Otherwise commit to GPU. Record decision in `results/power/benchmark_batched.json` and update STATE.md.
- [ ] **BENCH-03**: `run_sbf_iteration` updated to use `fit_batch_hierarchical` + `simulate_cohort_jax` by default. Old sequential path preserved behind `--legacy` flag (see VALID-05).
- [ ] **BENCH-04**: Entry point `08_run_power_iteration.py --benchmark` uses the new path and reports GPU utilization via periodic `nvidia-smi` sampling during the fit (inside the Python process, to detect idle GPU cycles).
- [ ] **BENCH-05**: JAX compilation cache hit verified across chunks — Chunk 0 compiles cold, Chunks 1 and 2 should start fitting within ~5 seconds (vs ~60s cold). Report in benchmark output.

### Production Run + Milestone Delivery (PROD)

- [ ] **PROD-01**: Full power sweep (600 SBF tasks across 3 chunks) executes successfully on the chosen platform (GPU or CPU comp, per BENCH-02).
- [ ] **PROD-02**: All 600 tasks complete; per-chunk parquet files aggregate cleanly into `results/power/power_master.csv` with no missing grid cells (`scripts/09_aggregate_power.py` runs without warnings).
- [ ] **PROD-03**: 4-panel publication figure (`scripts/10_plot_power_curves.py`) regenerated with real data, saved as both PDF and PNG. No placeholder values.
- [ ] **PROD-04**: `results/power/recommendation.md` (`scripts/11_write_recommendation.py`) contains a concrete N/group and trial count recommendation backed by real BF evidence, exclusion rates, and the omega_3 upper-bound caveat.
- [ ] **PROD-05**: Wave 3 auto-push wired into the production pipeline (`99_push_results.slurm` triggered via `afterany` from the post-processing job), so results land in git without manual intervention. Benchmark mode still skips push (unchanged).

## v2 Requirements (Deferred)

- **V2-HIER-01**: True hierarchical model with population-level hyperpriors (partial pooling) — would improve recovery for participants with poor fits but changes the statistical model and needs its own validation.
- **V2-HIER-02**: Per-participant step-size adaptation in NUTS via numpyro's `HMCGibbs` or custom kernel — the default batched NUTS adapts a single step size across all participants, which is a compromise.
- **V2-MLE-01**: MAP/MLE via `jaxopt.LBFGS` as a fast screening pass (rlwm pattern), with NUTS only for cases where MAP is ambiguous. ~50-100x speedup potential.
- **V2-HIER-03**: Multi-GPU `pmap` across chains if > 1 GPU is requested (currently `numpyro.set_host_device_count(1)`, sequential chains).

## Out of Scope

| Feature | Reason |
|---------|--------|
| Reimplementing HGF math in pure JAX | pyhgf's `scan_fn` is the source of truth; wrapping it in our own `lax.scan` gets the same benefit without maintenance burden |
| Changing statistical priors or contrasts | Scope creep — v1.2 is a compute/architecture refactor, not a model change |
| Multi-device `pmap` across GPUs | SLURM allocates 1 GPU per task in our setup; multi-GPU is v2 |
| GUI for power analysis | Batch HPC workflow, not interactive |
| Hierarchical hyperpriors (partial pooling) | Changes statistical semantics vs v1.1 — defer to v2 and validate separately |
| Real data ingestion | v2.0 milestone, unrelated to compute architecture |

## Success Criteria

1. Batched hierarchical logp passes bit-exact CPU test (VALID-01)
2. Small-batch sequential-vs-batched fit agrees within MCSE (VALID-02)
3. GPU benchmark completes and produces a feasibility decision (BENCH-01, BENCH-02)
4. Full power sweep runs on the chosen platform within a reasonable walltime budget (PROD-01)
5. v1.1 deliverables (`power_master.csv`, 4-panel figure, `recommendation.md`) populated with real data (PROD-02, PROD-03, PROD-04)
6. Legacy path still works for reproducibility (VALID-05)

## Key References

- rlwm_trauma_analysis `scripts/fitting/jax_likelihoods.py` — padding/masking pattern, stacked arrays
- rlwm_trauma_analysis `validation/diagnose_gpu.py` — vmap-vs-sequential benchmark (7-13x slower for their LBFGS case; different regime from our NUTS case)
- `project_utils/templates/guides/JAX_GPU_BAYESIAN_FITTING.md` — full writeup of lessons and patterns
- Current `src/prl_hgf/fitting/ops.py` — existing JAX Op with `lax.scan` over 420 trials, to be generalized to batched
- tapas HGF toolbox (MATLAB) `tapas_ehgf_binary.m` — Layer 2 per-trial state clamping pattern we are porting to JAX
- Schönbrodt & Wagenmakers (2018) — SBF design (v1.1 inherits; v1.2 preserves)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BATCH-01 | Phase 12 | Complete |
| BATCH-02 | Phase 12 | Complete |
| BATCH-03 | Phase 12 | Complete |
| BATCH-04 | Phase 12 | Complete |
| BATCH-05 | Phase 12 | Complete |
| BATCH-06 | Phase 12 | Complete |
| BATCH-07 | Phase 12 | Complete |
| VALID-01 | Phase 12 | Complete |
| VALID-02 | Phase 12 | Complete |
| JSIM-01 | Phase 13 | Complete |
| JSIM-02 | Phase 13 | Complete |
| JSIM-03 | Phase 13 | Complete |
| JSIM-04 | Phase 13 | Complete |
| JSIM-05 | Phase 13 | Complete |
| JSIM-06 | Phase 13 | Complete |
| VALID-04 | Phase 13 | Complete |
| BENCH-01 | Phase 14 | Pending |
| BENCH-02 | Phase 14 | Pending |
| BENCH-03 | Phase 14 | Pending |
| BENCH-04 | Phase 14 | Pending |
| BENCH-05 | Phase 14 | Pending |
| VALID-03 | Phase 14 | Pending |
| VALID-05 | Phase 14 | Pending |
| PROD-01 | Phase 15 | Pending |
| PROD-02 | Phase 15 | Pending |
| PROD-03 | Phase 15 | Pending |
| PROD-04 | Phase 15 | Pending |
| PROD-05 | Phase 15 | Pending |

**Coverage:** 28/28 v1.2 requirements mapped across 4 phases (12-15).

---
*Requirements defined: 2026-04-11*
*Traceability updated: 2026-04-11 (roadmap created)*
