# Audit Protocol: BlackJAX vs NumPyro NUTS (Phase 32)

**Date:** 2026-05-18
**Author:** Phase 32 (Sampler Audit Harness)
**Status:** PRE-REGISTERED

**Purpose:** This protocol MUST be committed before any audit code is written
or any audit run is submitted. It pre-registers the comparison design so that
no post-hoc rationalization of sampler differences is possible. All subsequent
Phase 32 plans (32-02 through 32-05) are gated by this document's existence
in the git history.

---

## 1. Fixed Hyperparameters

All head-to-head audit cells use identical hyperparameters across both
backends. Any deviation from these values invalidates the comparison.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `target_accept` | 0.95 | Project default; matches Phase 31 grid |
| `n_warmup` | 1000 | Standard adaptation budget; no cliff at this count |
| `n_draws` | 2000 | Sufficient for ESS estimation with 4 chains |
| `n_chains` | 4 | Minimum for R-hat computation; standard practice |
| `max_tree_depth` | 10 | Both backends cap tree expansion at 2^10 leapfrog steps |
| `mass_matrix` | diagonal (primary) | Fair comparison; dense is secondary analysis |
| `dtype` | fp32 (primary) | fp64 is BlackJAX-only secondary; excluded from head-to-head |
| `random_seed` | 42 (master) | Per-cell seed = 42 + cell_offset |
| `use_laplace_warmup` | false | Excluded from cross-backend comparison (BlackJAX-only feature) |

**Seed protocol:** Master seed = 42. Per-cell seed = master + cell_offset
where cell_offset is a deterministic function of (model_index, P_index,
backend_index). Same seed across backends ensures reproducibility but does NOT
produce identical chains (key-splitting differs between backends).

---

## 2. Config Equivalence Matrix

This table documents the mapping of every FitConfig knob to each backend's
API, identifies equivalence gaps, and specifies remediation obligations.

| Config knob | BlackJAX API | NumPyro API | Equivalent? | Action needed |
|-------------|--------------|-------------|-------------|---------------|
| `target_accept` | `window_adaptation(target_acceptance_rate=...)` | `NUTS(target_accept_prob=...)` | YES | None |
| `n_warmup` | `window_adaptation(num_steps=...)` | `MCMC(num_warmup=...)` | YES | None |
| `n_draws` | scan loop count | `MCMC(num_samples=...)` | YES | None |
| `n_chains` | vmap/shard chains | `MCMC(num_chains=...)` | YES | None |
| `max_tree_depth` | `blackjax.nuts(max_num_doublings=...)` | **NOT PASSED TO NUTS** | **NO** | Fix in Plan 32-02: add `max_tree_depth=max_tree_depth` to `NUTS(...)` constructor |
| `random_seed` | `jax.random.PRNGKey(random_seed)` | `jax.random.PRNGKey(random_seed)` | YES | None |
| `mass_matrix_kind="diagonal"` | `is_mass_matrix_diagonal=True` | `dense_mass=False` | YES | None |
| `mass_matrix_kind="dense"` | `is_mass_matrix_diagonal=False` | `dense_mass=True` | YES | None |
| `mass_matrix_kind="low_rank"` | no support (falls to dense) | no support | YES (both unsupported) | Document only |
| `use_laplace_warmup=True` | LBFGS warmup bypasses window_adaptation | **NOT AVAILABLE** | **NO** | Exclude from audit grid; BlackJAX-only secondary analysis |
| `use_fp64` | via `set_x64()` before JAX import | same if driver calls it | YES (driver-level) | Exclude from primary audit (BlackJAX-only secondary) |
| `use_shard_map` | auto chain dispatch (shard_map/vmap) | `chain_method="vectorized"` only | **NO** | NumPyro always vectorized; not configurable |
| Logp function | `build_logp_fn_batched` | same function object | YES | Verified by AUDIT-02 |
| Prior construction | `numpyro.distributions.log_prob(...)` | `numpyro.sample(...)` | YES (same dist objects from `HGFPriorSpec`) | Verified by AUDIT-02 |

### Gap 1: `max_tree_depth` not passed to NumPyro NUTS

**Status:** BLOCKING — must be fixed before any audit run.

**Current state:** The NumPyro path constructs
`NUTS(bound_model, target_accept_prob=..., dense_mass=...)` without passing
`max_tree_depth`. NumPyro's default is 10, which happens to match the audit
protocol's value of 10 — but config equivalence is structurally violated. If
any future run uses `max_tree_depth != 10`, the backends will silently diverge.

**Remediation:** Plan 32-02 must add `max_tree_depth=max_tree_depth` to the
`NUTS(...)` constructor call in `fit_batch_hierarchical` (NumPyro path).

**Verification:** AUDIT-02 logp parity test does not catch this (logp is
independent of tree depth). Structural inspection of the constructor call is
the verification.

### Gap 2: `use_laplace_warmup` not available in NumPyro

**Status:** EXCLUDED from audit grid.

**Current state:** `_laplace_warmup_params` feeds into `_run_blackjax_nuts` via
the `warmup_params` hook. The NumPyro path has no equivalent mechanism
(`numpyro.MCMC` uses internal dual-averaging warmup with no external override).

**Remediation:** Not fixable without modifying NumPyro internals. Excluded from
all cross-backend comparison cells. Treated as a BlackJAX-exclusive feature.
A separate BlackJAX-only analysis may compare diagonal-window vs Laplace-warmup
but this is NOT a backend comparison.

### Gap 3: `extra_fields` not requested in NumPyro `mcmc.run()`

**Status:** BLOCKING for metric parity — must be fixed before any audit run.

**Current state:** `mcmc.run()` is called without `extra_fields`. NumPyro's
`HMC.default_fields = ("z", "diverging")` — only divergence status is collected.
The audit requires `num_steps` (leapfrog count per draw) and
`mean_accept_prob` for ESS-per-grad-eval and acceptance-rate parity reporting.

**Remediation:** Plan 32-02 must add
`extra_fields=("num_steps", "mean_accept_prob")` to the `mcmc.run()` call.

---

## 3. Logp Function Specification

Both backends use the same logp factory and must produce numerically identical
log-density values at identical parameter inputs.

**Shared factory:** `build_logp_fn_batched(model_name, n_trials)` returns
`(batched_logp_fn, n_params)`. This function is called once and the same
function object is passed to both backend paths.

**BlackJAX path:**
- `_build_log_posterior(batched_logp_fn, input_data, observed, choices,
  trial_mask, n_participants, model_name, prior_spec)` returns
  `logdensity_fn: dict -> scalar`
- Prior log-probs computed via `numpyro.distributions` objects from
  `HGFPriorSpec.to_numpyro_dist()`
- Returns: `log_likelihood + sum(log_prior_i)`

**NumPyro path:**
- `_numpyro_model_{2,3}level(input_data, observed, choices, trial_mask, *,
  n_participants, batched_logp_fn, prior_spec)` declares priors via
  `numpyro.sample(...)` and likelihood via `numpyro.factor("hgf_loglike", logp)`
- Same `HGFPriorSpec.to_numpyro_dist()` objects used for prior distributions
- Returns: implicit (NumPyro model trace)

**AUDIT-02 verifies numerical parity:** Using
`numpyro.infer.util.log_density(model, args, kwargs, params)` on the NumPyro
model and direct evaluation of BlackJAX's `logdensity_fn(params)`, both at
identical fixed parameter values. Tolerance: `atol=1e-4, rtol=1e-4` (float32
accumulation differences).

---

## 4. Cohort Grid

### Primary head-to-head grid

| P (participants) | n_per_group | Models | Backends | Fits |
|------------------|-------------|--------|----------|------|
| 30 | 5 | hgf_2level, hgf_3level | blackjax, numpyro | 4 |
| 60 | 10 | hgf_2level, hgf_3level | blackjax, numpyro | 4 |
| 102 | 17 | hgf_2level, hgf_3level | blackjax, numpyro | 4 |
| 150 | 25 | hgf_2level, hgf_3level | blackjax, numpyro | 4 |
| 198 | 33 | hgf_2level, hgf_3level | blackjax, numpyro | 4 |

**Total head-to-head fits:** 5 P-values x 2 models x 2 backends = **20**

### A-vs-A noise-floor grid

Establishes natural MCMC variance baseline at a representative cohort size
(P=60). Same backend, same config, different seeds (42 and 43).

| Backend | Model | Seeds | Fits |
|---------|-------|-------|------|
| blackjax | hgf_2level | 42, 43 | 2 |
| blackjax | hgf_3level | 42, 43 | 2 |
| numpyro | hgf_2level | 42, 43 | 1 (seed=43 only; seed=42 already in head-to-head) |
| numpyro | hgf_3level | 42, 43 | 1 (seed=43 only; seed=42 already in head-to-head) |

**Total noise-floor fits:** 4 (backend x model) x 2 seeds - 2 overlapping
with head-to-head at P=60 = **6** additional fits

### Total SLURM tasks: 26

- 20 head-to-head
- 6 noise-floor (additional)

### Phase 31 gating

If Phase 31 benchmark results show that specific cells CRASH (e.g., 3-level
at P=198 with diagonal mass), those cells are accepted as TIMEOUT/CRASH
outcomes in the audit grid rather than being excluded. The grid is submitted
unconditionally; crash outcomes are valid data points (they indicate backend
limitations).

---

## 5. Metrics Logged Per Fit

Every audit fit writes a JSON result file with these fields (CSV columns for
aggregation):

### Identification columns

| Column | Type | Description |
|--------|------|-------------|
| `backend` | str | "blackjax" or "numpyro" |
| `model` | str | "hgf_2level" or "hgf_3level" |
| `n_participants` | int | P value from cohort grid |
| `mass_matrix` | str | "diagonal" or "dense" |
| `seed` | int | Per-cell random seed |
| `job_id` | str | SLURM job ID for traceability |

### Performance columns

| Column | Type | Description |
|--------|------|-------------|
| `walltime_s` | float | Total elapsed time (includes compile + sample) |
| `compile_time_s` | float | Estimated via fresh-process isolation (see Section 8) |
| `ess_bulk_min` | float | Minimum ESS across all parameters (ArviZ) |
| `ess_per_sec` | float | ess_bulk_min / walltime_s |
| `ess_per_grad_eval` | float | ess_bulk_min / total_leapfrog_steps |
| `divergent_count` | int | Total divergent transitions across all chains |
| `divergent_rate` | float | divergent_count / (n_chains x n_draws) |
| `rhat_max` | float | Maximum R-hat across all parameters (ArviZ) |
| `memory_peak_mb` | float | Peak RSS via tracemalloc |
| `status` | str | PASS, CRASH, TIMEOUT, or DIVERGENT |

### Recovery columns

| Column | Type | Description |
|--------|------|-------------|
| `recovery_corr_omega2` | float | Pearson r(true, posterior_mean) for omega_2 |
| `recovery_corr_beta` | float | Pearson r(true, posterior_mean) for beta |
| `recovery_corr_zeta` | float | Pearson r(true, posterior_mean) for zeta |

### Status classification rules

| Status | Condition |
|--------|-----------|
| PASS | rhat_max < 1.05 AND divergent_rate < 0.05 AND job completed |
| DIVERGENT | divergent_rate >= 0.05 AND job completed |
| TIMEOUT | SLURM walltime exceeded (24h) |
| CRASH | Python exception or non-zero exit code |

---

## 6. Recovery Correlation Specification

Recovery correlation measures how well each backend recovers known ground-truth
parameters from simulated data.

**Simulation procedure:**
1. Sample true parameters uniformly within prior support:
   - `omega_2 ~ Uniform(-6, -1)`
   - `beta ~ Uniform(1, 20)` (stored as `log_beta = log(beta)`)
   - `zeta ~ Uniform(-1, 1)`
   - For 3-level: `omega_3 ~ Uniform(-8, -4)`, `kappa ~ fixed`
2. Generate trial sequences via `simulate_batch` with the sampled parameters
3. Record true parameters in result JSON alongside fit results

**Recovery metric:**
- For each parameter `theta`:
  `recovery_corr_theta = pearsonr(true_theta_vector, posterior_mean_theta_vector)`
- Computed across all P participants within a single fit
- Values near 1.0 indicate good recovery; values near 0 indicate failure

**Minimum acceptable recovery:** `recovery_corr >= 0.5` for omega_2 and beta
at P >= 60 for a fit to count as scientifically valid (not just converged).

---

## 7. Compile Time Isolation

XLA JIT compilation cost is non-trivial (60-300s at large P) and must not
confound the backend comparison.

**Isolation strategy:** Each audit fit is a fresh SLURM task (one Python
process per cell). This ensures:
- No XLA cache from a prior fit contaminates walltime
- Each task pays its own compile cost
- Walltime = compile_time + sample_time (no separation needed for primary analysis)

**Compile time estimation for secondary analysis:**
- The A-vs-A noise-floor runs at P=60 (seeds 42 and 43) share the same array
  shapes. If both are fresh processes, both pay compile cost. To estimate
  compile time in isolation, compare walltime of P=60 audit cell against a
  hypothetical warm-cache run (same shape, second invocation).
- Since every task is a fresh process, `compile_time_s` is reported as NULL in
  the primary analysis. The A-vs-A variance absorbs compile variability.

**Rationale for fresh-process isolation:** Running multiple fits sequentially
in one process would allow JIT cache reuse from fit 1 to fit 2 (same array
shapes), making fit 2 appear 3-5x faster. This is a confound, not a real
backend advantage. Fresh process per cell eliminates this confound.

---

## 8. Decision Rules

Pre-specified decision criteria for the audit conclusion. These rules are
locked before any data is collected.

### Rule 1: Backend dominance

**Criterion:** Backend A dominates Backend B if:
- ESS-per-sec is higher for A in >= 80% of cells (16/20 head-to-head), AND
- The median ESS-per-sec difference exceeds the A-vs-A noise floor

**A-vs-A noise floor:** Defined as the maximum absolute difference in
ESS-per-sec between two same-backend runs at P=60 (across models). If the
BlackJAX noise floor is |ESS_seed42 - ESS_seed43| = X, then a BlackJAX vs
NumPyro difference must exceed X to be considered meaningful.

### Rule 2: Backend equivalence

**Criterion:** Backends are equivalent if:
- No backend dominates by > 20% ESS-per-sec across the grid, AND
- Both backends achieve PASS status in the same set of cells (no cell where
  one PASSES and the other CRASHES/DIVERGES)

### Rule 3: Config-dependent recommendation

**Criterion:** If dominance switches by model (2-level vs 3-level) or by P:
- Report per-class (model, P-range) recommendation
- Example: "BlackJAX preferred for 3-level at P>=102; backends equivalent for 2-level"

### Rule 4: Crash/timeout asymmetry

**Criterion:** If one backend CRASHes or TIMEOUTs in cells where the other
PASSES:
- Report as a reliability finding (not a speed finding)
- The backend that PASSES in more cells receives a reliability advantage
  regardless of ESS-per-sec in shared-PASS cells

### Reporting obligations

- Report ALL cells including CRASH/TIMEOUT (do not cherry-pick PASS-only)
- Report noise-floor magnitude alongside every comparison claim
- If noise-floor exceeds 30% of the observed difference, flag as "below noise floor"

---

## 9. Exclusions

The following configurations are explicitly excluded from the primary
cross-backend audit grid:

| Exclusion | Reason | Where documented |
|-----------|--------|------------------|
| `use_laplace_warmup=True` | BlackJAX-only feature; no NumPyro equivalent; including it confounds backend vs warmup-method | Gap 2 above |
| `use_shard_map=True` | Not consumed by NumPyro path (always vectorized); including it confounds backend vs dispatch-method | Config Equivalence Matrix |
| `use_fp64=True` | Both backends can use fp64 via `set_x64()`, but the primary comparison is fp32. fp64 is a BlackJAX-only secondary analysis for precision sensitivity | Section 1 |
| Dense mass at P>=150 for 3-level | Likely infeasible per Phase 27 DEPS-05 evidence (24h timeout at P=30 dense); included in grid but TIMEOUT is an accepted valid outcome | Phase 27 memo |

### Secondary analyses (not part of primary audit)

These analyses are informative but do NOT contribute to the primary backend
recommendation:

1. **BlackJAX diagonal vs BlackJAX Laplace-warmup:** Tests the mitigation
   ladder, not the backend. Reported in a separate section of the recommendation
   document.
2. **BlackJAX fp32 vs BlackJAX fp64:** Tests precision sensitivity. fp64
   may reduce divergences at a speed cost. BlackJAX-only secondary analysis.

---

## 10. SLURM Execution Specification

### Resource allocation per task

| Resource | Value | Rationale |
|----------|-------|-----------|
| Partition | gpu | Standard GPU partition |
| GPU | 1x A40 (48GB) | Sufficient for P<=198 batched cohort fits |
| CPUs | 4 | JAX host-side parallelism |
| Memory | 32GB | Headroom above observed 8-16GB for large P |
| Walltime | 24:00:00 | Conservative; most cells should finish in 1-6h |
| Exclude | m3g112 | Known cuSPARSE GPU lottery node |

### Array job structure

```
#SBATCH --array=0-25
# Task 0-19: head-to-head grid (5P x 2models x 2backends)
# Task 20-25: noise-floor grid (2backends x 2models x {additional seeds})
```

### GPU assertion (first line of driver)

```python
import jax
assert jax.devices()[0].platform == "gpu", (
    f"Expected GPU but got {jax.devices()[0].platform}. "
    "Resubmit with --exclude=m3g112."
)
```

### Output directory

```
models/power/audit_results/
  audit_{backend}_{model}_P{n}_{mass}_{seed}.json
```

---

## Appendix: Cell Offset Encoding

Deterministic cell_offset for seeding:

```
offset = (P_index * 4) + (model_index * 2) + backend_index
```

Where:
- P_index: 0=30, 1=60, 2=102, 3=150, 4=198
- model_index: 0=hgf_2level, 1=hgf_3level
- backend_index: 0=blackjax, 1=numpyro

Noise-floor runs use offset = 100 + (backend_index * 2) + model_index
(non-overlapping with head-to-head offsets 0-19).

---

## Appendix: Revision History

| Date | Change | Justification |
|------|--------|---------------|
| 2026-05-18 | Initial pre-registration | Phase 32 Plan 01 |
