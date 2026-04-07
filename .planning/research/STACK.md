# Technology Stack: BFDA Power Analysis Milestone

**Project:** PRL HGF Analysis — v1.1 Power Analysis
**Researched:** 2026-04-07
**Scope:** New capabilities only. Existing stack (pyhgf, PyMC, ArviZ, bambi, groupBMC, JAX) is validated and unchanged.

---

## Decision Summary

| Capability | Decision | Rationale |
|---|---|---|
| JZS BF computation | `pingouin.bayesfactor_ttest` (already a dep) | Verified JZS impl with `scipy.integrate.quad`; validated against JASP/R BayesFactor |
| BFDA orchestration | Custom Python module | No Python BFDA package exists; methodology is simple Monte Carlo |
| Cluster parallelism | SLURM array jobs (primary) | Embarrassingly parallel; matches existing cluster infrastructure |
| Within-node parallelism | `joblib` (already a transitive dep) | Optional; use only if packing multiple iterations per array task |
| Power curve visualization | `matplotlib` (already a dep) | Already in stack; no new dependency needed |
| Progress tracking (SLURM) | File-based logging, not tqdm | tqdm progress bars do not render in SLURM log files |

**Net new packages to add to `pyproject.toml`: zero.**

---

## JZS Bayes Factor Computation

### Decision: Use `pingouin.bayesfactor_ttest` — do NOT implement from scratch

`pingouin` is already declared in `pyproject.toml` (`pingouin>=0.5.5`). The current release is **0.6.1** (released 2026-03-28).

**Verification of implementation correctness (HIGH confidence):**

The GitHub source of `pingouin/bayesian.py` was inspected directly. The function `bayesfactor_ttest(t, nx, ny=None, paired=False, alternative='two-sided', r=0.707)` implements the Rouder et al. (2009) JZS prior using `scipy.integrate.quad` over the exact integrand from their Equation 2:

```
integrand(g) = (1 + N·g·r²)^(-1/2)
             · (1 + t²/((1 + N·g·r²)·df))^(-(df+1)/2)
             · (2π)^(-1/2) · g^(-3/2) · exp(-1/(2g))
```

`BF10 = ∫₀^∞ integrand(g) dg / (1 + t²/df)^(-(df+1)/2)`

One-sample case is explicitly handled: when `ny is None` or `ny == 1`, the function sets `n = nx`, `df = nx - 1`. This is precisely the use case for BFDA (testing whether a group-level parameter contrast differs from zero).

**Validated against JASP and the R BayesFactor package.**

**What NOT to do:** Do not re-implement Rouder Eq. 2 with raw `scipy.integrate.quad`. The pingouin implementation is tested, readable, and already available. Implementing from scratch introduces transcription error risk with no benefit.

**One-sided BF note (MEDIUM confidence, from docs):** pingouin computes one-sided BF by doubling the two-sided BF, which differs from R's BayesFactor package behavior. If the BFDA uses directional hypotheses, prefer two-sided BF with the standard BF₁₀ > 6 threshold, or verify against R output on a test case. For a symmetric design-prior scenario (testing group × session interaction), two-sided is the correct default.

**Cauchy scale default:** `r = 0.707` (= √2/2). This is the JASP/BayesFactor default and appropriate for medium effect sizes. The BFDA should parameterize `r` so power curves can be generated at multiple prior scales if needed.

### scipy.integrate.quad is the right tool (no JAX alternative)

JAX does not have a production-ready equivalent to `scipy.integrate.quad`. A GitHub issue (#27493, open as of research date) requests this feature — it does not exist yet. The JZS integral is 1D and fast; `scipy.integrate.quad` (21-point Gauss-Kronrod) completes in microseconds per call. This computation is not the bottleneck — MCMC is. There is no reason to seek a JAX-native solution.

---

## BFDA Orchestration

### Decision: Custom module, no external BFDA package

The R `BFDA` package (nicebread/BFDA) has no Python equivalent. No Python BFDA package exists as of 2026-04 (confirmed: WebSearch found no Python port).

This is not a problem. The BFDA algorithm for a fixed-N design is simple:

```
for each N in sample_sizes:
    for each iteration in 1..K:
        1. simulate synthetic cohort of N participants
        2. fit each participant with MCMC (existing pipeline)
        3. extract parameter posterior means
        4. run one-sample or paired t-test on contrast of interest
        5. compute BF10 via pingouin.bayesfactor_ttest
        6. record BF10
    power[N] = mean(BF10 > threshold)
```

The BFDA module should live at `src/prl_hgf/power/` and contain:
- `simulate.py` — cohort simulation (wraps existing simulation code)
- `compute_bf.py` — BF computation from fitted posteriors
- `orchestrate.py` — iteration loop and result aggregation
- `visualize.py` — power curves

---

## Cluster Parallelism

### Primary approach: SLURM array jobs

The existing cluster infrastructure (M3 MASSIVE) already uses SLURM array-style dispatch (see `cluster/04_fit_mcmc_gpu.slurm`). The natural structure for BFDA is:

```
sbatch --array=0-199 cluster/08_bfda_iteration.slurm
```

Each array task:
- Receives `$SLURM_ARRAY_TASK_ID` as its iteration index and reads N-level config from a pre-written parameter file
- Runs simulation + MCMC fitting for one iteration
- Writes result to `output/bfda/iter_{task_id}.parquet`

A separate aggregation job (with `--dependency=afterok:$ARRAY_JOB_ID`) reads all per-iteration files and computes power curves.

**Why this over joblib within a single job:**
- MCMC fitting (PyMC + JAX) is the compute bottleneck, not Python overhead
- Each iteration needs GPU access; array jobs allocate one GPU per task cleanly
- JAX has a documented incompatibility with `multiprocessing.fork` (confirmed via PyMC GitHub issues #1805, #6362, #7620). Fork-based parallelism inside a JAX process causes deadlocks. SLURM array jobs sidestep this entirely — each task is an independent process with its own JAX initialization
- Existing SLURM scripts already handle conda env activation, JAX cache dirs, and GPU verification

### Secondary (optional): joblib for CPU-only aggregation

`joblib` 1.5.3 (released 2025-12-15) is a transitive dependency (sklearn depends on it) and requires no new install. It can be used in the aggregation step for parallel BF computation across iterations if needed, but this step is trivially fast (microseconds per BF call) and likely does not need parallelism.

**If joblib is used inside a SLURM task**, set `multiprocessing_context="spawn"` or `"forkserver"` — never `"fork"` — due to JAX threading incompatibility.

---

## Power Curve Visualization

### Decision: matplotlib only

`matplotlib` (already a dependency, `>=3.4.0`) is sufficient. Power curves are line plots of P(BF₁₀ > threshold) vs N, with one curve per effect size. `seaborn` (also already a dep) can be used for styling.

`statsmodels.stats.power.TTestPower.plot_power` is not appropriate here because it computes frequentist power (1 - β), not Bayesian power (P(BF > threshold)). Do not use it.

---

## Recommended pyproject.toml Changes

**No changes needed.** All required packages are already declared:

| Package | Current constraint | BFDA use |
|---|---|---|
| `pingouin>=0.5.5` | Already declared | `bayesfactor_ttest` for JZS BF |
| `scipy>=1.10` | Already declared | `scipy.integrate.quad` (used internally by pingouin) |
| `matplotlib>=3.4.0` | Already declared | Power curve plots |
| `pandas>=2.0` | Already declared | Result aggregation |
| `numpy>=2.0.0` | Already declared | Array ops |

`joblib` does not need to be declared explicitly — it is a dependency of scikit-learn (transitive) and will be present in the environment.

---

## Integration Points with Existing Stack

| Existing component | How BFDA uses it |
|---|---|
| `src/prl_hgf/models/` | Simulation uses the same HGF model definitions |
| `src/prl_hgf/fitting/` | MCMC fitting reuses the existing `fit_participant()` interface |
| `cluster/04_fit_mcmc_gpu.slurm` | New `cluster/08_bfda_iteration.slurm` follows same template (conda activate, JAX cache, GPU verify) |
| `config.py` | BFDA paths (output dir, parameter grid file) added as constants |
| `configs/prl_analysis.yaml` | BFDA section added for N-levels, K iterations, effect sizes, BF threshold |

---

## What NOT to Add

| Rejected option | Reason |
|---|---|
| `rpy2` to call R's BFDA package | Heavy dependency, complex env management, pingouin covers the BF computation already |
| `torchquad` (GPU quadrature) | 1D integration is not the bottleneck; overkill and adds PyTorch dependency |
| `ipyparallel` | Cluster-level parallelism via SLURM arrays is cleaner; ipyparallel adds broker complexity |
| Custom JZS implementation via `scipy.integrate.quad` | pingouin already provides this, tested and validated |
| `statsmodels` power functions | Frequentist power, not Bayesian power — wrong abstraction |

---

## Sources

- [pingouin 0.6.1 on PyPI](https://pypi.org/project/pingouin/) — version confirmed 2026-03-28
- [pingouin bayesfactor_ttest documentation](https://pingouin-stats.org/generated/pingouin.bayesfactor_ttest.html) — function signature and mathematical formula
- [pingouin bayesian.py source](https://github.com/raphaelvallat/pingouin/blob/master/src/pingouin/bayesian.py) — `scipy.integrate.quad` usage and one-sample case handling confirmed
- [nicebread/BFDA R package](https://github.com/nicebread/BFDA) — confirmed no Python equivalent exists
- [joblib 1.5.3 on PyPI](https://pypi.org/project/joblib/) — version confirmed 2025-12-15
- [JAX multiprocessing fork issue #1805](https://github.com/jax-ml/jax/issues/1805) — fork incompatibility documented
- [PyMC issue #7620](https://github.com/pymc-devs/pymc/issues/7620) — fork causes deadlocks with JAX ops
- [JAX scipy.integrate.quad feature request #27493](https://github.com/jax-ml/jax/issues/27493) — confirms no JAX-native quad exists
- [Parallelizing Workloads with Slurm Job Arrays](https://blog.ronin.cloud/slurm-job-arrays/) — SLURM array job patterns
- [Schönbrodt & Wagenmakers 2018 BFDA paper](https://link.springer.com/article/10.3758/s13423-017-1230-y) — methodology reference
