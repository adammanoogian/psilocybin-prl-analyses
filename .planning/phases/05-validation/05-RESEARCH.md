# Phase 5: Validation & Comparison - Research

**Researched:** 2026-04-05
**Domain:** Parameter recovery analysis, Bayesian model comparison (random-effects BMS), ArviZ model evidence, scientific visualization
**Confidence:** MEDIUM (critical unknown resolved: Python BMS implementation exists; WAIC/LOO has a significant constraint requiring workaround)

---

## Summary

Phase 5 has two independent tracks: (1) parameter recovery analysis comparing true vs. recovered parameter values across simulated participants, and (2) formal Bayesian model comparison using the random-effects BMS algorithm from Rigoux et al. 2014.

The recovery track is straightforward: the batch fitting DataFrame already carries `true_*` columns alongside `mean` (posterior mean). Pearson r, bias (mean error), and RMSE are the standard metrics in the computational psychiatry literature. The standard for "good recovery" is r > 0.7; omega_3 is expected to fall below this and must be explicitly documented.

The BMS track has one critical blocker: the existing PyMC models use `pm.Potential` for the custom JAX likelihood, which prevents automatic log_likelihood storage in the ArviZ InferenceData. `az.waic` and `az.loo` require the `log_likelihood` group. The solution is to re-evaluate the logp function over posterior samples post-hoc and manually add a `log_likelihood` group to the InferenceData before calling ArviZ functions. For the group-level random-effects BMS, `groupBMC` (pip package, Python port of VBA_groupBMC) exists and takes a (n_subjects, n_models) log model evidence matrix.

**Primary recommendation:** Use WAIC (not LOO) as the per-subject model evidence metric because it requires only the total log-likelihood per subject (which can be computed post-hoc from the JAX logp function), avoids the complexity of pointwise LOO importance sampling, and maps cleanly to the scalar input the groupBMC algorithm needs. Implement a `compute_subject_log_evidence` helper that re-evaluates the logp function across all posterior samples and returns WAIC per subject.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.stats | bundled with scipy | Pearson r, RMSE, bias computation | Standard scientific Python; `pearsonr` returns r and p-value |
| numpy | project dependency | Array math for metrics | No alternative |
| matplotlib | project dependency | Scatter plots, bar plots, heatmaps | Standard visualization |
| seaborn | project dependency | Heatmaps (sns.heatmap for correlation matrix) | Cleaner defaults than raw matplotlib |
| groupBMC | 1.0 (pip) | Random-effects BMS (Rigoux 2014 algorithm) | Only Python implementation of VBA_groupBMC |
| arviz | project dependency | WAIC computation, InferenceData management | Already used in Phase 4 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | project dependency | Joining true vs. fitted DataFrames | Pivot and merge operations for recovery analysis |
| xarray | arviz dependency | Manually constructing log_likelihood DataArray | Required to add log_likelihood group to InferenceData |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| groupBMC | Implement VB-Dirichlet from scratch | groupBMC is tested, matches MATLAB reference; from-scratch risks subtle VB bugs |
| WAIC | LOO-CV (az.loo) | Both require log_likelihood group; WAIC is simpler to compute post-hoc for Potential models; LOO needs pointwise log-likelihood per trial which is harder with our aggregated JAX logp |
| scipy.stats.pearsonr | numpy.corrcoef | pearsonr returns p-value; corrcoef does not |

**Installation:**
```bash
pip install groupBMC==1.0
```

groupBMC is not in the existing pyproject.toml and must be added as a dependency.

---

## Architecture Patterns

### Recommended Project Structure

```
src/prl_hgf/
├── validation/          # New module for Phase 5
│   ├── __init__.py
│   ├── recovery.py      # Parameter recovery metrics and plots
│   ├── bms.py           # WAIC computation + groupBMC wrapper
│   └── plots.py         # Visualization helpers (scatter, bar, heatmap)
scripts/
└── 05_run_validation.py # Pipeline entry point
```

### Pattern 1: Recovery Analysis from Batch DataFrame

The batch fitting DataFrame has one row per (participant_id, group, session, parameter) with columns including `mean` (posterior mean) and `flagged`. The simulation batch DataFrame has one row per trial with `true_*` columns repeated on every row (constant within participant-session). Join these two DataFrames to build a recovery DataFrame.

**Join strategy:** The simulation DataFrame must be reduced to one row per (participant_id, group, session) before joining to the fitting results. Use `groupby(['participant_id', 'group', 'session']).first()` to extract the true parameter values.

**Example:**
```python
# Source: project DataFrames from Phase 3 (simulate_batch) and Phase 4 (fit_batch)

# Step 1: extract unique true parameter rows from simulation output
true_params = (
    sim_df.groupby(["participant_id", "group", "session"])[
        ["true_omega_2", "true_omega_3", "true_kappa", "true_beta", "true_zeta"]
    ]
    .first()
    .reset_index()
)

# Step 2: pivot fitting results to wide form
fitted_wide = (
    fit_df[fit_df["flagged"] == False]   # exclude unconverged fits
    .pivot_table(
        index=["participant_id", "group", "session"],
        columns="parameter",
        values="mean",
    )
    .reset_index()
)
# columns after pivot: participant_id, group, session, beta, kappa, omega_2, omega_3, zeta

# Step 3: merge
recovery_df = true_params.merge(fitted_wide, on=["participant_id", "group", "session"])
```

### Pattern 2: Per-Parameter Recovery Metrics

```python
# Source: scipy.stats documentation + standard computational psychiatry practice
from scipy import stats
import numpy as np

def compute_recovery_metrics(true_vals: np.ndarray, recovered_vals: np.ndarray) -> dict:
    """Compute r, bias, RMSE for one parameter."""
    mask = np.isfinite(true_vals) & np.isfinite(recovered_vals)
    t, r = true_vals[mask], recovered_vals[mask]
    r_val, p_val = stats.pearsonr(t, r)
    bias = float(np.mean(r - t))          # mean error (signed)
    rmse = float(np.sqrt(np.mean((r - t) ** 2)))
    return {"r": r_val, "p": p_val, "bias": bias, "rmse": rmse, "n": int(mask.sum())}
```

### Pattern 3: Recovery Scatter Plot

Standard in the computational psychiatry literature: one scatter panel per parameter, x-axis = true value, y-axis = posterior mean, diagonal identity line, r annotated.

```python
# Source: standard matplotlib/seaborn pattern verified against CPsy literature (PMC11951975)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_recovery_scatter(
    ax: plt.Axes,
    true_vals: np.ndarray,
    recovered_vals: np.ndarray,
    param_label: str,
    r_threshold: float = 0.7,
) -> None:
    mask = np.isfinite(true_vals) & np.isfinite(recovered_vals)
    t, r = true_vals[mask], recovered_vals[mask]
    r_val, p_val = stats.pearsonr(t, r)

    ax.scatter(t, r, alpha=0.4, s=20)

    # Identity line
    lims = [min(t.min(), r.min()), max(t.max(), r.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="identity")

    color = "green" if abs(r_val) >= r_threshold else "red"
    ax.set_title(f"{param_label}\nr = {r_val:.2f}, p = {p_val:.3f}", color=color)
    ax.set_xlabel("True value")
    ax.set_ylabel("Posterior mean")
```

### Pattern 4: WAIC Computation for pm.Potential Models

**Critical constraint:** The existing PyMC models use `pm.Potential("loglike", logp_op(...))`. ArviZ `az.waic` requires a `log_likelihood` group in InferenceData. This group is NOT populated automatically for `pm.Potential` — only for variables defined with `observed=` keyword.

**Solution:** Re-evaluate the logp Op over posterior samples post-hoc, then manually add the result to the InferenceData as a scalar `log_likelihood` variable.

```python
# Source: PyMC Discourse discussion + ArviZ InferenceData API
import xarray as xr
import numpy as np
import arviz as az

def compute_waic_for_potential_model(
    idata: az.InferenceData,
    logp_op,          # the Op returned by build_logp_ops_*level
    posterior_samples: dict,  # {param_name: array shape (chains, draws)}
) -> az.ELPDData:
    """Compute WAIC for a model whose likelihood was specified via pm.Potential.

    Strategy: evaluate logp_op at each posterior sample, collect the scalar
    total log-likelihood per sample, wrap as a (chains, draws, 1) DataArray,
    add to idata.log_likelihood, then call az.waic.
    """
    chains, draws = next(iter(posterior_samples.values())).shape
    loglike_vals = np.empty((chains, draws))

    for c in range(chains):
        for d in range(draws):
            params = {k: float(v[c, d]) for k, v in posterior_samples.items()}
            # evaluate the JAX-backed Op with numpy inputs
            loglike_vals[c, d] = float(logp_op(**params).eval())

    # Add as (chain, draw, obs) where obs dim = 1 (scalar likelihood)
    ll_da = xr.DataArray(
        loglike_vals[:, :, np.newaxis],
        dims=["chain", "draw", "loglike_dim_0"],
    )
    idata.add_groups({"log_likelihood": {"loglike": ll_da}})
    return az.waic(idata, var_name="loglike")
```

**Alternative simpler approach:** If the logp Op accepts keyword arguments matching parameter names from the InferenceData posterior, iterate over `idata.posterior` directly using xarray vectorized operations.

**Note:** WAIC returns an `ELPDData` object. The scalar quantity used as input to groupBMC is `waic_result.elpd_waic` (expected log pointwise predictive density, higher = better model). This should be negated or used as-is depending on the BMS input convention (groupBMC expects log model evidence, which is higher-is-better — same sign as elpd_waic).

### Pattern 5: groupBMC Usage

groupBMC implements the VBA_groupBMC algorithm (Variational Bayes on Dirichlet-Multinomial). Input is a (n_subjects, n_models) matrix of log model evidences.

```python
# Source: github.com/cpilab/group-bayesian-model-comparison README + VBA algorithm docs
from groupBMC.groupBMC import GroupBMC
import numpy as np

# L shape: (n_subjects, n_models) — e.g. (60, 2) for 2-level vs 3-level
# L[i, 0] = WAIC elpd for subject i under 2-level model
# L[i, 1] = WAIC elpd for subject i under 3-level model
L = np.column_stack([elpd_2level, elpd_3level])  # shape (n_subjects, 2)

result = GroupBMC(L).get_result()
# Expected output fields (from VBA_groupBMC MATLAB reference):
# result.alpha: Dirichlet posterior parameters, shape (n_models,)
# result.exp_r: expected model frequencies, shape (n_models,) — sums to 1
# result.xp: exceedance probabilities, shape (n_models,)
# result.pxp: protected exceedance probabilities
# result.bor: Bayesian Omnibus Risk (scalar)
```

**Per-group BMS:** Run GroupBMC separately for each group's subjects.

```python
for group_name in ["healthy_control", "pcs"]:
    mask = group_labels == group_name
    result_group = GroupBMC(L[mask]).get_result()
```

**Protected exceedance probability (PEP) formula** (from Rigoux 2014, if groupBMC does not compute it directly):
```
PEP = (1 - BOR) * EP + BOR / n_models
```

### Pattern 6: Parameter Correlation Matrix

Cross-parameter correlations from the posterior means reveal identifiability problems. A |r| > 0.8 between any two parameters indicates that the model cannot separate them — they trade off in the posterior.

```python
# Source: standard numpy/seaborn pattern
import seaborn as sns
import matplotlib.pyplot as plt

# fitted_wide columns: omega_2, omega_3, kappa, beta, zeta
param_cols = ["omega_2", "omega_3", "kappa", "beta", "zeta"]
corr_matrix = fitted_wide[param_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    ax=ax,
    center=0,
)
ax.set_title("Parameter correlation matrix (posterior means)")
```

### Anti-Patterns to Avoid

- **Joining trial-level sim_df directly to fit_df:** sim_df has one row per trial; fit_df has one row per parameter. Must reduce sim_df to one row per participant-session first.
- **Including flagged fits in recovery analysis:** Flagged participants (r_hat > 1.05 or ESS < 400) have unreliable posterior means. Filter them out before computing recovery metrics, but report how many were excluded.
- **Using az.loo instead of az.waic for Potential models:** LOO requires pointwise observation-level log-likelihoods; our logp Op returns a scalar total. WAIC can work with a single scalar per participant if computed from the variance of the total log-likelihood across posterior samples.
- **Using the wrong sign for groupBMC input:** groupBMC expects log model evidence where higher = better fit. WAIC elpd_waic is higher-is-better (it IS the log predictive density), so use elpd_waic directly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Group-level BMS VB algorithm | Custom Dirichlet-Multinomial VB loop | `groupBMC` pip package | The VB convergence, free-energy computation, and BOR calculation have been validated against the MATLAB reference |
| Pearson r with p-value | `np.corrcoef` | `scipy.stats.pearsonr` | pearsonr returns the p-value; corrcoef does not |
| Correlation heatmap annotation | Custom matplotlib text | `sns.heatmap(annot=True)` | seaborn handles cell annotation cleanly |

**Key insight:** The random-effects BMS algorithm (VB on Dirichlet-Multinomial model) appears simple but the convergence criterion, free-energy bound, BOR computation, and protected exceedance probability calculation each have subtle implementation details. The `groupBMC` package was written specifically to port MATLAB's VBA_groupBMC to Python and is the right tool.

---

## Common Pitfalls

### Pitfall 1: log_likelihood Group Missing from InferenceData

**What goes wrong:** `az.waic(idata)` raises `KeyError: 'log_likelihood'` or returns NaN. This will happen for every participant because `fit_participant` uses `pm.Potential` and does not pass `idata_kwargs={"log_likelihood": True}`.

**Why it happens:** PyMC only auto-populates the `log_likelihood` group for variables defined with `observed=` keyword. `pm.Potential` contributes to the joint density but PyMC does not know it represents a likelihood term.

**How to avoid:** Implement `compute_subject_log_evidence` as a post-hoc function that: (1) re-evaluates the logp Op over all posterior samples, (2) adds a `log_likelihood` group to idata, (3) calls `az.waic`. This requires loading the Idata objects from disk (if batch fitting saved them) or re-fitting is not needed — just re-evaluating the logp.

**Warning signs:** `az.waic` returning NaN or `ELPDData` with `p_waic=0`.

### Pitfall 2: Re-evaluating the JAX-backed Op is Slow

**What goes wrong:** Iterating over (chains × draws) in Python to evaluate the logp Op takes O(chains × draws × participants) time, which could be hours for 180 participants × 4 chains × 1000 draws.

**Why it happens:** The JAX Op compiles once but each Python-level call has overhead. The vectorized path (passing full posterior arrays) is much faster.

**How to avoid:** Use `pytensor.function` or `jax.vmap` to vectorize the evaluation across posterior samples. Alternatively, use the JAX function directly (bypassing the PyTensor Op wrapper) with `jax.vmap` over the posterior samples array. The `ops.py` module builds the JAX function before wrapping it in a PyTensor Op — check if the raw JAX function is exposed.

### Pitfall 3: omega_3 Recovery Expected to Fail

**What goes wrong:** omega_3 Pearson r is reported without context, causing readers to question the analysis.

**Why it happens:** Meta-volatility parameters are known to be poorly identified from binary response data alone (confirmed in PMC11951975 with r = 0.67 for HGF omega_3 with RT data; expected worse without RT).

**How to avoid:** Document the omega_3 recovery result explicitly regardless of its value. The success criteria say r > 0.7 is required for omega_2, beta, zeta but omega_3 and kappa recovery quality is to be documented, not gated. Add a warning annotation to the omega_3 scatter plot.

### Pitfall 4: groupBMC Input Has Wrong Shape

**What goes wrong:** `GroupBMC(L)` raises a shape error or produces nonsensical results.

**Why it happens:** The MATLAB reference uses (n_models × n_subjects) but the Python port uses (n_subjects × n_models). These are transposed.

**How to avoid:** Verify: `L.shape == (n_subjects, n_models)` before calling GroupBMC. The example in the README uses `L = np.array([[-100, -120, ...], ...])` where rows are subjects — confirm this.

### Pitfall 5: Excluding Flagged Fits Creates Biased Recovery

**What goes wrong:** Excluding all flagged participants removes outliers that could be the most informative about recovery failure modes.

**Why it happens:** Convergence failures often correlate with extreme true parameter values.

**How to avoid:** Run recovery analysis twice: once with all participants, once excluding flagged. Report both. The flagged-excluded version is the primary result; the full version is in an appendix.

### Pitfall 6: Per-Group BMS With Unequal Group Sizes

**What goes wrong:** With N=30 per group, per-group BMS has very low power compared to full-sample BMS (N=60).

**Why it happens:** Random-effects BMS needs enough subjects to estimate model frequencies reliably.

**How to avoid:** Report this as a limitation. The primary BMS result is the full-sample run. Per-group BMS is exploratory.

---

## Code Examples

### Recovery Metrics for All Parameters

```python
# Source: standard scipy + project DataFrames
from scipy import stats
import numpy as np

PARAM_MAP = {
    "omega_2": "true_omega_2",
    "omega_3": "true_omega_3",
    "kappa":   "true_kappa",
    "beta":    "true_beta",
    "zeta":    "true_zeta",
}

def compute_all_recovery_metrics(recovery_df: pd.DataFrame) -> pd.DataFrame:
    """Compute r, bias, RMSE for each parameter.

    Parameters
    ----------
    recovery_df : pandas.DataFrame
        Merged DataFrame with true_* columns and fitted posterior-mean
        columns named by parameter (omega_2, omega_3, kappa, beta, zeta).

    Returns
    -------
    pandas.DataFrame
        One row per parameter with columns: parameter, r, p, bias, rmse, n.
    """
    rows = []
    for fitted_col, true_col in PARAM_MAP.items():
        if fitted_col not in recovery_df.columns:
            continue
        true_vals = recovery_df[true_col].to_numpy(dtype=float)
        rec_vals  = recovery_df[fitted_col].to_numpy(dtype=float)
        mask = np.isfinite(true_vals) & np.isfinite(rec_vals)
        t, r = true_vals[mask], rec_vals[mask]
        r_val, p_val = stats.pearsonr(t, r)
        rows.append({
            "parameter": fitted_col,
            "r": r_val,
            "p": p_val,
            "bias": float(np.mean(r - t)),
            "rmse": float(np.sqrt(np.mean((r - t) ** 2))),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)
```

### Exceedance Probability Bar Plot

```python
# Source: standard matplotlib pattern
import matplotlib.pyplot as plt
import numpy as np

def plot_exceedance_probabilities(
    xp: np.ndarray,
    pxp: np.ndarray,
    model_names: list[str],
    title: str = "Bayesian Model Selection",
) -> plt.Figure:
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, xp,  width, label="Exceedance prob (EP)")
    ax.bar(x + width / 2, pxp, width, label="Protected EP (PEP)", alpha=0.8)
    ax.axhline(1.0 / len(model_names), linestyle="--", color="gray", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return fig
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Exceedance probability alone (Stephan 2009) | Protected exceedance probability + BOR (Rigoux 2014) | 2014 | PEP corrects for null case where all models are equally likely; BOR quantifies this risk |
| MATLAB-only BMS | groupBMC Python package | ~2020 | Python pipeline no longer requires MATLAB |
| LOO as default ArviZ comparison | LOO for standard models, WAIC for blackbox Potential models | ongoing | LOO is better in theory but WAIC is more tractable when logp is a scalar |

**Deprecated/outdated:**
- `az.compare()` with multiple InferenceData objects: this uses LOO or WAIC internally but assumes standard observed-variable models. Not usable here due to `pm.Potential` constraint — we bypass `az.compare` and call `az.waic` directly after manually adding the log_likelihood group.

---

## Open Questions

1. **groupBMC output field names**
   - What we know: `get_result()` returns a result object; VBA_groupBMC produces `alpha`, `exp_r`, `xp`, `pxp`, `bor`
   - What's unclear: the exact Python attribute names on the result object (may differ from MATLAB field names)
   - Recommendation: In the first task, run `print(dir(result))` and `print(result.__dict__)` after calling `GroupBMC(L).get_result()` to discover actual field names. Add an integration test that asserts the expected fields exist.

2. **JAX logp Op vectorization for WAIC**
   - What we know: The logp Op accepts scalar pytensor variables and returns a scalar; iterating per-sample in Python is slow
   - What's unclear: Whether the raw JAX function in `ops.py` is accessible for direct `jax.vmap` use
   - Recommendation: Read `ops.py` in detail during implementation. If the JAX function is exposed, use `jax.vmap` over posterior samples for O(10x) speedup.

3. **groupBMC pip package availability and maintenance status**
   - What we know: Package was `pip install groupBMC==1.0`; PyPI search returned no clear entry
   - What's unclear: Whether the package is on PyPI or only on GitHub; whether it installs cleanly with current Python/numpy
   - Recommendation: In the first task, run `pip install groupBMC` and verify. If the package is not on PyPI or fails to install, implement the VB algorithm from scratch using the MATLAB reference code (the algorithm is ~50 lines of core logic).

4. **WAIC scalar input convention for groupBMC**
   - What we know: WAIC returns `elpd_waic` (expected log predictive density, higher = better); groupBMC expects log model evidence (higher = better)
   - What's unclear: Whether groupBMC applies any internal scaling or log-sum-exp normalization that makes absolute values matter
   - Recommendation: Verify with the demo notebook values. If absolute values matter, use raw log-likelihood sums (not WAIC-corrected) as a fallback.

---

## Sources

### Primary (HIGH confidence)
- VBA_groupBMC.m (github.com/MBB-team/VBA-toolbox) — algorithm structure, inputs/outputs, VB iteration
- PyMC Discourse (discourse.pymc.io/t/pointwise-log-likelihood...12350) — pm.Potential cannot produce log_likelihood; CustomDist workaround
- PMC11951975 (Bayesian Workflow for Generative Modeling in Computational Psychiatry, 2025) — recovery thresholds, scatter plot standards, omega_3 caveat

### Secondary (MEDIUM confidence)
- github.com/cpilab/group-bayesian-model-comparison — Python BMS implementation exists, pip installable, takes (n_subjects, n_models) matrix
- mlisi.xyz/post/bms — VB algorithm walkthrough, alpha/r/g variables
- github.com/mattelisi/bmsR — confirms alpha, xp, pxp, bor as standard BMS output fields

### Tertiary (LOW confidence)
- WebSearch result: groupBMC==1.0 available on PyPI — not directly confirmed on PyPI; only GitHub README confirmed
- WebSearch result: WAIC elpd_waic maps directly to groupBMC log evidence input — reasonable assumption, not verified against groupBMC source

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — scipy, matplotlib, seaborn, arviz all verified as project dependencies
- Recovery analysis: HIGH — standard computational psychiatry practice verified in PMC11951975
- pm.Potential WAIC constraint: HIGH — confirmed by PyMC Discourse and PyMC documentation
- groupBMC existence and basic API: MEDIUM — GitHub README confirms existence; output field names need runtime verification
- groupBMC pip availability: LOW — README says `pip install groupBMC==1.0` but PyPI page returned error; must verify on first task

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable domain; groupBMC==1.0 is pinned version)
