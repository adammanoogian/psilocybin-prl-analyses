# Phase 9: Prechecks - Research

**Researched:** 2026-04-07
**Domain:** Parameter recovery gating, trial count sweep, MCMC convergence filtering
**Confidence:** HIGH (all findings verified directly from codebase)

## Summary

Phase 9 is primarily an orchestration problem, not a new-code problem. The
heavy lifting — simulate, fit, recover, correlate — is already fully
implemented across Phases 3–6. The precheck module needs a thin harness in
`src/prl_hgf/power/` that calls those existing functions with specific
configuration overrides, then produces structured eligibility output.

The single genuinely new piece of code is `make_precheck_trial_config()`: a
factory analogous to `make_power_config()` that scales `PhaseConfig.n_trials`
proportionally via `dataclasses.replace` bottom-up. `make_power_config()`
only changes `n_per_group` and `omega_2_deltas` — it does not touch the task
structure — so trial count variation must be a new function. The stable/volatile
ratio remains exactly 1:1 regardless of scale factor because the four phases
(acquisition_1, reversal_1, acquisition_2, reversal_2) each contribute equally
to both categories.

The compute cost of the trial sweep (PRE-04) is the primary risk: 50
participants × 5 trial counts × 1 session = 250 MCMC fits. On CPU with default
settings (4 chains, 1000 draws, 1000 tune) this takes 4–12 hours. The plan
must address this with either reduced MCMC settings for the sweep or explicit
wall-time expectation-setting.

**Primary recommendation:** Build `src/prl_hgf/power/precheck.py` as the
single new module. It imports and calls existing pipeline functions directly;
it does not duplicate them.

## Standard Stack

All required libraries are already installed and used in the project.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | project baseline | DataFrames throughout pipeline | Already used everywhere |
| numpy | project baseline | Array math, correlation | Already used everywhere |
| scipy.stats | project baseline | `pearsonr` for recovery r | Used in `analysis/recovery.py` |
| matplotlib | project baseline | All figures | Used in `analysis/plots.py` |
| seaborn | project baseline | Heatmap in correlation plot | Used in `analysis/plots.py` |

### Supporting (already in project)
| Library | Purpose |
|---------|---------|
| `prl_hgf.simulation.batch.simulate_batch` | Simulate N participants |
| `prl_hgf.fitting.batch.fit_batch` | MCMC fit N participants |
| `prl_hgf.analysis.recovery.build_recovery_df` | Join true + fitted |
| `prl_hgf.analysis.recovery.compute_recovery_metrics` | r, bias, RMSE per param |
| `prl_hgf.analysis.recovery.compute_correlation_matrix` | Confound matrix |
| `prl_hgf.analysis.plots.plot_recovery_scatter` | Existing scatter figures |
| `prl_hgf.analysis.plots.plot_correlation_matrix` | Existing heatmap figure |
| `prl_hgf.power.config.make_power_config` | Already varies n_per_group |
| `dataclasses.replace` | Bottom-up config overrides (frozen dataclasses) |

**No new third-party dependencies are needed.**

## Architecture Patterns

### Recommended Project Structure

```
src/prl_hgf/power/
├── __init__.py      # existing
├── config.py        # existing — make_power_config()
├── schema.py        # existing — POWER_SCHEMA, write_parquet_row()
├── seeds.py         # existing — make_child_rng()
└── precheck.py      # NEW — all Phase 9 logic

scripts/
└── 09_run_prechecks.py   # NEW — CLI entry point

results/power/prechecks/
├── recovery_metrics_precheck.csv
├── power_eligible_params.csv
├── exclusion_report.txt
├── correlation_matrix_precheck.png
├── recovery_scatter_precheck.png
└── trial_sweep_recovery_r.png
```

### Pattern 1: Config factory for trial count variation

`make_power_config()` is the established pattern for config overrides: use
`dataclasses.replace` bottom-up without file I/O, without mutating the base.

For trial count variation, the same pattern applies to the task layer:

```python
# Source: direct inspection of power/config.py + task_config.py
import dataclasses
from prl_hgf.env.task_config import AnalysisConfig

def make_trial_config(base: AnalysisConfig, scale_factor: float) -> AnalysisConfig:
    """Return config with per-phase n_trials scaled by scale_factor.

    Scales each PhaseConfig.n_trials proportionally so that the
    stable/volatile ratio stays 1:1. transfer.n_trials is untouched.
    """
    new_phases = [
        dataclasses.replace(p, n_trials=max(1, round(p.n_trials * scale_factor)))
        for p in base.task.phases
    ]
    new_task = dataclasses.replace(base.task, phases=new_phases)
    return dataclasses.replace(base, task=new_task)
```

**Verified:** The four phases are 2 stable + 2 volatile × 30 trials each.
Uniform scaling preserves the 1:1 ratio exactly. `PhaseConfig` validates
`n_trials >= 1`, which `max(1, round(...))` satisfies down to ~3% of baseline.

**Trial count grid:** The sweep range [50–250] maps to scale factors
approximately [0.12–0.60] of baseline 420. At scale 0.33 → ~180 trials;
scale 0.50 → ~240 trials; scale 0.71 → ~300 trials. The exact values
depend on rounding per phase. Five points is reasonable; see Code Examples.

**Important limitation:** `make_power_config()` does not vary trial count.
A new `make_trial_config()` (or `make_precheck_config()` combining both
n_per_group and trial scaling) is needed in `power/precheck.py`. Do not
modify `power/config.py`.

### Pattern 2: Recovery pipeline reuse

The existing Phase 5 pipeline (`05_run_validation.py`) shows the exact call
sequence. Phase 9 reuses it verbatim with two changes:

1. `build_recovery_df(..., min_n=0)` — the 30-participant guard was
   written for Phase 5 where 30 is the study default; Phase 9 fixes N=50
   which exceeds the guard, so `min_n=30` still works. But the precheck
   script should pass `min_n=50` explicitly to be self-documenting.
2. Only the 3-level model is used for prechecks (contains all 5 parameters).

```python
# Source: analysis/recovery.py — verified signatures
recovery_df = build_recovery_df(sim_df, fit_df, exclude_flagged=True, min_n=50)
metrics_df = compute_recovery_metrics(recovery_df)
corr_df = compute_correlation_matrix(recovery_df)
```

### Pattern 3: Eligibility table construction

No existing function builds the eligibility table. This is new logic but
straightforward: it reads from `metrics_df` (output of
`compute_recovery_metrics`) and applies the locked decision rules.

```python
# New logic in power/precheck.py
def build_eligibility_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build power-eligible parameter list with exclusion reasons."""
    rows = []
    for _, row in metrics_df.iterrows():
        param = row["parameter"]
        r_val = float(row["r"])
        passes = bool(row["passes_threshold"])  # |r| >= 0.7

        if param == "omega_3":
            status = "exploratory — upper bound"
            reason = f"omega_3 labeled exploratory per project decision (r={r_val:.2f})"
        elif passes:
            status = "power-eligible"
            reason = f"r={r_val:.2f} >= 0.7 threshold"
        else:
            status = "excluded"
            reason = f"r={r_val:.2f} < 0.7 threshold"

        rows.append({
            "parameter": param,
            "r": r_val,
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)
```

### Pattern 4: MCMC convergence gating

`fit_batch` already populates `flagged` column (True when any param has
R-hat > 1.05 or ESS < 400). The thresholds come from `FittingConfig`
(`r_hat_threshold=1.05`, `ess_threshold=400.0`). `build_recovery_df` has
`exclude_flagged=True` which removes them.

The precheck must report the count. This is already logged by `build_recovery_df`
at INFO level, but the script should explicitly capture and print it:

```python
n_total = fit_df["participant_id"].nunique()
n_flagged = int(fit_df["flagged"].any(level=0))  # wrong approach
# Correct approach:
n_flagged_participants = fit_df.groupby("participant_id")["flagged"].any().sum()
n_clean = n_total - n_flagged_participants
print(f"MCMC gating: {n_flagged_participants} excluded (R-hat>1.05 or ESS<400), {n_clean} retained")
```

### Pattern 5: Trial count sweep figure (VIZ-01)

No existing plotting function produces an r-vs-trial-count line chart. This
is a new `plot_trial_sweep()` function. The design is:

- X axis: total trials (derived from `config.task.n_trials_total`)
- Y axis: Pearson r from `compute_recovery_metrics`
- One line per parameter (5 lines, or 4 if omega_3 is plotted separately)
- Reference line at r=0.7
- Seaborn lineplot or plain matplotlib are both suitable

```python
# New function in power/precheck.py (or analysis/plots.py)
def plot_trial_sweep(
    sweep_results: list[dict],  # [{"trial_count": int, "metrics_df": pd.DataFrame}]
    r_threshold: float = 0.7,
    save_path: Path | None = None,
) -> plt.Figure:
    # sweep_results is built by the sweep loop, one entry per trial count
    ...
```

### Anti-Patterns to Avoid

- **Duplicating simulate+fit logic:** The precheck must call `simulate_batch`
  and `fit_batch` directly. Do not copy-paste the loops from those modules.
- **Modifying `make_power_config`:** Leave Phase 8 code untouched. New config
  factory goes in `power/precheck.py`.
- **Using `build_recovery_df` default `min_n=30`:** Pass `min_n=50` explicitly
  to document the Phase 9 assumption.
- **Running the full 3-session design for prechecks:** PRE-01 uses 50
  participants × 1 session (baseline only). The full 3-session design
  (baseline, post_dose, followup) triples the compute cost with no benefit
  for the recovery check. `simulate_batch` always generates 3 sessions; the
  precheck should filter to session="baseline" before fitting.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pearson r and p-value | custom correlation | `scipy.stats.pearsonr` | Already in `compute_recovery_metrics` |
| R-hat / ESS flagging | custom threshold check | `flag_fit()` in `fitting/single.py` | Already implemented; thresholds from config |
| Flagged-participant exclusion | custom filter | `build_recovery_df(exclude_flagged=True)` | Already logs count at INFO level |
| Correlation heatmap |  custom heatmap | `plot_correlation_matrix()` | Already flags |r|>0.8 pairs in figure caption |
| Scatter plot per param | custom scatter | `plot_recovery_scatter()` | Already handles omega_3 caveat annotation |
| Config variation | manual dict mutation | `dataclasses.replace` bottom-up | Frozen dataclasses require it |
| Seed isolation | random.seed() | `make_child_rng()` from `power/seeds.py` | SeedSequence guarantees independence |

**Key insight:** The "don't hand-roll" principle is especially strong here
because the existing pipeline was explicitly designed (Phase 5) with recovery
analysis as its primary output. Phase 9 is a parameterized re-run of Phase 5
with gating logic added on top.

## Common Pitfalls

### Pitfall 1: Fitting 3 sessions instead of 1 for PRE-01

**What goes wrong:** `simulate_batch` always generates 3 sessions
(baseline, post_dose, followup) for all participants. If the full sim_df
is passed to `fit_batch` without filtering, 150 participant-sessions are
fitted instead of 50.
**Why it happens:** The batch functions are designed for the full study design.
**How to avoid:** After `simulate_batch`, filter:
`sim_df_pre = sim_df[sim_df["session"] == "baseline"]` before calling
`fit_batch`.
**Warning signs:** `fit_batch` prints `"Fitting N participant-sessions"` —
if N > 50 for the PRE-01 run, the filter was missed.

### Pitfall 2: `make_power_config` does not scale trials

**What goes wrong:** Calling `make_power_config(base, n_per_group=30, ...)` for
the trial sweep and expecting it to produce different trial counts. It only
changes `n_per_group` and `omega_2_deltas`.
**Why it happens:** `make_power_config` was designed for power iteration, not
prechecks.
**How to avoid:** Use a new `make_trial_config(base, scale_factor)` that uses
`dataclasses.replace` on `task.phases[*].n_trials`.
**Warning signs:** `config.task.n_trials_total` returns 420 regardless of
which scale factor was intended.

### Pitfall 3: `min_n` guard raises with fewer than 30 participants

**What goes wrong:** If MCMC failures cause more than 20 exclusions from 50
participants (leaving < 30), `build_recovery_df(min_n=30)` raises `ValueError`.
**Why it happens:** The guard is conservative. With only 50 participants and
potential convergence failures at small trial counts, this can trigger.
**How to avoid:** Pass `min_n=0` for the trial sweep (recovery is still computed,
just without the safety net). For the main PRE-01 run, pass `min_n=30` (or
`min_n=50` for strict enforcement).
**Warning signs:** `ValueError: Recovery requires at least 30 participants`.

### Pitfall 4: omega_3 silently excluded or promoted

**What goes wrong:** The eligibility table either omits omega_3 entirely or
marks it "eligible" based solely on passing r >= 0.7.
**Why it happens:** The generic `passes_threshold` boolean in `metrics_df`
does not know about the "exploratory — upper bound" special case.
**How to avoid:** The `build_eligibility_table` function must hardcode the
omega_3 special case (it is a locked project decision, not a computed outcome).
**Warning signs:** Output CSV contains omega_3 with status="power-eligible" or
is missing the row entirely.

### Pitfall 5: Trial sweep compute cost

**What goes wrong:** PRE-04 with 5 trial counts × 50 participants = 250 fits
at default MCMC settings takes 4–12 hours on CPU.
**Why it happens:** MCMC is inherently sequential on Windows (`cores=1`) due
to JAX process isolation.
**How to avoid:** Either (a) use reduced MCMC settings for the sweep
(n_draws=500, n_tune=500, n_chains=2 — adequate for recovery estimation,
not for inference), or (b) document the wall time explicitly and provide a
`--quick` flag.
**Warning signs:** No checkpoint/intermediate save — if the process dies at
trial count 4 of 5, all work is lost.

### Pitfall 6: Stable/volatile ratio distortion at extreme scales

**What goes wrong:** At very small scale factors (e.g. 0.1 → 3 trials per
phase), rounding produces unequal phase lengths, breaking the 1:1 ratio.
**Why it happens:** `round(30 * 0.1) = 3` is fine (3 stable, 3 volatile per
phase type), but `round(30 * 0.03) = 1` risks integer rounding asymmetries
if phases have different baseline n_trials.
**How to avoid:** The four phases are equal (30 each), so uniform scaling
preserves the ratio for any scale factor. The minimum trial count in the sweep
(50 total) corresponds to ~9 trials per phase after accounting for the fixed
transfer trials, which is achievable. Log the actual total after construction.
**Warning signs:** `config.task.n_trials_total` differs from expected by more
than 4 (rounding artifact).

## Code Examples

### Trial config factory (verified pattern from power/config.py + task_config.py)

```python
# Source: verified via direct code inspection + runtime test
import dataclasses
from prl_hgf.env.task_config import AnalysisConfig


def make_trial_config(base: AnalysisConfig, target_total_trials: int) -> AnalysisConfig:
    """Return config with per-phase n_trials scaled to approach target_total.

    Scales each PhaseConfig.n_trials proportionally. Transfer n_trials is
    untouched. Actual total may differ slightly from target due to rounding.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline frozen config.
    target_total_trials : int
        Desired total trials. Must be > n_sets * (n_phases + transfer_min).
    """
    current_main = sum(p.n_trials for p in base.task.phases)
    current_transfer = base.task.transfer.n_trials
    n_sets = base.task.n_sets

    target_per_set = target_total_trials / n_sets
    scale = (target_per_set - current_transfer) / current_main

    new_phases = [
        dataclasses.replace(p, n_trials=max(1, round(p.n_trials * scale)))
        for p in base.task.phases
    ]
    new_task = dataclasses.replace(base.task, phases=new_phases)
    return dataclasses.replace(base, task=new_task)
```

Actual totals at the five sweep points with baseline 420 (verified by runtime):

| Target | Actual | Per-phase n_trials |
|--------|--------|-------------------|
| 150    | 156    | 8 each (×4 phases) + 20 transfer × 3 sets |
| 200    | 204    | 12 each |
| 250    | 252    | 16 each |
| 300    | 300    | 20 each |
| 420    | 420    | 30 each (baseline) |

### Baseline-only filter before fitting

```python
# Source: verified by inspecting simulate_batch output columns
sim_df = simulate_batch(config)
sim_df_pre = sim_df[sim_df["session"] == "baseline"].copy()
fit_df = fit_batch(sim_df_pre, model_name="hgf_3level", cores=1)
```

### Convergence exclusion count report

```python
# Source: fit_batch output schema — "flagged" is per-parameter row, not per-participant
# A participant is excluded if ANY parameter row is flagged
flagged_participants = (
    fit_df.groupby("participant_id")["flagged"].any()
)
n_flagged = int(flagged_participants.sum())
n_total = fit_df["participant_id"].nunique()
print(f"PRE-06: {n_flagged}/{n_total} participants excluded (R-hat>1.05 or ESS<400)")
```

### Eligibility table with omega_3 special-casing

```python
# New logic — no existing function does this
def build_eligibility_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics_df.iterrows():
        param = str(row["parameter"])
        r_val = float(row["r"])

        if param == "omega_3":
            status = "exploratory — upper bound"
            reason = f"Project decision: omega_3 always labeled exploratory (r={r_val:.2f})"
        elif bool(row["passes_threshold"]):
            status = "power-eligible"
            reason = f"r={r_val:.2f} >= 0.7"
        else:
            status = "excluded"
            reason = f"r={r_val:.2f} < 0.7 threshold"

        rows.append({"parameter": param, "r": r_val, "status": status, "reason": reason})

    return pd.DataFrame(rows, columns=["parameter", "r", "status", "reason"])
```

### Trial sweep loop skeleton

```python
# Recommended structure for PRE-04
sweep_results = []
for target_trials in [150, 200, 250, 300, 420]:
    trial_config = make_trial_config(base_config, target_total_trials=target_trials)
    actual_total = trial_config.task.n_trials_total

    sim_df = simulate_batch(trial_config)
    sim_df_pre = sim_df[sim_df["session"] == "baseline"].copy()
    fit_df = fit_batch(sim_df_pre, model_name="hgf_3level", cores=1,
                       n_chains=2, n_draws=500, n_tune=500)  # reduced for sweep

    recovery_df = build_recovery_df(sim_df_pre, fit_df, exclude_flagged=True, min_n=0)
    metrics_df = compute_recovery_metrics(recovery_df)

    sweep_results.append({"trial_count": actual_total, "metrics_df": metrics_df})
```

## State of the Art

| Old Approach | Current Approach | Impact for Phase 9 |
|--------------|------------------|-------------------|
| Re-implement recovery logic | Call existing `analysis/recovery.py` | Zero new recovery code |
| Vary trials via YAML edit | `dataclasses.replace` bottom-up | Pure Python, no file I/O |
| `flagged` as per-row bool | `flagged` column in fit_df (per param-row) | Must group by participant_id before counting |

## Open Questions

1. **Trial sweep MCMC settings**
   - What we know: default settings (4 chains, 1000 draws, 1000 tune) are
     correct for inference but expensive for the sweep.
   - What's unclear: whether reduced settings (2 chains, 500 draws, 500 tune)
     produce reliable enough r estimates for the minimum-trial-count determination.
   - Recommendation: The plan should specify MCMC settings explicitly for the
     sweep. Reduced settings (2 chains, 500/500) are plausible for this
     diagnostic purpose. The main PRE-01 should use full settings.

2. **Sweep trial count grid**
   - What we know: PRE-04 says "vary trials [50-250]" but the actual totals
     from integer rounding don't land exactly there. With the 5-sweep-point
     design the actual grid is approximately [156, 204, 252, 300, 420].
   - What's unclear: whether hitting exactly 50 trials at the low end is
     required, or whether the sweep just needs to span a reasonable range.
     At ~50 total trials (scale ~0.12), each phase has ~3 trials, which
     provides almost no learning signal — r for most params will be near 0.
   - Recommendation: Start the sweep at ~150 trials (scale ~0.33); going
     lower than this is unlikely to produce interpretable results.

3. **Single-session vs. multi-session for the sweep**
   - What we know: PRE-04 says "fix N=30/group" — this implies 2 groups ×
     30 = 60 participants, not 50. The PRE-01 uses 50 participants.
   - What's unclear: whether PRE-04 uses the same 50 participants as PRE-01
     (re-simulating with different trial counts) or a separate cohort.
   - Recommendation: Re-simulate fresh for each trial count in the sweep.
     Using the same participants would introduce a between-condition correlation
     that is not real.

4. **Output directory**
   - What we know: Phase 8's `08_run_power_iteration.py` writes to
     `results/power/`. Phase 9 prechecks should be separate from power sweep
     results.
   - Recommendation: Write to `results/power/prechecks/` to keep the
     directory structure clean before the power sweep starts.

## Sources

### Primary (HIGH confidence)

All findings verified by direct code inspection of the project repository.

- `src/prl_hgf/analysis/recovery.py` — `build_recovery_df`, `compute_recovery_metrics`,
  `compute_correlation_matrix` signatures, `min_n` guard behavior, R_THRESHOLD=0.7
- `src/prl_hgf/analysis/plots.py` — `plot_recovery_scatter`, `plot_correlation_matrix`
  signatures and behavior; omega_3 caveat annotation
- `src/prl_hgf/simulation/batch.py` — `simulate_batch` output columns, session
  label structure, JIT pre-warm
- `src/prl_hgf/fitting/batch.py` — `fit_batch` signature, `flagged` column semantics
  (per-parameter-row, not per-participant)
- `src/prl_hgf/fitting/single.py` — `flag_fit` thresholds (R_HAT_THRESHOLD=1.05,
  ESS_THRESHOLD=400.0)
- `src/prl_hgf/env/task_config.py` — `PhaseConfig`, `TaskConfig`, `AnalysisConfig`
  frozen dataclass structure; `n_trials_per_set`, `n_trials_total` properties
- `src/prl_hgf/env/simulator.py` — `generate_session` — trial structure is
  fully driven by config
- `src/prl_hgf/power/config.py` — `make_power_config` scope (n_per_group +
  omega_2_deltas only, no task modifications)
- `configs/prl_analysis.yaml` — current values: 4 phases × 30 trials + 20
  transfer × 3 sets = 420 total; 2 stable + 2 volatile phases; recovery_r_threshold=0.7
- Runtime verification: `dataclasses.replace` bottom-up on phases produces
  valid configs; scaling preserves 1:1 stable/volatile ratio (tested at
  scale factors 0.33, 0.50, 0.67, 1.0)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all from existing codebase, nothing new
- Architecture (new make_trial_config pattern): HIGH — verified runtime behavior
- Eligibility table logic: HIGH — derived from locked decisions in CLAUDE.md
- Pitfalls: HIGH — derived from direct code reading, not inference
- Compute time estimates: MEDIUM — rough estimates from code inspection;
  actual times depend on hardware and JAX JIT cache state

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable codebase; no external dependencies to expire)
