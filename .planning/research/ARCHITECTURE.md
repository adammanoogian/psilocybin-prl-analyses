# Architecture: BFDA Power Analysis Integration

**Domain:** Sequential Bayesian power analysis layered on an existing HGF
simulation–fit pipeline
**Researched:** 2026-04-07
**Overall confidence:** HIGH — findings are based on direct code reading of all
existing modules, not inference or documentation alone.

---

## Existing Architecture (Verified)

```
src/prl_hgf/
  env/           task_config.py (AnalysisConfig dataclass hierarchy)
                 simulator.py   (generate_session, Trial)
  models/        hgf_2level.py, hgf_3level.py, response.py
  simulation/    agent.py       (sample_participant_params, simulate_agent)
                 batch.py       (simulate_batch — takes AnalysisConfig)
  fitting/       single.py      (fit_participant)
                 batch.py       (fit_batch — takes pd.DataFrame)
                 ops.py, models.py
  analysis/      group.py       (build_estimates_wide, fit_group_model,
                                 extract_posterior_contrasts)
                 bms.py         (run_stratified_bms)
                 recovery.py    (build_recovery_df)
                 effect_sizes.py (compute_effect_sizes_table)
                 phase_stratification.py, plots.py, group_plots.py
  gui/           explorer.py
scripts/
  03_simulate_participants.py
  04_fit_participants.py
  05_run_validation.py
  06_group_analysis.py
configs/prl_analysis.yaml      (single source of truth)
config.py                      (Path constants)
```

Key signatures verified from source:

- `simulate_batch(config: AnalysisConfig, output_path=None) -> pd.DataFrame`
  Uses `config.simulation.n_participants_per_group` and
  `config.simulation.groups[name]` (GroupConfig with per-parameter mean/sd)
  and `config.simulation.session_deltas[name]` (SessionConfig with deltas).
  The `master_seed` drives all RNG; seeds are drawn upfront so adding
  participants does not disturb earlier seeds.

- `fit_batch(sim_df, model_name, n_chains, n_draws, n_tune, target_accept,
  random_seed, cores, output_path, ...) -> pd.DataFrame`
  Operates entirely on a DataFrame; no direct config dependency at call time.
  Outputs long-form rows: (participant_id, group, session, model, parameter,
  mean, sd, hdi_3%, hdi_97%, r_hat, ess, flagged).

- `build_estimates_wide(fit_df, model, exclude_flagged) -> pd.DataFrame`
  Pivots long-form fit results to one row per (participant, session) with
  parameter means as columns.

- `AnalysisConfig` is a frozen dataclass; it cannot be mutated. Its
  `SimulationConfig` subfield holds N-per-group and group distributions.

---

## What BFDA Needs vs What Exists

The power analysis loop is:

```
for (N, effect_size_d, iteration):
    config' = config with N and effect-size-modified group means
    sim_df  = simulate_batch(config')
    fit_df  = fit_batch(sim_df)
    wide    = build_estimates_wide(fit_df)
    BF      = compute_jzs_bf(wide, contrast="group:session_interaction")
    record(N, d, iteration, BF)
```

Gap analysis:

| Need | Exists | Gap |
|------|--------|-----|
| Varying N without editing YAML | No — N is baked into AnalysisConfig | Need `override_config()` helper |
| Varying effect sizes without editing YAML | No — group means baked in | Same helper |
| JZS Bayes factor for interaction contrast | No | New module: `power/bayes_factor.py` |
| BFDA stopping-rule evaluation | No | New module: `power/stopping_rule.py` |
| Single-iteration orchestrator callable | No | New module: `power/iteration.py` |
| SLURM array job script | No | New script: `scripts/08_power_slurm.sh` |
| Per-cell result recording (CSV rows) | No | Part of `power/iteration.py` |
| Aggregation across job array outputs | No | New script: `scripts/09_aggregate_power.py` |
| Power curve generation | No | New module: `power/curves.py` |

---

## Parameterization Strategy (Question 2)

### Recommendation: Runtime config override, never touch the YAML

`AnalysisConfig` is a frozen dataclass. The cleanest approach is a factory
function in the new `power/` package that constructs a modified
`AnalysisConfig` from the base YAML config plus runtime override scalars:

```python
# src/prl_hgf/power/config_override.py

def make_power_config(
    base_config: AnalysisConfig,
    n_per_group: int,
    effect_size_delta: dict[str, float],  # e.g. {"omega_2": 1.5}
    master_seed: int,
) -> AnalysisConfig:
    ...
```

This function creates new `GroupConfig`, `SessionConfig`, and
`SimulationConfig` objects (all frozen dataclasses accept keyword args) with
the substituted values, then wraps them in a new `AnalysisConfig`.

The effect size parameterization maps directly onto the existing
`session_deltas` structure. For a BFDA study of the group × session
interaction on omega_2, `effect_size_delta["omega_2"]` replaces
`psilocybin.session_deltas.omega_2_deltas[1]` (the post-dose delta). The
placebo group deltas are held at their YAML defaults (near zero), and the
psilocybin group deltas are set to `baseline_delta + d * pooled_sd` where
pooled_sd is estimated from the YAML group distributions.

Concretely, for a Cohen's d target:

```
psilocybin_post_dose_delta = d * pooled_sd(omega_2)
```

`pooled_sd` is `sqrt((sd_psilocybin² + sd_placebo²) / 2)`.

This keeps the YAML untouched and makes each job fully self-contained.

---

## New Module Layout (Question 1)

Recommended location: `src/prl_hgf/power/`

```
src/prl_hgf/power/
  __init__.py
  config_override.py   # make_power_config() factory
  bayes_factor.py      # compute_jzs_bf() — JZS BF for group×session contrast
  stopping_rule.py     # evaluate_bfda_stopping() — BF thresholds, sequential logic
  iteration.py         # run_power_iteration() — one (N, d, k) cell
  curves.py            # aggregate_power_results(), plot_power_curves()
```

### Component Responsibilities

| Component | Responsibility | Calls Into |
|-----------|---------------|-----------|
| `config_override.py` | Build modified AnalysisConfig without touching YAML | `env.task_config` dataclasses |
| `bayes_factor.py` | JZS BF on group-difference posterior draws | `analysis.group.extract_posterior_contrasts` |
| `stopping_rule.py` | Evaluate BF against H1/H0 thresholds (e.g. BF>10, BF<0.1) | `bayes_factor.py` |
| `iteration.py` | Orchestrate one (N, d, k) simulation+fit+BF cycle | `simulation.batch`, `fitting.batch`, `analysis.group`, `bayes_factor.py` |
| `curves.py` | Load aggregated CSVs, compute power at each cell, plot | pandas, matplotlib |

### New Scripts

```
scripts/
  07_power_single_iteration.py   # wrapper: reads CLI args, calls run_power_iteration()
  08_power_slurm_array.sh        # SLURM array job submitter
  09_aggregate_power.py          # gathers per-job CSVs, produces master results table
  10_plot_power_curves.py        # calls power/curves.py
```

This naming respects the existing `03_–06_` pipeline numbering and keeps
scripts callable independently (useful for local smoke tests before cluster
submission).

---

## SLURM Array Job Pattern (Question 3)

### Job Decomposition

Each SLURM array task corresponds to one `(N, d, k)` triple, where:
- N is sample size per group
- d is target Cohen's d for the primary contrast (omega_2 post-dose)
- k is the iteration index (0 to K-1)

This fully parallelizes across all three dimensions. At K=200, 7 N levels,
3 d levels: 4,200 total jobs. Each job does N_per_group × 2 groups × 3
sessions = up to 300 MCMC fits (at N=50). Wall time per job at N=50: ~2.5h
(300 fits × 30s).

### Array Index Encoding

A job index file (CSV or text) lists one `(N, d, k)` triple per line. The
SLURM array reads its row using `$SLURM_ARRAY_TASK_ID`:

```bash
# scripts/08_power_slurm_array.sh

#SBATCH --array=0-4199
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1    # JAX MCMC uses 1 core per chain internally

PARAMS_FILE="data/power/job_params.csv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PARAMS_FILE")
N=$(echo "$LINE" | cut -d, -f1)
D=$(echo "$LINE" | cut -d, -f2)
K=$(echo "$LINE" | cut -d, -f3)

python scripts/07_power_single_iteration.py \
    --n-per-group "$N" \
    --effect-size "$D" \
    --iteration "$K" \
    --output-dir "data/power/results/"
```

`job_params.csv` is generated once by a helper script:
`scripts/07a_generate_job_params.py`.

### Output Per Job

Each job writes a single-row CSV:

```
data/power/results/N{N}_d{D}_k{K:04d}.csv
```

Columns: `n_per_group, effect_size_d, iteration, bf_omega2, bf_kappa,
decision, n_flagged_fits, wall_time_s`

Single-row-per-job output avoids write conflicts entirely (no locking
needed). Jobs that fail write nothing; the aggregation script detects gaps.

---

## Results Aggregation (Questions 4 and 5)

### Data Flow from Jobs to Power Curves

```
SLURM array (4,200 jobs)
  -> data/power/results/N{N}_d{d}_k{k:04d}.csv   [one file per job]
       |
       v
scripts/09_aggregate_power.py
  -> pd.concat() all CSVs
  -> data/power/power_master.csv
       |  columns: n_per_group, effect_size_d, iteration, bf_omega2,
       |           bf_kappa, decision, n_flagged_fits
       v
scripts/10_plot_power_curves.py  (calls power/curves.py)
  -> power(N, d) = mean(decision == "H1") across iterations
  -> figures/power_curves.png
  -> data/power/power_summary.csv
       |  columns: n_per_group, effect_size_d, power_omega2, power_kappa,
       |           median_bf, n_iterations
```

### Aggregation Script Logic

```python
# scripts/09_aggregate_power.py
results_dir = DATA_DIR / "power" / "results"
csvs = sorted(results_dir.glob("N*_d*_k*.csv"))
master = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
master.to_csv(DATA_DIR / "power" / "power_master.csv", index=False)

# Flag missing jobs
expected = load_job_params()  # reads data/power/job_params.csv
completed = set(zip(master.n_per_group, master.effect_size_d, master.iteration))
missing = [(n, d, k) for n, d, k in expected if (n, d, k) not in completed]
if missing:
    print(f"WARNING: {len(missing)} jobs not yet completed")
```

This pattern is safe to run incrementally (re-run after partial completion to
check progress).

---

## Data Flow Diagram

```
configs/prl_analysis.yaml
        |
        | load_config()
        v
AnalysisConfig (base)
        |
        | make_power_config(N, d, seed)    [power/config_override.py]
        v
AnalysisConfig (power variant)
        |
        | simulate_batch()                 [simulation/batch.py — UNCHANGED]
        v
sim_df  (trial-level, N*2*3 participants)
        |
        | fit_batch()                      [fitting/batch.py — UNCHANGED]
        v
fit_df  (long-form: participant x session x parameter)
        |
        | build_estimates_wide()           [analysis/group.py — UNCHANGED]
        v
estimates_wide  (one row per participant-session)
        |
        | fit_group_model() + extract_posterior_contrasts()
        |                                  [analysis/group.py — UNCHANGED]
        v
posterior contrast samples (omega_2 group x session interaction)
        |
        | compute_jzs_bf()                 [power/bayes_factor.py — NEW]
        v
BF scalar
        |
        | evaluate_bfda_stopping()         [power/stopping_rule.py — NEW]
        v
decision: "H1" | "H0" | "inconclusive"
        |
        | write N{N}_d{d}_k{k}.csv        [iteration.py — NEW]
        v
data/power/results/  (4,200 files)
        |
        | 09_aggregate_power.py            [NEW script]
        v
data/power/power_master.csv
        |
        | 10_plot_power_curves.py          [NEW script]
        v
figures/power_curves.png
data/power/power_summary.csv
```

---

## Integration Points with Existing Components

### Unchanged (reused as-is)

| Module | How Power Analysis Uses It |
|--------|---------------------------|
| `env.task_config.load_config()` | Loads base config; power module wraps it |
| `env.task_config` dataclasses | `make_power_config()` constructs new frozen instances |
| `simulation.batch.simulate_batch()` | Called with power-variant config |
| `fitting.batch.fit_batch()` | Called with sim_df; all MCMC params passed as kwargs |
| `analysis.group.build_estimates_wide()` | Extracts posterior means |
| `analysis.group.fit_group_model()` | Fits bambi model for interaction |
| `analysis.group.extract_posterior_contrasts()` | Gets posterior draws for BF computation |
| `config.DATA_DIR` | Output paths hang off `DATA_DIR / "power" / ...` |

### Modified

None. The design deliberately avoids modifying any existing module. All new
functionality is additive.

### New Additions

| Path | Type | Purpose |
|------|------|---------|
| `src/prl_hgf/power/__init__.py` | Package | Exports `run_power_iteration`, `make_power_config` |
| `src/prl_hgf/power/config_override.py` | Module | Config factory without YAML mutation |
| `src/prl_hgf/power/bayes_factor.py` | Module | JZS BF from posterior draws |
| `src/prl_hgf/power/stopping_rule.py` | Module | BFDA decision rule evaluation |
| `src/prl_hgf/power/iteration.py` | Module | Single (N, d, k) orchestrator |
| `src/prl_hgf/power/curves.py` | Module | Power curve aggregation and plotting |
| `scripts/07a_generate_job_params.py` | Script | Write job_params.csv for SLURM |
| `scripts/07_power_single_iteration.py` | Script | CLI entry point for one job |
| `scripts/08_power_slurm_array.sh` | Bash | SLURM array job submitter |
| `scripts/09_aggregate_power.py` | Script | Concat per-job CSVs |
| `scripts/10_plot_power_curves.py` | Script | Power curve figures |
| `data/power/job_params.csv` | Data | (N, d, k) grid; generated, not committed |
| `data/power/results/` | Dir | Per-job output CSVs |
| `data/power/power_master.csv` | Data | Aggregated results |

---

## Key Design Decisions and Rationale

### Decision 1: Config override via factory, not YAML mutation

The existing `AnalysisConfig` is frozen. Mutating the YAML between jobs is
unsafe on a shared filesystem and impossible when 4,200 jobs run concurrently.
The factory function `make_power_config()` builds a fresh immutable config per
job. This is zero-risk and requires no changes to any existing module.

### Decision 2: One file per SLURM job, no shared writer

Writing to a shared CSV from thousands of concurrent jobs risks corruption
without a lock or database. One-file-per-job eliminates this entirely. The
aggregation step (sequential, after all jobs complete) is trivial with
`pd.concat`.

### Decision 3: Pass bambi model through analysis.group, not a custom path

`fit_group_model()` already returns an `az.InferenceData` from bambi.
`extract_posterior_contrasts()` already extracts posterior draws for the
group × session interaction. `compute_jzs_bf()` takes those draws and
computes the JZS BF. This avoids duplicating the bambi model fitting logic
and reuses the validated contrast-extraction code.

### Decision 4: SLURM array index = flat row in job_params.csv

Encoding (N, d, k) into the array index avoids off-by-one errors from
arithmetic index unpacking. The job reads its own parameters from a file
using the array task ID. This makes the mapping inspectable and allows
partial re-submission of failed jobs by rebuilding job_params.csv with only
the missing triples.

### Decision 5: bambi group model inside the power loop

`fit_group_model()` runs a bambi MCMC per iteration. At N=50, this adds
~5-10 min per job (bambi is cheaper than HGF fitting). This is acceptable.
If it becomes a bottleneck, the bambi step can be skipped in favor of a
simpler frequentist t-test BF approximation in `bayes_factor.py` (document
this as a future optimization flag).

---

## Suggested Build Order

The build order respects dependency chains. Each step can be tested before
the next is started.

**Step 1: `power/config_override.py`**
Lowest risk. No external dependencies beyond existing dataclasses. Write unit
tests verifying that N and d overrides produce the expected
`n_participants_per_group` and `omega_2_deltas`. This unblocks all subsequent
steps.

**Step 2: `power/bayes_factor.py`**
Requires understanding of JZS BF formula applied to posterior draws. Depends
on `analysis.group.extract_posterior_contrasts` output format, which is
verified. Can be unit-tested with synthetic posterior draw arrays before any
HGF fitting is done.

**Step 3: `power/stopping_rule.py`**
Simple threshold logic. Depends on BF scalar from Step 2. Trivial to test
in isolation.

**Step 4: `power/iteration.py` + `scripts/07_power_single_iteration.py`**
Integrates all prior steps. This is the first end-to-end test: run one
iteration locally with N=5 and fast MCMC settings to verify the pipeline
runs without errors. Use `--n-draws 50 --n-tune 50` flags.

**Step 5: `scripts/07a_generate_job_params.py`**
Generates the job grid. Can be reviewed before any cluster submission.

**Step 6: `scripts/08_power_slurm_array.sh`**
Submit 10 test jobs (small array) before full submission. Verify output file
naming and CSV schema.

**Step 7: `scripts/09_aggregate_power.py` + `power/curves.py` +
`scripts/10_plot_power_curves.py`**
Can be developed and tested against the 10 test-job outputs before the full
run completes.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mutating the YAML config between jobs

What goes wrong: concurrent SLURM jobs read/write the same file; race
conditions produce corrupted configs or wrong N values for some jobs.
What to do instead: `make_power_config()` factory — each job constructs its
own config in memory.

### Anti-Pattern 2: Passing N and d through environment variables

What goes wrong: SLURM environment variable injection is fragile and not
reproducible outside the cluster. What to do instead: explicit CLI arguments
to `07_power_single_iteration.py` parsed with `argparse`. The SLURM script
reads the job params CSV and passes them as arguments.

### Anti-Pattern 3: Running bambi with 4 chains per job on SLURM

What goes wrong: if `cores=4` is set for the bambi step and the SLURM job
requests 1 CPU, bambi will spawn 4 processes and oversubscribe the node.
What to do instead: keep `cores=1` (matching the existing pattern in
`04_fit_participants.py`) for HGF fitting; bambi's chains can run
sequentially or set `--cpus-per-task 4` in SLURM and pass `cores=4` only to
bambi, not to HGF fitting.

### Anti-Pattern 4: Reusing the same `master_seed` across iterations

What goes wrong: all K=200 iterations at a given (N, d) cell produce
identical simulated data, defeating the purpose of Monte Carlo power
estimation. What to do instead: derive `master_seed` from the iteration
index: `master_seed = base_seed + iteration * large_prime`. The
`make_power_config()` factory accepts `master_seed` as an explicit argument.

### Anti-Pattern 5: Writing results to a shared CSV with multiple writers

What goes wrong: file corruption, truncation, incomplete rows. What to do
instead: one file per job (described above), aggregated sequentially after
all jobs complete.

### Anti-Pattern 6: Running group-level bambi for every (N, d, k) cell with production-quality MCMC settings

What goes wrong: each iteration adds 5-10 min for bambi in addition to 2.5h
for HGF fitting. At 4,200 jobs this is tolerable, but with default `draws=2000,
tune=1000` the bambi portion could be the tail. What to do instead: use
reduced bambi settings inside the power loop (`draws=500, tune=500, chains=2`)
— sufficient for BF estimation from the group contrast posterior, not for
publication-quality inference. Document this as a separate MCMC budget from
the HGF fitting step.

---

## Scalability Notes

| Concern | At N=10/group | At N=50/group | At N=100/group |
|---------|--------------|--------------|---------------|
| Fits per iteration | 60 (2g×10p×3s) | 300 | 600 |
| Wall time per job | ~30 min | ~2.5h | ~5h |
| Total job-hours (4,200 jobs) | 2,100h | 10,500h | 21,000h |
| Memory per job | ~4 GB | ~8 GB | ~12 GB |

At N=10 the full grid runs in ~2,100 CPU-hours. Most clusters with 100+
cores can complete this in 24-48h. N=100 jobs should request 12h wall time
to be safe.

The JAX JIT pre-warm cost (~30s) is paid once per job, amortized across all
fits in that job. It is negligible for N>10.

---

## Sources

All findings based on direct source inspection of:

- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/simulation/batch.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/fitting/batch.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/analysis/group.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/analysis/bms.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/analysis/recovery.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/env/task_config.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/src/prl_hgf/simulation/agent.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/configs/prl_analysis.yaml`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/config.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/scripts/03_simulate_participants.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/scripts/04_fit_participants.py`
- `/c/Users/aman0087/Documents/Github/psilocybin_prl_analyses/scripts/06_group_analysis.py`

Confidence: HIGH. No inference required — all integration points confirmed
from actual function signatures and dataclass fields.
