# Phase 8: Config and Infrastructure - Research

**Researched:** 2026-04-07
**Domain:** Python frozen dataclasses, numpy SeedSequence, parquet I/O, SLURM array jobs
**Confidence:** HIGH

## Summary

Phase 8 wraps the existing pipeline (simulate_batch, fit_batch, build_estimates_wide)
with power-analysis infrastructure: a config factory that clones frozen dataclasses
without touching YAML, SeedSequence-based independent RNG for SLURM array tasks, and
tidy parquet output with a fixed schema.

All four research questions were answered from codebase inspection and stdlib docs.
The key insight is that `dataclasses.replace` is the correct, zero-boilerplate way to
produce modified copies of the deep-frozen `AnalysisConfig` hierarchy. SeedSequence
is already in numpy (no extra dependency); parquet requires adding `pyarrow` to
`environment_gpu.yml` and `pyproject.toml`. The SLURM array pattern follows the
existing templates almost verbatim.

**Primary recommendation:** Use `dataclasses.replace` at each layer of
`AnalysisConfig` → `SimulationConfig` → `SessionConfig` to produce a frozen override;
use `SeedSequence(master_seed).spawn(n_jobs)[SLURM_ARRAY_TASK_ID]` to generate the
per-job RNG; write each iteration to `results/power/<sweep_type>/<job_id>/iter_<N>.parquet`.

---

## Standard Stack

### Core (already in project environment)

| Library | Version constraint | Purpose | Why Standard |
|---------|-------------------|---------|--------------|
| `dataclasses` (stdlib) | Python 3.10+ | Frozen dataclass copy via `replace()` | Canonical approach, zero deps |
| `numpy.random.SeedSequence` | numpy>=2.0 (already pinned) | Guaranteed-independent child seeds | NumPy's own recommendation for parallel RNG |
| `numpy.random.default_rng` | numpy>=2.0 | Consume SeedSequence children | Modern Generator API |
| `pandas` | >=2.0 (already pinned) | DataFrame construction and parquet write | Already in stack |

### Must Add

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `pyarrow` | >=14.0 | pandas `.to_parquet()` / `.read_parquet()` back-end | Not in `environment_gpu.yml` or `pyproject.toml` — must add |

`fastparquet` is an alternative parquet engine but pyarrow is the standard choice
and is required anyway for pymc's ArviZ dependency chain; use pyarrow.

### Installation additions needed

In `environment_gpu.yml` under `pip:` block:
```
    - pyarrow>=14.0
```

In `pyproject.toml` dependencies list:
```
    "pyarrow>=14.0",
```

---

## Architecture Patterns

### Recommended Package Layout

```
src/prl_hgf/
└── power/
    ├── __init__.py          # public re-exports
    ├── config.py            # make_power_config()
    ├── seeds.py             # make_child_rng(master_seed, n_jobs, job_idx)
    ├── schema.py            # POWER_SCHEMA constant + validate_row()
    ├── runner.py            # run_one_iteration()
    └── io.py                # write_parquet_row()

cluster/
└── 08_power_sweep.slurm     # array job template

scripts/
└── 08_run_power_iteration.py  # entry point called by SLURM
```

### Pattern 1: Config Override via `dataclasses.replace`

**What:** Produce a frozen `AnalysisConfig` copy with `n_participants_per_group`
and session delta overrides applied, without reading or writing any YAML file.

**When to use:** Every call to `make_power_config(base_config, n_per_group, effect_size_delta, master_seed)`.

The `AnalysisConfig` hierarchy is three layers deep:
`AnalysisConfig.simulation` → `SimulationConfig`
`SimulationConfig.session_deltas` → `dict[str, SessionConfig]`
`SessionConfig.omega_2_deltas` → `list[float]`

`dataclasses.replace` works on any frozen dataclass and returns a new instance
without mutating the original. Nested replacements require explicit bottom-up
reconstruction.

```python
# Source: stdlib dataclasses.replace, verified locally
import dataclasses
from prl_hgf.env.task_config import (
    AnalysisConfig, SimulationConfig, SessionConfig
)

def make_power_config(
    base_config: AnalysisConfig,
    n_per_group: int,
    effect_size_delta: float,   # additive shift to psilocybin omega_2_deltas
    master_seed: int,
) -> AnalysisConfig:
    sim = base_config.simulation

    # Rebuild session_deltas with scaled effect
    new_deltas: dict[str, SessionConfig] = {}
    for group_name, sess_cfg in sim.session_deltas.items():
        if group_name == "psilocybin":
            new_omega_2_deltas = [d + effect_size_delta for d in sess_cfg.omega_2_deltas]
            new_sess = dataclasses.replace(
                sess_cfg, omega_2_deltas=new_omega_2_deltas
            )
        else:
            new_sess = sess_cfg  # placebo unchanged
        new_deltas[group_name] = new_sess

    new_sim = dataclasses.replace(
        sim,
        n_participants_per_group=n_per_group,
        master_seed=master_seed,
        session_deltas=new_deltas,
    )
    return dataclasses.replace(base_config, simulation=new_sim)
```

**Critical detail:** `SimulationConfig.__post_init__` validates that
`groups` and `session_deltas` have matching keys — both are preserved here.
`SessionConfig.__post_init__` validates that delta lists have length equal to
`session_labels` — only values change, not length.

### Pattern 2: SeedSequence for SLURM Array Independence

**What:** Generate one independent `numpy.random.Generator` per array task using
a fixed master seed.

**When to use:** At the top of the SLURM array task entry point script.

```python
# Source: numpy docs, verified with Python 3.10 / numpy 2.x locally
import numpy as np
import os

def make_child_rng(
    master_seed: int,
    n_jobs: int,
    job_index: int,
) -> np.random.Generator:
    """Return independent Generator for one SLURM array task."""
    children = np.random.SeedSequence(master_seed).spawn(n_jobs)
    return np.random.default_rng(children[job_index])
```

Usage in entry point script:
```python
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
rng = make_child_rng(master_seed=MASTER_SEED, n_jobs=N_JOBS, job_index=task_id)
```

**Key property:** `SeedSequence.spawn(n)` returns n children with provably
independent Philox BitGenerator streams. Each child's `entropy` equals the
parent's seed but with a different spawn path — streams are safe regardless
of how many draws each child takes.

**Critical detail:** `n_jobs` must be the same value every time to guarantee
that `children[task_id]` always corresponds to the same iteration index. Hard-code
it in the power config YAML section, not derived from squeue output.

### Pattern 3: Parquet Row Writes with Schema Enforcement

**What:** After each iteration, write a single-row parquet file using a fixed
column-order dict enforced by explicit pandas dtypes before writing.

**When to use:** At the end of `run_one_iteration()`.

```python
# Source: pandas 2.x docs, pyarrow engine
import pandas as pd
from pathlib import Path

POWER_SCHEMA: dict[str, str] = {
    "sweep_type":    "string",   # e.g. "n_per_group" or "effect_size"
    "effect_size":   "float64",
    "n_per_group":   "int64",
    "trial_count":   "int64",
    "iteration":     "int64",
    "parameter":     "string",   # e.g. "omega_2"
    "bf_value":      "float64",
    "bf_exceeds":    "bool",
    "bms_xp":        "float64",
    "bms_correct":   "bool",
    "recovery_r":    "float64",
    "n_divergences": "int64",
    "mean_rhat":     "float64",
}

def write_parquet_row(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    for col, dtype in POWER_SCHEMA.items():
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                f"Expected columns: {list(POWER_SCHEMA)}"
            )
        df[col] = df[col].astype(dtype)
    # Enforce column order
    df = df[list(POWER_SCHEMA)]
    df.to_parquet(output_path, index=False, engine="pyarrow")
```

**Output path convention** (no naming collisions):
```
results/power/<sweep_type>/job_<SLURM_ARRAY_JOB_ID>_task_<TASK_ID>/iter_<N>.parquet
```
Using both `$SLURM_ARRAY_JOB_ID` and `$SLURM_ARRAY_TASK_ID` in the path ensures
no collision even if the array is resubmitted.

### Pattern 4: SLURM Array Template

Based on existing `cluster/04_fit_mcmc_gpu.slurm` pattern:

```bash
#!/bin/bash
#SBATCH --job-name=prl_power
#SBATCH --output=cluster/logs/power_%A_%a.out
#SBATCH --error=cluster/logs/power_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%50        # %50 = max 50 concurrent

# %A = job array master ID, %a = task ID — prevents log collisions
```

The `%50` throttle is passed directly in the `--array` directive, consistent with
PWR-09. No additional flag needed.

### Anti-Patterns to Avoid

- **Direct field assignment on frozen dataclass:** `config.simulation.n_participants_per_group = 20` raises `FrozenInstanceError`. Always use `dataclasses.replace`.
- **Re-reading YAML inside `make_power_config`:** Defeats the purpose; the factory must take a base `AnalysisConfig` as input and return a new one without I/O.
- **Using `np.random.seed()` (legacy):** Global state, not safe for parallel jobs. Always use `SeedSequence` + `default_rng`.
- **Single parquet file shared across tasks:** Race condition on writes. Each task writes its own file; a separate aggregation step concatenates them.
- **Deriving `n_jobs` dynamically from environment:** `SeedSequence.spawn(n_jobs)` must be called with a fixed `n_jobs` constant, not inferred from `$SLURM_ARRAY_TASK_COUNT`. Store `n_jobs` in the power config YAML section.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Frozen config copies | Custom `copy()` method with manual field enumeration | `dataclasses.replace()` | Handles all fields automatically, type-safe, stdlib |
| Independent parallel RNG | Hash-based seed derivation | `np.random.SeedSequence.spawn()` | Provably independent streams, documented in NumPy |
| Parquet schema validation | Custom CSV with manual type checks | `pyarrow`-backed `pandas.to_parquet` with explicit dtype casting | Column types are enforced at write time, not on re-read |

---

## Common Pitfalls

### Pitfall 1: `dataclasses.replace` on dicts inside frozen dataclasses

**What goes wrong:** `SimulationConfig.session_deltas` is a `dict`. `dataclasses.replace`
passes it through as-is if you do not also rebuild it — you can accidentally share the
old dict reference. Since dicts are mutable, downstream code mutating the dict would
affect both old and new config.

**How to avoid:** Always construct a new dict explicitly:
```python
new_deltas = {k: (modified if k == "psilocybin" else v)
              for k, v in sim.session_deltas.items()}
new_sim = dataclasses.replace(sim, session_deltas=new_deltas)
```

**Warning signs:** Two configs that should differ on `session_deltas` print the same values.

### Pitfall 2: `SimulationConfig` validation catches mismatched keys

**What goes wrong:** `SimulationConfig.__post_init__` raises `ValueError` if
`set(groups) != set(session_deltas)`. If you rebuild `session_deltas` with a
misspelled group name or forget a group, the frozen constructor raises immediately.

**How to avoid:** Iterate over `sim.session_deltas.items()` to get the group names
from the existing config, not from a hard-coded list.

### Pitfall 3: SLURM array log collisions with naive `%j`

**What goes wrong:** `--output=logs/power_%j.out` uses the array master job ID,
meaning all 10 tasks write to the same log file.

**How to avoid:** Always use `%A_%a` in log file names: `power_%A_%a.out`.
The existing `04_fit_mcmc_gpu.slurm` uses `%j` (scalar job) — the array template
needs `%A_%a` instead.

### Pitfall 4: pyarrow not in prl_gpu environment

**What goes wrong:** `df.to_parquet(...)` raises `ImportError: Unable to find a
usable engine; tried using: 'pyarrow', 'fastparquet'`.

**How to avoid:** Add `pyarrow>=14.0` to both `environment_gpu.yml` (pip block)
and `pyproject.toml` before writing the first parquet file. Recreate the conda env
on the cluster (`mamba env update -f environment_gpu.yml --prune`).

### Pitfall 5: `SeedSequence` spawn count must be fixed

**What goes wrong:** If `n_jobs=10` the first run but later `n_jobs=20`, then
task index 5 now maps to a different child seed, breaking reproducibility.

**How to avoid:** Store `n_jobs` (total array size) as a field in the power config
YAML section and read it consistently. Never derive it from `$SLURM_ARRAY_TASK_COUNT`.

---

## Code Examples

### Constructing the `make_power_config` factory correctly

```python
# Verified pattern: dataclasses.replace on nested frozen hierarchy
import dataclasses
from prl_hgf.env.task_config import AnalysisConfig, SimulationConfig, SessionConfig

def make_power_config(
    base: AnalysisConfig,
    n_per_group: int,
    effect_size_delta: float,
    master_seed: int,
) -> AnalysisConfig:
    sim = base.simulation
    new_deltas = {}
    for group, sess in sim.session_deltas.items():
        if group == "psilocybin":
            sess = dataclasses.replace(
                sess,
                omega_2_deltas=[d + effect_size_delta for d in sess.omega_2_deltas],
            )
        new_deltas[group] = sess
    new_sim = dataclasses.replace(
        sim,
        n_participants_per_group=n_per_group,
        master_seed=master_seed,
        session_deltas=new_deltas,
    )
    return dataclasses.replace(base, simulation=new_sim)
```

### SeedSequence child RNG (entry point pattern)

```python
import os
import numpy as np

MASTER_SEED = 99999   # stored in power YAML section
N_JOBS = 10           # stored in power YAML section

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
children = np.random.SeedSequence(MASTER_SEED).spawn(N_JOBS)
rng = np.random.default_rng(children[task_id])
# Pass rng into simulate_batch via config.simulation.master_seed
# by calling rng.integers(0, 2**31) to derive a per-iteration seed
iteration_seed = int(rng.integers(0, 2**31))
```

### YAML power section addition (what to add to `prl_analysis.yaml`)

```yaml
# ---------------------------------------------------------------------------
# Power analysis parameters
# ---------------------------------------------------------------------------
power:
  # Grid of n_per_group values to sweep
  n_per_group_grid: [10, 15, 20, 25, 30, 40, 50]
  # Grid of effect size deltas (additive shift to psilocybin omega_2_deltas)
  effect_size_grid: [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
  # Number of Monte Carlo iterations per (n, d) combination
  n_iterations: 100
  # Master seed for SeedSequence — fixed for reproducibility
  master_seed: 20240101
  # Total SLURM array size (must match --array=0-<n_jobs-1>)
  n_jobs: 100
  # Bayes factor threshold for "sufficient evidence"
  bf_threshold: 10.0
```

### Downstream pipeline function signatures (verified from codebase)

`simulate_batch(config: AnalysisConfig, output_path: Path | None) -> pd.DataFrame`
- Reads `config.simulation.master_seed`, `config.simulation.n_participants_per_group`,
  `config.simulation.groups`, `config.simulation.session_deltas`.
- Returns tidy DataFrame with `true_omega_2`, `true_omega_3`, `true_kappa`, `true_beta`, `true_zeta` columns.

`fit_batch(sim_df, model_name, n_chains, n_draws, n_tune, target_accept, random_seed, cores) -> pd.DataFrame`
- Returns DataFrame with columns: `participant_id`, `group`, `session`, `model`,
  `parameter`, `mean`, `sd`, `hdi_3%`, `hdi_97%`, `r_hat`, `ess`, `flagged`.

`build_estimates_wide(fit_df, model, exclude_flagged) -> pd.DataFrame`
- Pivots to one row per (participant_id, group, session) with parameter means as columns.

---

## Effect Size → `session_deltas` Mapping

The `effect_size_delta` parameter in `make_power_config` is an **additive shift**
applied to `psilocybin.omega_2_deltas`. This is consistent with the existing
simulation design where omega_2_deltas are defined as additive shifts in the
natural (untransformed) parameter space.

Relationship to Cohen's d:
- The YAML records `psilocybin.omega_2.sd = 1.0` and `placebo.omega_2.sd = 1.0`.
- Pooled SD = sqrt((1.0^2 + 1.0^2) / 2) = 1.0.
- Therefore Cohen's d ≈ effect_size_delta / pooled_sd = effect_size_delta / 1.0.
- For a target d = 0.5: effect_size_delta = 0.5 × 1.0 = 0.5.

This means `effect_size_delta` equals Cohen's d numerically when both SDs = 1.0.
For other parameters (kappa, beta, zeta), the pooled SD differs — compute from
the YAML group distributions if those parameters are also swept.

---

## YAML Config Changes Required

The existing `prl_analysis.yaml` has no `power:` section. One must be added. The
analysis config loader (`load_config`) ignores unknown top-level keys (it only
parses `task`, `simulation`, `fitting`) — adding a `power:` block does NOT break
existing functionality.

A new `PowerConfig` dataclass and corresponding `load_power_config()` function
should be added to `task_config.py` (or a new `power_config.py` in `src/prl_hgf/power/`).
Given the requirement that power/ does not modify existing modules, create a separate
`src/prl_hgf/power/config.py` that loads the `power:` section independently via
`yaml.safe_load`.

---

## Open Questions

1. **Which parameter(s) are swept for effect size?**
   - What we know: Success criteria mention `effect_size_delta` as a single float applied to `omega_2_deltas`.
   - What's unclear: Whether `kappa_deltas` should also be scaled proportionally (kappa is a primary hypothesis parameter per CLAUDE.md).
   - Recommendation: Start with omega_2 only; add a `target_parameter` field to `make_power_config` if kappa sweeps are needed.

2. **Single parquet row per iteration vs. one row per parameter?**
   - What we know: The schema includes a `parameter` column, implying one row per (iteration × parameter).
   - What's unclear: Whether `bf_value`, `bms_xp`, etc. are computed per-parameter or once per iteration.
   - Recommendation: One row per (iteration × parameter), with BMS metrics (which are not parameter-specific) repeated across parameter rows for that iteration. This matches the tidy principle.

3. **n_divergences and mean_rhat scope**
   - What we know: `fit_batch` output has per-row `r_hat` and a `flagged` column.
   - What's unclear: Whether `n_divergences` is tracked per-participant or aggregated across the full batch. PyMC stores divergences in InferenceData but `fit_batch` does not return idata by default.
   - Recommendation: Use `fit_batch(..., return_idata=True)` and count divergences from the `sample_stats.diverging` array, or use `mean_rhat = fit_df.r_hat.mean()` and `n_divergences = fit_df.flagged.sum()` as a proxy.

---

## Sources

### Primary (HIGH confidence)
- Codebase: `src/prl_hgf/env/task_config.py` — full frozen dataclass hierarchy with `__post_init__` validators
- Codebase: `src/prl_hgf/simulation/batch.py` — `simulate_batch` exact signature and what it reads from config
- Codebase: `src/prl_hgf/fitting/batch.py` — `fit_batch` exact signature and output schema
- Codebase: `src/prl_hgf/analysis/group.py` — `build_estimates_wide` exact signature
- Codebase: `cluster/04_fit_mcmc_gpu.slurm` — established SLURM template pattern
- Codebase: `configs/prl_analysis.yaml` — existing config structure, group SD values
- stdlib `dataclasses.replace` — verified locally with nested frozen dataclasses
- `numpy.random.SeedSequence` — verified locally with numpy 2.x

### Secondary (MEDIUM confidence)
- `environment_gpu.yml` — confirmed pyarrow is absent, pandas>=2.0 is present
- `pyproject.toml` — confirmed pyarrow is not listed in dependencies

### Tertiary (LOW confidence)
- None — all findings are from direct codebase inspection or stdlib verification

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified from environment files and stdlib
- Architecture: HIGH — based on direct reading of all referenced modules
- Pitfalls: HIGH — derived from `__post_init__` validator inspection and verified dataclasses behavior
- Effect size mapping: MEDIUM — derived from YAML group SD values; exact formula needs domain confirmation

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (config structure is stable; changes only if Phase 7 API changes)
