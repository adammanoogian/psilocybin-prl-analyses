# Phase 14: Integration + GPU Benchmark + Decision Gate - Research

**Researched:** 2026-04-12
**Domain:** Python integration wiring, GPU benchmarking (nvidia-smi), JAX compilation cache, cross-platform MCMC consistency
**Confidence:** HIGH

## Summary

Phase 14 is primarily an integration and measurement phase. Both `fit_batch_hierarchical` (Phase 12) and `simulate_batch` via `simulate_cohort_jax` (Phase 13) are already implemented and tested. The work is:

1. Wire those into `run_sbf_iteration` (replacing the legacy `simulate_batch` + `fit_batch` calls) behind a `--legacy` flag that keeps the old path working.
2. Update `_run_benchmark` in `08_run_power_iteration.py` to time ONE FULL SBF iteration (300 participant-sessions × 2 models) rather than the current single-participant benchmark, and add periodic nvidia-smi sampling for GPU utilization, writing to `benchmark_batched.json`.
3. Implement the decision gate (50 GPU-hours threshold) and record in the JSON and STATE.md.
4. Verify JAX compilation cache persistence across chunks (BENCH-05).
5. Write VALID-03 (CPU vs GPU posterior mean agreement within 1%) and VALID-05 (legacy path smoke test).

The central tension is: `run_sbf_iteration` currently uses `fit_batch` (the legacy v1.1 sequential path — imported from the `fitting.batch` shim which delegates to `fitting.legacy.batch`). The new `fit_batch_hierarchical` in `fitting.hierarchical` returns `arviz.InferenceData` with a `participant` dimension, not a `pd.DataFrame` like `fit_batch`. This means `run_sbf_iteration` needs new code to extract per-participant posterior means from `InferenceData` into the DataFrame format the downstream BF contrasts and recovery metrics expect. That extraction logic is the primary new engineering in this phase.

**Primary recommendation:** Add a `use_legacy` parameter (default `False`) to `run_sbf_iteration` that switches between the batched and legacy paths. Add `--legacy` to `08_run_power_iteration.py` which passes `use_legacy=True`. Keep the public signature of `run_sbf_iteration` backward compatible.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| jax | >=0.4.26,<0.4.32 | JAX_COMPILATION_CACHE_DIR env var for cache persistence | Project constraint |
| pymc.sampling.jax (pmjax) | project constraint | `sample_numpyro_nuts` for batched MCMC | Locked decision: avoids _init_jitter bug |
| arviz | project constraint | InferenceData parsing after `fit_batch_hierarchical` | Already used in hierarchical.py |
| subprocess | stdlib | nvidia-smi queries for VRAM and utilization | Already used in _run_benchmark |
| threading | stdlib | Background periodic nvidia-smi sampling during fit | Standard pattern for non-blocking polling |
| json | stdlib | Benchmark output serialization | Already used in _run_benchmark |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2.0.0,<3.0 | Posterior extraction from InferenceData arrays | After idata returned from fit_batch_hierarchical |
| time.perf_counter | stdlib | Wall-clock timing of full iteration | Already used throughout _run_benchmark |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| threading + periodic nvidia-smi | pynvml (Python NVML bindings) | pynvml is not installed; nvidia-smi is universally available on cluster nodes; no new dependency needed |
| threading + periodic nvidia-smi | JAX device_put + memory stats | JAX memory stats track allocated bytes, not peak VRAM as nvidia-smi sees it; nvidia-smi is the ground truth for VRAM headroom |
| subprocess nvidia-smi | gpustat Python lib | Not a project dependency; subprocess is simpler |

## Architecture Patterns

### Recommended Project Structure

No new files required. All changes are in existing files:

```
scripts/
└── 08_run_power_iteration.py   # --benchmark and --legacy flag changes
src/prl_hgf/power/
└── iteration.py                # run_sbf_iteration: batched path + use_legacy param
tests/
└── test_power_iteration.py     # VALID-05 legacy smoke test
validation/
└── valid03_cross_platform.py   # VALID-03 CPU vs GPU agreement
results/power/
└── benchmark_batched.json      # Written by --benchmark run
```

### Pattern 1: IID Posterior Extraction from `fit_batch_hierarchical`

**What:** `fit_batch_hierarchical` returns `arviz.InferenceData` where posterior arrays have shape `(chain, draw, participant)`. To feed downstream code (`compute_all_contrasts`, `_extract_diagnostics`, `_compute_waic_table`), the batched path must extract per-participant posterior means into the same DataFrame schema that legacy `fit_batch` produces.

**Schema from legacy `fit_batch`:**
```
participant_id, group, session, model, parameter, mean, sd, hdi_3%, hdi_97%, r_hat, ess, flagged
```

**Extraction pattern:**
```python
# Source: hierarchical.py fit_batch_hierarchical + iteration.py existing schema
import arviz as az
import numpy as np

def _idata_to_fit_df(
    idata: az.InferenceData,
    participant_ids: list[str],
    participant_groups: list[str],
    participant_sessions: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Convert batched InferenceData to legacy fit_df schema."""
    rows = []
    posterior = idata.posterior
    # summary() computes mean, sd, hdi, r_hat, ess across chains/draws
    summary = az.summary(idata, var_names=[...], hdi_prob=0.94)
    # summary index is like "omega_2[0]", "omega_2[1]", ...
    for param in var_names:
        for i, (pid, grp, sess) in enumerate(zip(...)):
            row_key = f"{param}[{i}]"
            rows.append({
                "participant_id": pid,
                "group": grp,
                "session": sess,
                "model": model_name,
                "parameter": param,
                "mean": summary.loc[row_key, "mean"],
                "sd": summary.loc[row_key, "sd"],
                "hdi_3%": summary.loc[row_key, "hdi_3%"],
                "hdi_97%": summary.loc[row_key, "hdi_97%"],
                "r_hat": summary.loc[row_key, "r_hat"],
                "ess": summary.loc[row_key, "ess_bulk"],
                "flagged": ...,
            })
    return pd.DataFrame(rows)
```

**Key implementation note:** The `idata.posterior` has a `participant` dimension with `participant_id` coords (set in `fit_batch_hierarchical`). The `az.summary()` index format for shaped variables is `paramname[i]` (0-based). Cross-reference with `participant_ids` list preserved during `fit_batch_hierarchical`'s array stacking.

**Caveat:** `fit_batch_hierarchical` groups by `(participant_id, group, session)` internally and assigns coords. The ordering must be recovered from `idata.posterior.coords["participant"]` to correctly align participant metadata. Do NOT assume ordering matches the input DataFrame ordering.

### Pattern 2: `--legacy` Flag in `run_sbf_iteration`

**What:** Add `use_legacy: bool = False` to `run_sbf_iteration`. When `True`, delegate to the existing `simulate_batch` + `fit_batch` (legacy) calls unchanged. When `False`, call `simulate_batch` (already uses JAX vmap internally since Phase 13) + `fit_batch_hierarchical` for both models.

```python
def run_sbf_iteration(
    ...
    use_legacy: bool = False,
) -> list[dict]:
    ...
    if use_legacy:
        sim_df = simulate_batch(cfg)  # already JAX inside, but still same API
        fit_df_3, idata_3level = fit_batch(sim_df, "hgf_3level", ...)
        fit_df_2, idata_2level = fit_batch(sim_df, "hgf_2level", ...)
    else:
        sim_df = simulate_batch(cfg)  # same
        idata_3 = fit_batch_hierarchical(sim_df, "hgf_3level", ...)
        idata_2 = fit_batch_hierarchical(sim_df, "hgf_2level", ...)
        fit_df_3 = _idata_to_fit_df(idata_3, ...)
        fit_df_2 = _idata_to_fit_df(idata_2, ...)
        idata_3level = _idata_to_legacy_dict(idata_3, ...)  # for _compute_waic_table
        idata_2level = _idata_to_legacy_dict(idata_2, ...)
```

**Caveat:** `_compute_waic_table` and `_compute_bms_power` take `idata_3level: dict[tuple, object]` — a mapping of `(pid, grp, sess) -> InferenceData`. The batched path produces one joint `InferenceData` for ALL participants. A `_split_idata` helper is needed to slice the joint `InferenceData` into per-participant pieces for WAIC computation. This is the primary structural bridge.

### Pattern 3: Background GPU Utilization Sampling

**What:** During the fit call in `_run_benchmark`, spawn a background thread that polls `nvidia-smi` every N seconds. The thread stores peak VRAM and collects utilization samples. The main thread joins after the fit completes.

```python
import threading

class _GpuMonitor:
    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: list[dict] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    util, used, total = out.stdout.strip().split(", ")
                    self.samples.append({
                        "gpu_util_pct": float(util),
                        "vram_used_mb": float(used),
                        "vram_total_mb": float(total),
                    })
            except Exception:
                pass

    @property
    def peak_vram_mb(self) -> float:
        return max((s["vram_used_mb"] for s in self.samples), default=0.0)

    @property
    def mean_gpu_util_pct(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s["gpu_util_pct"] for s in self.samples]))
```

This pattern is LOW overhead: one subprocess call every 2 seconds during a multi-minute fit.

### Pattern 4: JAX Compilation Cache (`JAX_COMPILATION_CACHE_DIR`)

**What:** JAX (>=0.4.1) supports persistent compilation cache via `JAX_COMPILATION_CACHE_DIR` environment variable. When set, XLA compiled kernels are stored to disk and reloaded on subsequent runs with matching hardware + XLA version. BENCH-05 requires verifying that Chunk 1 and Chunk 2 show < 5 s JIT time vs ~60 s cold.

**How to verify:**
- Set `JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache` in SLURM job script (already the standard recommendation).
- Time the first `jax.jit`-compiled call in each chunk.
- In the benchmark JSON, record `jit_cold_s` (Chunk 0) vs `jit_warm_s` (Chunk 1+).
- The benchmark mode can simulate this by calling the compiled function twice and comparing timings.

**Important:** The compilation cache is keyed on XLA flags + hardware + JAX version. On homogeneous cluster nodes (same GPU model), cache hits are reliable. Cache misses occur if node GPU model differs between chunks.

**Configuration:** On the M3 cluster (from prior STATE.md), set in SLURM script:
```bash
export JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
```

### Pattern 5: Decision Gate Logic

**What:** Per BENCH-02: if `(per_iter_seconds × 600 / 3600) > 50`, recommend CPU `comp` partition.

`per_iter_seconds` = wall time for one `run_sbf_iteration` call (300 participant-sessions × 2 models).
`600` = iterations per chunk (200 task_ids × 3 N-subsamples, but SBF fits once — actual is 200 MCMC fits × 2 models per chunk).
`3600` = seconds per hour.
`50` = GPU-hour budget per chunk.

**Clarification needed:** The phase description says "300 participant-sessions × 2 models" for one iteration. For the benchmark, one full iteration = one `run_sbf_iteration` call at max N (50 per group × 2 groups × 3 sessions = 300 participant-sessions) × 2 models. The threshold formula as written uses `600` which equals `n_iterations=200` × 3 (effect sizes) / 1 chunk — i.e., tasks per chunk. This should be 200 (iterations per chunk, since there's 1 effect size per chunk). **Use the formula exactly as specified in the requirements:** `per_iter_seconds × 600 / 3600 > 50`.

### Anti-Patterns to Avoid

- **Calling `az.summary()` on the full joint InferenceData for all participants at once:** This is fine for small cohorts but at P=300 it produces a very wide summary DataFrame. Prefer extracting posterior arrays directly via `idata.posterior[param_name].values` (shape: chain × draw × participant) then computing mean/std/r_hat/ess per participant using numpy + arviz utilities.
- **Polling nvidia-smi too frequently:** Every 0.1 s would add noise and subprocess overhead. Use 2-second intervals.
- **Assuming `idata.posterior.coords["participant"]` order matches the DataFrame `groupby` order:** Always re-derive the participant ordering from the coords, not from assumptions.
- **Forgetting that `_compute_waic_table` needs `dict[tuple, InferenceData]` not a list:** The WAIC computation expects per-(pid, grp, sess) `InferenceData`. A helper `_split_idata` must slice the batched `InferenceData` along the `participant` dimension to produce per-participant `InferenceData` objects for WAIC.
- **Using `pm.sample()` for the batched path:** Locked decision — must use `pmjax.sample_numpyro_nuts()` due to `_init_jitter` bug.
- **Running the full benchmark on a CPU-only dev machine:** The benchmark will silently fall back to CPU; the decision gate will likely flag GPU as too slow. The benchmark is designed for SLURM GPU nodes only. Add a clear warning when `jax.devices()` has no GPU.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-participant posterior stats from InferenceData | Manual chain/draw loop + stats | `az.summary()` or direct `.values` + `np.mean/std` | ArviZ already handles r_hat, ess_bulk, hdi correctly |
| GPU memory query | Custom NVML Python binding | `subprocess` nvidia-smi | Already in codebase; no new dependency |
| Compilation timing | Custom JAX hook | `time.perf_counter()` before/after first JIT call | Sufficient for the < 5 s warm / ~60 s cold comparison |
| Cross-platform consistency test | Custom MCMC bridge | Same test code, different `JAX_PLATFORM_NAME` env var | CPU path: `JAX_PLATFORM_NAME=cpu`; GPU path: normal; compare posterior means |

**Key insight:** All the hard pieces (batched logp, vmapped simulation, MCMC sampling, InferenceData) are done. Phase 14 is plumbing + measurement, not new algorithms.

## Common Pitfalls

### Pitfall 1: `_idata_to_fit_df` Index Alignment

**What goes wrong:** `az.summary()` uses `varname[index]` indexing. The index is 0-based and corresponds to the order participants were added during `build_pymc_model_batched` → `fit_batch_hierarchical`. If the caller passes `sim_df` with participants in a different order than the groupby-order inside `fit_batch_hierarchical`, the participant ID-to-posterior mapping breaks.

**Why it happens:** `fit_batch_hierarchical` internally calls `sim_df.groupby(["participant_id", "group", "session"], sort=False)` which uses insertion order. The ordering of `sim_df` depends on `simulate_batch`, which processes groups/participants in alphabetical group order → sequential participant index → session index.

**How to avoid:** The `idata.posterior.coords["participant"]` list is the ground truth for participant ordering. Always build the reverse mapping from participant coords, not from the input DataFrame order.

**Warning signs:** Recovery metrics or BF contrast results that are clearly wrong (e.g., a placebo participant has psilocybin-group parameters).

### Pitfall 2: `_compute_waic_table` Expects Per-Participant `InferenceData`

**What goes wrong:** `_compute_waic_table` calls `compute_subject_waic(input_arr, obs_arr, choices_arr, idata, model_name)` for each `(pid, grp, sess)` key. It expects `idata` to be a single-participant `InferenceData`. Passing the full batched `InferenceData` will cause shape errors or silently compute WAIC across all participants.

**Why it happens:** The legacy `fit_batch` returns `idata_3level: dict[tuple, InferenceData]` where each InferenceData has a single participant. The batched path returns one joint InferenceData with P participants.

**How to avoid:** Write `_split_idata(idata, participant_idx) -> InferenceData` that slices `.posterior[var][..., participant_idx]` and creates a new `InferenceData` with that sliced posterior. Use `xarray.Dataset.isel(participant=i)` on the posterior group.

**Warning signs:** WAIC computation raises shape mismatches or returns suspiciously identical values across participants.

### Pitfall 3: JAX Compilation Cache May Not Persist Across SLURM Job Boundaries on All Clusters

**What goes wrong:** If the SLURM scratch filesystem is job-local (purged after job end), the compilation cache is lost between chunks. BENCH-05 assumes a persistent scratch directory that survives across the 3-chunk array.

**Why it happens:** Some cluster configurations use job-scoped scratch (`/scratch/job_$SLURM_JOB_ID/`) rather than user-scoped (`/scratch/$USER/`).

**How to avoid:** In the SLURM job script, use `JAX_COMPILATION_CACHE_DIR=$SCRATCH/$USER/jax_cache` where `$SCRATCH` is the persistent project allocation (on M3: `/scratch/ds_env` or similar). Verify with a two-call timing test in the benchmark: if second call takes < 5 s, cache is working.

**Warning signs:** Chunk 1 JIT time is still ~60 s (same as cold Chunk 0).

### Pitfall 4: Benchmark Timings on Wrong N

**What goes wrong:** BENCH-01 specifies "one full iteration (300 participant-sessions × 2 models)". The current `_run_benchmark` in `08_run_power_iteration.py` only fits ONE participant-session per model and extrapolates. The Phase 14 benchmark must fit the full cohort.

**Why it happens:** The old benchmark was a Phase 12 placeholder — it measured single-participant time as a proxy. Phase 14 requires measuring the actual batched fit time to know whether GPU amortization actually works.

**How to avoid:** In the new benchmark, call `run_sbf_iteration` (or the inner batched path directly) with `max_n = 50` (giving 300 participant-sessions) and time the full call. Replace the extrapolation projection with measured time.

**Warning signs:** Benchmark JSON says `"per_iteration_s": 10.0` but the extrapolation assumed 300× single-participant time.

### Pitfall 5: `fit_batch_hierarchical` Returns InferenceData, Not `(fit_df, idata_dict)`

**What goes wrong:** `run_sbf_iteration` currently does `fit_df_3, idata_3level = fit_batch(...)` expecting a tuple. After the switch to `fit_batch_hierarchical`, the return is a single `InferenceData`. Callers that destructure the tuple will fail.

**Why it happens:** The legacy `fit_batch` (v1.1) returns `(pd.DataFrame, dict[tuple, InferenceData])`. The new `fit_batch_hierarchical` returns `az.InferenceData` only.

**How to avoid:** The batched code path in `run_sbf_iteration` must explicitly call `_idata_to_fit_df` and `_split_idata` to reconstruct the downstream-expected data structures. These are new private helpers added to `iteration.py` (or a new `src/prl_hgf/power/_idata_utils.py` file).

## Code Examples

### Slicing batched InferenceData per participant
```python
# Source: xarray .isel() standard pattern
import arviz as az

def _split_idata(
    joint_idata: az.InferenceData,
    participant_idx: int,
) -> az.InferenceData:
    """Slice joint InferenceData to a single participant."""
    posterior_slice = joint_idata.posterior.isel(participant=participant_idx)
    return az.InferenceData(posterior=posterior_slice)
```

### Decision gate implementation
```python
# Source: Phase 14 BENCH-02 specification
def _apply_decision_gate(
    per_iter_seconds: float,
    n_iterations_per_chunk: int = 200,
) -> dict:
    """Apply BENCH-02 GPU-hour decision gate."""
    # Spec formula: per_iter_seconds × 600 / 3600 > 50
    gpu_hours_per_chunk = per_iter_seconds * 600 / 3600
    recommend_gpu = gpu_hours_per_chunk <= 50
    return {
        "per_iter_seconds": round(per_iter_seconds, 2),
        "gpu_hours_per_chunk": round(gpu_hours_per_chunk, 1),
        "decision": "gpu" if recommend_gpu else "cpu_comp",
        "decision_threshold_gpu_hours": 50,
    }
```

### VALID-03 cross-platform consistency test
```python
# Source: Phase 14 VALID-03 specification
import os
import numpy as np

def test_valid03_cpu_gpu_posterior_agreement():
    """CPU and GPU posterior means agree within 1% relative error."""
    # Both fits use same sim_df, same seed, same n_draws/n_chains
    # On dev machine: both run on CPU (JAX_PLATFORM_NAME=cpu is default)
    # On cluster: GPU fit uses default; CPU fit uses JAX_PLATFORM_NAME=cpu env
    # This test is run on both platforms and results compared post-hoc
    # (not in a single process, since you can't switch JAX platform mid-process)
    ...
    # Relative error: abs(cpu_mean - gpu_mean) / abs(cpu_mean + 1e-8) < 0.01
    assert np.abs(cpu_mean - gpu_mean) / (np.abs(cpu_mean) + 1e-8) < 0.01
```

**Implementation note on VALID-03:** You cannot switch `JAX_PLATFORM_NAME` within a single Python process. VALID-03 must either: (a) run the test twice — once with CPU env and once on GPU — and compare saved JSON outputs; or (b) mock the platform. The practical approach is a small script `validation/valid03_cross_platform.py` that saves posterior means to JSON, run twice (once on CPU, once on GPU node), then a comparison assertion. The planner should design VALID-03 as two separate runs + comparison, not a single pytest.

### `benchmark_batched.json` schema
```json
{
  "gpu_device": "NVIDIA L40S (or 'none (CPU only)')",
  "gpu_nvidia_smi": "L40S, 49152 MiB, 48000 MiB",
  "per_iteration_s": 142.5,
  "peak_vram_mb": 12800.0,
  "mean_gpu_util_pct": 87.3,
  "vram_total_mb": 49152.0,
  "jit_cold_s": 58.2,
  "jit_warm_s": 3.1,
  "decision": "gpu",
  "gpu_hours_per_chunk": 23.8,
  "decision_threshold_gpu_hours": 50,
  "benchmark_n_participant_sessions": 300,
  "mcmc_settings": {
    "chains": 2,
    "draws": 500,
    "tune": 500,
    "sampler": "numpyro"
  },
  "grid": {
    "n_per_group": [10, 15, 20, 25, 30, 40, 50],
    "max_n": 50,
    "effect_sizes": [0.3, 0.5, 0.7],
    "n_iterations": 200,
    "n_chunks": 3
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.1 `fit_batch`: sequential per-participant NUTS | v1.2 `fit_batch_hierarchical`: one joint NUTS call for P participants | Phase 12 | Amortises PCIe dispatch; GPU-feasible |
| `simulate_batch` with NumPy for-loop | `simulate_batch` with JAX vmap one-shot dispatch | Phase 13 | ~10x faster simulation for large cohorts |
| Old `_run_benchmark`: single participant, extrapolated | New `--benchmark`: full 300-session iteration, measured | Phase 14 | Ground truth timing for decision gate |

**Deprecated/outdated:**
- `fitting.batch` (shim): Still exported for legacy path, but new code should import from `fitting.hierarchical` and `fitting.legacy.batch` explicitly.
- `_prewarm_jit`: Used by the legacy `fit_batch` path. The batched path does not need explicit prewarming since the first `pmjax.sample_numpyro_nuts` call triggers JIT internally.

## Open Questions

1. **VALID-03 single-process vs two-run design**
   - What we know: JAX platform cannot be changed within a process (`jax.config.update("jax_platform_name", ...)` is a one-time global setting).
   - What's unclear: Whether to implement VALID-03 as (a) two shell-script invocations with JSON comparison, (b) a pytest that skips if no GPU, or (c) a dev-machine CPU-only approximation.
   - Recommendation: Planner should design VALID-03 as a comparison script (`validation/valid03_cross_platform.py`) that: reads two pre-saved JSON posterior summaries (one CPU, one GPU), and asserts 1% relative error on per-participant omega_2 means. The two JSONs are produced by running `08_run_power_iteration.py --benchmark` with different `JAX_PLATFORM_NAME` settings.

2. **`_compute_waic_table` compatibility with batched idata**
   - What we know: `compute_subject_waic` in `prl_hgf/analysis/bms.py` takes a single-participant `InferenceData`.
   - What's unclear: Whether `xarray.Dataset.isel(participant=i)` on a batched posterior produces an `InferenceData` that `compute_subject_waic` will accept without changes.
   - Recommendation: Write a unit test for `_split_idata` before integrating into `run_sbf_iteration`. If `compute_subject_waic` requires specific dimension names, adjust the slice accordingly.

3. **Decision gate formula — 600 vs 200**
   - What we know: The spec says `per_iter_seconds × 600 / 3600 > 50`. There are 200 iterations per chunk (one effect size per chunk). 600 = 3 × 200 (total iterations across all 3 chunks). The formula appears to measure total sweep GPU-hours across all chunks divided by the threshold.
   - What's unclear: Whether `600` is iterations-per-chunk (it's not — that's 200) or total iterations across all chunks.
   - Recommendation: Implement the formula exactly as specified (`× 600`). Document the formula's meaning in the benchmark JSON metadata field `decision_formula`. If the formula produces unexpected results on real GPU hardware, revisit — but match the spec exactly for BENCH-02.

4. **`simulate_batch` already uses JAX vmap (Phase 13)**
   - What we know: `simulate_batch` in Phase 13 was updated to use `_build_session_scanner` + `jax.vmap(_run_session)` internally. The function signature is unchanged.
   - What's unclear: Whether `run_sbf_iteration`'s batched path should also call `simulate_batch` (unchanged) or switch to `simulate_cohort_jax` directly.
   - Recommendation: Keep calling `simulate_batch` for both paths (it's already JAX-native). The distinction between batched and legacy paths is entirely in the MCMC step (`fit_batch_hierarchical` vs `fit_batch`), not in simulation.

## Sources

### Primary (HIGH confidence)
- Codebase: `src/prl_hgf/fitting/hierarchical.py` — `fit_batch_hierarchical` signature and return type (line 773-988), confirmed directly
- Codebase: `src/prl_hgf/power/iteration.py` — `run_sbf_iteration` current implementation, all downstream data consumers
- Codebase: `scripts/08_run_power_iteration.py` — existing `_run_benchmark` function, `--benchmark` flag, output format
- Codebase: `src/prl_hgf/simulation/batch.py` — confirmed already uses JAX vmap internally (Phase 13)
- Codebase: `.planning/STATE.md` — locked decisions: numpyro sampler, _init_jitter bug, decision gate threshold, factory pattern

### Secondary (MEDIUM confidence)
- Phase 13 RESEARCH.md — confirmed JAX PRNG threading and vmap patterns already validated in tests
- Phase 12 RESEARCH.md — confirmed `fit_batch_hierarchical` returns `az.InferenceData`

### Tertiary (LOW confidence)
- JAX compilation cache behavior across SLURM jobs: based on JAX documentation pattern; cluster-specific filesystem behavior not verified without access to actual cluster.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in codebase; no new dependencies
- Architecture (wiring): HIGH — both Phase 12 and 13 outputs confirmed in code; integration points clearly identified
- Architecture (idata extraction): MEDIUM — `az.summary()` index format and `xarray.Dataset.isel()` behavior assumed from training knowledge; needs verification in a quick test before writing `_split_idata`
- Architecture (nvidia-smi threading): HIGH — pattern already exists in `_run_benchmark`; threading extension is standard
- Decision gate: HIGH — formula exactly as specified in requirements
- VALID-03 design: MEDIUM — the two-run approach is the only feasible design given JAX single-platform-per-process constraint; exact file format TBD by planner
- Pitfalls: HIGH — all identified from direct code reading, not inference

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable; all dependencies pinned)
