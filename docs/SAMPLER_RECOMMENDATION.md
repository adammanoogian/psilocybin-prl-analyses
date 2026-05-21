# Sampler Recommendation Decision Tree

Quick-reference guide for choosing the right `FitConfig` for HGF model
fitting. Follow the decision tree or jump to the recommendation table.
For per-cell diagnostic evidence (walltimes, R-hat, divergences), see
[`CAPABILITY_MAP.md`](CAPABILITY_MAP.md).

---

## Decision Tree

```
Start
  |
  +-- Do you need group-level hyperpriors (hierarchical pooling)?
  |     |
  |     +-- NO  --> Mode A (no-pooling)
  |     |             |
  |     |             +-- 2-level HGF?
  |     |             |     +-- YES --> (A1) none_2level.yaml
  |     |             |     +-- NO (3-level) --> (A2) m1_laplace_fp64_3level.yaml
  |     |             |
  |     |             +-- Quick exploration only (no UQ needed)?
  |     |                   +-- YES --> (A3) laplace_only.yaml
  |     |
  |     +-- YES --> Mode B (hierarchical)
  |                   |
  |                   +-- 2-level HGF?
  |                   |     +-- YES --> (B1) hier_m1_2level.yaml
  |                   |
  |                   +-- 3-level HGF?
  |                         +-- YES --> (B2) hier_m1_m2_laplace_cov_3level.yaml
  |
  +-- Unsure about model complexity?
        +-- Start with 2-level; only move to 3-level if BMS favors it.
```

---

## Recommendation Table

| ID | Mode | Model | P_total | Config YAML | Mitigations | Status / Evidence |
|----|------|-------|---------|-------------|-------------|-------------------|
| A1 | A | 2-level | any | `none_2level.yaml` | None (diagonal mass) | Pending benchmark -- Phase 31 2-level cells at P>=150 timed out (pre-cuda13 env); resubmission needed |
| A2 | A | 3-level | <=50 | `m1_laplace_fp64_3level.yaml` | Dense mass + Laplace warmup + fp64 | PASS at P=50 (233s, job 54902462); TIMEOUT at P>=60 with all mitigations (Phase 31) |
| A2 | A | 3-level | >50 | -- | -- | Known infeasible: all NUTS configs (diagonal and dense+Laplace+fp64) TIMEOUT at 24h for P>=60 (CAPABILITY_MAP rows). Consider Mode B or Laplace fast-path |
| A3 | any | 2-level | any | `laplace_only.yaml` | n/a (MAP only) | PASS at P=160 PAT-RL (47s, job 55139039); pick_best_cue Laplace: pending benchmark |
| A3 | any | 3-level | any | `laplace_only.yaml` | n/a (MAP only) | FAIL for 3-level PAT-RL (beta overflow, kappa stuck at prior); pick_best_cue: pending |
| B1 | B | 2-level | any | `hier_m1_2level.yaml` | Dense mass, non-centered (via Mode B configs) | Pending benchmark -- Phase 34 Mode B sweep |
| B2 | B | 3-level | any | `hier_m1_m2_laplace_cov_3level.yaml` | Dense mass + non-centered + Laplace warmup + covariates | Pending benchmark -- Phase 34 Mode B sweep |

**Reading the table:** Pick the row matching your pooling mode, model, and
cohort size. The Config YAML column gives the file under `configs/fit/` to
load. Where status says "pending benchmark", the recommendation is based on
Phase 30 design rationale, not completed runs.

---

## FitConfig Quick Reference

```python
from prl_hgf.fitting import FitConfig

# Load a preset configuration
cfg = FitConfig.from_yaml("configs/fit/none_2level.yaml")

# Override specific fields via CLI or in code
cfg = FitConfig.from_yaml(
    "configs/fit/m1_laplace_fp64_3level.yaml",
)
```

All `configs/fit/*.yaml` files follow `schema_version: 1`. See
[`MITIGATION_COMPAT.md`](MITIGATION_COMPAT.md) for valid flag combinations.

---

## Caveats

- **omega_3 recovery is poor.** This is a known limitation in the HGF
  literature. Primary hypotheses should target omega_2 and kappa. Verify
  recovery via `validation/` scripts before interpreting group effects on
  omega_3.

- **Dense mass matrix is infeasible at large P.** Dense mass + P>=30 +
  1000 warmup steps timed out at 24h (DEPS-05, job 55198489). At P=300
  with fp64, dense mass causes OOM (6 GiB allocation during warmup,
  Phase 31 job 55529660). Use diagonal mass for large cohorts.

- **Laplace provides MAP estimates only.** No posterior uncertainty
  quantification, no credible intervals. Suitable for quick exploration
  and parameter point estimates, not for publishable inference.

- **Evidence is task-specific.** All NUTS benchmarks are from
  `pick_best_cue` (T=420). PAT-RL (T=192) has only Laplace evidence.
  Cliff locations and walltimes may differ across tasks -- run a pilot
  fit before committing to a full cohort.

- **Many benchmark cells remain pending.** Phase 31 2-level resubmission
  (post-cuda13 fix), Phase 32 sampler audit, and Phase 34 Mode B sweep
  are incomplete. This document will be updated as evidence lands. Check
  [`CAPABILITY_MAP.md`](CAPABILITY_MAP.md) for the latest status.
