# Toolbox benchmarks & method-validation scripts

Engineering/validation of the HGF fitting stack itself — **not** part of any
study's analysis pipeline. Repatriated 2026-08-08 from the psilocybin study
repo, where they had been swept along in the 2026-08-07 extraction.

| Script | Phase | What it validates |
|---|---|---|
| `04_sampler_audit.py` | 32 | BlackJAX vs NumPyro NUTS head-to-head |
| `05_sigma_identifiability.py` | 33 | Shared-sigma assumption, Mode B hierarchical |
| `06_recovery_validation_modeb.py` | 33 | Mode B recovery at P=200, covariate collinearity |
| `08_smoke_patrl_foundation.py` | 18–20 | PAT-RL task module composition |
| `14_benchmark_no_pooling.py` | 31 | Mode A grid sweep → capability map |
| `15_aggregate_bench31.py` | 31 | Aggregates the Mode A sweep |
| `16_aggregate_audit32.py` | 32 | Aggregates the sampler audit |
| `16_benchmark_hierarchical.py` | 34 | Mode B grid sweep → capability map |
| `17_aggregate_bench34.py` | 34 | Aggregates the Mode B sweep |
| `18_ppc_example.py` | — | PPC machinery demo (synthetic posterior; not inference) |
| `02_aggregate_variants.py` | 14.2 | Sampler-variant comparison log parser |

Matching SLURM jobs live in `cluster/archive/` (historical phase records —
paths updated to `scripts/benchmarks/`, but flags may reference options that
have since changed). Fit configs they consume are `configs/fit/*` (the
`none_*`, `benchmark_dense_*`, `m1_laplace*`, `hier_*` variants).
