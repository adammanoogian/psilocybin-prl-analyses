# Quick Task 002: Summary

## What Was Done

Created a comprehensive coding guide in the Obsidian Vault at:
`_inbox/Coding Guide - HGF Bayesian Fitting with JAX and NumPyro - 2026-04-14.md`

## Document Structure

1. **Mathematical Formulation** — Binary HGF update equations, 3-cue network topology, softmax-stickiness response model, prior specifications with rationale
2. **pyhgf Implementation** — Network API, scan function extraction, parameter injection pattern
3. **JAX Patterns** — lax.scan for trial sequences, NaN clamping (Tapas convention, mu_2_bound=14.0), vmap for cohort parallelism, data-as-arguments vs closures
4. **NumPyro MCMC** — Model definition (sample + factor), chain method selection table, ArviZ conversion
5. **HGF-Specific Lessons** — Partial feedback encoding, stickiness sentinel, attribute pytree keys, omega negativity constraint
6. **JAX-Specific Lessons** — No Python control flow on traced values, dtype consistency in lax.scan, compilation cache, shape-dependent recompilation
7. **PyMC Migration** — _init_jitter bug, closure constant problem, dim naming issues
8. **Cluster Deployment** — CUDA/PTX mismatch (the #1 pitfall), SLURM GPU template, chunk-based array jobs, GPU memory guide, pitfalls checklist
9. **References** — Mathys 2011/2014, Rigoux 2014, Weber 2024, Tapas toolbox

## Vault Updates

- `_INDEX.md`: Added entry under Coding Guides
- `_LOG.md`: Appended ingest operation record
- Wikilinks to: References/Bayesian Inference, References/Predictive Coding, Structured Notes/PRL HGF Analysis Planning
