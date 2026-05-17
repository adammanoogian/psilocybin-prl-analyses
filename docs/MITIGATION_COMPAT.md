# Mitigation Compatibility Matrix

Documents which `MitigationConfig` flag combinations are valid and tested.

## Flag Reference

| Flag | Field | Default | Wired In | Notes |
|------|-------|---------|----------|-------|
| Dense mass matrix | `mass_matrix_kind` | `"diagonal"` | Phase 29 | O(D^2) memory per chain |
| Laplace warmup | `use_laplace_warmup` | `False` | Phase 30 | Multi-start LBFGS init |
| FP64 precision | `use_fp64` | `False` | Phase 30 | Doubles memory; improves numerics |
| Multi-GPU sharding | `use_shard_map` | `False` | Phase 30 | Chain-axis shard_map |
| Non-centered params | `non_centered` | `()` | Phase 33 | Per-parameter reparameterization |

## Compatibility Rules

| Flag A | Flag B | Compatible? | Notes |
|--------|--------|-------------|-------|
| `use_laplace_warmup` | `backend="laplace"` | NO | Cannot Laplace-warm a Laplace fit |
| `non_centered` (any) | `pooling="none"` | NO | Non-centering requires hierarchical priors |
| `use_shard_map` | `backend="laplace"` | NO | Sharding is for MCMC chains only |
| `mass_matrix_kind="dense"` | Large P (>100) | CAUTION | Memory pre-flight check (Phase 29) |
| All other combinations | -- | YES | |

## GUARD-02 Test Coverage

Phase 28 tests the following grid:
- `{hgf_2level, hgf_3level} x {blackjax} x {diagonal} x {default priors, tight priors}`

Phases 29-33 progressively expand coverage:
- Phase 29 adds: `x {diagonal, dense}`
- Phase 30 adds: `x {fp64: T/F} x {shard_map: T/F} x {laplace_warmup: T/F}`
- Phase 33 adds: `x {non_centered} x {pooling: hierarchical}`

## Pre-flight Validation

`src/prl_hgf/fitting/preflight.py` validates all flag combinations before
launching the sampler. Currently a stub -- populated in Phases 29-30.

Checks planned:
1. **Memory estimate** (Phase 29): `D^2 x 8 x n_chains` for dense mass matrix
2. **Logical conflicts** (Phase 30): non_centered without hierarchical, laplace_warmup with backend=laplace
3. **Collinearity** (Phase 33): covariate x group correlation check
