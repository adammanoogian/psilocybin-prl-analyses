# Phase 32 Plan 01: Audit Protocol Pre-Registration Summary

**One-liner:** Pre-registered BlackJAX vs NumPyro audit protocol with 14-row config equivalence matrix, 26-cell cohort grid, and pre-specified decision rules.

## Outcome

AUDIT-01 satisfied. `.planning/AUDIT_PROTOCOL.md` (414 lines, 2497 words)
committed as standalone artifact BEFORE any audit code or run. Establishes
the pre-registration audit trail that gates Plans 32-02 through 32-05.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Write AUDIT_PROTOCOL.md | c90de9b | .planning/AUDIT_PROTOCOL.md |
| 2 | Commit protocol (gates all subsequent work) | c90de9b | same commit |

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| 26 SLURM tasks (20 head-to-head + 6 noise-floor) | A-vs-A noise floor at P=60 establishes MCMC variance baseline; 20 head-to-head covers full 5P x 2model x 2backend grid |
| use_laplace_warmup excluded from audit grid | BlackJAX-only feature; including it confounds backend vs warmup-method comparison |
| TIMEOUT accepted as valid outcome for dense P>=150 3-level | Phase 27 DEPS-05 evidence shows this config is likely infeasible; crashes are data |
| ESS-per-sec as primary metric with 80% dominance threshold | Pre-specified to prevent post-hoc cherry-picking of favorable metric |
| Fresh process per cell (no XLA cache sharing) | Prevents compile-time confound between sequential fits |

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- [x] `.planning/AUDIT_PROTOCOL.md` exists and is committed
- [x] Document specifies identical hyperparameters for both backends (Section 1)
- [x] Config equivalence gaps documented with fix obligations (Section 2, Gaps 1-3)
- [x] Cohort grid totals 26 SLURM tasks (Section 4)
- [x] All metrics enumerated (Section 5, 16 columns)
- [x] Decision rules pre-specified (Section 8, Rules 1-4)
- [x] Word count 2497 (>= 800 minimum)
- [x] Config Equivalence Matrix has 14 data rows (>= 10 minimum)

## Duration

~5 minutes (document creation + commit)

## Next Phase Readiness

Plan 32-02 is unblocked:
- AUDIT_PROTOCOL.md exists in git history as proof of pre-registration
- Gap 1 (max_tree_depth) and Gap 3 (extra_fields) documented with fix obligations
- No architectural decisions deferred
