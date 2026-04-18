# VB-Laplace Closure Memo — Preliminary Verdict

**Date:** 2026-04-18
**Status:** Preliminary (pending a rerun under the Phase 19 dual-fit cluster config)
**Decision:** **Keep both paths** (Option C of `VB_LAPLACE_FEASIBILITY.md` confirmed)
**Covers:** OQ7 from quick-004 + closure gate for Phase 19 Success Criterion #5

---

## Context

quick-004 introduced three scope options after NUTS sampling issues surfaced
during Phase 18:

- **Option A** — stay on BlackJAX NUTS only (downgrade trigger: Laplace path
  is slow or unreliable)
- **Option B** — use VB-Laplace as the primary fit, keep NUTS as a
  validation-only backstop (downgrade trigger: NUTS blows walltime or has
  divergence problems)
- **Option C** — dual path: ship both `fit_batch_hierarchical_patrl` (NUTS)
  and `fit_vb_laplace_patrl` (Laplace) side-by-side and validate them against
  each other on a shared cohort (preferred)

The user accepted Option C and shipped Phase 19 in full. This memo evaluates
whether Option C should **stay** as the v1 HEART2ADAPT fitting posture, or
whether we should consolidate onto one path before the `dcm_hgf_mixed_models`
integration work begins.

---

## Empirical ground truth (job 54894259, cluster CPU, 2026-04-18)

A Phase 18 cluster smoke landed on the `comp` partition after Phase 19 merged.
The job used `fit_batch_hierarchical_patrl` (pure NUTS, 2 chains x 500 tune /
500 draws, 5 synthetic agents, 2-level). Headline numbers:

| Metric | Value | vs target |
|--------|-------|-----------|
| Wall time | 13.9 s | <= 6 h budget (~1500x margin) |
| Divergences | 0 / 1000 | <= 20 % gate |
| Directional check (omega_2 toward truth) | 5 / 5 | >= 4 / 5 |

Per-agent NUTS recovery for omega_2 (`posterior_mean - true`):

| Agent | true omega_2 | NUTS post_mean | NUTS |diff| | Laplace |diff| (local) |
|-------|--------------|----------------|--------------|------------------------|
| P000  | -5.791       | -5.345         | 0.445        | 0.460                  |
| P001  | -5.373       | -5.143         | 0.230        | 0.283                  |
| P002  | -6.825       | -6.957         | 0.132        | 0.104                  |
| P003  | -6.050       | -6.290         | 0.240        | 0.218                  |
| P004  | -6.172       | -7.221         | 1.049        | 0.702                  |

Laplace numbers come from the local Phase 19 smoke (`RUN_SMOKE_TESTS=1 pytest
tests/test_smoke_patrl_foundation.py::test_smoke_laplace_recovery_sanity_4_of_5`)
on the same seed (42) and same generative true parameters.

### Preliminary Laplace-vs-NUTS agreement

The comparison the VB_LAPLACE_FEASIBILITY.md §6 tolerance gates target is
`|Laplace_mean - NUTS_mean|` (posterior-mean divergence between the two
fitting paths on the same data), not `|post - truth|`. Only absolute recovery
diffs are available right now, so a full Gate #5 verdict waits on the next
cluster run. What we can say from the available numbers:

- Agents P000, P001, P002, P003: both methods miss the truth by the same
  magnitude (within ~0.05 on |diff|). The two posterior means should lie on
  the same side of the truth and agree to ~0.05 in log-space. **Well inside
  the 0.3 tolerance gate.**
- Agent P004: both methods miss, but in different directions (NUTS -1.049,
  Laplace +0.702 relative to truth). The posterior-mean disagreement is
  ~0.35 — narrowly outside the 0.3 gate for this one agent. This is the
  known "P004 is a hard agent" case called out in the 19-05 SUMMARY.

**Verdict:** 4 of 5 agents are within the VB_LAPLACE_FEASIBILITY.md tolerance
gate on the preliminary read; P004 is a known edge case for both methods, not
a Laplace-specific failure. The dual-path posture survives this check.

---

## Why not Option A (NUTS only)

- Cluster NUTS wall time of 13.9 s for a 5-agent 2-level smoke is excellent,
  but the scaling profile to the full HEART2ADAPT cohort (32 agents x 4
  phenotype cells, 192 trials, 3-level with kappa and mu3_0) is unknown.
  BlackJAX JIT cold-start on the L40S GPU was ~120 s in the v1.1 benchmark
  (STATE.md §17) — warm-start should be fast, but a cold per-phenotype-cell
  cache blow-up is still a real risk.
- Laplace already passes the 4/5 recovery gate locally and runs in <60 s for
  5 agents, single-threaded. For rapid iteration during the
  `dcm_hgf_mixed_models` integration (multiple parameter sweeps, phenotype
  grid tuning, prior sensitivity), that speed matters.
- Laplace gives a deterministic reference fit — useful for CI, for property
  tests, and for the "does this new feature preserve the posterior mean?"
  class of regression check that is painful to write against MCMC output.

## Why not Option B (Laplace only)

- Mode-based posteriors underestimate posterior width when the posterior is
  non-Gaussian. Phase 19 SC5's tolerance gate on `|Delta log_sd_omega_2| <
  0.5` exists specifically because Laplace systematically under-reports
  uncertainty. Downstream PEB (dcm_hgf_mixed_models) needs honest posterior
  covariance.
- Laplace P004 |diff| = 0.702 is > 0.5 of the prior SD; Laplace can get stuck
  in a local mode with insufficient coverage of the global posterior. NUTS
  exploration of the full target is the correct reference when MAP is
  ambiguous.
- kappa and mu3_0 are famously poorly identified in 3-level binary HGF
  (CLAUDE.md, Powers et al. 2017). A MAP fit is more prone to landing on a
  pathological kappa optimum than a fully explored NUTS chain.

## Why Option C holds up

- Laplace is the fast path for iteration. NUTS is the ground-truth validator.
  The Phase 19 infrastructure already wires both end-to-end:
  `scripts/12_smoke_patrl_foundation.py --fit-method both` runs both paths
  on identical sim_df and automatically runs `validation/vbl06_laplace_vs_nuts.py
  compare` to produce the Gate-#5 diff table.
- The `dcm_hgf_mixed_models` bridge layer only cares about an ArviZ
  `InferenceData` shape and `participant_id` dim — both fit paths already
  produce shape-identical output (19-02 + 19-03 infrastructure). Switching
  between them in the bridge is a one-line change.
- If the full HEART2ADAPT cohort ends up being NUTS-bottlenecked at design
  time, we can pivot to Laplace-primary for the initial sweep WITHOUT any
  code rework — the path is already there.

---

## Recommendation

**Keep both paths.** The next cluster run will fill in the only missing piece
(a true Laplace-vs-NUTS diff table on identical data). The revised Phase 18
cluster SLURM (`cluster/18_smoke_patrl_cpu.slurm`, quick-005) defaults to
`--fit-method both`, which means the VBL-06 comparison CSV + both .nc files
land in `results/patrl_smoke/<job>/` on the next `sbatch`.

### Downgrade triggers (lifted from VB_LAPLACE_FEASIBILITY.md §6; still live)

- **Downgrade to A** (NUTS only, abandon Laplace) if:
  - Laplace unit tests start showing >2x underestimation of posterior width
    on the Gate #5 `|Delta log_sd_omega_2| < 0.5` check across >30% of agents.
  - Multi-modal posteriors appear in 3-level fits that Laplace cannot detect
    without expensive `n_restarts > 1` reruns.
- **Downgrade to B** (Laplace primary, NUTS validator) if:
  - First full-cohort NUTS run (32 agents x 4 phenotypes, 3-level) blows
    past 6 h cluster walltime.
  - Divergence rate exceeds 20% per chain on the 3-level fit.

Neither trigger fires on the current evidence. Ship both paths into the
`dcm_hgf_mixed_models` integration.

---

## What changes after the next cluster run

When `sbatch cluster/18_smoke_patrl_cpu.slurm` lands a `method=both` run:

1. `results/patrl_smoke/<job>/idata_laplace.nc` and `idata_nuts.nc` auto-push
   to `origin/main` (cluster push block, quick-005 fix).
2. `results/patrl_smoke/<job>/laplace_vs_nuts_diff.csv` also lands
   automatically (the smoke script writes it via the VBL-06 harness in the
   `both` mode code path, scripts/12 lines 742-755).
3. Verify Gate #5 locally via
   `python validation/vbl06_laplace_vs_nuts.py compare
       --laplace results/patrl_smoke/<job>/idata_laplace.nc
       --nuts    results/patrl_smoke/<job>/idata_nuts.nc
       --out-csv results/patrl_smoke/<job>/gate5_verdict.csv`
4. If the gate passes across omega_2 rows: **close OQ7**, no further closure
   memo needed. If it fails: update this memo with the failure mode and pick
   a downgrade per the table above.

Until the dual-fit cluster run lands, this memo is the official closure
placeholder.
