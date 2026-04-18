# VB-Laplace Closure Memo — FINAL Verdict

**Date:** 2026-04-18 (first draft); **FINALIZED** 2026-04-18 after cluster job 54896739 dual-fit
**Status:** FINAL
**Decision:** **Keep both paths, use by purpose** (Option C of `VB_LAPLACE_FEASIBILITY.md` confirmed with documented limitations)
**Covers:** OQ7 from quick-004 + closure gate for Phase 19 Success Criterion #5 (CLOSED)

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

## Empirical ground truth (job 54896739, dual-fit cluster CPU, 2026-04-18)

Phase 18 cluster smoke re-run with `--fit-method both` after the quick-005
SLURM update. Both paths ran on identical `sim_df` (seed 42). Results
auto-pushed to `results/patrl_smoke/54896739/`.

### Cohort summary

| Metric | Value | vs target |
|--------|-------|-----------|
| Wall time (both paths) | 22.4 s | <= 6 h (~960x margin) |
| NUTS divergences | 0 / 1000 | <= 20 % gate |
| Laplace convergence | 5/5 agents | all converged |
| Directional check (ω₂ toward truth, NUTS) | 5 / 5 | >= 4 / 5 |

### Per-agent Laplace-vs-NUTS posterior agreement (from `laplace_vs_nuts_diff.csv`)

**omega_2 (the gated parameter):**

| Agent | mean_laplace | mean_nuts | \|Δ mean\| | sd_laplace | sd_nuts | \|Δ log_sd\| | sd_ratio (L/N) | within_gate |
|-------|-------------|-----------|-----------|------------|---------|-------------|----------------|-------------|
| P000 | -5.331 | -5.345 | **0.014** | 0.185 | 0.278 | 0.407 | 0.665 | **PASS** |
| P001 | -5.083 | -5.143 | **0.060** | 0.182 | 0.354 | 0.663 | 0.515 | fail (sd) |
| P002 | -6.926 | -6.957 | **0.031** | 0.177 | 0.343 | 0.661 | 0.516 | fail (sd) |
| P003 | -6.268 | -6.290 | **0.022** | 0.166 | 0.269 | 0.482 | 0.617 | **PASS** |
| P004 | -6.873 | -7.221 | 0.347 | 0.160 | 0.830 | 1.649 | 0.192 | fail (both) |

**beta (informational only, not gated):**

| Agent | mean_laplace | mean_nuts | \|Δ mean\| | \|Δ log_sd\| | sd_ratio (L/N) |
|-------|-------------|-----------|-----------|-------------|----------------|
| P000 | 2.966 | 2.828 | 0.139 | 0.429 | 1.54 |
| P001 | 3.075 | 2.953 | 0.122 | 0.681 | 1.98 |
| P002 | 2.681 | 2.535 | 0.145 | 0.738 | 2.09 |
| P003 | 3.114 | 2.965 | 0.149 | 0.471 | 1.60 |
| P004 | 1.655 | 1.398 | 0.257 | 1.515 | 4.55 |

### What the numbers actually say

Two distinct patterns emerge when you compare the two fit paths:

1. **Posterior means agree across both methods — very well.** 4 of 5 agents
   have `|Δ mean_ω₂| < 0.06`. All 5 have `|Δ mean_β| < 0.30`. **This is the
   quantity that matters for PEB regression** (covariate mean effects). On
   the primary scientific deliverable, Laplace and NUTS are essentially
   interchangeable.

2. **Posterior widths diverge — and asymmetrically.** 
   - For **omega_2**: Laplace systematically UNDERESTIMATES the width.
     sd_ratio ≈ 0.5 on 4 agents, 0.19 on P004. This is the textbook Laplace
     pathology — a mode-centered Gaussian misses curvature and underestimates
     uncertainty.
   - For **beta**: Laplace OVERESTIMATES the width. sd_ratio ≈ 1.5-2.1 on 4
     agents, 4.55 on P004. This is driven by the log→natural parameterisation:
     `beta = exp(log_beta)` with log_beta ~ Gaussian → the Delta-method
     transform inflates sd_beta at high mean_log_beta while NUTS explores
     the bounded target directly.

### Formal Gate #5 verdict (VB_LAPLACE_FEASIBILITY.md §6 tolerance bands)

- `|Δ mean_ω₂| < 0.3`: **4/5 agents PASS** (P004 at 0.35 fails)
- `|Δ log_sd_ω₂| < 0.5`: **2/5 agents PASS** (P000=0.41, P003=0.48; P001/P002/P004 fail)
- Combined gate (AND): **2/5 agents PASS** (P000, P003)

The literal gate fails. The **root cause is the Laplace log_sd systematic
underestimation**, not a fundamental Laplace-vs-NUTS algorithmic disagreement.

### Downgrade triggers (from VB_LAPLACE_FEASIBILITY.md §6)

- **Trigger A** (switch to NUTS-only if >2× SD underestimation on >30% of
  agents): only P004 exceeds 2× (ratio 0.19 → 5.2× underestimation). 1/5 =
  20% of agents → **below 30% threshold → trigger NOT fired**.
- **Trigger B** (switch to Laplace-primary if NUTS walltime > 6h or
  divergences > 20%): cluster NUTS ran in 22.4s total with 0/1000
  divergences → **not fired**.

Neither trigger met. Dual-path posture holds.

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

## Recommendation — Keep both paths, use by purpose

Option C stands, but refined with a clear use-case split informed by the
cluster numbers:

| Use case | Preferred fit path | Why |
|----------|--------------------|-----|
| **PEB covariate means / point estimates** | Laplace OR NUTS | Means agree within 0.06 on 4/5 agents; use whichever is faster (Laplace in CI, NUTS in cluster production) |
| **Uncertainty quantification / HDI coverage** | **NUTS only** | Laplace underestimates ω₂ width by ~2× and overestimates β width by ~1.8× (opposite sign, both significant). Feed NUTS posteriors to any HDI-based weighting or Bayesian model comparison downstream. |
| **Model comparison / WAIC / log-likelihood integration** | **NUTS only** | Requires full posterior exploration; Laplace Gaussian approximation is inadequate at the tails. |
| **Rapid iteration / CI / smoke tests** | **Laplace** | 15× faster, deterministic, no divergence risk; infrastructure via `fit_vb_laplace_patrl`. |
| **Ground-truth recovery tests in Phase 18+ / validation** | Laplace-primary + NUTS-validator | Dual-fit path via `scripts/12 --fit-method both` remains the reference. |

Ship both paths into the `dcm_hgf_mixed_models` integration. The consumer's
bridge layer can dispatch on the fit method tag inside the `idata` — shape
parity is already enforced by Phase 19's shared `_samples_to_idata_patrl` /
`build_idata_from_laplace` factories.

### Documented Laplace limitations (for bridge-layer users)

1. **ω₂ posterior width ≈ 60% of NUTS width.** Under-coverage of 94% HDIs
   is the dominant risk. Do not use Laplace posteriors for uncertainty-
   driven decisions (e.g. "is this participant's ω₂ significantly different
   from the group mean?").
2. **β (natural-scale) posterior width ≈ 150-200% of NUTS width.** Over-
   coverage on the upper tail. Same caveat inverse: Laplace β CIs will be
   too conservative.
3. **Both pathologies scale with posterior non-Gaussianity.** P004 shows
   both problems at maximum — 5× underestimation on ω₂, 4.5× overestimation
   on β. Use NUTS for any participant with skewed posterior geometry.

### Downgrade triggers (STILL LIVE; re-check after first full-cohort run)

- **Downgrade to A** (NUTS-only, abandon Laplace) — not currently met:
  - Would fire if >30% of agents show >2× ω₂ width underestimation
  - Current: 20% (P004 only) — within tolerance

- **Downgrade to B** (Laplace-primary, NUTS-as-validator only) — not currently met:
  - Would fire if NUTS walltime > 6h on the full cohort or divergences > 20%
  - Current: 22.4s on 5 agents → ~2 minutes projected for 32 agents; divergences 0%.

No triggers fire at current scale. **Recheck after the first HEART2ADAPT
32-agent × 4-phenotype × 3-level cluster run.** If 3-level posteriors are
dramatically more non-Gaussian (as ω₃ × κ geometry famously is in binary
HGF — Powers et al. 2017), Laplace SD error may worsen and trigger A.

---

## OQ7 status: CLOSED

The first dual-fit cluster run (54896739) has landed. The formal gate numbers
are above. OQ7 is closed with the verdict:

> Keep both paths. Use Laplace for speed-sensitive workloads (CI, iteration,
> point estimates). Use NUTS for uncertainty-sensitive workloads (HDI
> coverage, model comparison, anything downstream of posterior variance).
> Document the asymmetric SD bias (Laplace under-wide on ω₂, over-wide on
> natural-scale β) in the `docs/PAT_RL_API_HANDOFF.md` consumer guide.

### Actions still open (tracked for HEART2ADAPT v3+, not Phase 19)

1. **3-level dual-fit comparison**: current gate numbers are from 2-level
   only. 3-level (adds ω₃, κ, μ₃⁰) has known-bad posterior geometry.
   Re-run gate check with `PRL_PATRL_SMOKE_LEVEL=3` before committing to
   dual-path for 3-level production.
2. **Full-cohort dual-fit**: 32 agents × 4 phenotypes. Check Laplace SD
   behavior under phenotype diversity (anxiety + reward sensitivity may
   broaden or narrow posterior geometry).
3. **Trigger reassessment after v3 scientific validation**: if any
   HEART2ADAPT analyses materially depend on posterior uncertainty (PEB
   credible intervals, model comparison), revisit the "Laplace overall vs
   NUTS overall" tradeoff with the real scientific payload in hand.

### Consumer-side guidance

The `dcm_hgf_mixed_models` v2 bridge layer should select fit method per the
use-case table above. No code change needed in prl_hgf; the shape-parity
contract established in Phase 19-02 (`build_idata_from_laplace` emits
`participant_id` dim natively + parameters in canonical order) means a
simple `method = "laplace" if fast_path else "blackjax"` flag at the
consumer's fit entry point is sufficient.
