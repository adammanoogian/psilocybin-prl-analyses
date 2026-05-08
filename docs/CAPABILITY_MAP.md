# HGF Fitting Capability Map

Empirical record of what this repo's HGF fitting pipeline can and cannot fit,
indexed by `(model, sampler, task, P_total, T)`. The aim is a plug-and-play
reference: external users (and sister repos like `dcm_hgf_mixed_models`) can
look up whether their cohort × model combination is validated, partially
validated, or known-broken before committing compute.

**This file is the source of truth.** Memory and handoff docs reference it,
not the other way around. Update it whenever a fit run produces a new data
point — see [How to update](#how-to-update) at the bottom.

---

## Conventions

| Field | Meaning |
|---|---|
| **Model** | `2-level` / `3-level` HGF (binary branches; see `CLAUDE.md` for arch) |
| **Sampler** | `Laplace` (`fit_vb_laplace_*`) / `NUTS` (BlackJAX via `fit_batch_hierarchical`) |
| **Task** | `pick_best_cue` (T=420) / `PAT-RL` (T=192) / other |
| **P_total** | Total subjects in the joint NUTS fit. For `pick_best_cue`: `n_per_group × n_groups × n_sessions`. For `PAT-RL`: `n_per_phenotype × 4`. |
| **T** | Trials per participant |
| **Status** | ✅ PASS / ⚠ PARTIAL / ❌ FAIL / 🔲 NOT TESTED |
| **Walltime** | Observed end-to-end wall (cold), or `--` if not measured |
| **Evidence** | SLURM job ID + commit hash that produced the result |
| **Diagnostics** | R-hat / ESS / divergences / recovery correlations (where measured) |

A cell is only PASS if **principled defaults** held: `target_accept=0.9`,
`max_tree_depth=10`, `is_mass_matrix_diagonal=True` (current default), no
prior tightening, no NUTS short-circuits. Mitigations (dense mass matrix,
reparam, prior tightening) get their own rows tagged `[mitigated]`.

---

## Capability table

### `pick_best_cue` — 3-cue PRL with criterion-based reversals (T=420)

| Model | Sampler | P_total | Status | Walltime | Evidence | Diagnostics |
|---|---|---|---|---|---|---|
| 3-level | NUTS | 30 (n/grp=5) | ❌ TIMEOUT | >8h | job 55143456 (A100 80GB, commit `cc9ee03`) | window_adaptation never completed in 8h walltime — same conditioning failure as P=300 row, just slower to manifest |
| 3-level | NUTS [BlackJAX 1.5 dense] | 30 (n/grp=5) | ❌ NOT CLEARED (TIMEOUT @ 24h) | >24h | DEPS-05 evidence: `.planning/phases/27-dependency-upgrade-chain/DEPS-05-evidence.json`, job 55198489 (m3g108, commit `4a3b541`) | window_adaptation (is_mass_matrix_diagonal=False, n_warmup=1000, n_chains=4) hit 24h walltime; BlackJAX 1.5 only exposes diagonal-vs-dense (no low-rank kwarg per 27-03 API probe), so dense O(D²)/step is the only non-diagonal option at this version and it is not feasible at P=30 — Phase 29 (M1 wiring + pre-flight) inherits |
| 3-level | NUTS | 60 (n/grp=10) | 🔲 PENDING | — | overnight job 2026-05-04 | — |
| 3-level | NUTS | 102 (n/grp=17) | 🔲 PENDING | — | overnight job 2026-05-04 | — |
| 3-level | NUTS | 150 (n/grp=25) | 🔲 PENDING | — | overnight job 2026-05-04 | — |
| 3-level | NUTS | 198 (n/grp=33) | 🔲 PENDING | — | overnight job 2026-05-04 | — |
| 3-level | NUTS | 50 | ✅ PASS | 233s | job 54902462, commit `05545b9` | warmup ~100–120s; in-process warm 2.52× |
| 3-level | NUTS | 300 (n/grp=50) | ❌ TIMEOUT | >10h | job 55139044 (this session) | window_adaptation never finished step 1/500; GPU at 100% util — **conditioning failure** |
| 2-level | NUTS | any | 🔲 NOT TESTED | — | — | The 2-level NUTS gap — see [Open gaps](#open-gaps) |

### PAT-RL — binary safe/dangerous approach-avoid (T=192)

| Model | Sampler | P_total | Status | Walltime | Evidence | Diagnostics |
|---|---|---|---|---|---|---|
| 2-level | Laplace | 160 (40 × 4 phenotypes) | ✅ PASS | ~47s CPU (54.2s observed) | job 55139039, commit `b78a51c` | r(ω₂)=0.935, r(β)=0.831, d(ω₂)=1.63, d(β)=2.34, \|cor\|=0.005 — handoff numbers reproduced |
| 3-level | Laplace | 160 | ❌ FAIL | ~3:58 (failed) | job 55139041, commit `b78a51c` | β → exp overflow in σ(β·EV+...); κ MAP stuck at prior mean — documented pathology, persists |
| 3-level | NUTS | any | 🔲 NOT TESTED | — | — | Deferred from Phase 20 to Phase 14-15 GPU benchmark; gated on the `pick_best_cue` 3-level cliff result |
| 2-level | NUTS | any | 🔲 NOT TESTED | — | — | Same gap as above |

---

## Open gaps

These are explicit "we don't know" cells. Each is a potential fit run that
would close a meaningful question.

1. **2-level NUTS at production scale** (any task). Hypothesis: 2-level should
   push much further than 3-level before any cliff because there's no κ × ω₃
   correlation in the posterior. Strong candidate for the production path if
   3-level NUTS caps below science cohort size. Run command in
   [Planned runs](#planned-runs) §B.

2. **3-level NUTS cliff location** (`pick_best_cue`). Currently bracketed at
   `50 ≤ P_max ≤ 300`. Overnight sweep (commit `136f6de`) probes
   `{30, 60, 102, 150, 198}` — will resolve to within ±25 of the cliff or
   prove no cliff exists in that range.

3. **`pick_best_cue` Laplace baseline.** Has Laplace ever been run on
   `pick_best_cue` at production scale? No row currently. May be the cheapest
   capable path if NUTS cliffs out.

4. **PAT-RL at intermediate P** (between 0 and 160). Only the headline
   P_total=160 row exists. If 3-level NUTS works at small P_total for PAT-RL,
   the 3-level Laplace pathology might be a Laplace-only problem, not a
   model problem.

---

## Planned runs

> **Pre-flight note (post-`b737fe6`).** Every BlackJAX NUTS run since commit
> `3995412` (Apr 2026) crashed at the post-warmup boundary on a duplicate
> `max_num_doublings` kwarg. Fix landed in `b737fe6`. Job 55143670 actually
> completed window_adaptation but never sampled. Job 55143456 timed out
> mid-warmup. The next submit will be the first true measurement of the
> 3-level / 2-level NUTS cliff in weeks — treat earlier "P_total>50 doesn't
> finish" rows as pre-fix data points whose root cause was the kwarg bug,
> not the cliff.

### A. Re-submit — 3-level NUTS cliff sweep

```bash
sbatch --time=24:00:00 \
  --export=ALL,N_PER_GROUP_OVERRIDES="5,10,17,25,33" \
  cluster/14_benchmark_gpu.slurm
```
Walks `pick_best_cue` 3-level NUTS at P_total ∈ {30, 60, 102, 150, 198}.
Walltime bumped from 8h → 24h: empirically warmup at n/grp=5 took >8h on
A100 (job 55143456 TIMEOUT) and >7.85h on L40S (job 55143670). The
P_total=50 anchor's "~15 min warmup" was a pre-cliff regime that no longer
extrapolates.

### B. Queued for after A — 2-level NUTS sweep at the same P grid

```bash
sbatch --time=24:00:00 \
  --export=ALL,BENCH_MODEL=hgf_2level,N_PER_GROUP_OVERRIDES="5,10,17,25,33,50" \
  cluster/14_benchmark_gpu.slurm
```
Drops in the 2-level NUTS column at the same P_total grid plus P_total=300
(2-level should easily handle the cohort size that broke 3-level). Resolves
the 2-level NUTS gap at production scale in a single run.

Requires the `BENCH_MODEL` env passthrough to be wired (see this commit's
diff for `cluster/14_benchmark_gpu.slurm` and `scripts/03_pre_analysis/03_run_power_iteration.py`).

### C. Conditional — `pick_best_cue` Laplace baseline

```bash
sbatch --time=00:30:00 \
  --export=ALL,SAMPLER=numpyro,LAPLACE_ONLY=1,N_PER_GROUP_OVERRIDES="50" \
  cluster/14_benchmark_gpu.slurm
```
Cheap (~few minutes). Run if A reveals NUTS cliffs early — gives us the
Laplace baseline on `pick_best_cue` as a fallback for the production path.

---

## Mitigation candidates (only if A or B prove a binding cliff)

These are principled algorithmic changes — not tuning shortcuts. They get
fired ONLY if the validated NUTS path (mitigation-free) cannot reach the
science cohort size.

### M1. Dense mass matrix (cheap, principled, ~1h)

**Diagnosis target:** 3-level posterior has tight ω₂/ω₃/κ correlations from
the volatility update equation. A diagonal mass matrix can't precondition
those off-diagonal directions, so window_adaptation can't find a step size ε
hitting `target_accept=0.9` without ε → 0 → max_tree_depth saturation.

**Change:** `src/prl_hgf/fitting/hierarchical.py:1415` —
`is_mass_matrix_diagonal=True` → `False`. Memory cost: O(D²) with D = total
parameter count. For 3-level × P=300 × 5 params/participant, D ≈ 1500 →
mass matrix is 1500² × 8 bytes = 18 MB. Trivial.

**Wire-up:** thread `is_mass_matrix_diagonal` kwarg through
`fit_batch_hierarchical` → `_run_blackjax_nuts` → `window_adaptation()`.
Default `True` (current behavior); CLI flag `--dense-mass-matrix` flips it.

**Expected impact:** if cliff is conditioning-driven (most likely), this
should push `P_max` 2-3× higher. If cliff persists, the issue is funnel
geometry → escalate to M2.

### M2. Non-centered reparameterization for hierarchical hypers (medium, ~1 day)

**Diagnosis target:** hierarchical models with tight hyperprior ↔ many
participant-level params produce funnel posterior geometry (Neal's funnel).
Standard NUTS-on-hierarchical practice is non-centered: sample
`z ~ N(0, 1)` then set `param = μ_hyper + σ_hyper · z`. Unconfounds the
participant-level posterior from the hyperprior posterior.

**Change scope:** likely in `src/prl_hgf/models/hierarchical.py` (or wherever
the participant-level priors are constructed). Each participant-level
parameter that takes a hyperprior gets the non-centered transform.

**Expected impact:** removes funnel divergences. If both M1 and M2 are
applied and the cliff persists, the joint problem is genuinely too
anisotropic for single-block NUTS → escalate to M3.

### M3. Sub-cohort sharding with shared hypers (heavy, ~1 week)

**Diagnosis target:** for science cohorts where even M1+M2 fail. Statistically
principled: Gibbs-block split of the joint posterior. Hypers sampled
centrally given all participants; participant-level params sampled per-shard
given current hypers, in parallel across GPUs.

**Change scope:** custom orchestrator in `src/prl_hgf/fitting/`. Multi-GPU
coordination via `jax.pmap` or explicit MPI. Significant; only justified if
the science cohort genuinely cannot be fit by M1+M2.

### Out of scope (already shelved or unrelated to cliff)

- **Parallel-scan likelihood (Phase 25, ELK / quasi-DEER).** Shelved per
  `memory/project_phase25_shelved.md` — addresses high-T regimes (T > 10k)
  and high-D state, not the P-cohort cliff. Revisit only if the user adds
  a continuous-state HGF or trial counts grow >10k.
- **Tightening priors / capping max_tree_depth.** Tuning shortcuts; would
  inflate `P_max` artificially without measuring the true capability. Not
  acceptable per the project's end goal (see
  `memory/project_end_goal_capability_map.md`).

---

## How to update

Whenever a fit run lands a new data point:

1. **Update the relevant table** with the row's status, walltime, evidence
   (SLURM job ID + commit hash from `git log`), and diagnostics.
2. **If the run revealed a cliff or a passing limit**, update [Open gaps](#open-gaps)
   to reflect the new bracket.
3. **Commit the map update alongside the analysis commit** so the map is
   git-coherent with the underlying evidence (idata, summary CSVs, .out logs).
4. **Run the closure-guard test**: `pytest tests/integration/test_capability_map.py`
   — confirms the table parses and no row is left with placeholder text.

A row is "complete" when all of: status (not PENDING), walltime, evidence
(real job ID, not `tbd`), and at least one diagnostic value are filled in.

---

## Source-of-truth notes

- **Memory pointers** (cross-conversation):
  `memory/project_end_goal_capability_map.md` — the meta-goal this map
  serves; `memory/project_phase25_shelved.md` — out-of-scope alt-path note.
- **Sister-repo consumers** (e.g., `dcm_hgf_mixed_models`) should pull this
  file's current rev to know what's safe to import. The PAT-RL handoff doc
  (`docs/PAT_RL_PHASE20_HANDOFF.md`) covers the API; this map covers the
  empirical envelope.
- **Phase 14.1-07 verification** (`docs/PHASE_14_1_07_VERIFICATION.md`)
  diagnosed the cache-scope bottleneck and unblocked the production NUTS
  path; the rows in this map are downstream of that fix.
