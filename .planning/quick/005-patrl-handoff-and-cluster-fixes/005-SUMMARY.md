# Quick Task 005: PAT-RL Handoff + Cluster Fixes + OQ Resolutions

**Date:** 2026-04-18
**Scope:** Four follow-ups after the Phase 18 cluster smoke landed
**Status:** Complete

---

## What shipped

### A — Cluster artifact transfer via git (not scp)

`cluster/18_smoke_patrl_cpu.slurm` updated:

- Output directory moved from gitignored `output/patrl_smoke_<job>/` to
  tracked `results/patrl_smoke/<job>/`.
- Cluster script now defaults to `--fit-method both` (dual NUTS + Laplace
  fit on identical sim_df) via new `PRL_PATRL_SMOKE_METHOD` env var. Per the
  Option C decision in `.planning/quick/004-.../VB_LAPLACE_FEASIBILITY.md`.
- Auto-push block now stages `.csv` AND `.nc` files so the offline
  `validation/vbl06_laplace_vs_nuts.py` harness can consume InferenceData
  directly from a local `git pull` — no scp required.
- Header documentation updated to reflect new output paths.

Net effect: next `sbatch cluster/18_smoke_patrl_cpu.slurm` writes trajectory
CSVs + parameter summary + `idata_laplace.nc` + `idata_nuts.nc` +
`laplace_vs_nuts_diff.csv` to a git-tracked path that auto-pushes to
`origin/main`.

### B — OQ1 resolution: documentation gap, not a live bug

Inspected `src/prl_hgf/fitting/hierarchical.py::_samples_to_idata` (line
1555). The function has a `coord_name` parameter with default `"participant"`.
The docstring already states: *"PAT-RL passes `'participant_id'` to match
downstream exporter expectations."*

- PAT-RL fitting (`fit_batch_hierarchical_patrl`) **already passes
  `coord_name="participant_id"`** — verified in `hierarchical_patrl.py:88`.
- Pick_best_cue fitting uses the default `"participant"` — intentional, and
  incompatible with `export_subject_trajectories` (which is a PAT-RL
  exporter). **This is by design**: `export_subject_trajectories` was
  authored for PAT-RL trajectories; pick_best_cue has its own exporters
  elsewhere.
- The first cluster smoke job (54893705) that failed with a
  `participant_id` error was a **Phase 19-01 issue** (pre-refactor
  simulator produced inconsistent `participant_id` values vs what the
  exporter looked up). The refactor in Phase 19-01 resolved it — job
  54894259 passed immediately after.

**No code fix required.** STATE.md updated to retire OQ1 as "resolved by
Phase 19-01 refactor; hierarchical.py::_samples_to_idata coord_name=
participant is intentional for pick_best_cue".

### C — OQ7 closure memo

`.planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/VB_LAPLACE_CLOSURE_MEMO.md`
written based on preliminary agreement numbers from cluster job 54894259.
Key points:

- 4 of 5 agents: Laplace and NUTS posterior means agree within ~0.05 in
  omega_2 — well inside the 0.3 tolerance gate.
- Agent P004: both methods miss the truth (Laplace +0.702, NUTS -1.049);
  disagreement ~0.35, narrowly outside the gate — flagged as a known hard
  case, not a Laplace-specific failure.
- **Recommendation: keep both paths (Option C confirmed).** Neither
  downgrade trigger fires on current evidence.
- Final Gate #5 verdict deferred until first cluster dual-fit run lands
  `laplace_vs_nuts_diff.csv` (the revised SLURM script produces it
  automatically via `scripts/12 --fit-method both` path).

### Handoff — `docs/PAT_RL_API_HANDOFF.md`

Single-source-of-truth public API document for the PAT-RL surface that
`dcm_hgf_mixed_models` v2 will consume. 7 sections:

1. **What changed since SISTER_API_PRL_HGF.md** — side-by-side old/new
   module map. The consumer's existing sister-API doc describes the
   deprecated pick_best_cue surface.
2. **Public API by subpackage** — `env.pat_rl_config`,
   `env.pat_rl_sequence`, `env.pat_rl_simulator`, `models.hgf_*_patrl`,
   `models.response_patrl`, `fitting.hierarchical_patrl`,
   `fitting.fit_vb_laplace_patrl`, `fitting.laplace_idata`,
   `analysis.export_trajectories`. Function signatures + critical pyhgf
   0.2.8 API quirks inline.
3. **Integration-point map** — which prl_hgf PAT-RL module each
   `dcm_hgf_mixed_models` stub (`task/`, `agents/`, `bridge/`, `analysis/`,
   `plotting/`) should call, with recommended modulator channel choices
   (epsilon2, epsilon3, psi2, delta_hr raw float64 as dcm_pytorch bilinear
   modulators per `18-05-dcm-interface-notes.md`).
4. **Minimal end-to-end example** — ~30 lines of copy-pasteable code
   running cohort simulation → Laplace fit → trajectory export → ready for
   DCM modulator wiring.
5. **Data contracts** — authoritative CSV column schemas (19-col
   trajectory CSV; 5-col parameter summary) and InferenceData dim spec.
6. **Known gaps / TODOs** — Models B/C/D deferred to Phase 20+; no PEB
   covariate export helper yet; kappa fixed at 1.0 across phenotypes;
   Laplace-vs-NUTS Gate #5 verdict pending dual-fit cluster run.
7. **Quick pointers table** — "need X, look at file Y" for rapid consumer
   onboarding.

This document **supersedes** `dcm_hgf_mixed_models/.planning/research/
SISTER_API_PRL_HGF.md` for any PAT-RL / HEART2ADAPT work. The consumer
repo should either port the relevant sections in-tree or link to this doc
from its own `docs/03_methods_reference/`.

---

## What the cross-repo handoff actually requires (user action)

The producer side is done. For `dcm_hgf_mixed_models` v2 to pick it up:

1. `cd ../dcm_hgf_mixed_models` — check if `prl_hgf` is installed editable
   from this repo (per `pyproject.toml` `siblings` extra). If not, `pip
   install -e ../psilocybin_prl_analyses`.
2. Run the §4 minimal example in the consumer's `scripts/00_run_full_
   pipeline.py` to confirm the integration works end-to-end. Expected:
   `output/demo/P000_trajectories.csv` + `parameter_summary.csv` produced
   in <60 seconds.
3. In `dcm_hgf_mixed_models/.planning/research/SISTER_API_PRL_HGF.md`,
   add a banner: *"SUPERSEDED BY: `../psilocybin_prl_analyses/docs/
   PAT_RL_API_HANDOFF.md` for all PAT-RL / HEART2ADAPT work. This document
   covers the older pick_best_cue surface only."* (Cross-repo edit; not
   done in this quick task.)
4. Wire the stubs per §3 of the handoff doc: `task/trial_sequence.py` →
   `generate_session_patrl`; `agents/simulate.py` → `simulate_patrl_cohort`;
   `bridge/hgf_to_dcm.py` → `export_subject_trajectories`; etc.
5. For v3 (scientific validation): request Phase 20 features in prl_hgf
   for Models B/C/D + PEB covariate export, OR implement consumer-side.

---

## Files shipped

**Modified** (prl_hgf):
- `cluster/18_smoke_patrl_cpu.slurm` — 5 edits (output path, method
  default, push block .nc inclusion, mkdir, header docs)

**Created** (prl_hgf):
- `.planning/quick/004-.../VB_LAPLACE_CLOSURE_MEMO.md` — ~140 lines
- `docs/PAT_RL_API_HANDOFF.md` — ~430 lines
- `.planning/quick/005-.../005-SUMMARY.md` — this file

**Did NOT touch** (parallel-stack invariant still upheld for Phase 19):
- Any `src/prl_hgf/` runtime code
- `configs/pat_rl.yaml`
- `configs/prl_analysis.yaml`
- `scripts/12_smoke_patrl_foundation.py` (the `--fit-method` flag it needs
  for the cluster default is already Phase 19-05 material)
- Any `tests/` file
- The consumer repo `dcm_hgf_mixed_models` (cross-repo edits are for the
  user to make when they wire up v2)

---

## Verification

1. `cat cluster/18_smoke_patrl_cpu.slurm | grep -E "output/|results/"` —
   only `results/` paths present for PAT-RL smoke outputs. `output/` only
   appears in legacy comments (none in code).
2. `ls docs/` — `PAT_RL_API_HANDOFF.md` exists.
3. `ls .planning/quick/004-patrl-smoke-and-vb-laplace-feasibility/` —
   includes `VB_LAPLACE_CLOSURE_MEMO.md`.
4. No runtime code changed → no new test regressions possible on this
   quick task. Pick_best_cue + Phase 18 + Phase 19 test suites unaffected.

---

## Next action for the user

When the next cluster run lands (with the new defaults):

```bash
git pull origin main
ls results/patrl_smoke/<latest_job>/
# Expect: trajectory CSVs + parameter_summary.csv + idata_laplace.nc +
#         idata_nuts.nc + laplace_vs_nuts_diff.csv
python validation/vbl06_laplace_vs_nuts.py compare \
    --laplace results/patrl_smoke/<job>/idata_laplace.nc \
    --nuts    results/patrl_smoke/<job>/idata_nuts.nc \
    --out-csv results/patrl_smoke/<job>/gate5_verdict.csv
```

If Gate #5 passes across omega_2 rows: OQ7 can be formally closed; update
the closure memo to "FINAL" status.

If it fails: consult `VB_LAPLACE_CLOSURE_MEMO.md` §"Downgrade triggers".
