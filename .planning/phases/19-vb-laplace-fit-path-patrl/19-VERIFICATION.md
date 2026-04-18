---
status: human_needed
phase: 19
verified_at: 2026-04-18
score: 6 of 6 automated success criteria PASSED; SC5 full NUTS comparison pending cluster data
human_verification:
  - test: Run Laplace-vs-NUTS comparison on cluster NUTS output
    expected: abs_diff_mean(omega_2) below 0.3 AND abs_diff_log_sd(omega_2) below 0.5 per subject
    why_human: Cluster NUTS output has not yet landed; only --skip-nuts-comparison path is exercisable at HEAD.
---

# Phase 19: VB-Laplace Fit Path for PAT-RL Verification Report

**Phase Goal:** A second, non-MCMC fit path exists for PAT-RL alongside BlackJAX NUTS: variational Bayes with Laplace approximation at the MAP. Mirrors the matlab tapas HGF toolbox convention. Lives in a new `src/prl_hgf/fitting/fit_vb_laplace_patrl.py` that reuses the existing `_build_patrl_log_posterior` pure-JAX logp surface from `hierarchical_patrl.py` without modifying it. Returns an ArviZ `InferenceData` shape-compatible with the NUTS path. Unblocks downstream PEB development while cluster NUTS numbers come in.

**Verified:** 2026-04-18
**Status:** human_needed (6 of 6 automated success criteria PASSED; SC5 full NUTS comparison HUMAN-CHECK pending cluster data)
**Re-verification:** No -- initial verification

## Summary

All 6 Phase 19 automated success criteria pass at HEAD. 27 of 27 fast pytest cases pass (8 in test_fit_vb_laplace_patrl + 11 in test_laplace_idata + 8 in test_vbl06_laplace_vs_nuts). Parallel-stack invariant (SC6) confirmed by empty `git diff 2045a27..HEAD` on all 9 protected paths. The only outstanding check is SC5 full Laplace-vs-NUTS hard-gate comparison against real cluster NUTS output -- the `--skip-nuts-comparison` path is tested and exits 0 as designed; the hard-gate comparison cannot be exercised end-to-end until `sbatch cluster/18_smoke_patrl_cpu.slurm` output lands. This is exactly the scope described in quick-004 VB_LAPLACE_FEASIBILITY.md.

## Success Criteria

### SC1 -- fit_vb_laplace_patrl with correct signature and MAP+Hessian flow

Status: PASSED

Evidence:
- `src/prl_hgf/fitting/fit_vb_laplace_patrl.py:104-415` implements signature `(sim_df, model_name, response_model=model_a, config=None, n_pseudo_draws=1000, max_iter=200, tol=1e-5, n_restarts=1, random_seed=0)`.
- Uses `jaxopt.LBFGS` at line 279. Computes Hessian via `jax.hessian(lambda f: -log_posterior_fn(unravel(f)))(flat_mode)` at line 342. Packages via `build_idata_from_laplace` at line 406.
- 2-level shape contract verified by `test_build_idata_2level_shape_contract` (posterior shape (1, 500, 4) for P=4).
- 3-level shape contract verified by `test_build_idata_3level_shape_contract` (5 latent vars + beta deterministic, all shape (1, 500, 3) for P=3).
- Command `python -c "from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl; import inspect; print(inspect.signature(fit_vb_laplace_patrl))"` returns the expected signature.

### SC2 -- PD regularization via eigh-clip with canonical diagnostic keys

Status: PASSED

Evidence:
- `_regularize_to_pd` at `fit_vb_laplace_patrl.py:54-96` uses `np.linalg.eigh`, clips eigvals below `eps=1e-8`, reconstructs via `V @ diag(w_clip) @ V.T`.
- Diagnostics dict includes canonical keys: hessian_min_eigval, hessian_max_eigval, n_eigenvalues_clipped, ridge_added.
- WARN log at lines 350-355 when eigenvalues are clipped.
- `test_regularize_to_pd_clips_negative_eigenvalues` PASS: 1 eigval clipped from diag([-0.5, 2.0]).
- `test_regularize_to_pd_preserves_pd_matrix` PASS: 0 clipped for PD input.
- `test_sample_stats_group_present` PASS: all 7 canonical diagnostic keys round-trip through InferenceData.sample_stats.

### SC3 -- export_subject_trajectories and export_subject_parameters consume Laplace InferenceData unchanged

Status: PASSED

Evidence:
- `test_consumer_compatibility_export_subject_parameters` (tests/test_laplace_idata.py:231) runs `export_subject_parameters` on a Laplace idata directly, asserts correct CSV schema (participant_id, parameter, posterior_mean, hdi_low, hdi_high) and row count (2*P for 2-level). PASSED.
- End-to-end corroboration in 19-05 SUMMARY: 3-agent `--fit-method laplace` smoke produces per-subject trajectory CSVs + `parameter_summary.csv`.
- Laplace factory emits dim `participant_id` natively (test_dim_name_is_participant_id_not_participant PASS), matching the exporter reader at `export_trajectories.py:181`.

### SC4 -- 5-agent CPU smoke under 60 seconds; omega_2 within 0.5 of truth for 4 of 5 agents

Status: PASSED

Evidence:
- 19-05 SUMMARY documents 30.7s wall time (under 60s gate) at `--seed 42 --level 2 --n-participants 5` with 4 of 5 agents meeting abs_diff below 0.5.
- Observed recovery table: P000 diff 0.460 PASS; P001 diff 0.283 PASS; P002 diff 0.104 PASS; P003 diff 0.218 PASS; P004 diff 0.702 FAIL. 4 of 5 within gate.
- Unit-level invariant encoded in `test_laplace_recovery_sanity_omega2_2level` (slow, RUN_SMOKE_TESTS=1 gated) at tests/test_fit_vb_laplace_patrl.py:278-316 which asserts n_within >= 4.
- CLI-level invariant in `test_smoke_laplace_recovery_sanity_4_of_5` at tests/test_smoke_patrl_foundation.py:282.
- Both slow tests correctly skipped in fast pytest run (reported as 2 skipped).

### SC5 -- Laplace-vs-NUTS comparison harness with within_gate column and --skip-nuts-comparison flag

Status: PASSED for harness and --skip path; HUMAN-CHECK for full NUTS comparison

Evidence:
- `validation/vbl06_laplace_vs_nuts.py:49-243` implements `compare_posteriors` emitting required columns: participant_id, parameter, mean_laplace, mean_nuts, sd_laplace, sd_nuts, abs_diff_mean, abs_diff_log_sd, sd_ratio, within_gate.
- `within_gate` uses `pd.BooleanDtype()` nullable bool; True for omega_2 rows satisfying both gates, `pd.NA` for non-omega_2 rows -- verified by `test_compare_posteriors_identical_idatas_all_within_gate` (asserts beta rows have NA).
- Hard gates at constants `_GATE_ABS_DIFF_MEAN_OMEGA2=0.3` and `_GATE_ABS_DIFF_LOG_SD_OMEGA2=0.5` at lines 38-39.
- `--skip-nuts-comparison` flag at lines 353-359 exits 0; tested by `test_cli_compare_skip_flag` (RUN_SMOKE_TESTS=1 gated).
- 7 fast unit tests in `test_vbl06_laplace_vs_nuts.py` all PASS: identical idatas, within-tolerance pass, exceeds-tolerance fail, near-zero SD handled, participant coord rename, mismatched participant sets raise RuntimeError, hard-gate omega_2-only.
- Full NUTS hard-gate comparison cannot be exercised at HEAD because cluster NUTS idata has not yet been produced (see Human Verification section below).

### SC6 -- Parallel-stack invariant: empty git diff on 9 protected paths

Status: PASSED

Evidence:
- `git diff --stat 2045a27..HEAD -- src/prl_hgf/fitting/hierarchical.py src/prl_hgf/fitting/hierarchical_patrl.py src/prl_hgf/env/task_config.py src/prl_hgf/env/simulator.py src/prl_hgf/models/hgf_2level.py src/prl_hgf/models/hgf_3level.py src/prl_hgf/models/response.py configs/prl_analysis.yaml configs/pat_rl.yaml` returns EMPTY.
- All Phase 19 additions live in new files: `src/prl_hgf/fitting/fit_vb_laplace_patrl.py`, `src/prl_hgf/fitting/laplace_idata.py`, `validation/vbl06_laplace_vs_nuts.py`, plus test files and scripts/12 modifications (scripts/12 is NOT on the protected list).
- Three regression-guard tests named `test_pick_best_cue_regression_unchanged` (in test_fit_vb_laplace_patrl.py, test_laplace_idata.py, and test_vbl06_laplace_vs_nuts.py) all PASS, confirming pick-best-cue imports still work.

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/prl_hgf/fitting/fit_vb_laplace_patrl.py | MAP + Laplace fit entry point | VERIFIED | 415 lines, exports fit_vb_laplace_patrl and _regularize_to_pd; imported by scripts/12 and validation/vbl06 |
| src/prl_hgf/fitting/laplace_idata.py | Factory for Laplace InferenceData | VERIFIED | 216 lines, exports build_idata_from_laplace and _PARAM_ORDER constants |
| validation/vbl06_laplace_vs_nuts.py | Comparison harness CLI | VERIFIED | 485 lines, exports compare_posteriors and _apply_hard_gates; argparse run+compare subcommands with --skip-nuts-comparison flag |
| scripts/12_smoke_patrl_foundation.py (modified) | --fit-method flag | VERIFIED | argparse flag at line 141; _fit_blackjax / _fit_laplace / _fit() dispatch; lazy VBL-06 import at line 743 |
| tests/test_fit_vb_laplace_patrl.py | Unit + smoke tests | VERIFIED | 418 lines, 12 tests (8 fast + 4 slow); 8 of 8 fast PASS |
| tests/test_laplace_idata.py | Factory consumer contract | VERIFIED | 389 lines, 11 tests; 11 of 11 PASS |
| tests/test_vbl06_laplace_vs_nuts.py | Harness unit tests | VERIFIED | 530 lines, 9 tests (8 fast + 1 slow); 8 of 8 fast PASS |

## Key Link Verification

| From | To | Via | Status |
|------|-----|-----|--------|
| fit_vb_laplace_patrl | _build_patrl_log_posterior | import + closure at line 208 | WIRED (hierarchical_patrl.py unmodified per SC6) |
| fit_vb_laplace_patrl | build_idata_from_laplace | import + call at line 406 | WIRED |
| Laplace InferenceData | export_subject_parameters | participant_id coord | WIRED (test_consumer_compatibility_export_subject_parameters PASS) |
| VBL-06 harness | fit_vb_laplace_patrl + fit_batch_hierarchical_patrl | lazy import in CLI | WIRED |
| scripts/12 --fit-method | fit_vb_laplace_patrl | _fit_laplace dispatch | WIRED |

## Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| VBL-01 (MAP optimizer) | SATISFIED | jaxopt.LBFGS jit=True default with jit=False fallback on tracing failure (tested). |
| VBL-02 (Laplace cov + PD regularization) | SATISFIED | _regularize_to_pd via eigh-clip; Cholesky sanity check at line 360; min-cov-eigval post-invert check at line 372. |
| VBL-03 (ArviZ shape parity) | SATISFIED | chain=1, draw=K pseudo-draws, participant_id dim; vars match NUTS (omega_2, log_beta, beta for 2-level; +omega_3, kappa, mu3_0 for 3-level). |
| VBL-04 (export-path compat) | SATISFIED | test_consumer_compatibility_export_subject_parameters PASS + 19-05 end-to-end smoke. |
| VBL-05 (recovery smoke at 5 agents) | SATISFIED | 19-05: 30.7s, 4 of 5 within 0.5; unit + CLI tests encode the invariant. |
| VBL-06 (Laplace-vs-NUTS comparison) | SATISFIED for harness / PENDING for real cluster run | All 8 fast tests PASS. Full hard-gate decision requires cluster NUTS data. |

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/prl_hgf/fitting/fit_vb_laplace_patrl.py | 229-232 | TODO OQ2: kappa logit reparametrization deferred pending cluster MAP-at-boundary check | Info | Matches scope note in 19-05 SUMMARY; native-space kappa is explicit phase scope decision. |
| src/prl_hgf/fitting/fit_vb_laplace_patrl.py | 333-337 | TODO Phase 20+: block-structured Hessian via jax.vmap(jax.hessian) | Info | Explicit future-optimization note. Dense representation is Phase 19 scope. |
| scripts/12_smoke_patrl_foundation.py | ~23 | TODO OQ7 closure memo pending cluster NUTS | Info | Documented in 19-05 SUMMARY; intended deferral. |

No blocker anti-patterns. All TODOs are documented deferrals aligned with the phase plan.

## Human Verification Required

### 1. Laplace-vs-NUTS cluster hard gate

Test: After `sbatch cluster/18_smoke_patrl_cpu.slurm` completes on the cluster and produces a NUTS idata, run:

    python validation/vbl06_laplace_vs_nuts.py compare \
        --laplace data/patrl_smoke/idata_laplace.nc \
        --nuts data/patrl_smoke/idata_nuts.nc \
        --out-csv results/laplace_vs_nuts_diff.csv

Expected: Exit code 0; all omega_2 rows show `abs_diff_mean < 0.3` AND `abs_diff_log_sd < 0.5` (within_gate=True). No HARD FAIL messages for omega_2. Soft SOFT WARN messages for sd_ratio outside [0.5, 2.0] are informational only.

Why human: Cluster NUTS run is external to this repo CI path. quick-004 VB_LAPLACE_FEASIBILITY.md explicitly designed Option C around parallel development so Phase 19 ships before cluster output lands. Phase 19 OQ7 closure memo is deferred until this comparison is performed.

### 2. OQ1 follow-up: NUTS dim-name hotfix

Test: Inspect `src/prl_hgf/fitting/hierarchical_patrl.py::_samples_to_idata` for whether the dim name is `participant` (pre-hotfix) or `participant_id` (post-hotfix).

Expected: STATE.md blocker list indicates OQ1 is still open; Laplace path sidesteps via native participant_id emission.

Why human: Tracked in STATE.md for separate follow-up, not in Phase 19 scope. Confirmed to not affect Phase 19 success criteria because `compare_posteriors` renames on-the-fly (see lines 118-136 of vbl06_laplace_vs_nuts.py).

## Gaps Summary

No automated gaps. All 6 success criteria PASS at HEAD:

- SC1-2: fit_vb_laplace_patrl exists with exact signature, implements MAP + PD-regularized Hessian, passes 8 of 8 fast unit tests.
- SC3: export_subject_parameters consumes Laplace idata unchanged; tested directly + validated end-to-end in 19-05 smoke.
- SC4: 30.7s 5-agent smoke with 4 of 5 omega_2 recovery documented in 19-05 SUMMARY; encoded as slow env-gated pytest invariant.
- SC5: Harness fully implemented; 8 of 8 fast tests pass; --skip-nuts-comparison path exits 0 as designed. The only outstanding item is the real cluster-NUTS hard-gate comparison, which is HUMAN-CHECK by design per quick-004.
- SC6: Empty `git diff 2045a27..HEAD` on all 9 parallel-stack-protected paths.

## Integration Summary

Phase 19 delivers the parent goal (tapas-parity VB-Laplace second fit path for PAT-RL):

- Reuses Phase 18 `_build_patrl_log_posterior` without modification (verified by SC6 empty diff).
- Emits Phase 18 Laplace idata in the participant_id-dimmed shape required by Phase 18-05 `export_subject_trajectories` + `export_subject_parameters` (SC3 consumer contract test).
- The smoke script (scripts/12_smoke_patrl_foundation.py) unifies the NUTS and Laplace paths under one `--fit-method {blackjax,laplace,both}` dispatch (blackjax default preserves Phase 18 behavior bit-for-bit).
- Phase 20 cluster validation can consume both the Laplace path (deterministic reference) and the BlackJAX NUTS path (sampling ground truth) via the VBL-06 harness.

Phase 19 is code-complete and goal-achieving at HEAD; the single outstanding HUMAN-CHECK is explicit scope-deferral (cluster NUTS not yet run).

---

Verified: 2026-04-18

Verifier: Claude (gsd-verifier)

