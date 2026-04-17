---
phase: 18-pat-rl-task-adaptation
plan: "02"
subsystem: env
tags: [pat-rl, trial-generator, binary-state, hazard, delta-hr, numpy-rng]

dependency-graph:
  requires: ["18-01"]
  provides: ["pat_rl_sequence.py", "PATRLTrial", "generate_session_patrl"]
  affects: ["18-03", "18-04", "18-05"]

tech-stack:
  added: []
  patterns:
    - "SeedSequence 4-way spawn for statistically independent RNG streams"
    - "Frozen dataclass for immutable trial records"
    - "State-conditioned Gaussian sampling with hard clamp bounds"

key-files:
  created:
    - src/prl_hgf/env/pat_rl_sequence.py
    - tests/test_env_pat_rl_sequence.py
  modified: []

decisions:
  - id: "SeedSequence-4-way-spawn"
    choice: "np.random.SeedSequence(seed).spawn(4) -> (state, mag, dhr, reserved)"
    rationale: "SeedSequence guarantees independence between streams even when seed integers are close; task-ID integer seeding would give correlated streams across participants"
  - id: "state-carried-forward-across-runs"
    choice: "initial_state for run N+1 = last state of run N"
    rationale: "Biological realism: context state is continuous; no artificial reset at run boundaries. Avoids biasing run-1 outcomes toward safe state."

metrics:
  duration: "~18 minutes"
  completed: "2026-04-17"
---

# Phase 18 Plan 02: PAT-RL Trial Sequence Generator Summary

**One-liner:** Binary-state hazard trial generator with SeedSequence-spawned RNG, 2x2 magnitude sampling, state-conditioned Delta-HR stub, and cumulative outcome timing.

## What Was Built

`src/prl_hgf/env/pat_rl_config.py` was already available from Plan 18-01. This plan adds the trial sequence generator that consumes it.

### PATRLTrial schema

| Field | Type | Description |
|-------|------|-------------|
| `trial_idx` | int | Session-level index 0..191 |
| `run_idx` | int | Run index 0..3 |
| `trial_in_run` | int | Trial within run 0..47 |
| `regime` | str | "stable" or "volatile" |
| `state` | int | Binary context: 0=safe, 1=dangerous |
| `reward_mag` | float | Reward magnitude from 2x2 cell |
| `shock_mag` | float | Shock magnitude from 2x2 cell |
| `delta_hr` | float | Anticipatory Delta-HR covariate (bpm) |
| `outcome_time_s` | float | Cumulative outcome onset time (seconds) |

### RNG stream split (SeedSequence 4-way spawn)

```
SeedSequence(seed).spawn(4)
  -> rng_state    : Bernoulli hazard flips for binary state
  -> rng_mag      : 2x2 cell index draw for magnitudes
  -> rng_dhr      : Gaussian Delta-HR stub samples
  -> _reserved    : Future outcome probability draws (Plan 18-04)
```

Using `SeedSequence` instead of arithmetic seed manipulation ensures each stream is statistically independent. The reserved 4th stream is allocated now so future plans can add outcome draws without changing seeds for existing streams.

### Delta-HR override for real-subject data

```python
trials = generate_session_patrl(
    config=cfg,
    seed=participant_id,
    delta_hr_override=subject_hr_array,  # shape (192,)
)
```

When `delta_hr_override` is provided, the stub is bypassed. Values are still clipped to `config.task.delta_hr_stub.bounds` ([-15, 10] bpm) to prevent extreme outliers from corrupting DCM regressors (Plan 18-05 trajectory export).

### Outcome timing formula

```
outcome_time_s[t] = run_offset + trial_in_run * trial_duration_s + cue_duration_s + anticipation_s
run_offset = run_idx * (trials_per_run * trial_duration_s + 15.0)
```

With default config (trial_duration_s=11.0, 48 trials/run, 15s gap):
- First trial outcome: 7.0 s (1.5 cue + 5.5 anticipation)
- Within-run spacing: 11.0 s
- Run boundary gap: 11.0 + 15.0 = 26.0 s minimum

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| SeedSequence 4-way spawn for state/mag/dHR/reserved | Task-ID integer seeding gives correlated streams; SeedSequence guarantees independence (existing project decision extended to PAT-RL) |
| Carry last state across run boundaries | Biological realism; avoids artificial reset artifact |
| delta_hr_override clipped to bounds before use | Prevents extreme real-subject outliers from corrupting trajectory export |

## Tests

9 pytest cases in `tests/test_env_pat_rl_sequence.py`:

| Test | Coverage |
|------|---------|
| `test_session_shape` | 192 trials, sequential idx, {0,1} states |
| `test_run_structure` | run_idx, trial_in_run, regime labels correct |
| `test_hazard_rate_approximate` | mean_stable < mean_volatile; within 3SD of Bernoulli expectation over 100 sessions |
| `test_magnitudes_cover_2x2` | All 4 (reward, shock) combos appear ≥10 times |
| `test_delta_hr_state_conditioned` | mean(dHR|dangerous) < 0, mean(dHR|safe) ≈ 0 (±0.3), all values in [-15, 10] over 500 sessions |
| `test_determinism` | Identical lists from same seed |
| `test_delta_hr_override_applied` | Override of 2.5 applied to all 192 trials |
| `test_outcome_time_s_monotone_increasing` | Strictly increasing; within-run spacing = 11.0 s; run gap ≥ 26.0 s |
| `test_pick_best_cue_still_loadable` | `prl_hgf.env.simulator.generate_session` imports without error |

## Verification Checklist

- [x] `src/prl_hgf/env/pat_rl_sequence.py` exports all 6 symbols in `__all__`
- [x] 9 pytest cases all pass
- [x] pick_best_cue regression: 34/34 tests pass (simulator, response, pat_rl_config)
- [x] ruff clean on both new files
- [x] mypy clean on both new files
- [x] Module does NOT import `simulator` (only in docstring)
- [x] Hazard-driven reversals verified statistically (100 sessions)
- [x] 2x2 magnitude coverage verified
- [x] Determinism under fixed seed verified

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

Plan 18-03 (HGF model builders for PAT-RL) can consume `trial.state` directly as binary input array. No interface changes anticipated.

Plan 18-05 (trajectory export) needs `outcome_time_s` and `run_idx` — both present in `PATRLTrial`.

Real-subject Delta-HR integration (Plan 18-04 fitting orchestrator): pass subject HR array as `delta_hr_override` to `generate_session_patrl`.
