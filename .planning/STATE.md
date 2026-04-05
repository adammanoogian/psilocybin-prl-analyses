# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Phase 4 (Fitting) — 04-01 complete (core MCMC engine); ready for 04-02 (batch fitting).

## Current Position

Phase: 4 of 7 (Fitting)
Plan: 1 of 2+ in phase (04-01 complete)
Status: In progress — Phase 4 Plan 1 complete; ready for 04-02 (batch fitting)
Last activity: 2026-04-05 — Completed 04-01-PLAN.md (core MCMC fitting engine)

Progress: [███████░░░] ~50% (7 of ~14 plans complete)

## Accumulated Context

### Decisions

| Decision | Rationale | Phase |
|----------|-----------|-------|
| pyhgf 0.2.8 chosen as HGF library | JAX-backed, PyMC integration, Network API supports custom topologies | Planning |
| 3 parallel binary HGF branches (one per cue) with shared volatility parent | Literature default; independent-volatility is a v2 variant | Planning |
| Softmax + stickiness response model (β, ζ) | Standard in computational psychiatry literature | Planning |
| Partial feedback: only chosen cue receives reward input | Standard PRL protocol | Planning |
| Random-effects BMS (Rigoux et al. 2014) for model comparison | Formal Bayesian model comparison at group level | Planning |
| Jupyter ipywidgets for GUI | VSCode compatible, no web server needed | Planning |
| ω₃ recovery caveat acknowledged upfront | Known issue in literature; primary hypotheses focus on ω₂ and κ | Planning |
| Task environment reads from PRL pick_best_cue config | No hardcoded task structure anywhere | Planning |
| Mixed-effects model for second-level group analysis | group × session × phase | Planning |
| ds_env (Python 3.10 conda env) as installation environment | pyhgf 0.2.8 Requires-Python <=3.13 excludes system Python 3.13 | 01-01 |
| Single merged prl_analysis.yaml | Task + analysis params in one file for single source of truth | 01-01 |
| Frozen dataclasses for all config types | Immutability prevents accidental mutation in downstream pipeline | 01-01 |
| /env/ (root-anchored) in .gitignore | Prevents ignoring src/prl_hgf/env/ subpackage | 01-01 |
| Phase n_trials 40 → 30 per phase | Plan spec: 3 sets × 4 phases × 30 + 3 × 20 transfer = 420 trials total | 01-02 |
| TransferConfig as separate dataclass from PhaseConfig | Transfer has no name field; avoids unused required field | 01-02 |
| pytest pythonpath includes "." (project root) | Root config.py must be importable during test collection | 01-02 |
| net.edges is a tuple (positional), not a dict | pyhgf 0.2.8: node N accessed as net.edges[N] by position index | 02-01 |
| extract_beliefs uses "mean" for continuous nodes and "expected_mean" for binary nodes | "mean" is log-odds posterior; "expected_mean" is sigmoid probability in [0,1] | 02-01 |
| observed mask must use int dtype (not bool) | JAX tracing multiplies observed against PE; int required for correctness | 02-01 |
| Response function uses expected_mean from binary INPUT_NODES (0,2,4) as mu1_k | Sigmoid-transformed P(reward\|cue k) in [0,1]; continuous-node log-odds not used in softmax | 02-02 |
| First trial stickiness = 0 via sentinel prev_choice=-1 | jnp.concatenate([-1], choices[:-1]) ensures no stickiness on first trial | 02-02 |
| jax.nn.log_softmax used (not manual formula) | Available in JAX 0.4+; cleaner and numerically stable | 02-02 |
| Test fixtures for response tests use helper functions, not session-scoped fixtures | JAX network state is mutable after input_data(); fresh network prevents state leakage | 02-02 |
| simulate_agent uses attribute carry (net.attributes = net.last_attributes) after each 1-trial input_data call | pyhgf 0.2.8 always restarts scan from self.attributes; carry pattern threads state forward correctly | 03-01 |
| Prior beliefs read from net.attributes[INPUT_NODE]["expected_mean"] BEFORE input_data | Gives P(reward\|cue) as prior for choice generation at each trial; cleaner than reading node_trajectories | 03-01 |
| Accuracy test threshold >= 0.80 (not strict >) | Single stochastic runs can hit exactly 80.0% at boundary; >= correctly represents ">80%" criterion | 03-01 |
| Numpy (not JAX) for simulation softmax | No gradients needed in simulation path; avoids JAX overhead | 03-01 |
| Seed array drawn upfront from master_seed before batch loop | Ensures reproducibility is not sensitive to loop order; changing n_participants_per_group does not alter earlier participant seeds | 03-02 |
| JIT pre-warm in batch.py (not agent.py) | Keeps agent.py single-responsibility; warmup is a batch-level orchestration concern | 03-02 |
| session_labels built as ["baseline"] + session_cfg.session_labels | Avoids hardcoding full 3-session list; derives non-baseline labels from YAML session_deltas config | 03-02 |
| Same rng_sim used for sample_participant_params and simulate_agent | RNG state advances after parameter sampling, providing additional entropy for trial choices | 03-02 |
| Two-Op split pattern (\_GradOp + \_LogpOp) wrapping JAX lax.scan | pyhgf pattern: forward Op delegates grad to separate GradOp; both instantiated once in factory scope | 04-01 |
| Shallow-copy parameter injection (dict(base\_attrs) + dict(attrs[idx])) | deepcopy breaks JAX traceability; shallow copy preserves it for lax.scan gradient flow | 04-01 |
| expected\_mean from binary INPUT\_NODES (0,2,4) for softmax mu1 | Sigmoid P(reward\|cue) in [0,1] is the correct quantity; continuous-node log-odds are NOT used | 04-01 |
| NaN guard returns -jnp.inf (not +inf) in logp Op | logp semantic: -inf = reject proposal; +inf would be incorrect and confuse NUTS | 04-01 |
| Kappa injected at both edge endpoints simultaneously | pyhgf stores coupling at both ends: node 6 volatility\_coupling\_children AND nodes 1,3,5 volatility\_coupling\_parents | 04-01 |
| omega\_2 prior upper=0.0 mandatory; 3-level NaN boundary ~-1.2 | 3-level model (shared volatility node) produces NaN for omega\_2 >= ~-1.2; prior with mu=-3 keeps sampler safe | 04-01 |
| cores=1 default on Windows for fit\_participant | JAX cross-process state issues on Windows; re-test cores=4 if batch runtime exceeds 8 hours | 04-01 |

### Pending Todos

- Phase 4, Plan 2: Batch fitting loop across all 180 participant-sessions
- Consider creating project-specific .venv with Python 3.10 (deferred from Phase 1)
- batch test suite is ~6-7 min per full run; consider excluding from CI fast runs with `-k "not slow"`

### Blockers/Concerns

- 3-level model NaN boundary at omega_2 >= ~-1.2 (handled by prior upper=0.0, but initial PyMC point shows -inf in point_logps; NUTS handles gracefully)
- ω₃ parameter recovery expected to be challenging (known issue in literature)
- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env or a Python 3.10 venv
- JAX forward pass takes ~1s per call due to JIT compilation (first call per session); acceptable for simulation but may slow fitting iteration
- `conda run -n ds_env python -c "..."` fails for multi-line scripts on Windows (conda 25.7.0); use a temp script file instead

## Session Continuity

Last session: 2026-04-05T22:07:00Z
Stopped at: Completed 04-01-PLAN.md — core MCMC fitting engine (Phase 4, plan 1)
Resume file: None — continue with 04-02-PLAN.md (batch fitting)
