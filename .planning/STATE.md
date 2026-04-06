# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Validated simulation-to-inference pipeline for HGF models on PRL pick_best_cue data.
**Current focus:** Phase 5 (Validation) — BMS module complete; pipeline script (05-03) next.

## Current Position

Phase: 5 of 7 (Validation) — In progress
Plan: 2 of 3 in phase (05-01, 05-02 complete)
Status: BMS module complete — groupBMC wrapper, WAIC post-hoc, EP bar plot, and tests done
Last activity: 2026-04-06 — Completed 05-02-PLAN.md (Bayesian model comparison module)

Progress: [██████████░] ~71% (10 of ~14 plans complete)

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
| Two-Op split pattern (_GradOp + _LogpOp) wrapping JAX lax.scan | pyhgf pattern: forward Op delegates grad to separate GradOp; both instantiated once in factory scope | 04-01 |
| Shallow-copy parameter injection (dict(base_attrs) + dict(attrs[idx])) | deepcopy breaks JAX traceability; shallow copy preserves it for lax.scan gradient flow | 04-01 |
| expected_mean from binary INPUT_NODES (0,2,4) for softmax mu1 | Sigmoid P(reward\|cue) in [0,1] is the correct quantity; continuous-node log-odds are NOT used | 04-01 |
| NaN guard returns -jnp.inf (not +inf) in logp Op | logp semantic: -inf = reject proposal; +inf would be incorrect and confuse NUTS | 04-01 |
| Kappa injected at both edge endpoints simultaneously | pyhgf stores coupling at both ends: node 6 volatility_coupling_children AND nodes 1,3,5 volatility_coupling_parents | 04-01 |
| omega_2 prior upper=0.0 mandatory; 3-level NaN boundary ~-1.2 | 3-level model (shared volatility node) produces NaN for omega_2 >= ~-1.2; prior with mu=-3 keeps sampler safe | 04-01 |
| cores=1 default on Windows for fit_participant | JAX cross-process state issues on Windows; re-test cores=4 if batch runtime exceeds 8 hours | 04-01 |
| Per-participant seed = random_seed + flat_idx | Ensures reproducible independent seeds without upfront allocation; simple and auditable | 04-02 |
| FittingConfig dataclass added to task_config.py | Pipeline script reads MCMC settings from config.fitting; keeps all params in one place | 04-02 |
| NaN-filled fallback rows on fit failure | Failed participants appear in output with flagged=True; not silently dropped | 04-02 |
| Session-scoped pytest fixture for simulated data | Amortizes expensive JAX JIT compile across 4 Op tests; 50-trial slice for speed | 04-02 |
| zip(axes, params, strict=True) in plot loop | ruff B905 requires explicit strict= parameter; axes and params always same length by construction | 05-01 |
| Fixture offset scale sd=1.0 for all params | With only 9 test participants, kappa needed adequate inter-subject spread vs 0.05 recovery noise | 05-01 |
| Recovery DataFrame wide form: one row per participant-session | Enables direct column-wise comparison of true_* and fitted values | 05-01 |
| compute_recovery_metrics skips parameters not in recovery_df | 2-level model omits omega_3 and kappa; graceful skip avoids KeyError | 05-01 |
| groupBMC 1.0 package used (not from-scratch VB fallback) | pip install groupBMC succeeded; implements Rigoux et al. 2014 VB algorithm exactly | 05-02 |
| GroupBMC(L) called with L transposed to (n_models, n_subjects) | groupBMC API requires (n_models, n_subjects); internal representation uses (n_subjects, n_models) | 05-02 |
| bor extracted from bmc.F1()-bmc.F0() (not GroupBMCResult attribute) | GroupBMCResult does not expose bor; GroupBMC.get_result() computes it internally but does not store it | 05-02 |
| Dataset.sizes used instead of Dataset.dims for chain/draw counts | xarray FutureWarning: dims will return a set in future; sizes always returns a mapping | 05-02 |
| WAIC loglike_dim_0 is a single scalar per sample (not per-trial) | pm.Potential computes trial-sum logp; ArviZ warns but value is valid as model evidence | 05-02 |

### Pending Todos

- Phase 5: Pipeline script (05-03) integrating recovery analysis + BMS on batch fit output
- Consider creating project-specific .venv with Python 3.10 (deferred from Phase 1)
- batch test suite is ~6-7 min per full run; consider excluding from CI fast runs with `-k "not slow"`
- compute_batch_waic requires idata storage in fitting pipeline — 05-03 must add .nc file saving

### Blockers/Concerns

- 3-level model NaN boundary at omega_2 >= ~-1.2 (handled by prior upper=0.0, but initial PyMC point shows -inf in point_logps; NUTS handles gracefully)
- ω₃ parameter recovery expected to be challenging (known issue in literature) — caveat annotation added to scatter plots
- System Python 3.13 incompatible with pyhgf 0.2.8 — all work must use ds_env or a Python 3.10 venv
- JAX forward pass takes ~1s per call due to JIT compilation (first call per session); acceptable for simulation but may slow fitting iteration
- `conda run -n ds_env python -c "..."` fails for multi-line scripts on Windows (conda 25.7.0); use a temp script file instead
- Full 180-participant batch estimated at ~3.1 hours (2-level) or ~4.5 hours (3-level) sequential on CPU; monitor and consider cores=4 testing
- WAIC batch runtime: 180 participants × 2 models × 4 chains × 1000 draws = 1.44M logp evaluations; budget 30-60 min total

## Session Continuity

Last session: 2026-04-06T13:33:12Z
Stopped at: Completed 05-02-PLAN.md — BMS module (bms.py, tests, groupBMC dependency)
Resume file: None — continue with 05-03 (pipeline script for validation + model comparison)
