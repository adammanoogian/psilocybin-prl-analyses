---
phase: quick
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  - cluster/00_setup_env_gpu.sh
  - cluster/01_diagnostic_gpu.slurm
  - cluster/04_fit_mcmc_gpu.slurm
  - cluster/05_validation_gpu.slurm
  - cluster/06_analysis.slurm
  - cluster/99_push_results.slurm
  - cluster/submit_full_pipeline.sh
  - environment_gpu.yml
  - scripts/smoke_test.sh
autonomous: true

must_haves:
  truths:
    - "Cluster env setup script creates prl_gpu conda env with JAX CUDA, PyMC, pyhgf on M3"
    - "SLURM jobs correctly chain simulate -> fit 2level -> fit 3level -> validate -> group analysis"
    - "Smoke test runs the full 4-script pipeline locally with 2 participants/group and passes"
  artifacts:
    - path: "environment_gpu.yml"
      provides: "Minimal GPU conda env definition for MCMC fitting"
      contains: "prl_gpu"
    - path: "cluster/00_setup_env_gpu.sh"
      provides: "Env creation script for M3"
      contains: "module load miniforge3"
    - path: "cluster/01_diagnostic_gpu.slurm"
      provides: "GPU sanity check SLURM job"
      contains: "jax.devices"
    - path: "cluster/04_fit_mcmc_gpu.slurm"
      provides: "MCMC fitting SLURM job for both models"
      contains: "04_fit_participants.py"
    - path: "cluster/submit_full_pipeline.sh"
      provides: "Wave orchestrator submitting all jobs with dependency chains"
      contains: "sbatch"
    - path: "cluster/99_push_results.slurm"
      provides: "Auto-push results to git branch after pipeline completes"
    - path: "scripts/smoke_test.sh"
      provides: "Local end-to-end pipeline test with 2 participants/group"
      contains: "n_participants_per_group"
  key_links:
    - from: "cluster/submit_full_pipeline.sh"
      to: "cluster/04_fit_mcmc_gpu.slurm"
      via: "sbatch dependency chain"
      pattern: "sbatch.*04_fit"
    - from: "cluster/04_fit_mcmc_gpu.slurm"
      to: "scripts/04_fit_participants.py"
      via: "python invocation"
      pattern: "python.*04_fit_participants"
    - from: "scripts/smoke_test.sh"
      to: "scripts/03_simulate_participants.py"
      via: "sequential script chain"
      pattern: "python.*03_simulate"
---

<objective>
Create cluster infrastructure for GPU-accelerated MCMC fitting on Monash M3 and
a local smoke test to verify the full pipeline chains end-to-end.

Purpose: The full pipeline (30 participants/group, 4 chains x 1000 draws) takes
3-5 hours CPU. Moving to GPU on M3 makes iterating feasible. The smoke test
validates the pipeline chain before committing to long cluster runs.

Output: cluster/ directory with SLURM jobs, environment_gpu.yml, and a local
smoke_test.sh script.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@config.py
@configs/prl_analysis.yaml
@pyproject.toml
@scripts/03_simulate_participants.py
@scripts/04_fit_participants.py
@scripts/05_run_validation.py
@scripts/06_group_analysis.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create cluster infrastructure (environment + SLURM jobs + orchestrator)</name>
  <files>
    environment_gpu.yml
    cluster/00_setup_env_gpu.sh
    cluster/01_diagnostic_gpu.slurm
    cluster/04_fit_mcmc_gpu.slurm
    cluster/05_validation_gpu.slurm
    cluster/06_analysis.slurm
    cluster/99_push_results.slurm
    cluster/submit_full_pipeline.sh
  </files>
  <action>
    Create all cluster infrastructure files following the patterns in
    `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\cluster\` and
    `C:\Users\aman0087\Documents\Github\project_utils\templates\hpc\`.

    **environment_gpu.yml** — Minimal GPU conda env for MCMC fitting ONLY:
    - name: prl_gpu
    - channels: conda-forge, defaults
    - python=3.10 (matching project requires-python and existing ds_env)
    - conda deps: pandas>=2.0, numpy>=2.0, scipy>=1.10, pyyaml>=6.0
    - pip deps: jax[cuda12]>=0.4.26,<0.4.32 (pyhgf 0.2.8 pins jax<0.4.32),
      pymc>=5.25.1, pytensor, arviz>=0.21.0, pyhgf>=0.2.8,<0.3, groupBMC>=1.0
    - NO matplotlib, seaborn, bambi, pingouin, ipywidgets, ptitprince
      (not needed for fitting/validation on cluster; group analysis needs bambi
      but we can add it or run analysis locally)
    - IMPORTANT: Actually, scripts/06_group_analysis.py imports bambi. Either:
      (a) include bambi in the GPU env, or (b) skip group analysis on cluster.
      Decision: include bambi>=0.13.0 in the env since it is needed for the
      analysis SLURM job. Also include pingouin>=0.5.5 (used in group analysis).
      Do NOT include matplotlib/seaborn/ipywidgets (only needed for plots, and
      06 has --skip-plots flag).
    - Also include the project itself via `pip install -e .` in the setup script
      (NOT in environment_gpu.yml) so src/prl_hgf is importable.

    **cluster/00_setup_env_gpu.sh** — Adapted from rlwm reference:
    - module load miniforge3 (no CUDA module)
    - Conda envs dir: /scratch/fc37/$USER/conda (via CONDA_ENVS_DIRS export
      before env create, to avoid home quota)
    - Supports --update, --fresh, --help flags
    - After env create/update, runs `pip install -e .` in project root to install
      prl_hgf package into the env
    - Verifies: python version, jax version, jax device detection
    - env name: prl_gpu (not rlwm_gpu)

    **cluster/01_diagnostic_gpu.slurm** — Quick GPU diagnostic:
    - #SBATCH: job-name=prl_diag, time=00:30:00, mem=32G, gres=gpu:1,
      partition=gpu, cpus-per-task=4
    - Logs to cluster/logs/diagnostic_%j.out/.err
    - Activates prl_gpu env (try by name, fallback to /scratch/fc37/$USER/conda/envs/prl_gpu)
    - _PROJECT="${PROJECT:-fc37}" pattern for portability
    - Sets JAX_COMPILATION_CACHE_DIR=/scratch/$_PROJECT/$USER/.jax_cache_gpu
    - Runs inline Python that:
      1. Checks jax.devices() for GPU
      2. Runs a small JAX computation on GPU
      3. Imports pymc, pyhgf to verify they load
      4. Times a small matrix multiply (1000x1000) on GPU vs CPU
    - Prints summary

    **cluster/04_fit_mcmc_gpu.slurm** — MCMC fitting (the main compute job):
    - #SBATCH: job-name=prl_fit, time=12:00:00, mem=64G, gres=gpu:A40:1,
      partition=gpu, cpus-per-task=4
    - Logs to cluster/logs/fit_mcmc_%j.out/.err
    - Config vars: MODEL="${MODEL:-hgf_2level}", INPUT="${INPUT:-}"
    - Env activation with _PROJECT="${PROJECT:-fc37}" pattern
    - JAX compilation cache setup
    - GPU verification (inline python jax.devices check)
    - Runs: python scripts/04_fit_participants.py --model $MODEL [--input $INPUT]
    - If INPUT not set, omit --input flag (script has its own default)
    - Multi-model dispatch: if MODEL contains spaces (multiple models), submit
      separate jobs for each (same pattern as GPU template)
    - Job resource report at end (nvidia-smi, sacct)

    **cluster/05_validation_gpu.slurm** — Validation (parameter recovery + BMS):
    - #SBATCH: job-name=prl_valid, time=04:00:00, mem=32G, gres=gpu:1,
      partition=gpu, cpus-per-task=4
    - Runs: python scripts/05_run_validation.py --skip-waic
    - (skip WAIC by default on cluster; it requires idata .nc files)
    - Env/cache/GPU verification same pattern as 04

    **cluster/06_analysis.slurm** — Group analysis (CPU-only, uses bambi):
    - #SBATCH: job-name=prl_analysis, time=02:00:00, mem=32G,
      partition=comp, cpus-per-task=4 (NO GPU needed)
    - Runs: python scripts/06_group_analysis.py --model 2level --skip-plots
    - Then: python scripts/06_group_analysis.py --model 3level --skip-plots
    - Uses prl_gpu env (has bambi installed), but comp partition (no GPU)

    **cluster/99_push_results.slurm** — Auto-push results:
    - Adapted from project_utils template (slurm_push_results_TEMPLATE.slurm)
    - #SBATCH: job-name=push_results, time=00:15:00, mem=4G, partition=comp
    - Stage patterns specific to this project:
      - data/fitted/*.csv (fitting results)
      - results/validation/*.csv, results/validation/*.png
      - results/group_analysis/*.csv
      - figures/*.png
    - Same branch-push strategy as template (creates results/slurm-YYYYMMDD branch)

    **cluster/submit_full_pipeline.sh** — Wave orchestrator:
    - Wave 1: Simulate (python scripts/03_simulate_participants.py — run inline
      before submitting SLURM jobs, since it is fast ~30s and produces the input CSV)
    - Wave 2: Fit hgf_2level + Fit hgf_3level (parallel GPU jobs via 04_fit_mcmc_gpu.slurm)
    - Wave 3: Validation (depends on both fit jobs via afterok)
    - Wave 4: Group analysis (depends on validation via afterok, runs on comp partition)
    - Wave 5: Push results (depends on analysis via afterany)
    - Supports flags: --skip-push, --skip-validation, --models="hgf_2level" (subset)
    - Prints job IDs, monitoring commands, cancel command
    - Uses `sbatch --parsable` and `--dependency=afterok:JOBID` chains

    All SLURM scripts must:
    - Use `mkdir -p cluster/logs` before any output
    - Use `cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"` for project root
    - Use `export PYTHONUNBUFFERED=1` for real-time log output
    - Have consistent header documentation with usage examples
  </action>
  <verify>
    - All 8 files exist under cluster/ and environment_gpu.yml at project root
    - `bash -n cluster/00_setup_env_gpu.sh` passes (syntax check)
    - `bash -n cluster/submit_full_pipeline.sh` passes (syntax check)
    - Each .slurm file has #SBATCH directives and proper env activation
    - environment_gpu.yml is valid YAML (python -c "import yaml; yaml.safe_load(open('environment_gpu.yml'))")
    - No references to rlwm_gpu (should be prl_gpu everywhere)
    - No jax>=0.5.0 references (must be <0.4.32 for pyhgf compatibility)
  </verify>
  <done>
    cluster/ directory contains all 7 scripts + environment_gpu.yml at root.
    Scripts reference prl_gpu env, correct project paths, correct script names.
    SLURM dependency chains are correct (fit -> validate -> analyse -> push).
  </done>
</task>

<task type="auto">
  <name>Task 2: Create and run local smoke test (2 participants/group end-to-end)</name>
  <files>
    scripts/smoke_test.sh
  </files>
  <action>
    Create `scripts/smoke_test.sh` that runs the full pipeline locally with a
    minimal participant count to verify scripts chain end-to-end.

    The script must:
    1. Set a temporary override for n_participants_per_group. The cleanest approach:
       - Use Python to temporarily patch the YAML config in-memory, OR
       - Create a temporary modified config file, OR
       - (Simplest) Use sed to temporarily modify configs/prl_analysis.yaml,
         run the pipeline, then restore it.
       Best approach: Use a Python wrapper that patches the config at runtime.
       Actually, the simplest and safest approach is:
       - Copy configs/prl_analysis.yaml to configs/prl_analysis.yaml.bak
       - Use sed to change `n_participants_per_group: 30` to `n_participants_per_group: 2`
       - Run the pipeline
       - Restore the original config from .bak
       - Use a trap to ensure restoration on exit/error

    2. Pipeline steps (sequential):
       a. `python scripts/03_simulate_participants.py`
          - Expect output: data/simulated/simulated_participants.csv
          - Quick check: file exists and has > 0 rows
       b. `python scripts/04_fit_participants.py --model hgf_2level`
          - Expect output: data/fitted/hgf_2level_results.csv
       c. `python scripts/04_fit_participants.py --model hgf_3level`
          - Expect output: data/fitted/hgf_3level_results.csv
       d. `python scripts/05_run_validation.py --skip-waic`
          - Expect output: results/validation/ directory with recovery CSVs
       e. `python scripts/06_group_analysis.py --model 2level --skip-plots`
          - Expect output: results/group_analysis/ directory with CSVs

    3. After each step:
       - Check exit code (bail out on failure with descriptive error)
       - Print timing (use $SECONDS or `date`)
       - Verify expected output file exists

    4. At the end:
       - Print summary table: step name, duration, status, output files
       - Print total elapsed time
       - Clean up: remove the smoke test outputs (optional --keep flag to retain)

    5. Header documentation:
       ```
       # Usage:
       #   bash scripts/smoke_test.sh           # Run and clean up
       #   bash scripts/smoke_test.sh --keep     # Run and keep outputs
       ```

    6. Use `set -euo pipefail` and trap for config restoration.

    7. The script should work with the ds_env conda environment (already active
       or activated by the user). Do NOT activate conda in the script — just
       document the prerequisite.

    After creating the script, RUN IT to verify the full pipeline works:
    ```
    cd C:/Users/aman0087/Documents/Github/psilocybin_prl_analyses
    bash scripts/smoke_test.sh --keep
    ```

    The smoke test with 2 participants/group should complete in ~5-15 minutes
    (MCMC is the bottleneck: 2 groups x 2 participants x 3 sessions x 2 models
    = 24 participant-session fits, each ~10-30s).

    IMPORTANT: If the smoke test fails on any step, diagnose and fix the issue.
    The goal is a GREEN end-to-end run. Do NOT mark this task as done until the
    smoke test passes.
  </action>
  <verify>
    - `bash scripts/smoke_test.sh --keep` completes with exit code 0
    - data/simulated/simulated_participants.csv exists with trial data for 2 participants/group
    - data/fitted/hgf_2level_results.csv exists with fitted parameter rows
    - data/fitted/hgf_3level_results.csv exists with fitted parameter rows
    - results/validation/ contains recovery metrics
    - results/group_analysis/ contains group-level CSVs
    - configs/prl_analysis.yaml is restored to n_participants_per_group: 30
  </verify>
  <done>
    Smoke test script runs the full 5-step pipeline with 2 participants/group,
    all steps pass, output files are produced, and the config is cleanly restored.
  </done>
</task>

</tasks>

<verification>
1. `ls cluster/*.slurm cluster/*.sh environment_gpu.yml scripts/smoke_test.sh` — all files exist
2. `bash -n cluster/submit_full_pipeline.sh && bash -n cluster/00_setup_env_gpu.sh && bash -n scripts/smoke_test.sh` — no syntax errors
3. `python -c "import yaml; yaml.safe_load(open('environment_gpu.yml'))"` — valid YAML
4. `grep -r "rlwm_gpu" cluster/ environment_gpu.yml` — should return nothing (must be prl_gpu)
5. `grep "jax\[cuda12\]" environment_gpu.yml` — confirms CUDA JAX is specified
6. Smoke test passed (Task 2 execution)
</verification>

<success_criteria>
- 8 cluster files created (7 in cluster/ + environment_gpu.yml at root)
- All SLURM scripts use prl_gpu env, fc37 project, correct script paths
- Wave orchestrator chains: simulate -> fit_2level || fit_3level -> validate -> analyse -> push
- Smoke test passes end-to-end with 2 participants/group
- Config file (prl_analysis.yaml) is cleanly restored after smoke test
- No regressions: existing tests still pass after changes
</success_criteria>

<output>
After completion, create `.planning/quick/001-cluster-gpu-setup-and-smoke-test/001-SUMMARY.md`
</output>
