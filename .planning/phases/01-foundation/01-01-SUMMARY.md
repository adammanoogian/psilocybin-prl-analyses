---
phase: 01-foundation
plan: "01"
subsystem: infra
tags: [python, setuptools, pyhgf, jax, pymc, pyyaml, ruff, mypy, pytest, dataclasses]

# Dependency graph
requires: []
provides:
  - Installable prl-hgf Python package (pip install -e .[dev])
  - Root config.py with PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, FIGURES_DIR, CONFIGS_DIR, SCRIPTS_DIR, DOCS_DIR, NOTEBOOKS_DIR path constants
  - configs/prl_analysis.yaml: full PRL pick_best_cue task structure plus simulation and fitting parameters
  - prl_hgf.env.task_config.load_config() returning validated AnalysisConfig dataclass hierarchy
  - Full directory scaffold: src/prl_hgf/{env,models,fitting,analysis}, configs/, scripts/, tests/, validation/, notebooks/, output/, figures/, docs/
  - CLAUDE.md, README.md, .gitignore, pyproject.toml with ruff/mypy/pytest tool configuration
affects:
  - 01-02-PLAN (task environment simulator reads from configs/prl_analysis.yaml via load_config)
  - 02-models (imports prl_hgf.env.task_config for task structure)
  - All subsequent phases (depend on installable package and config system)

# Tech tracking
tech-stack:
  added:
    - pyhgf>=0.2.8,<0.3 (JAX-backed HGF with PyMC integration)
    - jax>=0.4.26,<0.4.32
    - jaxlib>=0.4.26,<0.4.32
    - pymc>=5.25.1
    - numpy>=2.0.0,<3.0
    - pyyaml>=6.0
    - ruff>=0.4 (formatter + linter)
    - mypy>=1.10
    - pytest>=8.0 + pytest-cov>=5.0
  patterns:
    - Config via single root config.py with Path constants (no scattered hardcoded paths)
    - Task parameters in YAML, loaded via typed dataclass hierarchy with __post_init__ validation
    - Frozen dataclasses for immutable, validated config objects
    - from __future__ import annotations at top of every module
    - NumPy-style docstrings enforced by ruff pydocstyle convention = numpy
    - Three-layer naming: math symbols in internals, descriptive at API boundaries, domain English in scripts

key-files:
  created:
    - pyproject.toml
    - config.py
    - src/prl_hgf/__init__.py
    - src/prl_hgf/env/__init__.py
    - src/prl_hgf/models/__init__.py
    - src/prl_hgf/fitting/__init__.py
    - src/prl_hgf/analysis/__init__.py
    - configs/prl_analysis.yaml
    - src/prl_hgf/env/task_config.py
    - CLAUDE.md
    - README.md
    - .gitignore
  modified: []

key-decisions:
  - "Used ds_env (Python 3.10 conda env) as the installation environment — system Python 3.13 excluded by pyhgf 0.2.8 Requires-Python <=3.13 constraint"
  - "Merged task config and analysis config into one prl_analysis.yaml (not separate files) for single source of truth"
  - "Frozen dataclasses for all config types — immutability prevents accidental mutation in downstream code"
  - "env/ pattern in .gitignore changed to /env/ (root-anchored) to avoid ignoring src/prl_hgf/env/ subdirectory"

patterns-established:
  - "Config pattern: root config.py for paths, YAML for domain parameters, load_config() as the typed interface"
  - "Validation pattern: frozen dataclass + __post_init__ raising ValueError with expected vs. actual message"
  - "Import pattern: absolute imports only, from __future__ import annotations in every module"

# Metrics
duration: 21min
completed: 2026-04-04
---

# Phase 1 Plan 01: Project Scaffold and Config System Summary

**Installable prl-hgf package with setuptools build, pinned pyhgf/JAX/PyMC dependencies, root config.py path constants, and YAML-driven task config with typed frozen-dataclass validation**

## Performance

- **Duration:** 21 min
- **Started:** 2026-04-04T18:22:26Z
- **Completed:** 2026-04-04T18:43:07Z
- **Tasks:** 2
- **Files modified:** 12 created, 0 modified

## Accomplishments

- Installable `prl-hgf` package: `pip install -e ".[dev]"` succeeds in Python 3.10 conda environment with all dependencies (pyhgf 0.2.8, JAX, PyMC, numpy, pandas, scipy, matplotlib, seaborn, arviz, ipywidgets, pyyaml, ruff, mypy, pytest)
- Full project directory structure with all subpackages (env, models, fitting, analysis) as stubs ready for subsequent phases
- `configs/prl_analysis.yaml` encodes the complete PRL pick_best_cue task structure (3 cues, 4 phases, reward probabilities, partial feedback) plus simulation group distributions, session deltas, and fitting parameters — single source of truth
- `prl_hgf.env.task_config.load_config()` returns a validated, immutable `AnalysisConfig` dataclass that rejects invalid values (negative trial counts, probabilities > 1, invalid phase types, mismatched lengths) with expected vs. actual error messages

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project scaffold and pyproject.toml** - `2b92d70` (feat)
2. **Task 2: Create unified YAML config and config loader with validation** - `002909a` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `pyproject.toml` - setuptools build config, pinned dependencies, ruff/mypy/pytest tool configuration
- `config.py` - root Path constants: PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, FIGURES_DIR, CONFIGS_DIR, SCRIPTS_DIR, DOCS_DIR, NOTEBOOKS_DIR
- `src/prl_hgf/__init__.py` - package root with `__version__ = "0.1.0"`
- `src/prl_hgf/env/__init__.py` - task environment subpackage
- `src/prl_hgf/models/__init__.py` - HGF models subpackage stub
- `src/prl_hgf/fitting/__init__.py` - fitting subpackage stub
- `src/prl_hgf/analysis/__init__.py` - analysis subpackage stub
- `configs/prl_analysis.yaml` - full PRL task config + simulation + fitting + analysis parameters
- `src/prl_hgf/env/task_config.py` - frozen dataclass hierarchy + `load_config()` with yaml.safe_load
- `CLAUDE.md` - project AI guidelines (task structure, model params, conventions, architecture decisions)
- `README.md` - setup instructions, directory structure, config usage, key references
- `.gitignore` - excludes output/, figures/, data/, __pycache__, .venv, etc.

## Decisions Made

- **Python 3.10 environment:** System Python 3.13 is excluded by pyhgf 0.2.8 (`Requires-Python >=3.10,<=3.13`). The existing `ds_env` conda environment (Python 3.10.18) was used for installation and all verification.
- **Single merged YAML:** Task config and analysis config merged into one `prl_analysis.yaml` rather than two files, keeping all domain parameters in one place.
- **Frozen dataclasses:** All config types are frozen, preventing accidental mutation in downstream pipeline code and enabling use as dict keys if needed.
- **Root-anchored .gitignore for env/:** Changed `env/` to `/env/` in .gitignore to avoid accidentally ignoring `src/prl_hgf/env/` subdirectory.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed .gitignore `env/` pattern ignoring src subpackage**

- **Found during:** Task 1 (staging files for commit)
- **Issue:** The pattern `env/` in .gitignore matched `src/prl_hgf/env/` at any depth, causing git to ignore the subpackage directory
- **Fix:** Changed `env/` to `/env/` (root-anchored) for venv/ENV/env patterns; applied same anchoring to venv/ and ENV/
- **Files modified:** .gitignore
- **Verification:** `git add src/prl_hgf/env/__init__.py` succeeded after fix
- **Committed in:** 2b92d70 (Task 1 commit)

**2. [Rule 1 - Bug] Removed unused `field` import from task_config.py**

- **Found during:** Task 2 (ruff check verification)
- **Issue:** `from dataclasses import dataclass, field` — `field` was imported but not used; ruff F401
- **Fix:** Changed to `from dataclasses import dataclass`
- **Files modified:** src/prl_hgf/env/task_config.py
- **Verification:** `ruff check src/prl_hgf/env/task_config.py` passed with no errors
- **Committed in:** 002909a (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 - Bug)
**Impact on plan:** Both were minor correctness issues caught during verification. No scope creep.

## Issues Encountered

- `conda run -n ds_env python -c "..."` does not support multiline Python scripts as arguments on Windows. Worked around by writing temporary `.py` files and running them with `conda run -n ds_env python <path>`. Temp files deleted before commit.

## User Setup Required

None - no external service configuration required. Use `ds_env` conda environment for all development until a project-specific venv is created in Phase 1 Plan 02.

## Next Phase Readiness

- Package scaffold complete and installable in Python 3.10 environment
- Config system proven: `load_config()` returns correct values (n_cues=3, n_phases=4) and rejects invalid inputs
- Ready for Phase 1 Plan 02: task environment simulator (trial sequence generation from config)
- Note: A project-specific virtual environment (`.venv`) using Python 3.10 may be worth creating in Plan 02 to avoid relying on the shared `ds_env` conda environment

---
*Phase: 01-foundation*
*Completed: 2026-04-04*
