"""Central configuration for PRL HGF Analysis project.

This module contains all project paths and directory structure.
Task-specific parameters live in configs/prl_analysis.yaml
and are loaded via prl_hgf.env.task_config.load_config().
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
VALIDATION_DIR = RESULTS_DIR / "validation"
GROUP_ANALYSIS_DIR = RESULTS_DIR / "group_analysis"

for _directory in [OUTPUT_DIR, FIGURES_DIR, DOCS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)
