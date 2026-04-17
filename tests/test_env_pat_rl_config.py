"""Unit tests for the PAT-RL config loader (src/prl_hgf/env/pat_rl_config.py).

Covers round-trip loading, computed properties, and all __post_init__
validators. Also includes a pick_best_cue regression test to verify the
parallel-stack isolation: PAT-RL changes must not break the existing
pick_best_cue load_config path.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from prl_hgf.env.pat_rl_config import PATRLConfig, load_pat_rl_config

# ---------------------------------------------------------------------------
# Helper: write a modified config to tmp_path
# ---------------------------------------------------------------------------


def _load_default_raw() -> dict[str, Any]:
    """Load the default pat_rl.yaml as a plain dict."""
    from config import CONFIGS_DIR

    return yaml.safe_load((CONFIGS_DIR / "pat_rl.yaml").read_text())


def _write_config(tmp_path: Path, overrides: dict[str, Any]) -> Path:
    """Deep-copy default YAML dict, apply overrides, write to tmp_path.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    overrides : dict[str, Any]
        Nested dict of values to overlay onto the default config.  Uses
        recursive merge so callers can target deep keys without rewriting
        the entire document.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    raw = copy.deepcopy(_load_default_raw())
    _deep_merge(raw, overrides)
    out = tmp_path / "pat_rl_override.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in place."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> PATRLConfig:
    """Default PATRLConfig loaded once per module."""
    return load_pat_rl_config()


# ---------------------------------------------------------------------------
# Test 1: round-trip — check all top-level fields present and correct
# ---------------------------------------------------------------------------


def test_load_default_config_round_trip(cfg: PATRLConfig) -> None:
    """load_pat_rl_config() returns PATRLConfig with expected top-level values."""
    assert isinstance(cfg, PATRLConfig)
    assert cfg.task.n_trials == 192, (
        f"Expected task.n_trials=192, got {cfg.task.n_trials}"
    )
    assert cfg.task.n_runs == 4, (
        f"Expected task.n_runs=4, got {cfg.task.n_runs}"
    )
    assert cfg.task.trials_per_run == 48, (
        f"Expected task.trials_per_run=48, got {cfg.task.trials_per_run}"
    )
    assert cfg.task.hazards.stable == pytest.approx(0.03), (
        f"Expected hazards.stable=0.03, got {cfg.task.hazards.stable}"
    )
    expected_phenotypes = {
        "healthy",
        "anxious",
        "reward_sensitive",
        "anxious_reward_sensitive",
    }
    assert set(cfg.simulation.phenotypes.keys()) == expected_phenotypes, (
        f"Expected phenotype keys {sorted(expected_phenotypes)}, "
        f"got {sorted(cfg.simulation.phenotypes.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 2: computed property — trial_duration_s
# ---------------------------------------------------------------------------


def test_trial_duration_computed(cfg: PATRLConfig) -> None:
    """TimingConfig.trial_duration_s equals sum of its four components."""
    t = cfg.task.timing
    expected = t.cue_duration_s + t.anticipation_s + t.outcome_duration_s + t.iti_s
    assert cfg.task.timing.trial_duration_s == pytest.approx(expected), (
        f"Expected trial_duration_s={expected}, "
        f"got {cfg.task.timing.trial_duration_s}"
    )
    # Default YAML: 1.5 + 5.5 + 2.0 + 2.0 = 11.0
    assert cfg.task.timing.trial_duration_s == pytest.approx(11.0), (
        f"Expected default trial_duration_s=11.0, "
        f"got {cfg.task.timing.trial_duration_s}"
    )


# ---------------------------------------------------------------------------
# Test 3: n_runs * trials_per_run == n_trials enforced
# ---------------------------------------------------------------------------


def test_n_runs_times_trials_per_run_enforced(tmp_path: Path) -> None:
    """ValueError raised when n_runs * trials_per_run != n_trials.

    Uses n_trials=192, n_runs=4, trials_per_run=50 → product=200 != 192.
    Error message must contain both 192 and 200.
    """
    cfg_path = _write_config(
        tmp_path,
        {"task": {"n_trials": 192, "n_runs": 4, "trials_per_run": 50}},
    )
    with pytest.raises(ValueError, match="192") as exc_info:
        load_pat_rl_config(cfg_path)
    assert "200" in str(exc_info.value), (
        f"Expected '200' in error message, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Test 4: hazard bounds enforced
# ---------------------------------------------------------------------------


def test_hazard_bounds_enforced(tmp_path: Path) -> None:
    """ValueError raised when hazards.stable is outside (0, 1).

    Error message must mention '(0, 1)' and the bad value '1.5'.
    """
    cfg_path = _write_config(
        tmp_path,
        {"task": {"hazards": {"stable": 1.5, "volatile": 0.10}}},
    )
    with pytest.raises(ValueError, match="1.5") as exc_info:
        load_pat_rl_config(cfg_path)
    assert "(0, 1)" in str(exc_info.value), (
        f"Expected '(0, 1)' in error message, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Test 5: outcome probabilities must sum to 1
# ---------------------------------------------------------------------------


def test_outcome_probs_must_sum_to_one(tmp_path: Path) -> None:
    """ValueError raised when safe contingency probs sum to != 1.

    Uses reward=0.5, shock=0.1, nothing=0.1 → sum=0.7. Error message must
    contain both '0.7' and '1.0'.
    """
    cfg_path = _write_config(
        tmp_path,
        {
            "task": {
                "contingencies": {
                    "safe": {"reward": 0.5, "shock": 0.1, "nothing": 0.1}
                }
            }
        },
    )
    with pytest.raises(ValueError, match="0.7") as exc_info:
        load_pat_rl_config(cfg_path)
    assert "1.0" in str(exc_info.value), (
        f"Expected '1.0' in error message, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Test 6: phenotype keys enforced
# ---------------------------------------------------------------------------


def test_phenotype_keys_enforced(tmp_path: Path) -> None:
    """ValueError raised when phenotypes dict is missing a required key.

    Drops 'anxious_reward_sensitive'; error message must mention the
    missing key.
    """
    raw = copy.deepcopy(_load_default_raw())
    del raw["simulation"]["phenotypes"]["anxious_reward_sensitive"]
    cfg_path = tmp_path / "pat_rl_missing_phenotype.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="anxious_reward_sensitive"):
        load_pat_rl_config(cfg_path)


# ---------------------------------------------------------------------------
# Test 7: magnitudes must be positive
# ---------------------------------------------------------------------------


def test_magnitudes_must_be_positive(tmp_path: Path) -> None:
    """ValueError raised when a reward magnitude level is not > 0.

    Uses reward_levels=[0, 5]; 0 is not strictly positive.
    """
    cfg_path = _write_config(
        tmp_path,
        {"task": {"magnitudes": {"reward_levels": [0, 5]}}},
    )
    with pytest.raises(ValueError):
        load_pat_rl_config(cfg_path)


# ---------------------------------------------------------------------------
# Test 8: pick_best_cue regression — existing loader unaffected
# ---------------------------------------------------------------------------


def test_pick_best_cue_still_loadable() -> None:
    """PAT-RL module import must not break pick_best_cue load_config.

    Regression: importing pat_rl_config must not mutate task_config.py or
    prl_analysis.yaml paths. Asserts cfg.task.n_cues == 3.
    """
    from prl_hgf.env.task_config import load_config

    pbc_cfg = load_config()
    assert pbc_cfg.task.n_cues == 3, (
        f"Expected pick_best_cue task.n_cues=3, got {pbc_cfg.task.n_cues}"
    )
