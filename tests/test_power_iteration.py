"""Unit tests for :mod:`prl_hgf.power.iteration` — core power pipeline.

Tests cover:

- ``build_arrays_from_sim`` shape and value correctness
- Partial feedback encoding (unchosen cues have observed=0)
- ``run_power_iteration`` return structure (mock-based)
- Schema conformance via ``write_parquet_row``
- Dry-run backward compatibility for the entry point script

Run with::

    pytest tests/test_power_iteration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from prl_hgf.power.iteration import build_arrays_from_sim
from prl_hgf.power.schema import POWER_SCHEMA, write_parquet_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim_subset(
    n_trials: int = 5,
    cue_sequence: list[int] | None = None,
    reward_sequence: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal sim_df subset for one participant-session.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    cue_sequence : list[int] or None
        Chosen cue per trial.  If None, all trials choose cue 0.
    reward_sequence : list[float] or None
        Reward per trial.  If None, alternates 1.0 and 0.0.

    Returns
    -------
    pd.DataFrame
        Subset with ``trial``, ``cue_chosen``, ``reward`` columns.
    """
    if cue_sequence is None:
        cue_sequence = [0] * n_trials
    if reward_sequence is None:
        reward_sequence = [1.0 if t % 2 == 0 else 0.0 for t in range(n_trials)]

    return pd.DataFrame(
        {
            "trial": list(range(n_trials)),
            "cue_chosen": cue_sequence,
            "reward": reward_sequence,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: build_arrays_from_sim — shapes and basic values
# ---------------------------------------------------------------------------


def test_build_arrays_from_sim_matches_build_arrays() -> None:
    """Array shapes are correct and chosen cue gets reward + observed=1."""
    subset = _make_sim_subset(
        n_trials=5,
        cue_sequence=[0, 1, 2, 0, 1],
        reward_sequence=[1.0, 0.0, 1.0, 0.0, 1.0],
    )

    input_arr, obs_arr, choices_arr = build_arrays_from_sim(subset)

    assert input_arr.shape == (5, 3)
    assert obs_arr.shape == (5, 3)
    assert choices_arr.shape == (5,)

    # Verify each trial: chosen cue gets reward value + observed=1
    for t in range(5):
        cue = int(subset.iloc[t]["cue_chosen"])
        reward = float(subset.iloc[t]["reward"])
        assert input_arr[t, cue] == reward
        assert obs_arr[t, cue] == 1


# ---------------------------------------------------------------------------
# Test 2: build_arrays_from_sim — partial feedback encoding
# ---------------------------------------------------------------------------


def test_build_arrays_from_sim_partial_feedback() -> None:
    """Unchosen cues have input_data=0.0 and observed=0."""
    subset = _make_sim_subset(
        n_trials=3,
        cue_sequence=[0, 1, 2],
        reward_sequence=[1.0, 1.0, 1.0],
    )

    input_arr, obs_arr, choices_arr = build_arrays_from_sim(subset)

    # Trial 0: cue 0 chosen -> cues 1,2 unchosen
    assert input_arr[0, 1] == 0.0
    assert input_arr[0, 2] == 0.0
    assert obs_arr[0, 1] == 0
    assert obs_arr[0, 2] == 0

    # Trial 1: cue 1 chosen -> cues 0,2 unchosen
    assert input_arr[1, 0] == 0.0
    assert input_arr[1, 2] == 0.0
    assert obs_arr[1, 0] == 0
    assert obs_arr[1, 2] == 0

    # Trial 2: cue 2 chosen -> cues 0,1 unchosen
    assert input_arr[2, 0] == 0.0
    assert input_arr[2, 1] == 0.0
    assert obs_arr[2, 0] == 0
    assert obs_arr[2, 1] == 0

    np.testing.assert_array_equal(choices_arr, [0, 1, 2])


# ---------------------------------------------------------------------------
# Test 3: run_power_iteration — return structure (mock-based)
# ---------------------------------------------------------------------------


def test_run_power_iteration_return_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_power_iteration returns 3 dicts with exactly 13 POWER_SCHEMA keys."""
    import prl_hgf.power.iteration as iteration_mod

    # Mock simulate_batch
    mock_sim_df = pd.DataFrame(
        {
            "participant_id": ["P001"] * 3,
            "group": ["psilocybin"] * 3,
            "session": ["baseline"] * 3,
            "trial": [0, 1, 2],
            "cue_chosen": [0, 1, 2],
            "reward": [1.0, 0.0, 1.0],
            "true_omega_2": [-3.0] * 3,
            "true_omega_3": [-6.0] * 3,
            "true_kappa": [1.0] * 3,
            "true_beta": [3.0] * 3,
            "true_zeta": [0.5] * 3,
        }
    )
    monkeypatch.setattr(
        iteration_mod, "simulate_batch", lambda cfg: mock_sim_df
    )

    # Mock fit_batch — returns (fit_df, idata_dict)
    mock_fit_df = pd.DataFrame(
        {
            "participant_id": ["P001"],
            "group": ["psilocybin"],
            "session": ["baseline"],
            "model": ["hgf_3level"],
            "parameter": ["omega_2"],
            "mean": [-3.0],
            "sd": [0.1],
            "hdi_3%": [-3.2],
            "hdi_97%": [-2.8],
            "r_hat": [1.01],
            "ess": [500.0],
            "flagged": [False],
        }
    )
    mock_idata = MagicMock()
    mock_idata_dict = {("P001", "psilocybin", "baseline"): mock_idata}

    monkeypatch.setattr(
        iteration_mod,
        "fit_batch",
        lambda *a, **kw: (mock_fit_df.copy(), dict(mock_idata_dict)),
    )

    # Mock compute_all_contrasts
    mock_contrasts = [
        {"sweep_type": "did_postdose", "bf_value": 5.0, "bf_exceeds": False},
        {"sweep_type": "did_followup", "bf_value": 2.0, "bf_exceeds": False},
        {"sweep_type": "linear_trend", "bf_value": 1.5, "bf_exceeds": False},
    ]
    monkeypatch.setattr(
        iteration_mod,
        "compute_all_contrasts",
        lambda *a, **kw: mock_contrasts,
    )

    # Mock _compute_bms_power
    monkeypatch.setattr(
        iteration_mod, "_compute_bms_power", lambda *a, **kw: (0.8, True)
    )

    # Mock _extract_diagnostics
    monkeypatch.setattr(
        iteration_mod,
        "_extract_diagnostics",
        lambda *a, **kw: (0.9, 0, 1.01),
    )

    # Mock make_power_config to return a simple object with task.n_trials_total
    mock_cfg = MagicMock()
    mock_cfg.task.n_trials_total = 200
    monkeypatch.setattr(
        iteration_mod, "make_power_config", lambda *a, **kw: mock_cfg
    )

    from prl_hgf.power.config import PowerConfig

    power_config = PowerConfig(
        n_per_group_grid=[10],
        effect_size_grid=[0.5],
        n_iterations=1,
        master_seed=42,
        n_chunks=1,
        bf_threshold=10.0,
    )

    from prl_hgf.power.iteration import run_power_iteration

    results = run_power_iteration(
        base_config=MagicMock(),
        n_per_group=10,
        effect_size_delta=0.5,
        iteration=0,
        child_seed=42,
        power_config=power_config,
    )

    assert isinstance(results, list)
    assert len(results) == 3

    expected_keys = set(POWER_SCHEMA.keys())
    sweep_types = []
    for result in results:
        assert set(result.keys()) == expected_keys
        sweep_types.append(result["sweep_type"])

    assert sweep_types == ["did_postdose", "did_followup", "linear_trend"]


# ---------------------------------------------------------------------------
# Test 4: run_power_iteration — schema conformance via write_parquet_row
# ---------------------------------------------------------------------------


def test_run_power_iteration_schema_conformance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each result dict passes write_parquet_row schema validation."""
    import prl_hgf.power.iteration as iteration_mod

    # Same mocking as test 3
    mock_sim_df = pd.DataFrame(
        {
            "participant_id": ["P001"] * 3,
            "group": ["psilocybin"] * 3,
            "session": ["baseline"] * 3,
            "trial": [0, 1, 2],
            "cue_chosen": [0, 1, 2],
            "reward": [1.0, 0.0, 1.0],
            "true_omega_2": [-3.0] * 3,
            "true_omega_3": [-6.0] * 3,
            "true_kappa": [1.0] * 3,
            "true_beta": [3.0] * 3,
            "true_zeta": [0.5] * 3,
        }
    )
    monkeypatch.setattr(
        iteration_mod, "simulate_batch", lambda cfg: mock_sim_df
    )

    mock_fit_df = pd.DataFrame(
        {
            "participant_id": ["P001"],
            "group": ["psilocybin"],
            "session": ["baseline"],
            "model": ["hgf_3level"],
            "parameter": ["omega_2"],
            "mean": [-3.0],
            "sd": [0.1],
            "hdi_3%": [-3.2],
            "hdi_97%": [-2.8],
            "r_hat": [1.01],
            "ess": [500.0],
            "flagged": [False],
        }
    )
    mock_idata = MagicMock()
    mock_idata_dict = {("P001", "psilocybin", "baseline"): mock_idata}

    monkeypatch.setattr(
        iteration_mod,
        "fit_batch",
        lambda *a, **kw: (mock_fit_df.copy(), dict(mock_idata_dict)),
    )

    mock_contrasts = [
        {"sweep_type": "did_postdose", "bf_value": 5.0, "bf_exceeds": False},
        {"sweep_type": "did_followup", "bf_value": 2.0, "bf_exceeds": False},
        {"sweep_type": "linear_trend", "bf_value": 1.5, "bf_exceeds": False},
    ]
    monkeypatch.setattr(
        iteration_mod,
        "compute_all_contrasts",
        lambda *a, **kw: mock_contrasts,
    )
    monkeypatch.setattr(
        iteration_mod, "_compute_bms_power", lambda *a, **kw: (0.8, True)
    )
    monkeypatch.setattr(
        iteration_mod,
        "_extract_diagnostics",
        lambda *a, **kw: (0.9, 0, 1.01),
    )

    mock_cfg = MagicMock()
    mock_cfg.task.n_trials_total = 200
    monkeypatch.setattr(
        iteration_mod, "make_power_config", lambda *a, **kw: mock_cfg
    )

    from prl_hgf.power.config import PowerConfig
    from prl_hgf.power.iteration import run_power_iteration

    power_config = PowerConfig(
        n_per_group_grid=[10],
        effect_size_grid=[0.5],
        n_iterations=1,
        master_seed=42,
        n_chunks=1,
        bf_threshold=10.0,
    )

    results = run_power_iteration(
        base_config=MagicMock(),
        n_per_group=10,
        effect_size_delta=0.5,
        iteration=0,
        child_seed=42,
        power_config=power_config,
    )

    # Verify each result dict passes schema validation (no ValueError)
    for i, row in enumerate(results):
        out_path = tmp_path / f"test_{i}.parquet"
        write_parquet_row(row, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Test 5: Entry point dry-run backward compatibility
# ---------------------------------------------------------------------------


def test_entry_point_dry_run_still_works(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--dry-run writes a placeholder parquet file (backward compat)."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--chunk-id", "0",
            "--job-id", "test",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ],
    )

    # Import the entry point main() — sys.path is already set
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from importlib import import_module

    # Force fresh import in case cached
    mod_name = "08_run_power_iteration"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    entry_mod = import_module(mod_name)
    entry_mod.main()

    # Verify a parquet file was created
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) == 1
    assert "job_test_chunk_0000" in parquet_files[0].name
