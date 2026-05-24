"""Unit tests for :mod:`prl_hgf.power.iteration` — core power pipeline.

Tests cover:

- ``build_arrays_from_sim`` shape and value correctness
- Partial feedback encoding (unchosen cues have observed=0)
- ``run_power_iteration`` return structure (mock-based)
- Schema conformance via ``write_parquet_row``
- Dry-run backward compatibility for the entry point script
- ``_idata_to_fit_df`` schema correctness (VALID-05)
- ``_split_idata`` per-participant slice correctness (VALID-05)
- ``run_sbf_iteration`` wiring: legacy flag calls ``fit_batch`` (VALID-05)
- ``run_sbf_iteration`` wiring: default calls ``fit_batch_hierarchical``

Run with::

    pytest tests/test_power_iteration.py -v
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from prl_hgf.power.iteration import apply_decision_gate, build_arrays_from_sim
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


# ---------------------------------------------------------------------------
# Test 6: _idata_to_fit_df — schema correctness (VALID-05)
# ---------------------------------------------------------------------------


def test_idata_to_fit_df_schema() -> None:
    """_idata_to_fit_df produces correct schema and row count."""
    import arviz as az
    import xarray as xr

    from prl_hgf.power.iteration import _idata_to_fit_df

    n_participants = 2
    n_chains = 2
    n_draws = 50  # enough for ESS checks

    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            "omega_2": xr.DataArray(
                rng.normal(-3.0, 0.3, (n_chains, n_draws, n_participants)),
                dims=["chain", "draw", "participant"],
            ),
            "log_beta": xr.DataArray(
                rng.normal(1.0, 0.1, (n_chains, n_draws, n_participants)),
                dims=["chain", "draw", "participant"],
            ),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "participant": ["P001", "P002"],
            "participant_group": ("participant", ["psilocybin", "placebo"]),
            "participant_session": ("participant", ["baseline", "post_dose"]),
        },
    )

    idata = az.InferenceData(posterior=posterior)
    df = _idata_to_fit_df(idata, "hgf_3level")

    # Schema check: exactly the columns required by the legacy fit_df contract
    expected_columns = {
        "participant_id",
        "group",
        "session",
        "model",
        "parameter",
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "r_hat",
        "ess",
        "flagged",
    }
    assert set(df.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {set(df.columns)}"
    )

    # Row count: 2 participants × 2 parameters = 4 rows
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    # model column is uniform
    assert (df["model"] == "hgf_3level").all(), "model column mismatch"

    # participant_ids match the coord values
    assert set(df["participant_id"]) == {"P001", "P002"}

    # groups match coords
    p001_rows = df[df["participant_id"] == "P001"]
    assert (p001_rows["group"] == "psilocybin").all()

    # sessions match coords
    p002_rows = df[df["participant_id"] == "P002"]
    assert (p002_rows["session"] == "post_dose").all()

    # flagged is bool
    assert df["flagged"].dtype == bool or df["flagged"].dtype == object


# ---------------------------------------------------------------------------
# Test 7: _split_idata — single-participant slice (VALID-05)
# ---------------------------------------------------------------------------


def test_split_idata_produces_single_participant() -> None:
    """_split_idata removes the participant dimension for one participant."""
    import arviz as az
    import xarray as xr

    from prl_hgf.power.iteration import _split_idata

    n_participants = 3
    n_chains = 2
    n_draws = 10

    rng = np.random.default_rng(1)
    posterior = xr.Dataset(
        {
            "omega_2": xr.DataArray(
                rng.normal(-3.0, 0.5, (n_chains, n_draws, n_participants)),
                dims=["chain", "draw", "participant"],
            ),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "participant": ["P001", "P002", "P003"],
        },
    )

    idata = az.InferenceData(posterior=posterior)

    # Slice participant at index 1 (P002)
    single = _split_idata(idata, 1)

    # No participant dimension in sliced posterior
    assert "participant" not in single.posterior.dims, (
        f"Expected no participant dim, got dims: {single.posterior.dims}"
    )

    # Values match the original slice
    expected = idata.posterior["omega_2"].isel(participant=1).values
    actual = single.posterior["omega_2"].values
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Test 8: run_sbf_iteration — legacy flag calls fit_batch (VALID-05)
# ---------------------------------------------------------------------------


def test_run_sbf_iteration_legacy_flag_calls_fit_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VALID-05: use_legacy=True routes through fit_batch, not hierarchical."""
    import prl_hgf.power.iteration as iteration_mod

    # Minimal sim_df with required columns for fit_batch and downstream code
    mock_sim_df = pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P001"],
            "group": ["psilocybin", "psilocybin", "psilocybin"],
            "session": ["baseline", "baseline", "baseline"],
            "trial": [0, 1, 2],
            "cue_chosen": [0, 1, 2],
            "reward": [1.0, 0.0, 1.0],
            "true_omega_2": [-3.0, -3.0, -3.0],
            "true_omega_3": [-6.0, -6.0, -6.0],
            "true_kappa": [1.0, 1.0, 1.0],
            "true_beta": [3.0, 3.0, 3.0],
            "true_zeta": [0.5, 0.5, 0.5],
        }
    )
    monkeypatch.setattr(
        iteration_mod, "simulate_batch", lambda cfg: mock_sim_df
    )

    # Mock fit_batch returns (fit_df, idata_dict) — legacy signature
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
    mock_idata_dict = {("P001", "psilocybin", "baseline"): MagicMock()}

    fit_batch_call_count = [0]

    def mock_fit_batch(*args, **kwargs):
        fit_batch_call_count[0] += 1
        return mock_fit_df.copy(), dict(mock_idata_dict)

    monkeypatch.setattr(iteration_mod, "fit_batch", mock_fit_batch)

    # Mock make_power_config
    mock_cfg = MagicMock()
    mock_cfg.task.n_trials_total = 200
    monkeypatch.setattr(
        iteration_mod, "make_power_config", lambda *a, **kw: mock_cfg
    )

    # Mock downstream helpers to avoid real computation
    monkeypatch.setattr(
        iteration_mod, "_compute_waic_table", lambda *a, **kw: pd.DataFrame(
            columns=["participant_id", "model", "elpd_waic"]
        )
    )
    monkeypatch.setattr(
        iteration_mod, "compute_all_contrasts", lambda *a, **kw: [
            {"sweep_type": "did_postdose", "bf_value": 1.0, "bf_exceeds": False},
            {"sweep_type": "did_followup", "bf_value": 1.0, "bf_exceeds": False},
            {"sweep_type": "linear_trend", "bf_value": 1.0, "bf_exceeds": False},
        ]
    )
    monkeypatch.setattr(
        iteration_mod, "_bms_from_waic_table", lambda *a, **kw: (0.5, False)
    )
    monkeypatch.setattr(
        iteration_mod, "_extract_diagnostics", lambda *a, **kw: (0.8, 0, 1.01)
    )

    from prl_hgf.power.config import PowerConfig
    from prl_hgf.power.iteration import run_sbf_iteration

    power_config = PowerConfig(
        n_per_group_grid=[1],
        effect_size_grid=[0.5],
        n_iterations=1,
        master_seed=42,
        n_chunks=1,
        bf_threshold=10.0,
    )

    results = run_sbf_iteration(
        base_config=MagicMock(),
        effect_size_delta=0.5,
        iteration=0,
        child_seed=42,
        n_per_group_grid=[1],
        power_config=power_config,
        use_legacy=True,
    )

    # fit_batch must have been called (legacy path)
    assert fit_batch_call_count[0] > 0, (
        "fit_batch was not called with use_legacy=True"
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 9: run_sbf_iteration — default calls fit_batch_hierarchical
# ---------------------------------------------------------------------------


def test_run_sbf_iteration_default_calls_hierarchical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default path (use_legacy=False) calls fit_batch_hierarchical."""
    import arviz as az
    import xarray as xr

    import prl_hgf.power.iteration as iteration_mod

    mock_sim_df = pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P001"],
            "group": ["psilocybin", "psilocybin", "psilocybin"],
            "session": ["baseline", "baseline", "baseline"],
            "trial": [0, 1, 2],
            "cue_chosen": [0, 1, 2],
            "reward": [1.0, 0.0, 1.0],
            "true_omega_2": [-3.0, -3.0, -3.0],
            "true_omega_3": [-6.0, -6.0, -6.0],
            "true_kappa": [1.0, 1.0, 1.0],
            "true_beta": [3.0, 3.0, 3.0],
            "true_zeta": [0.5, 0.5, 0.5],
        }
    )
    monkeypatch.setattr(
        iteration_mod, "simulate_batch", lambda cfg: mock_sim_df
    )

    # Build a real (small) InferenceData for the mock to return
    rng = np.random.default_rng(99)
    posterior = xr.Dataset(
        {
            "omega_2": xr.DataArray(
                rng.normal(-3.0, 0.3, (2, 10, 1)),
                dims=["chain", "draw", "participant"],
            ),
        },
        coords={
            "chain": np.arange(2),
            "draw": np.arange(10),
            "participant": ["P001"],
            "participant_group": ("participant", ["psilocybin"]),
            "participant_session": ("participant", ["baseline"]),
        },
    )
    mock_idata = az.InferenceData(posterior=posterior)

    hierarchical_call_count = [0]

    def mock_fit_batch_hierarchical(*args, **kwargs):
        hierarchical_call_count[0] += 1
        return mock_idata

    # Patch at the module where the deferred import resolves
    monkeypatch.setattr(
        "prl_hgf.fitting.hierarchical.fit_batch_hierarchical",
        mock_fit_batch_hierarchical,
    )

    mock_cfg = MagicMock()
    mock_cfg.task.n_trials_total = 200
    monkeypatch.setattr(
        iteration_mod, "make_power_config", lambda *a, **kw: mock_cfg
    )

    monkeypatch.setattr(
        iteration_mod, "_compute_waic_table", lambda *a, **kw: pd.DataFrame(
            columns=["participant_id", "model", "elpd_waic"]
        )
    )
    monkeypatch.setattr(
        iteration_mod, "compute_all_contrasts", lambda *a, **kw: [
            {"sweep_type": "did_postdose", "bf_value": 1.0, "bf_exceeds": False},
            {"sweep_type": "did_followup", "bf_value": 1.0, "bf_exceeds": False},
            {"sweep_type": "linear_trend", "bf_value": 1.0, "bf_exceeds": False},
        ]
    )
    monkeypatch.setattr(
        iteration_mod, "_bms_from_waic_table", lambda *a, **kw: (0.5, False)
    )
    monkeypatch.setattr(
        iteration_mod, "_extract_diagnostics", lambda *a, **kw: (0.8, 0, 1.01)
    )

    from prl_hgf.power.config import PowerConfig
    from prl_hgf.power.iteration import run_sbf_iteration

    power_config = PowerConfig(
        n_per_group_grid=[1],
        effect_size_grid=[0.5],
        n_iterations=1,
        master_seed=42,
        n_chunks=1,
        bf_threshold=10.0,
    )

    results = run_sbf_iteration(
        base_config=MagicMock(),
        effect_size_delta=0.5,
        iteration=0,
        child_seed=42,
        n_per_group_grid=[1],
        power_config=power_config,
        use_legacy=False,
    )

    # fit_batch_hierarchical must have been called (batched path)
    assert hierarchical_call_count[0] > 0, (
        "fit_batch_hierarchical was not called with use_legacy=False"
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 10: apply_decision_gate — GPU recommended (low per_iter_s)
# ---------------------------------------------------------------------------


def test_apply_decision_gate_gpu_recommended() -> None:
    """100s iteration -> 16.7 GPU-hrs/chunk -> decision = gpu."""
    result = apply_decision_gate(per_iter_seconds=100.0)

    # 100 * 600 / 3600 = 16.666... rounds to 16.7
    assert result["decision"] == "gpu", (
        f"Expected 'gpu', got '{result['decision']}'"
    )
    assert result["gpu_hours_per_chunk"] == 16.7, (
        f"Expected 16.7, got {result['gpu_hours_per_chunk']}"
    )


# ---------------------------------------------------------------------------
# Test 11: apply_decision_gate — CPU recommended (high per_iter_s)
# ---------------------------------------------------------------------------


def test_apply_decision_gate_cpu_recommended() -> None:
    """500s iteration -> 83.3 GPU-hrs/chunk -> decision = cpu_comp."""
    result = apply_decision_gate(per_iter_seconds=500.0)

    # 500 * 600 / 3600 = 83.333... rounds to 83.3
    assert result["decision"] == "cpu_comp", (
        f"Expected 'cpu_comp', got '{result['decision']}'"
    )
    assert result["gpu_hours_per_chunk"] > 50


# ---------------------------------------------------------------------------
# Test 12: apply_decision_gate — exact boundary (50.0 GPU-hrs -> gpu)
# ---------------------------------------------------------------------------


def test_apply_decision_gate_boundary() -> None:
    """300s -> exactly 50.0 GPU-hrs/chunk -> decision = gpu (uses <=, not <)."""
    result = apply_decision_gate(per_iter_seconds=300.0)

    # 300 * 600 / 3600 = 50.0 exactly; gate uses > 50, so 50 == gpu
    assert result["decision"] == "gpu", (
        f"Boundary case: expected 'gpu' (<=50), got '{result['decision']}'"
    )
    assert result["gpu_hours_per_chunk"] == 50.0


# ---------------------------------------------------------------------------
# Test 13: apply_decision_gate — schema completeness
# ---------------------------------------------------------------------------


def test_apply_decision_gate_schema() -> None:
    """apply_decision_gate returns all required fields for benchmark JSON."""
    result = apply_decision_gate(per_iter_seconds=100.0)

    required_fields = {
        "per_iter_seconds",
        "gpu_hours_per_chunk",
        "decision",
        "decision_threshold_gpu_hours",
        "decision_formula",
    }
    assert required_fields <= set(result.keys()), (
        f"Missing fields: {required_fields - set(result.keys())}"
    )
    assert result["decision_threshold_gpu_hours"] == 50
    assert result["decision_formula"] == "per_iter_seconds * 600 / 3600 > 50"
    assert result["per_iter_seconds"] == 100.0


# ---------------------------------------------------------------------------
# Test 14: _GpuMonitor — graceful degradation when nvidia-smi is absent
# ---------------------------------------------------------------------------


def test_gpu_monitor_no_nvidia_smi() -> None:
    """_GpuMonitor records no samples and returns 0.0 when nvidia-smi absent."""
    # Import _GpuMonitor from the script module
    script_path = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(script_path))

    mod_name = "08_run_power_iteration"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    script_mod = importlib.import_module(mod_name)
    _GpuMonitor = script_mod._GpuMonitor  # type: ignore[attr-defined]

    monitor = _GpuMonitor(interval_s=0.05)  # short interval for test speed

    # Patch subprocess.run in the script module's namespace
    with patch(f"{mod_name}.subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        monitor.start()
        time.sleep(0.15)  # allow a couple of polling cycles
        monitor.stop()

    assert monitor.samples == [], (
        f"Expected empty samples when nvidia-smi absent, got {monitor.samples}"
    )
    assert monitor.peak_vram_mb == 0.0, (
        f"Expected peak_vram_mb=0.0, got {monitor.peak_vram_mb}"
    )
    assert monitor.mean_gpu_util_pct == 0.0, (
        f"Expected mean_gpu_util_pct=0.0, got {monitor.mean_gpu_util_pct}"
    )
