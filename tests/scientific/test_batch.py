"""Unit tests for batch simulation.

Tests use a small configuration (2 participants/group instead of 30) to keep
runtime reasonable (~60-90 s) while still exercising the full batch pipeline.

All tests are marked ``@pytest.mark.scientific`` since they invoke pyhgf network
operations which trigger JAX JIT compilation on the first call.

Run all:  ``pytest tests/test_batch.py -v``
"""

from __future__ import annotations

import pandas as pd
import pytest

from prl_hgf.env.task_config import (
    AnalysisConfig,
    SimulationConfig,
    load_config,
)
from prl_hgf.simulation.batch import simulate_batch

# ---------------------------------------------------------------------------
# Expected columns in batch output (SIM-06 spec)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "participant_id",
    "group",
    "session",
    "session_idx",
    "trial",
    "cue_chosen",
    "reward",
    "cue_0_prob",
    "cue_1_prob",
    "cue_2_prob",
    "phase_label",
    "phase_name",
    "best_cue",
    "true_omega_2",
    "true_omega_3",
    "true_kappa",
    "true_beta",
    "true_zeta",
    "model",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(n_per_group: int = 2) -> AnalysisConfig:
    """Return an AnalysisConfig with n_participants_per_group overridden.

    Since all dataclasses are frozen, construct a new SimulationConfig and
    a new AnalysisConfig wrapping the same task and group distributions as
    the real config.

    Parameters
    ----------
    n_per_group : int
        Participants per group in the small test config.

    Returns
    -------
    AnalysisConfig
        Config identical to the real one except for a reduced cohort size.
    """
    real = load_config()
    small_sim = SimulationConfig(
        n_participants_per_group=n_per_group,
        master_seed=real.simulation.master_seed,
        groups=real.simulation.groups,
        session_deltas=real.simulation.session_deltas,
    )
    return AnalysisConfig(task=real.task, simulation=small_sim, fitting=real.fitting)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_batch_output_shape():
    """Batch DataFrame has correct number of rows and all expected columns."""
    cfg = _small_config(n_per_group=2)
    n_groups = len(cfg.simulation.groups)
    n_sessions = 3
    n_trials = cfg.task.n_trials_total
    expected_rows = n_groups * 2 * n_sessions * n_trials  # 2 per group

    df = simulate_batch(cfg)

    assert df.shape[0] == expected_rows, (
        f"Expected {expected_rows} rows, got {df.shape[0]}. "
        f"(n_groups={n_groups}, n_per_group=2, n_sessions={n_sessions}, "
        f"n_trials={n_trials})"
    )
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns in output: {sorted(missing)}"


@pytest.mark.scientific
def test_batch_column_values():
    """Column value domains match specification."""
    cfg = _small_config(n_per_group=2)
    df = simulate_batch(cfg)

    # Group labels
    actual_groups = set(df["group"].unique())
    expected_groups = {"psilocybin", "placebo"}
    assert actual_groups == expected_groups, (
        f"Expected groups {expected_groups}, got {actual_groups}"
    )

    # Session labels
    actual_sessions = set(df["session"].unique())
    expected_sessions = {"baseline", "post_dose", "followup"}
    assert actual_sessions == expected_sessions, (
        f"Expected sessions {expected_sessions}, got {actual_sessions}"
    )

    # Cue chosen in {0, 1, 2}
    bad_cues = df["cue_chosen"].loc[~df["cue_chosen"].isin([0, 1, 2])]
    assert bad_cues.empty, f"cue_chosen out of {{0,1,2}}: {bad_cues.unique()}"

    # Reward in {0, 1}
    bad_rewards = df["reward"].loc[~df["reward"].isin([0, 1])]
    assert bad_rewards.empty, f"reward out of {{0,1}}: {bad_rewards.unique()}"

    # Participant IDs: 4 unique (2 per group x 2 groups)
    n_unique_ids = df["participant_id"].nunique()
    assert n_unique_ids == 4, f"Expected 4 unique participant_ids, got {n_unique_ids}"

    # Model column
    assert (df["model"] == "hgf_3level").all(), (
        "Expected all model values to be 'hgf_3level'"
    )

    # true_kappa > 0 (clipping to PARAM_BOUNDS[kappa] = (0.01, 2.0))
    assert (df["true_kappa"] > 0).all(), (
        f"true_kappa has non-positive values: min={df['true_kappa'].min()}"
    )

    # true_beta > 0 (clipping to PARAM_BOUNDS[beta] = (0.01, 20.0))
    assert (df["true_beta"] > 0).all(), (
        f"true_beta has non-positive values: min={df['true_beta'].min()}"
    )


@pytest.mark.scientific
def test_batch_group_param_differences():
    """Psilocybin group shows larger omega_2 shift than placebo at post_dose.

    From config: psilocybin omega_2_delta=+1.5, placebo omega_2_delta=+0.3.
    Even with N=2, the psilocybin group's post_dose shift should exceed placebo.
    """
    cfg = _small_config(n_per_group=2)
    df = simulate_batch(cfg)

    unique = df.drop_duplicates(subset=["participant_id", "session"])

    def mean_omega2(group, session):
        mask = (unique["group"] == group) & (unique["session"] == session)
        return unique.loc[mask, "true_omega_2"].mean()

    psi_shift = mean_omega2("psilocybin", "post_dose") - mean_omega2(
        "psilocybin", "baseline"
    )
    plc_shift = mean_omega2("placebo", "post_dose") - mean_omega2(
        "placebo", "baseline"
    )

    assert psi_shift > plc_shift, (
        f"Expected psilocybin omega_2 shift ({psi_shift:.4f}) > "
        f"placebo omega_2 shift ({plc_shift:.4f}) at post_dose."
    )


@pytest.mark.scientific
def test_batch_session_deltas_visible():
    """Psilocybin group omega_2 is higher at post_dose than at baseline.

    From config: psilocybin omega_2_deltas[0] = +1.5 at post_dose.
    With N=2 and fixed seed, each participant's post_dose omega_2 should
    exceed their baseline omega_2 by ~1.5.
    """
    cfg = _small_config(n_per_group=2)
    df = simulate_batch(cfg)

    pc = df[df["group"] == "psilocybin"].drop_duplicates(
        subset=["participant_id", "session"]
    )
    baseline_vals = pc[pc["session"] == "baseline"].set_index("participant_id")[
        "true_omega_2"
    ]
    postdose_vals = pc[pc["session"] == "post_dose"].set_index("participant_id")[
        "true_omega_2"
    ]

    # Same participants must appear in both sessions
    common_ids = baseline_vals.index.intersection(postdose_vals.index)
    assert len(common_ids) > 0, (
        "No common participant_ids between baseline and post_dose"
    )

    for pid in common_ids:
        baseline_val = baseline_vals[pid]
        postdose_val = postdose_vals[pid]
        assert postdose_val > baseline_val, (
            f"Participant {pid}: post_dose omega_2 ({postdose_val:.4f}) should "
            f"exceed baseline omega_2 ({baseline_val:.4f}) by ~1.5"
        )


@pytest.mark.scientific
def test_batch_reproducible():
    """Two runs with the same config produce identical DataFrames."""
    cfg = _small_config(n_per_group=2)
    df1 = simulate_batch(cfg)
    df2 = simulate_batch(cfg)

    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_exact=True,
        obj="simulate_batch reproducibility",
    )


@pytest.mark.scientific
def test_batch_csv_output(tmp_path):
    """CSV output is saved correctly and readable with correct row count."""
    cfg = _small_config(n_per_group=2)
    n_groups = len(cfg.simulation.groups)
    n_sessions = 3
    n_trials = cfg.task.n_trials_total
    expected_rows = n_groups * 2 * n_sessions * n_trials

    output_file = tmp_path / "test_output.csv"
    df = simulate_batch(cfg, output_path=output_file)

    assert output_file.exists(), f"Output CSV not created at {output_file}"

    df_loaded = pd.read_csv(output_file)
    assert len(df_loaded) == expected_rows, (
        f"CSV has {len(df_loaded)} rows, expected {expected_rows}"
    )
    assert len(df_loaded) == len(df), (
        f"CSV row count ({len(df_loaded)}) != returned DataFrame row count ({len(df)})"
    )
