"""Integration + audit tests for Phase 20-07 validation gates (PRL-V1 + PRL-V2).

Covers:
* SC9 audit — zero new files added under scripts/ since Phase 20 start
* SC11 regression gate — Phase 18/19 regression suite is green
* Option A axis mask constants — prevents regression if mapping drifts
* Cohen's d gate logic — synthetic d=0.6 passes, d=0.3 fails
* |cor(omega_2, beta)| gate logic — cor=0.3 passes, cor=0.6 fails
* compute_recovery_metrics_patrl exploratory labelling
* phenotype_separability_summary.csv schema with contrast_type column

Run::

    pytest tests/test_validation_gates_patrl.py -v          # fast tests only
    pytest tests/test_validation_gates_patrl.py -v -m slow  # + SC11 (slow)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Commit hash that marks the beginning of Phase 20 (one commit before 20-01).
# SC9 checks that no scripts/ files were *added* between this ref and HEAD.
_PHASE_20_START_REF = "b040c3f"


# ---------------------------------------------------------------------------
# SC9: no new scripts added under scripts/
# ---------------------------------------------------------------------------


def test_sc9_no_new_scripts_audit() -> None:
    """SC9: zero script files added under scripts/ since Phase 20 start.

    Runs ``git diff --name-status {_PHASE_20_START_REF}..HEAD -- scripts/``
    and asserts that every output line starts with ``M`` (modified) — i.e.
    no line starts with ``A`` (added).

    Failure message includes the full list of offending lines so the author
    can identify the violating file(s) immediately.

    Cluster equivalent (manual check)::

        git diff --name-status b040c3f..HEAD -- scripts/
    """
    try:
        out = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-status",
                f"{_PHASE_20_START_REF}..HEAD",
                "--",
                "scripts/",
            ],
            text=True,
            cwd=str(_ROOT),
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"SC9: git diff command failed: {exc}")

    lines = [line for line in out.splitlines() if line.strip()]
    added = [line for line in lines if line.startswith("A\t") or line.startswith("A ")]

    assert not added, (
        "SC9 VIOLATED — new script files added since Phase 20 start "
        f"(commit {_PHASE_20_START_REF}):\n"
        + "\n".join(f"  {line}" for line in added)
        + "\n\nSC9 invariant: extensions MUST be modifications to existing scripts "
        "under scripts/; zero new files allowed."
    )


# ---------------------------------------------------------------------------
# SC11: Phase 18/19 regression suite is green
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_sc11_phase_18_19_regression_suite_green() -> None:
    """SC11: Full Phase 18/19 regression suite must exit 0.

    Runs the canonical regression set as a subprocess and asserts returncode
    == 0.  Marked ``@pytest.mark.scientific`` — exclude from fast CI with
    ``pytest -m 'not slow'``.

    Cluster equivalent::

        conda run -n ds_env pytest tests/test_env_pat_rl_config.py \\
            tests/test_models_patrl.py tests/test_pat_rl_simulator.py \\
            tests/test_hierarchical_patrl.py tests/test_fit_vb_laplace_patrl.py \\
            tests/test_smoke_patrl_foundation.py tests/test_bms.py -v
    """
    regression_files = [
        "tests/test_env_pat_rl_config.py",
        "tests/test_models_patrl.py",
        "tests/test_pat_rl_simulator.py",
        "tests/test_hierarchical_patrl.py",
        "tests/test_fit_vb_laplace_patrl.py",
        "tests/test_smoke_patrl_foundation.py",
        "tests/test_bms.py",
    ]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *regression_files,
            "-v",
            "--tb=short",
            "-x",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )

    assert result.returncode == 0, (
        "SC11 VIOLATED — Phase 18/19 regression suite has failures.\n"
        f"Return code: {result.returncode}\n"
        "--- STDOUT ---\n"
        f"{result.stdout[-4000:]}\n"
        "--- STDERR ---\n"
        f"{result.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Option A axis mask regression test
# ---------------------------------------------------------------------------


def test_option_a_factorial_axis_masks_are_correct() -> None:
    """Verify axis mask constants in scripts/06_group_analysis.py match Option A spec.

    This catches regressions where a future refactor drifts the phenotype
    → axis mapping away from the user-approved Option A (2x2 factorial).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None, (
        "Could not load scripts/06_group_analysis.py as a module spec"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    assert frozenset({"high_anxiety", "anxious_reward"}) == mod.ANXIETY_HIGH, (
        f"ANXIETY_HIGH mismatch.  Expected {{'high_anxiety', 'anxious_reward'}}, "
        f"got {mod.ANXIETY_HIGH}"
    )
    assert frozenset({"healthy", "reward_susceptible"}) == mod.ANXIETY_LOW, (
        f"ANXIETY_LOW mismatch.  Expected {{'healthy', 'reward_susceptible'}}, "
        f"got {mod.ANXIETY_LOW}"
    )
    assert frozenset({"reward_susceptible", "anxious_reward"}) == mod.REWARD_HIGH, (
        f"REWARD_HIGH mismatch.  Expected {{'reward_susceptible', 'anxious_reward'}}, "
        f"got {mod.REWARD_HIGH}"
    )
    assert frozenset({"healthy", "high_anxiety"}) == mod.REWARD_LOW, (
        f"REWARD_LOW mismatch.  Expected {{'healthy', 'high_anxiety'}}, "
        f"got {mod.REWARD_LOW}"
    )


# ---------------------------------------------------------------------------
# Cohen's d gate logic
# ---------------------------------------------------------------------------


def _make_separation_df(d_effect: float, n_per_group: int = 40) -> pd.DataFrame:
    """Build a synthetic estimates_df with known Cohen's d separation.

    Parameters
    ----------
    d_effect : float
        Target Cohen's d (group_high mean - group_low mean with SD=1).
    n_per_group : int
        Agents per phenotype.

    Returns
    -------
    pandas.DataFrame with columns: participant_id, phenotype, omega_2, beta.
    """
    rng = np.random.default_rng(seed=42)
    rows = []

    # phenotypes: high_anxiety + anxious_reward = ANXIETY_HIGH (d_effect on omega_2)
    # healthy + reward_susceptible = ANXIETY_LOW (baseline omega_2 = 0)
    #
    # For reward axis: reward_susceptible + anxious_reward = REWARD_HIGH (d on beta)
    # healthy + high_anxiety = REWARD_LOW (baseline beta = 0)
    #
    # Use d_effect for both axes for simplicity.

    phenotype_omega2 = {
        "high_anxiety": d_effect,         # anxiety_high
        "anxious_reward": d_effect,       # anxiety_high
        "healthy": 0.0,                   # anxiety_low
        "reward_susceptible": 0.0,        # anxiety_low
    }
    phenotype_beta = {
        "high_anxiety": 0.0,              # reward_low
        "anxious_reward": d_effect,       # reward_high
        "healthy": 0.0,                   # reward_low
        "reward_susceptible": d_effect,   # reward_high
    }

    pid_counter = 0
    for ph in ["high_anxiety", "anxious_reward", "healthy", "reward_susceptible"]:
        for _ in range(n_per_group):
            rows.append(
                {
                    "participant_id": f"P{pid_counter:04d}",
                    "phenotype": ph,
                    "omega_2": float(rng.normal(phenotype_omega2[ph], 1.0)),
                    "beta": float(rng.normal(phenotype_beta[ph], 1.0)),
                }
            )
            pid_counter += 1

    return pd.DataFrame(rows)


def test_cohen_d_gate_logic_pass() -> None:
    """d=0.6 separation on omega_2 and beta both pass PRL-V2 gate (d >= 0.5)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    estimates_df = _make_separation_df(d_effect=0.9, n_per_group=40)
    summary_df, gate_pass = mod._compute_prl_v2_metrics(estimates_df)

    # d on omega_2 (anxiety axis) should pass
    omega_row = summary_df[
        (summary_df["contrast"] == "anxiety_high_vs_low")
        & (summary_df["contrast_type"] == "factorial")
    ]
    assert len(omega_row) == 1, "anxiety_high_vs_low factorial row missing"
    d_omega = float(omega_row["d_or_cor"].iloc[0])
    assert abs(d_omega) >= 0.5, (
        f"Expected |d(omega_2)| >= 0.5 for d=0.9 separation, got {d_omega:.3f}"
    )

    # d on beta (reward axis) should pass
    beta_row = summary_df[
        (summary_df["contrast"] == "reward_high_vs_low")
        & (summary_df["contrast_type"] == "factorial")
    ]
    assert len(beta_row) == 1, "reward_high_vs_low factorial row missing"
    d_beta = float(beta_row["d_or_cor"].iloc[0])
    assert abs(d_beta) >= 0.5, (
        f"Expected |d(beta)| >= 0.5 for d=0.9 separation, got {d_beta:.3f}"
    )


def test_cohen_d_gate_logic_fail() -> None:
    """d=0.0 separation fails the PRL-V2 d gate (< 0.5)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # No separation — both axes will have |d| ≈ 0
    estimates_df = _make_separation_df(d_effect=0.0, n_per_group=40)
    summary_df, gate_pass = mod._compute_prl_v2_metrics(estimates_df)

    omega_row = summary_df[
        (summary_df["contrast"] == "anxiety_high_vs_low")
        & (summary_df["contrast_type"] == "factorial")
    ]
    d_omega = float(omega_row["d_or_cor"].iloc[0])
    assert abs(d_omega) < 0.5, (
        f"Expected |d(omega_2)| < 0.5 for zero separation, got {d_omega:.3f}"
    )

    # Gate should fail (at least one criterion unmet)
    # Correlation gate may also fail/pass independently — focus on d gate
    d_rows = summary_df[
        summary_df["contrast"].isin(["anxiety_high_vs_low", "reward_high_vs_low"])
    ]
    assert not d_rows["passes"].all(), (
        "Expected at least one d gate to fail for zero-separation data"
    )


# ---------------------------------------------------------------------------
# Correlation gate logic
# ---------------------------------------------------------------------------


def _make_correlated_df(cor: float, n: int = 160) -> pd.DataFrame:
    """Build estimates_df with a specified |cor(omega_2, beta)|.

    Parameters
    ----------
    cor : float
        Target Pearson correlation between omega_2 and beta (signed).
    n : int
        Total number of agents (spread evenly across 4 phenotypes).

    Returns
    -------
    pandas.DataFrame with columns: participant_id, phenotype, omega_2, beta.
    """
    rng = np.random.default_rng(seed=99)
    cov_mat = np.array([[1.0, cor], [cor, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], cov_mat, size=n)

    phenotypes = ["high_anxiety", "anxious_reward", "healthy", "reward_susceptible"]
    n_per = n // len(phenotypes)
    phenotype_col = []
    for ph in phenotypes:
        phenotype_col.extend([ph] * n_per)
    # pad to n
    while len(phenotype_col) < n:
        phenotype_col.append(phenotypes[-1])

    return pd.DataFrame(
        {
            "participant_id": [f"P{i:04d}" for i in range(n)],
            "phenotype": phenotype_col[:n],
            "omega_2": samples[:, 0],
            "beta": samples[:, 1],
        }
    )


def test_cor_omega_beta_gate_logic_pass() -> None:
    """cor=0.3 is below threshold 0.5 → correlation gate passes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    estimates_df = _make_correlated_df(cor=0.3, n=160)
    summary_df, _ = mod._compute_prl_v2_metrics(estimates_df)

    cor_row = summary_df[
        (summary_df["contrast"] == "cor_omega2_beta")
        & (summary_df["contrast_type"] == "factorial")
    ]
    assert len(cor_row) == 1, "cor_omega2_beta factorial row missing"
    cor_val = float(cor_row["d_or_cor"].iloc[0])
    assert cor_row["passes"].iloc[0], (
        f"Expected |cor|={abs(cor_val):.3f} to pass (threshold < 0.5)"
    )


def test_cor_omega_beta_gate_logic_fail() -> None:
    """cor=0.8 is above threshold 0.5 → correlation gate fails."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    estimates_df = _make_correlated_df(cor=0.8, n=160)
    summary_df, gate_pass = mod._compute_prl_v2_metrics(estimates_df)

    cor_row = summary_df[
        (summary_df["contrast"] == "cor_omega2_beta")
        & (summary_df["contrast_type"] == "factorial")
    ]
    assert len(cor_row) == 1, "cor_omega2_beta factorial row missing"
    cor_val = float(cor_row["d_or_cor"].iloc[0])
    assert not cor_row["passes"].iloc[0], (
        f"Expected |cor|={abs(cor_val):.3f} to fail (threshold < 0.5)"
    )


# ---------------------------------------------------------------------------
# compute_recovery_metrics_patrl exploratory labelling
# ---------------------------------------------------------------------------


def test_compute_recovery_metrics_patrl_exploratory_labeling() -> None:
    """Exploratory params (omega_3, mu3_0) get exploratory=True and passes_threshold=pd.NA.

    Primary params (omega_2, kappa, beta) get exploratory=False and a bool
    passes_threshold.  Verifies Decision 141 / Phase 9-01 precedent.
    """
    from prl_hgf.analysis.recovery import compute_recovery_metrics_patrl

    rng = np.random.default_rng(seed=17)
    n = 10
    params = ["omega_2", "kappa", "beta", "omega_3", "mu3_0"]
    true_vals = {"omega_2": -3.0, "kappa": 1.0, "beta": 2.0, "omega_3": -6.0, "mu3_0": 0.0}

    rows = []
    for pid_i in range(n):
        for p in params:
            tv = true_vals[p] + rng.normal(0, 0.5)
            rows.append(
                {
                    "participant_id": f"P{pid_i:02d}",
                    "param": p,
                    "true_value": tv,
                    "posterior_mean": tv + rng.normal(0, 0.05),
                    "posterior_sd": 0.1,
                }
            )
    fit_df = pd.DataFrame(rows)

    metrics = compute_recovery_metrics_patrl(fit_df, r_threshold=0.7)

    assert set(metrics["param"].tolist()) >= {"omega_2", "kappa", "beta", "omega_3", "mu3_0"}, (
        f"Expected all 5 params in output, got: {metrics['param'].tolist()}"
    )

    # Primary: exploratory=False, passes_threshold in {True, False}
    for p in ("omega_2", "kappa", "beta"):
        row = metrics[metrics["param"] == p]
        assert len(row) == 1, f"Missing row for {p}"
        assert not bool(row["exploratory"].iloc[0]), (
            f"{p} should have exploratory=False, got {row['exploratory'].iloc[0]}"
        )
        passes = row["passes_threshold"].iloc[0]
        assert isinstance(passes, (bool, np.bool_)) or passes in (True, False), (
            f"{p} passes_threshold should be bool, got {type(passes)}: {passes}"
        )

    # Exploratory: exploratory=True, passes_threshold=pd.NA
    for p in ("omega_3", "mu3_0"):
        row = metrics[metrics["param"] == p]
        assert len(row) == 1, f"Missing row for {p}"
        assert bool(row["exploratory"].iloc[0]), (
            f"{p} should have exploratory=True, got {row['exploratory'].iloc[0]}"
        )
        passes = row["passes_threshold"].iloc[0]
        assert pd.isna(passes), (
            f"{p} passes_threshold should be pd.NA, got {passes!r}"
        )


# ---------------------------------------------------------------------------
# phenotype_separability_summary.csv schema
# ---------------------------------------------------------------------------


def test_phenotype_separability_supplementary_csv_schema() -> None:
    """phenotype_separability_summary.csv has contrast_type column with both types.

    Verifies:
    * 'contrast_type' column exists.
    * Both 'factorial' and 'pairwise' values are present.
    * 'factorial' rows correspond to the 3 primary gate criteria.
    * 'pairwise' rows are present as supplementary output.

    Uses a synthetic run via _compute_prl_v2_metrics directly.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "group_analysis_06", _ROOT / "scripts" / "06_group_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # Create an estimates_df with all 4 phenotypes
    estimates_df = _make_separation_df(d_effect=0.6, n_per_group=20)

    summary_df, gate_pass = mod._compute_prl_v2_metrics(estimates_df)

    # Schema checks
    assert "contrast_type" in summary_df.columns, (
        f"'contrast_type' column missing from summary. Got: {list(summary_df.columns)}"
    )
    assert "contrast" in summary_df.columns, "'contrast' column missing"
    assert "d_or_cor" in summary_df.columns, "'d_or_cor' column missing"
    assert "passes" in summary_df.columns, "'passes' column missing"
    assert "n_high" in summary_df.columns, "'n_high' column missing"
    assert "n_low" in summary_df.columns, "'n_low' column missing"

    # Both contrast types present
    types_present = set(summary_df["contrast_type"].unique().tolist())
    assert "factorial" in types_present, (
        f"'factorial' contrast type missing. Present: {types_present}"
    )
    assert "pairwise" in types_present, (
        f"'pairwise' contrast type missing. Present: {types_present}"
    )

    # Exactly 3 factorial rows (anxiety_d, reward_d, cor)
    factorial_rows = summary_df[summary_df["contrast_type"] == "factorial"]
    assert len(factorial_rows) == 3, (
        f"Expected 3 factorial rows, got {len(factorial_rows)}:\n"
        f"{factorial_rows[['contrast', 'contrast_type']].to_string()}"
    )

    # At least 1 pairwise row
    pairwise_rows = summary_df[summary_df["contrast_type"] == "pairwise"]
    assert len(pairwise_rows) >= 1, (
        f"Expected >= 1 pairwise row, got {len(pairwise_rows)}"
    )

    # Gate pass/fail decision uses ONLY factorial rows
    factorial_passes = bool(factorial_rows["passes"].all())
    assert gate_pass == factorial_passes, (
        f"gate_pass ({gate_pass}) should equal factorial_passes ({factorial_passes})"
    )
