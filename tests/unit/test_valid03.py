"""Unit tests for VALID-03 cross-platform comparison logic.

These tests exercise :func:`~tests.scientific.test_valid03_cross_platform.compare_results`
with synthetic JSON data.  They do NOT run MCMC — the actual cross-platform
fit is a manual validation step on the cluster.

The four cases tested are:
1. Identical result files: must pass.
2. Values differing by < 0.5% (well within 1% tolerance): must pass.
3. One parameter / participant exceeds 5% relative error: must fail.
4. Near-zero parameter values (mean ~ 0.001): denominator uses
   ``abs(mean_a) + 1e-8`` so no false failure from near-zero division.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable so we can import from tests.scientific.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tests.scientific.test_valid03_cross_platform import compare_results  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, name: str, posterior_means: dict) -> Path:
    """Write a minimal VALID-03 JSON file to tmp_path."""
    data = {
        "platform": "cpu",
        "n_per_group": 2,
        "seed": 42,
        "n_chains": 2,
        "n_draws": 100,
        "n_tune": 100,
        "participant_ids": list(next(iter(posterior_means.values())).keys()),
        "parameters": list(posterior_means.keys()),
        "posterior_means": posterior_means,
    }
    path = tmp_path / name
    with path.open("w") as fh:
        json.dump(data, fh)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compare_results_identical(tmp_path: Path) -> None:
    """Identical JSON files: all relative errors are 0, must return True."""
    posterior_means = {
        "omega_2": {"P01": -3.5, "P02": -4.1, "P03": -3.8},
        "beta": {"P01": 1.2, "P02": 0.9, "P03": 1.5},
    }
    path_a = _write_json(tmp_path, "a.json", posterior_means)
    path_b = _write_json(tmp_path, "b.json", posterior_means)

    result = compare_results(path_a, path_b, rtol=0.01)

    assert result is True


def test_compare_results_within_tolerance(tmp_path: Path) -> None:
    """Values differing by < 0.5% (< rtol=1%): must return True."""
    means_a = {
        "omega_2": {"P01": -3.500, "P02": -4.100, "P03": -3.800},
        "beta": {"P01": 1.200, "P02": 0.900, "P03": 1.500},
    }
    # Introduce 0.3% relative difference on each value.
    factor = 0.003
    means_b = {
        param: {pid: val * (1.0 + factor) for pid, val in pid_vals.items()}
        for param, pid_vals in means_a.items()
    }
    path_a = _write_json(tmp_path, "a.json", means_a)
    path_b = _write_json(tmp_path, "b.json", means_b)

    result = compare_results(path_a, path_b, rtol=0.01)

    assert result is True


def test_compare_results_exceeds_tolerance(tmp_path: Path) -> None:
    """One participant exceeds 5% relative error: must return False."""
    means_a = {
        "omega_2": {"P01": -3.500, "P02": -4.100, "P03": -3.800},
        "beta": {"P01": 1.200, "P02": 0.900, "P03": 1.500},
    }
    means_b = {
        # omega_2 for P02 differs by ~5% (well above rtol=1%)
        "omega_2": {"P01": -3.500, "P02": -4.100 * 1.05, "P03": -3.800},
        "beta": {"P01": 1.200, "P02": 0.900, "P03": 1.500},
    }
    path_a = _write_json(tmp_path, "a.json", means_a)
    path_b = _write_json(tmp_path, "b.json", means_b)

    result = compare_results(path_a, path_b, rtol=0.01)

    assert result is False


def test_compare_results_near_zero_parameter(tmp_path: Path) -> None:
    """Near-zero parameter values do not trigger false failures.

    When mean_a ~ 0.001, dividing by ``abs(mean_a)`` alone could be dominated
    by numerical noise.  The denominator ``abs(mean_a) + 1e-8`` ensures a
    stable relative error.  Both platforms report identical means here so the
    result must be True.
    """
    means_a = {
        "zeta": {"P01": 0.001, "P02": 0.0005, "P03": 0.0002},
        "omega_2": {"P01": -3.5, "P02": -4.1, "P03": -3.8},
    }
    # Same values on both platforms (identical); rel_err = 0.
    path_a = _write_json(tmp_path, "a.json", means_a)
    path_b = _write_json(tmp_path, "b.json", means_a)

    result = compare_results(path_a, path_b, rtol=0.01)

    assert result is True
