"""Unit tests for make_power_config and PowerConfig.

All tests are pure dataclass operations — no JAX, no pyhgf, no MCMC. They
run in milliseconds and do not require any marks.

Run with::

    pytest tests/test_power_config.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from prl_hgf.env.task_config import AnalysisConfig, load_config
from prl_hgf.power.config import PowerConfig, load_power_config, make_power_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base() -> AnalysisConfig:
    """Return the base AnalysisConfig loaded from configs/prl_analysis.yaml."""
    return load_config()


# ---------------------------------------------------------------------------
# make_power_config — field overrides
# ---------------------------------------------------------------------------


def test_make_power_config_returns_frozen_config(base: AnalysisConfig) -> None:
    """make_power_config returns AnalysisConfig with n_per_group and seed set.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    result = make_power_config(
        base, n_per_group=20, effect_size_delta=0.5, master_seed=9999
    )
    assert isinstance(result, AnalysisConfig)
    assert result.simulation.n_participants_per_group == 20
    assert result.simulation.master_seed == 9999
    # Unchanged sub-configs must be the same object (not copied)
    assert result.task == base.task
    assert result.fitting == base.fitting


def test_make_power_config_applies_effect_size_delta(
    base: AnalysisConfig,
) -> None:
    """Psilocybin omega_2_deltas shift by effect_size_delta; placebo unchanged.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    delta = 0.5
    result = make_power_config(
        base, n_per_group=20, effect_size_delta=delta, master_seed=1
    )

    base_psi = base.simulation.session_deltas["psilocybin"]
    result_psi = result.simulation.session_deltas["psilocybin"]
    for orig, shifted in zip(
        base_psi.omega_2_deltas, result_psi.omega_2_deltas, strict=True
    ):
        assert shifted == pytest.approx(orig + delta)

    base_pla = base.simulation.session_deltas["placebo"]
    result_pla = result.simulation.session_deltas["placebo"]
    assert result_pla.omega_2_deltas == pytest.approx(base_pla.omega_2_deltas)


# ---------------------------------------------------------------------------
# make_power_config — immutability of base
# ---------------------------------------------------------------------------


def test_make_power_config_does_not_mutate_base(base: AnalysisConfig) -> None:
    """Calling make_power_config must not change any field on base.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    orig_n = base.simulation.n_participants_per_group
    orig_omega_2 = list(
        base.simulation.session_deltas["psilocybin"].omega_2_deltas
    )

    make_power_config(base, n_per_group=99, effect_size_delta=2.0, master_seed=0)

    assert base.simulation.n_participants_per_group == orig_n
    assert list(
        base.simulation.session_deltas["psilocybin"].omega_2_deltas
    ) == orig_omega_2


# ---------------------------------------------------------------------------
# make_power_config — preservation of other delta fields
# ---------------------------------------------------------------------------


def test_make_power_config_preserves_non_omega2_deltas(
    base: AnalysisConfig,
) -> None:
    """kappa_deltas, beta_deltas, zeta_deltas for psilocybin are unchanged.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    result = make_power_config(
        base, n_per_group=20, effect_size_delta=1.0, master_seed=1
    )

    base_psi = base.simulation.session_deltas["psilocybin"]
    result_psi = result.simulation.session_deltas["psilocybin"]

    assert result_psi.kappa_deltas == pytest.approx(base_psi.kappa_deltas)
    assert result_psi.beta_deltas == pytest.approx(base_psi.beta_deltas)
    assert result_psi.zeta_deltas == pytest.approx(base_psi.zeta_deltas)
    assert result_psi.session_labels == base_psi.session_labels


# ---------------------------------------------------------------------------
# make_power_config — group key consistency
# ---------------------------------------------------------------------------


def test_make_power_config_preserves_group_keys(base: AnalysisConfig) -> None:
    """Result session_deltas keys match groups keys and contain expected groups.

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    result = make_power_config(
        base, n_per_group=20, effect_size_delta=0.0, master_seed=1
    )

    expected_groups = {"psilocybin", "placebo"}
    assert set(result.simulation.groups) == expected_groups
    assert set(result.simulation.session_deltas) == expected_groups
    assert set(result.simulation.groups) == set(result.simulation.session_deltas)


# ---------------------------------------------------------------------------
# load_power_config
# ---------------------------------------------------------------------------


def test_power_config_loads_from_yaml() -> None:
    """load_power_config returns PowerConfig matching YAML power: section."""
    pc = load_power_config()

    assert pc.n_per_group_grid == [10, 15, 20, 25, 30, 40, 50]
    assert pc.master_seed == 20240101
    assert pc.n_jobs == 100
    assert pc.bf_threshold == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# PowerConfig validation
# ---------------------------------------------------------------------------


def test_power_config_validation() -> None:
    """PowerConfig.__post_init__ raises ValueError for invalid field values."""
    valid_kwargs = {
        "n_per_group_grid": [10, 20],
        "effect_size_grid": [0.0, 0.5],
        "n_iterations": 10,
        "master_seed": 42,
        "n_jobs": 4,
        "bf_threshold": 10.0,
    }

    with pytest.raises(ValueError, match="n_iterations"):
        PowerConfig(**{**valid_kwargs, "n_iterations": 0})

    with pytest.raises(ValueError, match="n_jobs"):
        PowerConfig(**{**valid_kwargs, "n_jobs": 0})

    with pytest.raises(ValueError, match="bf_threshold"):
        PowerConfig(**{**valid_kwargs, "bf_threshold": -1.0})


# ---------------------------------------------------------------------------
# make_power_config — frozen result
# ---------------------------------------------------------------------------


def test_make_power_config_result_is_frozen(base: AnalysisConfig) -> None:
    """The returned AnalysisConfig must be frozen (cannot set attributes).

    Parameters
    ----------
    base : AnalysisConfig
        Baseline config fixture.
    """
    result = make_power_config(
        base, n_per_group=10, effect_size_delta=0.0, master_seed=1
    )

    with pytest.raises(FrozenInstanceError):
        result.simulation = base.simulation  # type: ignore[misc]
