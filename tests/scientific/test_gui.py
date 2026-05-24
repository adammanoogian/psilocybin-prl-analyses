"""Unit tests for the ParamExplorer GUI module.

Uses the Agg matplotlib backend so tests run headless without a display.
The backend is set before any matplotlib import to ensure it takes effect.

All tests share a single module-scoped ``explorer`` fixture to amortise
the expensive JAX JIT warm-up (~5-10 s) across the full test session.

Mark: ``@pytest.mark.scientific`` — JAX JIT compilation makes these tests slow
on first run.  Exclude with ``pytest -m "not slow"`` for fast CI loops.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede any matplotlib.pyplot import

import numpy as np
import pytest

pytest.importorskip("ipywidgets")

from prl_hgf.gui.explorer import PROFILES, ParamExplorer

# ---------------------------------------------------------------------------
# Module-scoped fixture (shared across all tests to avoid re-warming JAX)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
@pytest.mark.scientific
def explorer() -> ParamExplorer:
    """Instantiate ParamExplorer once per module (JAX warm-up is expensive).

    Returns
    -------
    ParamExplorer
        A fully initialised explorer instance using default config and
        3-level model (the default toggle value).
    """
    return ParamExplorer()


# ---------------------------------------------------------------------------
# Test 1: Slider keys
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_explorer_has_all_sliders(explorer: ParamExplorer) -> None:
    """ParamExplorer._sliders must have exactly the expected 7 keys."""
    expected = {"omega_2", "omega_3", "kappa", "beta", "zeta", "mu_1_0", "mu_3_0"}
    assert set(explorer._sliders.keys()) == expected


# ---------------------------------------------------------------------------
# Test 2: continuous_update=False
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_sliders_continuous_update_false(explorer: ParamExplorer) -> None:
    """Every slider must have continuous_update=False (fire only on release)."""
    for name, slider in explorer._sliders.items():
        assert slider.continuous_update is False, (
            f"Slider '{name}' has continuous_update=True; expected False."
        )


# ---------------------------------------------------------------------------
# Test 3: Initial belief cache populated
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_initial_beliefs_cached(explorer: ParamExplorer) -> None:
    """After init, _cached_beliefs_ must contain all three cue belief keys."""
    assert explorer._cached_beliefs_ is not None, (
        "_cached_beliefs_ is None after initialisation."
    )
    for key in ("p_reward_cue0", "p_reward_cue1", "p_reward_cue2"):
        assert key in explorer._cached_beliefs_, (
            f"Expected key '{key}' missing from _cached_beliefs_."
        )


# ---------------------------------------------------------------------------
# Test 4: Initial choice probability cache populated
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_initial_choice_probs_cached(explorer: ParamExplorer) -> None:
    """After init, _cached_choice_probs_ must have shape (n_trials, 3) summing to 1."""
    cp = explorer._cached_choice_probs_
    assert cp is not None, "_cached_choice_probs_ is None after initialisation."
    n_trials = len(explorer._trials)
    assert cp.shape == (n_trials, 3), (
        f"Expected shape ({n_trials}, 3), got {cp.shape}."
    )
    row_sums = cp.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, np.ones(n_trials), atol=1e-5,
        err_msg="Choice probability rows do not sum to 1.",
    )


# ---------------------------------------------------------------------------
# Test 5: HGF param change triggers new forward pass (beliefs change)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_hgf_param_triggers_forward_pass(explorer: ParamExplorer) -> None:
    """Changing omega_2 slider must produce different p_reward_cue0 beliefs."""
    # Capture state before change
    old_p0 = explorer._cached_beliefs_["p_reward_cue0"].copy()

    # Move omega_2 far enough to guarantee a different trajectory
    original_val = explorer._sliders["omega_2"].value
    new_val = -5.0 if original_val != -5.0 else -3.0
    explorer._sliders["omega_2"].value = new_val

    new_p0 = explorer._cached_beliefs_["p_reward_cue0"]
    assert not np.allclose(old_p0, new_p0, atol=1e-6), (
        "p_reward_cue0 did not change after omega_2 slider was moved. "
        f"old[0]={old_p0[0]:.4f}, new[0]={new_p0[0]:.4f}."
    )

    # Restore original value for subsequent tests
    explorer._sliders["omega_2"].value = original_val


# ---------------------------------------------------------------------------
# Test 6: Response param change skips forward pass (same beliefs object)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_response_param_skips_forward_pass(explorer: ParamExplorer) -> None:
    """Changing beta must reuse cached beliefs (same dict object) but update choice probs."""
    # Force a forward pass first to ensure cache is fresh
    explorer._run_forward_pass()

    beliefs_before = explorer._cached_beliefs_
    cp_before = (
        explorer._cached_choice_probs_.copy()
        if explorer._cached_choice_probs_ is not None
        else None
    )

    # Change beta (response param — must NOT trigger forward pass)
    original_beta = explorer._sliders["beta"].value
    new_beta = 5.0 if original_beta != 5.0 else 1.0
    explorer._sliders["beta"].value = new_beta

    # Same dict object → no new forward pass
    assert explorer._cached_beliefs_ is beliefs_before, (
        "_cached_beliefs_ was replaced when beta was changed. "
        "Expected the same dict object (no forward pass)."
    )

    # Choice probs SHOULD have changed
    if cp_before is not None:
        assert not np.allclose(
            explorer._cached_choice_probs_, cp_before, atol=1e-6
        ), "Choice probs did not change after beta slider was moved."

    # Restore
    explorer._sliders["beta"].value = original_beta


# ---------------------------------------------------------------------------
# Test 7: Preset loads correct slider values
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_preset_loads_values(explorer: ParamExplorer) -> None:
    """Triggering post-concussion preset sets all slider values from PROFILES."""
    preset_name = "post-concussion"
    explorer._preset_toggle.value = preset_name

    profile = PROFILES[preset_name]
    for param, expected in profile.items():
        actual = explorer._sliders[param].value
        assert abs(actual - expected) < 1e-9, (
            f"Slider '{param}' has value {actual!r} after preset load; "
            f"expected {expected!r}."
        )


# ---------------------------------------------------------------------------
# Test 8: Model toggle hides 3-level sliders when 2-level selected
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_model_toggle_hides_3level_sliders(explorer: ParamExplorer) -> None:
    """Selecting 2-level model must set display='none' on omega_3, kappa, mu_3_0."""
    explorer._model_toggle.value = "2-level"

    for name in ("omega_3", "kappa", "mu_3_0"):
        display_val = explorer._sliders[name].layout.display
        assert display_val == "none", (
            f"Slider '{name}' has layout.display={display_val!r} after 2-level toggle; "
            f"expected 'none'."
        )


# ---------------------------------------------------------------------------
# Test 9: Model toggle shows 3-level sliders when 3-level selected
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_model_toggle_shows_3level_sliders(explorer: ParamExplorer) -> None:
    """Selecting 3-level model must set display='' on omega_3, kappa, mu_3_0."""
    explorer._model_toggle.value = "3-level"

    for name in ("omega_3", "kappa", "mu_3_0"):
        display_val = explorer._sliders[name].layout.display
        assert display_val == "", (
            f"Slider '{name}' has layout.display={display_val!r} after 3-level toggle; "
            f"expected ''."
        )


# ---------------------------------------------------------------------------
# Test 10: Ground-truth probability shape and value range
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_ground_truth_probs_shape(explorer: ParamExplorer) -> None:
    """_gt_probs must have shape (420, 3) with all values in [0, 1]."""
    gt = explorer._gt_probs
    assert gt.shape == (420, 3), (
        f"Expected _gt_probs shape (420, 3), got {gt.shape}."
    )
    assert gt.min() >= 0.0, f"_gt_probs contains values < 0: min={gt.min()}"
    assert gt.max() <= 1.0, f"_gt_probs contains values > 1: max={gt.max()}"


# ---------------------------------------------------------------------------
# Test 11: Figure has four axes
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_figure_has_four_axes(explorer: ParamExplorer) -> None:
    """The explorer figure must have exactly 4 axes."""
    assert len(explorer._axes) == 4, (
        f"Expected 4 axes, got {len(explorer._axes)}."
    )


# ---------------------------------------------------------------------------
# Test 12: Effective learning rate shape and range
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_effective_lr_shape(explorer: ParamExplorer) -> None:
    """_cached_lr_ must have shape (420, 3) with values in (0, 1]."""
    lr = explorer._cached_lr_
    assert lr is not None, "_cached_lr_ is None after initialisation."
    assert lr.shape == (420, 3), (
        f"Expected _cached_lr_ shape (420, 3), got {lr.shape}."
    )
    assert lr.min() > 0.0, (
        f"_cached_lr_ contains non-positive values: min={lr.min()}"
    )
    assert lr.max() <= 1.0 + 1e-6, (
        f"_cached_lr_ contains values > 1: max={lr.max()}"
    )
