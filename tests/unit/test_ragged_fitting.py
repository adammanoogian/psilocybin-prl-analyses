"""Unit tests for variable-length (ragged) cohort support.

Criterion-based reversal schedules produce sessions with differing trial
counts. ``fit_batch_hierarchical`` handles this by right-padding every
session to the cohort maximum and carrying a ``(P, T_max)`` validity mask
that (a) zeros the log-likelihood of padded trials and (b) freezes the HGF
belief state across them (``_clamped_scan``).

These tests verify the two invariants that make padding safe:

1. **Rectangular identity** — a cohort with equal trial counts pads to
   arrays bit-identical to a plain ``np.stack`` with an all-ones mask.
2. **Padding inertness** — a padded session's log-likelihood equals the
   unpadded session's log-likelihood, and is invariant to the values
   placed in the pad region.

All tests run the 2-level model on tiny arrays (T <= 16) on CPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from prl_hgf.fitting.hierarchical import (  # noqa: E402
    _pad_and_stack,
    build_logp_fn_batched,
)

_T_SHORT = 12
_T_LONG = 16


def _make_session(n_trials: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one partial-feedback session's (input_data, observed, choices)."""
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, 3, size=n_trials)
    rewards = rng.integers(0, 2, size=n_trials).astype(float)

    input_data = np.zeros((n_trials, 3), dtype=float)
    observed = np.zeros((n_trials, 3), dtype=int)
    for t in range(n_trials):
        input_data[t, choices[t]] = rewards[t]
        observed[t, choices[t]] = 1
    return input_data, observed, choices


# ---------------------------------------------------------------------------
# _pad_and_stack
# ---------------------------------------------------------------------------


class TestPadAndStack:
    def test_rectangular_identity(self) -> None:
        """Equal trial counts: no padding, all-ones mask, plain-stack arrays."""
        sessions = [_make_session(_T_SHORT, seed) for seed in (1, 2, 3)]
        inputs, observed, choices = (list(x) for x in zip(*sessions))

        in_arr, obs_arr, ch_arr, mask = _pad_and_stack(inputs, observed, choices)

        np.testing.assert_array_equal(in_arr, np.stack(inputs))
        np.testing.assert_array_equal(obs_arr, np.stack(observed))
        np.testing.assert_array_equal(ch_arr, np.stack(choices))
        np.testing.assert_array_equal(mask, np.ones((3, _T_SHORT), dtype=np.float32))

    def test_ragged_shapes_and_mask(self) -> None:
        """Ragged counts pad to T_max with a prefix-of-ones mask and zero pads."""
        counts = [_T_SHORT, _T_LONG, 14]
        sessions = [_make_session(n, seed=n) for n in counts]
        inputs, observed, choices = (list(x) for x in zip(*sessions))

        in_arr, obs_arr, ch_arr, mask = _pad_and_stack(inputs, observed, choices)

        assert in_arr.shape == (3, _T_LONG, 3)
        assert obs_arr.shape == (3, _T_LONG, 3)
        assert ch_arr.shape == (3, _T_LONG)
        assert mask.shape == (3, _T_LONG)

        for i, n in enumerate(counts):
            np.testing.assert_array_equal(mask[i, :n], 1.0)
            np.testing.assert_array_equal(mask[i, n:], 0.0)
            np.testing.assert_array_equal(in_arr[i, :n], inputs[i])
            np.testing.assert_array_equal(in_arr[i, n:], 0.0)
            np.testing.assert_array_equal(obs_arr[i, n:], 0)
            np.testing.assert_array_equal(ch_arr[i, n:], 0)


# ---------------------------------------------------------------------------
# Masked log-likelihood invariants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def session_short() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _make_session(_T_SHORT, seed=42)


def _logp_batch(
    n_trials: int,
    input_data: np.ndarray,
    observed: np.ndarray,
    choices: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Evaluate the 2-level batched logp for a single-participant batch."""
    import jax.numpy as jnp

    fn, n_params = build_logp_fn_batched(model_name="hgf_2level", n_trials=n_trials)
    assert n_params == 3
    params = (
        jnp.array([-3.0]),  # omega_2
        jnp.array([3.0]),  # beta
        jnp.array([0.5]),  # zeta
    )
    return float(
        fn(
            *params,
            jnp.array(input_data[None], dtype=jnp.float32),
            jnp.array(observed[None], dtype=jnp.int32),
            jnp.array(choices[None], dtype=jnp.int32),
            jnp.array(mask[None], dtype=jnp.float32),
        )
    )


class TestMaskedLogp:
    def test_padded_matches_unpadded(self, session_short) -> None:
        """Padding to T_max with the mask reproduces the unpadded logp."""
        input_data, observed, choices = session_short

        logp_unpadded = _logp_batch(
            _T_SHORT,
            input_data,
            observed,
            choices,
            np.ones(_T_SHORT),
        )

        (in_p, obs_p, ch_p, mask_p) = _pad_and_stack(
            [input_data, _make_session(_T_LONG, seed=7)[0]],
            [observed, _make_session(_T_LONG, seed=7)[1]],
            [choices, _make_session(_T_LONG, seed=7)[2]],
        )
        logp_padded = _logp_batch(_T_LONG, in_p[0], obs_p[0], ch_p[0], mask_p[0])

        assert logp_padded == pytest.approx(logp_unpadded, rel=1e-5)

    def test_pad_values_are_inert(self, session_short) -> None:
        """Garbage in the pad region cannot change the logp (belief freeze)."""
        input_data, observed, choices = session_short
        n_pad = _T_LONG - _T_SHORT

        mask = np.concatenate([np.ones(_T_SHORT), np.zeros(n_pad)])

        def padded_with(fill_reward: float, fill_choice: int) -> float:
            in_p = np.concatenate(
                [input_data, np.full((n_pad, 3), fill_reward)], axis=0
            )
            obs_p = np.concatenate([observed, np.ones((n_pad, 3), dtype=int)], axis=0)
            ch_p = np.concatenate(
                [choices, np.full(n_pad, fill_choice, dtype=int)], axis=0
            )
            return _logp_batch(_T_LONG, in_p, obs_p, ch_p, mask)

        logp_zeros = padded_with(0.0, 0)
        logp_garbage = padded_with(1.0, 2)

        assert logp_garbage == pytest.approx(logp_zeros, rel=1e-6)
