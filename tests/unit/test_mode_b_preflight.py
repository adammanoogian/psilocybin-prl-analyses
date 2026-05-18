"""Unit tests for Mode B pre-flight collinearity check."""

from __future__ import annotations

import numpy as np
import pytest

from prl_hgf.fitting.preflight import check_covariate_collinearity


class TestCovariateCollinearity:
    """Tests for check_covariate_collinearity."""

    def test_passes_orthogonal_covariate(self):
        """Orthogonal covariate and group should not raise."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        group_idx = np.array([0] * 100 + [1] * 100)
        # Should not raise
        check_covariate_collinearity(x, group_idx, threshold=0.7)

    def test_raises_collinear_covariate(self):
        """Highly collinear covariate should raise ValueError."""
        group_idx = np.array([0] * 100 + [1] * 100)
        x = group_idx.astype(float) + np.random.default_rng(42).standard_normal(
            200
        ) * 0.05
        with pytest.raises(ValueError, match="collinearity check failed"):
            check_covariate_collinearity(x, group_idx, threshold=0.7)

    def test_error_message_includes_actual_r(self):
        """Error message should show actual |r| value."""
        group_idx = np.array([0] * 50 + [1] * 50)
        x = group_idx.astype(float)  # perfect correlation
        with pytest.raises(ValueError, match=r"\|cor.*= 1\.000"):
            check_covariate_collinearity(x, group_idx)

    def test_error_message_includes_remediation(self):
        """Error message should suggest group-mean-centering."""
        group_idx = np.array([0] * 50 + [1] * 50)
        x = group_idx.astype(float)
        with pytest.raises(ValueError, match="group-mean-centered"):
            check_covariate_collinearity(x, group_idx)

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        rng = np.random.default_rng(123)
        group_idx = np.array([0] * 100 + [1] * 100)
        # Create moderate correlation via deterministic construction
        x_moderate = group_idx.astype(float) + rng.standard_normal(200) * 0.8
        r = abs(np.corrcoef(x_moderate, group_idx)[0, 1])
        # Should pass at default threshold 0.7 (moderate noise r ~ 0.5-0.6)
        if r < 0.7:
            check_covariate_collinearity(x_moderate, group_idx, threshold=0.7)
        # Should fail at strict threshold 0.3
        if r > 0.3:
            with pytest.raises(ValueError):
                check_covariate_collinearity(
                    x_moderate, group_idx, threshold=0.3
                )

    def test_none_covariate_not_accepted(self):
        """Function requires arrays, not None.

        The None guard is enforced at the call site in hierarchical.py,
        not in the function itself.
        """
        with pytest.raises((TypeError, AttributeError)):
            check_covariate_collinearity(None, np.array([0, 1]))

    def test_expected_vs_actual_in_message(self):
        """Error message includes both expected threshold and actual |r|."""
        group_idx = np.array([0] * 50 + [1] * 50)
        x = group_idx.astype(float)
        with pytest.raises(ValueError) as exc_info:
            check_covariate_collinearity(x, group_idx, threshold=0.7)
        msg = str(exc_info.value)
        assert "Expected |r| < 0.7" in msg
        assert "got 1.000" in msg
