"""TDD tests for memory pre-flight estimator."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestEstimateMassMatrixMemory:
    """Tests for estimate_mass_matrix_memory formula."""

    def test_diagonal_returns_zero(self):
        from prl_hgf.fitting.preflight import estimate_mass_matrix_memory

        assert estimate_mass_matrix_memory(120, "diagonal", 4, False) == 0

    def test_dense_no_pmap(self):
        from prl_hgf.fitting.preflight import estimate_mass_matrix_memory

        # D=120, n_chains=4, no pmap: 120^2 * 8 * 4 * 1 = 460800
        assert estimate_mass_matrix_memory(120, "dense", 4, False) == 460_800

    def test_dense_with_pmap(self):
        from prl_hgf.fitting.preflight import estimate_mass_matrix_memory

        # D=120, n_chains=4, pmap: 120^2 * 8 * 4 * 4 = 1843200
        assert estimate_mass_matrix_memory(120, "dense", 4, True) == 1_843_200


class TestValidateFitConfig:
    """Tests for validate_fit_config memory guard."""

    def test_refuses_dense_over_threshold(self):
        from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig
        from prl_hgf.fitting.preflight import validate_fit_config
        from prl_hgf.fitting.priors import HGFPriorSpec

        cfg = FitConfig(
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=4),
            mitigation=MitigationConfig(mass_matrix_kind="dense"),
        )
        spec = HGFPriorSpec.default_3level()

        # Mock device with only 1 MB memory -- D=400 (4 params x 100 participants)
        # Memory = 400^2 * 8 * 4 * 1 = 5,120,000 bytes > 25% of 1MB = 262,144
        with (
            patch(
                "prl_hgf.fitting.preflight._get_device_memory_bytes",
                return_value=1_000_000,
            ),
            pytest.raises(ValueError, match="low_rank"),
        ):
            validate_fit_config(cfg, spec, n_participants=100)

    def test_allows_dense_under_threshold(self):
        from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig
        from prl_hgf.fitting.preflight import validate_fit_config
        from prl_hgf.fitting.priors import HGFPriorSpec

        cfg = FitConfig(
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=4),
            mitigation=MitigationConfig(mass_matrix_kind="dense"),
        )
        spec = HGFPriorSpec.default_3level()

        # Mock device with 80 GB -- D=20 (4 params x 5 participants)
        # Memory = 20^2 * 8 * 4 * 1 = 12,800 bytes << 25% of 80GB
        with patch(
            "prl_hgf.fitting.preflight._get_device_memory_bytes",
            return_value=80_000_000_000,
        ):
            validate_fit_config(cfg, spec, n_participants=5)  # should not raise

    def test_diagonal_always_passes(self):
        from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig
        from prl_hgf.fitting.preflight import validate_fit_config
        from prl_hgf.fitting.priors import HGFPriorSpec

        cfg = FitConfig(
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=4),
            mitigation=MitigationConfig(mass_matrix_kind="diagonal"),
        )
        spec = HGFPriorSpec.default_3level()

        # Mock device with only 1 byte -- should still pass with diagonal
        with patch(
            "prl_hgf.fitting.preflight._get_device_memory_bytes",
            return_value=1,
        ):
            validate_fit_config(cfg, spec, n_participants=1000)  # should not raise

    def test_error_message_content(self):
        from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig
        from prl_hgf.fitting.preflight import validate_fit_config
        from prl_hgf.fitting.priors import HGFPriorSpec

        cfg = FitConfig(
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=4),
            mitigation=MitigationConfig(mass_matrix_kind="dense"),
        )
        spec = HGFPriorSpec.default_3level()

        with patch(
            "prl_hgf.fitting.preflight._get_device_memory_bytes",
            return_value=1_000_000,
        ):
            with pytest.raises(ValueError) as exc_info:
                validate_fit_config(cfg, spec, n_participants=100)
            msg = str(exc_info.value)
            assert "low_rank" in msg
            assert "M3" in msg
