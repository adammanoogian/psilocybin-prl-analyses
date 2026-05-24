"""Unit tests for HGFPriorSpec and PriorDist."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from prl_hgf.fitting.priors import HGFPriorSpec, PriorDist

# ---------------------------------------------------------------------------
# Paths to configs/priors/ YAML files
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "priors"


# ---------------------------------------------------------------------------
# PriorDist tests
# ---------------------------------------------------------------------------


class TestPriorDist:
    """Tests for PriorDist dataclass."""

    def test_normal_to_numpyro_dist(self):
        """Normal family returns numpyro Normal with correct params."""
        import numpyro.distributions as dist

        p = PriorDist("normal", loc=1.0, scale=2.0)
        d = p.to_numpyro_dist()
        assert isinstance(d, dist.Normal)
        assert float(d.loc) == 1.0
        assert float(d.scale) == 2.0

    def test_truncated_normal_to_numpyro_dist(self):
        """TruncatedNormal family returns a truncated distribution."""
        from numpyro.distributions import Distribution

        p = PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0)
        d = p.to_numpyro_dist()
        assert isinstance(d, Distribution)
        # Should give higher log_prob at loc than far from it
        lp_at_loc = float(d.log_prob(jnp.array(-3.0)))
        lp_far = float(d.log_prob(jnp.array(-10.0)))
        assert lp_at_loc > lp_far

    def test_half_normal_to_numpyro_dist(self):
        """HalfNormal family returns correct numpyro dist."""
        import numpyro.distributions as dist

        p = PriorDist("half_normal", scale=1.5)
        d = p.to_numpyro_dist()
        assert isinstance(d, dist.HalfNormal)
        assert float(d.scale) == 1.5

    def test_unknown_family_raises(self):
        """Unknown family raises ValueError."""
        p = PriorDist("cauchy", loc=0.0, scale=1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown distribution family"):
            p.to_numpyro_dist()

    def test_log_prob_normal(self):
        """log_prob delegates to numpyro dist correctly."""
        p = PriorDist("normal", loc=0.0, scale=1.0)
        # Standard normal at 0 should give log(1/sqrt(2*pi))
        expected = -0.5 * np.log(2.0 * np.pi)
        actual = float(p.log_prob(jnp.array(0.0)))
        assert abs(actual - expected) < 1e-5

    def test_log_prob_truncated_concentrates_mass(self):
        """TruncatedNormal concentrates mass within bounds."""
        p = PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0)
        # Value well within bounds should have higher lp than boundary
        lp_within = float(p.log_prob(jnp.array(-3.0)))
        lp_at_high = float(p.log_prob(jnp.array(0.0)))
        assert lp_within > lp_at_high


# ---------------------------------------------------------------------------
# HGFPriorSpec factory tests
# ---------------------------------------------------------------------------


class TestHGFPriorSpecFactories:
    """Tests for HGFPriorSpec class methods."""

    def test_default_2level_matches_hardcoded(self):
        """default_2level() matches the original hardcoded priors."""
        spec = HGFPriorSpec.default_2level()
        assert spec.omega_2.family == "truncated_normal"
        assert spec.omega_2.loc == -3.0
        assert spec.omega_2.scale == 2.0
        assert spec.omega_2.high == 0.0
        assert spec.log_beta.family == "normal"
        assert spec.log_beta.loc == 0.0
        assert spec.log_beta.scale == 1.5
        assert spec.zeta.family == "normal"
        assert spec.zeta.loc == 0.0
        assert spec.zeta.scale == 2.0
        assert spec.omega_3 is None
        assert spec.kappa is None
        assert spec.mu3_0 is None
        assert not spec.is_3level

    def test_default_3level_matches_hardcoded(self):
        """default_3level() matches the original hardcoded priors."""
        spec = HGFPriorSpec.default_3level()
        assert spec.omega_2.family == "truncated_normal"
        assert spec.omega_2.loc == -3.0
        assert spec.omega_2.scale == 2.0
        assert spec.omega_2.high == 0.0
        assert spec.log_beta.family == "normal"
        assert spec.log_beta.loc == 0.0
        assert spec.log_beta.scale == 1.5
        assert spec.zeta.family == "normal"
        assert spec.zeta.loc == 0.0
        assert spec.zeta.scale == 2.0
        assert spec.omega_3 is not None
        assert spec.omega_3.family == "truncated_normal"
        assert spec.omega_3.loc == -6.0
        assert spec.omega_3.scale == 2.0
        assert spec.omega_3.high == 0.0
        assert spec.is_3level

    def test_tight_3level_differs_only_in_omega3(self):
        """tight_3level() matches default_3level() except for omega_3."""
        default = HGFPriorSpec.default_3level()
        tight = HGFPriorSpec.tight_3level()
        # Shared priors are identical
        assert tight.omega_2 == default.omega_2
        assert tight.log_beta == default.log_beta
        assert tight.zeta == default.zeta
        # omega_3 differs
        assert tight.omega_3 != default.omega_3
        assert tight.omega_3.family == "normal"
        assert tight.omega_3.loc == -6.0
        assert tight.omega_3.scale == 1.0

    def test_is_3level_property(self):
        """is_3level reflects presence of omega_3."""
        assert not HGFPriorSpec.default_2level().is_3level
        assert HGFPriorSpec.default_3level().is_3level
        assert HGFPriorSpec.tight_3level().is_3level


# ---------------------------------------------------------------------------
# YAML round-trip tests
# ---------------------------------------------------------------------------


class TestYAMLRoundTrip:
    """Tests for YAML serialization/deserialization."""

    def test_roundtrip_2level(self, tmp_path):
        """2-level spec survives YAML round-trip."""
        spec = HGFPriorSpec.default_2level()
        yaml_path = tmp_path / "test_2level.yaml"
        spec.to_yaml(yaml_path)
        loaded = HGFPriorSpec.from_yaml(yaml_path)
        assert loaded == spec

    def test_roundtrip_3level(self, tmp_path):
        """3-level spec survives YAML round-trip."""
        spec = HGFPriorSpec.default_3level()
        yaml_path = tmp_path / "test_3level.yaml"
        spec.to_yaml(yaml_path)
        loaded = HGFPriorSpec.from_yaml(yaml_path)
        assert loaded == spec

    def test_roundtrip_tight_3level(self, tmp_path):
        """Tight 3-level spec survives YAML round-trip."""
        spec = HGFPriorSpec.tight_3level()
        yaml_path = tmp_path / "test_tight.yaml"
        spec.to_yaml(yaml_path)
        loaded = HGFPriorSpec.from_yaml(yaml_path)
        assert loaded == spec

    def test_roundtrip_with_all_fields(self, tmp_path):
        """Spec with kappa and mu3_0 survives YAML round-trip."""
        spec = HGFPriorSpec(
            omega_2=PriorDist("normal", loc=-4.0, scale=2.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
            omega_3=PriorDist("normal", loc=-6.0, scale=2.0),
            kappa=PriorDist("truncated_normal", loc=1.0, scale=0.5, low=0.0, high=2.0),
            mu3_0=PriorDist("normal", loc=1.0, scale=1.0),
        )
        yaml_path = tmp_path / "test_full.yaml"
        spec.to_yaml(yaml_path)
        loaded = HGFPriorSpec.from_yaml(yaml_path)
        assert loaded == spec


# ---------------------------------------------------------------------------
# configs/priors/ YAML file tests
# ---------------------------------------------------------------------------


class TestConfigPriorYAMLFiles:
    """Tests that configs/priors/ YAML files load correctly."""

    def test_pick_best_cue_2level(self):
        """pick_best_cue_2level.yaml matches default_2level."""
        path = _CONFIGS_DIR / "pick_best_cue_2level.yaml"
        spec = HGFPriorSpec.from_yaml(path)
        assert spec == HGFPriorSpec.default_2level()

    def test_pick_best_cue_3level(self):
        """pick_best_cue_3level.yaml matches default_3level."""
        path = _CONFIGS_DIR / "pick_best_cue_3level.yaml"
        spec = HGFPriorSpec.from_yaml(path)
        assert spec == HGFPriorSpec.default_3level()

    def test_pick_best_cue_3level_tight(self):
        """pick_best_cue_3level_tight.yaml matches tight_3level."""
        path = _CONFIGS_DIR / "pick_best_cue_3level_tight.yaml"
        spec = HGFPriorSpec.from_yaml(path)
        assert spec == HGFPriorSpec.tight_3level()

    def test_pat_rl_2level(self):
        """pat_rl_2level.yaml loads and has normal omega_2."""
        path = _CONFIGS_DIR / "pat_rl_2level.yaml"
        spec = HGFPriorSpec.from_yaml(path)
        assert spec.omega_2.family == "normal"
        assert spec.omega_2.loc == -4.0
        assert spec.omega_2.scale == 2.0
        assert not spec.is_3level

    def test_pat_rl_3level(self):
        """pat_rl_3level.yaml loads with kappa and mu3_0."""
        path = _CONFIGS_DIR / "pat_rl_3level.yaml"
        spec = HGFPriorSpec.from_yaml(path)
        assert spec.is_3level
        assert spec.omega_3 is not None
        assert spec.kappa is not None
        assert spec.kappa.family == "truncated_normal"
        assert spec.kappa.low == 0.0
        assert spec.kappa.high == 2.0
        assert spec.mu3_0 is not None
        assert spec.mu3_0.loc == 1.0


# ---------------------------------------------------------------------------
# Parity test: to_numpyro_dist and log_prob consistency
# ---------------------------------------------------------------------------


class TestParityNumpyroLogProb:
    """Verify that PriorDist.log_prob equals numpyro dist.log_prob."""

    @pytest.mark.parametrize(
        "prior",
        [
            PriorDist("normal", loc=0.0, scale=1.5),
            PriorDist("normal", loc=-3.0, scale=2.0),
            PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0),
            PriorDist("truncated_normal", loc=-6.0, scale=2.0, high=0.0),
            PriorDist("normal", loc=-6.0, scale=1.0),
            PriorDist("half_normal", scale=1.0),
        ],
    )
    def test_logprob_parity(self, prior):
        """PriorDist.log_prob matches its numpyro dist at test points."""
        test_points = jnp.array([-5.0, -3.0, -1.0, 0.0, 1.0])
        d = prior.to_numpyro_dist()
        for x in test_points:
            expected = float(d.log_prob(x))
            actual = float(prior.log_prob(x))
            assert abs(actual - expected) < 1e-6, (
                f"Mismatch at x={float(x)}: expected={expected}, actual={actual}"
            )
