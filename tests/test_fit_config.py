"""Unit tests for FitConfig dataclass hierarchy."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import yaml

from prl_hgf.fitting.config import (
    CovariateConfig,
    FitConfig,
    MitigationConfig,
    SamplerConfig,
)


class TestSamplerConfig:
    """Tests for SamplerConfig construction."""

    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.backend == "blackjax"
        assert cfg.n_chains == 4
        assert cfg.n_draws == 1000
        assert cfg.n_warmup == 1000
        assert cfg.target_accept == 0.95
        assert cfg.max_tree_depth == 10
        assert cfg.random_seed == 42

    def test_explicit_values(self):
        cfg = SamplerConfig(
            backend="numpyro",
            n_chains=8,
            n_draws=2000,
            n_warmup=500,
            target_accept=0.9,
            max_tree_depth=12,
            random_seed=123,
        )
        assert cfg.backend == "numpyro"
        assert cfg.n_chains == 8
        assert cfg.n_draws == 2000
        assert cfg.n_warmup == 500
        assert cfg.target_accept == 0.9
        assert cfg.max_tree_depth == 12
        assert cfg.random_seed == 123


class TestMitigationConfig:
    """Tests for MitigationConfig construction and hashing."""

    def test_defaults(self):
        cfg = MitigationConfig()
        assert cfg.mass_matrix_kind == "diagonal"
        assert cfg.use_laplace_warmup is False
        assert cfg.use_fp64 is False
        assert cfg.use_shard_map is False
        assert cfg.non_centered == ()

    def test_non_centered_tuple(self):
        cfg = MitigationConfig(non_centered=("omega2", "kappa"))
        assert cfg.non_centered == ("omega2", "kappa")
        assert isinstance(cfg.non_centered, tuple)

    def test_hash_stability(self):
        """Same values produce the same hash across constructions."""
        cfg1 = MitigationConfig(
            mass_matrix_kind="dense",
            use_fp64=True,
            non_centered=("omega2", "kappa"),
        )
        cfg2 = MitigationConfig(
            mass_matrix_kind="dense",
            use_fp64=True,
            non_centered=("omega2", "kappa"),
        )
        assert hash(cfg1) == hash(cfg2)

    def test_hash_different_values(self):
        """Different configs produce different hashes."""
        cfg1 = MitigationConfig(mass_matrix_kind="diagonal")
        cfg2 = MitigationConfig(mass_matrix_kind="dense")
        assert hash(cfg1) != hash(cfg2)

    def test_hash_usable_in_dict(self):
        """MitigationConfig can be used as dict key (JIT cache)."""
        cfg = MitigationConfig(non_centered=("omega2",))
        cache = {cfg: "compiled_fn"}
        assert cache[cfg] == "compiled_fn"


class TestCovariateConfig:
    """Tests for CovariateConfig construction."""

    def test_defaults(self):
        cfg = CovariateConfig()
        assert cfg.pooling == "none"

    def test_hierarchical(self):
        cfg = CovariateConfig(pooling="hierarchical")
        assert cfg.pooling == "hierarchical"


class TestFitConfig:
    """Tests for FitConfig construction and serialization."""

    def test_defaults(self):
        cfg = FitConfig()
        assert cfg.schema_version == 1
        assert cfg.model_name == "hgf_2level"
        assert cfg.sampler == SamplerConfig()
        assert cfg.mitigation == MitigationConfig()
        assert cfg.covariate == CovariateConfig()
        assert cfg.log_every == 0
        assert cfg.progressbar is True

    def test_explicit_construction(self):
        cfg = FitConfig(
            schema_version=2,
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=8),
            mitigation=MitigationConfig(mass_matrix_kind="dense"),
            covariate=CovariateConfig(pooling="hierarchical"),
            log_every=50,
            progressbar=False,
        )
        assert cfg.schema_version == 2
        assert cfg.model_name == "hgf_3level"
        assert cfg.sampler.n_chains == 8
        assert cfg.mitigation.mass_matrix_kind == "dense"
        assert cfg.covariate.pooling == "hierarchical"
        assert cfg.log_every == 50
        assert cfg.progressbar is False

    def test_frozen(self):
        """FitConfig is immutable."""
        cfg = FitConfig()
        with pytest.raises(AttributeError):
            cfg.model_name = "hgf_3level"  # type: ignore[misc]


class TestFitConfigYaml:
    """Tests for YAML round-trip serialization."""

    def test_round_trip_defaults(self, tmp_path: Path):
        """Write defaults -> read -> assert equal."""
        cfg = FitConfig()
        path = tmp_path / "test.yaml"
        cfg.to_yaml(path)
        loaded = FitConfig.from_yaml(path)
        assert loaded == cfg

    def test_round_trip_explicit(self, tmp_path: Path):
        """Write explicit config -> read -> assert equal."""
        cfg = FitConfig(
            schema_version=1,
            model_name="hgf_3level",
            sampler=SamplerConfig(backend="numpyro", n_chains=8, n_draws=2000),
            mitigation=MitigationConfig(
                mass_matrix_kind="dense",
                use_fp64=True,
                non_centered=("omega2", "kappa"),
            ),
            covariate=CovariateConfig(pooling="hierarchical"),
            log_every=100,
            progressbar=False,
        )
        path = tmp_path / "explicit.yaml"
        cfg.to_yaml(path)
        loaded = FitConfig.from_yaml(path)
        assert loaded == cfg

    def test_round_trip_canonical_configs(self):
        """All canonical configs in configs/fit/ round-trip correctly."""
        configs_dir = Path(__file__).parent.parent / "configs" / "fit"
        if not configs_dir.exists():
            pytest.skip("configs/fit/ not found")
        for yaml_file in sorted(configs_dir.glob("*.yaml")):
            cfg = FitConfig.from_yaml(yaml_file)
            assert cfg.schema_version == 1

    def test_unknown_top_key_warns(self, tmp_path: Path):
        """Unknown top-level key produces UserWarning."""
        data = {
            "schema_version": 1,
            "model_name": "hgf_2level",
            "unknown_key": "should_warn",
        }
        path = tmp_path / "unknown.yaml"
        with path.open("w") as f:
            yaml.dump(data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = FitConfig.from_yaml(path)
            assert len(w) == 1
            assert "unknown_key" in str(w[0].message)
        assert cfg.model_name == "hgf_2level"

    def test_unknown_nested_key_warns(self, tmp_path: Path):
        """Unknown nested key produces UserWarning."""
        data = {
            "schema_version": 1,
            "sampler": {
                "backend": "blackjax",
                "bogus_field": 999,
            },
        }
        path = tmp_path / "nested_unknown.yaml"
        with path.open("w") as f:
            yaml.dump(data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = FitConfig.from_yaml(path)
            assert len(w) == 1
            assert "bogus_field" in str(w[0].message)
        assert cfg.sampler.backend == "blackjax"

    def test_missing_keys_get_defaults(self, tmp_path: Path):
        """Missing keys get their default values."""
        data = {"schema_version": 1, "model_name": "hgf_3level"}
        path = tmp_path / "minimal.yaml"
        with path.open("w") as f:
            yaml.dump(data, f)

        cfg = FitConfig.from_yaml(path)
        assert cfg.model_name == "hgf_3level"
        assert cfg.sampler == SamplerConfig()
        assert cfg.mitigation == MitigationConfig()
        assert cfg.covariate == CovariateConfig()
        assert cfg.log_every == 0
        assert cfg.progressbar is True

    def test_empty_yaml_gets_all_defaults(self, tmp_path: Path):
        """Completely empty YAML gets all defaults."""
        path = tmp_path / "empty.yaml"
        path.write_text("")

        cfg = FitConfig.from_yaml(path)
        assert cfg == FitConfig()

    def test_schema_version_preserved(self, tmp_path: Path):
        """schema_version survives round-trip."""
        cfg = FitConfig(schema_version=2)
        path = tmp_path / "versioned.yaml"
        cfg.to_yaml(path)
        loaded = FitConfig.from_yaml(path)
        assert loaded.schema_version == 2

    def test_non_centered_list_to_tuple(self, tmp_path: Path):
        """YAML list for non_centered becomes tuple in dataclass."""
        data = {
            "schema_version": 1,
            "mitigation": {
                "non_centered": ["omega2", "kappa", "mu3_0"],
            },
        }
        path = tmp_path / "nc.yaml"
        with path.open("w") as f:
            yaml.dump(data, f)

        cfg = FitConfig.from_yaml(path)
        assert cfg.mitigation.non_centered == ("omega2", "kappa", "mu3_0")
        assert isinstance(cfg.mitigation.non_centered, tuple)

    def test_non_centered_null_to_empty_tuple(self, tmp_path: Path):
        """YAML null for non_centered becomes empty tuple."""
        data = {
            "schema_version": 1,
            "mitigation": {
                "non_centered": None,
            },
        }
        path = tmp_path / "nc_null.yaml"
        with path.open("w") as f:
            yaml.dump(data, f)

        cfg = FitConfig.from_yaml(path)
        assert cfg.mitigation.non_centered == ()


class TestFitConfigJson:
    """Tests for JSON serialization."""

    def test_to_json_valid(self):
        """to_json produces valid JSON."""
        cfg = FitConfig()
        j = cfg.to_json()
        parsed = json.loads(j)
        assert parsed["schema_version"] == 1
        assert parsed["model_name"] == "hgf_2level"
        assert parsed["sampler"]["backend"] == "blackjax"

    def test_to_json_reconstructable(self):
        """JSON output can reconstruct equivalent config."""
        cfg = FitConfig(
            model_name="hgf_3level",
            sampler=SamplerConfig(n_chains=8),
            mitigation=MitigationConfig(non_centered=("omega2",), use_fp64=True),
        )
        j = cfg.to_json()
        parsed = json.loads(j)

        # Reconstruct from parsed JSON
        sampler = SamplerConfig(**parsed["sampler"])
        mit_data = parsed["mitigation"]
        mit_data["non_centered"] = tuple(mit_data["non_centered"])
        mitigation = MitigationConfig(**mit_data)
        covariate = CovariateConfig(**parsed["covariate"])
        reconstructed = FitConfig(
            schema_version=parsed["schema_version"],
            model_name=parsed["model_name"],
            sampler=sampler,
            mitigation=mitigation,
            covariate=covariate,
            log_every=parsed["log_every"],
            progressbar=parsed["progressbar"],
        )
        assert reconstructed == cfg

    def test_to_json_compact(self):
        """JSON is compact (no extra whitespace)."""
        cfg = FitConfig()
        j = cfg.to_json()
        assert " " not in j  # compact separators, no spaces
