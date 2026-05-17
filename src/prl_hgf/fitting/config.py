"""FitConfig dataclass hierarchy for sampler/mitigation/covariate settings.

Frozen dataclasses with YAML round-trip serialization. Pure data model --
no integration with fitting functions. Validation happens at fit-launch time.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class SamplerConfig:
    """NUTS / Laplace sampler settings."""

    backend: Literal["blackjax", "numpyro", "laplace"] = "blackjax"
    n_chains: int = 4
    n_draws: int = 1000
    n_warmup: int = 1000
    target_accept: float = 0.95
    max_tree_depth: int = 10
    random_seed: int = 42


@dataclass(frozen=True)
class MitigationConfig:
    """Mitigation flags for conditioning cliff."""

    mass_matrix_kind: Literal["diagonal", "dense"] = "diagonal"
    use_laplace_warmup: bool = False
    use_fp64: bool = False
    use_shard_map: bool = False
    non_centered: tuple[str, ...] = ()

    def __hash__(self) -> int:
        """Stable hash for JIT cache reuse.

        Uses a representation that preserves tuple values despite
        ``dataclasses.asdict`` converting tuples to lists.
        """
        items: list[tuple[str, object]] = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            # Ensure tuple stays tuple for consistent hashing
            if isinstance(val, (list, tuple)):
                val = tuple(val)
            items.append((f.name, val))
        return hash(tuple(sorted(items)))


@dataclass(frozen=True)
class CovariateConfig:
    """Hierarchical pooling and covariate settings."""

    pooling: Literal["none", "hierarchical"] = "none"


@dataclass(frozen=True)
class FitConfig:
    """Complete fitting configuration -- single source of truth.

    Parameters
    ----------
    schema_version : int
        Schema version for forward compatibility.
    model_name : str
        Model identifier (e.g. ``"hgf_2level"``, ``"hgf_3level"``).
    sampler : SamplerConfig
        Sampler backend and tuning parameters.
    mitigation : MitigationConfig
        Mitigation flags for conditioning cliff.
    covariate : CovariateConfig
        Hierarchical pooling and covariate settings.
    log_every : int
        Log every N iterations (0 = no logging).
    progressbar : bool
        Show progress bar during sampling.
    """

    schema_version: int = 1
    model_name: str = "hgf_2level"
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    covariate: CovariateConfig = field(default_factory=CovariateConfig)
    log_every: int = 0
    progressbar: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> FitConfig:
        """Load a FitConfig from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        FitConfig
            Parsed configuration with defaults for missing keys.

        Warns
        -----
        UserWarning
            If unrecognized keys are present at any level.
        """
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        # Top-level known keys
        top_fields = {f.name for f in dataclasses.fields(cls)}
        sampler_fields = {f.name for f in dataclasses.fields(SamplerConfig)}
        mitigation_fields = {f.name for f in dataclasses.fields(MitigationConfig)}
        covariate_fields = {f.name for f in dataclasses.fields(CovariateConfig)}

        # Warn on unknown top-level keys
        unknown_top = set(raw.keys()) - top_fields
        if unknown_top:
            warnings.warn(
                f"Unknown top-level config keys ignored: {sorted(unknown_top)}",
                UserWarning,
                stacklevel=2,
            )

        # Extract schema_version
        schema_version = raw.pop("schema_version", 1)

        # Extract nested configs
        sampler_raw = raw.pop("sampler", {}) or {}
        mitigation_raw = raw.pop("mitigation", {}) or {}
        covariate_raw = raw.pop("covariate", {}) or {}

        # Warn on unknown nested keys
        unknown_sampler = set(sampler_raw.keys()) - sampler_fields
        if unknown_sampler:
            warnings.warn(
                f"Unknown sampler config keys ignored: {sorted(unknown_sampler)}",
                UserWarning,
                stacklevel=2,
            )

        unknown_mitigation = set(mitigation_raw.keys()) - mitigation_fields
        if unknown_mitigation:
            warnings.warn(
                f"Unknown mitigation config keys ignored: {sorted(unknown_mitigation)}",
                UserWarning,
                stacklevel=2,
            )

        unknown_covariate = set(covariate_raw.keys()) - covariate_fields
        if unknown_covariate:
            warnings.warn(
                f"Unknown covariate config keys ignored: {sorted(unknown_covariate)}",
                UserWarning,
                stacklevel=2,
            )

        # Filter to known keys only
        sampler_kwargs = {k: v for k, v in sampler_raw.items() if k in sampler_fields}
        mitigation_kwargs = {
            k: v for k, v in mitigation_raw.items() if k in mitigation_fields
        }
        covariate_kwargs = {
            k: v for k, v in covariate_raw.items() if k in covariate_fields
        }

        # Handle non_centered: YAML list -> Python tuple
        if "non_centered" in mitigation_kwargs:
            nc = mitigation_kwargs["non_centered"]
            if nc is None:
                mitigation_kwargs["non_centered"] = ()
            else:
                mitigation_kwargs["non_centered"] = tuple(nc)

        # Build sub-configs
        sampler = SamplerConfig(**sampler_kwargs)
        mitigation = MitigationConfig(**mitigation_kwargs)
        covariate = CovariateConfig(**covariate_kwargs)

        # Remaining top-level keys (after popping nested sections)
        top_kwargs = {k: v for k, v in raw.items() if k in top_fields}

        return cls(
            schema_version=schema_version,
            sampler=sampler,
            mitigation=mitigation,
            covariate=covariate,
            **top_kwargs,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Dump configuration to a YAML file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dataclasses.asdict(self)
        # Convert non_centered list back to list for clean YAML
        # (dataclasses.asdict already does this, but be explicit)
        if "mitigation" in data and "non_centered" in data["mitigation"]:
            data["mitigation"]["non_centered"] = list(
                data["mitigation"]["non_centered"]
            )
        with path.open("w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def to_json(self) -> str:
        """Serialize to compact JSON for idata.attrs provenance.

        Returns
        -------
        str
            JSON string representation of the configuration.
        """
        data = dataclasses.asdict(self)
        # Ensure non_centered is a list for JSON serialization
        if "mitigation" in data and "non_centered" in data["mitigation"]:
            data["mitigation"]["non_centered"] = list(
                data["mitigation"]["non_centered"]
            )
        return json.dumps(data, separators=(",", ":"))
