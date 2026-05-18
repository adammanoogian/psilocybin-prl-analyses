"""HGFPriorSpec -- single source of truth for prior distributions.

Both BlackJAX closure path and NumPyro model path consume the same
HGFPriorSpec object, eliminating prior drift between backends.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class PriorDist:
    """Single prior distribution specification.

    Parameters
    ----------
    family : str
        Distribution family name.
    loc : float
        Location parameter (mean for Normal).
    scale : float
        Scale parameter (std for Normal).
    low : float or None
        Lower truncation bound (None = no lower bound).
    high : float or None
        Upper truncation bound (None = no upper bound).
    """

    family: Literal["normal", "truncated_normal", "half_normal"]
    loc: float = 0.0
    scale: float = 1.0
    low: float | None = None
    high: float | None = None

    def to_numpyro_dist(self):
        """Convert to a numpyro.distributions object.

        Returns
        -------
        numpyro.distributions.Distribution
            The corresponding numpyro distribution.
        """
        import numpyro.distributions as dist

        if self.family == "normal":
            return dist.Normal(loc=self.loc, scale=self.scale)
        elif self.family == "truncated_normal":
            kwargs: dict = {"loc": self.loc, "scale": self.scale}
            if self.low is not None:
                kwargs["low"] = self.low
            if self.high is not None:
                kwargs["high"] = self.high
            return dist.TruncatedNormal(**kwargs)
        elif self.family == "half_normal":
            return dist.HalfNormal(scale=self.scale)
        else:
            msg = f"Unknown distribution family: {self.family}"
            raise ValueError(msg)

    def log_prob(self, value):
        """Evaluate log-probability (delegates to numpyro dist).

        Parameters
        ----------
        value : jnp.ndarray
            Point(s) at which to evaluate.

        Returns
        -------
        jnp.ndarray
            Log-probability values.
        """
        return self.to_numpyro_dist().log_prob(value)


@dataclass(frozen=True)
class HGFPriorSpec:
    """Complete prior specification for HGF model parameters.

    Supports two pooling modes:

    - ``pooling="none"`` (Mode A): Independent per-participant priors.
      Hyperprior fields are ``None``.
    - ``pooling="hierarchical"`` (Mode B): Group-level hyperpriors
      (mu_p, sigma_p) per cognitive parameter with non-centered
      reparameterization support.  Participant-level priors are
      ``Normal`` (required for LocScaleReparam).

    Parameters
    ----------
    omega_2 : PriorDist
        Prior for tonic volatility (log-space learning rate).
    log_beta : PriorDist
        Prior for log inverse temperature.
    zeta : PriorDist
        Prior for stickiness / choice perseveration.
    omega_3 : PriorDist or None
        Prior for meta-volatility (3-level only).
    kappa : PriorDist or None
        Prior for volatility coupling (None = frozen at 1.0).
    mu3_0 : PriorDist or None
        Prior for initial volatility belief (3-level only).
    pooling : str
        Pooling mode: ``"none"`` for Mode A, ``"hierarchical"`` for Mode B.
    omega_2_mu_hyper : PriorDist or None
        Hyperprior on group mean of omega_2.
    omega_2_sigma_hyper : PriorDist or None
        Hyperprior on shared sigma of omega_2.
    log_beta_mu_hyper : PriorDist or None
        Hyperprior on group mean of log_beta.
    log_beta_sigma_hyper : PriorDist or None
        Hyperprior on shared sigma of log_beta.
    zeta_mu_hyper : PriorDist or None
        Hyperprior on group mean of zeta.
    zeta_sigma_hyper : PriorDist or None
        Hyperprior on shared sigma of zeta.
    omega_3_mu_hyper : PriorDist or None
        Hyperprior on group mean of omega_3 (3-level only).
    omega_3_sigma_hyper : PriorDist or None
        Hyperprior on shared sigma of omega_3 (3-level only).
    """

    omega_2: PriorDist
    log_beta: PriorDist
    zeta: PriorDist
    omega_3: PriorDist | None = None
    kappa: PriorDist | None = None
    mu3_0: PriorDist | None = None

    # --- Mode B: pooling and hyperprior fields ---
    pooling: Literal["none", "hierarchical"] = "none"
    omega_2_mu_hyper: PriorDist | None = None
    omega_2_sigma_hyper: PriorDist | None = None
    log_beta_mu_hyper: PriorDist | None = None
    log_beta_sigma_hyper: PriorDist | None = None
    zeta_mu_hyper: PriorDist | None = None
    zeta_sigma_hyper: PriorDist | None = None
    omega_3_mu_hyper: PriorDist | None = None
    omega_3_sigma_hyper: PriorDist | None = None

    @property
    def is_3level(self) -> bool:
        """Whether this spec includes 3-level parameters."""
        return self.omega_3 is not None

    @property
    def is_hierarchical(self) -> bool:
        """Whether this spec uses hierarchical (Mode B) pooling."""
        return self.pooling == "hierarchical"

    @classmethod
    def default_2level(cls) -> HGFPriorSpec:
        """Default 2-level priors matching current hardcoded values."""
        return cls(
            omega_2=PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
        )

    @classmethod
    def default_3level(cls) -> HGFPriorSpec:
        """Default 3-level priors (wider omega_3 -- unified default)."""
        return cls(
            omega_2=PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
            omega_3=PriorDist("truncated_normal", loc=-6.0, scale=2.0, high=0.0),
        )

    @classmethod
    def tight_3level(cls) -> HGFPriorSpec:
        """3-level priors with tight omega_3 (legacy Phase 14.2 mitigation).

        The tight prior collapses the (mu_3, omega_3) funnel arm at
        extreme-negative omega_3.  Superseded by mass-matrix + Laplace
        warmup mitigations (Phases 29-30).
        """
        return cls(
            omega_2=PriorDist("truncated_normal", loc=-3.0, scale=2.0, high=0.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
            omega_3=PriorDist("normal", loc=-6.0, scale=1.0),
        )

    @classmethod
    def default_2level_hierarchical(cls) -> HGFPriorSpec:
        """Default 2-level hierarchical (Mode B) priors.

        Participant-level priors are Normal (not TruncatedNormal)
        because LocScaleReparam requires unconstrained support.
        The hyperprior centering provides soft constraint.

        Hyperprior defaults (Boehm 2018 convention):

        - mu_omega2 ~ Normal(-3, 1)
        - sigma_omega2 ~ HalfNormal(1)
        - mu_log_beta ~ Normal(0, 1.5)
        - sigma_log_beta ~ HalfNormal(1)
        - mu_zeta ~ Normal(0, 2)
        - sigma_zeta ~ HalfNormal(1)
        """
        return cls(
            omega_2=PriorDist("normal", loc=-3.0, scale=2.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
            pooling="hierarchical",
            omega_2_mu_hyper=PriorDist("normal", loc=-3.0, scale=1.0),
            omega_2_sigma_hyper=PriorDist("half_normal", scale=1.0),
            log_beta_mu_hyper=PriorDist("normal", loc=0.0, scale=1.5),
            log_beta_sigma_hyper=PriorDist("half_normal", scale=1.0),
            zeta_mu_hyper=PriorDist("normal", loc=0.0, scale=2.0),
            zeta_sigma_hyper=PriorDist("half_normal", scale=1.0),
        )

    @classmethod
    def default_3level_hierarchical(cls) -> HGFPriorSpec:
        """Default 3-level hierarchical (Mode B) priors.

        Participant-level priors are Normal (not TruncatedNormal)
        because LocScaleReparam requires unconstrained support.
        The hyperprior centering provides soft constraint.

        Hyperprior defaults (Boehm 2018 convention):

        - mu_omega2 ~ Normal(-3, 1)
        - sigma_omega2 ~ HalfNormal(1)
        - mu_log_beta ~ Normal(0, 1.5)
        - sigma_log_beta ~ HalfNormal(1)
        - mu_zeta ~ Normal(0, 2)
        - sigma_zeta ~ HalfNormal(1)
        - mu_omega3 ~ Normal(-6, 1)
        - sigma_omega3 ~ HalfNormal(1)
        """
        return cls(
            omega_2=PriorDist("normal", loc=-3.0, scale=2.0),
            log_beta=PriorDist("normal", loc=0.0, scale=1.5),
            zeta=PriorDist("normal", loc=0.0, scale=2.0),
            omega_3=PriorDist("normal", loc=-6.0, scale=2.0),
            pooling="hierarchical",
            omega_2_mu_hyper=PriorDist("normal", loc=-3.0, scale=1.0),
            omega_2_sigma_hyper=PriorDist("half_normal", scale=1.0),
            log_beta_mu_hyper=PriorDist("normal", loc=0.0, scale=1.5),
            log_beta_sigma_hyper=PriorDist("half_normal", scale=1.0),
            zeta_mu_hyper=PriorDist("normal", loc=0.0, scale=2.0),
            zeta_sigma_hyper=PriorDist("half_normal", scale=1.0),
            omega_3_mu_hyper=PriorDist("normal", loc=-6.0, scale=1.0),
            omega_3_sigma_hyper=PriorDist("half_normal", scale=1.0),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> HGFPriorSpec:
        """Load prior spec from YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML prior specification file.

        Returns
        -------
        HGFPriorSpec
            Parsed prior specification.
        """
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)

        def _parse_prior(d: dict) -> PriorDist:
            return PriorDist(
                family=d["family"],
                loc=d.get("loc", 0.0),
                scale=d.get("scale", 1.0),
                low=d.get("low"),
                high=d.get("high"),
            )

        spec_kwargs: dict = {
            "omega_2": _parse_prior(raw["omega_2"]),
            "log_beta": _parse_prior(raw["log_beta"]),
            "zeta": _parse_prior(raw["zeta"]),
        }
        if "omega_3" in raw:
            spec_kwargs["omega_3"] = _parse_prior(raw["omega_3"])
        if "kappa" in raw:
            spec_kwargs["kappa"] = _parse_prior(raw["kappa"])
        if "mu3_0" in raw:
            spec_kwargs["mu3_0"] = _parse_prior(raw["mu3_0"])

        # Mode B hyperprior fields
        if "pooling" in raw:
            spec_kwargs["pooling"] = raw["pooling"]
        hyper_fields = [
            "omega_2_mu_hyper",
            "omega_2_sigma_hyper",
            "log_beta_mu_hyper",
            "log_beta_sigma_hyper",
            "zeta_mu_hyper",
            "zeta_sigma_hyper",
            "omega_3_mu_hyper",
            "omega_3_sigma_hyper",
        ]
        for field in hyper_fields:
            if field in raw:
                spec_kwargs[field] = _parse_prior(raw[field])

        return cls(**spec_kwargs)

    def to_yaml(self, path: str | Path) -> None:
        """Dump prior spec to YAML.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {}

        # Include pooling if hierarchical
        if self.pooling != "none":
            data["pooling"] = self.pooling

        for fname in [
            "omega_2",
            "log_beta",
            "zeta",
            "omega_3",
            "kappa",
            "mu3_0",
            "omega_2_mu_hyper",
            "omega_2_sigma_hyper",
            "log_beta_mu_hyper",
            "log_beta_sigma_hyper",
            "zeta_mu_hyper",
            "zeta_sigma_hyper",
            "omega_3_mu_hyper",
            "omega_3_sigma_hyper",
        ]:
            val = getattr(self, fname)
            if val is not None:
                d = dataclasses.asdict(val)
                # Remove None entries for cleaner YAML
                d = {k: v for k, v in d.items() if v is not None}
                data[fname] = d

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
