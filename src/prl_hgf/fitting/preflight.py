"""Pre-flight validation for FitConfig before launching sampler."""

from __future__ import annotations

import warnings

from prl_hgf.fitting.config import FitConfig
from prl_hgf.fitting.priors import HGFPriorSpec

_PARAMS_PER_PARTICIPANT = {
    "hgf_2level": 3,  # omega_2, log_beta, zeta
    "hgf_3level": 4,  # omega_2, omega_3, log_beta, zeta
}


def estimate_mass_matrix_memory(
    n_params: int,
    mass_matrix_kind: str,
    n_chains: int,
    use_pmap: bool,
) -> int:
    """Estimate bytes for mass matrix allocation.

    Parameters
    ----------
    n_params : int
        Total dimension D of the joint parameter vector.
    mass_matrix_kind : str
        "diagonal" or "dense".
    n_chains : int
        Number of MCMC chains.
    use_pmap : bool
        Whether pmap replicates the matrix across devices.

    Returns
    -------
    int
        Estimated bytes. Zero for diagonal (stored as 1D vector).
    """
    if mass_matrix_kind == "diagonal":
        return 0
    return n_params**2 * 8 * n_chains * (4 if use_pmap else 1)


def _get_device_memory_bytes() -> int:
    """Detect available device memory in bytes.

    Returns
    -------
    int
        Device memory in bytes. Falls back to system RAM, then 8 GB default.
    """
    try:
        import jax

        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats and "bytes_limit" in stats:
                return stats["bytes_limit"]
    except Exception:
        pass

    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass

    warnings.warn(
        "Cannot detect device memory (no JAX GPU, no psutil). "
        "Using conservative 8 GB default for pre-flight check.",
        UserWarning,
        stacklevel=2,
    )
    return 8 * 1024**3


def _human_bytes(n: int | float) -> str:
    """Format byte count as human-readable string.

    Parameters
    ----------
    n : int or float
        Number of bytes.

    Returns
    -------
    str
        Human-readable byte string (e.g. "4.9 MB").
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def validate_fit_config(
    fit_config: FitConfig,
    prior_spec: HGFPriorSpec,
    n_participants: int = 1,
) -> None:
    """Validate config + prior spec compatibility before launching fit.

    Refuses dense mass matrix configurations when estimated memory exceeds
    25% of detected device memory.

    Parameters
    ----------
    fit_config : FitConfig
        Configuration to validate.
    prior_spec : HGFPriorSpec
        Prior specification to validate against config.
    n_participants : int
        Number of participants in the cohort.

    Raises
    ------
    ValueError
        If dense mass matrix memory exceeds 25% of device memory.
    """
    mass_matrix_kind = fit_config.mitigation.mass_matrix_kind

    # low_rank falls through to dense in practice (BlackJAX 1.5 has no API)
    if mass_matrix_kind == "low_rank":
        mass_matrix_kind = "dense"

    if mass_matrix_kind == "diagonal":
        return

    n_free = _PARAMS_PER_PARTICIPANT.get(fit_config.model_name, 3)
    d = n_free * n_participants
    n_chains = fit_config.sampler.n_chains

    try:
        import jax

        use_pmap = jax.device_count() >= n_chains
    except Exception:
        use_pmap = False

    estimated = estimate_mass_matrix_memory(d, mass_matrix_kind, n_chains, use_pmap)
    device_mem = _get_device_memory_bytes()
    threshold = int(device_mem * 0.25)

    if estimated > threshold:
        msg = (
            f"Dense mass matrix estimated at {_human_bytes(estimated)} "
            f"exceeds 25% of device memory ({_human_bytes(threshold)} limit). "
            f"D={d} (n_params_per_participant={n_free} x P={n_participants}), "
            f"n_chains={n_chains}, pmap={use_pmap}. "
            f"Alternatives: use mass_matrix_kind='low_rank' (when BlackJAX adds "
            f"support), reduce n_participants, or run on M3 cluster with more "
            f"GPU memory (Phase 35)."
        )
        raise ValueError(msg)
