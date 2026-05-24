"""Unit tests for ``prl_hgf.viz.schema``.

Covers:
- NetworkSpec frozen-ness (CONTEXT Decision 4): field assignment raises.
- NetworkSpec asdict + json.dumps round-trip.
- Dual coord-name probe (SCHEMA-03, Decision 131): build_network_spec
  detects both ``"participant"`` (NUTS default, pick_best_cue) and
  ``"participant_id"`` (Laplace + PAT-RL NUTS) via
  ``set(idata.posterior.dims)`` (NOT ``data_vars``).
- Laplace r_hat suppression (SCHEMA-04): when sample_stats contains a
  ``laplace`` data_var, every NodeSpec.r_hat is None.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from typing import cast

import arviz as az
import numpy as np
import pytest

from prl_hgf.models.hgf_3level import build_3level_network
from prl_hgf.models.hgf_3level_patrl import build_3level_network_patrl
from prl_hgf.viz import NetworkSpec, NodeSpec, build_network_spec


@pytest.fixture(scope="session")
def patrl_3l_network():
    """Session-scoped PAT-RL 3-level Network (3 nodes)."""
    return build_3level_network_patrl()


@pytest.fixture(scope="session")
def pbc_3l_network():
    """Session-scoped pick_best_cue 3-level Network (7 nodes)."""
    return build_3level_network()


def _make_synthetic_idata(coord_name: str, with_laplace: bool) -> az.InferenceData:
    """Build minimal synthetic idata for coord-name + Laplace-marker probes.

    Parameters
    ----------
    coord_name
        Either ``"participant"`` or ``"participant_id"``. Selects which
        participant coordinate dim the synthetic posterior carries.
    with_laplace
        If True, add a ``laplace`` data_var to ``sample_stats`` -- simulates
        the VB-Laplace idata signature (see
        ``src/prl_hgf/fitting/laplace_idata.py`` L268-270).

    Returns
    -------
    arviz.InferenceData
        Synthetic idata with ``posterior`` containing 2 variables
        (omega_2, omega_3) sharing the selected participant dim, and
        an optional ``sample_stats`` group with the Laplace marker.
    """
    n_chains, n_draws, n_participants = 1, 10, 2
    post = {
        "omega_2": np.zeros((n_chains, n_draws, n_participants)),
        "omega_3": np.zeros((n_chains, n_draws, n_participants)),
    }
    dims = {"omega_2": [coord_name], "omega_3": [coord_name]}
    coords = {coord_name: list(range(n_participants))}

    sample_stats = None
    if with_laplace:
        sample_stats = {"laplace": np.ones((n_chains, n_draws), dtype=bool)}

    # cast() satisfies mypy: az.from_dict stub returns Any in arviz typeshed
    # (see Decision 143). Zero runtime overhead.
    return cast(
        az.InferenceData,
        az.from_dict(
            posterior=post,
            sample_stats=sample_stats,
            coords=coords,
            dims=dims,
        ),
    )


def test_networkspec_is_frozen(patrl_3l_network):
    """CONTEXT Decision 4: NetworkSpec is @dataclass(frozen=True)."""
    spec = build_network_spec(patrl_3l_network)
    with pytest.raises(FrozenInstanceError):
        spec.n_nodes = 99  # type: ignore[misc]


def test_nodespec_is_frozen(patrl_3l_network):
    """NodeSpec is @dataclass(frozen=True)."""
    spec = build_network_spec(patrl_3l_network)
    with pytest.raises(FrozenInstanceError):
        spec.nodes[0].idx = 99  # type: ignore[misc]


def test_asdict_json_round_trip(patrl_3l_network):
    """NetworkSpec round-trips via dataclasses.asdict + json.dumps."""
    spec = build_network_spec(patrl_3l_network)
    d = asdict(spec)
    s = json.dumps(d)
    recovered = json.loads(s)
    assert recovered["n_nodes"] == 3
    assert len(recovered["nodes"]) == 3
    assert recovered["source_pyhgf_version"] == spec.source_pyhgf_version


def test_no_schema_version_or_extra_field(patrl_3l_network):
    """CONTEXT Decision 4: No schema_version field, no extra dict field."""
    spec = build_network_spec(patrl_3l_network)
    d = asdict(spec)
    assert "schema_version" not in d
    assert "extra" not in d
    for node in d["nodes"]:
        assert "extra" not in node
        assert "schema_version" not in node


def test_coord_name_detected_participant_id(patrl_3l_network):
    """SCHEMA-03 / Decision 131: participant_id coord (Laplace, PAT-RL NUTS)."""
    idata = _make_synthetic_idata("participant_id", with_laplace=False)
    spec = build_network_spec(patrl_3l_network, idata=idata)
    assert spec.coord_name == "participant_id"


def test_coord_name_detected_participant(pbc_3l_network):
    """SCHEMA-03 / Decision 131: participant coord (NUTS default, pick_best_cue)."""
    idata = _make_synthetic_idata("participant", with_laplace=False)
    spec = build_network_spec(pbc_3l_network, idata=idata)
    assert spec.coord_name == "participant"


def test_coord_name_none_without_idata(patrl_3l_network):
    """coord_name is None when idata is not provided."""
    spec = build_network_spec(patrl_3l_network)
    assert spec.coord_name is None


def test_coord_name_explicit_override(patrl_3l_network):
    """When coord_name is explicitly passed, it takes precedence over probe."""
    idata = _make_synthetic_idata("participant_id", with_laplace=False)
    spec = build_network_spec(patrl_3l_network, idata=idata, coord_name="participant")
    assert spec.coord_name == "participant"


def test_r_hat_suppressed_on_laplace_idata(patrl_3l_network):
    """SCHEMA-04: when sample_stats has a 'laplace' data_var, r_hat is None.

    Uses ``"laplace" in idata.sample_stats.data_vars`` membership -- NOT
    ``sample_stats.laplace == True`` (xarray pitfall from research
    Pattern 6).
    """
    idata = _make_synthetic_idata("participant_id", with_laplace=True)
    spec = build_network_spec(patrl_3l_network, idata=idata)
    assert spec.is_laplace is True
    for node in spec.nodes:
        assert node.r_hat is None, (
            f"NodeSpec idx={node.idx} r_hat must be None for Laplace "
            f"idata (SCHEMA-04); got {node.r_hat}"
        )


def test_r_hat_not_suppressed_on_nuts_idata(pbc_3l_network):
    """When sample_stats lacks 'laplace' marker, is_laplace=False and r_hat present.

    r_hat emission is allowed (here: empty tuple indicating presence
    signal; Phase 24 POST-01 populates actual values).
    """
    idata = _make_synthetic_idata("participant", with_laplace=False)
    spec = build_network_spec(pbc_3l_network, idata=idata)
    assert spec.is_laplace is False
    for node in spec.nodes:
        # r_hat will be () (empty tuple indicating presence in scaffold);
        # Phase 24 POST-01 will populate actual values.
        assert node.r_hat is not None, (
            f"NodeSpec idx={node.idx} r_hat must be non-None for NUTS "
            f"idata (scaffold returns empty tuple); got {node.r_hat}"
        )


def test_invalid_role_rejected_post_init():
    """NodeSpec.__post_init__ rejects invalid role strings."""
    with pytest.raises(ValueError, match="role"):
        NodeSpec(idx=0, kind="binary-state", level=1, role="bogus", branch_idx=0)


def test_invalid_kind_rejected_post_init():
    """NodeSpec.__post_init__ rejects invalid kind strings."""
    with pytest.raises(ValueError, match="kind"):
        NodeSpec(
            idx=0,
            kind="nonsense-state",
            level=1,
            role="input",
            branch_idx=0,
        )


def test_duplicate_nodes_rejected_post_init():
    """NetworkSpec.__post_init__ rejects duplicate node idx entries."""
    dup = NodeSpec(idx=0, kind="binary-state", level=1, role="input", branch_idx=0)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        NetworkSpec(n_nodes=2, input_idxs=(0,), nodes=(dup, dup))
