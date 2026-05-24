"""Canary integration tests for ``prl_hgf.viz.inspector.inspect_network``.

Covers:
- Node count dedup (Pitfall P3): pick_best_cue 3-level returns 7
  nodes, not 9, because the shared volatility parent (node 6) is
  visited once via ``set[int]`` dedup.
- Input-node emission (Pitfall 8): every ``input_idxs`` member appears
  in the emitted ``nodes`` list. We do NOT filter by ``node_type`` like
  pyhgf's upstream ``plot_network`` does.
- Strict unknown-kind guard (CONTEXT Decision 3): fabricated edge with
  an out-of-range integer ``node_type`` raises ValueError naming the
  offending idx, observed code, and known-codes set.
"""

from __future__ import annotations

import pytest

from prl_hgf.models.hgf_2level import build_2level_network
from prl_hgf.models.hgf_2level_patrl import build_2level_network_patrl
from prl_hgf.models.hgf_3level import build_3level_network
from prl_hgf.models.hgf_3level_patrl import build_3level_network_patrl
from prl_hgf.viz import inspect_network


@pytest.fixture(scope="session")
def patrl_3level_network():
    """Session-scoped PAT-RL 3-level Network (3 nodes, input_idxs=(0,))."""
    return build_3level_network_patrl()


@pytest.fixture(scope="session")
def patrl_2level_network():
    """Session-scoped PAT-RL 2-level Network (2 nodes, input_idxs=(0,))."""
    return build_2level_network_patrl()


@pytest.fixture(scope="session")
def pick_best_cue_3level_network():
    """Session-scoped pick_best_cue 3-level Network (7 nodes, 3 branches sharing vol parent)."""
    return build_3level_network()


@pytest.fixture(scope="session")
def pick_best_cue_2level_network():
    """Session-scoped pick_best_cue 2-level Network (6 nodes: 3 binary + 3 continuous)."""
    return build_2level_network()


def test_patrl_3level_emits_three_nodes(patrl_3level_network):
    """PAT-RL 3-level Network has 3 nodes; inspector emits all three."""
    result = inspect_network(patrl_3level_network)
    assert result["n_nodes"] == 3
    assert len(result["nodes"]) == 3


def test_patrl_2level_emits_two_nodes(patrl_2level_network):
    """PAT-RL 2-level Network has 2 nodes; inspector emits both."""
    result = inspect_network(patrl_2level_network)
    assert result["n_nodes"] == 2
    assert len(result["nodes"]) == 2


def test_pick_best_cue_3level_dedup(pick_best_cue_3level_network):
    """pick_best_cue 3-level has 3 branches sharing volatility parent 6.

    Naive per-branch traversal would emit 9 nodes. BFS with ``set[int]``
    dedup must emit 7.
    """
    result = inspect_network(pick_best_cue_3level_network)
    assert len(result["nodes"]) == 7, (
        f"Expected 7 (shared vol parent deduplicated), got "
        f"{len(result['nodes'])}. Emitted idxs: "
        f"{[n['idx'] for n in result['nodes']]}"
    )
    idxs = [n["idx"] for n in result["nodes"]]
    assert len(set(idxs)) == 7, f"Duplicate idxs: {idxs}"


def test_pick_best_cue_2level_emits_six_nodes(pick_best_cue_2level_network):
    """pick_best_cue 2-level has 3 binary + 3 continuous = 6 nodes."""
    result = inspect_network(pick_best_cue_2level_network)
    assert len(result["nodes"]) == 6


@pytest.mark.parametrize(
    "fixture_name,expected_inputs",
    [
        ("patrl_3level_network", (0,)),
        ("patrl_2level_network", (0,)),
        ("pick_best_cue_3level_network", (0, 2, 4)),
        ("pick_best_cue_2level_network", (0, 2, 4)),
    ],
)
def test_all_input_idxs_emitted(request, fixture_name, expected_inputs):
    """Pitfall 8 guard: every input_idx appears as an emitted NodeSpec.

    pyhgf's upstream plot_network filters binary-state nodes
    (node_type == 1) -- the inspector must NOT do that.
    """
    network = request.getfixturevalue(fixture_name)
    result = inspect_network(network)
    emitted_idxs = {n["idx"] for n in result["nodes"]}
    for input_idx in expected_inputs:
        assert input_idx in emitted_idxs, (
            f"input_idx {input_idx} missing from emitted nodes "
            f"{sorted(emitted_idxs)}. Inspector incorrectly filtered "
            f"by node_type."
        )


def test_unknown_node_type_raises_value_error(patrl_3level_network):
    """Strict guard (CONTEXT Decision 3): unknown integer node_type codes.

    Must raise ValueError with idx, observed code, and known-codes list.
    """
    from types import SimpleNamespace

    # Fabricate an edge with node_type=99 at index 0.
    real_edge = patrl_3level_network.edges[0]
    fake_edge = SimpleNamespace(
        node_type=99,
        value_parents=real_edge.value_parents,
        volatility_parents=real_edge.volatility_parents,
        value_children=real_edge.value_children,
        volatility_children=real_edge.volatility_children,
        coupling_fn=getattr(real_edge, "coupling_fn", ()),
    )
    # Splice the fake edge into a shallow copy of edges.
    edges = list(patrl_3level_network.edges)
    edges[0] = fake_edge
    fake_network = SimpleNamespace(
        attributes=patrl_3level_network.attributes,
        edges=tuple(edges),
        input_idxs=patrl_3level_network.input_idxs,
        n_nodes=patrl_3level_network.n_nodes,
    )

    with pytest.raises(ValueError) as excinfo:
        inspect_network(fake_network)
    msg = str(excinfo.value)
    assert "99" in msg, f"ValueError did not name observed code: {msg}"
    assert "0" in msg, f"ValueError did not name offending idx: {msg}"
    assert "Known codes" in msg or "known" in msg.lower(), (
        f"ValueError did not include known-codes set: {msg}"
    )


def test_inspector_does_not_import_from_prl_hgf():
    """VIZ-02 guard: inspector.py imports zero symbols from prl_hgf.*.

    This is a structural isolation check. Inspector is a reader of
    pyhgf Network objects passed to it -- it must not depend on any
    prl_hgf module.
    """
    from pathlib import Path

    from prl_hgf import viz  # noqa: F401

    src = Path(viz.__file__).parent / "inspector.py"
    content = src.read_text()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("from prl_hgf") or stripped.startswith("import prl_hgf"):
            raise AssertionError(
                f"inspector.py must not import from prl_hgf.*; got: {line}"
            )
