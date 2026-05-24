"""Unit tests for ``prl_hgf.viz.roles``.

Covers:
- ``infer_role`` adjacency-only derivation across 2-level, 3-level,
  multimodal 2-level, and shared-volatility-parent topologies (VIZ-05,
  P16 guard).
- ``assign_levels_and_branches`` BFS dedup and shared-node branch_idx
  promotion to ``None``.
"""

from __future__ import annotations

import pytest

from prl_hgf.models.hgf_2level import build_2level_network
from prl_hgf.models.hgf_2level_patrl import build_2level_network_patrl
from prl_hgf.models.hgf_3level import build_3level_network
from prl_hgf.models.hgf_3level_patrl import build_3level_network_patrl
from prl_hgf.viz.roles import assign_levels_and_branches, infer_role

# Multimodal PAT-RL is a parallel 2-level variant with two binary input
# channels sharing one continuous belief parent. The source module may
# not be present in every checkout (it is a recent parallel-stack
# addition and not part of the default pick_best_cue pipeline), so the
# import is soft-gated here. Tests requiring it are decorated with
# ``@pytest.mark.skipif(not _HAS_MULTIMODAL, ...)`` rather than
# hard-erroring at collection time.
try:
    from prl_hgf.models.hgf_2level_multimodal_patrl import (
        build_2level_multimodal_network_patrl,
    )

    _HAS_MULTIMODAL = True
except ImportError:  # pragma: no cover - depends on local checkout state
    _HAS_MULTIMODAL = False

    def build_2level_multimodal_network_patrl():  # type: ignore[misc]
        """Placeholder so fixture decorator resolves; skipped at runtime."""
        raise RuntimeError("multimodal builder unavailable (test should be skipped)")


@pytest.fixture(scope="session")
def patrl_3l():
    """Session-scoped PAT-RL 3-level Network (input_idxs=(0,), 3 nodes)."""
    return build_3level_network_patrl()


@pytest.fixture(scope="session")
def patrl_2l():
    """Session-scoped PAT-RL 2-level Network (input_idxs=(0,), 2 nodes)."""
    return build_2level_network_patrl()


@pytest.fixture(scope="session")
def pbc_3l():
    """Session-scoped pick_best_cue 3-level Network (input_idxs=(0,2,4), 7 nodes)."""
    return build_3level_network()


@pytest.fixture(scope="session")
def pbc_2l():
    """Session-scoped pick_best_cue 2-level Network (input_idxs=(0,2,4), 6 nodes)."""
    return build_2level_network()


@pytest.fixture(scope="session")
def multimodal_2l():
    """Session-scoped 2-level multimodal PAT-RL Network.

    Two binary input nodes (0, 1) both coupled as value-children to a
    single shared continuous-state belief parent (2).
    """
    if not _HAS_MULTIMODAL:
        pytest.skip("hgf_2level_multimodal_patrl not available")
    return build_2level_multimodal_network_patrl()


def test_input_role_via_input_idxs(patrl_3l):
    """role='input' when idx is in input_idxs."""
    for input_idx in patrl_3l.input_idxs:
        assert infer_role(input_idx, patrl_3l.edges, patrl_3l.input_idxs) == "input"


def test_volatility_role_pick_best_cue_node6(pbc_3l):
    """pick_best_cue node 6 appears in nodes 1/3/5 volatility_parents.

    Role must be 'volatility' by adjacency-structure alone -- NOT by
    reading per-node attribute dict keys (P16 guard).
    """
    assert infer_role(6, pbc_3l.edges, pbc_3l.input_idxs) == "volatility"


def test_value_role_fallback_pick_best_cue_node1(pbc_3l):
    """Node 1 is neither an input nor referenced as any volatility_parent.

    Falls through to the 'value' default.
    """
    assert infer_role(1, pbc_3l.edges, pbc_3l.input_idxs) == "value"


def test_volatility_role_patrl_3level():
    """PAT-RL 3-level has one volatility parent (the last node)."""
    net = build_3level_network_patrl()
    # Find the node whose idx appears in another node's volatility_parents
    vol_nodes = set()
    for e in net.edges:
        if e.volatility_parents is not None:
            vol_nodes.update(e.volatility_parents)
    assert len(vol_nodes) >= 1
    for idx in vol_nodes:
        assert infer_role(idx, net.edges, net.input_idxs) == "volatility"


@pytest.mark.skipif(
    not _HAS_MULTIMODAL,
    reason="hgf_2level_multimodal_patrl module not available in this checkout",
)
def test_multimodal_2level_role_coverage(multimodal_2l):
    """Multimodal 2-level: 2 binary inputs + 1 shared continuous belief.

    Both inputs are 'input'; the shared belief is 'value' (no
    volatility_parents reference it -- 2-level has no volatility layer).
    """
    roles = {
        idx: infer_role(idx, multimodal_2l.edges, multimodal_2l.input_idxs)
        for idx in range(multimodal_2l.n_nodes)
    }
    for input_idx in multimodal_2l.input_idxs:
        assert roles[input_idx] == "input"
    non_input_idxs = [
        i for i in range(multimodal_2l.n_nodes) if i not in multimodal_2l.input_idxs
    ]
    for idx in non_input_idxs:
        assert roles[idx] == "value", (
            f"Multimodal 2-level node {idx} expected 'value' (no "
            f"volatility layer present), got {roles[idx]}"
        )


def test_shared_parent_branch_promoted_to_none(pbc_3l):
    """pick_best_cue 3-level shared volatility parent (node 6).

    ``branch_idx`` must be promoted to ``None`` via BFS contested-branch
    detection.
    """
    _levels, branches, _order = assign_levels_and_branches(
        pbc_3l.edges, pbc_3l.input_idxs
    )
    assert 6 in branches
    assert branches[6] is None, (
        f"Shared volatility parent node 6 must have branch_idx=None; got {branches[6]}"
    )


def test_bfs_dedup_order_and_count_pbc_3level(pbc_3l):
    """BFS emits each node exactly once and covers all 7 nodes.

    pick_best_cue 3-level has 7 unique nodes -- the shared volatility
    parent (node 6) is deduplicated across the 3 branches.
    """
    levels, _branches, order = assign_levels_and_branches(
        pbc_3l.edges, pbc_3l.input_idxs
    )
    assert len(order) == 7
    assert len(set(order)) == 7, f"Duplicates in BFS order: {order}"
    # Input nodes are level 1; node 6 (shared) is level 3.
    for input_idx in pbc_3l.input_idxs:
        assert levels[input_idx] == 1
    assert levels[6] == 3


def test_patrl_3level_bfs_three_nodes(patrl_3l):
    """PAT-RL 3-level BFS discovers all 3 nodes, single-branch."""
    _levels, branches, order = assign_levels_and_branches(
        patrl_3l.edges, patrl_3l.input_idxs
    )
    assert len(order) == 3
    # Single-branch network: every node should have branch_idx == 0.
    for idx, br in branches.items():
        assert br == 0, f"PAT-RL 3-level node {idx} expected branch_idx=0, got {br}"


def test_roles_does_not_read_node_parameters():
    """P16 guard: roles.py does not read from node_parameters dict keys.

    Structural source-level check -- infer_role and
    assign_levels_and_branches must only reference edge adjacency fields.
    """
    from pathlib import Path

    from prl_hgf.viz import roles as roles_module

    src = Path(roles_module.__file__).read_text()
    assert "node_parameters" not in src, (
        "roles.py must not read node_parameters (Pitfall P16). "
        "Role must be adjacency-only."
    )
