"""HGF model definitions for the PRL pick_best_cue pipeline.

Provides two model variants:

* **2-level binary HGF**: Three parallel binary branches (one per cue) with
  independent level-1 continuous-state belief nodes.  Suitable for stationary
  or slowly changing environments.

* **3-level binary HGF**: Same three branches with an additional shared
  continuous-state volatility parent, allowing the model to track
  environment-wide changes in learning rate (meta-volatility).

Both models are built as :class:`pyhgf.model.Network` instances using
``add_nodes()`` calls with explicit edge wiring.  Partial feedback (unchosen
cues do not update) is handled via the ``observed`` mask passed to
:meth:`~pyhgf.model.Network.input_data`.

Public symbols
--------------
Factory functions:
    :func:`build_2level_network`, :func:`build_3level_network`

Input/output helpers:
    :func:`prepare_input_data`, :func:`extract_beliefs`,
    :func:`extract_beliefs_3level`

Response function:
    :func:`softmax_stickiness_surprise`

Node index constants:
    :data:`INPUT_NODES`, :data:`BELIEF_NODES`, :data:`VOLATILITY_NODE`,
    :data:`N_CUES`
"""

from __future__ import annotations

from prl_hgf.models.hgf_2level import (
    BELIEF_NODES,
    INPUT_NODES,
    N_CUES,
    build_2level_network,
    extract_beliefs,
    prepare_input_data,
)
from prl_hgf.models.hgf_3level import (
    VOLATILITY_NODE,
    build_3level_network,
    extract_beliefs_3level,
)
from prl_hgf.models.response import softmax_stickiness_surprise

__all__ = [
    "build_2level_network",
    "build_3level_network",
    "prepare_input_data",
    "extract_beliefs",
    "extract_beliefs_3level",
    "softmax_stickiness_surprise",
    "INPUT_NODES",
    "BELIEF_NODES",
    "VOLATILITY_NODE",
    "N_CUES",
]
