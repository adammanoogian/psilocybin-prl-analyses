"""3-level binary-state HGF for the PAT-RL task (parallel stack).

Extends the 2-level PAT-RL model by adding a continuous-state volatility
parent (node 2) that modulates the learning rate of the value parent (node 1)
via volatility coupling of strength ``kappa``.

Node layout
-----------
::

    Node 0: binary-state   — binary state input (safe=0 / dangerous=1)
    Node 1: continuous-state — value parent (posterior belief mu_2 in log-odds)
    Node 2: continuous-state — volatility parent (mu_3 in log-volatility space)

Input nodes (for ``net.input_idxs``): (0,)
Belief node (continuous-state, source of ``mu2``): 1
Volatility node (source of ``mu3``): 2

Parameter mapping
-----------------
+---------------+-------------------+-----------+
| Project name  | pyhgf attribute   | Node(s)   |
+===============+===================+===========+
| omega_2       | tonic_volatility  | 1         |
+---------------+-------------------+-----------+
| omega_3       | tonic_volatility  | 2         |
+---------------+-------------------+-----------+
| kappa         | coupling strength | edge 2→1  |
+---------------+-------------------+-----------+
| mu3_0         | mean (initial)    | 2         |
+---------------+-------------------+-----------+

Notes
-----
The ω₃ recovery caveat applies: ω₃ is known to be poorly recovered with
binary data.  Primary hypotheses focus on ω₂ and κ.

The kappa coupling strength is passed via the ``volatility_children`` tuple
``([BELIEF_NODE], [kappa])``; this is the pyhgf 0.2.x API for setting
per-child coupling strengths at construction time.

Do **not** modify this file to add pick_best_cue concerns.  The PAT-RL and
pick_best_cue pipelines are parallel stacks sharing no imports below the
``pyhgf`` library level.
"""

from __future__ import annotations

import numpy as np
from pyhgf.model import Network

from prl_hgf.models.hgf_2level_patrl import (
    BELIEF_NODE,
    INPUT_NODE,
    extract_beliefs_patrl,
)

# ---------------------------------------------------------------------------
# Module-level constants (exported for downstream use)
# ---------------------------------------------------------------------------

#: Node index for the continuous-state volatility parent.
VOLATILITY_NODE: int = 2

# Re-export shared constants so callers can import everything from here.
__all__ = [
    "INPUT_NODE",
    "BELIEF_NODE",
    "VOLATILITY_NODE",
    "build_3level_network_patrl",
    "extract_beliefs_patrl_3level",
]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_3level_network_patrl(
    omega_2: float = -4.0,
    omega_3: float = -6.0,
    kappa: float = 1.0,
    mu3_0: float = 1.0,
) -> Network:
    """Build a 3-level binary HGF with a single input channel.

    Constructs the same binary input + value parent as
    :func:`~prl_hgf.models.hgf_2level_patrl.build_2level_network_patrl`,
    then adds a single volatility parent (node 2) connected to node 1 via
    volatility coupling of strength ``kappa``.

    Parameters
    ----------
    omega_2 : float, optional
        Tonic volatility for the level-1 value parent (node 1).
        Default ``-4.0``.
    omega_3 : float, optional
        Tonic meta-volatility for the level-2 volatility parent (node 2).
        Default ``-6.0``.
    kappa : float, optional
        Volatility coupling strength between node 2 and node 1.
        Default ``1.0``.
    mu3_0 : float, optional
        Initial prior mean for the volatility node (node 2).
        Default ``1.0``.

    Returns
    -------
    pyhgf.model.Network
        Network with ``input_idxs = (0,)`` and three nodes
        (input + value + volatility).

    Examples
    --------
    >>> net = build_3level_network_patrl()
    >>> net.input_idxs
    (0,)
    >>> len(net.edges)
    3
    """
    net = Network()
    net.add_nodes(kind="binary-state")  # node 0: binary input
    net.add_nodes(
        kind="continuous-state",
        value_children=INPUT_NODE,  # node 1: value parent of node 0
        node_parameters={"tonic_volatility": omega_2, "mean": 0.0, "precision": 1.0},
    )
    # Node 2: volatility parent of node 1.
    # kappa is set via the coupling tuple in volatility_children (pyhgf 0.2.x API).
    net.add_nodes(
        kind="continuous-state",
        volatility_children=([BELIEF_NODE], [kappa]),  # couple to belief node
        node_parameters={
            "tonic_volatility": omega_3,
            "mean": mu3_0,
            "precision": 1.0,
        },
    )  # node 2
    # Explicitly set input node index (avoids auto-detection ambiguity)
    net.input_idxs = (INPUT_NODE,)
    return net


# ---------------------------------------------------------------------------
# Belief extraction helper (3-level extension)
# ---------------------------------------------------------------------------


def extract_beliefs_patrl_3level(net: Network) -> dict[str, np.ndarray]:
    """Extract belief trajectories from a 3-level forward-pass network.

    Extends :func:`~prl_hgf.models.hgf_2level_patrl.extract_beliefs_patrl`
    with the volatility node trajectories.

    Must be called after :meth:`~pyhgf.model.Network.input_data` has been run.

    Parameters
    ----------
    net : pyhgf.model.Network
        3-level network object after calling ``input_data``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with all keys from
        :func:`~prl_hgf.models.hgf_2level_patrl.extract_beliefs_patrl` plus:

        * ``"mu3"``: posterior mean of the volatility parent (node 2,
          ``"mean"`` field), shape ``(n_trials,)``.
        * ``"sigma3"``: posterior standard deviation of node 2
          (``1 / precision``), shape ``(n_trials,)``.
        * ``"epsilon3"``: volatility prediction error from node 1's temp
          sub-dict (``"volatility_prediction_error"``), shape
          ``(n_trials,)``.  Falls back to zeros if key absent.

    Examples
    --------
    >>> import numpy as np
    >>> net = build_3level_network_patrl()
    >>> u = np.zeros(10)
    >>> net.input_data(input_data=u[:, None], time_steps=np.ones(10))
    >>> beliefs = extract_beliefs_patrl_3level(net)
    >>> {"mu3", "sigma3", "epsilon3"}.issubset(beliefs.keys())
    True
    """
    # Start with the 2-level beliefs (mu2, sigma2, p_state, expected_precision)
    beliefs = extract_beliefs_patrl(net)

    traj_vol = net.node_trajectories[VOLATILITY_NODE]
    beliefs["mu3"] = np.asarray(traj_vol["mean"])
    beliefs["sigma3"] = 1.0 / np.asarray(traj_vol["precision"])

    # volatility_prediction_error is in node 1 (BELIEF_NODE) temp sub-dict
    traj_belief = net.node_trajectories[BELIEF_NODE]
    temp_belief = traj_belief.get("temp", {})
    beliefs["epsilon3"] = np.asarray(
        temp_belief.get(
            "volatility_prediction_error",
            np.zeros_like(beliefs["mu3"]),
        )
    )

    return beliefs
