"""2-level binary-state HGF for the PAT-RL task (parallel stack).

Topology: single ``binary-state`` input node (idx 0) whose value parent is a
single ``continuous-state`` node (idx 1) with ``tonic_volatility = omega_2``.
Contrast with :mod:`prl_hgf.models.hgf_2level` which has three parallel
binary branches for pick_best_cue partial feedback.

Node layout
-----------
::

    Node 0: binary-state   — binary state input (safe=0 / dangerous=1)
    Node 1: continuous-state — value parent (posterior belief mu_2 in log-odds)

Input nodes (for ``net.input_idxs``): (0,)
Belief node (continuous-state, source of ``mu2``): 1

Parameter mapping
-----------------
+---------------+-------------------+----------+
| Project name  | pyhgf attribute   | Node(s)  |
+===============+===================+==========+
| omega_2       | tonic_volatility  | 1        |
+---------------+-------------------+----------+

Notes
-----
Do **not** modify this file to add pick_best_cue concerns.  The PAT-RL and
pick_best_cue pipelines are parallel stacks sharing no imports below the
``pyhgf`` library level.
"""

from __future__ import annotations

import numpy as np
from pyhgf.model import Network

# ---------------------------------------------------------------------------
# Module-level constants (exported for downstream use)
# ---------------------------------------------------------------------------

#: Node index for the single binary-state input node.
INPUT_NODE: int = 0

#: Node index for the continuous-state value parent (level-1 belief).
BELIEF_NODE: int = 1


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_2level_network_patrl(omega_2: float = -4.0) -> Network:
    """Build a 2-level binary HGF with a single input channel.

    Creates one binary-HGF branch: a binary-state input node (idx 0) and a
    continuous-state value parent (idx 1).  The topology is a strict scalar
    version of the pick_best_cue 3-branch model.

    Parameters
    ----------
    omega_2 : float, optional
        Tonic volatility of the continuous value parent.  Default ``-4.0``.

    Returns
    -------
    pyhgf.model.Network
        Network with ``input_idxs = (0,)`` and one belief node (idx 1).

    Examples
    --------
    >>> net = build_2level_network_patrl()
    >>> net.input_idxs
    (0,)
    >>> len(net.edges)
    2
    """
    net = Network()
    net.add_nodes(kind="binary-state")  # node 0: binary input
    net.add_nodes(
        kind="continuous-state",
        value_children=INPUT_NODE,  # node 1: value parent of node 0
        node_parameters={"tonic_volatility": omega_2, "mean": 0.0, "precision": 1.0},
    )
    # Explicitly set input node index (avoids auto-detection ambiguity)
    net.input_idxs = (INPUT_NODE,)
    return net


# ---------------------------------------------------------------------------
# Belief extraction helper
# ---------------------------------------------------------------------------


def extract_beliefs_patrl(net: Network) -> dict[str, np.ndarray]:
    """Extract per-trial belief trajectories from a fitted network.

    Must be called after :meth:`~pyhgf.model.Network.input_data` has been
    run so that ``net.node_trajectories`` is populated.

    Parameters
    ----------
    net : pyhgf.model.Network
        Network object after calling ``input_data``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with keys:

        * ``"mu2"``: continuous value posterior mean (log-odds), shape
          ``(n_trials,)``.
        * ``"sigma2"``: posterior variance ``1 / precision``, shape
          ``(n_trials,)``.
        * ``"p_state"``: sigmoid of ``mu2``; probability of state=1
          (dangerous), shape ``(n_trials,)``.
        * ``"expected_precision"``: from ``temp["effective_precision"]`` of
          node 1 (pyhgf 0.2.x attribute path), shape ``(n_trials,)``.
          Falls back to zeros if key absent (forward compatibility).

    Examples
    --------
    >>> import numpy as np
    >>> net = build_2level_network_patrl()
    >>> u = np.zeros(10)
    >>> net.input_data(input_data=u[:, None], time_steps=np.ones(10))
    >>> beliefs = extract_beliefs_patrl(net)
    >>> set(beliefs.keys()) >= {"mu2", "sigma2", "p_state", "expected_precision"}
    True
    """
    traj = net.node_trajectories[BELIEF_NODE]
    mu2 = np.asarray(traj["mean"])
    sigma2 = 1.0 / np.asarray(traj["precision"])
    p_state = 1.0 / (1.0 + np.exp(-mu2))
    # pyhgf 0.2.x: effective_precision lives in the temp sub-dict
    temp = traj.get("temp", {})
    exp_prec = np.asarray(temp.get("effective_precision", np.zeros_like(mu2)))
    return {
        "mu2": mu2,
        "sigma2": sigma2,
        "p_state": p_state,
        "expected_precision": exp_prec,
    }


__all__ = [
    "INPUT_NODE",
    "BELIEF_NODE",
    "build_2level_network_patrl",
    "extract_beliefs_patrl",
]
