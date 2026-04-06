"""GUI module for the PRL HGF analysis pipeline.

Provides the :class:`~prl_hgf.gui.explorer.ParamExplorer` interactive
widget for exploring HGF parameter effects on belief trajectories,
learning rates, and choice probabilities in a Jupyter notebook.
"""

from __future__ import annotations

from prl_hgf.gui.explorer import ParamExplorer

__all__ = ["ParamExplorer"]
