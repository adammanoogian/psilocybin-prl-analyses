"""RNG seeding utilities for parallel SLURM array tasks.

Uses :class:`numpy.random.SeedSequence` to generate independent child
random number generators for each array task. This guarantees that no two
tasks share a state vector, regardless of the number of jobs or master seed.
"""

from __future__ import annotations

import numpy as np


def make_child_rng(
    master_seed: int,
    n_jobs: int,
    job_index: int,
) -> np.random.Generator:
    """Return an independent RNG for a single SLURM array task.

    Uses :class:`numpy.random.SeedSequence` to spawn ``n_jobs`` child
    sequences from ``master_seed``, then wraps ``children[job_index]`` in a
    :class:`numpy.random.Generator` via :func:`numpy.random.default_rng`.
    The SeedSequence algorithm guarantees that all child seeds have distinct
    internal state vectors, so no two tasks produce correlated streams.

    Parameters
    ----------
    master_seed : int
        Master RNG seed passed to :class:`numpy.random.SeedSequence`.
    n_jobs : int
        Total number of array tasks (must be >= 1).
    job_index : int
        Zero-based index of this task (must satisfy ``0 <= job_index < n_jobs``).

    Returns
    -------
    numpy.random.Generator
        Independent generator for this task.

    Raises
    ------
    ValueError
        If ``n_jobs < 1``.
    ValueError
        If ``job_index`` is outside ``[0, n_jobs)``.

    Examples
    --------
    >>> rng = make_child_rng(master_seed=42, n_jobs=10, job_index=0)
    >>> isinstance(rng, np.random.Generator)
    True
    """
    if n_jobs < 1:
        raise ValueError(
            f"n_jobs must be >= 1, got {n_jobs}."
        )
    if not (0 <= job_index < n_jobs):
        raise ValueError(
            f"job_index must be in [0, n_jobs), "
            f"expected 0 <= job_index < {n_jobs}, got {job_index}."
        )
    children = np.random.SeedSequence(master_seed).spawn(n_jobs)
    return np.random.default_rng(children[job_index])
