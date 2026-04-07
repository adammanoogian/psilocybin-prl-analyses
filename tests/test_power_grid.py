"""Unit tests for :mod:`prl_hgf.power.grid` decode utilities."""

from __future__ import annotations

import pytest

from prl_hgf.power.grid import decode_task_id, total_grid_size

# Standard grids used across all tests
N_GRID: list[int] = [10, 15, 20, 25, 30, 40, 50]
D_GRID: list[float] = [0.3, 0.5, 0.7]
N_ITER: int = 200


class TestTotalGridSize:
    """Tests for :func:`total_grid_size`."""

    def test_total_grid_size(self) -> None:
        """7 sample sizes x 3 effect sizes x 200 iterations = 4200."""
        assert total_grid_size(N_GRID, D_GRID, N_ITER) == 4200


class TestDecodeTaskId:
    """Tests for :func:`decode_task_id`."""

    def test_decode_first_task(self) -> None:
        """Task 0 maps to the first cell: (N=10, d=0.3, iter=0)."""
        assert decode_task_id(0, N_GRID, D_GRID, N_ITER) == (10, 0.3, 0)

    def test_decode_last_task(self) -> None:
        """Task 4199 maps to the last cell: (N=50, d=0.7, iter=199)."""
        assert decode_task_id(4199, N_GRID, D_GRID, N_ITER) == (50, 0.7, 199)

    def test_decode_boundary_between_n_levels(self) -> None:
        """Task 600 is the first task of the N=15 block.

        N=10 block spans tasks 0..599 (3 d-levels x 200 iters = 600).
        """
        assert decode_task_id(600, N_GRID, D_GRID, N_ITER) == (15, 0.3, 0)

    def test_decode_boundary_between_d_levels(self) -> None:
        """Task 200 is the first task of d=0.5 within the N=10 block.

        Within N=10: d=0.3 occupies tasks 0..199, so d=0.5 starts at 200.
        """
        assert decode_task_id(200, N_GRID, D_GRID, N_ITER) == (10, 0.5, 0)

    def test_decode_out_of_range_raises(self) -> None:
        """Task ID equal to total grid size must raise IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            decode_task_id(4200, N_GRID, D_GRID, N_ITER)

    def test_roundtrip_all_tasks(self) -> None:
        """Every task ID maps to a unique (n, d, iter) tuple."""
        seen: set[tuple[int, float, int]] = set()
        for task_id in range(4200):
            result = decode_task_id(task_id, N_GRID, D_GRID, N_ITER)
            assert result not in seen, (
                f"Duplicate decode for task_id={task_id}: {result}"
            )
            seen.add(result)
        assert len(seen) == 4200
