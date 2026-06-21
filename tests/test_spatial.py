"""Tests for spatial indexing backends (H3 hex and rectangular grid)."""

from __future__ import annotations

import numpy as np
import pytest

from garland.spatial import H3HexGrid, RectangularGrid, create_spatial_grid


def _assign_cluster(grid, center_x: float, center_y: float, n: int, spread: float = 30.0):
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(center_x, spread, n), 0, grid.width).astype(np.float32)
    y = np.clip(rng.normal(center_y, spread, n), 0, grid.height).astype(np.float32)
    grid.assign_positions(x, y)
    return x, y


class TestRectangularGrid:
    def test_cell_of_matches_row_col_layout(self):
        grid = RectangularGrid(width=1000.0, height=1000.0, cell_size=200.0)
        x = np.array([100.0, 450.0], dtype=np.float32)
        y = np.array([100.0, 650.0], dtype=np.float32)
        grid.assign_positions(x, y)
        assert grid.cell_of(0) == 0  # row 0, col 0
        assert grid.cell_of(1) == 3 * grid.cols + 2  # row 3, col 2

    def test_dilated_zone_reaches_k_min(self):
        grid = RectangularGrid(width=2000.0, height=2000.0, cell_size=200.0)
        _assign_cluster(grid, 500.0, 500.0, n=80)
        center = grid.cell_of(0)
        zone = grid.dilated_zone(center, k_min=50)
        total = sum(grid.zone_population(cid) for cid in zone)
        assert total >= 50
        assert center in zone


class TestH3HexGrid:
    def test_assign_positions_registers_cells(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=200)
        assert grid.n_cells > 0
        assert len(grid.cell_ids) == 200

    def test_cell_of_agrees_with_cell_ids_property(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=100)
        for idx in range(100):
            assert grid.cell_of(idx) == int(grid.cell_ids[idx])

    def test_dilated_zone_uses_hex_rings(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=120)
        center = grid.cell_of(0)
        zone = grid.dilated_zone(center, k_min=50)
        total = sum(grid.zone_population(cid) for cid in zone)
        assert total >= 50
        assert center in zone
        assert len(zone) >= 1

    def test_cell_center_within_domain(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=50)
        for cell_id in np.unique(grid.cell_ids):
            cx, cy = grid.cell_center(int(cell_id))
            assert 0 <= cx <= grid.width
            assert 0 <= cy <= grid.height


class TestSpatialFactory:
    def test_create_hex_backend(self):
        grid = create_spatial_grid(backend="hex", width=1000.0, height=1000.0)
        assert isinstance(grid, H3HexGrid)

    def test_create_rect_backend(self):
        grid = create_spatial_grid(backend="rect", width=1000.0, height=1000.0)
        assert isinstance(grid, RectangularGrid)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown spatial backend"):
            create_spatial_grid(backend="triangular")  # type: ignore[arg-type]
