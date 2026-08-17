"""Tests for spatial indexing backends (H3 hex and rectangular grid)."""

from __future__ import annotations

import numpy as np
import pytest

from garland.spatial import H3HexGrid, RectangularGrid, create_spatial_grid


def _reference_rectangular_zone(grid, center_cell: int, k_min: int, population_fn):
    center_row = center_cell // grid.cols
    center_col = center_cell % grid.cols
    zone_cells = [center_cell]
    total_pop = population_fn(center_cell)
    ring = 1
    while total_pop < k_min and ring < max(grid.rows, grid.cols):
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                if abs(dr) != ring and abs(dc) != ring:
                    continue
                row, col = center_row + dr, center_col + dc
                if 0 <= row < grid.rows and 0 <= col < grid.cols:
                    cell_id = row * grid.cols + col
                    if cell_id not in zone_cells:
                        zone_cells.append(cell_id)
                        total_pop += population_fn(cell_id)
        ring += 1
    return zone_cells


def _reference_hex_zone(grid, center_cell: int, k_min: int, population_fn):
    h3_index = grid._int_to_h3[center_cell]
    zone_cells = [center_cell]
    total_pop = population_fn(center_cell)
    ring = 1
    while total_pop < k_min and ring <= 64:
        for neighbor in grid._h3.grid_ring(h3_index, ring):
            cell_id = grid._register_h3_cell(neighbor)
            if cell_id not in zone_cells:
                zone_cells.append(cell_id)
                total_pop += population_fn(cell_id)
        ring += 1
    return zone_cells


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

    def test_dilated_zone_uses_supplied_population_function(self):
        grid = RectangularGrid(width=1000.0, height=1000.0, cell_size=200.0)
        center = 2 * grid.cols + 2
        zone = grid.dilated_zone(center, k_min=25, population_fn=lambda _: 10)
        assert len(zone) == 9
        assert len(zone) < grid.rows * grid.cols

    def test_dilated_zone_is_sensitive_to_respondent_density(self):
        grid = RectangularGrid(width=2000.0, height=2000.0, cell_size=200.0)
        center = 5 * grid.cols + 5
        footprints = [
            len(
                grid.dilated_zone(
                    center,
                    k_min=100,
                    population_fn=lambda _cell, value=value: value,
                )
            )
            for value in (2, 5, 20)
        ]
        assert footprints[0] > footprints[1] > footprints[2]
        assert footprints[0] - footprints[2] >= 20


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

    def test_dilated_zone_uses_supplied_population_function(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=10)
        center = grid.cell_of(0)
        zone = grid.dilated_zone(center, k_min=25, population_fn=lambda _: 10)
        assert len(zone) >= 3

    def test_dilated_zone_is_sensitive_to_respondent_density(self):
        grid = H3HexGrid(width=2000.0, height=2000.0, resolution=9)
        _assign_cluster(grid, 1000.0, 1000.0, n=10)
        center = grid.cell_of(0)
        footprints = [
            len(
                grid.dilated_zone(
                    center,
                    k_min=100,
                    population_fn=lambda _cell, value=value: value,
                )
            )
            for value in (2, 5, 20)
        ]
        assert footprints[0] > footprints[1] > footprints[2]
        assert footprints[0] - footprints[2] >= 20

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

    @staticmethod
    def test_unknown_backend_raises():
        with pytest.raises(ValueError, match="Unknown spatial backend"):
            create_spatial_grid(backend="triangular")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["rect", "hex"])
def test_dilation_preserves_zone_and_population_metrics_against_reference(backend):
    grid = create_spatial_grid(
        backend=backend,
        width=2000.0,
        height=2000.0,
        cell_size=200.0,
    )
    _assign_cluster(grid, 1000.0, 1000.0, n=200)
    centers = [int(cell_id) for cell_id in np.unique(grid.cell_ids)[:3]]

    def population_fn(cell_id: int) -> int:
        return (cell_id * 17) % 11 + 1

    for center_cell in centers:
        for k_min in (10, 100, 1000):
            if backend == "rect":
                reference = _reference_rectangular_zone(grid, center_cell, k_min, population_fn)
            else:
                reference = _reference_hex_zone(grid, center_cell, k_min, population_fn)
            optimized = grid.dilated_zone(center_cell, k_min, population_fn)
            assert optimized == reference
            assert {
                "cell_count": len(optimized),
                "population": sum(population_fn(cell_id) for cell_id in optimized),
                "meets_k": sum(population_fn(cell_id) for cell_id in optimized) >= k_min,
            } == {
                "cell_count": len(reference),
                "population": sum(population_fn(cell_id) for cell_id in reference),
                "meets_k": sum(population_fn(cell_id) for cell_id in reference) >= k_min,
            }
