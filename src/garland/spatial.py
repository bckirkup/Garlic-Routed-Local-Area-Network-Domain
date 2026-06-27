"""Hierarchical spatial indexing for the GARLAND testbed.

Supports rectangular grid cells and H3 hexagonal indexing with a shared
public API (``cell_of``, ``dilated_zone``, ``agents_in_cell``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SpatialBackend = Literal["hex", "rect"]

METERS_PER_DEG_LAT = 111_320.0


def meters_to_latlng(
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
    origin_lat: float,
    origin_lng: float,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Convert local meter coordinates to latitude/longitude."""
    lat = origin_lat + y / METERS_PER_DEG_LAT
    lng_scale = METERS_PER_DEG_LAT * np.cos(np.radians(origin_lat))
    lng = origin_lng + x / lng_scale
    return lat, lng


def latlng_to_meters(
    lat: float,
    lng: float,
    origin_lat: float,
    origin_lng: float,
) -> tuple[float, float]:
    """Convert latitude/longitude to local meter coordinates."""
    y = (lat - origin_lat) * METERS_PER_DEG_LAT
    lng_scale = METERS_PER_DEG_LAT * np.cos(np.radians(origin_lat))
    x = (lng - origin_lng) * lng_scale
    return x, y


class SpatialIndex(ABC):
    """Shared spatial index interface for privacy protocol zone queries."""

    width: float
    height: float
    cell_size: float
    n_cells: int

    @abstractmethod
    def assign_positions(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        """Bulk-assign agent positions and recompute cell memberships."""

    @abstractmethod
    def cell_of(self, agent_idx: int) -> int:
        """Return the cell_id for an agent by index."""

    @property
    @abstractmethod
    def cell_ids(self) -> NDArray[np.int32]:
        """Per-agent cell IDs (same order as positions passed to assign_positions)."""

    @abstractmethod
    def agents_in_cell(self, cell_id: int) -> list[int]:
        """Return indices of agents in a given cell."""

    @abstractmethod
    def agents_in_radius(self, x: float, y: float, radius: float) -> NDArray[np.intp]:
        """Return agent indices within Euclidean radius of (x, y)."""

    @abstractmethod
    def zone_population(self, cell_id: int) -> int:
        """Population count within a single cell."""

    @abstractmethod
    def dilated_zone(self, center_cell: int, k_min: int) -> list[int]:
        """Expand zone outward from center_cell until population >= k_min."""

    @abstractmethod
    def cell_center(self, cell_id: int) -> tuple[float, float]:
        """Return (x, y) center coordinates of a cell."""

    @abstractmethod
    def density_at_cell(self, cell_id: int) -> float:
        """Population density (agents per km²) at a cell."""


class RectangularGrid(SpatialIndex):
    """Rectangular cell-based spatial index."""

    def __init__(self, width: float = 10_000.0, height: float = 10_000.0, cell_size: float = 200.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))
        self.n_cells = self.cols * self.rows
        self._x: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._y: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._cell_ids: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._cell_agents: dict[int, list[int]] = {}

    def assign_positions(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        self._x = x
        self._y = y
        col = np.clip((x / self.cell_size).astype(np.int32), 0, self.cols - 1)
        row = np.clip((y / self.cell_size).astype(np.int32), 0, self.rows - 1)
        self._cell_ids = row * self.cols + col
        self._rebuild_cell_agents()

    def _rebuild_cell_agents(self) -> None:
        self._cell_agents = {}
        if len(self._cell_ids) == 0:
            return
        order = np.argsort(self._cell_ids, kind="stable")
        sorted_cells = self._cell_ids[order]
        boundaries = np.concatenate(
            ([0], np.nonzero(np.diff(sorted_cells))[0] + 1, [len(sorted_cells)])
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            cell_id = int(sorted_cells[start])
            self._cell_agents[cell_id] = order[start:end].tolist()

    def cell_of(self, agent_idx: int) -> int:
        return int(self._cell_ids[agent_idx])

    @property
    def cell_ids(self) -> NDArray[np.int32]:
        return self._cell_ids

    def agents_in_cell(self, cell_id: int) -> list[int]:
        return self._cell_agents.get(cell_id, [])

    def agents_in_radius(self, x: float, y: float, radius: float) -> NDArray[np.intp]:
        dx = self._x - x
        dy = self._y - y
        dist_sq = dx * dx + dy * dy
        return np.nonzero(dist_sq <= radius * radius)[0]

    def zone_population(self, cell_id: int) -> int:
        return len(self._cell_agents.get(cell_id, []))

    def dilated_zone(self, center_cell: int, k_min: int) -> list[int]:
        center_row = center_cell // self.cols
        center_col = center_cell % self.cols
        zone_cells = [center_cell]
        total_pop = self.zone_population(center_cell)
        ring = 1
        while total_pop < k_min and ring < max(self.rows, self.cols):
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if abs(dr) != ring and abs(dc) != ring:
                        continue
                    row, col = center_row + dr, center_col + dc
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        cid = row * self.cols + col
                        if cid not in zone_cells:
                            zone_cells.append(cid)
                            total_pop += self.zone_population(cid)
            ring += 1
        return zone_cells

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        row = cell_id // self.cols
        col = cell_id % self.cols
        cx = (col + 0.5) * self.cell_size
        cy = (row + 0.5) * self.cell_size
        return (cx, cy)

    def density_at_cell(self, cell_id: int) -> float:
        area_km2 = (self.cell_size / 1000.0) ** 2
        return self.zone_population(cell_id) / area_km2 if area_km2 > 0 else 0.0


class H3HexGrid(SpatialIndex):
    """H3 hexagonal spatial index with integer cell IDs mapped from H3 indices."""

    def __init__(
        self,
        width: float = 10_000.0,
        height: float = 10_000.0,
        cell_size: float = 200.0,
        resolution: int = 9,
        origin_lat: float = 40.0,
        origin_lng: float = -74.0,
    ):
        import h3

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.resolution = resolution
        self.origin_lat = origin_lat
        self.origin_lng = origin_lng
        self._h3 = h3
        self.n_cells = 0
        self._x: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._y: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._cell_ids: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._cell_agents: dict[int, list[int]] = {}
        self._h3_to_int: dict[str, int] = {}
        self._int_to_h3: dict[int, str] = {}
        self._next_cell_id = 0
        self._hex_area_km2 = h3.average_hexagon_area(resolution, unit="km^2")

    def _register_h3_cell(self, h3_index: str) -> int:
        cell_id = self._h3_to_int.get(h3_index)
        if cell_id is None:
            cell_id = self._next_cell_id
            self._next_cell_id += 1
            self._h3_to_int[h3_index] = cell_id
            self._int_to_h3[cell_id] = h3_index
            self.n_cells = self._next_cell_id
        return cell_id

    def _xy_to_h3(self, x: float, y: float) -> str:
        lat, lng = meters_to_latlng(x, y, self.origin_lat, self.origin_lng)
        return self._h3.latlng_to_cell(float(lat), float(lng), self.resolution)

    def assign_positions(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        self._x = x
        self._y = y
        cell_ids = np.empty(len(x), dtype=np.int32)
        for idx, (px, py) in enumerate(zip(x, y)):
            h3_index = self._xy_to_h3(float(px), float(py))
            cell_ids[idx] = self._register_h3_cell(h3_index)
        self._cell_ids = cell_ids
        self._rebuild_cell_agents()

    def _rebuild_cell_agents(self) -> None:
        self._cell_agents = {}
        if len(self._cell_ids) == 0:
            return
        order = np.argsort(self._cell_ids, kind="stable")
        sorted_cells = self._cell_ids[order]
        boundaries = np.concatenate(
            ([0], np.nonzero(np.diff(sorted_cells))[0] + 1, [len(sorted_cells)])
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            cell_id = int(sorted_cells[start])
            self._cell_agents[cell_id] = order[start:end].tolist()

    def cell_of(self, agent_idx: int) -> int:
        return int(self._cell_ids[agent_idx])

    @property
    def cell_ids(self) -> NDArray[np.int32]:
        return self._cell_ids

    def agents_in_cell(self, cell_id: int) -> list[int]:
        return self._cell_agents.get(cell_id, [])

    def agents_in_radius(self, x: float, y: float, radius: float) -> NDArray[np.intp]:
        dx = self._x - x
        dy = self._y - y
        dist_sq = dx * dx + dy * dy
        return np.nonzero(dist_sq <= radius * radius)[0]

    def zone_population(self, cell_id: int) -> int:
        return len(self._cell_agents.get(cell_id, []))

    def dilated_zone(self, center_cell: int, k_min: int) -> list[int]:
        h3_index = self._int_to_h3[center_cell]
        zone_cells = [center_cell]
        total_pop = self.zone_population(center_cell)
        ring = 1
        max_rings = 64
        while total_pop < k_min and ring <= max_rings:
            for neighbor in self._h3.grid_ring(h3_index, ring):
                cid = self._register_h3_cell(neighbor)
                if cid not in zone_cells:
                    zone_cells.append(cid)
                    total_pop += self.zone_population(cid)
            ring += 1
        return zone_cells

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        h3_index = self._int_to_h3[cell_id]
        lat, lng = self._h3.cell_to_latlng(h3_index)
        return latlng_to_meters(lat, lng, self.origin_lat, self.origin_lng)

    def density_at_cell(self, cell_id: int) -> float:
        return self.zone_population(cell_id) / self._hex_area_km2 if self._hex_area_km2 > 0 else 0.0


def create_spatial_grid(
    width: float = 10_000.0,
    height: float = 10_000.0,
    cell_size: float = 200.0,
    backend: SpatialBackend = "hex",
    h3_resolution: int = 9,
    origin_lat: float = 40.0,
    origin_lng: float = -74.0,
) -> SpatialIndex:
    """Create a spatial index for the requested backend."""
    if backend == "rect":
        return RectangularGrid(width=width, height=height, cell_size=cell_size)
    if backend == "hex":
        return H3HexGrid(
            width=width,
            height=height,
            cell_size=cell_size,
            resolution=h3_resolution,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
        )
    raise ValueError(f"Unknown spatial backend {backend!r}; expected 'hex' or 'rect'")


# Backward-compatible alias used in tests and older call sites.
SpatialGrid = RectangularGrid
