"""Hierarchical spatial indexing for the GARLAND testbed.

Implements a coordinate grid that can scale to H3-style hexagonal indexing.
Provides efficient nearest-neighbor queries and zone-based aggregation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpatialGrid:
    """Cell-based spatial index supporting zone queries and density estimation.

    Parameters
    ----------
    width : float
        Spatial domain width in meters.
    height : float
        Spatial domain height in meters.
    cell_size : float
        Grid cell edge length in meters (default 200m ≈ city block).
    """

    def __init__(self, width: float = 10_000.0, height: float = 10_000.0, cell_size: float = 200.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))
        self.n_cells = self.cols * self.rows
        # Agent positions stored as flat arrays for vectorized ops
        self._x: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._y: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._cell_ids: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        # Reverse map: cell_id -> list of agent indices
        self._cell_agents: dict[int, list[int]] = {}

    def assign_positions(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        """Bulk-assign agent positions and recompute cell memberships."""
        self._x = x
        self._y = y
        col = np.clip((x / self.cell_size).astype(np.int32), 0, self.cols - 1)
        row = np.clip((y / self.cell_size).astype(np.int32), 0, self.rows - 1)
        self._cell_ids = row * self.cols + col
        # Rebuild reverse map (sorted indices → consecutive cell runs)
        self._cell_agents = {}
        if len(self._cell_ids) == 0:
            return
        order = np.argsort(self._cell_ids, kind="stable")
        sorted_cells = self._cell_ids[order]
        boundaries = np.concatenate(
            ([0], np.where(np.diff(sorted_cells) != 0)[0] + 1, [len(sorted_cells)])
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            cell_id = int(sorted_cells[start])
            self._cell_agents[cell_id] = order[start:end].tolist()

    def cell_of(self, agent_idx: int) -> int:
        """Return the cell_id for an agent by index."""
        return int(self._cell_ids[agent_idx])

    def agents_in_cell(self, cell_id: int) -> list[int]:
        """Return indices of agents in a given cell."""
        return self._cell_agents.get(cell_id, [])

    def agents_in_radius(self, x: float, y: float, radius: float) -> NDArray[np.intp]:
        """Return agent indices within Euclidean radius of (x, y)."""
        dx = self._x - x
        dy = self._y - y
        dist_sq = dx * dx + dy * dy
        return np.where(dist_sq <= radius * radius)[0]

    def zone_population(self, cell_id: int) -> int:
        """Population count within a single cell."""
        return len(self._cell_agents.get(cell_id, []))

    def dilated_zone(self, center_cell: int, k_min: int) -> list[int]:
        """Expand zone outward from center_cell until population >= k_min.

        Returns list of cell_ids forming the dilated zone.
        """
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
                    r, c = center_row + dr, center_col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        cid = r * self.cols + c
                        if cid not in zone_cells:
                            zone_cells.append(cid)
                            total_pop += self.zone_population(cid)
            ring += 1
        return zone_cells

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        """Return (x, y) center coordinates of a cell."""
        row = cell_id // self.cols
        col = cell_id % self.cols
        cx = (col + 0.5) * self.cell_size
        cy = (row + 0.5) * self.cell_size
        return (cx, cy)

    def density_at_cell(self, cell_id: int) -> float:
        """Population density (agents per km²) at a cell."""
        area_km2 = (self.cell_size / 1000.0) ** 2
        return self.zone_population(cell_id) / area_km2 if area_km2 > 0 else 0.0
