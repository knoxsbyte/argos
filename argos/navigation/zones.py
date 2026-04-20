"""
argos.navigation.zones — Room zone management for multi-robot coverage.

ZoneManager partitions a room into zones (one per robot) and tracks per-zone
cleaning progress on a binary occupancy grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Grid resolution: each cell represents this many metres on a side.
_GRID_RES = 0.1  # 10 cm cells


@dataclass
class Zone:
    """A rectangular sub-region of the room assigned to one robot.

    bounds = (x_min, y_min, x_max, y_max) in metres.
    coverage_pct is a fraction [0, 1] of cells marked cleaned.
    """

    zone_id: str
    bounds: tuple[float, float, float, float]
    assigned_robot: str | None = None
    coverage_pct: float = 0.0
    priority: int = 1

    # Internal occupancy grid: True = cleaned.  Populated lazily.
    _grid: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        x_min, y_min, x_max, y_max = self.bounds
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Zone {self.zone_id}: invalid bounds {self.bounds}")
        cols = max(1, round((x_max - x_min) / _GRID_RES))
        rows = max(1, round((y_max - y_min) / _GRID_RES))
        self._grid = np.zeros((rows, cols), dtype=bool)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x, y) to (row, col) grid indices (clamped)."""
        x_min, y_min, x_max, y_max = self.bounds
        col = int((x - x_min) / _GRID_RES)
        row = int((y - y_min) / _GRID_RES)
        rows, cols = self._grid.shape
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        return row, col

    def _to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert (row, col) grid indices to world centre coordinates."""
        x_min, y_min, _, _ = self.bounds
        x = x_min + (col + 0.5) * _GRID_RES
        y = y_min + (row + 0.5) * _GRID_RES
        return x, y

    def mark_cleaned(self, x: float, y: float, radius: float = 0.3) -> None:
        """Mark all cells within radius of (x, y) as cleaned."""
        r_cells = max(1, int(radius / _GRID_RES))
        row_c, col_c = self._to_grid(x, y)
        rows, cols = self._grid.shape
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    r = row_c + dr
                    c = col_c + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        self._grid[r, c] = True
        total = self._grid.size
        cleaned = int(self._grid.sum())
        self.coverage_pct = cleaned / total if total > 0 else 0.0

    def nearest_uncleaned(
        self, current_x: float, current_y: float
    ) -> tuple[float, float] | None:
        """Return world coordinates of the nearest uncleaned cell centre."""
        uncleaned = np.argwhere(~self._grid)
        if uncleaned.size == 0:
            return None
        cur_row, cur_col = self._to_grid(current_x, current_y)
        dists = np.hypot(
            uncleaned[:, 0] - cur_row,
            uncleaned[:, 1] - cur_col,
        )
        idx = int(np.argmin(dists))
        return self._to_world(int(uncleaned[idx, 0]), int(uncleaned[idx, 1]))

    def is_complete(self, threshold: float = 0.95) -> bool:
        return self.coverage_pct >= threshold

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    @property
    def area(self) -> float:
        return self.width * self.height


class ZoneManager:
    """Manages zone partitioning and coverage tracking for one room.

    Parameters
    ----------
    room_bounds:
        (x_min, y_min, x_max, y_max) extent of the room in metres.
    """

    def __init__(self, room_bounds: tuple[float, float, float, float]) -> None:
        x_min, y_min, x_max, y_max = room_bounds
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Invalid room_bounds: {room_bounds}")
        self.room_bounds = room_bounds
        self.zones: dict[str, Zone] = {}

    # ------------------------------------------------------------------
    # Partitioning strategies
    # ------------------------------------------------------------------

    def partition(
        self,
        num_robots: int,
        strategy: str = "strips",
        robot_positions: list[tuple[float, float]] | None = None,
    ) -> list[Zone]:
        """Divide the room into zones for num_robots.

        Parameters
        ----------
        num_robots:
            Number of zones to create (one per robot).
        strategy:
            "strips"   — vertical strips of equal width.
            "quadrant" — NxN grid (nearest square to num_robots).
            "voronoi"  — Voronoi cells from robot_positions (falls back to
                         strips when positions are not provided).

        Returns
        -------
        list[Zone]
            Created zones, also stored in self.zones.
        """
        if num_robots < 1:
            raise ValueError("num_robots must be >= 1")

        if strategy == "strips":
            zones = self._partition_strips(num_robots)
        elif strategy == "quadrant":
            zones = self._partition_quadrant(num_robots)
        elif strategy == "voronoi":
            if robot_positions and len(robot_positions) == num_robots:
                zones = self._partition_voronoi(num_robots, robot_positions)
            else:
                logger.warning(
                    "Voronoi partitioning requested but robot_positions not provided "
                    "or length mismatch — falling back to strips."
                )
                zones = self._partition_strips(num_robots)
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose: strips, quadrant, voronoi"
            )

        for zone in zones:
            self.zones[zone.zone_id] = zone
            logger.debug(
                "Zone %s created: bounds=%s area=%.2f m²",
                zone.zone_id, zone.bounds, zone.area,
            )

        logger.info(
            "Partitioned room %s into %d zones (strategy=%s)",
            self.room_bounds, len(zones), strategy,
        )
        return zones

    def _partition_strips(self, num_robots: int) -> list[Zone]:
        """Divide room into vertical strips of equal width."""
        x_min, y_min, x_max, y_max = self.room_bounds
        total_width = x_max - x_min
        strip_width = total_width / num_robots
        zones = []
        for i in range(num_robots):
            zx_min = x_min + i * strip_width
            zx_max = x_min + (i + 1) * strip_width
            zones.append(Zone(
                zone_id=f"zone_{i}",
                bounds=(zx_min, y_min, zx_max, y_max),
                priority=i + 1,
            ))
        return zones

    def _partition_quadrant(self, num_robots: int) -> list[Zone]:
        """Divide room into an NxM grid with N*M >= num_robots."""
        x_min, y_min, x_max, y_max = self.room_bounds
        # Choose grid dimensions whose product >= num_robots, as square as possible
        cols = math.ceil(math.sqrt(num_robots))
        rows = math.ceil(num_robots / cols)
        cell_w = (x_max - x_min) / cols
        cell_h = (y_max - y_min) / rows
        zones = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= num_robots:
                    break
                zones.append(Zone(
                    zone_id=f"zone_{idx}",
                    bounds=(
                        x_min + c * cell_w,
                        y_min + r * cell_h,
                        x_min + (c + 1) * cell_w,
                        y_min + (r + 1) * cell_h,
                    ),
                    priority=idx + 1,
                ))
        return zones

    def _partition_voronoi(
        self,
        num_robots: int,
        robot_positions: list[tuple[float, float]],
    ) -> list[Zone]:
        """Assign each grid cell to its nearest robot; return bounding boxes.

        This is an approximate Voronoi that computes bounding boxes of each
        robot's Voronoi region rather than exact polygons, which is sufficient
        for rectangular room coverage.
        """
        x_min, y_min, x_max, y_max = self.room_bounds
        res = _GRID_RES
        xs = np.arange(x_min + res / 2, x_max, res)
        ys = np.arange(y_min + res / 2, y_max, res)
        xx, yy = np.meshgrid(xs, ys)  # shape (rows, cols)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

        seeds = np.array(robot_positions)  # (num_robots, 2)
        # For each point find nearest seed
        dists = np.linalg.norm(
            points[:, None, :] - seeds[None, :, :], axis=2
        )  # (N, num_robots)
        assignments = np.argmin(dists, axis=1)  # (N,)

        zones = []
        for i in range(num_robots):
            mask = assignments == i
            if not mask.any():
                # Edge case: no cells assigned; create tiny zone at seed
                sx, sy = robot_positions[i]
                zones.append(Zone(
                    zone_id=f"zone_{i}",
                    bounds=(sx - res, sy - res, sx + res, sy + res),
                    priority=i + 1,
                ))
                continue
            region = points[mask]
            bx_min = float(region[:, 0].min()) - res / 2
            by_min = float(region[:, 1].min()) - res / 2
            bx_max = float(region[:, 0].max()) + res / 2
            by_max = float(region[:, 1].max()) + res / 2
            # Clamp to room bounds
            bx_min = max(bx_min, x_min)
            by_min = max(by_min, y_min)
            bx_max = min(bx_max, x_max)
            by_max = min(by_max, y_max)
            zones.append(Zone(
                zone_id=f"zone_{i}",
                bounds=(bx_min, by_min, bx_max, by_max),
                priority=i + 1,
            ))
        return zones

    # ------------------------------------------------------------------
    # Assignment and tracking
    # ------------------------------------------------------------------

    def assign_robot(self, zone_id: str, robot_id: str) -> None:
        """Assign a robot to a zone."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone {zone_id!r} not found")
        self.zones[zone_id].assigned_robot = robot_id
        logger.debug("Robot %s assigned to zone %s", robot_id, zone_id)

    def get_robot_zone(self, robot_id: str) -> Zone | None:
        """Return the zone assigned to robot_id, or None."""
        for zone in self.zones.values():
            if zone.assigned_robot == robot_id:
                return zone
        return None

    def update_coverage(
        self,
        zone_id: str,
        robot_position: tuple[float, float],
        radius: float = 0.3,
    ) -> None:
        """Mark area around robot_position as cleaned in zone_id."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone {zone_id!r} not found")
        self.zones[zone_id].mark_cleaned(
            robot_position[0], robot_position[1], radius=radius
        )

    def get_next_uncleaned(
        self,
        zone_id: str,
        current_pos: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return the nearest uncleaned point in zone_id from current_pos."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone {zone_id!r} not found")
        return self.zones[zone_id].nearest_uncleaned(current_pos[0], current_pos[1])

    def is_zone_complete(self, zone_id: str, threshold: float = 0.95) -> bool:
        """Return True when zone coverage meets or exceeds threshold."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone {zone_id!r} not found")
        return self.zones[zone_id].is_complete(threshold)

    def overall_coverage(self) -> float:
        """Weighted-average coverage across all zones (by area)."""
        if not self.zones:
            return 0.0
        total_area = sum(z.area for z in self.zones.values())
        if total_area == 0.0:
            return 0.0
        return sum(
            z.coverage_pct * z.area for z in self.zones.values()
        ) / total_area

    def unassigned_zones(self) -> list[Zone]:
        """Return zones that have not yet been assigned to a robot."""
        return [z for z in self.zones.values() if z.assigned_robot is None]

    def summary(self) -> dict:
        """Return a serialisable summary of all zones."""
        return {
            "room_bounds": self.room_bounds,
            "overall_coverage": round(self.overall_coverage(), 4),
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "bounds": z.bounds,
                    "assigned_robot": z.assigned_robot,
                    "coverage_pct": round(z.coverage_pct, 4),
                    "priority": z.priority,
                    "area_sqm": round(z.area, 3),
                }
                for z in self.zones.values()
            ],
        }

    def __repr__(self) -> str:
        return (
            f"<ZoneManager room={self.room_bounds} "
            f"zones={len(self.zones)} "
            f"coverage={self.overall_coverage():.1%}>"
        )
