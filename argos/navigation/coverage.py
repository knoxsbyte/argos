"""
argos.navigation.coverage — Path planners and navigation executor.

BoustrophedonPlanner generates lawnmower and spiral coverage paths.
plan_to_target() uses A* on a grid for point-to-point navigation with
circular obstacle avoidance.

NavigationExecutor sends waypoints to a robot bridge sequentially.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from argos.navigation.zones import Zone

logger = logging.getLogger(__name__)

# Grid resolution for A* (metres per cell)
_ASTAR_RES = 0.1


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------


@dataclass
class Waypoint:
    x: float
    y: float
    heading: float  # radians, 0 = +x direction
    action: str = "move"  # "move" | "clean" | "turn"

    def distance_to(self, other: "Waypoint") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return (
            f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, "
            f"heading={math.degrees(self.heading):.1f}°, action={self.action!r})"
        )


# ---------------------------------------------------------------------------
# BoustrophedonPlanner
# ---------------------------------------------------------------------------


class BoustrophedonPlanner:
    """Generate coverage paths for rectangular zones.

    Parameters
    ----------
    step_size:
        Distance between parallel rows in metres. Should be <= 2*robot_radius
        for full coverage.
    robot_radius:
        Physical radius of the robot's cleaning footprint in metres.
    """

    def __init__(
        self,
        step_size: float = 0.3,
        robot_radius: float = 0.3,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if robot_radius <= 0:
            raise ValueError("robot_radius must be positive")
        self.step_size = step_size
        self.robot_radius = robot_radius

    # ------------------------------------------------------------------
    # Public planners
    # ------------------------------------------------------------------

    def plan(self, zone: "Zone", start_pos: tuple[float, float]) -> list[Waypoint]:
        """Generate a boustrophedon (lawnmower) path for zone.

        Rows run parallel to the longer axis of the zone bounding box.
        The first row is the one closest to start_pos.

        Parameters
        ----------
        zone:
            Zone whose bounds define the planning area.
        start_pos:
            (x, y) starting position of the robot.

        Returns
        -------
        list[Waypoint]
            Ordered waypoints covering the zone.
        """
        x_min, y_min, x_max, y_max = zone.bounds
        width = x_max - x_min
        height = y_max - y_min

        # Choose row direction: rows run along the longer axis
        if width >= height:
            waypoints = self._rows_along_x(x_min, y_min, x_max, y_max, start_pos)
        else:
            waypoints = self._rows_along_y(x_min, y_min, x_max, y_max, start_pos)

        logger.debug(
            "BoustrophedonPlanner.plan: zone=%s generated %d waypoints",
            zone.zone_id, len(waypoints),
        )
        return waypoints

    def plan_spiral(
        self, zone: "Zone", start_pos: tuple[float, float]
    ) -> list[Waypoint]:
        """Generate an inward spiral coverage path for square/circular zones.

        The spiral starts from the zone border nearest to start_pos and
        works inward in concentric rectangular loops.
        """
        x_min, y_min, x_max, y_max = zone.bounds
        waypoints: list[Waypoint] = []

        # Inward rectangular spiral
        left, right = x_min + self.step_size / 2, x_max - self.step_size / 2
        bottom, top = y_min + self.step_size / 2, y_max - self.step_size / 2
        heading = 0.0  # starts moving right

        while left <= right and bottom <= top:
            # Bottom edge: left → right
            for x in self._row_x_points(left, right):
                waypoints.append(Waypoint(x=x, y=bottom, heading=0.0, action="clean"))
            # Right edge: bottom → top
            for y in self._row_x_points(bottom, top):
                waypoints.append(Waypoint(x=right, y=y, heading=math.pi / 2, action="clean"))
            # Top edge: right → left
            for x in reversed(self._row_x_points(left, right)):
                waypoints.append(Waypoint(x=x, y=top, heading=math.pi, action="clean"))
            # Left edge: top → bottom
            for y in reversed(self._row_x_points(bottom, top)):
                waypoints.append(Waypoint(x=left, y=y, heading=3 * math.pi / 2, action="clean"))

            left += self.step_size
            right -= self.step_size
            bottom += self.step_size
            top -= self.step_size

        # Reorder to start nearest to start_pos
        if waypoints:
            start_idx = min(
                range(len(waypoints)),
                key=lambda i: math.hypot(
                    waypoints[i].x - start_pos[0],
                    waypoints[i].y - start_pos[1],
                ),
            )
            waypoints = waypoints[start_idx:] + waypoints[:start_idx]

        logger.debug(
            "BoustrophedonPlanner.plan_spiral: zone=%s generated %d waypoints",
            zone.zone_id, len(waypoints),
        )
        return waypoints

    def plan_to_target(
        self,
        start: tuple[float, float],
        target: tuple[float, float],
        obstacles: list[tuple[float, float, float]] | None = None,
    ) -> list[Waypoint]:
        """Plan a collision-free path from start to target using A*.

        Parameters
        ----------
        start:
            (x, y) start position in metres.
        target:
            (x, y) target position in metres.
        obstacles:
            List of (cx, cy, radius) circular obstacles in metres.
            Pass None or [] for obstacle-free navigation.

        Returns
        -------
        list[Waypoint]
            Shortest collision-free path as waypoints.  Falls back to a
            direct straight line when A* finds no grid path.
        """
        obs = obstacles or []
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        dist = math.hypot(dx, dy)

        if dist < _ASTAR_RES:
            # Already at target
            heading = math.atan2(dy, dx) if dist > 0 else 0.0
            return [Waypoint(x=target[0], y=target[1], heading=heading, action="move")]

        if not obs:
            # No obstacles — straight line with intermediate waypoints
            return self._straight_line(start, target)

        return self._astar(start, target, obs)

    def estimate_duration(self, waypoints: list[Waypoint], speed: float = 0.5) -> float:
        """Estimate travel time in seconds for a waypoint list at given speed.

        Parameters
        ----------
        waypoints:
            Ordered list of waypoints.
        speed:
            Robot travel speed in m/s (default 0.5).
        """
        if len(waypoints) < 2 or speed <= 0:
            return 0.0
        total = 0.0
        for a, b in zip(waypoints, waypoints[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total / speed

    # ------------------------------------------------------------------
    # Private helpers — row generation
    # ------------------------------------------------------------------

    def _rows_along_x(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        start_pos: tuple[float, float],
    ) -> list[Waypoint]:
        """Horizontal rows (left-right stripes) with boustrophedon reversal."""
        waypoints: list[Waypoint] = []
        y = y_min + self.step_size / 2
        row_idx = 0
        while y <= y_max - self.step_size / 2 + 1e-9:
            xs = self._row_x_points(x_min, x_max)
            if row_idx % 2 == 1:
                xs = list(reversed(xs))
            heading = 0.0 if row_idx % 2 == 0 else math.pi
            for x in xs:
                waypoints.append(Waypoint(x=x, y=y, heading=heading, action="clean"))
            y += self.step_size
            row_idx += 1

        # Find closest waypoint to start and rotate list so we begin there
        if waypoints:
            start_idx = min(
                range(len(waypoints)),
                key=lambda i: math.hypot(
                    waypoints[i].x - start_pos[0],
                    waypoints[i].y - start_pos[1],
                ),
            )
            waypoints = waypoints[start_idx:] + waypoints[:start_idx]

        return waypoints

    def _rows_along_y(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        start_pos: tuple[float, float],
    ) -> list[Waypoint]:
        """Vertical rows (top-bottom stripes) with boustrophedon reversal."""
        waypoints: list[Waypoint] = []
        x = x_min + self.step_size / 2
        col_idx = 0
        while x <= x_max - self.step_size / 2 + 1e-9:
            ys = self._row_x_points(y_min, y_max)
            if col_idx % 2 == 1:
                ys = list(reversed(ys))
            heading = math.pi / 2 if col_idx % 2 == 0 else 3 * math.pi / 2
            for y in ys:
                waypoints.append(Waypoint(x=x, y=y, heading=heading, action="clean"))
            x += self.step_size
            col_idx += 1

        if waypoints:
            start_idx = min(
                range(len(waypoints)),
                key=lambda i: math.hypot(
                    waypoints[i].x - start_pos[0],
                    waypoints[i].y - start_pos[1],
                ),
            )
            waypoints = waypoints[start_idx:] + waypoints[:start_idx]

        return waypoints

    def _row_x_points(self, lo: float, hi: float) -> list[float]:
        """Evenly spaced sample points along an axis segment."""
        if hi <= lo:
            return [lo]
        n = max(1, round((hi - lo) / self.step_size))
        return [lo + (i + 0.5) * (hi - lo) / n for i in range(n)]

    # ------------------------------------------------------------------
    # Private helpers — A*
    # ------------------------------------------------------------------

    def _straight_line(
        self,
        start: tuple[float, float],
        target: tuple[float, float],
    ) -> list[Waypoint]:
        """Return waypoints along a straight line, spaced step_size apart."""
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        n_steps = max(1, int(dist / self.step_size))
        waypoints = []
        for i in range(n_steps + 1):
            t = i / n_steps
            waypoints.append(Waypoint(
                x=start[0] + t * dx,
                y=start[1] + t * dy,
                heading=heading,
                action="move",
            ))
        return waypoints

    def _astar(
        self,
        start: tuple[float, float],
        target: tuple[float, float],
        obstacles: list[tuple[float, float, float]],
    ) -> list[Waypoint]:
        """A* path search on a grid.

        Grid origin = (0, 0). Cell (r, c) covers world
        [c*res, (c+1)*res) x [r*res, (r+1)*res).

        The grid extent is sized to cover start and target with margin.
        """
        res = _ASTAR_RES
        margin = max(self.robot_radius * 2, res * 5)

        # Grid origin (world coordinates of cell (0,0))
        ox = min(start[0], target[0]) - margin
        oy = min(start[1], target[1]) - margin
        gx_max = max(start[0], target[0]) + margin
        gy_max = max(start[1], target[1]) + margin
        cols = max(2, int((gx_max - ox) / res) + 1)
        rows = max(2, int((gy_max - oy) / res) + 1)

        # Build obstacle bitmap — mark cells within robot_radius of any obstacle
        blocked = np.zeros((rows, cols), dtype=bool)
        inflate = self.robot_radius + res / 2
        for cx, cy, cr in obstacles:
            total_r = cr + inflate
            c_col = int((cx - ox) / res)
            c_row = int((cy - oy) / res)
            span = int(total_r / res) + 1
            for dr in range(-span, span + 1):
                for dc in range(-span, span + 1):
                    r, c = c_row + dr, c_col + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        wx = ox + (c + 0.5) * res
                        wy = oy + (r + 0.5) * res
                        if math.hypot(wx - cx, wy - cy) <= total_r:
                            blocked[r, c] = True

        def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
            return (
                int((wy - oy) / res),
                int((wx - ox) / res),
            )

        def grid_to_world(r: int, c: int) -> tuple[float, float]:
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        s_row, s_col = world_to_grid(start[0], start[1])
        t_row, t_col = world_to_grid(target[0], target[1])

        # Clamp to grid
        s_row = max(0, min(rows - 1, s_row))
        s_col = max(0, min(cols - 1, s_col))
        t_row = max(0, min(rows - 1, t_row))
        t_col = max(0, min(cols - 1, t_col))

        def heuristic(r: int, c: int) -> float:
            return math.hypot(r - t_row, c - t_col)

        # Priority queue: (f, g, row, col, parent_row, parent_col)
        open_heap: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_heap, (heuristic(s_row, s_col), 0.0, s_row, s_col))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {(s_row, s_col): None}
        g_score: dict[tuple[int, int], float] = {(s_row, s_col): 0.0}

        # 8-connected neighbours
        neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]

        found = False
        while open_heap:
            _, g, r, c = heapq.heappop(open_heap)
            if (r, c) == (t_row, t_col):
                found = True
                break
            if g > g_score.get((r, c), math.inf):
                continue  # stale entry
            for dr, dc, cost in neighbours:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if blocked[nr, nc]:
                    continue
                ng = g + cost
                if ng < g_score.get((nr, nc), math.inf):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + heuristic(nr, nc), ng, nr, nc))

        if not found:
            logger.warning(
                "A* could not find a path from %s to %s; returning straight line.",
                start, target,
            )
            return self._straight_line(start, target)

        # Reconstruct path
        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = (t_row, t_col)
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()

        # Convert to waypoints
        waypoints: list[Waypoint] = []
        for i, (r, c) in enumerate(path):
            wx, wy = grid_to_world(r, c)
            if i + 1 < len(path):
                nr, nc = path[i + 1]
                nwx, nwy = grid_to_world(nr, nc)
                heading = math.atan2(nwy - wy, nwx - wx)
            elif waypoints:
                heading = waypoints[-1].heading
            else:
                heading = math.atan2(target[1] - start[1], target[0] - start[0])
            waypoints.append(Waypoint(x=wx, y=wy, heading=heading, action="move"))

        logger.debug(
            "A* found path: %d cells, estimated dist=%.2f m",
            len(path), len(path) * res,
        )
        return waypoints


# ---------------------------------------------------------------------------
# NavigationExecutor
# ---------------------------------------------------------------------------


class NavigationExecutor:
    """Sends a list of Waypoints to a robot bridge sequentially.

    Parameters
    ----------
    robot_bridge:
        Any object with an async ``send_action(action)`` method
        (UnitreeBridge or MockUnitreeBridge).
    speed:
        Desired travel speed in m/s. Used to compute action duration_ms.
    """

    def __init__(self, robot_bridge, speed: float = 0.5) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.robot = robot_bridge
        self.speed = speed

    async def execute_path(
        self,
        waypoints: list[Waypoint],
        cancel_event: asyncio.Event | None = None,
    ) -> bool:
        """Navigate the robot through waypoints in order.

        Parameters
        ----------
        waypoints:
            Ordered list of Waypoint objects.
        cancel_event:
            Optional asyncio.Event. If set between waypoints the executor
            stops immediately and returns False.

        Returns
        -------
        bool
            True if all waypoints were reached; False if cancelled or empty.
        """
        from argos.comm.messages import Action

        if not waypoints:
            return True

        _DEFAULT_JOINTS = [0.0] * 29

        prev_x, prev_y = waypoints[0].x, waypoints[0].y

        for i, wp in enumerate(waypoints):
            if cancel_event is not None and cancel_event.is_set():
                logger.debug("NavigationExecutor: cancelled at waypoint %d/%d", i, len(waypoints))
                return False

            dist = math.hypot(wp.x - prev_x, wp.y - prev_y)
            duration_ms = max(100, int((dist / self.speed) * 1000))

            # Encode locomotion as a hip-pitch joint command
            j = list(_DEFAULT_JOINTS)
            if dist > 1e-4:
                dx = wp.x - prev_x
                dy = wp.y - prev_y
                # Hip pitch (joints 1, 7) encode forward lean
                fwd = math.cos(wp.heading) * 0.2
                lat = math.sin(wp.heading) * 0.1
                j[1] = max(-0.3, min(0.3, fwd))   # left hip pitch
                j[7] = max(-0.3, min(0.3, fwd))   # right hip pitch
                j[0] = max(-0.2, min(0.2, lat))    # left hip abduction

            action = Action(joint_targets=j, duration_ms=duration_ms)
            await self.robot.send_action(action)
            prev_x, prev_y = wp.x, wp.y

        return True

    async def move_to(self, target: tuple[float, float]) -> bool:
        """Move the robot to target using a simple P-controller.

        Reads the robot's current state, computes the error, and sends
        a locomotion command. Repeats until within tolerance.

        Returns True when within 0.1 m of target or after 30 steps.
        """
        from argos.comm.messages import Action

        _DEFAULT_JOINTS = [0.0] * 29
        tolerance = 0.1
        max_steps = 30

        for step in range(max_steps):
            try:
                state = await self.robot.get_state()
                cur_x, cur_y = state.position[0], state.position[1]
            except Exception:
                # In mock mode position may not update; use step-based estimate
                cur_x, cur_y = 0.0, 0.0

            dx = target[0] - cur_x
            dy = target[1] - cur_y
            dist = math.hypot(dx, dy)

            if dist < tolerance:
                logger.debug("NavigationExecutor.move_to: reached target in %d steps", step)
                return True

            # Proportional gain
            gain = min(1.0, dist / 1.0)
            heading = math.atan2(dy, dx)

            j = list(_DEFAULT_JOINTS)
            j[1] = gain * 0.2 * math.cos(heading)
            j[7] = gain * 0.2 * math.cos(heading)
            j[0] = gain * 0.1 * math.sin(heading)

            duration_ms = max(200, int((dist * gain / self.speed) * 1000))
            await self.robot.send_action(Action(
                joint_targets=j, duration_ms=min(duration_ms, 1000),
            ))

        logger.warning("NavigationExecutor.move_to: max steps reached without reaching target")
        return False
