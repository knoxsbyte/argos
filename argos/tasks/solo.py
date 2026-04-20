"""
argos.tasks.solo — Single-robot cleaning task implementations.

Each class corresponds to a solo task defined in configs/tasks/cleaning.yaml.
Tasks send Action commands to the robot via robot.send_action() and return a
detailed TaskResult with timing and coverage metrics.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

from argos.comm.messages import Action
from argos.tasks.base import BaseTask, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_JOINTS = [0.0] * 29


def _sweep_arm_joints(phase: float) -> list[float]:
    """Return joint targets that produce a sweeping motion.

    phase ∈ [0, 2π] drives a sinusoidal left-arm sweep while keeping the
    rest of the body at neutral.
    """
    j = list(_DEFAULT_JOINTS)
    # Left shoulder (index 15): lateral oscillation ±0.6 rad
    j[15] = 0.6 * math.sin(phase)
    # Left elbow (index 17): 0.4 rad flex through sweep
    j[17] = -0.4 + 0.3 * math.sin(phase + math.pi / 4)
    # Right arm mirrors with opposite phase (holding dustpan)
    j[21] = -0.3 * math.sin(phase)
    j[23] = -0.3
    return j


def _vacuum_arm_joints(phase: float) -> list[float]:
    """Forward push/pull pattern for vacuuming."""
    j = list(_DEFAULT_JOINTS)
    j[15] = 0.2 * math.sin(phase)  # gentle side sway
    j[17] = -0.5 + 0.2 * math.sin(phase)  # elbow push-pull
    j[16] = 0.4  # shoulder forward flex
    return j


def _mop_arm_joints(phase: float) -> list[float]:
    """Bimanual mop push with slow back-and-forth."""
    j = list(_DEFAULT_JOINTS)
    # Both arms forward, alternating push-pull
    j[16] = 0.5 + 0.1 * math.sin(phase)   # left shoulder flex
    j[22] = 0.5 + 0.1 * math.sin(phase)   # right shoulder flex
    j[17] = -0.3 * math.cos(phase)         # left elbow
    j[23] = -0.3 * math.cos(phase)         # right elbow
    return j


def _wipe_horizontal_joints(phase: float) -> list[float]:
    """Back-and-forth horizontal wipe with right arm."""
    j = list(_DEFAULT_JOINTS)
    j[21] = 0.7 * math.sin(phase)   # right shoulder lateral
    j[23] = -0.4                    # elbow fixed flex
    j[22] = 0.3                     # shoulder forward
    return j


def _wipe_vertical_joints(phase: float, reach: float = 0.0) -> list[float]:
    """Vertical up-down squeegee motion."""
    j = list(_DEFAULT_JOINTS)
    j[22] = 0.5 + reach * 0.3       # shoulder forward flex (more = higher reach)
    j[23] = -0.5 + 0.4 * math.sin(phase)  # elbow drives vertical
    j[21] = 0.1                     # slight outward
    return j


def _grasp_action(open_left: float = 0.0, open_right: float = 1.0) -> Action:
    return Action(
        joint_targets=list(_DEFAULT_JOINTS),
        gripper_left=open_left,
        gripper_right=open_right,
        duration_ms=800,
    )


async def _locomotion_step(robot, dx: float, dy: float, duration_ms: int = 1000) -> None:
    """Send a locomotion step action (body translation via hip joints)."""
    j = list(_DEFAULT_JOINTS)
    # Simplified: drive hip pitch joints to simulate forward motion
    j[1] = 0.2 * (dx / max(abs(dx), 0.001))   # left hip pitch
    j[7] = 0.2 * (dx / max(abs(dx), 0.001))   # right hip pitch
    j[0] = 0.1 * (dy / max(abs(dy), 0.001))   # left hip abduction for lateral
    await robot.send_action(Action(joint_targets=j, duration_ms=duration_ms))


# ---------------------------------------------------------------------------
# Floor tasks
# ---------------------------------------------------------------------------


class SweepFloorTask(BaseTask):
    """Dry-sweep a floor zone using boustrophedon coverage pattern.

    params:
        zone_bounds: [x_min, y_min, x_max, y_max] (metres)
        start_pos:   [x, y]  (optional, default [0, 0])
        step_size:   row spacing in metres (default 0.3)
    """

    task_type = "sweep_floor"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        b = self.params.get("zone_bounds")
        if b is None:
            return False
        if len(b) != 4:
            return False
        if b[2] <= b[0] or b[3] <= b[1]:
            return False
        return True

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        start = self._begin()
        logger.info("[%s] SweepFloorTask starting on %s", self.task_id, robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Invalid zone_bounds parameter",
            ))

        bounds = self.params["zone_bounds"]
        start_pos = self.params.get("start_pos", [bounds[0], bounds[1]])
        step_size = self.params.get("step_size", 0.3)

        # Build coverage rows (boustrophedon)
        from argos.navigation.coverage import BoustrophedonPlanner
        from argos.navigation.zones import Zone

        zone = Zone(
            zone_id=f"{self.task_id}_zone",
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        )
        planner = BoustrophedonPlanner(step_size=step_size)
        waypoints = planner.plan(zone, tuple(start_pos))

        total_waypoints = len(waypoints)
        completed = 0
        phase = 0.0

        for wp in waypoints:
            if self.is_cancelled():
                logger.info("[%s] SweepFloorTask cancelled at waypoint %d/%d",
                            self.task_id, completed, total_waypoints)
                return self._finish(TaskResult(
                    success=False,
                    duration_seconds=self._elapsed(),
                    error_message="Cancelled",
                    metrics={"waypoints_completed": completed, "total_waypoints": total_waypoints},
                ))

            # Send locomotion toward waypoint
            await _locomotion_step(robot, wp.x - start_pos[0], wp.y - start_pos[1],
                                   duration_ms=800)
            start_pos = [wp.x, wp.y]

            # Sweeping arm motion (2 cycles per waypoint)
            for _ in range(2):
                phase += math.pi / 2
                joints = _sweep_arm_joints(phase)
                await robot.send_action(Action(joint_targets=joints, duration_ms=400))
                await asyncio.sleep(0.05)

            completed += 1

        coverage = completed / max(total_waypoints, 1)
        duration = self._elapsed()
        logger.info("[%s] SweepFloorTask done in %.1fs, coverage=%.1f%%",
                    self.task_id, duration, coverage * 100)
        return self._finish(TaskResult(
            success=coverage >= 0.95,
            duration_seconds=duration,
            metrics={
                "waypoints_completed": completed,
                "total_waypoints": total_waypoints,
                "coverage_fraction": round(coverage, 3),
                "zone_area_sqm": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
            },
        ))


class VacuumFloorTask(BaseTask):
    """Vacuum a floor zone using the handheld vacuum tool.

    params:
        zone_bounds: [x_min, y_min, x_max, y_max]
        tool:        "handheld_vacuum" | "upright_vacuum" (default: handheld_vacuum)
        start_pos:   [x, y] (optional)
    """

    task_type = "vacuum_floor"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        b = self.params.get("zone_bounds")
        if b is None or len(b) != 4:
            return False
        return b[2] > b[0] and b[3] > b[1]

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] VacuumFloorTask starting on %s", self.task_id, robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Invalid zone_bounds parameter",
            ))

        bounds = self.params["zone_bounds"]
        start_pos = self.params.get("start_pos", [bounds[0], bounds[1]])
        tool = self.params.get("tool", "handheld_vacuum")

        from argos.navigation.coverage import BoustrophedonPlanner
        from argos.navigation.zones import Zone

        # Vacuum needs tighter rows for better particulate pickup
        step_size = 0.25
        zone = Zone(
            zone_id=f"{self.task_id}_zone",
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        )
        planner = BoustrophedonPlanner(step_size=step_size)
        waypoints = planner.plan(zone, tuple(start_pos))

        total_waypoints = len(waypoints)
        completed = 0
        phase = 0.0
        current_pos = list(start_pos)

        for wp in waypoints:
            if self.is_cancelled():
                return self._finish(TaskResult(
                    success=False,
                    duration_seconds=self._elapsed(),
                    error_message="Cancelled",
                    metrics={"waypoints_completed": completed, "tool": tool},
                ))

            await _locomotion_step(robot, wp.x - current_pos[0],
                                   wp.y - current_pos[1], duration_ms=900)
            current_pos = [wp.x, wp.y]

            # Vacuum push-pull arm motion
            phase += math.pi / 3
            joints = _vacuum_arm_joints(phase)
            await robot.send_action(Action(joint_targets=joints, duration_ms=500))
            await asyncio.sleep(0.05)
            completed += 1

        coverage = completed / max(total_waypoints, 1)
        duration = self._elapsed()
        logger.info("[%s] VacuumFloorTask done in %.1fs coverage=%.1f%%",
                    self.task_id, duration, coverage * 100)
        return self._finish(TaskResult(
            success=coverage >= 0.95,
            duration_seconds=duration,
            metrics={
                "waypoints_completed": completed,
                "total_waypoints": total_waypoints,
                "coverage_fraction": round(coverage, 3),
                "tool": tool,
                "zone_area_sqm": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
            },
        ))


class MopFloorTask(BaseTask):
    """Wet-mop a hard floor surface with slower, overlapping strokes.

    params:
        zone_bounds:  [x_min, y_min, x_max, y_max]
        start_pos:    [x, y] (optional)
        dwell_time_s: seconds to pause after each stroke (default 0.5)
    """

    task_type = "mop_floor"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        b = self.params.get("zone_bounds")
        if b is None or len(b) != 4:
            return False
        return b[2] > b[0] and b[3] > b[1]

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] MopFloorTask starting on %s", self.task_id, robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Invalid zone_bounds parameter",
            ))

        bounds = self.params["zone_bounds"]
        start_pos = self.params.get("start_pos", [bounds[0], bounds[1]])
        dwell = self.params.get("dwell_time_s", 0.5)

        from argos.navigation.coverage import BoustrophedonPlanner
        from argos.navigation.zones import Zone

        # Wider step for mop (mop head ≈ 0.4 m)
        zone = Zone(
            zone_id=f"{self.task_id}_zone",
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        )
        planner = BoustrophedonPlanner(step_size=0.4, robot_radius=0.2)
        waypoints = planner.plan(zone, tuple(start_pos))

        total_waypoints = len(waypoints)
        completed = 0
        phase = 0.0
        current_pos = list(start_pos)

        for wp in waypoints:
            if self.is_cancelled():
                return self._finish(TaskResult(
                    success=False,
                    duration_seconds=self._elapsed(),
                    error_message="Cancelled",
                    metrics={"waypoints_completed": completed},
                ))

            # Mopping is slower — longer duration
            await _locomotion_step(robot, wp.x - current_pos[0],
                                   wp.y - current_pos[1], duration_ms=1200)
            current_pos = [wp.x, wp.y]

            # Bimanual mop stroke
            for sub_phase in range(3):
                phase += math.pi / 3
                joints = _mop_arm_joints(phase)
                await robot.send_action(Action(joint_targets=joints, duration_ms=600))
                await asyncio.sleep(0.02)

            # Dwell so wet mop lifts grime
            await asyncio.sleep(dwell)
            completed += 1

        coverage = completed / max(total_waypoints, 1)
        duration = self._elapsed()
        logger.info("[%s] MopFloorTask done in %.1fs coverage=%.1f%%",
                    self.task_id, duration, coverage * 100)
        return self._finish(TaskResult(
            success=coverage >= 0.92,
            duration_seconds=duration,
            metrics={
                "waypoints_completed": completed,
                "total_waypoints": total_waypoints,
                "coverage_fraction": round(coverage, 3),
                "zone_area_sqm": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
            },
        ))


# ---------------------------------------------------------------------------
# Surface wiping tasks
# ---------------------------------------------------------------------------


class WipeSurfaceTask(BaseTask):
    """Wipe a horizontal/vertical surface with back-and-forth arm strokes.

    params:
        surface_pos:  [x, y, z] world position of surface centre
        surface_width_m: extent to cover left-right (default 0.6)
        num_passes:   number of wipe passes (default 4)
    """

    task_type = "wipe_surface"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        pos = self.params.get("surface_pos")
        if pos is None or len(pos) != 3:
            return False
        return True

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] WipeSurfaceTask starting on %s", self.task_id, robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="surface_pos [x, y, z] required",
            ))

        surface_pos = self.params["surface_pos"]
        num_passes = self.params.get("num_passes", 4)
        width = self.params.get("surface_width_m", 0.6)

        # Navigate to surface
        from argos.navigation.coverage import BoustrophedonPlanner
        approach_waypoints = BoustrophedonPlanner().plan_to_target(
            start=(0.0, 0.0),
            target=(surface_pos[0], surface_pos[1]),
        )
        for wp in approach_waypoints:
            if self.is_cancelled():
                return self._finish(TaskResult(
                    success=False,
                    duration_seconds=self._elapsed(),
                    error_message="Cancelled during approach",
                ))
            await robot.send_action(Action(
                joint_targets=list(_DEFAULT_JOINTS),
                duration_ms=int(500 * (1.0 / max(len(approach_waypoints), 1)) + 300),
            ))

        # Spray then wipe
        # Simulate spray: open left gripper (spray bottle trigger)
        await robot.send_action(_grasp_action(open_left=1.0, open_right=0.0))
        await asyncio.sleep(0.3)
        await robot.send_action(_grasp_action(open_left=0.0, open_right=0.0))

        # Execute wipe passes
        passes_done = 0
        for i in range(num_passes):
            if self.is_cancelled():
                break
            phase = (i / num_passes) * 2 * math.pi
            joints = _wipe_horizontal_joints(phase)
            await robot.send_action(Action(joint_targets=joints, duration_ms=600))
            await asyncio.sleep(0.05)
            # Return stroke
            joints_back = _wipe_horizontal_joints(phase + math.pi)
            await robot.send_action(Action(joint_targets=joints_back, duration_ms=600))
            await asyncio.sleep(0.05)
            passes_done += 1

        coverage = passes_done / num_passes
        duration = self._elapsed()
        logger.info("[%s] WipeSurfaceTask done in %.1fs passes=%d/%d",
                    self.task_id, duration, passes_done, num_passes)
        return self._finish(TaskResult(
            success=coverage >= 0.85 and not self.is_cancelled(),
            duration_seconds=duration,
            metrics={
                "passes_completed": passes_done,
                "total_passes": num_passes,
                "surface_pos": surface_pos,
                "width_m": width,
                "coverage_fraction": round(coverage, 3),
            },
        ))


class WipeWindowTask(BaseTask):
    """Clean glass windows and mirrors with vertical squeegee strokes.

    params:
        window_pos:   [x, y] world position of window centre
        window_height_m: total window height (default 1.5)
        window_width_m:  window width (default 1.0)
        num_columns:  vertical columns to cover (default 3)
    """

    task_type = "wipe_window"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        pos = self.params.get("window_pos")
        if pos is None or len(pos) < 2:
            return False
        h = self.params.get("window_height_m", 1.5)
        return 0.1 <= h <= 2.5

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] WipeWindowTask starting on %s", self.task_id, robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="window_pos [x, y] required; height must be 0.1-2.5 m",
            ))

        window_pos = self.params["window_pos"]
        height = self.params.get("window_height_m", 1.5)
        width = self.params.get("window_width_m", 1.0)
        num_columns = self.params.get("num_columns", 3)

        # Approach window
        from argos.navigation.coverage import BoustrophedonPlanner
        approach = BoustrophedonPlanner().plan_to_target(
            start=(0.0, 0.0),
            target=(window_pos[0], window_pos[1]),
        )
        for wp in approach:
            if self.is_cancelled():
                return self._finish(TaskResult(
                    success=False,
                    duration_seconds=self._elapsed(),
                    error_message="Cancelled during approach",
                ))
            await robot.send_action(Action(
                joint_targets=list(_DEFAULT_JOINTS), duration_ms=400,
            ))

        # Spray window
        await robot.send_action(_grasp_action(open_left=1.0, open_right=0.0))
        await asyncio.sleep(0.5)
        await robot.send_action(_grasp_action(open_left=0.0, open_right=0.0))

        # Vertical squeegee strokes per column
        cols_done = 0
        for col in range(num_columns):
            if self.is_cancelled():
                break
            # Reach level: 0 = bottom of robot reach, 1 = max reach
            reach = height / 2.0  # normalised to [0, 1.25]
            for stroke in range(4):
                phase = (stroke / 4) * 2 * math.pi
                joints = _wipe_vertical_joints(phase, reach=min(reach / 1.25, 1.0))
                await robot.send_action(Action(joint_targets=joints, duration_ms=500))
                await asyncio.sleep(0.04)
            cols_done += 1

        coverage = cols_done / num_columns
        duration = self._elapsed()
        logger.info("[%s] WipeWindowTask done in %.1fs cols=%d/%d",
                    self.task_id, duration, cols_done, num_columns)
        return self._finish(TaskResult(
            success=coverage >= 0.92 and not self.is_cancelled(),
            duration_seconds=duration,
            metrics={
                "columns_completed": cols_done,
                "total_columns": num_columns,
                "window_height_m": height,
                "window_width_m": width,
                "coverage_fraction": round(coverage, 3),
            },
        ))


# ---------------------------------------------------------------------------
# Object manipulation tasks
# ---------------------------------------------------------------------------


class PickUpObjectTask(BaseTask):
    """Detect object by name, navigate to it, grasp, and move to target.

    params:
        object_name:  string identifier of the object (e.g. "cup")
        object_pos:   [x, y, z] known/detected position (optional mock)
        target_pos:   [x, y, z] where to place the object
        grasp_type:   "power" | "pinch" (default "power")
    """

    task_type = "pick_up_object"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        if not self.params.get("object_name"):
            return False
        target = self.params.get("target_pos")
        if target is None or len(target) != 3:
            return False
        return True

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] PickUpObjectTask: target=%s on %s",
                    self.task_id, self.params.get("object_name"), robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="object_name and target_pos [x,y,z] required",
            ))

        object_name = self.params["object_name"]
        object_pos = self.params.get("object_pos", [1.0, 0.0, 0.3])
        target_pos = self.params["target_pos"]

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False, duration_seconds=self._elapsed(), error_message="Cancelled",
            ))

        # Phase 1: Navigate to object
        from argos.navigation.coverage import BoustrophedonPlanner
        approach = BoustrophedonPlanner().plan_to_target(
            start=(0.0, 0.0),
            target=(object_pos[0], object_pos[1]),
        )
        for wp in approach:
            if self.is_cancelled():
                return self._finish(TaskResult(
                    success=False, duration_seconds=self._elapsed(), error_message="Cancelled",
                ))
            await robot.send_action(Action(joint_targets=list(_DEFAULT_JOINTS), duration_ms=500))

        # Phase 2: Open gripper, lower arm, close gripper (grasp)
        # Open
        await robot.send_action(_grasp_action(open_left=0.0, open_right=1.0))
        # Lower arm toward object height
        j_reach = list(_DEFAULT_JOINTS)
        j_reach[22] = 0.6   # right shoulder flex forward
        j_reach[23] = -0.8  # right elbow flex down
        await robot.send_action(Action(joint_targets=j_reach, duration_ms=800))

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False, duration_seconds=self._elapsed(), error_message="Cancelled",
            ))

        # Grasp
        await robot.send_action(_grasp_action(open_left=0.0, open_right=0.0))
        # Lift
        j_lift = list(_DEFAULT_JOINTS)
        j_lift[22] = 0.3
        j_lift[23] = -0.3
        await robot.send_action(Action(joint_targets=j_lift, duration_ms=600))

        # Phase 3: Navigate to target
        carry_path = BoustrophedonPlanner().plan_to_target(
            start=(object_pos[0], object_pos[1]),
            target=(target_pos[0], target_pos[1]),
        )
        for wp in carry_path:
            if self.is_cancelled():
                # Drop object at current position
                await robot.send_action(_grasp_action(open_left=0.0, open_right=1.0))
                return self._finish(TaskResult(
                    success=False, duration_seconds=self._elapsed(), error_message="Cancelled",
                ))
            await robot.send_action(Action(joint_targets=j_lift, duration_ms=500))

        # Phase 4: Place object
        j_place = list(_DEFAULT_JOINTS)
        j_place[22] = 0.6
        j_place[23] = -0.7
        await robot.send_action(Action(joint_targets=j_place, duration_ms=700))
        await robot.send_action(_grasp_action(open_left=0.0, open_right=1.0))
        # Retract arm
        await robot.send_action(Action(joint_targets=list(_DEFAULT_JOINTS), duration_ms=500))

        duration = self._elapsed()
        logger.info("[%s] PickUpObjectTask done in %.1fs object=%s",
                    self.task_id, duration, object_name)
        return self._finish(TaskResult(
            success=True,
            duration_seconds=duration,
            metrics={
                "object_name": object_name,
                "pick_pos": object_pos,
                "place_pos": target_pos,
                "object_relocated": 1.0,
                "object_undamaged": 1.0,
            },
        ))


class SortItemsTask(BaseTask):
    """Pick up multiple items and sort them to predefined bin locations.

    params:
        items: list of {"name": str, "pos": [x,y,z], "category": str}
        bins:  dict of category -> [x, y, z] target position
    """

    task_type = "sort_items"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        items = self.params.get("items")
        bins = self.params.get("bins")
        if not items or not bins:
            return False
        if not isinstance(items, list) or not isinstance(bins, dict):
            return False
        return len(items) > 0

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] SortItemsTask: %d items on %s",
                    self.task_id, len(self.params.get("items", [])), robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="items list and bins dict required",
            ))

        items = self.params["items"]
        bins = self.params["bins"]
        sorted_count = 0
        flagged = []
        current_pos = [0.0, 0.0]

        for item in items:
            if self.is_cancelled():
                break

            item_name = item.get("name", "unknown")
            item_pos = item.get("pos", [1.0, 0.0, 0.3])
            category = item.get("category", "misc")

            if category not in bins:
                logger.warning("[%s] Unknown category %s for item %s; flagging",
                               self.task_id, category, item_name)
                flagged.append(item_name)
                continue

            bin_pos = bins[category]

            # Reuse PickUpObjectTask logic inline
            sub_task = PickUpObjectTask(
                task_id=f"{self.task_id}_pick_{item_name}",
                params={
                    "object_name": item_name,
                    "object_pos": item_pos,
                    "target_pos": bin_pos,
                },
            )
            # Transfer cancel event
            if self.is_cancelled():
                sub_task._cancel_event.set()

            sub_result = await sub_task.execute([robot])
            if sub_result.success:
                sorted_count += 1
                current_pos = [item_pos[0], item_pos[1]]

        duration = self._elapsed()
        total_items = len(items)
        sorted_fraction = sorted_count / max(total_items, 1)

        logger.info("[%s] SortItemsTask done in %.1fs sorted=%d/%d flagged=%d",
                    self.task_id, duration, sorted_count, total_items, len(flagged))
        return self._finish(TaskResult(
            success=sorted_fraction >= 0.95 and not self.is_cancelled(),
            duration_seconds=duration,
            metrics={
                "items_sorted": sorted_count,
                "total_items": total_items,
                "items_correctly_sorted_fraction": round(sorted_fraction, 3),
                "flagged_items": flagged,
            },
        ))


class TakeOutTrashTask(BaseTask):
    """Grasp trash bag from bin, carry to disposal location.

    params:
        bin_locations: list of [x, y, z] bin positions
        disposal_pos:  [x, y] disposal point position
    """

    task_type = "take_out_trash"
    min_robots = 1
    cooperative = False

    def validate_params(self) -> bool:
        bins = self.params.get("bin_locations")
        disposal = self.params.get("disposal_pos")
        if not bins or not disposal:
            return False
        if not isinstance(bins, list) or len(disposal) < 2:
            return False
        return len(bins) > 0

    async def execute(self, robots: list) -> TaskResult:
        robot = robots[0]
        self._begin()
        logger.info("[%s] TakeOutTrashTask: %d bins on %s",
                    self.task_id, len(self.params.get("bin_locations", [])), robot.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="bin_locations list and disposal_pos required",
            ))

        bin_locations = self.params["bin_locations"]
        disposal_pos = self.params["disposal_pos"]
        bins_emptied = 0
        current_pos = [0.0, 0.0]

        from argos.navigation.coverage import BoustrophedonPlanner
        planner = BoustrophedonPlanner()

        for bin_pos in bin_locations:
            if self.is_cancelled():
                break

            # Navigate to bin
            to_bin = planner.plan_to_target(
                start=tuple(current_pos),
                target=(bin_pos[0], bin_pos[1]),
            )
            for wp in to_bin:
                if self.is_cancelled():
                    break
                await robot.send_action(Action(
                    joint_targets=list(_DEFAULT_JOINTS), duration_ms=500,
                ))

            if self.is_cancelled():
                break

            # Reach down and grasp bag (hook grip on right hand)
            j_reach = list(_DEFAULT_JOINTS)
            j_reach[22] = 0.4
            j_reach[23] = -0.9
            await robot.send_action(Action(joint_targets=j_reach, duration_ms=700))
            # Hook grip: partially close
            await robot.send_action(_grasp_action(open_left=0.0, open_right=0.3))

            # Lift bag
            j_carry = list(_DEFAULT_JOINTS)
            j_carry[22] = 0.2
            j_carry[23] = -0.2
            await robot.send_action(Action(joint_targets=j_carry, duration_ms=600))
            current_pos = [bin_pos[0], bin_pos[1]]

            # Navigate to disposal
            to_disposal = planner.plan_to_target(
                start=tuple(current_pos),
                target=tuple(disposal_pos[:2]),
            )
            for wp in to_disposal:
                if self.is_cancelled():
                    break
                await robot.send_action(Action(joint_targets=j_carry, duration_ms=600))

            # Release bag at disposal
            await robot.send_action(_grasp_action(open_left=0.0, open_right=1.0))
            await robot.send_action(Action(
                joint_targets=list(_DEFAULT_JOINTS), duration_ms=500,
            ))
            current_pos = list(disposal_pos[:2])
            bins_emptied += 1

        total_bins = len(bin_locations)
        emptied_fraction = bins_emptied / max(total_bins, 1)
        duration = self._elapsed()

        logger.info("[%s] TakeOutTrashTask done in %.1fs bins=%d/%d",
                    self.task_id, duration, bins_emptied, total_bins)
        return self._finish(TaskResult(
            success=emptied_fraction >= 1.0 and not self.is_cancelled(),
            duration_seconds=duration,
            metrics={
                "bins_emptied": bins_emptied,
                "total_bins": total_bins,
                "bins_emptied_fraction": round(emptied_fraction, 3),
            },
        ))
