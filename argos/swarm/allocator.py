"""
argos.swarm.allocator — Auction-based Multi-Robot Task Allocation (MRTA).

Each robot submits a bid (cost) for each unassigned ready task. Tasks are
greedily assigned to the lowest bidder. Cooperative tasks requiring multiple
robots form the cheapest available team.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

from argos.comm import MockUnitreeBridge, RobotState, UnitreeBridge
from argos.swarm.dependency import TaskDAG, TaskNode

logger = logging.getLogger(__name__)

AnyBridge = Union[UnitreeBridge, MockUnitreeBridge]

# Penalty weights for the bid cost function.
# Thresholds mirror BatteryMonitor.LOW_THRESHOLD / CRITICAL_THRESHOLD so that
# auction costs naturally prefer robots with healthy batteries.
_BATTERY_LOW_THRESHOLD: float = 40.0      # matches BatteryMonitor.LOW_THRESHOLD
_BATTERY_CRITICAL_THRESHOLD: float = 15.0 # matches BatteryMonitor.CRITICAL_THRESHOLD
_BATTERY_PENALTY_SCALE: float = 50.0      # cost added per % below low threshold
_BATTERY_CRITICAL_PENALTY: float = 1000.0 # near-infinite cost — don't assign to critical robots
_LOAD_PENALTY_PER_TASK: float = 10.0      # cost penalty per already-assigned task
_TASK_LOCATION_DEFAULT: list[float] = [0.0, 0.0, 0.0]  # fallback task location


@dataclass
class RobotBid:
    """A single bid from one robot for one task.

    Lower ``cost`` is better. The allocator selects the robot(s) with the
    lowest aggregate cost.
    """

    robot_id: str
    task_id: str
    cost: float


class AuctionAllocator:
    """Greedy auction-based task allocator for a robot swarm.

    Parameters
    ----------
    robots:
        List of connected bridge instances (real or mock).
    """

    def __init__(self, robots: list[AnyBridge]) -> None:
        self.robots: list[AnyBridge] = list(robots)
        # robot_id → list of task_ids currently assigned (for load computation)
        self._current_load: dict[str, list[str]] = {
            r.robot_id: [] for r in self.robots
        }

    # ------------------------------------------------------------------
    # Primary allocation
    # ------------------------------------------------------------------

    def assign(
        self,
        dag: TaskDAG,
        robot_states: dict[str, RobotState],
    ) -> dict[str, list[str]]:
        """Compute task assignments for all currently ready tasks.

        Parameters
        ----------
        dag:
            The task dependency graph.
        robot_states:
            Latest state snapshot per robot, keyed by ``robot_id``.

        Returns
        -------
        dict[str, list[str]]
            ``{robot_id: [task_id, ...]}``. A cooperative task appears in the
            lists of *all* assigned robots.
        """
        assignments: dict[str, list[str]] = {r.robot_id: [] for r in self.robots}
        ready_tasks = dag.get_ready_tasks()

        # Only consider tasks not already assigned.
        unassigned = [t for t in ready_tasks if not t.assigned_robots]

        for task in unassigned:
            if task.cooperative or task.min_robots > 1:
                assigned_ids = self._assign_cooperative_task(
                    task, self.robots, robot_states
                )
            else:
                assigned_ids = self._assign_solo_task(task, self.robots, robot_states)

            if assigned_ids:
                task.assigned_robots = assigned_ids
                for rid in assigned_ids:
                    self._current_load.setdefault(rid, []).append(task.task_id)
                    assignments.setdefault(rid, []).append(task.task_id)
                logger.info(
                    "AuctionAllocator: task %s (%s) → robots %s.",
                    task.task_id,
                    task.task_type,
                    assigned_ids,
                )
            else:
                logger.warning(
                    "AuctionAllocator: no robots available for task %s (%s).",
                    task.task_id,
                    task.task_type,
                )

        return assignments

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        dag: TaskDAG,
        robot_states: dict[str, RobotState],
        failed_robot_id: str | None = None,
    ) -> dict[str, list[str]]:
        """Reassign tasks when a robot disconnects or fails.

        Parameters
        ----------
        dag:
            The task dependency graph.
        robot_states:
            Current states of all *remaining* robots.
        failed_robot_id:
            ID of the robot that failed. Its pending tasks are reset to
            unassigned ``pending`` status so they can be reallocated.

        Returns
        -------
        dict[str, list[str]]
            New assignments for all affected tasks.
        """
        if failed_robot_id:
            logger.warning(
                "AuctionAllocator.rebalance: robot %s failed; reassigning tasks.",
                failed_robot_id,
            )
            # Remove failed robot from pool.
            self.robots = [r for r in self.robots if r.robot_id != failed_robot_id]
            self._current_load.pop(failed_robot_id, None)

            # Reset tasks that were assigned to the failed robot.
            for task in dag.get_all_tasks():
                if failed_robot_id in task.assigned_robots and task.status in (
                    "pending",
                    "active",
                ):
                    task.assigned_robots = [
                        r for r in task.assigned_robots if r != failed_robot_id
                    ]
                    if not task.assigned_robots:
                        task.status = "pending"  # put back in pool

        # Run a fresh assignment pass.
        return self.assign(dag, robot_states)

    def release_task(self, task_id: str, robot_id: str) -> None:
        """Remove *task_id* from *robot_id*'s load tracker (called on completion)."""
        load = self._current_load.get(robot_id, [])
        if task_id in load:
            load.remove(task_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_solo_task(
        self,
        task: TaskNode,
        available: list[AnyBridge],
        robot_states: dict[str, RobotState],
    ) -> list[str]:
        """Select the single best robot for a solo task."""
        bids = self._collect_bids(task, available, robot_states)
        if not bids:
            return []
        best = min(bids, key=lambda b: b.cost)
        return [best.robot_id]

    def _assign_cooperative_task(
        self,
        task: TaskNode,
        available: list[AnyBridge],
        robot_states: dict[str, RobotState],
    ) -> list[str]:
        """Select the best robot *team* for a cooperative task.

        Returns a list of ``min_robots`` robot IDs (or fewer if not enough
        robots are available). Picks the team with the lowest total bid cost.
        """
        bids = self._collect_bids(task, available, robot_states)
        if len(bids) < task.min_robots:
            logger.warning(
                "AuctionAllocator: only %d robots available for cooperative task %s "
                "(requires %d).",
                len(bids),
                task.task_id,
                task.min_robots,
            )
            if not bids:
                return []
            # Assign as many as we have.
            bids.sort(key=lambda b: b.cost)
            return [b.robot_id for b in bids]

        # Sort by cost and take the cheapest team of size min_robots.
        bids.sort(key=lambda b: b.cost)
        team = bids[: task.min_robots]
        return [b.robot_id for b in team]

    def _collect_bids(
        self,
        task: TaskNode,
        available: list[AnyBridge],
        robot_states: dict[str, RobotState],
    ) -> list[RobotBid]:
        """Collect one bid per available robot for *task*."""
        bids: list[RobotBid] = []
        for robot in available:
            rid = robot.robot_id
            state = robot_states.get(rid)
            if state is None:
                # No state available — skip (robot may have just connected).
                logger.debug(
                    "AuctionAllocator: no state for robot %s; skipping.", rid
                )
                continue
            cost = self._compute_bid(state, task, rid)
            bids.append(RobotBid(robot_id=rid, task_id=task.task_id, cost=cost))
        return bids

    def _compute_bid(
        self,
        robot_state: RobotState,
        task: TaskNode,
        robot_id: str,
    ) -> float:
        """Compute the bid cost for a robot/task pair.

        Cost components:
        - **distance**: Euclidean distance from robot's current XY position to
          the task's target location (taken from ``task.params["location"]`` if
          present, otherwise assumed to be at the origin).
        - **battery penalty**: A flat penalty added when battery < 20%, scaled
          by how far below 20% the robot is.
        - **load penalty**: A linear penalty for each task already assigned to
          the robot, discouraging overloading.
        """
        # Distance cost — use XY only (Z is robot height, not relevant for navigation).
        robot_pos = robot_state.position[:2]
        task_location = task.params.get("location", _TASK_LOCATION_DEFAULT)
        if isinstance(task_location, (list, tuple)) and len(task_location) >= 2:
            task_xy = list(task_location[:2])
        else:
            task_xy = [0.0, 0.0]

        distance = math.sqrt(
            (robot_pos[0] - task_xy[0]) ** 2 + (robot_pos[1] - task_xy[1]) ** 2
        )

        # Battery penalty — penalise robots that might not complete the task.
        battery = robot_state.battery_percent

        # Critical robots should never be assigned new tasks.
        if battery < _BATTERY_CRITICAL_THRESHOLD:
            return _BATTERY_CRITICAL_PENALTY
        battery_penalty = (
            _BATTERY_PENALTY_SCALE * (_BATTERY_LOW_THRESHOLD - battery) / _BATTERY_LOW_THRESHOLD
            if battery < _BATTERY_LOW_THRESHOLD
            else 0.0
        )

        # Load penalty — prefer underutilised robots.
        current_tasks = len(self._current_load.get(robot_id, []))
        load_penalty = current_tasks * _LOAD_PENALTY_PER_TASK

        total_cost = distance + battery_penalty + load_penalty
        logger.debug(
            "AuctionAllocator: bid robot=%s task=%s dist=%.2f bat_pen=%.2f "
            "load_pen=%.2f total=%.2f",
            robot_id,
            task.task_id,
            distance,
            battery_penalty,
            load_penalty,
            total_cost,
        )
        return total_cost
