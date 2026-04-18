"""
argos.swarm.coordinator — SwarmCoordinator: top-level orchestration for ARGOS.

Ties together the LLMTaskPlanner, AuctionAllocator, CooperativeCoordinator,
and the robot registry into a single async execution pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Union

from argos.comm import MockUnitreeBridge, RobotRegistry, RobotState, UnitreeBridge
from argos.swarm.allocator import AuctionAllocator
from argos.swarm.cooperative import CooperativeCoordinator
from argos.swarm.dependency import TaskDAG, TaskNode
from argos.swarm.planner import LLMTaskPlanner

logger = logging.getLogger(__name__)

AnyBridge = Union[UnitreeBridge, MockUnitreeBridge]

# How often the monitor loop polls for task progress (seconds).
_MONITOR_INTERVAL: float = 2.0
# Maximum retries for a single task before marking it permanently failed.
_MAX_TASK_RETRIES: int = 2
# How long to wait (seconds) for a stalled active task before declaring it failed.
_STALL_TIMEOUT: float = 300.0


class SwarmCoordinator:
    """Orchestrates a full goal-to-completion cleaning mission.

    Parameters
    ----------
    registry:
        The robot registry holding all connected bridges.
    planner:
        LLM-based task planner that produces a TaskDAG.
    allocator:
        Auction-based task allocator.
    cooperative:
        PEFA cooperative task coordinator.
    event_callback:
        Optional callback ``(event_name, data_dict) → None`` invoked on key
        lifecycle events (task_started, task_done, task_failed, goal_complete).
    """

    def __init__(
        self,
        registry: RobotRegistry,
        planner: LLMTaskPlanner,
        allocator: AuctionAllocator,
        cooperative: CooperativeCoordinator,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner
        self.allocator = allocator
        self.cooperative = cooperative
        self._event_callback = event_callback

        self._dag: TaskDAG | None = None
        self._active_futures: dict[str, asyncio.Task] = {}
        self._task_retries: dict[str, int] = {}
        self._monitor_task: asyncio.Task | None = None
        self._task_start_times: dict[str, float] = {}
        self._running: bool = False
        self._goal_start: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_goal(
        self,
        goal: str,
        scene_info: dict[str, Any] | None = None,
    ) -> str:
        """Execute a complete cleaning goal end-to-end.

        Parameters
        ----------
        goal:
            Natural-language goal, e.g. ``"clean the kitchen"``.
        scene_info:
            Optional perception data (rooms, surfaces, objects). If not
            provided an empty dict is used and the planner uses heuristics.

        Returns
        -------
        str
            Human-readable summary of the mission outcome.
        """
        self._goal_start = time.monotonic()
        self._running = True
        self._active_futures = {}
        self._task_retries = {}
        self._task_start_times = {}

        scene = scene_info or {}
        robots = self.registry.list_all()
        num_robots = len(robots)

        if num_robots == 0:
            logger.error("SwarmCoordinator: no robots registered; cannot execute goal.")
            return "FAILED: no robots available."

        logger.info(
            "SwarmCoordinator: goal=%r, robots=%d, scene_keys=%s.",
            goal,
            num_robots,
            list(scene.keys()),
        )

        # 1. Plan
        self._emit_event("planning_started", {"goal": goal, "num_robots": num_robots})
        dag = self.planner.decompose(goal, scene, num_robots)
        self._dag = dag
        logger.info("SwarmCoordinator: DAG built — %s.", dag)
        self._emit_event(
            "planning_complete",
            {"num_tasks": dag.graph.number_of_nodes(), "dag": dag.to_dict()},
        )

        if dag.graph.number_of_nodes() == 0:
            return "FAILED: planner produced an empty task plan."

        # Start the monitor loop.
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="swarm-monitor"
        )

        # 2. Main execution loop
        try:
            result = await self._execution_loop(dag, robots)
        except Exception as exc:
            logger.exception("SwarmCoordinator: unhandled error in execution loop: %s", exc)
            result = f"FAILED: unexpected error — {exc}"
        finally:
            self._running = False
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

        elapsed = time.monotonic() - self._goal_start
        summary = f"{result} (elapsed: {elapsed:.1f}s)"
        self._emit_event("goal_complete", {"summary": summary, "elapsed_s": elapsed})
        return summary

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the current swarm state."""
        dag = self._dag
        if dag is None:
            return {"status": "idle", "robots": [], "tasks": []}

        robots = self.registry.list_all()
        robot_summaries = []
        for r in robots:
            robot_summaries.append(
                {
                    "robot_id": r.robot_id,
                    "alive": r.is_alive(),
                    "busy": self.registry.is_busy(r.robot_id),
                }
            )

        tasks_info = []
        for task in dag.get_all_tasks():
            tasks_info.append(
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "assigned_robots": task.assigned_robots,
                    "retries": self._task_retries.get(task.task_id, 0),
                }
            )

        active_coop = self.cooperative.list_active_sessions()
        return {
            "status": "running" if self._running else "idle",
            "elapsed_s": round(time.monotonic() - self._goal_start, 1)
            if self._goal_start
            else 0.0,
            "robots": robot_summaries,
            "tasks": tasks_info,
            "active_cooperative_sessions": active_coop,
            "dag_complete": dag.is_complete(),
        }

    async def emergency_stop(self) -> None:
        """Send stop commands to all robots immediately."""
        logger.critical("SwarmCoordinator: EMERGENCY STOP triggered.")
        self._running = False

        # Cancel all active task futures.
        for task_id, fut in list(self._active_futures.items()):
            if not fut.done():
                fut.cancel()
                logger.warning("SwarmCoordinator: cancelled task future %s.", task_id)

        # Cancel monitor.
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        # Send zero-velocity action to every robot.
        from argos.comm import Action

        stop_action = Action(
            joint_targets=[0.0] * 29,
            gripper_left=0.0,
            gripper_right=0.0,
            duration_ms=100,
        )
        robots = self.registry.list_all()
        await asyncio.gather(
            *(self._safe_send_action(r, stop_action) for r in robots),
            return_exceptions=True,
        )
        logger.critical(
            "SwarmCoordinator: stop commands sent to %d robot(s).", len(robots)
        )
        self._emit_event("emergency_stop", {"robots": [r.robot_id for r in robots]})

    # ------------------------------------------------------------------
    # Internal execution pipeline
    # ------------------------------------------------------------------

    async def _execution_loop(
        self, dag: TaskDAG, robots: list[AnyBridge]
    ) -> str:
        """Main task dispatch and completion loop."""
        while not dag.is_complete() and self._running:
            # Collect current robot states.
            robot_states = await self._collect_states(robots)

            # Assign newly-ready tasks.
            assignments = self.allocator.assign(dag, robot_states)

            # Launch tasks that have been assigned but not yet started.
            for task in dag.get_ready_tasks():
                if task.task_id in self._active_futures:
                    continue  # already running
                if not task.assigned_robots:
                    continue  # no robots assigned yet

                task.status = "active"
                self._task_start_times[task.task_id] = time.monotonic()

                assigned_bridges = [
                    r for r in robots if r.robot_id in task.assigned_robots
                ]
                if not assigned_bridges:
                    logger.warning(
                        "SwarmCoordinator: task %s has assignments %s but no matching bridges.",
                        task.task_id,
                        task.assigned_robots,
                    )
                    continue

                # Mark assigned robots as busy.
                for rid in task.assigned_robots:
                    try:
                        self.registry.set_busy(rid, True)
                    except KeyError:
                        pass

                if task.cooperative or len(assigned_bridges) > 1:
                    coro = self._execute_cooperative_task(task, assigned_bridges)
                else:
                    coro = self._execute_solo_task(task, assigned_bridges[0])

                future = asyncio.create_task(
                    self._run_task_with_retry(task, coro, dag, robots, robot_states),
                    name=f"task-{task.task_id}",
                )
                self._active_futures[task.task_id] = future
                self._emit_event(
                    "task_started",
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "robots": task.assigned_robots,
                    },
                )
                logger.info(
                    "SwarmCoordinator: launched task %s (%s) on %s.",
                    task.task_id,
                    task.task_type,
                    task.assigned_robots,
                )

            # Prune completed futures.
            done_ids = [
                tid for tid, fut in self._active_futures.items() if fut.done()
            ]
            for tid in done_ids:
                self._active_futures.pop(tid, None)

            if dag.has_failed() and not self._active_futures:
                # No running tasks and something failed — cannot proceed.
                break

            await asyncio.sleep(0.5)

        # Wait for any still-running tasks.
        if self._active_futures:
            await asyncio.gather(
                *self._active_futures.values(), return_exceptions=True
            )

        if dag.is_complete():
            return "SUCCESS: all tasks completed"
        failed_tasks = [
            t.task_id for t in dag.get_all_tasks() if t.status == "failed"
        ]
        return f"PARTIAL: {len(failed_tasks)} task(s) failed: {failed_tasks}"

    async def _run_task_with_retry(
        self,
        task: TaskNode,
        initial_coro: Any,
        dag: TaskDAG,
        robots: list[AnyBridge],
        robot_states: dict[str, RobotState],
    ) -> None:
        """Wrap task execution with up to _MAX_TASK_RETRIES retry attempts."""
        success = False
        retries = 0

        # Run the initial coroutine.
        try:
            success = await initial_coro
        except Exception as exc:
            logger.error(
                "SwarmCoordinator: task %s raised exception: %s.", task.task_id, exc
            )

        while not success and retries < _MAX_TASK_RETRIES:
            retries += 1
            self._task_retries[task.task_id] = retries
            logger.warning(
                "SwarmCoordinator: task %s failed; retry %d/%d.",
                task.task_id,
                retries,
                _MAX_TASK_RETRIES,
            )
            self._emit_event(
                "task_retry",
                {"task_id": task.task_id, "attempt": retries + 1},
            )

            # Rebalance to pick best robots for retry.
            fresh_states = await self._collect_states(robots)
            new_assignments = self.allocator.rebalance(dag, fresh_states)
            new_assigned = new_assignments.get(task.task_id, task.assigned_robots)
            task.assigned_robots = new_assigned if new_assigned else task.assigned_robots

            assigned_bridges = [
                r for r in robots if r.robot_id in task.assigned_robots
            ]
            if not assigned_bridges:
                logger.error(
                    "SwarmCoordinator: no robots available for retry of task %s.",
                    task.task_id,
                )
                break

            try:
                if task.cooperative or len(assigned_bridges) > 1:
                    success = await self._execute_cooperative_task(
                        task, assigned_bridges
                    )
                else:
                    success = await self._execute_solo_task(task, assigned_bridges[0])
            except Exception as exc:
                logger.error(
                    "SwarmCoordinator: task %s retry %d raised: %s.",
                    task.task_id,
                    retries,
                    exc,
                )

        # Release busy flags.
        for rid in task.assigned_robots:
            self.allocator.release_task(task.task_id, rid)
            try:
                self.registry.set_busy(rid, False)
            except KeyError:
                pass

        if success:
            dag.mark_done(task.task_id)
            self._emit_event(
                "task_done",
                {"task_id": task.task_id, "task_type": task.task_type},
            )
            logger.info("SwarmCoordinator: task %s DONE.", task.task_id)
        else:
            dag.mark_failed(task.task_id)
            self._emit_event(
                "task_failed",
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "retries": retries,
                },
            )
            logger.error(
                "SwarmCoordinator: task %s FAILED after %d retries.",
                task.task_id,
                retries,
            )

    async def _execute_solo_task(self, task: TaskNode, robot: AnyBridge) -> bool:
        """Execute a single-robot task by invoking the robot's policy stub.

        In production, this would dispatch to the robot's policy inference
        service. Here we simulate execution time and return success.
        """
        logger.info(
            "SwarmCoordinator: executing solo task %s (%s) on %s.",
            task.task_id,
            task.task_type,
            robot.robot_id,
        )
        try:
            from argos.comm import Action

            # Build a task-appropriate action command.
            action = Action(
                joint_targets=[0.0] * 29,
                gripper_left=0.5,
                gripper_right=0.5,
                duration_ms=min(task.duration_estimate * 1000, 30_000),
            )
            await robot.send_action(action.clipped())
            # Simulate task execution duration (capped at 5s for integration).
            await asyncio.sleep(min(task.duration_estimate, 5.0))
            return True
        except Exception as exc:
            logger.error(
                "SwarmCoordinator: solo task %s on %s failed: %s.",
                task.task_id,
                robot.robot_id,
                exc,
            )
            return False

    async def _execute_cooperative_task(
        self, task: TaskNode, robots: list[AnyBridge]
    ) -> bool:
        """Execute a multi-robot task via the PEFA protocol."""
        logger.info(
            "SwarmCoordinator: executing cooperative task %s (%s) on %s.",
            task.task_id,
            task.task_type,
            [r.robot_id for r in robots],
        )
        return await self.cooperative.start_cooperative_task(task, robots)

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Background loop: detect stalled tasks and emit progress events."""
        logger.debug("SwarmCoordinator: monitor loop started.")
        while self._running:
            try:
                await asyncio.sleep(_MONITOR_INTERVAL)
                if self._dag is None:
                    continue

                now = time.monotonic()
                for task in self._dag.get_all_tasks():
                    if task.status != "active":
                        continue
                    started = self._task_start_times.get(task.task_id, now)
                    elapsed = now - started
                    if elapsed > _STALL_TIMEOUT:
                        logger.warning(
                            "SwarmCoordinator: task %s has been active for %.0fs "
                            "(stall timeout=%.0fs); marking failed.",
                            task.task_id,
                            elapsed,
                            _STALL_TIMEOUT,
                        )
                        self._emit_event(
                            "task_stalled",
                            {
                                "task_id": task.task_id,
                                "elapsed_s": elapsed,
                            },
                        )
                        # Cancel the task future so the main loop can retry/fail it.
                        fut = self._active_futures.get(task.task_id)
                        if fut and not fut.done():
                            fut.cancel()

                # Emit a periodic progress heartbeat.
                status = self.get_status()
                done_count = sum(
                    1 for t in status["tasks"] if t["status"] == "done"
                )
                total_count = len(status["tasks"])
                self._emit_event(
                    "progress",
                    {
                        "done": done_count,
                        "total": total_count,
                        "elapsed_s": status["elapsed_s"],
                    },
                )

            except asyncio.CancelledError:
                logger.debug("SwarmCoordinator: monitor loop cancelled.")
                return
            except Exception as exc:
                logger.exception(
                    "SwarmCoordinator: error in monitor loop: %s", exc
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _collect_states(
        self, robots: list[AnyBridge]
    ) -> dict[str, RobotState]:
        """Fetch the latest state from all robots concurrently."""
        async def _fetch(r: AnyBridge) -> tuple[str, RobotState | None]:
            try:
                state = await asyncio.wait_for(r.get_state(), timeout=2.0)
                return r.robot_id, state
            except Exception as exc:
                logger.warning(
                    "SwarmCoordinator: could not fetch state for %s: %s.",
                    r.robot_id,
                    exc,
                )
                return r.robot_id, None

        pairs = await asyncio.gather(*(_fetch(r) for r in robots))
        return {rid: s for rid, s in pairs if s is not None}

    async def _safe_send_action(
        self, robot: AnyBridge, action: Any
    ) -> None:
        """Send an action to a robot, swallowing all exceptions."""
        try:
            await robot.send_action(action)
        except Exception as exc:
            logger.warning(
                "SwarmCoordinator: could not send action to %s: %s.",
                robot.robot_id,
                exc,
            )

    def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Fire the event callback if registered."""
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_name, data)
        except Exception as exc:
            logger.warning(
                "SwarmCoordinator: event_callback raised for %s: %s.", event_name, exc
            )
