"""
argos.swarm.cooperative — PEFA (Propose-Execute-Feedback-Adjust) protocol.

Manages multi-robot cooperative tasks where two or more robots must act in
synchrony. Each PEFASession drives a single cooperative task from proposal
through execution and adjusts on partial failure (up to 3 attempts).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Union

from argos.comm import Action, CoopMessage, CoopPhase, MockUnitreeBridge, UnitreeBridge
from argos.swarm.dependency import TaskNode

logger = logging.getLogger(__name__)

AnyBridge = Union[UnitreeBridge, MockUnitreeBridge]

_CONFIRM_TIMEOUT: float = 5.0   # seconds to wait for all robots to confirm
_EXECUTE_TIMEOUT: float = 30.0  # seconds allowed for the execution phase
_MAX_ATTEMPTS: int = 3


class PEFASession:
    """Manages a single cooperative task through the PEFA protocol loop.

    Parameters
    ----------
    task:
        The cooperative TaskNode to execute.
    robot_bridges:
        Ordered list of bridges; ``robots[0]`` is the lead robot.
    session_id:
        Unique identifier for this cooperation episode. Auto-generated if empty.
    """

    def __init__(
        self,
        task: TaskNode,
        robot_bridges: list[AnyBridge],
        session_id: str = "",
    ) -> None:
        self.task = task
        self.robots: list[AnyBridge] = list(robot_bridges)
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.phase: str = "propose"
        self.confirmations: dict[str, bool] = {}
        self._attempt: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> bool:
        """Execute the full PEFA loop.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if all attempts fail.
        """
        self._start_time = time.monotonic()
        logger.info(
            "PEFASession %s: starting task %s (%s) with %d robots.",
            self.session_id,
            self.task.task_id,
            self.task.task_type,
            len(self.robots),
        )

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._attempt = attempt
            logger.info(
                "PEFASession %s: attempt %d/%d.", self.session_id, attempt, _MAX_ATTEMPTS
            )

            # PROPOSE
            self.phase = "propose"
            plan = await self._propose()
            logger.debug("PEFASession %s: plan proposed — %s.", self.session_id, plan)

            # CONFIRM
            self.phase = "confirm"
            confirmed = await self._confirm(plan)
            if not confirmed:
                logger.warning(
                    "PEFASession %s: confirmation failed on attempt %d.",
                    self.session_id,
                    attempt,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(0.5)
                continue

            # EXECUTE
            self.phase = "execute"
            results = await self._execute(plan)
            logger.debug(
                "PEFASession %s: execution results — %s.", self.session_id, results
            )

            # FEEDBACK / ADJUST
            self.phase = "feedback"
            success = await self._feedback_adjust(results, plan, attempt)
            if success:
                self.phase = "complete"
                elapsed = time.monotonic() - self._start_time
                logger.info(
                    "PEFASession %s: task %s completed in %.1fs after %d attempt(s).",
                    self.session_id,
                    self.task.task_id,
                    elapsed,
                    attempt,
                )
                return True

        logger.error(
            "PEFASession %s: task %s failed after %d attempts.",
            self.session_id,
            self.task.task_id,
            _MAX_ATTEMPTS,
        )
        return False

    # ------------------------------------------------------------------
    # PEFA phases
    # ------------------------------------------------------------------

    async def _propose(self) -> dict:
        """Lead robot computes the action plan for the cooperative task.

        Returns a plan dict with:
        - ``actions``: per-robot action specifications
        - ``sync_point``: shared timing anchor (monotonic seconds)
        - ``task_type``: echoed from the task node
        - ``params``: echoed task params
        """
        lead = self.robots[0]
        try:
            state = await lead.get_state()
        except Exception as exc:
            logger.warning(
                "PEFASession %s: could not fetch lead state — %s; using defaults.",
                self.session_id, exc
            )
            from argos.comm import RobotState
            state = RobotState()

        # Build per-robot action stubs. The lead robot computes positions
        # relative to its current pose; follower robots mirror symmetrically.
        actions: dict[str, dict] = {}
        for i, robot in enumerate(self.robots):
            # Generate a safe default action (home pose with slight variation).
            joint_targets = [0.0] * 29
            # Apply minor offset for non-lead robots to avoid collision.
            if i > 0:
                joint_targets[12] = 0.1 * i   # waist yaw offset
            actions[robot.robot_id] = {
                "joint_targets": joint_targets,
                "gripper_left": 0.0,
                "gripper_right": 0.0,
                "duration_ms": self.task.duration_estimate * 1000 // max(1, _MAX_ATTEMPTS),
                "role": "lead" if i == 0 else f"follower_{i}",
            }

        plan = {
            "session_id": self.session_id,
            "task_type": self.task.task_type,
            "params": self.task.params,
            "actions": actions,
            "sync_point": time.monotonic(),
            "lead_robot": lead.robot_id,
            "attempt": self._attempt,
        }

        # Emit a PROPOSE CoopMessage (best-effort; bridge may not support pub/sub).
        await self._broadcast_coop_message(
            phase=CoopPhase.PROPOSE,
            sender_id=lead.robot_id,
            payload={"plan_summary": {"task_type": self.task.task_type}},
        )

        return plan

    async def _confirm(self, plan: dict) -> bool:
        """Broadcast plan to all robots and wait for readiness confirmation.

        Returns ``True`` if all robots confirm within ``_CONFIRM_TIMEOUT`` seconds.
        """
        self.confirmations = {}
        lead_id = plan.get("lead_robot", self.robots[0].robot_id)

        async def _check_robot_ready(robot: AnyBridge) -> tuple[str, bool]:
            rid = robot.robot_id
            try:
                # In a real system we would send the plan over the robot's
                # communication channel and wait for an ACK. Here we verify
                # the robot is still alive and fetch its state.
                state = await asyncio.wait_for(robot.get_state(), timeout=_CONFIRM_TIMEOUT)
                # Robot is considered ready if battery > 5% and connected.
                ready = state.battery_percent > 5.0 and robot.is_alive()
                return rid, ready
            except asyncio.TimeoutError:
                logger.warning(
                    "PEFASession %s: robot %s did not confirm within %.1fs.",
                    self.session_id, rid, _CONFIRM_TIMEOUT,
                )
                return rid, False
            except Exception as exc:
                logger.warning(
                    "PEFASession %s: robot %s confirmation error — %s.",
                    self.session_id, rid, exc,
                )
                return rid, False

        results = await asyncio.gather(
            *(_check_robot_ready(r) for r in self.robots), return_exceptions=False
        )
        for rid, ready in results:
            self.confirmations[rid] = ready

        all_confirmed = all(self.confirmations.values())
        failed = [rid for rid, ok in self.confirmations.items() if not ok]
        if failed:
            logger.warning(
                "PEFASession %s: robots not ready: %s.", self.session_id, failed
            )

        await self._broadcast_coop_message(
            phase=CoopPhase.CONFIRM,
            sender_id=lead_id,
            payload={"confirmations": self.confirmations},
        )
        return all_confirmed

    async def _execute(self, plan: dict) -> dict[str, bool]:
        """Send synchronised actions to all robots simultaneously.

        Uses ``asyncio.gather`` so all send_action calls are fired concurrently.

        Returns
        -------
        dict[str, bool]
            ``{robot_id: success_flag}``
        """
        actions_spec = plan.get("actions", {})

        async def _send_to_robot(robot: AnyBridge) -> tuple[str, bool]:
            rid = robot.robot_id
            spec = actions_spec.get(rid, {})
            try:
                action = Action(
                    joint_targets=spec.get("joint_targets", [0.0] * 29),
                    gripper_left=float(spec.get("gripper_left", 0.0)),
                    gripper_right=float(spec.get("gripper_right", 0.0)),
                    duration_ms=int(spec.get("duration_ms", 500)),
                )
                await asyncio.wait_for(
                    robot.send_action(action.clipped()),
                    timeout=_EXECUTE_TIMEOUT,
                )
                # Wait for the action duration to simulate execution.
                await asyncio.sleep(action.duration_ms / 1000.0)
                return rid, True
            except asyncio.TimeoutError:
                logger.error(
                    "PEFASession %s: robot %s timed out during execution.",
                    self.session_id, rid,
                )
                return rid, False
            except Exception as exc:
                logger.error(
                    "PEFASession %s: robot %s execution error — %s.",
                    self.session_id, rid, exc,
                )
                return rid, False

        results_list = await asyncio.gather(
            *(_send_to_robot(r) for r in self.robots), return_exceptions=False
        )
        results: dict[str, bool] = dict(results_list)

        lead_id = plan.get("lead_robot", self.robots[0].robot_id)
        await self._broadcast_coop_message(
            phase=CoopPhase.EXECUTE,
            sender_id=lead_id,
            payload={"results": results},
        )
        return results

    async def _feedback_adjust(
        self, results: dict[str, bool], plan: dict, attempt: int
    ) -> bool:
        """Analyse execution results. Return True on full success, False to retry.

        On partial failure (some robots failed), adjusts the plan by removing
        failed robots from the participant list for the next attempt and logs
        a warning. The caller's loop will increment the attempt counter.
        """
        failed_robots = [rid for rid, ok in results.items() if not ok]
        succeeded = [rid for rid, ok in results.items() if ok]

        if not failed_robots:
            logger.info(
                "PEFASession %s: all %d robots succeeded.", self.session_id, len(results)
            )
            await self._broadcast_coop_message(
                phase=CoopPhase.COMPLETE,
                sender_id=plan.get("lead_robot", self.robots[0].robot_id),
                payload={"success": True, "attempt": attempt},
            )
            return True

        logger.warning(
            "PEFASession %s: %d robot(s) failed: %s. Succeeded: %s.",
            self.session_id,
            len(failed_robots),
            failed_robots,
            succeeded,
        )

        if attempt >= _MAX_ATTEMPTS:
            return False

        # Adjust: remove failed robots from future attempts if we still have
        # enough to meet min_robots requirement.
        remaining = [r for r in self.robots if r.robot_id not in failed_robots]
        if len(remaining) >= self.task.min_robots:
            logger.info(
                "PEFASession %s: adjusting team — continuing with %d robot(s).",
                self.session_id,
                len(remaining),
            )
            self.robots = remaining
        else:
            logger.warning(
                "PEFASession %s: too few robots remaining (%d < %d required); "
                "retrying with full team.",
                self.session_id,
                len(remaining),
                self.task.min_robots,
            )

        # Brief pause before retry.
        await asyncio.sleep(1.0)
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _broadcast_coop_message(
        self,
        phase: CoopPhase,
        sender_id: str,
        payload: dict,
    ) -> None:
        """Emit a CoopMessage to all robots (best-effort, non-blocking)."""
        msg = CoopMessage(
            session_id=self.session_id,
            phase=phase,
            sender_id=sender_id,
            receiver_id="*",
            payload=payload,
        )
        logger.debug(
            "PEFASession %s: broadcast phase=%s payload_keys=%s.",
            self.session_id,
            phase.value,
            list(payload.keys()),
        )
        # In a full implementation, messages would be published over the robot
        # communication bus. For now we log them so the protocol is traceable.
        _ = msg  # suppress "unused variable" warnings

    def get_status(self) -> dict:
        """Return a serialisable status snapshot for this session."""
        return {
            "session_id": self.session_id,
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "phase": self.phase,
            "attempt": self._attempt,
            "robots": [r.robot_id for r in self.robots],
            "confirmations": dict(self.confirmations),
            "elapsed_s": round(time.monotonic() - self._start_time, 2)
            if self._start_time
            else 0.0,
        }


class CooperativeCoordinator:
    """Manages all active PEFA sessions across the swarm.

    Usage::

        coordinator = CooperativeCoordinator()
        success = await coordinator.start_cooperative_task(task, [bridge_a, bridge_b])
    """

    def __init__(self) -> None:
        self.active_sessions: dict[str, PEFASession] = {}
        self._completed_sessions: dict[str, dict] = {}

    async def start_cooperative_task(
        self,
        task: TaskNode,
        robot_bridges: list[AnyBridge],
    ) -> bool:
        """Create and run a :class:`PEFASession` for *task*.

        Parameters
        ----------
        task:
            The cooperative task node to execute.
        robot_bridges:
            Bridges for all robots assigned to this task.

        Returns
        -------
        bool
            ``True`` on success.
        """
        session_id = f"{task.task_id}-{uuid.uuid4().hex[:6]}"
        session = PEFASession(task=task, robot_bridges=robot_bridges, session_id=session_id)
        self.active_sessions[session_id] = session
        logger.info(
            "CooperativeCoordinator: starting session %s for task %s.",
            session_id,
            task.task_id,
        )
        try:
            success = await session.run()
        except Exception as exc:
            logger.exception(
                "CooperativeCoordinator: unhandled exception in session %s — %s.",
                session_id,
                exc,
            )
            success = False
        finally:
            final_status = session.get_status()
            self.active_sessions.pop(session_id, None)
            self._completed_sessions[session_id] = {
                **final_status,
                "success": success,
            }

        return success

    def get_session_status(self, session_id: str) -> dict:
        """Return the status dict for an active or recently completed session.

        Returns an empty dict if the session_id is unknown.
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get_status()
        if session_id in self._completed_sessions:
            return self._completed_sessions[session_id]
        return {}

    def list_active_sessions(self) -> list[dict]:
        """Return status snapshots for all currently running sessions."""
        return [s.get_status() for s in self.active_sessions.values()]

    def list_completed_sessions(self) -> list[dict]:
        """Return status records for all finished sessions."""
        return list(self._completed_sessions.values())
