"""
Singleton registry of all connected G1 robots.

RobotRegistry tracks live UnitreeBridge (or MockUnitreeBridge) instances,
monitors heartbeats, and notifies callers when a robot goes offline.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Union

from argos.comm.messages import HeartbeatMessage
from argos.comm.unitree_bridge import MockUnitreeBridge, UnitreeBridge

logger = logging.getLogger(__name__)

AnyBridge = Union[UnitreeBridge, MockUnitreeBridge]

# How often the monitor loop checks each robot (seconds).
_MONITOR_INTERVAL: float = 1.0
# How many consecutive missed heartbeat checks before a robot is declared dead.
_MAX_MISSED: int = 3


class RobotRegistry:
    """Singleton registry that tracks all active robot bridges.

    Usage::

        registry = RobotRegistry()
        robot_id = await registry.register(bridge)
        …
        await registry.deregister(robot_id)

    The registry spawns an asyncio background task to monitor heartbeats.
    Call ``shutdown()`` to cancel it cleanly.
    """

    _instance: RobotRegistry | None = None

    def __new__(cls) -> "RobotRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:  # type: ignore[has-type]
            return
        self._initialised = True

        # robot_id → bridge
        self._robots: dict[str, AnyBridge] = {}
        # robot_id → consecutive missed heartbeat count
        self._missed: dict[str, int] = {}
        # robot_id → "is currently executing a task" flag
        self._busy: dict[str, bool] = {}
        # Callbacks registered via on_robot_lost()
        self._lost_callbacks: list[Callable[[str], None]] = []

        self._monitor_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        logger.debug("RobotRegistry initialised.")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self, bridge: AnyBridge) -> str:
        """Add a bridge to the registry and start monitoring it.

        Parameters
        ----------
        bridge:
            A connected (or mock) UnitreeBridge instance.

        Returns
        -------
        str
            The robot_id assigned by the registry.  If the bridge already
            exposes a robot_id property it is used as a prefix, otherwise a
            UUID is generated.
        """
        robot_id = self._generate_id(bridge)
        async with self._lock:
            if robot_id in self._robots:
                logger.warning("Robot %s is already registered — skipping.", robot_id)
                return robot_id
            self._robots[robot_id] = bridge
            self._missed[robot_id] = 0
            self._busy[robot_id] = False
            logger.info("Registered robot %s.", robot_id)

        # Start the monitor task the first time a robot is added.
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(
                self._heartbeat_monitor(), name="registry-heartbeat-monitor"
            )
        return robot_id

    async def deregister(self, robot_id: str) -> None:
        """Remove a robot from the registry and disconnect it if still alive."""
        async with self._lock:
            bridge = self._robots.pop(robot_id, None)
            self._missed.pop(robot_id, None)
            self._busy.pop(robot_id, None)

        if bridge is None:
            logger.warning("deregister called for unknown robot_id '%s'.", robot_id)
            return

        try:
            await bridge.disconnect()
        except Exception as exc:
            logger.warning("Error disconnecting %s during deregister: %s", robot_id, exc)

        logger.info("Deregistered robot %s.", robot_id)

        # Stop the monitor task if no robots remain.
        if not self._robots and self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, robot_id: str) -> AnyBridge | None:
        """Return the bridge for *robot_id*, or None if not registered."""
        return self._robots.get(robot_id)

    def list_all(self) -> list[AnyBridge]:
        """Return all registered bridges regardless of status."""
        return list(self._robots.values())

    def list_available(self) -> list[AnyBridge]:
        """Return bridges that are alive and not currently busy."""
        return [
            bridge
            for robot_id, bridge in self._robots.items()
            if bridge.is_alive() and not self._busy.get(robot_id, False)
        ]

    def set_busy(self, robot_id: str, busy: bool) -> None:
        """Mark a robot as busy (executing a task) or free."""
        if robot_id not in self._robots:
            raise KeyError(f"Unknown robot_id: {robot_id!r}")
        self._busy[robot_id] = busy
        logger.debug("Robot %s busy=%s.", robot_id, busy)

    def is_busy(self, robot_id: str) -> bool:
        return self._busy.get(robot_id, False)

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """Send *message* to all registered robots concurrently.

        Each robot receives the message via its bridge's send_action() for
        action dicts, or via a CycloneDDS publication for other payloads.
        For simplicity this implementation logs the broadcast; the actual
        transport is project-specific.
        """
        robot_ids = list(self._robots.keys())
        if not robot_ids:
            logger.warning("broadcast() called but no robots are registered.")
            return

        logger.info("Broadcasting message to %d robot(s): %s", len(robot_ids), message)

        async def _send_one(rid: str, bridge: AnyBridge) -> None:
            try:
                # If it's an action message, dispatch it; otherwise log only.
                if "joint_targets" in message:
                    from argos.comm.messages import Action
                    action = Action(**message)
                    await bridge.send_action(action)
                else:
                    logger.debug("Broadcast payload for %s: %s", rid, message)
            except Exception as exc:
                logger.error("Broadcast to %s failed: %s", rid, exc)

        await asyncio.gather(
            *(_send_one(rid, bridge) for rid, bridge in self._robots.items()),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # Heartbeat monitor
    # ------------------------------------------------------------------

    async def _heartbeat_monitor(self) -> None:
        """Background task: check each robot every second, drop dead ones."""
        logger.info("Heartbeat monitor started.")
        while True:
            try:
                await asyncio.sleep(_MONITOR_INTERVAL)
                await self._check_all()
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor stopped.")
                return
            except Exception as exc:
                logger.exception("Unexpected error in heartbeat monitor: %s", exc)

    async def _check_all(self) -> None:
        """Single pass over all registered robots."""
        to_remove: list[str] = []
        async with self._lock:
            snapshot = dict(self._robots)

        for robot_id, bridge in snapshot.items():
            if bridge.is_alive():
                async with self._lock:
                    self._missed[robot_id] = 0
                # Emit a heartbeat log every check (callers may subscribe via
                # on_robot_lost; a heartbeat-received callback can be added).
                try:
                    state = await bridge.get_state()
                    hb = HeartbeatMessage.from_state(robot_id, state)
                    logger.debug(
                        "Heartbeat OK — %s, battery=%.1f%%",
                        robot_id,
                        hb.state_summary.get("battery_percent", "?"),
                    )
                except Exception as exc:
                    logger.warning("Could not fetch state for %s: %s", robot_id, exc)
            else:
                async with self._lock:
                    self._missed[robot_id] = self._missed.get(robot_id, 0) + 1
                    missed = self._missed[robot_id]

                logger.warning(
                    "Robot %s missed heartbeat check %d/%d.",
                    robot_id, missed, _MAX_MISSED,
                )
                if missed >= _MAX_MISSED:
                    to_remove.append(robot_id)

        for robot_id in to_remove:
            logger.error("Robot %s declared lost after %d missed checks.", robot_id, _MAX_MISSED)
            async with self._lock:
                self._robots.pop(robot_id, None)
                self._missed.pop(robot_id, None)
                self._busy.pop(robot_id, None)
            self._fire_lost(robot_id)

    def _fire_lost(self, robot_id: str) -> None:
        for cb in self._lost_callbacks:
            try:
                cb(robot_id)
            except Exception as exc:
                logger.warning("on_robot_lost callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_robot_lost(self, callback: Callable[[str], None]) -> None:
        """Register *callback* to be called when a robot goes offline.

        The callback receives the robot_id as its sole argument and is called
        synchronously inside the monitor task's event loop.
        """
        self._lost_callbacks.append(callback)
        logger.debug("Registered on_robot_lost callback: %s.", callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel the monitor task and deregister all robots."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        robot_ids = list(self._robots.keys())
        for rid in robot_ids:
            await self.deregister(rid)

        logger.info("RobotRegistry shut down.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(bridge: AnyBridge) -> str:
        """Derive a registry key from the bridge's own robot_id property."""
        base = getattr(bridge, "robot_id", None)
        if base:
            return base
        return f"robot-{uuid.uuid4().hex[:8]}"
