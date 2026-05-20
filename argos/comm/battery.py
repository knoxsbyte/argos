"""Battery monitoring and charging dock management for ARGOS swarm robots."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class ChargeState(Enum):
    NOMINAL   = "nominal"    # > 40 %
    LOW       = "low"        # 15–40 %
    CRITICAL  = "critical"   # < 15 % — return to dock immediately
    CHARGING  = "charging"
    FULL      = "full"       # >= 98 %


@dataclass
class ChargingDock:
    dock_id: str
    position: tuple[float, float]   # (x, y) in room metres
    max_robots: int = 1
    occupied_by: list[str] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return len(self.occupied_by) < self.max_robots

    def assign(self, robot_id: str) -> bool:
        if self.is_available and robot_id not in self.occupied_by:
            self.occupied_by.append(robot_id)
            return True
        return False

    def release(self, robot_id: str) -> None:
        self.occupied_by = [r for r in self.occupied_by if r != robot_id]


@dataclass
class BatteryStatus:
    robot_id: str
    percent: float
    state: ChargeState
    estimated_minutes_remaining: float
    dock_assigned: str | None = None
    last_updated: float = field(default_factory=time.time)


class BatteryMonitor:
    """
    Tracks battery levels for all registered robots, assigns charging docks
    when a robot goes critical, and fires callbacks on state transitions.

    Usage:
        monitor = BatteryMonitor(docks=[ChargingDock("dock-1", (0.5, 0.5))])
        monitor.on_critical(lambda rid: coordinator.send_to_dock(rid))
        await monitor.start(registry)
    """

    LOW_THRESHOLD      = 40.0   # %
    CRITICAL_THRESHOLD = 15.0   # %
    FULL_THRESHOLD     = 98.0   # %
    POLL_INTERVAL      = 5.0    # seconds

    # Rough discharge rate used for time-remaining estimate (% per minute at full load)
    DISCHARGE_RATE_PCT_PER_MIN = 0.8

    def __init__(self, docks: list[ChargingDock] | None = None) -> None:
        self._docks: list[ChargingDock] = docks or [
            ChargingDock("dock-A", (0.5, 0.5)),
            ChargingDock("dock-B", (0.5, 3.5)),
        ]
        self._statuses: dict[str, BatteryStatus] = {}
        self._callbacks: dict[str, list[Callable[[str], None]]] = {
            "low":      [],
            "critical": [],
            "full":     [],
        }
        self._task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, registry) -> None:
        """Begin polling loop. Pass a RobotRegistry instance."""
        self._registry = registry
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("BatteryMonitor started — %d docks registered", len(self._docks))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── callbacks ─────────────────────────────────────────────────────────────

    def on_low(self, cb: Callable[[str], None]) -> None:
        self._callbacks["low"].append(cb)

    def on_critical(self, cb: Callable[[str], None]) -> None:
        self._callbacks["critical"].append(cb)

    def on_full(self, cb: Callable[[str], None]) -> None:
        self._callbacks["full"].append(cb)

    # ── public queries ────────────────────────────────────────────────────────

    def get_status(self, robot_id: str) -> BatteryStatus | None:
        return self._statuses.get(robot_id)

    def all_statuses(self) -> list[BatteryStatus]:
        return list(self._statuses.values())

    def find_nearest_dock(
        self, robot_position: tuple[float, float]
    ) -> ChargingDock | None:
        """Return the nearest available dock to robot_position."""
        available = [d for d in self._docks if d.is_available]
        if not available:
            return None
        rx, ry = robot_position
        return min(
            available,
            key=lambda d: (d.position[0] - rx) ** 2 + (d.position[1] - ry) ** 2,
        )

    def assign_dock(self, robot_id: str, robot_position: tuple[float, float]) -> ChargingDock | None:
        """Find and assign the nearest available dock. Returns None if all full."""
        dock = self.find_nearest_dock(robot_position)
        if dock and dock.assign(robot_id):
            if robot_id in self._statuses:
                self._statuses[robot_id].dock_assigned = dock.dock_id
            logger.info("%s → assigned to %s", robot_id, dock.dock_id)
            return dock
        logger.warning("%s: no available dock", robot_id)
        return None

    def release_dock(self, robot_id: str) -> None:
        for dock in self._docks:
            dock.release(robot_id)
        if robot_id in self._statuses:
            self._statuses[robot_id].dock_assigned = None

    def dock_summary(self) -> list[dict]:
        return [
            {
                "dock_id":   d.dock_id,
                "position":  d.position,
                "occupied":  d.occupied_by,
                "available": d.is_available,
            }
            for d in self._docks
        ]

    # ── internal ──────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.debug("BatteryMonitor poll error: %s", exc)
            await asyncio.sleep(self.POLL_INTERVAL)

    async def _poll_once(self) -> None:
        for robot in self._registry.list_all():
            try:
                state = await robot.get_state()
            except Exception:
                continue

            pct  = state.battery_percent
            prev = self._statuses.get(robot.robot_id)

            charge_state = self._classify(pct, prev)
            minutes_left = pct / self.DISCHARGE_RATE_PCT_PER_MIN if charge_state not in (
                ChargeState.CHARGING, ChargeState.FULL
            ) else float("inf")

            status = BatteryStatus(
                robot_id=robot.robot_id,
                percent=pct,
                state=charge_state,
                estimated_minutes_remaining=minutes_left,
                dock_assigned=prev.dock_assigned if prev else None,
            )
            self._statuses[robot.robot_id] = status

            self._fire_transitions(robot.robot_id, prev, status)

    def _classify(self, pct: float, prev: BatteryStatus | None) -> ChargeState:
        if prev and prev.state == ChargeState.CHARGING:
            if pct >= self.FULL_THRESHOLD:
                return ChargeState.FULL
            return ChargeState.CHARGING
        if pct >= self.FULL_THRESHOLD:
            return ChargeState.FULL
        if pct < self.CRITICAL_THRESHOLD:
            return ChargeState.CRITICAL
        if pct < self.LOW_THRESHOLD:
            return ChargeState.LOW
        return ChargeState.NOMINAL

    def _fire_transitions(
        self,
        robot_id: str,
        prev: BatteryStatus | None,
        current: BatteryStatus,
    ) -> None:
        prev_state = prev.state if prev else None

        if current.state == ChargeState.CRITICAL and prev_state != ChargeState.CRITICAL:
            logger.warning("CRITICAL battery: %s @ %.1f%%", robot_id, current.percent)
            for cb in self._callbacks["critical"]:
                cb(robot_id)

        elif current.state == ChargeState.LOW and prev_state == ChargeState.NOMINAL:
            logger.info("LOW battery: %s @ %.1f%%", robot_id, current.percent)
            for cb in self._callbacks["low"]:
                cb(robot_id)

        elif current.state in (ChargeState.FULL,) and prev_state == ChargeState.CHARGING:
            logger.info("FULL: %s — ready to deploy", robot_id)
            for cb in self._callbacks["full"]:
                cb(robot_id)
