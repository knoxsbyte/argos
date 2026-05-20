"""
argos.comm — Communication layer for the ARGOS robot framework.

Exports the primary public API for robot bridges, the registry, configuration,
and all shared message types.
"""

from argos.comm.messages import (
    Action,
    CoopMessage,
    CoopPhase,
    HeartbeatMessage,
    RobotInfo,
    RobotState,
    TaskMessage,
    TaskStatus,
)
from argos.comm.robot_registry import RobotRegistry
from argos.comm.unitree_bridge import G1Config, MockUnitreeBridge, UnitreeBridge
from argos.comm.battery import BatteryMonitor, BatteryStatus, ChargeState, ChargingDock

__all__ = [
    # Bridges
    "UnitreeBridge",
    "MockUnitreeBridge",
    # Registry
    "RobotRegistry",
    # Configuration
    "G1Config",
    # Message types
    "RobotState",
    "Action",
    "RobotInfo",
    "TaskMessage",
    "TaskStatus",
    "CoopMessage",
    "CoopPhase",
    "HeartbeatMessage",
    # Battery
    "BatteryMonitor",
    "BatteryStatus",
    "ChargeState",
    "ChargingDock",
]
