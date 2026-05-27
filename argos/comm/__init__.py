"""
argos.comm — Communication layer for the ARGOS robot framework.

Exports the primary public API for robot bridges, the registry, configuration,
and all shared message types. Supports both Unitree G1 (29 DOF humanoid) and
Unitree Go2 (12 DOF quadruped) robots.
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
from argos.comm.robot_model import (
    GO2_SPEC,
    G1_SPEC,
    ROBOT_SPECS,
    RobotModel,
    RobotSpec,
    get_spec,
    list_models,
)
from argos.comm.robot_registry import RobotRegistry
from argos.comm.unitree_bridge import G1Config, MockUnitreeBridge, UnitreeBridge
from argos.comm.go2_bridge import Go2Bridge, Go2Config, MockGo2Bridge
from argos.comm.battery import BatteryMonitor, BatteryStatus, ChargeState, ChargingDock

__all__ = [
    # G1 bridge
    "UnitreeBridge",
    "MockUnitreeBridge",
    "G1Config",
    # Go2 bridge
    "Go2Bridge",
    "MockGo2Bridge",
    "Go2Config",
    # Robot model abstraction
    "RobotModel",
    "RobotSpec",
    "G1_SPEC",
    "GO2_SPEC",
    "ROBOT_SPECS",
    "get_spec",
    "list_models",
    # Registry
    "RobotRegistry",
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
