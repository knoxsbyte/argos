"""
Shared data structures for ARGOS robot communication layer.

All message types are Pydantic models for validation and serialization,
plus Python dataclasses where plain containers suffice.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    FAILED = "FAILED"


class CoopPhase(str, Enum):
    PROPOSE = "PROPOSE"
    CONFIRM = "CONFIRM"
    EXECUTE = "EXECUTE"
    COMPLETE = "COMPLETE"


# ---------------------------------------------------------------------------
# Core robot state
# ---------------------------------------------------------------------------


class RobotState(BaseModel):
    """Full state snapshot from a Unitree robot (G1 or Go2).

    Joint ordering follows the Unitree SDK convention:
      - G1:  29 DOF (legs, waist, arms, grippers)
      - Go2: 12 DOF (FL/FR/BL/BR × hip-yaw, hip-pitch, knee)

    IMU values are in SI units: accel [m/s²], gyro [rad/s].
    Position is in the world frame [m]. Orientation is a unit quaternion
    stored as (w, x, y, z).
    """

    robot_model: str = Field(
        default="unitree_g1",
        description="Robot model identifier, e.g. 'unitree_g1' or 'unitree_go2'.",
    )
    battery_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="State-of-charge as a percentage [0, 100].",
    )
    joint_positions: list[float] = Field(
        default_factory=lambda: [0.0] * 29,
        description="Joint angles in radians. 29 values for G1, 12 for Go2.",
    )
    joint_velocities: list[float] = Field(
        default_factory=lambda: [0.0] * 29,
        description="Joint angular velocities in rad/s. 29 values for G1, 12 for Go2.",
    )
    imu_accel: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 9.81],
        description="Linear acceleration [ax, ay, az] in m/s².",
    )
    imu_gyro: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Angular velocity [gx, gy, gz] in rad/s.",
    )
    position: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="World-frame position [x, y, z] in metres.",
    )
    orientation: list[float] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0],
        description="Unit quaternion [w, x, y, z].",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of the state snapshot.",
    )

    @field_validator("joint_positions", "joint_velocities")
    @classmethod
    def _check_dof(cls, v: list[float]) -> list[float]:
        if len(v) not in (12, 29):
            raise ValueError(
                f"Expected 12 (Go2) or 29 (G1) joint values, got {len(v)}"
            )
        return v

    @property
    def dof(self) -> int:
        """Return the number of degrees of freedom inferred from joint count."""
        return len(self.joint_positions)

    @field_validator("imu_accel", "imu_gyro", "position")
    @classmethod
    def _check_xyz(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"Expected 3 values, got {len(v)}")
        return v

    @field_validator("orientation")
    @classmethod
    def _check_quat(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(f"Expected 4 values (w,x,y,z), got {len(v)}")
        return v

    def to_summary(self) -> dict[str, Any]:
        """Compact summary for heartbeats and logs."""
        return {
            "battery_percent": round(self.battery_percent, 1),
            "position": [round(p, 3) for p in self.position],
            "orientation": [round(q, 4) for q in self.orientation],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Action command
# ---------------------------------------------------------------------------

# G1 joint position limits (radians) — conservative safe values.
# Index corresponds to the same ordering as RobotState.joint_positions.
JOINT_LIMITS_LOW: list[float] = [
    -2.87, -3.40, -1.30, -1.25, -2.18, -2.00,  # left leg  (0-5)
    -2.87, -0.09, -1.30, -1.25, -2.18, -2.00,  # right leg (6-11)
    -0.52, -0.52, -3.14,                         # waist     (12-14)
    -2.87, -1.57, -3.14, -1.57, -3.14, -1.57,  # left arm  (15-20)
    -2.87, -3.14, -3.14, -1.57, -3.14, -1.57,  # right arm (21-26)
    -0.5,  -0.5,                                  # left  gripper fingers (27-28)
]
JOINT_LIMITS_HIGH: list[float] = [
    2.87,  0.09,  1.30,  2.16,  2.18,  2.00,   # left leg
    2.87,  3.40,  1.30,  2.16,  2.18,  2.00,   # right leg
    0.52,  0.52,  3.14,                          # waist
    2.87,  3.14,  3.14,  1.57,  3.14,  1.57,   # left arm
    2.87,  1.57,  3.14,  1.57,  3.14,  1.57,   # right arm
    0.5,   0.5,                                   # left gripper fingers
]


class Action(BaseModel):
    """Motor command sent to a Unitree robot (G1 or Go2).

    For G1: joint_targets should have 29 values.
    For Go2: joint_targets should have 12 values; gripper fields are unused.
    duration_ms is how long the motion should take; the bridge interpolates.
    """

    joint_targets: list[float] = Field(
        default_factory=lambda: [0.0] * 29,
        description="Target joint positions in radians. 29 for G1, 12 for Go2.",
    )
    gripper_left: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Left gripper opening fraction [0=closed, 1=open]. G1 only.",
    )
    gripper_right: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Right gripper opening fraction [0=closed, 1=open]. G1 only.",
    )
    duration_ms: int = Field(
        default=500,
        gt=0,
        description="Duration for executing the action in milliseconds.",
    )

    @field_validator("joint_targets")
    @classmethod
    def _check_dof(cls, v: list[float]) -> list[float]:
        if len(v) not in (12, 29):
            raise ValueError(
                f"Expected 12 (Go2) or 29 (G1) joint targets, got {len(v)}"
            )
        return v

    def clipped(self) -> "Action":
        """Return a copy with joint_targets clipped to safe hardware limits."""
        clipped_targets = [
            float(max(lo, min(hi, val)))
            for val, lo, hi in zip(
                self.joint_targets, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
            )
        ]
        return Action(
            joint_targets=clipped_targets,
            gripper_left=self.gripper_left,
            gripper_right=self.gripper_right,
            duration_ms=self.duration_ms,
        )


# ---------------------------------------------------------------------------
# Robot info / registry record
# ---------------------------------------------------------------------------


class RobotInfo(BaseModel):
    """Static metadata about a connected Unitree robot (G1 or Go2)."""

    robot_id: str = Field(description="Unique identifier assigned by the registry.")
    name: str = Field(description="Human-readable name, e.g. 'G1-Alpha' or 'Go2-Scout'.")
    ip: str = Field(description="IP address of the robot.")
    model: str = Field(default="Unitree G1", description="Human-readable robot model string.")
    robot_model: str = Field(
        default="unitree_g1",
        description="Machine-readable model ID: 'unitree_g1' | 'unitree_go2'.",
    )
    dof: int = Field(default=29, description="Degrees of freedom (29 for G1, 12 for Go2).")
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form capability flags, e.g. {'lidar': True, 'has_arms': False}.",
    )


# ---------------------------------------------------------------------------
# Task coordination
# ---------------------------------------------------------------------------


class TaskMessage(BaseModel):
    """A task dispatched to one or more robots by the task manager."""

    task_id: str = Field(description="UUID of the task.")
    task_type: str = Field(
        description="Task kind, e.g. 'sweep_zone', 'pick_object', 'patrol'."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific parameters.",
    )
    assigned_robot: str | None = Field(
        default=None,
        description="robot_id of the assigned robot; None = unassigned.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task lifecycle state.",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the task was created.",
    )
    updated_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp of the last status change.",
    )

    def transition(self, new_status: TaskStatus) -> "TaskMessage":
        """Return a copy with the status updated and updated_at refreshed."""
        return self.model_copy(
            update={"status": new_status, "updated_at": time.time()}
        )


# ---------------------------------------------------------------------------
# Robot–robot cooperation
# ---------------------------------------------------------------------------


class CoopMessage(BaseModel):
    """Multi-robot coordination message exchanged during joint tasks."""

    session_id: str = Field(
        description="Shared session identifier linking all messages in a coop episode."
    )
    phase: CoopPhase = Field(description="Current cooperation phase.")
    sender_id: str = Field(description="robot_id of the sending robot.")
    receiver_id: str = Field(
        description="robot_id of the intended recipient; '*' = broadcast."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Phase-specific data (e.g. proposed trajectory, confirmation token).",
    )
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class HeartbeatMessage(BaseModel):
    """Periodic liveness signal emitted by each connected robot."""

    robot_id: str = Field(description="robot_id of the sender.")
    timestamp: float = Field(default_factory=time.time)
    state_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact state snapshot (see RobotState.to_summary()).",
    )

    @classmethod
    def from_state(cls, robot_id: str, state: RobotState) -> "HeartbeatMessage":
        """Convenience constructor that populates state_summary automatically."""
        return cls(robot_id=robot_id, state_summary=state.to_summary())
