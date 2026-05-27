"""
argos.comm.robot_model — Central robot-model registry for ARGOS.

Every robot-type-specific constant (DOF, joint names, joint limits, locomotion
type, sensor topics, capability flags) lives here so the rest of the framework
can branch on RobotModel without scattering magic numbers throughout the code.

Usage::

    from argos.comm.robot_model import RobotModel, RobotSpec, get_spec, GO2_SPEC

    spec = get_spec("unitree_go2")
    print(spec.dof)          # 12
    print(spec.has_arms)     # False
    print(spec.locomotion_type)  # "quadruped"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class RobotModel(str, Enum):
    """Supported ARGOS robot models."""

    G1  = "unitree_g1"
    GO2 = "unitree_go2"


# ---------------------------------------------------------------------------
# RobotSpec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """Immutable specification for one robot model.

    Parameters
    ----------
    model:
        Enum member that identifies this model.
    dof:
        Total controllable degrees of freedom (joint count).
    max_speed_ms:
        Nominal maximum locomotion speed in m/s.
    has_arms:
        True if the robot has arm links (manipulation capable).
    has_grippers:
        True if the robot has end-effector grippers.
    locomotion_type:
        ``"biped"`` | ``"quadruped"``
    joint_names:
        Ordered tuple of joint names matching joint_positions indices.
    joint_limits_low:
        Lower position limit (radians) for each joint.
    joint_limits_high:
        Upper position limit (radians) for each joint.
    camera_topics:
        DDS/ROS topic names for camera streams (in order: RGB, depth, …).
    lidar_topic:
        DDS/ROS topic name for the primary LiDAR point cloud.
    footprint_m:
        Approximate robot footprint diameter in metres (used for coverage).
    height_m:
        Nominal standing / operational height in metres.
    """

    model:               RobotModel
    dof:                 int
    max_speed_ms:        float
    has_arms:            bool
    has_grippers:        bool
    locomotion_type:     str
    joint_names:         tuple[str, ...]
    joint_limits_low:    tuple[float, ...]
    joint_limits_high:   tuple[float, ...]
    camera_topics:       tuple[str, ...]
    lidar_topic:         str
    footprint_m:         float
    height_m:            float

    # ── convenience helpers ───────────────────────────────────────────────

    def clip_joints(self, positions: list[float]) -> list[float]:
        """Return *positions* clipped to hardware limits."""
        return [
            float(max(lo, min(hi, v)))
            for v, lo, hi in zip(positions, self.joint_limits_low, self.joint_limits_high)
        ]

    def zero_joints(self) -> list[float]:
        """Return a zero-position vector of the correct length."""
        return [0.0] * self.dof

    def can_manipulate(self) -> bool:
        """Return True if this robot can perform manipulation tasks."""
        return self.has_arms and self.has_grippers

    def is_quadruped(self) -> bool:
        return self.locomotion_type == "quadruped"

    def is_biped(self) -> bool:
        return self.locomotion_type == "biped"


# ---------------------------------------------------------------------------
# G1 Spec — 29-DOF biped humanoid
# ---------------------------------------------------------------------------

#: Joint names for the Unitree G1 in SDK order (29 joints).
_G1_JOINT_NAMES: tuple[str, ...] = (
    # Left leg (0-5)
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    # Right leg (6-11)
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    # Waist (12-14)
    "waist_yaw", "waist_roll", "waist_pitch",
    # Left arm (15-20)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_yaw",
    # Right arm (21-26)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_yaw",
    # Grippers (27-28)
    "left_gripper", "right_gripper",
)

_G1_JOINT_LOW: tuple[float, ...] = (
    -2.87, -3.40, -1.30, -1.25, -2.18, -2.00,   # left leg
    -2.87, -0.09, -1.30, -1.25, -2.18, -2.00,   # right leg
    -0.52, -0.52, -3.14,                          # waist
    -2.87, -1.57, -3.14, -1.57, -3.14, -1.57,   # left arm
    -2.87, -3.14, -3.14, -1.57, -3.14, -1.57,   # right arm
    -0.5,  -0.5,                                   # grippers
)

_G1_JOINT_HIGH: tuple[float, ...] = (
    2.87,  0.09,  1.30,  2.16,  2.18,  2.00,    # left leg
    2.87,  3.40,  1.30,  2.16,  2.18,  2.00,    # right leg
    0.52,  0.52,  3.14,                           # waist
    2.87,  3.14,  3.14,  1.57,  3.14,  1.57,    # left arm
    2.87,  1.57,  3.14,  1.57,  3.14,  1.57,    # right arm
    0.5,   0.5,                                    # grippers
)

G1_SPEC = RobotSpec(
    model            = RobotModel.G1,
    dof              = 29,
    max_speed_ms     = 2.0,
    has_arms         = True,
    has_grippers     = True,
    locomotion_type  = "biped",
    joint_names      = _G1_JOINT_NAMES,
    joint_limits_low = _G1_JOINT_LOW,
    joint_limits_high= _G1_JOINT_HIGH,
    camera_topics    = (
        "rt/camera/left_wrist/color",
        "rt/camera/right_wrist/color",
        "rt/camera/head/color",
        "rt/camera/head/depth",
    ),
    lidar_topic      = "rt/livox/lidar",
    footprint_m      = 0.45,
    height_m         = 1.30,
)


# ---------------------------------------------------------------------------
# Go2 Spec — 12-DOF quadruped
# ---------------------------------------------------------------------------

#: Joint names for the Unitree Go2 in SDK order (12 joints).
#: Layout: FL = Front Left, FR = Front Right, BL = Back Left, BR = Back Right.
#: Each leg: hip_yaw (abduction), hip_pitch, knee.
_GO2_JOINT_NAMES: tuple[str, ...] = (
    "fl_hip_yaw",   "fl_hip_pitch",  "fl_knee",    # front-left  (0-2)
    "fr_hip_yaw",   "fr_hip_pitch",  "fr_knee",    # front-right (3-5)
    "bl_hip_yaw",   "bl_hip_pitch",  "bl_knee",    # back-left   (6-8)
    "br_hip_yaw",   "br_hip_pitch",  "br_knee",    # back-right  (9-11)
)

_GO2_JOINT_LOW: tuple[float, ...] = (
    -0.863, -1.571, -2.775,   # FL
    -0.863, -1.571, -2.775,   # FR
    -0.863, -1.571, -2.775,   # BL
    -0.863, -1.571, -2.775,   # BR
)

_GO2_JOINT_HIGH: tuple[float, ...] = (
    0.863,  3.927,  -0.646,   # FL
    0.863,  3.927,  -0.646,   # FR
    0.863,  3.927,  -0.646,   # BL
    0.863,  3.927,  -0.646,   # BR
)

GO2_SPEC = RobotSpec(
    model            = RobotModel.GO2,
    dof              = 12,
    max_speed_ms     = 3.5,
    has_arms         = False,
    has_grippers     = False,
    locomotion_type  = "quadruped",
    joint_names      = _GO2_JOINT_NAMES,
    joint_limits_low = _GO2_JOINT_LOW,
    joint_limits_high= _GO2_JOINT_HIGH,
    camera_topics    = (
        "rt/camera/head/color",
        "rt/camera/head/depth",
    ),
    lidar_topic      = "rt/livox/lidar",
    footprint_m      = 0.55,
    height_m         = 0.70,
)


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

ROBOT_SPECS: dict[RobotModel, RobotSpec] = {
    RobotModel.G1:  G1_SPEC,
    RobotModel.GO2: GO2_SPEC,
}


def get_spec(model: str | RobotModel) -> RobotSpec:
    """Return the :class:`RobotSpec` for *model*.

    Parameters
    ----------
    model:
        Either a :class:`RobotModel` enum member or its string value
        (e.g. ``"unitree_g1"`` or ``"unitree_go2"``).

    Raises
    ------
    ValueError
        If *model* is not a known robot model.
    """
    if isinstance(model, RobotModel):
        key = model
    else:
        try:
            key = RobotModel(model)
        except ValueError:
            valid = [m.value for m in RobotModel]
            raise ValueError(
                f"Unknown robot model: {model!r}. Valid models: {valid}"
            ) from None
    return ROBOT_SPECS[key]


def list_models() -> list[str]:
    """Return the string values of all registered robot models."""
    return [m.value for m in RobotModel]
