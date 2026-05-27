"""
argos.comm.go2_bridge — Async bridge to the Unitree Go2 quadruped robot.

The Go2 has 12 DOF (four legs × 3 joints: hip-yaw, hip-pitch, knee).
It has no arms or grippers, so only floor-based tasks are supported.
Communication uses the Unitree SDK2 / CycloneDDS transport, same as the G1.

When the real SDK is not installed a :class:`MockGo2Bridge` is provided that
simulates the characteristic trot-gait oscillation of a quadruped.

Usage::

    from argos.comm.go2_bridge import Go2Config, MockGo2Bridge

    cfg    = Go2Config(robot_ip="192.168.1.12", name="Go2-Scout")
    bridge = MockGo2Bridge(cfg)
    await bridge.connect()
    state  = await bridge.get_state()
    print(state.joint_positions)   # 12-element list
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from argos.comm.messages import Action, RobotState
from argos.comm.robot_model import GO2_SPEC, RobotModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import — falls back to mock automatically
# ---------------------------------------------------------------------------

try:
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    _GO2_SDK_AVAILABLE = True
    logger.info("unitree_sdk2_python detected — Go2 real SDK mode active.")
except ImportError:
    _GO2_SDK_AVAILABLE = False
    logger.warning(
        "unitree_sdk2_python not installed. Go2 running in mock/simulation mode."
    )

try:
    import pyrealsense2 as rs  # type: ignore[import]
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

try:
    import open3d as o3d  # type: ignore[import]
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Go2Config(BaseModel):
    """Connection and operational parameters for a single Go2 robot."""

    robot_ip: str = Field(description="IP address of the Go2 robot.")
    name: str = Field(default="Go2", description="Human-readable robot name.")
    control_freq: int = Field(
        default=50,
        gt=0,
        description="Control loop frequency in Hz.",
    )
    network_interface: str = Field(
        default="eth0",
        description="Network interface used for CycloneDDS communication.",
    )
    connect_timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum seconds to wait during connect().",
    )
    robot_model: str = Field(
        default=RobotModel.GO2.value,
        description="Robot model identifier — always 'unitree_go2' for Go2.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Quadruped trot-gait oscillation parameters (12 joints)
# ---------------------------------------------------------------------------

# Trot gait: diagonal pairs (FL+BR) and (FR+BL) swing together.
# Amplitude and frequency for each joint in the default trot.
_GO2_TROT_AMPS: list[float] = [
    0.05, 0.35, 0.60,   # FL hip-yaw, hip-pitch, knee
    0.05, 0.35, 0.60,   # FR hip-yaw, hip-pitch, knee
    0.05, 0.35, 0.60,   # BL hip-yaw, hip-pitch, knee
    0.05, 0.35, 0.60,   # BR hip-yaw, hip-pitch, knee
]
_GO2_TROT_FREQS: list[float] = [
    0.3, 2.2, 2.2,      # FL
    0.3, 2.2, 2.2,      # FR
    0.3, 2.2, 2.2,      # BL
    0.3, 2.2, 2.2,      # BR
]
# Phase offsets implement the trot diagonal pairing (0=FL/BR swing, π=FR/BL swing)
_GO2_TROT_PHASES: list[float] = [
    0.0,   0.0,   0.0,   # FL — phase group A
    math.pi, math.pi, math.pi,  # FR — phase group B
    math.pi, math.pi, math.pi,  # BL — phase group B
    0.0,   0.0,   0.0,   # BR — phase group A
]


# ---------------------------------------------------------------------------
# Real hardware bridge
# ---------------------------------------------------------------------------


class Go2Bridge:
    """Async interface to a single Unitree Go2 quadruped robot.

    State updates arrive via CycloneDDS callbacks and are deposited into an
    asyncio.Queue so that async consumers never block on SDK I/O.

    The Go2 has 12 joints, no arms, no grippers.  ``get_pointcloud()``
    returns Livox MID-360 data when the Livox ROS2 driver bridge is active.
    """

    def __init__(self, config: Go2Config) -> None:
        self.config = config
        self._spec = GO2_SPEC
        self._connected: bool = False
        self._last_heartbeat: float = 0.0

        self._state_queue: asyncio.Queue[RobotState] = asyncio.Queue(maxsize=10)
        self._latest_state: RobotState = RobotState(
            joint_positions=self._spec.zero_joints(),
            joint_velocities=self._spec.zero_joints(),
            robot_model=RobotModel.GO2.value,
        )

        self._channel_factory: Any = None
        self._state_subscriber: Any = None
        self._sport_client: Any = None
        self._rs_pipeline: Any = None
        self._rs_align: Any = None

        logger.debug("Go2Bridge created for %s (%s)", config.name, config.robot_ip)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialise CycloneDDS and subscribe to Go2 state topics."""
        if not _GO2_SDK_AVAILABLE:
            logger.error(
                "Cannot connect: unitree_sdk2_python is not installed. "
                "Use MockGo2Bridge for development."
            )
            return False

        logger.info("Connecting to Go2 %s at %s …", self.config.name, self.config.robot_ip)
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._sdk_init),
                timeout=self.config.connect_timeout,
            )
            self._connected = True
            self._last_heartbeat = time.time()
            logger.info("Connected to Go2 %s.", self.config.name)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "Timeout connecting to Go2 %s after %.1fs.",
                self.config.name,
                self.config.connect_timeout,
            )
            return False
        except Exception as exc:
            logger.exception("Unexpected error connecting to Go2 %s: %s", self.config.name, exc)
            return False

    def _sdk_init(self) -> None:
        """Blocking SDK initialisation — runs in executor thread."""
        ChannelFactory.Instance().Init(0, self.config.network_interface)
        self._state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_subscriber.Init(self._on_lowstate, 10)
        self._sport_client = SportClient()
        self._sport_client.SetTimeout(self.config.connect_timeout)
        self._sport_client.Init()

    def _on_lowstate(self, msg: Any) -> None:
        """CycloneDDS callback — runs in SDK thread, must be non-blocking."""
        try:
            state = self._parse_lowstate(msg)
            self._latest_state = state
            self._last_heartbeat = time.time()
            if self._state_queue.full():
                try:
                    self._state_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._state_queue.put_nowait(state)
        except Exception as exc:
            logger.warning("Go2 error parsing lowstate: %s", exc)

    def _parse_lowstate(self, msg: Any) -> RobotState:
        """Convert a raw SDK LowState message into a 12-DOF RobotState."""
        joint_pos = [msg.motor_state[i].q  for i in range(12)]
        joint_vel = [msg.motor_state[i].dq for i in range(12)]
        imu = msg.imu_state
        return RobotState(
            robot_model=RobotModel.GO2.value,
            battery_percent=float(getattr(msg, "power_v", 24.0)) / 0.24,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_accel=[
                imu.accelerometer[0],
                imu.accelerometer[1],
                imu.accelerometer[2],
            ],
            imu_gyro=[
                imu.gyroscope[0],
                imu.gyroscope[1],
                imu.gyroscope[2],
            ],
            orientation=[
                imu.quaternion[0],
                imu.quaternion[1],
                imu.quaternion[2],
                imu.quaternion[3],
            ],
            timestamp=time.time(),
        )

    async def disconnect(self) -> None:
        """Cleanly shut down SDK channels and sensor pipelines."""
        logger.info("Disconnecting Go2 %s …", self.config.name)
        self._connected = False
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None
        self._state_subscriber = None
        self._sport_client = None
        logger.info("Disconnected Go2 %s.", self.config.name)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    async def get_state(self) -> RobotState:
        """Return the most recent 12-DOF robot state."""
        period = 1.0 / self.config.control_freq
        try:
            state = await asyncio.wait_for(
                self._state_queue.get(), timeout=period * 2
            )
            self._latest_state = state
            return state
        except asyncio.TimeoutError:
            logger.debug("Go2 get_state timed out; returning cached state.")
            return self._latest_state

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    async def send_action(self, action: Action) -> None:
        """Validate and dispatch a 12-DOF motor command to the Go2.

        The action's joint_targets are clipped to Go2 hardware limits before
        transmission via the SportClient velocity interface.
        """
        if not self._connected or self._sport_client is None:
            raise RuntimeError(f"Go2 {self.config.name} is not connected.")

        clipped = self._spec.clip_joints(action.joint_targets[:12])
        logger.debug(
            "Sending action to Go2 %s: duration_ms=%d",
            self.config.name,
            action.duration_ms,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sdk_send_action, clipped, action.duration_ms
        )

    def _sdk_send_action(self, targets: list[float], duration_ms: int) -> None:
        """Blocking SDK write — runs in executor thread."""
        self._sport_client.SetJointPositionCmd(
            targets,
            duration=duration_ms / 1000.0,
        )

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------

    async def get_camera_frame(self) -> np.ndarray:
        """Return the latest RGB frame from the Go2 head RealSense D430."""
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed.")
        await self._ensure_realsense()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_rgb)

    async def get_depth_frame(self) -> np.ndarray:
        """Return the latest depth frame in metres as a float32 HW array."""
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed.")
        await self._ensure_realsense()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_depth)

    async def get_pointcloud(self) -> np.ndarray:
        """Return a Livox MID-360 LiDAR point cloud as an (N, 3) float32 array."""
        if not _O3D_AVAILABLE:
            raise RuntimeError("open3d is not installed; cannot read LiDAR data.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_lidar)

    async def _ensure_realsense(self) -> None:
        if self._rs_pipeline is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._init_realsense)

    def _init_realsense(self) -> None:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        self._rs_align = rs.align(rs.stream.color)
        self._rs_pipeline = pipeline
        logger.info("RealSense D430 initialised on Go2 %s.", self.config.name)

    def _capture_rgb(self) -> np.ndarray:
        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        return np.asanyarray(aligned.get_color_frame().get_data())

    def _capture_depth(self) -> np.ndarray:
        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        raw = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32)
        return raw / 1000.0

    def _capture_lidar(self) -> np.ndarray:
        raise NotImplementedError(
            "Livox MID-360 capture requires the Livox ROS2 driver bridge. "
            "Use ros2_bridge module or MockGo2Bridge for development."
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def is_alive(self) -> bool:
        """Return True if a heartbeat was received within the last 2 seconds."""
        return self._connected and (time.time() - self._last_heartbeat) < 2.0

    @property
    def robot_id(self) -> str:
        return f"{self.config.name}@{self.config.robot_ip}"


# ---------------------------------------------------------------------------
# Mock bridge — for development / CI without hardware
# ---------------------------------------------------------------------------


class MockGo2Bridge:
    """Simulated Go2 bridge that produces physically plausible fake data.

    Joint positions follow a trot gait oscillation (diagonal pairs in-phase,
    opposite pairs 180° out-of-phase). Battery drains slowly and the robot
    wanders randomly in XY. All async methods complete instantly.
    """

    def __init__(self, config: Go2Config) -> None:
        self.config = config
        self._spec = GO2_SPEC
        self._connected: bool = False
        self._start_time: float = 0.0
        self._battery: float = 100.0
        self._position: list[float] = [0.0, 0.0, 0.35]  # standing height
        self._orientation: list[float] = [1.0, 0.0, 0.0, 0.0]
        self._last_heartbeat: float = 0.0
        self._assigned_id: str = f"mock-{config.name}@{config.robot_ip}"
        self._rng = np.random.default_rng(seed=99)

        # Standing joint pose (nominal trot stance)
        self._nominal: list[float] = [
            0.0,  0.8, -1.5,   # FL
            0.0,  0.8, -1.5,   # FR
            0.0,  0.8, -1.5,   # BL
            0.0,  0.8, -1.5,   # BR
        ]

        logger.debug("MockGo2Bridge created for %s.", config.name)

    async def connect(self) -> bool:
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._connected = True
        logger.info("MockGo2Bridge: simulated connect for %s.", self.config.name)
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("MockGo2Bridge: simulated disconnect for %s.", self.config.name)

    async def get_state(self) -> RobotState:
        """Return a 12-element joint state simulating a trot gait."""
        self._tick()
        t = time.time() - self._start_time

        joint_pos: list[float] = []
        joint_vel: list[float] = []
        for i in range(12):
            amp   = _GO2_TROT_AMPS[i]
            freq  = _GO2_TROT_FREQS[i]
            phase = _GO2_TROT_PHASES[i]
            theta = 2 * math.pi * freq * t + phase + i * 0.1
            pos   = self._nominal[i] + amp * math.sin(theta)
            vel   = amp * 2 * math.pi * freq * math.cos(theta)
            joint_pos.append(float(pos))
            joint_vel.append(float(vel))

        return RobotState(
            robot_model=RobotModel.GO2.value,
            battery_percent=self._battery,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_accel=[
                float(self._rng.normal(0.0, 0.08)),
                float(self._rng.normal(0.0, 0.08)),
                9.81 + float(self._rng.normal(0.0, 0.05)),
            ],
            imu_gyro=[
                float(self._rng.normal(0.0, 0.01)),
                float(self._rng.normal(0.0, 0.01)),
                float(self._rng.normal(0.0, 0.01)),
            ],
            position=list(self._position),
            orientation=list(self._orientation),
            timestamp=time.time(),
        )

    def _tick(self) -> None:
        """Advance simulated state: drain battery, move position."""
        self._last_heartbeat = time.time()
        elapsed = time.time() - self._start_time
        # Go2 battery lasts ~2 hours; drain ~0.83% per minute
        self._battery = max(0.0, 100.0 - elapsed / 72.0)
        # Random walk in XY at higher speed than G1
        self._position[0] += float(self._rng.normal(0.0, 0.002))
        self._position[1] += float(self._rng.normal(0.0, 0.002))

    async def send_action(self, action: Action) -> None:
        """Accept a 12-DOF action and log it (no real hardware)."""
        targets = action.joint_targets[:12] if len(action.joint_targets) >= 12 else action.joint_targets
        clipped = self._spec.clip_joints(targets)
        logger.debug(
            "MockGo2Bridge: action received for %s, duration_ms=%d.",
            self.config.name,
            action.duration_ms,
        )

    async def get_camera_frame(self) -> np.ndarray:
        """Return a synthetic 480×848 RGB frame (gradient + noise)."""
        frame = np.zeros((480, 848, 3), dtype=np.uint8)
        t = int(time.time() * 30) % 256
        frame[:, :, 0] = np.linspace(t, (t + 100) % 256, 848, dtype=np.uint8)
        frame[:, :, 2] = np.linspace(0, 180, 480, dtype=np.uint8).reshape(-1, 1)
        noise = self._rng.integers(0, 15, size=(480, 848, 3), dtype=np.uint8)
        return np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    async def get_depth_frame(self) -> np.ndarray:
        """Return a synthetic 480×848 depth map in metres."""
        base = np.full((480, 848), 1.5, dtype=np.float32)
        noise = self._rng.normal(0.0, 0.015, size=(480, 848)).astype(np.float32)
        return base + noise

    async def get_pointcloud(self) -> np.ndarray:
        """Return a synthetic (N, 3) float32 point cloud simulating a floor + walls."""
        n = 12_000
        pts = self._rng.uniform(-4.0, 4.0, size=(n, 3)).astype(np.float32)
        # Dense floor plane at z≈0
        floor_n = n // 3
        pts[:floor_n, 2] = self._rng.uniform(-0.03, 0.03, size=floor_n).astype(np.float32)
        # Walls at ±3 m
        wall_n = n // 6
        pts[floor_n:floor_n + wall_n, 0] = self._rng.uniform(2.9, 3.0, size=wall_n).astype(np.float32)
        return pts

    def is_connected(self) -> bool:
        return self._connected

    def is_alive(self) -> bool:
        return self._connected and (time.time() - self._last_heartbeat) < 2.0

    @property
    def robot_id(self) -> str:
        return self._assigned_id
