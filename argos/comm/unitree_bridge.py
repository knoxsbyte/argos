"""
Async bridge to the Unitree G1 humanoid robot via unitree_sdk2_python.

When the real SDK is not installed (development / CI), a MockSDK fallback
is used automatically so the rest of ARGOS can be exercised without hardware.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from argos.comm.messages import Action, RobotState
from argos.comm.robot_model import G1_SPEC, RobotModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import with graceful mock fallback
# ---------------------------------------------------------------------------

try:
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import G1LocoClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    _SDK_AVAILABLE = True
    logger.info("unitree_sdk2_python detected — real SDK mode active.")
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning(
        "unitree_sdk2_python not installed. Running in mock/simulation mode."
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


class G1Config(BaseModel):
    """Connection and operational parameters for a single G1 robot."""

    ip: str = Field(description="IP address of the G1 robot.")
    name: str = Field(default="G1", description="Human-readable robot name.")
    dof: int = Field(default=29, description="Degrees of freedom.")
    robot_model: str = Field(
        default=RobotModel.G1.value,
        description="Robot model identifier — always 'unitree_g1' for G1Config.",
    )
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

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Joint limits — loaded from the central G1_SPEC so there is one source of truth
# ---------------------------------------------------------------------------

_JOINT_LOW  = list(G1_SPEC.joint_limits_low)
_JOINT_HIGH = list(G1_SPEC.joint_limits_high)


def _clip_joints(targets: list[float]) -> list[float]:
    return G1_SPEC.clip_joints(targets)


# ---------------------------------------------------------------------------
# Real hardware bridge
# ---------------------------------------------------------------------------


class UnitreeBridge:
    """Async interface to a single Unitree G1 humanoid robot.

    State updates arrive via CycloneDDS callbacks and are deposited into an
    asyncio.Queue so that async consumers never block on SDK I/O.
    """

    def __init__(self, config: G1Config) -> None:
        self.config = config
        self._connected: bool = False
        self._last_heartbeat: float = 0.0

        # Thread-safe state pipeline: SDK callback → queue → async consumer.
        self._state_queue: asyncio.Queue[RobotState] = asyncio.Queue(maxsize=10)
        self._latest_state: RobotState = RobotState()

        # SDK objects (set during connect)
        self._channel_factory: Any = None
        self._state_subscriber: Any = None
        self._loco_client: Any = None

        # RealSense pipeline (optional)
        self._rs_pipeline: Any = None
        self._rs_align: Any = None

        logger.debug("UnitreeBridge created for %s (%s)", config.name, config.ip)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialise CycloneDDS and subscribe to robot state topics.

        Returns True on success, False if connection times out.
        """
        if not _SDK_AVAILABLE:
            logger.error(
                "Cannot connect: unitree_sdk2_python is not installed. "
                "Use MockUnitreeBridge for development."
            )
            return False

        logger.info("Connecting to %s at %s …", self.config.name, self.config.ip)
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._sdk_init),
                timeout=self.config.connect_timeout,
            )
            self._connected = True
            self._last_heartbeat = time.time()
            logger.info("Connected to %s.", self.config.name)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "Timeout connecting to %s after %.1fs.",
                self.config.name,
                self.config.connect_timeout,
            )
            return False
        except Exception as exc:
            logger.exception("Unexpected error connecting to %s: %s", self.config.name, exc)
            return False

    def _sdk_init(self) -> None:
        """Blocking SDK initialisation — runs in executor thread."""
        ChannelFactory.Instance().Init(0, self.config.network_interface)
        self._state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_subscriber.Init(self._on_lowstate, 10)
        self._loco_client = G1LocoClient()
        self._loco_client.SetTimeout(self.config.connect_timeout)
        self._loco_client.Init()

    def _on_lowstate(self, msg: Any) -> None:
        """CycloneDDS callback — runs in SDK thread, must be non-blocking."""
        try:
            state = self._parse_lowstate(msg)
            self._latest_state = state
            self._last_heartbeat = time.time()
            # Non-blocking put; drop oldest if queue is full.
            if self._state_queue.full():
                try:
                    self._state_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._state_queue.put_nowait(state)
        except Exception as exc:
            logger.warning("Error parsing lowstate: %s", exc)

    @staticmethod
    def _parse_lowstate(msg: Any) -> RobotState:
        """Convert a raw SDK LowState message into a RobotState."""
        joint_pos = [msg.motor_state[i].q for i in range(29)]
        joint_vel = [msg.motor_state[i].dq for i in range(29)]
        imu = msg.imu_state
        return RobotState(
            battery_percent=float(msg.power_v) / 0.29,  # approx. from voltage
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_accel=[imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2]],
            imu_gyro=[imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]],
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
        logger.info("Disconnecting %s …", self.config.name)
        self._connected = False
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None
        # SDK channels are cleaned up by garbage collection.
        self._state_subscriber = None
        self._loco_client = None
        logger.info("Disconnected %s.", self.config.name)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    async def get_state(self) -> RobotState:
        """Return the most recent robot state.

        Awaits up to one control period for a fresh state; falls back to the
        cached value if none arrives in time.
        """
        period = 1.0 / self.config.control_freq
        try:
            state = await asyncio.wait_for(
                self._state_queue.get(), timeout=period * 2
            )
            self._latest_state = state
            return state
        except asyncio.TimeoutError:
            logger.debug("get_state timed out; returning cached state.")
            return self._latest_state

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    async def send_action(self, action: Action) -> None:
        """Validate and dispatch a motor command to the robot.

        Joint targets are clipped to hardware limits before transmission.
        """
        if not self._connected or self._loco_client is None:
            raise RuntimeError(f"{self.config.name} is not connected.")

        safe_action = action.clipped()
        logger.debug(
            "Sending action to %s: duration_ms=%d",
            self.config.name,
            safe_action.duration_ms,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._sdk_send_action, safe_action
        )

    def _sdk_send_action(self, action: Action) -> None:
        """Blocking SDK write — runs in executor thread."""
        # G1LocoClient exposes Move / ArmCommand depending on firmware.
        # Here we use the low-level joint position interface.
        self._loco_client.SetJointPositionCmd(
            action.joint_targets,
            duration=action.duration_ms / 1000.0,
        )

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------

    async def get_camera_frame(self) -> np.ndarray:
        """Return the latest RGB frame from the RealSense D435 as HWC uint8."""
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

    async def _ensure_realsense(self) -> None:
        if self._rs_pipeline is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._init_realsense)

    def _init_realsense(self) -> None:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        align_to = rs.stream.color
        self._rs_align = rs.align(align_to)
        self._rs_pipeline = pipeline
        logger.info("RealSense D435 initialised on %s.", self.config.name)

    def _capture_rgb(self) -> np.ndarray:
        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def _capture_depth(self) -> np.ndarray:
        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        depth_frame = aligned.get_depth_frame()
        raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        # RealSense depth units are millimetres; convert to metres.
        return raw / 1000.0

    async def get_pointcloud(self) -> np.ndarray:
        """Return a Livox MID360 LiDAR point cloud as an (N, 3) float32 array."""
        if not _O3D_AVAILABLE:
            raise RuntimeError("open3d is not installed; cannot read LiDAR data.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_lidar)

    def _capture_lidar(self) -> np.ndarray:
        # Livox SDK2 publishes on a local UDP port; open3d can read it via
        # the Livox ROS2 driver or directly. For now we use the open3d
        # LiDAR reader abstraction (requires livox_ros_driver2 bridge).
        raise NotImplementedError(
            "Livox MID360 capture requires the Livox ROS2 driver bridge. "
            "Use ROS2Bridge.get_pointcloud() or connect via the ros2_bridge module."
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
        """Unique robot identifier derived from config name and IP."""
        return f"{self.config.name}@{self.config.ip}"


# ---------------------------------------------------------------------------
# Mock bridge — for development / CI without hardware
# ---------------------------------------------------------------------------


class MockUnitreeBridge:
    """Simulated G1 bridge that produces physically plausible fake data.

    Joint positions oscillate sinusoidally, battery drains slowly, and the
    robot wanders randomly in the XY plane. All async methods complete
    instantly so unit tests run at full speed.
    """

    _OSCILLATION_AMPS = [0.3, 0.2, 0.15, 0.1, 0.25, 0.2] * 4 + [0.1, 0.05, 0.05, 0.05, 0.05]
    _OSCILLATION_FREQS = [0.5, 0.7, 0.4, 0.6, 0.5, 0.3] * 4 + [0.2, 0.1, 0.1, 0.1, 0.1]

    def __init__(self, config: G1Config) -> None:
        self.config = config
        self._connected: bool = False
        self._start_time: float = 0.0
        self._battery: float = 100.0
        self._position: list[float] = [0.0, 0.0, 0.85]  # standing height
        self._orientation: list[float] = [1.0, 0.0, 0.0, 0.0]
        self._last_heartbeat: float = 0.0
        self._assigned_id: str = f"mock-{config.name}@{config.ip}"
        self._rng = np.random.default_rng(seed=42)

        # Pad oscillation params to exactly 29 joints
        amps = (self._OSCILLATION_AMPS * 2)[:29]
        freqs = (self._OSCILLATION_FREQS * 2)[:29]
        self._amps: list[float] = amps
        self._freqs: list[float] = freqs

        logger.debug("MockUnitreeBridge created for %s.", config.name)

    async def connect(self) -> bool:
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._connected = True
        logger.info("MockUnitreeBridge: simulated connect for %s.", self.config.name)
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("MockUnitreeBridge: simulated disconnect for %s.", self.config.name)

    async def get_state(self) -> RobotState:
        self._tick()
        t = time.time() - self._start_time
        joint_pos = [
            amp * math.sin(2 * math.pi * freq * t + i)
            for i, (amp, freq) in enumerate(zip(self._amps, self._freqs))
        ]
        joint_vel = [
            amp * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t + i)
            for i, (amp, freq) in enumerate(zip(self._amps, self._freqs))
        ]
        return RobotState(
            robot_model=RobotModel.G1.value,
            battery_percent=self._battery,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_accel=[
                float(self._rng.normal(0.0, 0.05)),
                float(self._rng.normal(0.0, 0.05)),
                9.81 + float(self._rng.normal(0.0, 0.02)),
            ],
            imu_gyro=[
                float(self._rng.normal(0.0, 0.005)),
                float(self._rng.normal(0.0, 0.005)),
                float(self._rng.normal(0.0, 0.005)),
            ],
            position=list(self._position),
            orientation=list(self._orientation),
            timestamp=time.time(),
        )

    def _tick(self) -> None:
        """Advance simulated state: drain battery, move position."""
        self._last_heartbeat = time.time()
        elapsed = time.time() - self._start_time
        # Drain ~1% per minute
        self._battery = max(0.0, 100.0 - elapsed / 60.0)
        # Random walk in XY
        self._position[0] += float(self._rng.normal(0.0, 0.001))
        self._position[1] += float(self._rng.normal(0.0, 0.001))

    async def send_action(self, action: Action) -> None:
        safe = action.clipped()
        logger.debug(
            "MockUnitreeBridge: action received for %s, duration_ms=%d.",
            self.config.name,
            safe.duration_ms,
        )

    async def get_camera_frame(self) -> np.ndarray:
        """Return a synthetic 480×640 RGB frame (gradient + noise)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t = int(time.time() * 30) % 256
        frame[:, :, 0] = np.linspace(t, (t + 128) % 256, 640, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 200, 480, dtype=np.uint8).reshape(-1, 1)
        noise = self._rng.integers(0, 20, size=(480, 640, 3), dtype=np.uint8)
        return np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    async def get_depth_frame(self) -> np.ndarray:
        """Return a synthetic 480×640 depth map in metres."""
        base = np.full((480, 640), 2.0, dtype=np.float32)
        noise = self._rng.normal(0.0, 0.02, size=(480, 640)).astype(np.float32)
        return base + noise

    async def get_pointcloud(self) -> np.ndarray:
        """Return a synthetic (N, 3) float32 point cloud."""
        n = 10_000
        pts = self._rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
        # Simulate a flat floor at z≈0
        pts[:n // 4, 2] = self._rng.uniform(-0.05, 0.05, size=n // 4).astype(np.float32)
        return pts

    def is_connected(self) -> bool:
        return self._connected

    def is_alive(self) -> bool:
        return self._connected and (time.time() - self._last_heartbeat) < 2.0

    @property
    def robot_id(self) -> str:
        return self._assigned_id
