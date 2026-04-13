"""
Optional ROS2 interface for the ARGOS communication layer.

ROS2Bridge presents the same interface as UnitreeBridge but reads robot state
and sensor data from ROS2 topics published by the Unitree ROS2 driver.

If rclpy is not installed, importing this module raises ImportError with a
helpful installation message.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

_rclpy_err_msg: str = ""

try:
    import rclpy  # type: ignore[import]
    from rclpy.node import Node  # type: ignore[import]
    from rclpy.qos import (  # type: ignore[import]
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image, PointCloud2  # type: ignore[import]

    try:
        from unitree_ros2_msgs.msg import LowState as UnitreeLowState  # type: ignore[import]
        _UNITREE_MSGS = True
    except ImportError:
        _UNITREE_MSGS = False

    _RCLPY_AVAILABLE = True
except ImportError as _rclpy_err:
    _RCLPY_AVAILABLE = False
    _rclpy_err_msg = str(_rclpy_err)
    # Provide stub names so class bodies that reference these at definition
    # time do not raise NameError when rclpy is absent.
    Node = object  # type: ignore[assignment,misc]
    QoSProfile = None  # type: ignore[assignment]
    QoSDurabilityPolicy = None  # type: ignore[assignment]
    QoSHistoryPolicy = None  # type: ignore[assignment]
    QoSReliabilityPolicy = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    PointCloud2 = None  # type: ignore[assignment]
    UnitreeLowState = None  # type: ignore[assignment]
    _UNITREE_MSGS = False

from argos.comm.messages import Action, RobotState
from argos.comm.unitree_bridge import G1Config, _clip_joints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic names (override via subclass or config if needed)
# ---------------------------------------------------------------------------

TOPIC_STATE = "/g1/state"
TOPIC_COLOR_IMAGE = "/g1/camera/color/image_raw"
TOPIC_LIDAR = "/g1/lidar/points"


def _require_rclpy() -> None:
    """Raise ImportError if rclpy is not available."""
    if not _RCLPY_AVAILABLE:
        raise ImportError(
            "rclpy is not installed. Install it with:\n"
            "  pip install argos[ros2]\n"
            "or follow the ROS2 Humble/Iron installation guide at "
            "https://docs.ros.org/en/humble/Installation.html\n"
            f"(Original error: {_rclpy_err_msg})"
        ) from None


# ---------------------------------------------------------------------------
# ROS2 node that spins in a background thread
# ---------------------------------------------------------------------------


class _ARGOSRobotNode(Node):  # type: ignore[misc]
    """Internal ROS2 node that subscribes to robot topics and caches data."""

    def __init__(self, config: G1Config, state_queue: asyncio.Queue[RobotState]) -> None:
        super().__init__(f"argos_{config.name.lower().replace('-', '_')}_bridge")
        self._config = config
        self._state_queue = state_queue
        self._loop: asyncio.AbstractEventLoop | None = None

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # State subscriber
        if _UNITREE_MSGS:
            self._state_sub = self.create_subscription(
                UnitreeLowState, TOPIC_STATE, self._on_state, sensor_qos
            )
        else:
            logger.warning(
                "unitree_ros2_msgs not found; /g1/state subscription skipped."
            )
            self._state_sub = None

        # Camera subscriber
        self._latest_image: np.ndarray | None = None
        self._image_sub = self.create_subscription(
            Image, TOPIC_COLOR_IMAGE, self._on_image, sensor_qos
        )

        # LiDAR subscriber
        self._latest_cloud: np.ndarray | None = None
        self._cloud_sub = self.create_subscription(
            PointCloud2, TOPIC_LIDAR, self._on_cloud, sensor_qos
        )

        self._last_state_time: float = 0.0

    # ------------------------------------------------------------------
    # Callbacks (ROS2 thread)
    # ------------------------------------------------------------------

    def _on_state(self, msg: Any) -> None:
        """Parse a Unitree LowState ROS2 message into RobotState."""
        try:
            joint_pos = [msg.motor_state[i].q for i in range(29)]
            joint_vel = [msg.motor_state[i].dq for i in range(29)]
            imu = msg.imu_state
            state = RobotState(
                battery_percent=float(getattr(msg, "battery_percent", 100.0)),
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
            self._last_state_time = time.time()

            if self._loop is not None and not self._loop.is_closed():
                # Thread-safe bridge to the asyncio event loop.
                if self._state_queue.full():
                    try:
                        self._state_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                asyncio.run_coroutine_threadsafe(
                    self._state_queue.put(state), self._loop
                )
        except Exception as exc:
            logger.warning("Error parsing ROS2 state message: %s", exc)

    def _on_image(self, msg: Any) -> None:
        """Convert a sensor_msgs/Image to an HWC uint8 numpy array."""
        try:
            height = msg.height
            width = msg.width
            encoding = msg.encoding.lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if encoding in ("bgr8", "rgb8"):
                frame = data.reshape((height, width, 3))
            elif encoding == "bgra8":
                frame = data.reshape((height, width, 4))[:, :, :3]
            elif encoding == "mono8":
                frame = np.stack([data.reshape((height, width))] * 3, axis=-1)
            else:
                logger.warning("Unsupported image encoding: %s", encoding)
                return

            self._latest_image = frame
        except Exception as exc:
            logger.warning("Error parsing ROS2 image: %s", exc)

    def _on_cloud(self, msg: Any) -> None:
        """Convert a sensor_msgs/PointCloud2 to an (N, 3) float32 array."""
        try:
            import struct

            point_step = msg.point_step
            data = bytes(msg.data)
            n_points = len(data) // point_step
            # Standard XYZ fields are at offsets 0, 4, 8 (float32 each).
            pts = np.frombuffer(
                data,
                dtype=np.float32,
                count=n_points * (point_step // 4),
            ).reshape(n_points, point_step // 4)
            # Extract XYZ columns (fields 0,1,2 → offsets 0,4,8 bytes → indices 0,1,2).
            self._latest_cloud = pts[:, :3].copy()
        except Exception as exc:
            logger.warning("Error parsing ROS2 PointCloud2: %s", exc)


# ---------------------------------------------------------------------------
# Public ROS2Bridge
# ---------------------------------------------------------------------------


class ROS2Bridge:
    """UnitreeBridge-compatible interface that reads from ROS2 topics.

    Raises ImportError on construction if rclpy is not installed.
    """

    def __init__(self, config: G1Config) -> None:
        _require_rclpy()
        self.config = config
        self._connected: bool = False
        self._state_queue: asyncio.Queue[RobotState] = asyncio.Queue(maxsize=10)
        self._latest_state: RobotState = RobotState()
        self._node: _ARGOSRobotNode | None = None
        self._spin_thread: threading.Thread | None = None
        self._executor: rclpy.executors.SingleThreadedExecutor | None = None

        logger.debug("ROS2Bridge created for %s.", config.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialise rclpy, create the subscriber node, and start spinning."""
        try:
            if not rclpy.ok():
                rclpy.init()

            loop = asyncio.get_running_loop()
            self._node = _ARGOSRobotNode(self.config, self._state_queue)
            self._node._loop = loop

            self._executor = rclpy.executors.SingleThreadedExecutor()
            self._executor.add_node(self._node)

            self._spin_thread = threading.Thread(
                target=self._spin_forever, daemon=True, name="ros2-spin"
            )
            self._spin_thread.start()

            # Wait briefly for the first state message.
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._state_queue.get()),
                    timeout=self.config.connect_timeout,
                )
                self._connected = True
                logger.info("ROS2Bridge connected for %s.", self.config.name)
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    "No state message received on %s within %.1fs — "
                    "connected anyway (topics may not be publishing yet).",
                    TOPIC_STATE,
                    self.config.connect_timeout,
                )
                self._connected = True
                return True

        except Exception as exc:
            logger.exception("Failed to initialise ROS2Bridge: %s", exc)
            return False

    def _spin_forever(self) -> None:
        """Thread target: spin the ROS2 executor until shutdown."""
        try:
            self._executor.spin()
        except Exception as exc:
            logger.error("ROS2 spin thread exited with error: %s", exc)

    async def disconnect(self) -> None:
        self._connected = False
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        logger.info("ROS2Bridge disconnected for %s.", self.config.name)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    async def get_state(self) -> RobotState:
        period = 1.0 / self.config.control_freq
        try:
            state = await asyncio.wait_for(
                self._state_queue.get(), timeout=period * 2
            )
            self._latest_state = state
            return state
        except asyncio.TimeoutError:
            logger.debug("ROS2Bridge: get_state timed out; returning cached state.")
            return self._latest_state

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def send_action(self, action: Action) -> None:
        """Publish an action as a ROS2 joint command.

        Currently logs the command; full implementation requires a
        /g1/joint_command publisher with the appropriate message type.
        """
        if not self._connected:
            raise RuntimeError(f"{self.config.name} ROS2Bridge is not connected.")
        clipped = _clip_joints(action.joint_targets)
        logger.info(
            "ROS2Bridge send_action for %s: %d joints, duration_ms=%d.",
            self.config.name,
            len(clipped),
            action.duration_ms,
        )
        # TODO: publish to /g1/joint_command when message type is available.

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    async def get_camera_frame(self) -> np.ndarray:
        """Return the latest RGB frame received on the color image topic."""
        if self._node is None or self._node._latest_image is None:
            raise RuntimeError(
                f"No image available yet on {TOPIC_COLOR_IMAGE}. "
                "Ensure the camera driver is publishing."
            )
        return self._node._latest_image.copy()

    async def get_depth_frame(self) -> np.ndarray:
        """Depth frames are not available via the standard color topic.

        Override TOPIC_COLOR_IMAGE or add a dedicated depth subscriber to
        support depth frames over ROS2.
        """
        raise NotImplementedError(
            "Depth frames require a separate /g1/camera/depth/image_raw subscriber. "
            "Subscribe to that topic and expose it via a subclass."
        )

    async def get_pointcloud(self) -> np.ndarray:
        """Return the latest LiDAR point cloud as an (N, 3) float32 array."""
        if self._node is None or self._node._latest_cloud is None:
            raise RuntimeError(
                f"No point cloud available yet on {TOPIC_LIDAR}. "
                "Ensure the LiDAR driver is publishing."
            )
        return self._node._latest_cloud.copy()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def is_alive(self) -> bool:
        if self._node is None:
            return False
        return self._connected and (time.time() - self._node._last_state_time) < 2.0

    @property
    def robot_id(self) -> str:
        return f"ros2-{self.config.name}@{self.config.ip}"
