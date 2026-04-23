"""
argos.perception.mapping — 3-D room mapping for ARGOS.

PointCloudProcessor handles raw Livox MID360 LiDAR scans: coordinate
transformation, voxel-downsampling, floor extraction, and occupancy grids.

RoomMapper is the high-level async interface used during robot operation;
it drives the robot through a 360° scan and builds a persistent map.

When open3d is not installed both classes degrade to lightweight mock
implementations so the rest of ARGOS can be exercised without hardware.
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional open3d import with mock fallback
# ---------------------------------------------------------------------------

try:
    import open3d as o3d  # type: ignore[import]
    _O3D_AVAILABLE = True
    logger.info("open3d detected — real point-cloud processing active.")
except ImportError:
    _O3D_AVAILABLE = False
    o3d = None  # type: ignore[assignment]
    logger.warning("open3d not installed. PointCloudProcessor running in mock mode.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3×3 rotation matrix."""
    # Normalise defensively
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _build_transform(robot_pose: tuple[float, ...]) -> np.ndarray:
    """Build a 4×4 homogeneous transform from robot_pose.

    robot_pose = (x, y, z, qx, qy, qz, qw).
    When only 3 values are given they are interpreted as (x, y, yaw) and a
    vertical translation of 0.85 m (standing height) is assumed.
    """
    if len(robot_pose) == 7:
        tx, ty, tz, qx, qy, qz, qw = robot_pose
        R = _quat_to_rotation_matrix(qx, qy, qz, qw)
    elif len(robot_pose) == 3:
        tx, ty, yaw = robot_pose
        tz = 0.85
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"robot_pose must have 3 or 7 elements, got {len(robot_pose)}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ---------------------------------------------------------------------------
# PointCloudProcessor
# ---------------------------------------------------------------------------

class PointCloudProcessor:
    """Processes Livox MID360 LiDAR point clouds into a persistent 3-D map.

    Scans are accumulated in world-frame coordinates, voxel-downsampled for
    memory efficiency, and exposed as occupancy grids and room bounds.

    When open3d is unavailable, all methods operate on plain NumPy arrays
    and return plausible mock values so downstream code stays exercisable.
    """

    def __init__(self) -> None:
        self.voxel_size: float = 0.05   # 5 cm voxels
        self._has_o3d = _O3D_AVAILABLE

        if self._has_o3d:
            self.accumulated_cloud: Any = o3d.geometry.PointCloud()
        else:
            # Fallback: accumulate points as a plain (N, 3) float32 array
            self._accumulated_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Scan processing
    # ------------------------------------------------------------------

    def process_scan(
        self,
        points: np.ndarray,
        robot_pose: tuple[float, ...],
    ) -> Any:
        """Transform raw points from robot frame to world frame.

        Parameters
        ----------
        points:
            (N, 3) float32 array of LiDAR points in robot body frame.
        robot_pose:
            Either (x, y, z, qx, qy, qz, qw) or (x, y, yaw).

        Returns
        -------
        An open3d.geometry.PointCloud (or a mock object when o3d is absent).
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {points.shape}")

        T = _build_transform(robot_pose)
        pts_h = np.hstack([points.astype(np.float64),
                           np.ones((len(points), 1), dtype=np.float64)])
        world_pts = (T @ pts_h.T).T[:, :3].astype(np.float32)

        if self._has_o3d:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(world_pts.astype(np.float64))
            return cloud
        else:
            return _MockPointCloud(world_pts)

    def accumulate(self, cloud: Any) -> None:
        """Add *cloud* to the accumulated map and voxel-downsample.

        Automatically handles both real o3d PointClouds and mock objects.
        """
        if self._has_o3d and isinstance(cloud, o3d.geometry.PointCloud):
            self.accumulated_cloud += cloud
            self.accumulated_cloud = self.accumulated_cloud.voxel_down_sample(
                self.voxel_size
            )
        else:
            new_pts = np.asarray(cloud.points, dtype=np.float32)
            if new_pts.ndim == 2 and new_pts.shape[1] == 3:
                self._accumulated_pts = np.vstack([self._accumulated_pts, new_pts])
                self._accumulated_pts = self._voxel_downsample_numpy(
                    self._accumulated_pts, self.voxel_size
                )

    # ------------------------------------------------------------------
    # Floor extraction
    # ------------------------------------------------------------------

    def extract_floor(self, cloud: Any) -> tuple[np.ndarray, float]:
        """Fit a floor plane via RANSAC and return (normal, floor_z).

        Parameters
        ----------
        cloud:
            Point cloud (o3d.PointCloud or mock).

        Returns
        -------
        (normal_vector, floor_height_z) — normal is a unit 3-D vector,
        floor_height_z is the z-coordinate of the floor in world frame.
        """
        pts = self._cloud_to_numpy(cloud)
        if pts.shape[0] < 10:
            logger.warning("extract_floor: too few points (%d), returning defaults.", pts.shape[0])
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0

        if self._has_o3d and isinstance(cloud, o3d.geometry.PointCloud):
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=500,
            )
            a, b, c, d = plane_model
            normal = np.array([a, b, c], dtype=np.float32)
            norm_len = float(np.linalg.norm(normal))
            if norm_len > 1e-6:
                normal /= norm_len
            # Floor height: -d / c when c ≠ 0
            floor_z = float(-d / c) if abs(c) > 1e-6 else 0.0
            return normal, floor_z
        else:
            return self._ransac_floor_numpy(pts)

    # ------------------------------------------------------------------
    # Room bounds
    # ------------------------------------------------------------------

    def extract_room_bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) of the accumulated map."""
        pts = self._get_accumulated_numpy()
        if pts.shape[0] == 0:
            return (-5.0, -5.0, 5.0, 5.0)

        # Use 5th/95th percentile to ignore stray outlier points
        x_min = float(np.percentile(pts[:, 0], 5))
        y_min = float(np.percentile(pts[:, 1], 5))
        x_max = float(np.percentile(pts[:, 0], 95))
        y_max = float(np.percentile(pts[:, 1], 95))
        return (x_min, y_min, x_max, y_max)

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def get_occupancy_grid(self, resolution: float = 0.1) -> np.ndarray:
        """Project the 3-D cloud to a 2-D occupancy grid.

        Returns
        -------
        HxW uint8 array:
            - 0   = free space
            - 128 = unknown
            - 255 = occupied

        The grid origin aligns with the room's (x_min, y_min) corner.
        Cell (row, col) corresponds to world position:
            x = x_min + col * resolution
            y = y_min + row * resolution
        """
        pts = self._get_accumulated_numpy()
        if pts.shape[0] == 0:
            # Return a 100×100 unknown grid as a sensible default
            return np.full((100, 100), 128, dtype=np.uint8)

        x_min, y_min, x_max, y_max = self.extract_room_bounds()
        cols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        rows = max(1, int(math.ceil((y_max - y_min) / resolution)))

        grid = np.full((rows, cols), 128, dtype=np.uint8)  # start as unknown

        # Project points: only those above floor (z > 0.05 m) count as occupied
        wall_pts = pts[pts[:, 2] > 0.05]
        if wall_pts.shape[0] > 0:
            col_idx = np.clip(
                ((wall_pts[:, 0] - x_min) / resolution).astype(int), 0, cols - 1
            )
            row_idx = np.clip(
                ((wall_pts[:, 1] - y_min) / resolution).astype(int), 0, rows - 1
            )
            grid[row_idx, col_idx] = 255  # occupied

        # Floor-level points → free
        floor_pts = pts[(pts[:, 2] >= -0.05) & (pts[:, 2] <= 0.05)]
        if floor_pts.shape[0] > 0:
            col_idx = np.clip(
                ((floor_pts[:, 0] - x_min) / resolution).astype(int), 0, cols - 1
            )
            row_idx = np.clip(
                ((floor_pts[:, 1] - y_min) / resolution).astype(int), 0, rows - 1
            )
            # Mark free only where not already occupied
            mask = grid[row_idx, col_idx] != 255
            grid[row_idx[mask], col_idx[mask]] = 0

        return grid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_map(self, path: str) -> None:
        """Save the accumulated point cloud to *path* (PCD or PLY format)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self._has_o3d:
            o3d.io.write_point_cloud(str(p), self.accumulated_cloud)
            logger.info("Map saved to %s (%d points).",
                        p, len(self.accumulated_cloud.points))
        else:
            pts = self._accumulated_pts
            np.save(str(p) + ".npy" if not str(p).endswith(".npy") else str(p), pts)
            logger.info("Mock map saved to %s (%d points).", p, len(pts))

    def load_map(self, path: str) -> None:
        """Load a previously saved point cloud from *path*."""
        p = Path(path)
        if not p.exists():
            logger.error("Map file not found: %s", p)
            return

        if self._has_o3d and str(p).endswith((".pcd", ".ply")):
            self.accumulated_cloud = o3d.io.read_point_cloud(str(p))
            logger.info("Map loaded from %s (%d points).",
                        p, len(self.accumulated_cloud.points))
        else:
            npy_path = str(p) + ".npy" if not str(p).endswith(".npy") else str(p)
            if Path(npy_path).exists():
                self._accumulated_pts = np.load(npy_path).astype(np.float32)
                logger.info("Mock map loaded from %s (%d points).",
                            npy_path, len(self._accumulated_pts))
            else:
                logger.error("Cannot load map: unsupported format or file missing.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cloud_to_numpy(self, cloud: Any) -> np.ndarray:
        if self._has_o3d and isinstance(cloud, o3d.geometry.PointCloud):
            return np.asarray(cloud.points, dtype=np.float32)
        return np.asarray(cloud.points, dtype=np.float32)

    def _get_accumulated_numpy(self) -> np.ndarray:
        if self._has_o3d:
            return np.asarray(self.accumulated_cloud.points, dtype=np.float32)
        return self._accumulated_pts

    @staticmethod
    def _voxel_downsample_numpy(pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """Simple numpy voxel downsampling: keep one point per occupied voxel."""
        if pts.shape[0] == 0:
            return pts
        voxel_idx = np.floor(pts / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_idx, axis=0, return_index=True)
        return pts[unique_indices]

    @staticmethod
    def _ransac_floor_numpy(pts: np.ndarray, iterations: int = 200,
                             threshold: float = 0.02) -> tuple[np.ndarray, float]:
        """Minimal RANSAC plane fitting without open3d."""
        best_inliers = 0
        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        best_d = 0.0
        rng = np.random.default_rng(seed=0)
        n = pts.shape[0]

        for _ in range(iterations):
            idx = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2).astype(np.float32)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-6:
                continue
            normal /= norm_len
            d = -float(np.dot(normal, p0))
            distances = np.abs(pts @ normal.reshape(3, 1) + d).ravel()
            inliers = int(np.count_nonzero(distances < threshold))
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_d = d

        floor_z = float(-best_d / best_normal[2]) if abs(best_normal[2]) > 1e-6 else 0.0
        return best_normal, floor_z


# ---------------------------------------------------------------------------
# Mock PointCloud for when open3d is absent
# ---------------------------------------------------------------------------

class _MockPointCloud:
    """Minimal stand-in for o3d.geometry.PointCloud."""

    def __init__(self, pts: np.ndarray) -> None:
        self.points = pts  # (N, 3) float32


# ---------------------------------------------------------------------------
# RoomMapper
# ---------------------------------------------------------------------------

class RoomMapper:
    """High-level async room mapper combining LiDAR and depth camera data.

    Drives the robot through a 360° scan, accumulates LiDAR point clouds,
    maintains an occupancy grid, and exposes navigation helpers.
    """

    # Angular step for the 360° scan (radians)
    _SCAN_STEP_RAD: float = math.radians(30.0)   # 12 poses × 30° = 360°
    # Rotation speed for the scanning turn (rad/s, approximate)
    _ROTATION_SPEED: float = math.radians(30.0)  # 30°/s → ~1 s per step

    def __init__(self, robot_bridge: Any) -> None:
        self.robot = robot_bridge
        self.pc_processor = PointCloudProcessor()
        self.occupancy_grid: np.ndarray | None = None
        self.room_bounds: tuple[float, float, float, float] | None = None
        self._grid_resolution: float = 0.1        # metres per cell
        self._grid_origin: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def scan_room(self, duration: float = 10.0) -> None:
        """Command the robot to rotate 360° and accumulate LiDAR scans.

        The robot performs discrete yaw steps, pausing briefly at each
        position to gather a clean LiDAR return.  Total rotation takes
        approximately *duration* seconds.

        After scanning, the occupancy grid and room bounds are recomputed.
        """
        logger.info("RoomMapper: starting 360° room scan (%.1f s).", duration)
        num_steps = 12
        step_duration = duration / num_steps

        for i in range(num_steps):
            await self.update()
            await asyncio.sleep(step_duration)

        self._recompute_map()
        logger.info(
            "RoomMapper: scan complete. Bounds=%s, free_area=%.1f m².",
            self.room_bounds,
            self.get_free_area(),
        )

    async def update(self) -> None:
        """Capture one LiDAR scan and merge it into the accumulated map.

        Call this periodically during normal operation to keep the map fresh.
        """
        try:
            raw_pts = await self.robot.get_pointcloud()  # (N, 3) float32
            state = await self.robot.get_state()
            pos = state.position           # [x, y, z]
            ori = state.orientation        # [w, x, y, z]

            # Reorder to (x, y, z, qx, qy, qz, qw)
            robot_pose = (
                pos[0], pos[1], pos[2],
                ori[1], ori[2], ori[3], ori[0],  # w→last position
            )
            cloud = self.pc_processor.process_scan(raw_pts, robot_pose)
            self.pc_processor.accumulate(cloud)
            self._recompute_map()

        except NotImplementedError:
            # Real LiDAR not wired up yet (ROS2 bridge needed)
            logger.debug("RoomMapper.update: LiDAR not available, skipping.")
        except Exception as exc:
            logger.warning("RoomMapper.update error: %s", exc)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def get_free_area(self) -> float:
        """Return the free floor area in square metres from the occupancy grid."""
        if self.occupancy_grid is None:
            return 0.0
        free_cells = int(np.count_nonzero(self.occupancy_grid == 0))
        cell_area = self._grid_resolution ** 2
        return float(free_cells) * cell_area

    def find_nearest_obstacle(
        self,
        position: tuple[float, float],
        heading: float,
        max_range: float = 10.0,
    ) -> float:
        """Return distance (metres) to the nearest obstacle in *heading* direction.

        Performs a discrete ray-cast through the occupancy grid.

        Parameters
        ----------
        position:
            Robot position ``(x, y)`` in world frame.
        heading:
            Direction to cast the ray (radians, CCW from +x axis).
        max_range:
            Maximum range to search (metres).
        """
        if self.occupancy_grid is None:
            return max_range

        grid = self.occupancy_grid
        res = self._grid_resolution
        ox, oy = self._grid_origin
        rows, cols = grid.shape
        step_size = res / 2.0  # half-cell steps for accuracy

        px, py = position
        dx = math.cos(heading) * step_size
        dy = math.sin(heading) * step_size

        num_steps = int(max_range / step_size)
        for i in range(1, num_steps + 1):
            wx = px + i * dx
            wy = py + i * dy
            col = int((wx - ox) / res)
            row = int((wy - oy) / res)
            if col < 0 or col >= cols or row < 0 or row >= rows:
                return float(i) * step_size  # out of map → treat as obstacle
            if grid[row, col] == 255:
                return float(i) * step_size

        return max_range

    def is_position_safe(
        self,
        x: float,
        y: float,
        robot_radius: float = 0.3,
    ) -> bool:
        """Return True if the circle at (x, y) with *robot_radius* is obstacle-free.

        Checks a disc of cells in the occupancy grid rather than a single
        point, making it suitable for robot footprint clearance queries.
        """
        if self.occupancy_grid is None:
            return True  # optimistic when no map is available

        grid = self.occupancy_grid
        res = self._grid_resolution
        ox, oy = self._grid_origin
        rows, cols = grid.shape

        radius_cells = int(math.ceil(robot_radius / res))
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r = cy + dr
                c = cx + dc
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    return False   # outside map bounds → unsafe
                if grid[r, c] == 255:
                    return False   # occupied → unsafe

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute_map(self) -> None:
        """Refresh the occupancy grid and room bounds from the accumulated cloud."""
        self.room_bounds = self.pc_processor.extract_room_bounds()
        self.occupancy_grid = self.pc_processor.get_occupancy_grid(
            resolution=self._grid_resolution
        )
        if self.room_bounds is not None:
            self._grid_origin = (self.room_bounds[0], self.room_bounds[1])
