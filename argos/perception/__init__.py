"""
argos.perception — Perception pipeline for the ARGOS room-cleaning framework.

Provides semantic scene understanding via three subsystems:

* **Scene graph** (``scene.py``) — live room model: surfaces, objects, priorities.
* **Detection** (``detection.py``) — YOLOv8 object detection + CV-based dirt detection.
* **Mapping** (``mapping.py``) — LiDAR-based 3-D mapping and occupancy grids.

All optional hardware dependencies (open3d, ultralytics) degrade gracefully to
mock implementations when not installed, so the module is always importable.
"""

from argos.perception.scene import (
    DetectedObject,
    ObjectCategory,
    Room,
    SceneGraph,
    Surface,
    SurfaceType,
)
from argos.perception.detection import (
    BedMakingDetector,
    DirtDetector,
    ObjectDetector,
)
from argos.perception.mapping import (
    PointCloudProcessor,
    RoomMapper,
)

__all__ = [
    # Scene graph
    "SceneGraph",
    "Room",
    "Surface",
    "DetectedObject",
    "ObjectCategory",
    "SurfaceType",
    # Detection
    "ObjectDetector",
    "DirtDetector",
    "BedMakingDetector",
    # Mapping
    "RoomMapper",
    "PointCloudProcessor",
]
