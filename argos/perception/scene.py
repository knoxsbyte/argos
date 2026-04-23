"""
argos.perception.scene — Semantic scene graph for ARGOS room-cleaning framework.

Maintains a live model of the environment: rooms, surfaces, and detected objects.
All coordinates are in the world frame (metres). The scene graph is the single
source of truth consumed by the task planner and the LLM reasoning layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

STALE_OBJECT_TIMEOUT: float = 30.0   # seconds before an unseen object is pruned
DIRTY_SURFACE_THRESHOLD: float = 0.1  # fraction above which a surface is "dirty"
PICKUP_PRIORITY_WEIGHT: float = 2.0   # extra score for objects needing pickup
DIRTY_PRIORITY_WEIGHT: float = 1.5    # extra score for dirty surfaces


class SurfaceType(Enum):
    FLOOR = "floor"
    COUNTER = "counter"
    TABLE = "table"
    BED = "bed"
    SHELF = "shelf"
    WINDOW = "window"
    WALL = "wall"


class ObjectCategory(Enum):
    CLUTTER = "clutter"
    TRASH = "trash"
    FURNITURE = "furniture"
    CLEANING_TOOL = "cleaning_tool"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DetectedObject:
    """A semantically-labelled object observed in the scene."""

    object_id: str
    category: ObjectCategory
    label: str                           # e.g. "plastic_bottle", "sock"
    position: np.ndarray                 # 3-D position (x, y, z) in world frame
    bounding_box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in image pixels
    confidence: float
    needs_pickup: bool = False
    needs_cleaning: bool = False

    # Internal book-keeping — not exposed in to_dict().
    _last_seen: float = field(default_factory=time.time, repr=False, compare=False)

    def touch(self) -> None:
        """Refresh the last-seen timestamp to prevent stale removal."""
        self._last_seen = time.time()

    def is_stale(self, timeout: float = STALE_OBJECT_TIMEOUT) -> bool:
        """Return True if the object has not been seen within *timeout* seconds."""
        return (time.time() - self._last_seen) > timeout

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category.value,
            "label": self.label,
            "position": self.position.tolist(),
            "bounding_box": list(self.bounding_box),
            "confidence": round(self.confidence, 3),
            "needs_pickup": self.needs_pickup,
            "needs_cleaning": self.needs_cleaning,
        }


@dataclass
class Surface:
    """A planar surface (floor, table, counter …) that can accumulate dirt."""

    surface_id: str
    surface_type: SurfaceType
    bounds: np.ndarray                        # (N, 3) corner points in world frame
    dirty_regions: list[np.ndarray] = field(default_factory=list)  # list of 3-D pts
    dirty_fraction: float = 0.0              # 0.0–1.0

    def centroid(self) -> np.ndarray:
        """Return the mean position of the surface corners."""
        return self.bounds.mean(axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "surface_type": self.surface_type.value,
            "bounds": self.bounds.tolist(),
            "dirty_fraction": round(self.dirty_fraction, 3),
            "dirty_region_count": len(self.dirty_regions),
        }


@dataclass
class Room:
    """Spatial container representing one room of the environment."""

    room_id: str
    bounds: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) metres
    floor_area: float                            # square metres
    surfaces: dict[str, Surface] = field(default_factory=dict)
    objects: dict[str, DetectedObject] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "bounds": list(self.bounds),
            "floor_area": round(self.floor_area, 2),
            "surfaces": {sid: s.to_dict() for sid, s in self.surfaces.items()},
            "objects": {oid: o.to_dict() for oid, o in self.objects.items()},
        }


# ---------------------------------------------------------------------------
# Scene graph
# ---------------------------------------------------------------------------


class SceneGraph:
    """Full semantic model of the environment.

    Objects are identified by *object_id* strings.  New detections are merged
    by matching label + approximate 3-D position (within 0.5 m).  Objects not
    seen for more than STALE_OBJECT_TIMEOUT seconds are pruned automatically.
    """

    MERGE_DISTANCE: float = 0.5  # metres — detections closer than this are merged

    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        self.current_room: Room | None = None
        self._update_count: int = 0
        self._object_counter: int = 0
        self._surface_counter: int = 0

    # ------------------------------------------------------------------
    # Primary update entry-point
    # ------------------------------------------------------------------

    def update_from_perception(
        self,
        objects: list[DetectedObject],
        surfaces: list[Surface],
        robot_pose: tuple[float, float, float],
    ) -> None:
        """Merge new detections into the scene graph and remove stale objects.

        Parameters
        ----------
        objects:
            Freshly detected objects from the perception pipeline (2-D bbox +
            3-D position already expressed in world frame).
        surfaces:
            Updated surface state (dirty regions, dirty_fraction).
        robot_pose:
            Current robot position ``(x, y, yaw)`` used to create a room on
            first call if none exists yet.
        """
        self._ensure_room(robot_pose)
        room = self.current_room
        assert room is not None  # guaranteed by _ensure_room

        # --- merge / insert objects -----------------------------------------
        for det in objects:
            matched_id = self._find_matching_object(room, det)
            if matched_id is not None:
                existing = room.objects[matched_id]
                # Update position with running average (EMA α = 0.3)
                existing.position = 0.7 * existing.position + 0.3 * det.position
                existing.confidence = max(existing.confidence, det.confidence)
                existing.needs_pickup = det.needs_pickup
                existing.needs_cleaning = det.needs_cleaning
                existing.touch()
            else:
                self._object_counter += 1
                new_id = f"obj_{self._object_counter:04d}"
                det.object_id = new_id
                det._last_seen = time.time()
                room.objects[new_id] = det

        # --- merge / insert surfaces ----------------------------------------
        for surf in surfaces:
            if surf.surface_id in room.surfaces:
                existing_surf = room.surfaces[surf.surface_id]
                existing_surf.dirty_fraction = surf.dirty_fraction
                existing_surf.dirty_regions = surf.dirty_regions
            else:
                self._surface_counter += 1
                room.surfaces[surf.surface_id] = surf

        # --- prune stale objects --------------------------------------------
        stale_ids = [
            oid for oid, obj in room.objects.items() if obj.is_stale()
        ]
        for oid in stale_ids:
            logger.debug("SceneGraph: pruning stale object %s (%s)", oid,
                         room.objects[oid].label)
            del room.objects[oid]

        self._update_count += 1
        logger.debug(
            "SceneGraph update #%d: %d objects, %d surfaces, pruned %d stale",
            self._update_count,
            len(room.objects),
            len(room.surfaces),
            len(stale_ids),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_dirty_surfaces(self) -> list[Surface]:
        """Return all surfaces whose dirty_fraction exceeds the threshold."""
        if self.current_room is None:
            return []
        return [
            s for s in self.current_room.surfaces.values()
            if s.dirty_fraction > DIRTY_SURFACE_THRESHOLD
        ]

    def get_cluttered_objects(self) -> list[DetectedObject]:
        """Return all objects that need pickup."""
        if self.current_room is None:
            return []
        return [
            o for o in self.current_room.objects.values()
            if o.needs_pickup
        ]

    def get_cleaning_priority_list(self) -> list[tuple[str, float]]:
        """Return a priority-sorted list of ``(task_description, priority_score)``.

        Priority scoring:
        - Dirty surface: ``dirty_fraction * DIRTY_PRIORITY_WEIGHT``
        - Object needing pickup: ``confidence * PICKUP_PRIORITY_WEIGHT``
        """
        tasks: list[tuple[str, float]] = []

        if self.current_room is None:
            return tasks

        # Dirty surfaces
        for surf in self.get_dirty_surfaces():
            score = surf.dirty_fraction * DIRTY_PRIORITY_WEIGHT
            centroid = surf.centroid()
            desc = (
                f"clean_{surf.surface_type.value}:{surf.surface_id} "
                f"@ ({centroid[0]:.2f},{centroid[1]:.2f})"
            )
            tasks.append((desc, round(score, 4)))

        # Clutter / trash pickup
        for obj in self.get_cluttered_objects():
            score = obj.confidence * PICKUP_PRIORITY_WEIGHT
            desc = (
                f"pickup_{obj.category.value}:{obj.label}:{obj.object_id} "
                f"@ ({obj.position[0]:.2f},{obj.position[1]:.2f})"
            )
            tasks.append((desc, round(score, 4)))

        # Objects needing cleaning (e.g. stained furniture)
        if self.current_room:
            for obj in self.current_room.objects.values():
                if obj.needs_cleaning and not obj.needs_pickup:
                    score = obj.confidence * 1.2
                    desc = (
                        f"clean_object:{obj.label}:{obj.object_id} "
                        f"@ ({obj.position[0]:.2f},{obj.position[1]:.2f})"
                    )
                    tasks.append((desc, round(score, 4)))

        tasks.sort(key=lambda t: t[1], reverse=True)
        return tasks

    def get_task_locations(self, task_type: str) -> list[tuple[float, float]]:
        """Return 2-D ``(x, y)`` positions relevant to *task_type*.

        Recognised task_type values:
        - ``"pickup"`` — positions of objects needing pickup
        - ``"clean_surface"`` — centroids of dirty surfaces
        - ``"furniture"`` — positions of furniture objects
        - ``"all_objects"`` — positions of every tracked object
        """
        if self.current_room is None:
            return []

        locations: list[tuple[float, float]] = []

        if task_type == "pickup":
            for obj in self.current_room.objects.values():
                if obj.needs_pickup:
                    locations.append((float(obj.position[0]), float(obj.position[1])))

        elif task_type == "clean_surface":
            for surf in self.get_dirty_surfaces():
                c = surf.centroid()
                locations.append((float(c[0]), float(c[1])))

        elif task_type == "furniture":
            for obj in self.current_room.objects.values():
                if obj.category == ObjectCategory.FURNITURE:
                    locations.append((float(obj.position[0]), float(obj.position[1])))

        elif task_type == "all_objects":
            for obj in self.current_room.objects.values():
                locations.append((float(obj.position[0]), float(obj.position[1])))

        else:
            logger.warning("get_task_locations: unknown task_type %r", task_type)

        return locations

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for the LLM planner."""
        return {
            "update_count": self._update_count,
            "current_room_id": self.current_room.room_id if self.current_room else None,
            "rooms": {rid: r.to_dict() for rid, r in self.rooms.items()},
            "summary": {
                "dirty_surfaces": len(self.get_dirty_surfaces()),
                "objects_needing_pickup": len(self.get_cluttered_objects()),
                "priority_tasks": self.get_cleaning_priority_list()[:5],
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_room(self, robot_pose: tuple[float, float, float]) -> None:
        """Create a default room if none exists yet."""
        if not self.rooms:
            room = Room(
                room_id="room_000",
                bounds=(-10.0, -10.0, 10.0, 10.0),
                floor_area=400.0,
            )
            self.rooms["room_000"] = room
            self.current_room = room
            logger.info("SceneGraph: created default room_000.")
        elif self.current_room is None:
            self.current_room = next(iter(self.rooms.values()))

    def _find_matching_object(
        self, room: Room, det: DetectedObject
    ) -> str | None:
        """Return the ID of an existing object that matches *det*, or None."""
        for oid, obj in room.objects.items():
            if obj.label != det.label:
                continue
            dist = float(np.linalg.norm(obj.position - det.position))
            if dist < self.MERGE_DISTANCE:
                return oid
        return None
