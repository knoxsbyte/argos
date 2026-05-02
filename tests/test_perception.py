"""Tests for perception/ module."""
import numpy as np
import pytest
from argos.perception.scene import (
    SceneGraph, Room, Surface, DetectedObject,
    ObjectCategory, SurfaceType,
)
from argos.perception.detection import ObjectDetector, DirtDetector, BedMakingDetector
from argos.perception.mapping import PointCloudProcessor


# ── SceneGraph ────────────────────────────────────────────────────────────────

def test_scene_graph_update():
    graph = SceneGraph()
    obj = DetectedObject(
        object_id="o1",
        category=ObjectCategory.CLUTTER,
        label="bottle",
        position=np.array([1.0, 2.0, 0.0]),
        bounding_box=(10, 20, 50, 80),
        confidence=0.85,
        needs_pickup=True,
    )
    surface = Surface(
        surface_id="s1",
        surface_type=SurfaceType.FLOOR,
        bounds=np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]]),
        dirty_fraction=0.3,
    )
    graph.update_from_perception([obj], [surface], (0.0, 0.0, 0.0))
    assert len(graph.get_cluttered_objects()) >= 1
    assert len(graph.get_dirty_surfaces()) >= 1


def test_scene_graph_to_dict():
    graph = SceneGraph()
    d = graph.to_dict()
    assert isinstance(d, dict)


def test_scene_graph_cleaning_priority():
    graph = SceneGraph()
    surface = Surface(
        surface_id="s2",
        surface_type=SurfaceType.COUNTER,
        bounds=np.zeros((4, 3)),
        dirty_fraction=0.8,
    )
    graph.update_from_perception([], [surface], (0.0, 0.0, 0.0))
    priorities = graph.get_cleaning_priority_list()
    assert isinstance(priorities, list)


def test_scene_graph_task_locations():
    graph = SceneGraph()
    locs = graph.get_task_locations("sweep_floor")
    assert isinstance(locs, list)


# ── ObjectDetector ────────────────────────────────────────────────────────────

def test_object_detector_mock():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)
    assert isinstance(results, list)


def test_object_detector_with_depth():
    detector = ObjectDetector()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    intrinsics = {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
    results = detector.detect_with_depth(rgb, depth, intrinsics)
    assert isinstance(results, list)


# ── DirtDetector ──────────────────────────────────────────────────────────────

def test_dirt_detector_clean_frame():
    detector = DirtDetector()
    frame = np.ones((224, 224, 3), dtype=np.uint8) * 200  # uniform gray
    fraction = detector.estimate_dirty_fraction(frame)
    assert 0.0 <= fraction <= 1.0


def test_dirt_detector_dirty_patch():
    detector = DirtDetector()
    frame = np.ones((224, 224, 3), dtype=np.uint8) * 200
    # Add a colored patch that stands out
    frame[80:100, 80:100] = [180, 20, 20]
    fraction = detector.estimate_dirty_fraction(frame)
    assert 0.0 <= fraction <= 1.0


def test_dirt_detector_mask_shape():
    detector = DirtDetector()
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    mask = detector.detect_dirt(frame)
    assert mask.shape == (224, 224)


def test_clutter_on_floor_detection():
    detector = DirtDetector()
    depth = np.ones((224, 224), dtype=np.float32) * 0.8  # floor at 0.8m
    depth[100:120, 100:120] = 0.6   # object 20cm above floor
    mask = detector.detect_clutter_on_floor(depth, floor_height=0.8)
    assert mask.shape == (224, 224)


# ── BedMakingDetector ─────────────────────────────────────────────────────────

def test_bed_detector_assess():
    detector = BedMakingDetector()
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    depth = np.ones((480, 640), dtype=np.float32)
    result = detector.assess_bed_state(frame, depth)
    assert "is_made" in result
    assert "wrinkle_severity" in result
    assert "sheet_coverage" in result


# ── PointCloudProcessor ───────────────────────────────────────────────────────

def test_pointcloud_process_scan():
    processor = PointCloudProcessor()
    points = np.random.uniform(-3, 3, (500, 3)).astype(np.float32)
    robot_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    cloud = processor.process_scan(points, robot_pose)
    assert cloud is not None


def test_pointcloud_occupancy_grid():
    processor = PointCloudProcessor()
    points = np.random.uniform(-3, 3, (500, 3)).astype(np.float32)
    points[:, 2] = np.random.uniform(0.1, 2.0, 500)  # above floor
    robot_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    processor.process_scan(points, robot_pose)
    processor.accumulate(processor.process_scan(points, robot_pose))
    grid = processor.get_occupancy_grid()
    assert grid.ndim == 2
    assert grid.dtype == np.uint8


def test_pointcloud_room_bounds():
    processor = PointCloudProcessor()
    points = np.array([
        [-3, -2, 0.1], [3, -2, 0.1], [3, 2, 0.1], [-3, 2, 0.1],
    ], dtype=np.float32)
    robot_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    processor.process_scan(points, robot_pose)
    processor.accumulate(processor.process_scan(points, robot_pose))
    bounds = processor.extract_room_bounds()
    assert len(bounds) == 4
