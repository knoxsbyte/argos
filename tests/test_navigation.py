"""Tests for navigation/ — zones and coverage planning."""
import numpy as np
import pytest
from argos.navigation.zones import Zone, ZoneManager
from argos.navigation.coverage import BoustrophedonPlanner, Waypoint


# ── ZoneManager ───────────────────────────────────────────────────────────────

def test_zone_partition_strips():
    mgr = ZoneManager(room_bounds=(0, 0, 6, 4))
    zones = mgr.partition(num_robots=2, strategy="strips")
    assert len(zones) == 2
    # Zones should together cover the room width
    widths = [z.bounds[2] - z.bounds[0] for z in zones]
    assert abs(sum(widths) - 6.0) < 0.01


def test_zone_partition_quadrant():
    mgr = ZoneManager(room_bounds=(0, 0, 6, 4))
    zones = mgr.partition(num_robots=4, strategy="quadrant")
    assert len(zones) == 4


def test_zone_assign_robot():
    mgr = ZoneManager(room_bounds=(0, 0, 6, 4))
    zones = mgr.partition(num_robots=2, strategy="strips")
    mgr.assign_robot(zones[0].zone_id, "G1-A")
    mgr.assign_robot(zones[1].zone_id, "G1-B")
    assert mgr.get_robot_zone("G1-A") is not None
    assert mgr.get_robot_zone("G1-B") is not None


def test_zone_coverage_tracking():
    mgr = ZoneManager(room_bounds=(0, 0, 6, 4))
    zones = mgr.partition(num_robots=1, strategy="strips")
    zid = zones[0].zone_id

    assert not mgr.is_zone_complete(zid)
    # Simulate robot cleaning the entire zone
    mgr.update_coverage(zid, robot_position=(3.0, 2.0), radius=10.0)
    assert mgr.is_zone_complete(zid, threshold=0.5)


def test_overall_coverage_zero_at_start():
    mgr = ZoneManager(room_bounds=(0, 0, 4, 4))
    mgr.partition(num_robots=2, strategy="strips")
    cov = mgr.overall_coverage()
    assert 0.0 <= cov <= 1.0


# ── BoustrophedonPlanner ──────────────────────────────────────────────────────

def test_planner_generates_waypoints():
    planner = BoustrophedonPlanner(step_size=0.5)
    zone = Zone(zone_id="z0", bounds=(0.0, 0.0, 3.0, 2.0))
    wps = planner.plan(zone, start_pos=(0.0, 0.0))
    assert len(wps) > 0
    assert all(isinstance(w, Waypoint) for w in wps)


def test_planner_covers_zone():
    planner = BoustrophedonPlanner(step_size=0.3, robot_radius=0.3)
    zone = Zone(zone_id="z1", bounds=(0.0, 0.0, 3.0, 2.0))
    wps = planner.plan(zone, start_pos=(0.0, 0.0))
    # All waypoints should be inside the zone bounds
    for wp in wps:
        assert zone.bounds[0] - 0.1 <= wp.x <= zone.bounds[2] + 0.1
        assert zone.bounds[1] - 0.1 <= wp.y <= zone.bounds[3] + 0.1


def test_planner_spiral():
    planner = BoustrophedonPlanner()
    zone = Zone(zone_id="z2", bounds=(0.0, 0.0, 2.0, 2.0))
    wps = planner.plan_spiral(zone, start_pos=(1.0, 1.0))
    assert len(wps) > 0


def test_planner_to_target_no_obstacles():
    planner = BoustrophedonPlanner()
    wps = planner.plan_to_target((0.0, 0.0), (2.0, 2.0))
    assert len(wps) >= 2
    # Last waypoint should be near target
    last = wps[-1]
    assert abs(last.x - 2.0) < 0.5
    assert abs(last.y - 2.0) < 0.5


def test_planner_duration_estimate():
    planner = BoustrophedonPlanner()
    zone = Zone(zone_id="z3", bounds=(0.0, 0.0, 4.0, 3.0))
    wps = planner.plan(zone, start_pos=(0.0, 0.0))
    duration = planner.estimate_duration(wps, speed=0.5)
    assert duration > 0.0
