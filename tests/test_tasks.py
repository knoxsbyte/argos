"""Tests for tasks/ module."""
import asyncio
import pytest
from argos.tasks.base import TaskStatus, TaskResult
from argos.tasks.library import TaskLibrary
from argos.tasks.solo import SweepFloorTask, WipeSurfaceTask, PickUpObjectTask
from argos.tasks.cooperative import MakeBedTask
from argos.comm.unitree_bridge import MockUnitreeBridge, G1Config


# ── TaskLibrary ───────────────────────────────────────────────────────────────

def test_library_lists_tasks():
    lib = TaskLibrary.get_instance()
    types = lib.list_types()
    assert len(types) >= 8
    assert "sweep_floor" in types
    assert "make_bed" in types


def test_library_create_solo():
    lib = TaskLibrary.get_instance()
    task = lib.create("sweep_floor", "t-sweep-1", {"zone": "A"})
    assert task.task_type == "sweep_floor"
    assert task.min_robots == 1
    assert not task.cooperative


def test_library_create_cooperative():
    lib = TaskLibrary.get_instance()
    task = lib.create("make_bed", "t-bed-1", {})
    assert task.cooperative is True
    assert task.min_robots >= 2


def test_library_get_config():
    lib = TaskLibrary.get_instance()
    cfg = lib.get_config("sweep_floor")
    assert isinstance(cfg, dict)


# ── SweepFloorTask ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sweep_floor_executes():
    cfg = G1Config(ip="10.1.0.1", name="G1-sweep")
    robot = MockUnitreeBridge(cfg)
    await robot.connect()

    task = SweepFloorTask("sweep-1", {"zone_bounds": (0.0, 0.0, 2.0, 2.0)})
    result = await task.execute([robot])

    assert isinstance(result, TaskResult)
    assert result.success is True
    assert result.duration_seconds > 0

    await robot.disconnect()


@pytest.mark.asyncio
async def test_sweep_floor_cancel():
    cfg = G1Config(ip="10.1.0.2", name="G1-cancel")
    robot = MockUnitreeBridge(cfg)
    await robot.connect()

    task = SweepFloorTask("sweep-cancel", {"zone_bounds": (0.0, 0.0, 5.0, 5.0)})

    async def cancel_soon():
        await asyncio.sleep(0.05)
        await task.cancel()

    asyncio.create_task(cancel_soon())
    result = await task.execute([robot])

    assert result.success is False
    assert task.status in (TaskStatus.CANCELLED, TaskStatus.FAILED)

    await robot.disconnect()


@pytest.mark.asyncio
async def test_wipe_surface_executes():
    cfg = G1Config(ip="10.1.0.3", name="G1-wipe")
    robot = MockUnitreeBridge(cfg)
    await robot.connect()

    task = WipeSurfaceTask("wipe-1", {"target": "counter", "position": (1.0, 0.5)})
    result = await task.execute([robot])

    assert isinstance(result, TaskResult)
    await robot.disconnect()


# ── MakeBedTask (cooperative) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_make_bed_two_robots():
    cfg_a = G1Config(ip="10.2.0.1", name="G1-bed-A")
    cfg_b = G1Config(ip="10.2.0.2", name="G1-bed-B")
    robot_a = MockUnitreeBridge(cfg_a)
    robot_b = MockUnitreeBridge(cfg_b)
    await robot_a.connect()
    await robot_b.connect()

    task = MakeBedTask("bed-1", {"bed_pos": [2.0, 1.5]})
    result = await task.execute([robot_a, robot_b])

    assert isinstance(result, TaskResult)
    assert result.success is True

    await robot_a.disconnect()
    await robot_b.disconnect()
