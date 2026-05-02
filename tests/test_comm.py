"""Tests for the comm/ layer."""
import asyncio
import pytest
from argos.comm.messages import (
    RobotState, Action, TaskMessage, CoopMessage,
    TaskStatus, CoopPhase, HeartbeatMessage,
)
from argos.comm.unitree_bridge import MockUnitreeBridge, G1Config
from argos.comm.robot_registry import RobotRegistry


# ── Messages ──────────────────────────────────────────────────────────────────

def test_robot_state_defaults():
    state = RobotState()
    assert len(state.joint_positions) == 29
    assert len(state.joint_velocities) == 29
    assert 0.0 <= state.battery_percent <= 100.0


def test_action_clipped():
    action = Action(joint_targets=[999.0] * 29)
    clipped = action.clipped()
    assert all(abs(v) <= 10.0 for v in clipped.joint_targets)


def test_task_message_transition():
    msg = TaskMessage(task_id="t1", task_type="sweep_floor",
                      assigned_robot="G1-A", status=TaskStatus.PENDING)
    updated = msg.transition(TaskStatus.ACTIVE)   # returns a copy
    assert updated.status == TaskStatus.ACTIVE
    assert msg.status == TaskStatus.PENDING        # original unchanged


def test_coop_message_phases():
    msg = CoopMessage(session_id="s1", phase=CoopPhase.PROPOSE,
                      sender_id="G1-A", receiver_id="G1-B", payload={})
    assert msg.phase == CoopPhase.PROPOSE


def test_heartbeat_from_state():
    state = RobotState(battery_percent=75.0)
    hb = HeartbeatMessage.from_state("G1-A", state)
    assert hb.robot_id == "G1-A"
    assert "battery_percent" in hb.state_summary


# ── MockUnitreeBridge ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_bridge_connect():
    cfg = G1Config(ip="192.168.1.10", name="G1-test")
    bridge = MockUnitreeBridge(cfg)
    ok = await bridge.connect()
    assert ok is True
    assert bridge.is_connected()
    await bridge.disconnect()
    assert not bridge.is_connected()


@pytest.mark.asyncio
async def test_mock_bridge_state():
    cfg = G1Config(ip="192.168.1.10", name="G1-test")
    bridge = MockUnitreeBridge(cfg)
    await bridge.connect()
    state = await bridge.get_state()
    assert isinstance(state, RobotState)
    assert len(state.joint_positions) == 29
    await bridge.disconnect()


@pytest.mark.asyncio
async def test_mock_bridge_send_action():
    cfg = G1Config(ip="192.168.1.10", name="G1-test")
    bridge = MockUnitreeBridge(cfg)
    await bridge.connect()
    action = Action(joint_targets=[0.0] * 29)
    await bridge.send_action(action)   # must not raise
    await bridge.disconnect()


@pytest.mark.asyncio
async def test_mock_bridge_camera():
    cfg = G1Config(ip="192.168.1.10", name="G1-test")
    bridge = MockUnitreeBridge(cfg)
    await bridge.connect()
    frame = await bridge.get_camera_frame()
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    await bridge.disconnect()


# ── RobotRegistry ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_register_deregister():
    registry = RobotRegistry()
    cfg = G1Config(ip="192.168.1.20", name="G1-reg")
    bridge = MockUnitreeBridge(cfg)
    await bridge.connect()

    robot_id = await registry.register(bridge)
    assert robot_id is not None
    assert registry.get(robot_id) is bridge
    assert bridge in registry.list_all()

    await registry.deregister(robot_id)
    assert registry.get(robot_id) is None


@pytest.mark.asyncio
async def test_registry_list_available():
    registry = RobotRegistry()
    cfg = G1Config(ip="192.168.1.21", name="G1-avail")
    bridge = MockUnitreeBridge(cfg)
    await bridge.connect()
    rid = await registry.register(bridge)

    available = registry.list_available()
    assert bridge in available

    registry.set_busy(rid, True)
    available_after = registry.list_available()
    assert bridge not in available_after

    await registry.deregister(rid)
