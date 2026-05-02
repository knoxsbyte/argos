"""Tests for policy/ module — all run in mock mode (no GPU needed)."""
import numpy as np
import pytest
from argos.policy.base import MockPolicy, PolicyConfig, PolicyObservation, PolicyOutput
from argos.policy.router import PolicyRouter
from argos.comm.messages import RobotState, Action


def _make_obs(instruction: str = "sweep the floor") -> PolicyObservation:
    return PolicyObservation(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        depth=None,
        robot_state=RobotState(),
        language_instruction=instruction,
        timestep=0,
    )


# ── MockPolicy ────────────────────────────────────────────────────────────────

def test_mock_policy_load():
    cfg = PolicyConfig(model_name="mock")
    policy = MockPolicy(cfg)
    policy.load()
    assert policy.is_loaded


def test_mock_policy_predict_returns_action():
    cfg = PolicyConfig(model_name="mock")
    policy = MockPolicy(cfg)
    policy.load()
    obs = _make_obs()
    out = policy.predict(obs)
    assert isinstance(out, PolicyOutput)
    assert isinstance(out.action, Action)
    assert len(out.action.joint_targets) == 29


def test_mock_policy_done_signal():
    cfg = PolicyConfig(model_name="mock", action_chunk_size=8)
    policy = MockPolicy(cfg)
    policy.load()
    obs = _make_obs()
    # Should signal done after ~10 steps
    done = False
    for _ in range(15):
        out = policy.predict(obs)
        if out.done_signal:
            done = True
            break
    assert done


def test_mock_policy_reset():
    cfg = PolicyConfig(model_name="mock")
    policy = MockPolicy(cfg)
    policy.load()
    policy.reset()  # must not raise


def test_mock_policy_latency_tracking():
    cfg = PolicyConfig(model_name="mock")
    policy = MockPolicy(cfg)
    policy.load()
    obs = _make_obs()
    for _ in range(5):
        policy.predict(obs)
    assert policy.avg_inference_ms >= 0.0


# ── PolicyRouter ──────────────────────────────────────────────────────────────

def test_router_returns_policy_for_all_tasks():
    router = PolicyRouter()
    task_types = [
        "sweep_floor", "vacuum_floor", "mop_floor", "wipe_surface",
        "wipe_window", "pick_up_object", "sort_items", "make_bed",
        "change_sheets", "move_furniture", "take_out_trash", "organize_shelf",
    ]
    for tt in task_types:
        policy = router.get_policy(tt)
        assert policy is not None
        assert policy.is_loaded


def test_router_caches_policy():
    router = PolicyRouter()
    p1 = router.get_policy("sweep_floor")
    p2 = router.get_policy("sweep_floor")
    assert p1 is p2   # same instance returned


def test_router_unknown_task_falls_back():
    router = PolicyRouter()
    policy = router.get_policy("nonexistent_task_xyz")
    assert policy is not None


def test_router_unload_all():
    router = PolicyRouter()
    router.get_policy("sweep_floor")
    router.unload_all()   # must not raise


# ── VLA / Diffusion / ACT in mock mode ───────────────────────────────────────

def test_vla_mock_mode():
    from argos.policy.vla import OpenVLAPolicy
    cfg = PolicyConfig(model_name="openvla/openvla-7b")
    policy = OpenVLAPolicy(cfg)
    policy.load()
    out = policy.predict(_make_obs("organize the shelf"))
    assert len(out.action.joint_targets) == 29


def test_diffusion_mock_mode():
    from argos.policy.diffusion import DiffusionPolicy
    cfg = PolicyConfig(model_name="diffusion")
    policy = DiffusionPolicy(cfg)
    policy.load()
    for _ in range(3):
        out = policy.predict(_make_obs("sweep the floor"))
    assert len(out.action.joint_targets) == 29
    policy.reset()


def test_act_mock_mode():
    from argos.policy.act import ACTPolicy
    cfg = PolicyConfig(model_name="act", action_chunk_size=8)
    policy = ACTPolicy(cfg)
    policy.load()
    for _ in range(3):
        out = policy.predict(_make_obs("wipe the counter"))
    assert len(out.action.joint_targets) == 29
    policy.reset()
