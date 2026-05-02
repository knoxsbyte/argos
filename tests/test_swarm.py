"""Tests for the swarm/ coordination layer."""
import asyncio
import pytest
from argos.swarm.dependency import TaskDAG, TaskNode
from argos.swarm.allocator import AuctionAllocator
from argos.swarm.planner import LLMTaskPlanner
from argos.comm.messages import RobotState
from argos.comm.unitree_bridge import MockUnitreeBridge, G1Config


# ── TaskDAG ───────────────────────────────────────────────────────────────────

def test_dag_add_and_ready():
    dag = TaskDAG()
    t1 = TaskNode("t1", "sweep_floor", {})
    t2 = TaskNode("t2", "wipe_surface", {})
    dag.add_task(t1)
    dag.add_task(t2)
    dag.add_dependency("t2", "t1")   # t2 depends on t1

    ready = dag.get_ready_tasks()
    assert any(t.task_id == "t1" for t in ready)
    assert not any(t.task_id == "t2" for t in ready)


def test_dag_mark_done_unblocks():
    dag = TaskDAG()
    t1 = TaskNode("t1", "sweep_floor", {})
    t2 = TaskNode("t2", "wipe_surface", {})
    dag.add_task(t1)
    dag.add_task(t2)
    dag.add_dependency("t2", "t1")

    dag.mark_done("t1")
    ready = dag.get_ready_tasks()
    assert any(t.task_id == "t2" for t in ready)


def test_dag_complete_when_all_done():
    dag = TaskDAG()
    t1 = TaskNode("t1", "sweep_floor", {})
    dag.add_task(t1)
    assert not dag.is_complete()
    dag.mark_done("t1")
    assert dag.is_complete()


def test_dag_serialization():
    dag = TaskDAG()
    t1 = TaskNode("t1", "sweep_floor", {"zone": "A"})
    dag.add_task(t1)
    data = dag.to_dict()
    assert "tasks" in data
    assert any(t["task_id"] == "t1" for t in data["tasks"])


def test_dag_critical_path():
    dag = TaskDAG()
    for i in range(3):
        dag.add_task(TaskNode(f"t{i}", "sweep_floor", {}, duration_estimate=60))
    dag.add_dependency("t1", "t0")
    dag.add_dependency("t2", "t1")
    path = dag.get_critical_path()
    assert len(path) >= 1


def test_dag_cycle_detection():
    dag = TaskDAG()
    dag.add_task(TaskNode("t1", "sweep_floor", {}))
    dag.add_task(TaskNode("t2", "wipe_surface", {}))
    dag.add_dependency("t2", "t1")
    with pytest.raises(Exception):
        dag.add_dependency("t1", "t2")   # would create cycle


# ── AuctionAllocator ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allocator_assigns_solo_tasks():
    cfg_a = G1Config(ip="10.0.0.1", name="G1-A")
    cfg_b = G1Config(ip="10.0.0.2", name="G1-B")
    robot_a = MockUnitreeBridge(cfg_a)
    robot_b = MockUnitreeBridge(cfg_b)
    await robot_a.connect()
    await robot_b.connect()

    allocator = AuctionAllocator([robot_a, robot_b])

    dag = TaskDAG()
    dag.add_task(TaskNode("t1", "sweep_floor", {"zone": "A"}, min_robots=1))
    dag.add_task(TaskNode("t2", "wipe_surface", {"zone": "B"}, min_robots=1))

    states = {
        robot_a.robot_id: await robot_a.get_state(),
        robot_b.robot_id: await robot_b.get_state(),
    }
    assignments = allocator.assign(dag, states)

    total_assigned = sum(len(v) for v in assignments.values())
    assert total_assigned == 2   # both tasks assigned

    await robot_a.disconnect()
    await robot_b.disconnect()


@pytest.mark.asyncio
async def test_allocator_cooperative_task():
    cfg_a = G1Config(ip="10.0.0.3", name="G1-C")
    cfg_b = G1Config(ip="10.0.0.4", name="G1-D")
    robot_a = MockUnitreeBridge(cfg_a)
    robot_b = MockUnitreeBridge(cfg_b)
    await robot_a.connect()
    await robot_b.connect()

    allocator = AuctionAllocator([robot_a, robot_b])

    dag = TaskDAG()
    dag.add_task(TaskNode("t1", "make_bed", {}, min_robots=2, cooperative=True))

    states = {
        robot_a.robot_id: await robot_a.get_state(),
        robot_b.robot_id: await robot_b.get_state(),
    }
    assignments = allocator.assign(dag, states)

    # Both robots should be assigned to the cooperative task
    assigned_robots = [rid for rid, tasks in assignments.items() if "t1" in tasks]
    assert len(assigned_robots) >= 2

    await robot_a.disconnect()
    await robot_b.disconnect()


# ── LLMTaskPlanner (fallback mode) ────────────────────────────────────────────

def test_planner_fallback_clean():
    planner = LLMTaskPlanner(api_key="test-key")
    dag = planner._fallback_decompose("clean the bedroom", num_robots=2)
    assert not dag.is_complete()
    ready = dag.get_ready_tasks()
    assert len(ready) >= 1


def test_planner_fallback_bed():
    planner = LLMTaskPlanner(api_key="test-key")
    dag = planner._fallback_decompose("make the bed", num_robots=2)
    types = [t.task_type for t in dag.get_ready_tasks()]
    # should include a bed-related task
    all_types = [n.task_type for n in dag.graph.nodes.values()
                 if hasattr(n, 'task_type')] if hasattr(dag.graph, 'nodes') else types
    assert any("bed" in t or "sweep" in t or "wipe" in t for t in types)


def test_planner_parse_valid_json():
    planner = LLMTaskPlanner(api_key="test-key")
    json_str = '''
    {
      "tasks": [
        {"id": "t1", "type": "sweep_floor", "params": {"zone": "A"}, "min_robots": 1},
        {"id": "t2", "type": "wipe_surface", "params": {}, "min_robots": 1}
      ],
      "dependencies": [{"from": "t2", "depends_on": "t1"}]
    }
    '''
    dag = planner._parse_response(json_str)
    assert not dag.is_complete()
    ready = dag.get_ready_tasks()
    assert any(t.task_id == "t1" for t in ready)
