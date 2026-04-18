"""
argos.swarm.dependency — Task Dependency Graph for ARGOS swarm coordination.

Uses a NetworkX DiGraph to model task ordering constraints. Nodes are TaskNode
instances; directed edges represent "must-complete-before" relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """A single task in the swarm's execution plan.

    Attributes
    ----------
    task_id:
        Unique identifier for this task (e.g. ``"t1"``).
    task_type:
        Task kind matching a key in ``configs/tasks/cleaning.yaml``
        (e.g. ``"sweep_floor"``, ``"make_bed"``).
    params:
        Task-specific parameters (zone, target surface, etc.).
    min_robots:
        Minimum number of robots required to execute this task.
    cooperative:
        Whether this task requires coordinated multi-robot execution via PEFA.
    duration_estimate:
        Expected execution time in seconds.
    assigned_robots:
        robot_ids currently assigned to this task.
    status:
        Lifecycle state: ``pending`` | ``active`` | ``done`` | ``failed``.
    """

    task_id: str
    task_type: str
    params: dict[str, Any]
    min_robots: int = 1
    cooperative: bool = False
    duration_estimate: int = 60
    assigned_robots: list[str] = field(default_factory=list)
    status: str = "pending"


class TaskDAG:
    """Directed Acyclic Graph of tasks with dependency tracking.

    Edges run from *dependent* → *dependency*, meaning an edge (A, B)
    encodes "B must complete before A starts." ``add_dependency(from_id, to_id)``
    inserts edge (from_id, to_id) so ``to_id`` blocks ``from_id``.

    Usage::

        dag = TaskDAG()
        dag.add_task(TaskNode("t1", "sweep_floor", {"zone": "A"}))
        dag.add_task(TaskNode("t2", "mop_floor", {"zone": "A"}))
        dag.add_dependency("t2", "t1")   # mop after sweep
        ready = dag.get_ready_tasks()    # returns [t1]
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_task(self, node: TaskNode) -> None:
        """Insert a task node into the graph.

        If a node with the same ``task_id`` already exists it is replaced.
        """
        self.graph.add_node(node.task_id, task=node)
        logger.debug("TaskDAG: added task %s (%s).", node.task_id, node.task_type)

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Record that *to_id* must complete before *from_id* starts.

        Parameters
        ----------
        from_id:
            The task that depends on another.
        to_id:
            The prerequisite task that must finish first.

        Raises
        ------
        KeyError:
            If either task_id is not present in the graph.
        ValueError:
            If the edge would introduce a cycle.
        """
        if from_id not in self.graph:
            raise KeyError(f"Task '{from_id}' not found in DAG.")
        if to_id not in self.graph:
            raise KeyError(f"Task '{to_id}' not found in DAG.")

        # Edge direction: to_id → from_id means "to_id is a prerequisite of
        # from_id". graph.predecessors(from_id) will then yield to_id, so
        # get_ready_tasks() correctly blocks from_id until to_id is done.
        self.graph.add_edge(to_id, from_id)

        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(to_id, from_id)
            raise ValueError(
                f"Adding dependency {from_id!r} → {to_id!r} would create a cycle."
            )
        logger.debug("TaskDAG: dependency %s → %s added.", from_id, to_id)

    def mark_done(self, task_id: str) -> None:
        """Mark a task as successfully completed."""
        self._set_status(task_id, "done")

    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed (will block dependents)."""
        self._set_status(task_id, "failed")

    def _set_status(self, task_id: str, status: str) -> None:
        if task_id not in self.graph:
            raise KeyError(f"Task '{task_id}' not found in DAG.")
        self.graph.nodes[task_id]["task"].status = status
        logger.debug("TaskDAG: task %s → %s.", task_id, status)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> TaskNode | None:
        """Return the TaskNode for *task_id*, or None if not present."""
        data = self.graph.nodes.get(task_id)
        return data["task"] if data else None

    def get_ready_tasks(self) -> list[TaskNode]:
        """Return tasks whose predecessors have all completed.

        A task is *ready* when:
        - Its status is ``pending``.
        - Every predecessor node has status ``done``.
        """
        ready: list[TaskNode] = []
        for node_id, data in self.graph.nodes(data=True):
            task: TaskNode = data["task"]
            if task.status != "pending":
                continue
            predecessors = list(self.graph.predecessors(node_id))
            all_done = all(
                self.graph.nodes[pred]["task"].status == "done"
                for pred in predecessors
            )
            if all_done:
                ready.append(task)
        return ready

    def get_all_tasks(self) -> list[TaskNode]:
        """Return all task nodes in topological order."""
        try:
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            order = list(self.graph.nodes)
        return [self.graph.nodes[n]["task"] for n in order]

    def is_complete(self) -> bool:
        """Return True when every task has status ``done``."""
        return all(
            data["task"].status == "done"
            for _, data in self.graph.nodes(data=True)
        )

    def has_failed(self) -> bool:
        """Return True if any task is in the ``failed`` state."""
        return any(
            data["task"].status == "failed"
            for _, data in self.graph.nodes(data=True)
        )

    def get_critical_path(self) -> list[TaskNode]:
        """Return the sequence of tasks on the longest (critical) path.

        Uses task ``duration_estimate`` as edge weights. If the graph is empty
        returns an empty list.
        """
        if not self.graph.nodes:
            return []

        # Build a weighted copy where edge weight = duration of the source node.
        weighted = nx.DiGraph()
        for node_id, data in self.graph.nodes(data=True):
            weighted.add_node(node_id, duration=data["task"].duration_estimate)
        for u, v in self.graph.edges():
            # Edge u → v means u is a prerequisite of v; weight is u's duration.
            weighted.add_edge(u, v, weight=self.graph.nodes[u]["task"].duration_estimate)

        # Find the path with maximum total duration.
        try:
            # nx.dag_longest_path uses edge weights when provided.
            path = nx.dag_longest_path(weighted, weight="weight")
        except nx.NetworkXUnfeasible:
            logger.warning("TaskDAG: graph has cycles; cannot compute critical path.")
            return []

        return [self.graph.nodes[n]["task"] for n in path]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the DAG to a plain dictionary (JSON-compatible)."""
        tasks = []
        for _, data in self.graph.nodes(data=True):
            t: TaskNode = data["task"]
            tasks.append(
                {
                    "task_id": t.task_id,
                    "task_type": t.task_type,
                    "params": t.params,
                    "min_robots": t.min_robots,
                    "cooperative": t.cooperative,
                    "duration_estimate": t.duration_estimate,
                    "assigned_robots": list(t.assigned_robots),
                    "status": t.status,
                }
            )
        # Edge (u, v) means u is a prerequisite of v, so v "from" depends_on u.
        dependencies = [
            {"from": v, "depends_on": u} for u, v in self.graph.edges()
        ]
        return {"tasks": tasks, "dependencies": dependencies}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDAG":
        """Reconstruct a TaskDAG from the dict produced by :meth:`to_dict`."""
        dag = cls()
        for t in data.get("tasks", []):
            node = TaskNode(
                task_id=t["task_id"],
                task_type=t["task_type"],
                params=t.get("params", {}),
                min_robots=t.get("min_robots", 1),
                cooperative=t.get("cooperative", False),
                duration_estimate=t.get("duration_estimate", 60),
                assigned_robots=t.get("assigned_robots", []),
                status=t.get("status", "pending"),
            )
            dag.add_task(node)
        for dep in data.get("dependencies", []):
            try:
                dag.add_dependency(dep["from"], dep["depends_on"])
            except (KeyError, ValueError) as exc:
                logger.warning("TaskDAG.from_dict: skipping bad dependency — %s", exc)
        return dag

    def __repr__(self) -> str:  # pragma: no cover
        counts = {"pending": 0, "active": 0, "done": 0, "failed": 0}
        for _, data in self.graph.nodes(data=True):
            counts[data["task"].status] = counts.get(data["task"].status, 0) + 1
        return (
            f"TaskDAG(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"pending={counts['pending']}, active={counts['active']}, "
            f"done={counts['done']}, failed={counts['failed']})"
        )
