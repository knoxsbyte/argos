"""
argos.swarm.planner — LLM-based task planner for ARGOS swarm coordination.

Uses the Anthropic Claude API to decompose a natural-language cleaning goal
into a TaskDAG. Falls back to rule-based heuristics if the API fails or
returns malformed JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import yaml

from argos.swarm.dependency import TaskDAG, TaskNode

logger = logging.getLogger(__name__)

# Path to the task library config relative to the package root.
_CLEANING_YAML = (
    Path(__file__).parent.parent.parent / "configs" / "tasks" / "cleaning.yaml"
)

# JSON schema description embedded in the planning prompt.
_JSON_SCHEMA = """\
{
  "tasks": [
    {
      "id": "<string>",
      "type": "<task_type from the library>",
      "params": {"<key>": "<value>"},
      "min_robots": <int, default 1>,
      "cooperative": <bool, default false>,
      "duration_estimate": <int seconds, optional>
    }
  ],
  "dependencies": [
    {"from": "<task_id>", "depends_on": "<task_id that must finish first>"}
  ]
}"""


class LLMTaskPlanner:
    """Decompose a natural-language cleaning goal into a :class:`TaskDAG`.

    Parameters
    ----------
    model:
        Anthropic model string.
    api_key:
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        api_key: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self._task_library: dict[str, Any] = self._load_task_library()
        logger.info(
            "LLMTaskPlanner initialised with model=%s, %d task types loaded.",
            model,
            len(self._task_library),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        goal: str,
        scene_info: dict[str, Any],
        num_robots: int = 2,
    ) -> TaskDAG:
        """Decompose a natural-language *goal* into a :class:`TaskDAG`.

        Parameters
        ----------
        goal:
            High-level cleaning goal, e.g. ``"clean the kitchen"``.
        scene_info:
            Perception output: rooms detected, objects on surfaces, floor types, etc.
        num_robots:
            Number of robots available for assignment.

        Returns
        -------
        TaskDAG
            Parsed and validated task graph ready for allocation.
        """
        prompt = self._build_prompt(goal, scene_info, num_robots)
        logger.info("LLMTaskPlanner: calling Claude for goal=%r", goal)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            logger.debug("LLMTaskPlanner: raw response:\n%s", response_text)
            dag = self._parse_response(response_text)
            if dag.graph.number_of_nodes() == 0:
                logger.warning(
                    "LLMTaskPlanner: parsed DAG is empty; falling back to heuristics."
                )
                return self._fallback_decompose(goal, num_robots)
            return dag
        except anthropic.APIError as exc:
            logger.error(
                "LLMTaskPlanner: Anthropic API error (%s); using fallback.", exc
            )
            return self._fallback_decompose(goal, num_robots)
        except Exception as exc:
            logger.exception(
                "LLMTaskPlanner: unexpected error during decompose; using fallback. %s",
                exc,
            )
            return self._fallback_decompose(goal, num_robots)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self, goal: str, scene_info: dict[str, Any], num_robots: int
    ) -> str:
        """Build a structured prompt for task decomposition."""
        task_list_lines: list[str] = []
        for name, spec in self._task_library.items():
            min_r = spec.get("min_robots", 1)
            max_r = spec.get("max_robots", 4)
            coop = spec.get("type", "solo") == "cooperative"
            dur = spec.get("duration_estimate", {})
            base = dur.get("base_seconds", 60) if isinstance(dur, dict) else dur
            task_list_lines.append(
                f"  - {name}: min_robots={min_r}, max_robots={max_r}, "
                f"cooperative={coop}, duration_estimate_base={base}s"
            )
        task_list_str = "\n".join(task_list_lines)

        scene_str = json.dumps(scene_info, indent=2) if scene_info else "{}"

        return f"""\
You are a task planner for a swarm of {num_robots} Unitree G1 humanoid robots \
performing cleaning operations.

## Goal
{goal}

## Scene Information
{scene_str}

## Available Task Types
{task_list_str}

## Instructions
Decompose the goal into a minimal, ordered set of tasks from the available task types.
- Assign `min_robots` appropriately (cooperative tasks need ≥ 2 robots).
- Set `cooperative: true` for tasks that require simultaneous multi-robot coordination.
- Define `dependencies` so tasks execute in the correct order.
- Keep the plan concise — prefer fewer, well-scoped tasks.
- Only use task types from the list above.
- All task IDs must be unique strings (e.g. "t1", "t2", ...).

## Output Format
Respond with ONLY valid JSON matching this schema — no prose, no markdown fences:
{_JSON_SCHEMA}
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str) -> TaskDAG:
        """Parse a JSON response from the LLM into a :class:`TaskDAG`.

        Handles responses wrapped in markdown code fences gracefully.
        """
        # Strip optional markdown fences (```json ... ```)
        cleaned = response_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Find the first { ... } block in case there is surrounding text.
        brace_match = re.search(r"\{[\s\S]+\}", cleaned)
        if brace_match:
            cleaned = brace_match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLMTaskPlanner: JSON decode error — %s. Returning empty DAG.", exc
            )
            return TaskDAG()

        return self._build_dag_from_data(data)

    def _build_dag_from_data(self, data: dict[str, Any]) -> TaskDAG:
        """Construct a TaskDAG from a parsed JSON dict."""
        dag = TaskDAG()
        tasks_data: list[dict[str, Any]] = data.get("tasks", [])
        for t in tasks_data:
            task_id = str(t.get("id", ""))
            task_type = str(t.get("type", ""))
            if not task_id or not task_type:
                logger.warning(
                    "LLMTaskPlanner: skipping malformed task entry: %s", t
                )
                continue
            if task_type not in self._task_library:
                logger.warning(
                    "LLMTaskPlanner: unknown task type %r — skipping.", task_type
                )
                continue

            lib_spec = self._task_library[task_type]
            lib_dur = lib_spec.get("duration_estimate", {})
            lib_base = (
                lib_dur.get("base_seconds", 60)
                if isinstance(lib_dur, dict)
                else int(lib_dur)
            )

            node = TaskNode(
                task_id=task_id,
                task_type=task_type,
                params=t.get("params", {}),
                min_robots=int(t.get("min_robots", lib_spec.get("min_robots", 1))),
                cooperative=bool(
                    t.get(
                        "cooperative",
                        lib_spec.get("type", "solo") == "cooperative",
                    )
                ),
                duration_estimate=int(t.get("duration_estimate", lib_base)),
            )
            dag.add_task(node)

        for dep in data.get("dependencies", []):
            from_id = str(dep.get("from", ""))
            depends_on = str(dep.get("depends_on", ""))
            if not from_id or not depends_on:
                continue
            try:
                dag.add_dependency(from_id, depends_on)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "LLMTaskPlanner: skipping invalid dependency %s → %s: %s",
                    from_id,
                    depends_on,
                    exc,
                )

        return dag

    # ------------------------------------------------------------------
    # Fallback heuristics
    # ------------------------------------------------------------------

    def _fallback_decompose(self, goal: str, num_robots: int) -> TaskDAG:
        """Rule-based task decomposition when the LLM is unavailable.

        Keyword mapping:
        - Any goal → ``sweep_floor``
        - "clean" / "floor" / "kitchen" → ``mop_floor`` after sweep
        - "surface" / "counter" / "table" / "kitchen" → ``wipe_surface``
        - "bedroom" / "bed" → ``make_bed`` (cooperative if ≥ 2 robots)
        - "trash" / "bin" / "garbage" → ``take_out_trash``
        - "window" / "glass" / "mirror" → ``wipe_window``
        - "vacuum" / "carpet" → ``vacuum_floor`` instead of sweep
        - "shelf" / "organise" / "organize" → ``organize_shelf``
        """
        goal_lower = goal.lower()
        dag = TaskDAG()
        counter = 0

        def _next_id() -> str:
            nonlocal counter
            counter += 1
            return f"t{counter}"

        def _dur(task_type: str) -> int:
            spec = self._task_library.get(task_type, {})
            dur = spec.get("duration_estimate", {})
            return dur.get("base_seconds", 60) if isinstance(dur, dict) else 60

        # Always start with floor cleaning.
        if any(kw in goal_lower for kw in ("vacuum", "carpet")):
            floor_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=floor_id,
                    task_type="vacuum_floor",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("vacuum_floor"),
                )
            )
        else:
            floor_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=floor_id,
                    task_type="sweep_floor",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("sweep_floor"),
                )
            )

        # Mopping after sweeping for relevant keywords.
        if any(kw in goal_lower for kw in ("clean", "floor", "kitchen", "mop")):
            mop_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=mop_id,
                    task_type="mop_floor",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("mop_floor"),
                )
            )
            dag.add_dependency(mop_id, floor_id)

        # Surface wiping.
        if any(
            kw in goal_lower
            for kw in ("surface", "counter", "table", "wipe", "kitchen")
        ):
            wipe_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=wipe_id,
                    task_type="wipe_surface",
                    params={"target": "counter"},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("wipe_surface"),
                )
            )
            # Wipe surfaces in parallel with floor tasks — no dependency needed.

        # Bed making for bedroom goals.
        if any(kw in goal_lower for kw in ("bedroom", "bed", "sheet")):
            bed_min = 2 if num_robots >= 2 else 1
            bed_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=bed_id,
                    task_type="make_bed",
                    params={},
                    min_robots=bed_min,
                    cooperative=bed_min >= 2,
                    duration_estimate=_dur("make_bed"),
                )
            )

        # Trash collection.
        if any(kw in goal_lower for kw in ("trash", "bin", "garbage", "rubbish")):
            trash_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=trash_id,
                    task_type="take_out_trash",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("take_out_trash"),
                )
            )

        # Window / glass cleaning.
        if any(kw in goal_lower for kw in ("window", "glass", "mirror")):
            win_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=win_id,
                    task_type="wipe_window",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("wipe_window"),
                )
            )

        # Shelf organisation.
        if any(kw in goal_lower for kw in ("shelf", "organis", "organiz")):
            shelf_id = _next_id()
            dag.add_task(
                TaskNode(
                    task_id=shelf_id,
                    task_type="organize_shelf",
                    params={},
                    min_robots=1,
                    cooperative=False,
                    duration_estimate=_dur("organize_shelf"),
                )
            )

        logger.info(
            "LLMTaskPlanner: fallback decomposition → %d tasks.",
            dag.graph.number_of_nodes(),
        )
        return dag

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_task_library(self) -> dict[str, Any]:
        """Load and return the task_library section from cleaning.yaml."""
        yaml_path = _CLEANING_YAML
        if not yaml_path.exists():
            logger.warning(
                "LLMTaskPlanner: cleaning.yaml not found at %s; task library empty.",
                yaml_path,
            )
            return {}
        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            library: dict[str, Any] = raw.get("task_library", {})
            logger.debug(
                "LLMTaskPlanner: loaded %d task types from %s.", len(library), yaml_path
            )
            return library
        except Exception as exc:
            logger.error("LLMTaskPlanner: failed to load cleaning.yaml — %s", exc)
            return {}
