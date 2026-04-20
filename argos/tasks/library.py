"""
argos.tasks.library — TaskLibrary: singleton registry that loads cleaning.yaml
and instantiates the correct BaseTask subclass from a task_type string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from argos.tasks.base import BaseTask

logger = logging.getLogger(__name__)

# Mapping from task_type string -> concrete class.
# Populated at import time after the subclasses are imported below.
_REGISTRY: dict[str, type[BaseTask]] = {}


def _build_registry() -> None:
    """Populate _REGISTRY with all known concrete task classes."""
    from argos.tasks.solo import (
        MopFloorTask,
        PickUpObjectTask,
        SortItemsTask,
        SweepFloorTask,
        TakeOutTrashTask,
        VacuumFloorTask,
        WipeSurfaceTask,
        WipeWindowTask,
    )
    from argos.tasks.cooperative import (
        ChangeSheetsTask,
        MakeBedTask,
        MoveFurnitureTask,
        OrganizeShelfTask,
    )

    for cls in (
        SweepFloorTask,
        VacuumFloorTask,
        MopFloorTask,
        WipeSurfaceTask,
        WipeWindowTask,
        PickUpObjectTask,
        SortItemsTask,
        TakeOutTrashTask,
        MakeBedTask,
        ChangeSheetsTask,
        MoveFurnitureTask,
        OrganizeShelfTask,
    ):
        _REGISTRY[cls.task_type] = cls  # type: ignore[attr-defined]


# Default config path — relative to this file's package root
_DEFAULT_CONFIG = (
    Path(__file__).parent.parent.parent / "configs" / "tasks" / "cleaning.yaml"
)


class TaskLibrary:
    """Singleton registry that maps task_type strings to task classes.

    Usage::

        lib = TaskLibrary.get_instance()
        task = lib.create("sweep_floor", task_id="t1", params={"zone_bounds": [0,0,5,4]})
    """

    _instance: TaskLibrary | None = None

    def __init__(self, config_path: Path | None = None) -> None:
        if not _REGISTRY:
            _build_registry()

        resolved = config_path or _DEFAULT_CONFIG
        self._config_path = resolved
        self._tasks: dict[str, dict[str, Any]] = {}

        if resolved.exists():
            self._load(resolved)
        else:
            logger.warning(
                "TaskLibrary: cleaning.yaml not found at %s — "
                "library will operate without YAML metadata.",
                resolved,
            )

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "TaskLibrary":
        if cls._instance is None:
            cls._instance = TaskLibrary()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing with custom config paths)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        task_library = raw.get("task_library", {})
        for task_type, config in task_library.items():
            self._tasks[task_type] = config
            logger.debug("TaskLibrary: loaded config for %r", task_type)

        logger.info(
            "TaskLibrary: loaded %d task types from %s", len(self._tasks), path
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def create(
        self,
        task_type: str,
        task_id: str,
        params: dict | None = None,
    ) -> BaseTask:
        """Instantiate the correct BaseTask subclass for task_type.

        Parameters
        ----------
        task_type:
            Key matching a cleaning.yaml entry and a registered class,
            e.g. "sweep_floor".
        task_id:
            Unique identifier for this task instance.
        params:
            Task-specific parameter dict. Defaults to empty dict.

        Raises
        ------
        ValueError
            If task_type is not registered.
        """
        if task_type not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unknown task_type {task_type!r}. "
                f"Available types: {available}"
            )

        cls = _REGISTRY[task_type]
        task = cls(task_id=task_id, params=params or {})
        logger.debug("TaskLibrary.create: %r -> %s(id=%r)", task_type, cls.__name__, task_id)
        return task

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_types(self) -> list[str]:
        """Return sorted list of all registered task type names."""
        return sorted(_REGISTRY)

    def get_config(self, task_type: str) -> dict[str, Any]:
        """Return the YAML config dict for task_type.

        Returns an empty dict if task_type was not found in cleaning.yaml.
        """
        return dict(self._tasks.get(task_type, {}))

    def is_cooperative(self, task_type: str) -> bool:
        """Return True if task_type requires multi-robot coordination."""
        if task_type in _REGISTRY:
            return _REGISTRY[task_type].cooperative  # type: ignore[attr-defined]
        # Fall back to YAML config
        cfg = self._tasks.get(task_type, {})
        return cfg.get("type", "solo") == "cooperative"

    def min_robots(self, task_type: str) -> int:
        """Return the minimum number of robots needed for task_type."""
        if task_type in _REGISTRY:
            return _REGISTRY[task_type].min_robots  # type: ignore[attr-defined]
        cfg = self._tasks.get(task_type, {})
        return int(cfg.get("min_robots", 1))

    def get_registry(self) -> dict[str, type[BaseTask]]:
        """Return the raw class registry (task_type -> class)."""
        return dict(_REGISTRY)

    def __repr__(self) -> str:
        return (
            f"<TaskLibrary config={self._config_path} "
            f"types={len(_REGISTRY)} yaml_entries={len(self._tasks)}>"
        )
