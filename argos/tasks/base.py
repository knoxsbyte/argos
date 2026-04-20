"""
argos.tasks.base — Abstract base class for all ARGOS cleaning tasks.

All concrete task classes extend BaseTask and implement task_type, min_robots,
cooperative, execute(), and validate_params().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    success: bool
    duration_seconds: float
    error_message: str | None = None
    metrics: dict = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base for every ARGOS cleaning task.

    Subclasses must implement:
      - task_type property (str matching cleaning.yaml key)
      - min_robots property (int)
      - cooperative property (bool)
      - execute(robots) -> TaskResult
      - validate_params() -> bool
    """

    def __init__(self, task_id: str, params: dict) -> None:
        self.task_id = task_id
        self.params = params
        self.status = TaskStatus.PENDING
        self.start_time: float | None = None
        self.result: TaskResult | None = None

        import asyncio
        self._cancel_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type key matching cleaning.yaml entries."""

    @property
    @abstractmethod
    def min_robots(self) -> int:
        """Minimum number of robots required to execute this task."""

    @property
    @abstractmethod
    def cooperative(self) -> bool:
        """True if the task requires multi-robot coordination."""

    @abstractmethod
    async def execute(self, robots: list) -> TaskResult:
        """Execute the task on the provided robots. Return a TaskResult."""

    @abstractmethod
    def validate_params(self) -> bool:
        """Return True if self.params are valid for this task type."""

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def cancel(self) -> None:
        """Signal cancellation; task execute() should observe is_cancelled()."""
        self._cancel_event.set()
        self.status = TaskStatus.CANCELLED

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _begin(self) -> float:
        """Mark task active and record start time. Returns start timestamp."""
        self.status = TaskStatus.ACTIVE
        self.start_time = time.monotonic()
        return self.start_time

    def _finish(self, result: TaskResult) -> TaskResult:
        """Store result and update status."""
        self.result = result
        self.status = TaskStatus.DONE if result.success else TaskStatus.FAILED
        return result

    def _elapsed(self) -> float:
        """Seconds since _begin() was called."""
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this task."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "min_robots": self.min_robots,
            "cooperative": self.cooperative,
            "params": self.params,
            "start_time": self.start_time,
            "result": {
                "success": self.result.success,
                "duration_seconds": self.result.duration_seconds,
                "error_message": self.result.error_message,
                "metrics": self.result.metrics,
            }
            if self.result is not None
            else None,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.task_id!r} "
            f"status={self.status.value}>"
        )
