"""
argos.tasks — Task definitions and execution engines for ARGOS.

Public API
----------
Solo tasks (single-robot):
    SweepFloorTask, VacuumFloorTask, MopFloorTask,
    WipeSurfaceTask, WipeWindowTask,
    PickUpObjectTask, SortItemsTask, TakeOutTrashTask

Cooperative tasks (multi-robot):
    MakeBedTask, ChangeSheetsTask, MoveFurnitureTask, OrganizeShelfTask

Base types:
    BaseTask, TaskResult, TaskStatus

Factory:
    TaskLibrary
"""

from argos.tasks.base import BaseTask, TaskResult, TaskStatus
from argos.tasks.cooperative import (
    ChangeSheetsTask,
    MakeBedTask,
    MoveFurnitureTask,
    OrganizeShelfTask,
)
from argos.tasks.library import TaskLibrary
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

__all__ = [
    # Base
    "BaseTask",
    "TaskResult",
    "TaskStatus",
    # Solo
    "SweepFloorTask",
    "VacuumFloorTask",
    "MopFloorTask",
    "WipeSurfaceTask",
    "WipeWindowTask",
    "PickUpObjectTask",
    "SortItemsTask",
    "TakeOutTrashTask",
    # Cooperative
    "MakeBedTask",
    "ChangeSheetsTask",
    "MoveFurnitureTask",
    "OrganizeShelfTask",
    # Factory
    "TaskLibrary",
]
