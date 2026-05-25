"""argos.navigation.room — Session-scoped room and zone registry."""
from __future__ import annotations

import logging
from typing import ClassVar

from argos.navigation.zones import ZoneManager

logger = logging.getLogger(__name__)

_DEFAULT_BOUNDS: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 6.0)


class RoomRegistry:
    """Singleton registry — one ZoneManager per named room.

    All state lives in-process for the duration of the REPL session.
    Use ``get_instance()`` everywhere rather than constructing directly.
    """

    _instance: ClassVar[RoomRegistry | None] = None

    def __init__(self) -> None:
        self._managers: dict[str, ZoneManager] = {}
        self._active: str | None = None

    @classmethod
    def get_instance(cls) -> RoomRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.add_room("default", _DEFAULT_BOUNDS)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (used in tests)."""
        cls._instance = None

    # ── room CRUD ──────────────────────────────────────────────────────────────

    def add_room(
        self,
        name: str,
        bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    ) -> ZoneManager:
        mgr = ZoneManager(bounds)
        self._managers[name] = mgr
        if self._active is None:
            self._active = name
        logger.info("Room %r registered: bounds=%s", name, bounds)
        return mgr

    def get_room(self, name: str | None = None) -> ZoneManager | None:
        key = name or self._active
        return self._managers.get(key) if key else None

    def active_name(self) -> str | None:
        return self._active

    def set_active(self, name: str) -> bool:
        if name in self._managers:
            self._active = name
            return True
        return False

    def list_rooms(self) -> list[str]:
        return list(self._managers.keys())

    def summary(self, name: str | None = None) -> dict | None:
        mgr = self.get_room(name)
        return mgr.summary() if mgr else None
