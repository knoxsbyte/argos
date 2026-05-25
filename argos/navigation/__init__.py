"""
argos.navigation — Path planning and zone management for ARGOS robots.

Public API
----------
Zone management:
    Zone, ZoneManager

Coverage path planning:
    Waypoint, BoustrophedonPlanner

Navigation execution:
    NavigationExecutor
"""

from argos.navigation.coverage import BoustrophedonPlanner, NavigationExecutor, Waypoint
from argos.navigation.room import RoomRegistry
from argos.navigation.zones import Zone, ZoneManager

__all__ = [
    "Zone",
    "ZoneManager",
    "RoomRegistry",
    "Waypoint",
    "BoustrophedonPlanner",
    "NavigationExecutor",
]
