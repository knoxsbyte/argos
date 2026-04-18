"""
argos.swarm — Swarm coordination layer for the ARGOS robot framework.

Provides LLM-based task planning, auction-based allocation, PEFA cooperative
execution, and the top-level SwarmCoordinator that ties everything together.
"""

from argos.swarm.allocator import AuctionAllocator
from argos.swarm.cooperative import CooperativeCoordinator, PEFASession
from argos.swarm.coordinator import SwarmCoordinator
from argos.swarm.dependency import TaskDAG, TaskNode
from argos.swarm.planner import LLMTaskPlanner

__all__ = [
    "SwarmCoordinator",
    "LLMTaskPlanner",
    "AuctionAllocator",
    "CooperativeCoordinator",
    "TaskDAG",
    "TaskNode",
    "PEFASession",
]
