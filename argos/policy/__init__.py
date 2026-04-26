"""
argos.policy — Policy inference module for ARGOS robot task execution.

Provides abstract and concrete policy implementations for language-conditioned
and visuomotor robot control, plus a router for task-type-based policy selection.

All ML-dependent classes degrade gracefully to MockPolicy behaviour when
GPU/model weights are unavailable.
"""

from argos.policy.base import (
    BasePolicy,
    MockPolicy,
    PolicyConfig,
    PolicyObservation,
    PolicyOutput,
)
from argos.policy.act import ACTPolicy
from argos.policy.diffusion import DiffusionPolicy
from argos.policy.router import PolicyRouter
from argos.policy.vla import OpenVLAPolicy

__all__ = [
    # Base abstractions
    "BasePolicy",
    "MockPolicy",
    "PolicyConfig",
    "PolicyObservation",
    "PolicyOutput",
    # Concrete policies
    "OpenVLAPolicy",
    "DiffusionPolicy",
    "ACTPolicy",
    # Router
    "PolicyRouter",
]
