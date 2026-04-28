"""
argos.training.sim — Simulation environment backends for ARGOS training.

Provides:
  CleaningEnv      — MuJoCo-based environment (default, mock fallback available)
  IsaacCleaningEnv — Isaac Lab environment (requires NVIDIA Isaac Sim)
"""

from argos.training.sim.mujoco_env import CleaningEnv
from argos.training.sim.isaac_env import IsaacCleaningEnv

__all__ = ["CleaningEnv", "IsaacCleaningEnv"]
