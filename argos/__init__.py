"""
ARGOS — Autonomous Robot Group Operations System
=================================================

A Python framework for coordinating Unitree G1 humanoid robot swarms
to perform structured cleaning and manipulation tasks in real environments.

Key subsystems
--------------
argos.swarm       — Swarm coordinator, peer discovery, task allocation
argos.comm        — Low-level robot communication (SDK + gRPC)
argos.perception  — Room mapping, object detection, dirt detection
argos.policy      — Diffusion / ACT / VLM policy inference
argos.navigation  — Path planning and whole-body locomotion
argos.tasks       — Task definitions, execution engines, success checks
argos.training    — RL / IL training pipelines and sim utilities
argos.cli         — Textual TUI and Typer CLI entry-points
"""

__version__ = "0.1.0"
__author__ = "ARGOS Contributors"
__license__ = "MIT"
