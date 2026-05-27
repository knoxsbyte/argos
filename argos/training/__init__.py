"""
argos.training — ML training pipeline for ARGOS humanoid robot cleaning.

Public API
----------
Ingestion:
  VideoIngestor     — extracts frames from cleaning demo videos
  Episode           — a complete demonstration episode
  VideoFrame        — a single decoded frame with optional depth

Preprocessing:
  PoseEstimator     — human hand/body pose estimation (MediaPipe or optical-flow)
  ActionLabeler     — segments video and converts wrist motion → robot actions

Dataset:
  LeRobotDatasetBuilder — builds LeRobot-format HDF5 datasets

Fine-tuning:
  LoRAFinetuner     — LoRA fine-tuning for OpenVLA / Diffusion Policy
  FinetuneConfig    — dataclass with all training hyperparameters

Evaluation:
  PolicyEvaluator   — runs policies in simulation and reports results
  EvalConfig        — evaluation configuration
  EvalResult        — per-task evaluation result container

Simulation:
  CleaningEnv       — MuJoCo room environment (gymnasium-compatible)
"""

from argos.training.ingest import Episode, VideoFrame, VideoIngestor
from argos.training.preprocess import ActionLabeler, PoseEstimator
from argos.training.dataset import LeRobotDatasetBuilder
from argos.training.finetune import FinetuneConfig, LoRAFinetuner
from argos.training.evaluate import EvalConfig, EvalResult, PolicyEvaluator
from argos.training.checkpoints import CheckpointRecord, CheckpointRegistry
from argos.training.sim.mujoco_env import CleaningEnv

__all__ = [
    # Ingestion
    "VideoIngestor",
    "Episode",
    "VideoFrame",
    # Preprocessing
    "PoseEstimator",
    "ActionLabeler",
    # Dataset
    "LeRobotDatasetBuilder",
    # Fine-tuning
    "LoRAFinetuner",
    "FinetuneConfig",
    # Checkpoints
    "CheckpointRegistry",
    "CheckpointRecord",
    # Evaluation
    "PolicyEvaluator",
    "EvalConfig",
    "EvalResult",
    # Simulation
    "CleaningEnv",
]
