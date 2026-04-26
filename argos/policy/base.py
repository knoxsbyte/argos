"""
argos.policy.base — Abstract base class for all ARGOS inference policies.

All concrete policy classes extend BasePolicy and implement load(), predict(),
and reset(). The module also provides PolicyConfig, PolicyObservation,
PolicyOutput, and a MockPolicy for testing without GPU/models.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from argos.comm import Action, RobotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PolicyConfig(BaseModel):
    """Pydantic model for policy configuration."""

    model_name: str = Field(description="Model identifier or HuggingFace repo ID.")
    checkpoint_path: str | None = Field(
        default=None,
        description="Local path to a checkpoint directory or state-dict file.",
    )
    device: str = Field(
        default="cpu",
        description='Torch device string: "cpu", "cuda", "cuda:0", etc.',
    )
    inference_freq: float = Field(
        default=10.0,
        gt=0.0,
        description="Target inference frequency in Hz. predict() must complete within 1/freq seconds.",
    )
    action_chunk_size: int = Field(
        default=8,
        gt=0,
        description="Number of future actions to predict simultaneously (ACT / diffusion).",
    )

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Observation / output containers
# ---------------------------------------------------------------------------


@dataclass
class PolicyObservation:
    """Input to a policy at one timestep."""

    image: np.ndarray
    """HxWx3 uint8 RGB frame from the robot camera."""

    robot_state: RobotState
    """Current joint positions, velocities, IMU, etc."""

    language_instruction: str
    """Natural-language description of the task being executed."""

    depth: np.ndarray | None = None
    """HxW float32 depth map in metres (optional)."""

    timestep: int = 0
    """Monotonically increasing step counter within an episode."""


@dataclass
class PolicyOutput:
    """Output from a policy at one timestep."""

    action: Action
    """Joint targets and gripper commands to send to the robot."""

    confidence: float
    """Scalar in [0, 1] indicating how confident the policy is in this action."""

    done_signal: bool = False
    """True when the policy believes the current task is complete."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Policy-specific extras (e.g. attention maps, token logits)."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BasePolicy(ABC):
    """Abstract base class for all ARGOS inference policies.

    Subclasses must implement load(), predict(), and reset().
    The _time_predict() wrapper automatically tracks per-call latency.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self.is_loaded: bool = False
        self._inference_count: int = 0
        self._total_inference_ms: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (may take several seconds).

        Must set self.is_loaded = True on success.
        Must never raise — fall back to mock_mode on failure.
        """

    @abstractmethod
    def predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Run one inference step.

        Must complete in less than 1 / config.inference_freq seconds.
        Called via _time_predict() to maintain latency statistics.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal episode state between task episodes."""

    # ------------------------------------------------------------------
    # Optional teardown
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory and unload model weights.

        Subclasses may override; base implementation is a no-op.
        """
        self.is_loaded = False
        logger.debug("%s: unloaded.", self.__class__.__name__)

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    @property
    def avg_inference_ms(self) -> float:
        """Average predict() wall-clock time in milliseconds."""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_ms / self._inference_count

    def _time_predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Wrap predict() with wall-clock timing and update stats."""
        t0 = time.perf_counter()
        output = self.predict(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0
        self._inference_count += 1
        self._total_inference_ms += elapsed_ms

        budget_ms = 1_000.0 / self.config.inference_freq
        if elapsed_ms > budget_ms:
            logger.warning(
                "%s: predict() took %.1f ms (budget %.1f ms at %.1f Hz).",
                self.__class__.__name__,
                elapsed_ms,
                budget_ms,
                self.config.inference_freq,
            )
        return output

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} model={self.config.model_name!r} "
            f"loaded={self.is_loaded} inferences={self._inference_count}>"
        )


# ---------------------------------------------------------------------------
# Mock policy — safe zero-noise actions, no ML dependencies
# ---------------------------------------------------------------------------


class MockPolicy(BasePolicy):
    """Mock policy for testing — returns safe near-zero actions with small Gaussian noise.

    Signals done after approximately 10 steps so that test loops terminate.
    Never requires GPU or any ML library.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        if config is None:
            config = PolicyConfig(model_name="mock")
        super().__init__(config)
        self._step: int = 0
        self._rng = np.random.default_rng(seed=0)

    def load(self) -> None:
        """Instant load — nothing to fetch."""
        self.is_loaded = True
        logger.debug("MockPolicy: loaded (no-op).")

    def predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Return a near-zero action with small Gaussian noise."""
        self._step += 1

        noise = self._rng.normal(0.0, 0.01, size=29).tolist()
        action = Action(
            joint_targets=noise,
            gripper_left=float(abs(self._rng.normal(0.0, 0.02))),
            gripper_right=float(abs(self._rng.normal(0.0, 0.02))),
            duration_ms=int(1_000.0 / max(self.config.inference_freq, 1.0)),
        )
        # Clip to hardware limits via the model's own method
        action = action.clipped()

        done = self._step >= 10
        confidence = max(0.0, 1.0 - self._step * 0.05)

        return PolicyOutput(
            action=action,
            confidence=confidence,
            done_signal=done,
            metadata={"step": self._step, "policy": "mock"},
        )

    def reset(self) -> None:
        """Reset step counter for a new episode."""
        self._step = 0
        logger.debug("MockPolicy: reset.")
