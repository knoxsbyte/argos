"""
argos.policy.router — Policy selection and lifecycle manager.

PolicyRouter maps task types to the appropriate policy implementation,
lazy-loads policies on first use, caches loaded instances to avoid
repeated weight loading, and provides the main async control loop for
executing a policy on a robot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from argos.comm import Action, RobotState
from argos.policy.base import BasePolicy, MockPolicy, PolicyConfig, PolicyObservation, PolicyOutput

if TYPE_CHECKING:
    from argos.tasks.base import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy policy imports — avoid heavy ML imports at module load time
# ---------------------------------------------------------------------------


def _import_vla() -> type[BasePolicy]:
    from argos.policy.vla import OpenVLAPolicy
    return OpenVLAPolicy


def _import_diffusion() -> type[BasePolicy]:
    from argos.policy.diffusion import DiffusionPolicy
    return DiffusionPolicy


def _import_act() -> type[BasePolicy]:
    from argos.policy.act import ACTPolicy
    return ACTPolicy


_POLICY_FACTORIES: dict[str, callable] = {
    "vla": _import_vla,
    "diffusion": _import_diffusion,
    "act": _import_act,
    "mock": lambda: MockPolicy,
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class PolicyRouter:
    """Selects and manages the right policy for each task type.

    Policies are lazy-loaded on first request and cached thereafter.
    If a policy fails to load, the router substitutes a MockPolicy so the
    control loop can always proceed safely.
    """

    # Default routing table: task_type → policy_type
    TASK_POLICY_MAP: dict[str, str] = {
        "sweep_floor": "diffusion",
        "vacuum_floor": "diffusion",
        "mop_floor": "diffusion",
        "wipe_surface": "act",
        "wipe_window": "act",
        "pick_up_object": "act",
        "sort_items": "vla",
        "make_bed": "act",
        "change_sheets": "act",
        "move_furniture": "diffusion",
        "take_out_trash": "act",
        "organize_shelf": "vla",
    }

    def __init__(self, config: dict | None = None) -> None:
        # Loaded policy instances, keyed by policy_type ("vla", "act", etc.)
        self._policies: dict[str, BasePolicy] = {}
        # Per-type PolicyConfig overrides set via configure()
        self._policy_configs: dict[str, PolicyConfig] = {}
        # Merge caller overrides with the defaults
        self.routing_table: dict[str, str] = {
            **self.TASK_POLICY_MAP,
            **(config or {}),
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, policy_type: str, config: PolicyConfig) -> None:
        """Set a PolicyConfig for a policy type before it is loaded.

        If the policy has already been loaded this config takes effect on
        the next call to get_policy() for an unloaded instance.
        """
        self._policy_configs[policy_type] = config
        logger.debug("PolicyRouter: configured policy_type='%s'.", policy_type)

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self, task_type: str) -> BasePolicy:
        """Return the loaded policy for task_type.

        Lazy-loads on the first call. Subsequent calls return the cached
        instance. If the mapped policy type cannot be loaded, a MockPolicy
        is substituted and a warning is emitted.
        """
        policy_type = self.routing_table.get(task_type)
        if policy_type is None:
            logger.warning(
                "PolicyRouter: no mapping for task_type='%s'. Falling back to mock.",
                task_type,
            )
            policy_type = "mock"

        if policy_type in self._policies:
            return self._policies[policy_type]

        policy = self._load_policy(policy_type)
        self._policies[policy_type] = policy
        return policy

    def _load_policy(self, policy_type: str) -> BasePolicy:
        """Instantiate and load a policy by type string.

        Returns a MockPolicy on any failure.
        """
        factory = _POLICY_FACTORIES.get(policy_type)
        if factory is None:
            logger.warning(
                "PolicyRouter: unknown policy_type='%s'. Using mock.", policy_type
            )
            return self._make_mock()

        config = self._policy_configs.get(policy_type) or PolicyConfig(
            model_name=policy_type
        )

        try:
            policy_cls = factory()
            policy: BasePolicy = policy_cls(config)
            policy.load()
            if not policy.is_loaded:
                raise RuntimeError("policy.is_loaded is False after load()")
            logger.info(
                "PolicyRouter: loaded policy_type='%s' (%s).",
                policy_type,
                policy_cls.__name__,
            )
            return policy
        except Exception as exc:
            logger.warning(
                "PolicyRouter: failed to load policy_type='%s' (%s). Using mock.",
                policy_type,
                exc,
            )
            return self._make_mock()

    def _make_mock(self) -> MockPolicy:
        config = PolicyConfig(model_name="mock")
        mock = MockPolicy(config)
        mock.load()
        return mock

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def preload(self, task_types: list[str]) -> None:
        """Eagerly load policies for the given task types.

        Safe to call at startup so the first predict() call is not penalised
        by model loading latency. Skips types already loaded.
        """
        seen: set[str] = set()
        for task_type in task_types:
            policy_type = self.routing_table.get(task_type, "mock")
            if policy_type in seen or policy_type in self._policies:
                continue
            seen.add(policy_type)
            logger.info(
                "PolicyRouter: preloading policy_type='%s' for task='%s'.",
                policy_type,
                task_type,
            )
            policy = self._load_policy(policy_type)
            self._policies[policy_type] = policy

    def unload_all(self) -> None:
        """Unload all cached policies and free GPU memory."""
        for policy_type, policy in list(self._policies.items()):
            try:
                policy.unload()
                logger.info("PolicyRouter: unloaded policy_type='%s'.", policy_type)
            except Exception as exc:
                logger.warning(
                    "PolicyRouter: error unloading policy_type='%s': %s", policy_type, exc
                )
        self._policies.clear()

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    async def run_policy_loop(
        self,
        task: "BaseTask",
        robot: object,
        policy: BasePolicy,
        max_steps: int = 1000,
    ) -> bool:
        """Execute a policy control loop for one task episode.

        Runs at policy.config.inference_freq Hz. At each step:
          1. Acquire camera frame and robot state from the robot bridge.
          2. Build a PolicyObservation and call policy.predict().
          3. Send the resulting Action to the robot.
          4. Stop on done_signal, task cancellation, or max_steps exceeded.

        Args:
            task:      The ARGOS task being executed (provides instruction and
                       cancellation signal).
            robot:     A UnitreeBridge or MockUnitreeBridge instance.
            policy:    The loaded policy to run.
            max_steps: Hard limit on inference steps per episode.

        Returns:
            True  if the policy emitted done_signal.
            False if cancelled or max_steps was reached.
        """
        if not policy.is_loaded:
            logger.warning(
                "PolicyRouter.run_policy_loop: policy is not loaded; calling load()."
            )
            policy.load()

        policy.reset()

        period = 1.0 / policy.config.inference_freq
        instruction = getattr(task, "params", {}).get(
            "instruction", task.task_type if hasattr(task, "task_type") else "clean"
        )

        logger.info(
            "PolicyRouter: starting control loop (max_steps=%d, freq=%.1f Hz, task=%r).",
            max_steps,
            policy.config.inference_freq,
            instruction,
        )

        step = 0
        while step < max_steps:
            if task.is_cancelled():
                logger.info("PolicyRouter: task cancelled at step %d.", step)
                return False

            loop_start = time.perf_counter()

            # --- Observe ---
            try:
                image = await robot.get_camera_frame()
            except Exception as exc:
                logger.warning("PolicyRouter: camera error at step %d: %s.", step, exc)
                image = np.zeros((480, 640, 3), dtype=np.uint8)

            try:
                depth = await robot.get_depth_frame()
            except Exception:
                depth = None

            try:
                robot_state: RobotState = await robot.get_state()
            except Exception as exc:
                logger.warning("PolicyRouter: state error at step %d: %s.", step, exc)
                robot_state = RobotState()

            obs = PolicyObservation(
                image=image,
                depth=depth,
                robot_state=robot_state,
                language_instruction=instruction,
                timestep=step,
            )

            # --- Infer ---
            output: PolicyOutput = policy._time_predict(obs)

            # --- Act ---
            try:
                await robot.send_action(output.action)
            except Exception as exc:
                logger.warning("PolicyRouter: send_action error at step %d: %s.", step, exc)

            step += 1

            if output.done_signal:
                logger.info(
                    "PolicyRouter: done_signal received at step %d "
                    "(avg latency %.1f ms).",
                    step,
                    policy.avg_inference_ms,
                )
                return True

            # --- Rate limit ---
            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        logger.warning(
            "PolicyRouter: reached max_steps=%d without done_signal.", max_steps
        )
        return False
