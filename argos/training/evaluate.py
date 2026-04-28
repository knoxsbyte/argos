"""
argos.training.evaluate — Policy evaluation harness for ARGOS.

Runs trained policies in simulation (MuJoCo or mock) for a configurable
number of episodes per task type and produces structured EvalResult objects
and a markdown summary report.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for policy evaluation."""

    num_episodes: int = 50
    max_steps_per_episode: int = 500
    task_types: list[str] | None = None   # None = all available tasks
    render: bool = False
    record_video: bool = False
    output_dir: Path | None = None


@dataclass
class EvalResult:
    """Evaluation results for a single task type."""

    task_type: str
    success_rate: float
    avg_completion_time: float            # seconds
    avg_steps: float
    failure_reasons: dict[str, int] = field(default_factory=dict)  # reason → count
    episode_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PolicyEvaluator
# ---------------------------------------------------------------------------


class PolicyEvaluator:
    """Evaluates trained policies in a simulation environment.

    Compatible with CleaningEnv (MuJoCo) or any object that exposes the
    gymnasium Env interface: reset(), step(), is_success().
    When sim_env is None, a mock environment is used internally.
    """

    # Known task types with human-readable names
    _ALL_TASK_TYPES: list[str] = [
        "sweep_floor",
        "vacuum_floor",
        "mop_floor",
        "wipe_surface",
        "make_bed",
        "dust_surfaces",
        "empty_trash",
        "tidy_clutter",
        "sanitise_surface",
        "generic_cleaning",
    ]

    # Failure reason categories
    _FAILURE_TIMEOUT = "timeout"
    _FAILURE_COLLISION = "collision"
    _FAILURE_STUCK = "stuck"
    _FAILURE_POLICY_ERROR = "policy_error"

    def __init__(self, policy, sim_env, config: EvalConfig) -> None:
        """
        Parameters
        ----------
        policy:
            Any object with a predict(obs) → action method, or None for mock.
        sim_env:
            A gymnasium-compatible environment, or None to use a mock env.
        config:
            EvalConfig instance.
        """
        self.policy = policy
        self.env = sim_env
        self.config = config

        if self.config.output_dir is not None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[EvalResult]:
        """Run num_episodes per task_type and return one EvalResult per task.

        progress_callback(task_type, episode_idx, total_episodes)
        """
        task_types = self.config.task_types or self._available_task_types()
        results: list[EvalResult] = []

        for task_type in task_types:
            logger.info("Evaluating task: %s", task_type)
            result = self._evaluate_task_with_progress(
                task_type, progress_callback
            )
            results.append(result)
            logger.info(
                "  %s — success_rate=%.2f  avg_steps=%.1f",
                task_type, result.success_rate, result.avg_steps,
            )

        return results

    def evaluate_task(self, task_type: str) -> EvalResult:
        """Run evaluation for a single task type."""
        return self._evaluate_task_with_progress(task_type, progress_callback=None)

    def generate_report(self, results: list[EvalResult]) -> str:
        """Generate a markdown evaluation report."""
        lines: list[str] = [
            "# ARGOS Policy Evaluation Report\n",
            f"**Date:** {_now_str()}",
            f"**Episodes per task:** {self.config.num_episodes}",
            f"**Max steps per episode:** {self.config.max_steps_per_episode}",
            "",
            "## Results\n",
            "| Task | Success Rate | Avg Steps | Avg Time (s) |",
            "|------|-------------|-----------|--------------|",
        ]

        for r in results:
            lines.append(
                f"| {r.task_type} | {r.success_rate * 100:.1f}% "
                f"| {r.avg_steps:.1f} | {r.avg_completion_time:.2f} |"
            )

        lines.append("")
        lines.append("## Failure Analysis\n")

        for r in results:
            if not r.failure_reasons:
                continue
            lines.append(f"### {r.task_type}")
            total_failures = sum(r.failure_reasons.values())
            for reason, count in sorted(r.failure_reasons.items(), key=lambda x: -x[1]):
                pct = 100.0 * count / max(total_failures, 1)
                lines.append(f"- **{reason}**: {count} ({pct:.1f}%)")
            lines.append("")

        # Overall summary
        if results:
            overall_sr = np.mean([r.success_rate for r in results])
            lines.append(f"## Overall Success Rate: {overall_sr * 100:.1f}%\n")
            status = (
                "PASS" if overall_sr >= 0.8
                else ("MARGINAL" if overall_sr >= 0.6 else "FAIL")
            )
            lines.append(f"**Status:** {status}")

        return "\n".join(lines)

    def save_results(self, results: list[EvalResult], output_path: Path) -> None:
        """Save evaluation results as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = []
        for r in results:
            serialisable.append({
                "task_type": r.task_type,
                "success_rate": r.success_rate,
                "avg_completion_time": r.avg_completion_time,
                "avg_steps": r.avg_steps,
                "failure_reasons": r.failure_reasons,
                "num_episodes": len(r.episode_results),
                "episodes": r.episode_results,
            })

        with output_path.open("w") as f:
            json.dump({"results": serialisable, "timestamp": _now_str()}, f, indent=2)

        logger.info("Results saved to %s.", output_path)

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_task_with_progress(
        self,
        task_type: str,
        progress_callback: Callable | None,
    ) -> EvalResult:
        episode_results: list[dict] = []
        failure_reasons: dict[str, int] = {}
        n = self.config.num_episodes

        for ep_idx in range(n):
            ep_result = self._run_episode(task_type)
            episode_results.append(ep_result)

            if not ep_result["success"] and ep_result.get("failure_reason"):
                reason = ep_result["failure_reason"]
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            if progress_callback is not None:
                progress_callback(task_type, ep_idx + 1, n)

        successes = [r for r in episode_results if r["success"]]
        success_rate = len(successes) / max(n, 1)
        avg_steps = float(np.mean([r["steps"] for r in episode_results])) if episode_results else 0.0
        avg_time = float(np.mean([r["duration"] for r in episode_results])) if episode_results else 0.0

        return EvalResult(
            task_type=task_type,
            success_rate=success_rate,
            avg_completion_time=avg_time,
            avg_steps=avg_steps,
            failure_reasons=failure_reasons,
            episode_results=episode_results,
        )

    def _run_episode(self, task_type: str) -> dict:
        """Run a single episode and return a result dict.

        Returns {success, steps, duration, failure_reason}.
        """
        t_start = time.perf_counter()
        failure_reason: str | None = None

        try:
            env = self.env
            if env is None:
                # Use mock environment
                env = _MockEnv(task_type=task_type, max_steps=self.config.max_steps_per_episode)

            obs, _ = env.reset()
            if self.policy is not None:
                self.policy.reset() if hasattr(self.policy, "reset") else None

            prev_action = None
            stuck_counter = 0

            for step in range(self.config.max_steps_per_episode):
                # Get action from policy
                try:
                    if self.policy is not None:
                        action = self._policy_predict(obs)
                    else:
                        action = _random_action(29)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Policy error at step %d: %s", step, exc)
                    failure_reason = self._FAILURE_POLICY_ERROR
                    break

                # Detect stuck (action barely changing)
                if prev_action is not None:
                    delta = float(np.max(np.abs(np.array(action) - np.array(prev_action))))
                    if delta < 1e-4:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0

                    if stuck_counter > 30:
                        failure_reason = self._FAILURE_STUCK
                        break

                prev_action = action

                obs, reward, terminated, truncated, info = env.step(
                    np.array(action, dtype=np.float32)
                )

                # Check for collision in info
                if info.get("collision", False):
                    failure_reason = self._FAILURE_COLLISION

                if terminated:
                    if hasattr(env, "is_success") and env.is_success():
                        failure_reason = None  # success
                    elif not info.get("success", False):
                        failure_reason = failure_reason or self._FAILURE_COLLISION
                    break

                if truncated:
                    failure_reason = self._FAILURE_TIMEOUT
                    break

            else:
                # Loop exhausted without break
                failure_reason = self._FAILURE_TIMEOUT

            success = failure_reason is None and (
                (hasattr(env, "is_success") and env.is_success())
                or (info.get("success", False) if "info" in dir() else False)
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Episode failed with exception: %s", exc)
            failure_reason = self._FAILURE_POLICY_ERROR
            success = False
            step = 0

        duration = time.perf_counter() - t_start

        return {
            "success": success,
            "steps": step + 1 if "step" in dir() else 0,
            "duration": round(duration, 3),
            "failure_reason": failure_reason,
        }

    def _policy_predict(self, obs: dict) -> np.ndarray:
        """Call policy.predict() with appropriate obs formatting."""
        if hasattr(self.policy, "predict"):
            # Try generic predict(obs_dict) first
            try:
                result = self.policy.predict(obs)
                # Handle both raw array and PolicyOutput-like object
                if hasattr(result, "action"):
                    action = result.action
                    if hasattr(action, "joint_targets"):
                        return np.array(action.joint_targets, dtype=np.float32)
                    return np.array(action, dtype=np.float32)
                return np.array(result, dtype=np.float32)
            except Exception:  # noqa: BLE001
                pass
        return _random_action(29)

    def _available_task_types(self) -> list[str]:
        """Return task types from the env if available, else all known types."""
        if self.env is not None and hasattr(self.env, "task_type"):
            return [self.env.task_type]
        return self._ALL_TASK_TYPES[:3]  # default: first 3 for quick eval


# ---------------------------------------------------------------------------
# Mock environment (no MuJoCo)
# ---------------------------------------------------------------------------


class _MockEnv:
    """Minimal mock environment for evaluation without MuJoCo."""

    def __init__(self, task_type: str = "sweep_floor", max_steps: int = 500) -> None:
        self.task_type = task_type
        self._max_steps = max_steps
        self._step_count = 0
        self._coverage = 0.0
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self._step_count = 0
        self._coverage = 0.0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1
        # Simulate gradual coverage increase based on action magnitude
        delta = float(np.linalg.norm(action)) * 0.001
        self._coverage = min(1.0, self._coverage + delta + self._rng.exponential(0.002))
        reward = delta
        terminated = self._coverage >= 0.95
        truncated = self._step_count >= self._max_steps
        info = {"success": terminated, "coverage": self._coverage, "collision": False}
        return self._obs(), reward, terminated, truncated, info

    def is_success(self) -> bool:
        return self._coverage >= 0.95

    def _obs(self) -> dict:
        h, w = 224, 224
        return {
            "rgb": self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
            "depth": self._rng.random((h, w)).astype(np.float32),
            "robot_state": np.zeros(29, dtype=np.float32),
            "language_instruction": f"Perform {self.task_type}.",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_action(dim: int) -> np.ndarray:
    """Return a small random action vector."""
    return np.random.normal(0.0, 0.01, dim).astype(np.float32)


def _now_str() -> str:
    """Current UTC time as ISO-8601 string."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
