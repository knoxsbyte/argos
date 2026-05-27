"""
argos.training.augment — Data augmentation for robot demonstration episodes.

Expands a limited set of training episodes by applying visual and kinematic
transforms. Each augmented copy introduces controlled variation so the trained
policy generalises beyond the exact footage captured.

Augmentation strategies
-----------------------
horizontal_flip  Mirror frames left-right; negate X-axis state/action components.
color_jitter     Random brightness, contrast, and saturation shift on RGB frames.
gaussian_noise   Zero-mean Gaussian noise added to state and action vectors.
speed_jitter     Resample episode at 0.8–1.2× playback speed (temporal stretch).

Usage::

    aug = DataAugmentor(strategies=["horizontal_flip", "color_jitter"])
    expanded = aug.augment_dataset(episodes, factor=3)
    # len(expanded) == len(episodes) * 3  (1 original + 2 augmented copies each)
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from argos.training.ingest import Episode, VideoFrame

logger = logging.getLogger(__name__)

ALL_STRATEGIES: tuple[str, ...] = (
    "horizontal_flip",
    "color_jitter",
    "gaussian_noise",
    "speed_jitter",
)


@dataclass
class AugmentConfig:
    """Hyperparameters for each augmentation strategy."""

    # color_jitter
    brightness_range: tuple[float, float] = (0.75, 1.25)
    contrast_range:   tuple[float, float] = (0.80, 1.20)
    saturation_range: tuple[float, float] = (0.80, 1.20)
    # gaussian_noise — fraction of the typical state/action range
    noise_std: float = 0.02
    # speed_jitter — episode replayed at this speed multiple relative to original
    speed_range: tuple[float, float] = (0.80, 1.20)


# ---------------------------------------------------------------------------
# Frame-level helpers (no OpenCV dependency)
# ---------------------------------------------------------------------------

def _flip_frame(frame: VideoFrame) -> VideoFrame:
    """Mirror RGB (and depth if present) horizontally."""
    return VideoFrame(
        frame_idx=frame.frame_idx,
        timestamp=frame.timestamp,
        rgb=frame.rgb[:, ::-1, :].copy(),
        depth=frame.depth[:, ::-1].copy() if frame.depth is not None else None,
    )


def _jitter_frame(
    frame: VideoFrame,
    brightness: float,
    contrast: float,
    saturation: float,
) -> VideoFrame:
    """Apply colour jitter to one RGB frame (pure NumPy — no OpenCV required)."""
    rgb  = frame.rgb.astype(np.float32)
    rgb *= brightness
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    rgb  = mean + contrast * (rgb - mean)
    lum  = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., np.newaxis]
    rgb  = lum + saturation * (rgb - lum)
    return VideoFrame(
        frame_idx=frame.frame_idx,
        timestamp=frame.timestamp,
        rgb=np.clip(rgb, 0, 255).astype(np.uint8),
        depth=frame.depth,
    )


# ---------------------------------------------------------------------------
# DataAugmentor
# ---------------------------------------------------------------------------

class DataAugmentor:
    """Applies configurable augmentation strategies to :class:`Episode` objects.

    Parameters
    ----------
    strategies:
        Any subset of ``ALL_STRATEGIES``. Defaults to all four.
    config:
        Fine-grained hyperparameters for each strategy.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        config: AugmentConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.strategies = list(strategies or ALL_STRATEGIES)
        self.config     = config or AugmentConfig()

        unknown = [s for s in self.strategies if s not in ALL_STRATEGIES]
        if unknown:
            raise ValueError(
                f"Unknown strategies: {unknown}. Choose from {ALL_STRATEGIES}."
            )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── public API ────────────────────────────────────────────────────────────

    def augment_episode(self, episode: Episode) -> Episode:
        """Return one new augmented copy of *episode*.

        Each call samples fresh random parameters so repeated calls on the
        same source episode produce distinct variants.
        """
        aug = copy.deepcopy(episode)

        if "speed_jitter" in self.strategies:
            aug = self._speed_jitter(aug)

        if "horizontal_flip" in self.strategies and random.random() < 0.5:
            aug = self._horizontal_flip(aug)

        if "color_jitter" in self.strategies:
            aug = self._color_jitter(aug)

        if "gaussian_noise" in self.strategies:
            aug = self._gaussian_noise(aug)

        aug.metadata = {
            **aug.metadata,
            "augmented":       True,
            "source_episode":  episode.episode_id,
            "strategies_used": self.strategies,
        }
        return aug

    def augment_dataset(
        self,
        episodes: list[Episode],
        factor: int = 2,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Episode]:
        """Expand *episodes* by *factor*, keeping originals at the front.

        ``factor=3`` returns 3× as many episodes: 1 original + 2 augmented
        copies per source episode.

        Parameters
        ----------
        episodes:
            Source episodes.
        factor:
            Dataset size multiplier (>= 1).
        progress_callback:
            Called as ``progress_callback(done, total)`` after each augmented
            episode is generated.

        Returns
        -------
        list[Episode]
        """
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if factor == 1:
            return list(episodes)

        result  = list(episodes)
        total   = len(episodes) * (factor - 1)
        done    = 0

        for ep in episodes:
            for copy_idx in range(factor - 1):
                aug = self.augment_episode(ep)
                aug.episode_id = f"{ep.episode_id}_aug{copy_idx + 1}"
                result.append(aug)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        logger.info(
            "DataAugmentor: %d episodes × factor=%d → %d total  strategies=%s",
            len(episodes), factor, len(result), self.strategies,
        )
        return result

    def summary(self) -> dict:
        """Return a serialisable config summary."""
        return {
            "strategies": self.strategies,
            "config": {
                "brightness_range": self.config.brightness_range,
                "contrast_range":   self.config.contrast_range,
                "saturation_range": self.config.saturation_range,
                "noise_std":        self.config.noise_std,
                "speed_range":      self.config.speed_range,
            },
        }

    # ── strategy implementations ──────────────────────────────────────────────

    def _horizontal_flip(self, episode: Episode) -> Episode:
        episode.frames = [_flip_frame(f) for f in episode.frames]
        if "actions" in episode.metadata:
            actions = np.array(episode.metadata["actions"], dtype=np.float32)
            if actions.ndim == 2 and actions.shape[1] >= 1:
                actions[:, 0] *= -1
                episode.metadata["actions"] = actions.tolist()
        return episode

    def _color_jitter(self, episode: Episode) -> Episode:
        b = random.uniform(*self.config.brightness_range)
        c = random.uniform(*self.config.contrast_range)
        s = random.uniform(*self.config.saturation_range)
        episode.frames = [_jitter_frame(f, b, c, s) for f in episode.frames]
        return episode

    def _gaussian_noise(self, episode: Episode) -> Episode:
        """Add small Gaussian noise to state/action vectors in metadata."""
        for key in ("states", "actions"):
            if key in episode.metadata:
                arr   = np.array(episode.metadata[key], dtype=np.float32)
                noise = np.random.normal(0.0, self.config.noise_std, arr.shape).astype(np.float32)
                episode.metadata[key] = (arr + noise).tolist()
        return episode

    def _speed_jitter(self, episode: Episode) -> Episode:
        """Resample frames to simulate faster or slower playback."""
        if len(episode.frames) < 2:
            return episode

        speed  = random.uniform(*self.config.speed_range)
        n_orig = len(episode.frames)
        n_new  = max(2, int(round(n_orig / speed)))

        indices = np.linspace(0, n_orig - 1, n_new)
        new_frames: list[VideoFrame] = []
        for i, fi in enumerate(indices):
            src = episode.frames[int(fi)]
            new_frames.append(VideoFrame(
                frame_idx=i,
                timestamp=round(src.timestamp / speed, 4),
                rgb=src.rgb,
                depth=src.depth,
            ))

        episode.frames                   = new_frames
        episode.fps                      = episode.fps * speed
        episode.duration                 = episode.duration / speed
        episode.metadata["speed_factor"] = round(speed, 3)
        return episode
