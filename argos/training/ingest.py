"""
argos.training.ingest — Video ingestion pipeline for ARGOS training data.

Extracts frames from cleaning demonstration videos (MP4/AVI/MOV),
samples at a target FPS, resizes, and bundles into Episode objects
ready for preprocessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. VideoIngestor will run in mock mode.")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Keyword → task_type mapping used by _infer_task_type
_TASK_KEYWORDS: dict[str, str] = {
    "sweep":    "sweep_floor",
    "broom":    "sweep_floor",
    "vacuum":   "vacuum_floor",
    "hoover":   "vacuum_floor",
    "mop":      "mop_floor",
    "wipe":     "wipe_surface",
    "scrub":    "wipe_surface",
    "wash":     "wipe_surface",
    "clean":    "wipe_surface",
    "bed":      "make_bed",
    "sheet":    "make_bed",
    "pillow":   "make_bed",
    "dust":     "dust_surfaces",
    "shelf":    "dust_surfaces",
    "trash":    "empty_trash",
    "bin":      "empty_trash",
    "garbage":  "empty_trash",
    "tidy":     "tidy_clutter",
    "clutter":  "tidy_clutter",
    "organise": "tidy_clutter",
    "organize": "tidy_clutter",
    "sanitise": "sanitise_surface",
    "sanitize": "sanitise_surface",
    "disinfect":"sanitise_surface",
}


@dataclass
class VideoFrame:
    """A single decoded video frame with optional depth."""

    frame_idx: int
    timestamp: float          # seconds from video start
    rgb: np.ndarray           # HxWx3 uint8
    depth: np.ndarray | None  # HxW float32, None if not available


@dataclass
class Episode:
    """A complete demonstration episode extracted from a video file."""

    episode_id: str
    video_path: str
    frames: list[VideoFrame]
    fps: float
    duration: float
    metadata: dict = field(default_factory=dict)
    # metadata keys: task_type, language_instruction, source_fps, total_source_frames


# ---------------------------------------------------------------------------
# VideoIngestor
# ---------------------------------------------------------------------------


class VideoIngestor:
    """Extracts frames from cleaning video footage.

    When opencv-python is not installed, returns a single synthetic
    mock frame per file so the rest of the pipeline can be exercised
    without a GPU or camera hardware.
    """

    def __init__(
        self,
        target_fps: float = 15.0,
        target_size: tuple[int, int] = (224, 224),
        max_frames: int = 1000,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        self.target_fps = target_fps
        self.target_size = target_size  # (width, height)
        self.max_frames = max_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        video_path: Path,
        task_type: str,
        language_instruction: str = "",
    ) -> Episode:
        """Load a single video file and return an Episode.

        Steps:
        1. Open with cv2.VideoCapture (or mock path if unavailable).
        2. Sample at target_fps (skip frames to match source fps).
        3. Resize each frame to target_size.
        4. Convert BGR → RGB.
        5. If a paired depth video exists (same stem + '_depth'): load it.
        6. Return Episode.
        """
        video_path = Path(video_path)
        episode_id = video_path.stem

        if not _CV2_AVAILABLE:
            return self._mock_episode(video_path, task_type, language_instruction, episode_id)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video %s — returning mock episode.", video_path)
            cap.release()
            return self._mock_episode(video_path, task_type, language_instruction, episode_id)

        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total_source_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_source_frames / source_fps if source_fps > 0 else 0.0

        # Load paired depth video if available
        depth_cap = self._open_depth_cap(video_path)
        depth_frames: dict[int, np.ndarray] = {}
        if depth_cap is not None:
            depth_frames = self._load_depth_frames(depth_cap, source_fps)
            depth_cap.release()

        frames: list[VideoFrame] = list(
            self._sample_frames(cap, source_fps=source_fps, depth_frames=depth_frames)
        )
        cap.release()

        if not language_instruction:
            language_instruction = self._default_instruction(task_type)

        return Episode(
            episode_id=episode_id,
            video_path=str(video_path),
            frames=frames,
            fps=self.target_fps,
            duration=duration,
            metadata={
                "task_type": task_type,
                "language_instruction": language_instruction,
                "source_fps": source_fps,
                "total_source_frames": total_source_frames,
            },
        )

    def ingest_directory(
        self,
        video_dir: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Episode]:
        """Scan directory for video files and ingest each one.

        For each video, looks for a sidecar ``metadata.json`` (same stem,
        same directory) with optional keys ``task_type`` and
        ``language_instruction``.  If absent, infers task_type from the
        filename/directory name.

        progress_callback(current, total, filename) — called after each file.
        """
        video_dir = Path(video_dir)
        video_files = sorted(
            p for p in video_dir.rglob("*")
            if p.suffix.lower() in _VIDEO_EXTENSIONS
            and "_depth" not in p.stem  # skip depth sidecars
        )

        episodes: list[Episode] = []
        total = len(video_files)

        for idx, vf in enumerate(video_files, start=1):
            meta = self._load_sidecar(vf)
            task_type = meta.get("task_type") or self._infer_task_type(vf)
            instruction = meta.get("language_instruction", "")

            try:
                ep = self.ingest_file(vf, task_type=task_type, language_instruction=instruction)
                episodes.append(ep)
                logger.info("Ingested %s (%d frames).", vf.name, len(ep.frames))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest %s: %s", vf, exc)

            if progress_callback is not None:
                progress_callback(idx, total, vf.name)

        return episodes

    def validate_episode(self, episode: Episode) -> tuple[bool, str]:
        """Check that the episode meets minimum quality requirements.

        Returns (is_valid, reason) where reason is empty string on success.
        """
        if len(episode.frames) < 5:
            return False, f"Too few frames: {len(episode.frames)} (min 5)"

        if len(episode.frames) > self.max_frames:
            return False, (
                f"Too many frames: {len(episode.frames)} (max {self.max_frames}). "
                "Re-ingest with a lower target_fps or higher max_frames."
            )

        sample = episode.frames[0].rgb
        if sample.ndim != 3 or sample.shape[2] != 3:
            return False, f"Frame RGB has unexpected shape: {sample.shape}"

        if sample.dtype != np.uint8:
            return False, f"Frame RGB dtype is {sample.dtype}, expected uint8"

        if episode.fps <= 0:
            return False, f"Invalid fps: {episode.fps}"

        return True, ""

    # ------------------------------------------------------------------
    # Internal frame sampling
    # ------------------------------------------------------------------

    def _sample_frames(
        self,
        cap: "cv2.VideoCapture",
        source_fps: float | None = None,
        depth_frames: dict[int, np.ndarray] | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield VideoFrame objects sampled at target_fps from a VideoCapture."""
        if source_fps is None or source_fps <= 0:
            source_fps = self.target_fps

        # Frame step: how many source frames to advance per target frame
        frame_step = max(1, source_fps / self.target_fps)
        next_source_idx = 0.0
        target_idx = 0
        source_frame_count = 0

        w, h = self.target_size  # width, height

        while target_idx < self.max_frames:
            desired_source = round(next_source_idx)

            # Seek if we need to skip frames
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos != desired_source:
                cap.set(cv2.CAP_PROP_POS_FRAMES, desired_source)

            ret, bgr = cap.read()
            if not ret:
                break

            source_frame_count += 1
            timestamp = desired_source / source_fps

            # Resize
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

            # BGR → RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth = None
            if depth_frames is not None:
                depth = depth_frames.get(desired_source)

            yield VideoFrame(
                frame_idx=target_idx,
                timestamp=timestamp,
                rgb=rgb,
                depth=depth,
            )

            target_idx += 1
            next_source_idx += frame_step

    # ------------------------------------------------------------------
    # Depth video helpers
    # ------------------------------------------------------------------

    def _open_depth_cap(self, video_path: Path) -> "cv2.VideoCapture | None":
        """Look for a paired depth video (same stem + '_depth') next to the RGB video."""
        for ext in [video_path.suffix, ".mp4", ".avi"]:
            depth_path = video_path.with_stem(video_path.stem + "_depth").with_suffix(ext)
            if depth_path.exists():
                cap = cv2.VideoCapture(str(depth_path))
                if cap.isOpened():
                    logger.debug("Paired depth video found: %s", depth_path)
                    return cap
                cap.release()
        return None

    def _load_depth_frames(
        self,
        depth_cap: "cv2.VideoCapture",
        source_fps: float,
    ) -> dict[int, np.ndarray]:
        """Load all depth frames from a depth video into a dict keyed by source frame index."""
        frames: dict[int, np.ndarray] = {}
        w, h = self.target_size
        idx = 0
        while True:
            ret, frame = depth_cap.read()
            if not ret:
                break
            # Depth videos are typically grayscale or single-channel; take first channel
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if gray.shape[:2] != (h, w):
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
            # Normalise to float32 metres (assume 16-bit scale or 0-255 → 0-10m)
            depth_m = gray.astype(np.float32) / 25.5  # 255 → 10m
            frames[idx] = depth_m
            idx += 1
        return frames

    # ------------------------------------------------------------------
    # Metadata / inference helpers
    # ------------------------------------------------------------------

    def _load_sidecar(self, video_path: Path) -> dict:
        """Load sidecar metadata JSON if present."""
        sidecar = video_path.with_suffix(".json")
        if sidecar.exists():
            try:
                with sidecar.open() as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not parse sidecar %s: %s", sidecar, exc)
        return {}

    def _infer_task_type(self, video_path: Path) -> str:
        """Infer task_type from filename and parent directory name using keywords."""
        text = (video_path.stem + " " + video_path.parent.name).lower()
        for keyword, task_type in _TASK_KEYWORDS.items():
            if keyword in text:
                return task_type
        return "generic_cleaning"

    @staticmethod
    def _default_instruction(task_type: str) -> str:
        """Return a sensible language instruction for a given task_type."""
        _INSTRUCTIONS: dict[str, str] = {
            "sweep_floor":      "Sweep the floor clean.",
            "vacuum_floor":     "Vacuum the floor.",
            "mop_floor":        "Mop the floor.",
            "wipe_surface":     "Wipe the surface clean.",
            "make_bed":         "Make the bed neatly.",
            "dust_surfaces":    "Dust all surfaces.",
            "empty_trash":      "Empty the trash bin.",
            "tidy_clutter":     "Tidy up the clutter.",
            "sanitise_surface": "Sanitise the surface.",
            "generic_cleaning": "Clean the room.",
        }
        return _INSTRUCTIONS.get(task_type, "Perform the cleaning task.")

    # ------------------------------------------------------------------
    # Mock fallback (no cv2)
    # ------------------------------------------------------------------

    def _mock_episode(
        self,
        video_path: Path,
        task_type: str,
        language_instruction: str,
        episode_id: str,
    ) -> Episode:
        """Return a minimal synthetic episode when cv2 is unavailable."""
        rng = np.random.default_rng(seed=abs(hash(str(video_path))) & 0xFFFFFFFF)
        h, w = self.target_size[1], self.target_size[0]
        n_frames = min(30, self.max_frames)
        frames = [
            VideoFrame(
                frame_idx=i,
                timestamp=i / self.target_fps,
                rgb=rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                depth=None,
            )
            for i in range(n_frames)
        ]
        if not language_instruction:
            language_instruction = self._default_instruction(task_type)
        return Episode(
            episode_id=episode_id,
            video_path=str(video_path),
            frames=frames,
            fps=self.target_fps,
            duration=n_frames / self.target_fps,
            metadata={
                "task_type": task_type,
                "language_instruction": language_instruction,
                "source_fps": self.target_fps,
                "total_source_frames": n_frames,
                "mock": True,
            },
        )
