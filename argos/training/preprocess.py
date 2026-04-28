"""
argos.training.preprocess — Pose estimation and action labeling for ARGOS.

PoseEstimator: estimates human hand/body pose from video frames using
MediaPipe Holistic when available, falling back to optical-flow-based
hand motion estimation.

ActionLabeler: segments video into discrete action clips and converts
human wrist motion into robot action vectors suitable for imitation learning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from argos.training.ingest import Episode, VideoFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Optical-flow fallback unavailable; using zero-velocity.")

try:
    import mediapipe as _mp  # type: ignore[import]
    _MP_AVAILABLE = True
    logger.info("MediaPipe detected — full holistic pose estimation active.")
except ImportError:
    _mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False
    logger.warning("mediapipe not installed. PoseEstimator using optical-flow fallback.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# G1 has 29 DoF (arms + torso + grippers)
_ACTION_DIM = 29
# MediaPipe hand landmark count
_HAND_LANDMARKS = 21
# MediaPipe pose landmark count
_POSE_LANDMARKS = 33
# Wrist landmark indices in MediaPipe pose
_LEFT_WRIST_IDX = 15
_RIGHT_WRIST_IDX = 16
# Hand openness: distance between tip of index (8) and thumb (4) relative to palm
_INDEX_TIP_IDX = 8
_THUMB_TIP_IDX = 4
_WRIST_IDX = 0  # in hand landmark set


def _zeros_hand() -> np.ndarray:
    return np.zeros((_HAND_LANDMARKS, 3), dtype=np.float32)


def _zeros_pose() -> np.ndarray:
    return np.zeros((_POSE_LANDMARKS, 3), dtype=np.float32)


def _zeros_3() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _empty_pose_dict() -> dict:
    return {
        "left_hand": None,
        "right_hand": None,
        "pose": None,
        "wrist_velocity_left": _zeros_3(),
        "wrist_velocity_right": _zeros_3(),
    }


# ---------------------------------------------------------------------------
# PoseEstimator
# ---------------------------------------------------------------------------


class PoseEstimator:
    """Estimates human hand/body pose from video frames.

    Uses MediaPipe Holistic (hands + body) when available.
    Falls back to simple optical flow for hand motion estimation when
    MediaPipe is absent.  Falls back to zero velocities when neither
    MediaPipe nor cv2 are available.
    """

    def __init__(self, use_mediapipe: bool = True) -> None:
        self._holistic = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

        if use_mediapipe and _MP_AVAILABLE:
            try:
                self._holistic = _mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                )
                logger.debug("MediaPipe Holistic initialised.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("MediaPipe Holistic init failed (%s); using optical-flow fallback.", exc)
                self._holistic = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray) -> dict:
        """Estimate pose for a single RGB frame (HxWx3 uint8).

        Returns
        -------
        dict with keys:
            ``left_hand``   : np.ndarray (21, 3) or None
            ``right_hand``  : np.ndarray (21, 3) or None
            ``pose``        : np.ndarray (33, 3) or None
            ``wrist_velocity_left``  : np.ndarray (3,)
            ``wrist_velocity_right`` : np.ndarray (3,)
        """
        if self._holistic is not None:
            return self._estimate_mediapipe(frame)
        if _CV2_AVAILABLE:
            return self._estimate_optical_flow(frame)
        return _empty_pose_dict()

    def estimate_sequence(self, frames: list[np.ndarray]) -> list[dict]:
        """Estimate pose for each frame and compute wrist velocities across frames."""
        if not frames:
            return []

        results = [self.estimate(f) for f in frames]

        # Compute velocities as finite differences of wrist positions across frames
        for i in range(len(results)):
            for side, wrist_idx in [("left", _LEFT_WRIST_IDX), ("right", _RIGHT_WRIST_IDX)]:
                vel_key = f"wrist_velocity_{side}"
                pose_key = "pose"

                if i == 0:
                    results[i][vel_key] = _zeros_3()
                    continue

                prev_pose = results[i - 1].get(pose_key)
                curr_pose = results[i].get(pose_key)

                if prev_pose is not None and curr_pose is not None:
                    delta = curr_pose[wrist_idx] - prev_pose[wrist_idx]
                    results[i][vel_key] = delta.astype(np.float32)
                else:
                    results[i][vel_key] = results[i - 1].get(vel_key, _zeros_3()).copy()

        return results

    # ------------------------------------------------------------------
    # MediaPipe path
    # ------------------------------------------------------------------

    def _estimate_mediapipe(self, frame: np.ndarray) -> dict:
        """Run MediaPipe Holistic on a RGB frame."""
        result = self._holistic.process(frame)

        def _landmarks_to_array(lm_list, n: int) -> np.ndarray | None:
            if lm_list is None:
                return None
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in lm_list.landmark],
                dtype=np.float32,
            )
            if arr.shape[0] != n:
                return None
            return arr

        left_hand = _landmarks_to_array(result.left_hand_landmarks, _HAND_LANDMARKS)
        right_hand = _landmarks_to_array(result.right_hand_landmarks, _HAND_LANDMARKS)
        pose = _landmarks_to_array(result.pose_landmarks, _POSE_LANDMARKS)

        # Wrist velocity: derive from pose if available, else hand landmarks
        wrist_vel_left = _zeros_3()
        wrist_vel_right = _zeros_3()
        if pose is not None and hasattr(self, "_prev_pose"):
            prev_pose = self._prev_pose
            if prev_pose is not None:
                wrist_vel_left = pose[_LEFT_WRIST_IDX] - prev_pose[_LEFT_WRIST_IDX]
                wrist_vel_right = pose[_RIGHT_WRIST_IDX] - prev_pose[_RIGHT_WRIST_IDX]

        self._prev_pose = pose  # type: ignore[attr-defined]

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose": pose,
            "wrist_velocity_left": wrist_vel_left.astype(np.float32),
            "wrist_velocity_right": wrist_vel_right.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Optical-flow fallback
    # ------------------------------------------------------------------

    def _estimate_optical_flow(self, frame: np.ndarray) -> dict:
        """Use sparse Lucas-Kanade optical flow to estimate hand velocity."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        result = _empty_pose_dict()

        if self._prev_gray is None or self._prev_pts is None:
            # Detect good features to track (hands tend to be in lower 2/3 of frame)
            mask = np.zeros_like(gray)
            mask[h // 3:, :] = 255
            pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=20, qualityLevel=0.01, minDistance=10, mask=mask
            )
            self._prev_gray = gray
            self._prev_pts = pts
            return result

        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_gray = gray
            return result

        # Track points
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        good_prev = self._prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) > 0:
            flow = good_next - good_prev  # (N, 2)
            # Split into left/right halves of frame
            mid_x = w // 2
            left_mask = good_prev[:, 0] < mid_x
            right_mask = ~left_mask

            if left_mask.any():
                mean_left = flow[left_mask].mean(axis=0)  # (2,)
                result["wrist_velocity_left"] = np.array(
                    [mean_left[0] / w, mean_left[1] / h, 0.0], dtype=np.float32
                )

            if right_mask.any():
                mean_right = flow[right_mask].mean(axis=0)
                result["wrist_velocity_right"] = np.array(
                    [mean_right[0] / w, mean_right[1] / h, 0.0], dtype=np.float32
                )

        self._prev_gray = gray
        self._prev_pts = good_next.reshape(-1, 1, 2) if len(good_next) > 0 else None
        return result

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None


# ---------------------------------------------------------------------------
# ActionLabeler
# ---------------------------------------------------------------------------


class ActionLabeler:
    """Segments video into discrete action clips and assigns robot action vectors.

    The labeler converts human wrist motion (from PoseEstimator output)
    into robot-compatible action arrays of shape (T, 29).
    """

    # Action type → dominant motion axis / pattern
    _ACTION_PATTERNS: dict[str, str] = {
        "sweep_floor":      "lateral_sweep",
        "vacuum_floor":     "lateral_sweep",
        "mop_floor":        "lateral_sweep",
        "wipe_surface":     "circular_wipe",
        "make_bed":         "bilateral_spread",
        "dust_surfaces":    "vertical_wipe",
        "empty_trash":      "lift_and_carry",
        "tidy_clutter":     "pick_and_place",
        "sanitise_surface": "circular_wipe",
        "generic_cleaning": "lateral_sweep",
    }

    def __init__(
        self,
        min_segment_frames: int = 15,
        velocity_threshold: float = 0.02,
    ) -> None:
        self.min_segment_frames = min_segment_frames
        self.velocity_threshold = velocity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, episode: "Episode", poses: list[dict]) -> list[dict]:
        """Find action segments by detecting motion starts/stops.

        Returns a list of segment dicts:
        {
          "start_frame": int,
          "end_frame":   int,
          "action_type": str,
          "confidence":  float,
        }
        """
        if not poses:
            return []

        energy = self._motion_energy(poses)
        task_type = episode.metadata.get("task_type", "generic_cleaning")

        # Smooth energy with a simple box filter
        kernel_size = max(3, self.min_segment_frames // 3)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energy, kernel, mode="same")

        segments: list[dict] = []
        in_segment = False
        seg_start = 0

        for i, e in enumerate(smoothed):
            if not in_segment and e > self.velocity_threshold:
                in_segment = True
                seg_start = i
            elif in_segment and (e <= self.velocity_threshold or i == len(smoothed) - 1):
                in_segment = False
                seg_end = i
                seg_len = seg_end - seg_start
                if seg_len >= self.min_segment_frames:
                    motion_pattern = energy[seg_start:seg_end]
                    action_type = self._infer_action_type(motion_pattern, task_type)
                    # Confidence: how far above threshold the mean energy is
                    mean_e = float(motion_pattern.mean()) if len(motion_pattern) > 0 else 0.0
                    confidence = float(
                        np.clip((mean_e - self.velocity_threshold) / (self.velocity_threshold + 1e-6), 0.0, 1.0)
                    )
                    segments.append({
                        "start_frame": seg_start,
                        "end_frame": seg_end,
                        "action_type": action_type,
                        "confidence": round(confidence, 3),
                    })

        return segments

    def label_segment(
        self,
        frames: list["VideoFrame"],
        poses: list[dict],
        task_type: str,
    ) -> np.ndarray:
        """Convert human wrist motion to robot action vector array.

        Mapping:
        - Wrist position delta → arm joint velocity targets (indices 0-13: left arm, 14-27: right arm)
        - Hand openness → gripper command (index 28: left gripper, index 27: right gripper)
        - Remaining joints (torso): zeroed

        Returns (T, action_dim) float32 array where action_dim = 29.
        """
        T = len(poses)
        if T == 0:
            return np.zeros((0, _ACTION_DIM), dtype=np.float32)

        actions = np.zeros((T, _ACTION_DIM), dtype=np.float32)

        for t, pose_dict in enumerate(poses):
            # Left arm joints (0-6): map from left wrist velocity
            vel_left = pose_dict.get("wrist_velocity_left", _zeros_3())
            vel_right = pose_dict.get("wrist_velocity_right", _zeros_3())

            # Scale velocities to joint-space targets (heuristic mapping)
            scale = 5.0  # normalised landmark velocity → radians
            # Left arm: 7 joints (shoulder x3, elbow x1, wrist x3)
            actions[t, 0:3] = vel_left * scale            # shoulder
            actions[t, 3] = np.linalg.norm(vel_left) * scale * 0.5  # elbow
            actions[t, 4:7] = vel_left * scale * 0.3     # wrist

            # Right arm: 7 joints (indices 7-13)
            actions[t, 7:10] = vel_right * scale
            actions[t, 10] = np.linalg.norm(vel_right) * scale * 0.5
            actions[t, 11:14] = vel_right * scale * 0.3

            # Torso (indices 14-20): zeros (no reliable torso mapping from monocular)
            # Left leg (21-24), right leg (25-28): zeros

            # Gripper commands: index 27 (right) and 28 (left)
            left_hand = pose_dict.get("left_hand")
            right_hand = pose_dict.get("right_hand")
            actions[t, 28] = self._hand_openness(left_hand)   # left gripper
            actions[t, 27] = self._hand_openness(right_hand)  # right gripper

        # Clip to plausible joint velocity range [-1, 1]
        actions = np.clip(actions, -1.0, 1.0)
        return actions.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _motion_energy(self, poses: list[dict]) -> np.ndarray:
        """Compute per-frame motion energy from wrist velocity magnitudes."""
        energy = np.zeros(len(poses), dtype=np.float32)
        for i, p in enumerate(poses):
            vel_l = p.get("wrist_velocity_left", _zeros_3())
            vel_r = p.get("wrist_velocity_right", _zeros_3())
            energy[i] = float(np.linalg.norm(vel_l) + np.linalg.norm(vel_r))
        return energy

    def _infer_action_type(self, motion_pattern: np.ndarray, task_type: str) -> str:
        """Map motion pattern to action type string.

        Heuristic: look at variance and mean to characterise the motion,
        then consult the task type to pick the best label.
        """
        if len(motion_pattern) == 0:
            return self._ACTION_PATTERNS.get(task_type, "generic_motion")

        mean_e = float(motion_pattern.mean())
        var_e = float(motion_pattern.var())

        # High variance → pick-and-place or discrete action
        # Low variance, steady → sweep/wipe
        if var_e > 0.01 and mean_e > 0.05:
            pattern = "pick_and_place"
        elif mean_e > 0.08:
            pattern = "lateral_sweep"
        elif mean_e > 0.03:
            pattern = "circular_wipe"
        else:
            pattern = "slow_approach"

        # Override with task-specific pattern if confident
        task_pattern = self._ACTION_PATTERNS.get(task_type)
        if task_pattern is not None and mean_e > self.velocity_threshold * 2:
            return task_pattern

        return pattern

    @staticmethod
    def _hand_openness(hand_landmarks: np.ndarray | None) -> float:
        """Compute hand openness (0=closed, 1=open) from MediaPipe hand landmarks.

        Uses the normalised distance between thumb tip (4) and index tip (8)
        relative to the palm size (wrist to middle-finger base).
        """
        if hand_landmarks is None or hand_landmarks.shape[0] < 9:
            return 0.5  # default: half-open

        thumb_tip = hand_landmarks[_THUMB_TIP_IDX]
        index_tip = hand_landmarks[_INDEX_TIP_IDX]
        wrist = hand_landmarks[_WRIST_IDX]
        middle_base = hand_landmarks[9] if hand_landmarks.shape[0] > 9 else wrist

        tip_dist = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(wrist - middle_base)) + 1e-6

        openness = float(np.clip(tip_dist / (palm_size * 1.5), 0.0, 1.0))
        return openness
