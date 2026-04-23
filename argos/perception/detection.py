"""
argos.perception.detection — Object and dirt detection for ARGOS.

Two main concerns:
1. ObjectDetector   — YOLOv8-based detection of household objects in RGB frames.
2. DirtDetector     — CV-based detection of dirt/stains on surfaces.
3. BedMakingDetector — Specialised heuristic for assessing bed state.

When ultralytics is not installed the ObjectDetector falls back to a mock
implementation that returns synthetic detections so the rest of the pipeline
can be exercised without a GPU or model weights.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from argos.perception.scene import DetectedObject, ObjectCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies with graceful fallback
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. DirtDetector and BedMakingDetector using numpy fallbacks.")

try:
    from ultralytics import YOLO as _YOLO  # type: ignore[import]
    _YOLO_AVAILABLE = True
    logger.info("ultralytics detected — real YOLOv8 mode active.")
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. ObjectDetector running in mock mode.")


# ---------------------------------------------------------------------------
# COCO class → ObjectCategory + pickup/cleaning flags
# ---------------------------------------------------------------------------

# (category, needs_pickup, needs_cleaning)
_COCO_MAP: dict[str, tuple[ObjectCategory, bool, bool]] = {
    # --- clutter that should be picked up ---
    "bottle":        (ObjectCategory.CLUTTER, True,  False),
    "cup":           (ObjectCategory.CLUTTER, True,  False),
    "wine glass":    (ObjectCategory.CLUTTER, True,  False),
    "fork":          (ObjectCategory.CLUTTER, True,  False),
    "knife":         (ObjectCategory.CLUTTER, True,  False),
    "spoon":         (ObjectCategory.CLUTTER, True,  False),
    "bowl":          (ObjectCategory.CLUTTER, True,  False),
    "banana":        (ObjectCategory.TRASH,   True,  False),
    "apple":         (ObjectCategory.TRASH,   True,  False),
    "orange":        (ObjectCategory.TRASH,   True,  False),
    "sandwich":      (ObjectCategory.TRASH,   True,  False),
    "pizza":         (ObjectCategory.TRASH,   True,  False),
    "cake":          (ObjectCategory.TRASH,   True,  False),
    "book":          (ObjectCategory.CLUTTER, True,  False),
    "cell phone":    (ObjectCategory.CLUTTER, True,  False),
    "remote":        (ObjectCategory.CLUTTER, True,  False),
    "keyboard":      (ObjectCategory.CLUTTER, True,  False),
    "mouse":         (ObjectCategory.CLUTTER, True,  False),
    "scissors":      (ObjectCategory.CLUTTER, True,  False),
    "toothbrush":    (ObjectCategory.CLUTTER, True,  False),
    "hair drier":    (ObjectCategory.CLUTTER, True,  False),
    "umbrella":      (ObjectCategory.CLUTTER, True,  False),
    "handbag":       (ObjectCategory.CLUTTER, True,  False),
    "backpack":      (ObjectCategory.CLUTTER, True,  False),
    "suitcase":      (ObjectCategory.CLUTTER, False, False),
    "sports ball":   (ObjectCategory.CLUTTER, True,  False),
    "frisbee":       (ObjectCategory.CLUTTER, True,  False),
    "skis":          (ObjectCategory.CLUTTER, False, False),
    "snowboard":     (ObjectCategory.CLUTTER, False, False),
    "kite":          (ObjectCategory.CLUTTER, True,  False),
    "baseball bat":  (ObjectCategory.CLUTTER, False, False),
    "baseball glove":(ObjectCategory.CLUTTER, True,  False),
    "skateboard":    (ObjectCategory.CLUTTER, False, False),
    "surfboard":     (ObjectCategory.CLUTTER, False, False),
    "tennis racket": (ObjectCategory.CLUTTER, False, False),
    "clock":         (ObjectCategory.CLUTTER, False, False),
    "vase":          (ObjectCategory.CLUTTER, False, False),
    "tie":           (ObjectCategory.CLUTTER, True,  False),
    "suitcase":      (ObjectCategory.CLUTTER, False, False),
    # --- furniture ---
    "chair":         (ObjectCategory.FURNITURE, False, True),
    "couch":         (ObjectCategory.FURNITURE, False, True),
    "bed":           (ObjectCategory.FURNITURE, False, True),
    "dining table":  (ObjectCategory.FURNITURE, False, True),
    "toilet":        (ObjectCategory.FURNITURE, False, True),
    "tv":            (ObjectCategory.FURNITURE, False, False),
    "laptop":        (ObjectCategory.FURNITURE, False, False),
    "microwave":     (ObjectCategory.FURNITURE, False, True),
    "oven":          (ObjectCategory.FURNITURE, False, True),
    "toaster":       (ObjectCategory.FURNITURE, False, True),
    "sink":          (ObjectCategory.FURNITURE, False, True),
    "refrigerator":  (ObjectCategory.FURNITURE, False, True),
    "potted plant":  (ObjectCategory.FURNITURE, False, False),
    # --- cleaning tools ---
    "broom":         (ObjectCategory.CLEANING_TOOL, False, False),
    "mop":           (ObjectCategory.CLEANING_TOOL, False, False),
    "vacuum":        (ObjectCategory.CLEANING_TOOL, False, False),
}

_DEFAULT_ENTRY: tuple[ObjectCategory, bool, bool] = (ObjectCategory.UNKNOWN, False, False)


def _coco_to_detected_object(
    label: str,
    bbox: tuple[int, int, int, int],
    confidence: float,
    position: np.ndarray | None = None,
) -> DetectedObject:
    """Convert a COCO detection into a DetectedObject."""
    cat, needs_pickup, needs_cleaning = _COCO_MAP.get(label, _DEFAULT_ENTRY)
    if position is None:
        position = np.zeros(3, dtype=np.float32)
    return DetectedObject(
        object_id=str(uuid.uuid4())[:8],
        category=cat,
        label=label,
        position=position,
        bounding_box=bbox,
        confidence=confidence,
        needs_pickup=needs_pickup,
        needs_cleaning=needs_cleaning,
    )


# ---------------------------------------------------------------------------
# Object detector
# ---------------------------------------------------------------------------


class ObjectDetector:
    """Detects objects in RGB frames using YOLOv8.

    Falls back to a mock that returns synthetic detections when ultralytics
    or the model weights are unavailable (useful for unit tests and CI).
    """

    CATEGORY_MAP = {k: v[0] for k, v in _COCO_MAP.items()}

    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self._model: Any = None

        if _YOLO_AVAILABLE:
            try:
                self._model = _YOLO(f"yolov8{model_size}.pt")
                self._model.to(device)
                logger.info("YOLOv8%s loaded on %s.", model_size, device)
            except Exception as exc:
                logger.warning(
                    "Failed to load YOLOv8%s: %s — using mock detections.", model_size, exc
                )
                self._model = None
        else:
            logger.warning("YOLOv8 unavailable — ObjectDetector in mock mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """Run YOLOv8 on *frame* (HxWx3 BGR uint8).

        Returns a list of DetectedObject instances with 2-D bounding boxes.
        3-D positions are left at the origin; use detect_with_depth() for
        full 3-D localisation.
        """
        if self._model is None:
            return self._mock_detect(frame)

        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_with_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_intrinsics: dict[str, float],
    ) -> list[DetectedObject]:
        """Detect objects and compute their 3-D world-frame positions.

        Uses the standard pinhole back-projection:
            X = (u − cx) * Z / fx
            Y = (v − cy) * Z / fy
            Z = depth at (u, v)

        Parameters
        ----------
        rgb:
            HxWx3 BGR uint8 frame.
        depth:
            HxW float32 depth in metres (aligned to rgb).
        camera_intrinsics:
            Dict with keys ``fx``, ``fy``, ``cx``, ``cy``.
        """
        detections = self.detect(rgb)
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        h, w = depth.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bounding_box
            # Centre of bounding box, clamped to image bounds
            u = int(np.clip((x1 + x2) / 2, 0, w - 1))
            v = int(np.clip((y1 + y2) / 2, 0, h - 1))

            # Sample a 5×5 patch around the centre and use the median depth
            # to reduce the impact of depth holes / noise.
            u0, u1 = max(0, u - 2), min(w, u + 3)
            v0, v1 = max(0, v - 2), min(h, v + 3)
            patch = depth[v0:v1, u0:u1]
            valid = patch[patch > 0.0]
            if valid.size == 0:
                # No valid depth — leave position at origin
                continue
            Z = float(np.median(valid))

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            det.position = np.array([X, Y, Z], dtype=np.float32)

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(self, results: list[Any]) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # (N, 4)
            confs = result.boxes.conf.cpu().numpy()          # (N,)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            names: dict[int, str] = result.names

            for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                if float(conf) < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                label = names.get(cls_id, "unknown")
                det = _coco_to_detected_object(label, (x1, y1, x2, y2), float(conf))
                detections.append(det)

        return detections

    def _mock_detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """Return a fixed set of synthetic detections for dev/test purposes."""
        h, w = frame.shape[:2]
        return [
            _coco_to_detected_object(
                label="bottle",
                bbox=(w // 4, h // 4, w // 4 + 60, h // 4 + 120),
                confidence=0.82,
                position=np.array([1.0, 0.5, 0.0], dtype=np.float32),
            ),
            _coco_to_detected_object(
                label="cup",
                bbox=(w // 2, h // 3, w // 2 + 50, h // 3 + 60),
                confidence=0.74,
                position=np.array([1.5, -0.3, 0.0], dtype=np.float32),
            ),
        ]


# ---------------------------------------------------------------------------
# Dirt detector
# ---------------------------------------------------------------------------


class DirtDetector:
    """Detects dirt, stains, and mess on surfaces using classical CV.

    No external model is required.  The pipeline works in LAB colour space
    to separate luminance from chrominance, making it robust to lighting
    variation.

    When opencv-python is not installed a numpy-only fallback is used that
    operates on the raw BGR channels directly (slightly less accurate but
    fully functional for development and testing).
    """

    # LAB colour-anomaly thresholds
    _LAB_ANOMALY_THRESH: float = 18.0   # Euclidean distance in LAB space
    _MORPH_KERNEL_SIZE: int = 5

    def __init__(self) -> None:
        self._has_cv2 = _CV2_AVAILABLE
        if self._has_cv2:
            self._kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._MORPH_KERNEL_SIZE, self._MORPH_KERNEL_SIZE),
            )
        else:
            # Simple square kernel for numpy fallback
            k = self._MORPH_KERNEL_SIZE
            self._kernel = np.ones((k, k), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_dirt(
        self,
        frame: np.ndarray,
        surface_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a binary uint8 mask (255 = dirty, 0 = clean).

        Algorithm
        ---------
        1. Convert BGR → LAB (or use channel deviation when cv2 is absent).
        2. Compute per-pixel deviation from a local mean.
        3. Threshold the magnitude of colour/brightness deviation.
        4. Morphological close + open to reduce noise.
        5. Apply *surface_mask* if provided.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image.")

        if self._has_cv2:
            raw_mask = self._detect_dirt_cv2(frame)
        else:
            raw_mask = self._detect_dirt_numpy(frame)

        if surface_mask is not None:
            smask = surface_mask.astype(np.uint8)
            if smask.shape != frame.shape[:2]:
                # Nearest-neighbour resize via numpy index tiling
                h, w = frame.shape[:2]
                sh, sw = smask.shape
                row_idx = (np.arange(h) * sh // h).astype(int)
                col_idx = (np.arange(w) * sw // w).astype(int)
                smask = smask[np.ix_(row_idx, col_idx)]
            raw_mask = (raw_mask.astype(bool) & smask.astype(bool)).astype(np.uint8) * 255

        return raw_mask

    def estimate_dirty_fraction(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> float:
        """Estimate the fraction ``[0.0, 1.0]`` of the surface that is dirty.

        Parameters
        ----------
        frame:
            Full HxWx3 BGR frame.
        roi:
            Optional ``(x1, y1, x2, y2)`` region of interest in pixel coords.
            When given, analysis is restricted to this rectangle.
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0
        else:
            crop = frame

        mask = self.detect_dirt(crop)
        total_pixels = mask.size
        if total_pixels == 0:
            return 0.0
        dirty_pixels = int(np.count_nonzero(mask))
        return float(dirty_pixels) / float(total_pixels)

    def detect_clutter_on_floor(
        self,
        depth: np.ndarray,
        floor_height: float = 0.0,
    ) -> np.ndarray:
        """Return a binary mask of floor-clutter regions from a depth image.

        Pixels where the measured depth is more than 5 cm shallower than
        *floor_height* are marked as cluttered (elevated objects on floor).

        Parameters
        ----------
        depth:
            HxW float32 depth in metres (camera frame, z = forward depth).
        floor_height:
            Expected floor depth (metres).  Anything shallower by >5 cm
            is considered floor clutter.
        """
        if depth.ndim != 2:
            raise ValueError("depth must be a 2-D HxW array.")

        clutter_threshold = floor_height - 0.05  # 5 cm above floor

        valid = depth > 0.0
        elevated = valid & (depth < clutter_threshold)
        mask = elevated.astype(np.uint8) * 255

        if self._has_cv2:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        else:
            mask = self._morph_open_numpy(mask)
            mask = self._morph_close_numpy(mask)

        return mask

    # ------------------------------------------------------------------
    # Private cv2 path
    # ------------------------------------------------------------------

    def _detect_dirt_cv2(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        blur = cv2.GaussianBlur(lab, (31, 31), sigmaX=0, sigmaY=0)
        delta_a = lab[:, :, 1] - blur[:, :, 1]
        delta_b = lab[:, :, 2] - blur[:, :, 2]
        colour_anomaly = np.sqrt(delta_a ** 2 + delta_b ** 2)
        delta_l = np.abs(lab[:, :, 0] - blur[:, :, 0])
        combined = np.maximum(colour_anomaly, delta_l * 0.7)
        raw_mask = (combined > self._LAB_ANOMALY_THRESH).astype(np.uint8) * 255
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self._kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self._kernel)
        return opened

    # ------------------------------------------------------------------
    # Private numpy-only fallback path
    # ------------------------------------------------------------------

    def _detect_dirt_numpy(self, frame: np.ndarray) -> np.ndarray:
        """Numpy-only dirt detection using channel deviation from local mean.

        Uses a 2-D prefix-sum box filter (kernel half-width = k, kernel size
        = 2k × 2k) to approximate a Gaussian blur without any external
        libraries.  Pixels whose colour deviates strongly from the local mean
        are marked as dirty.
        """
        img = frame.astype(np.float32)
        h, w = img.shape[:2]

        # Box filter via prefix sums with padding of k on each side.
        # Padded shape: (h + 2k, w + 2k, 3).  After cumsum, cs has the same
        # shape.  The box sum for an h×w region starting at origin is:
        #   cs[k:k+h, k:k+w] - cs[0:h, k:k+w] - cs[k:k+h, 0:w] + cs[0:h, 0:w]
        # which gives exactly h rows and w columns.
        k = 16  # half-width → kernel = 2k × 2k = 32 × 32
        padded = np.pad(img, ((k, k), (k, k), (0, 0)), mode='reflect')
        cs = np.cumsum(np.cumsum(padded.astype(np.float64), axis=0), axis=1)
        blur = (
            cs[k:k + h, k:k + w]
            - cs[0:h,   k:k + w]
            - cs[k:k + h, 0:w]
            + cs[0:h,     0:w]
        ).astype(np.float32) / (k * k)

        # Colour deviation across channels
        delta = np.abs(img - blur)
        colour_anomaly = np.sqrt(np.sum(delta ** 2, axis=2))

        raw_mask = (colour_anomaly > (self._LAB_ANOMALY_THRESH * 1.5)).astype(np.uint8) * 255
        raw_mask = self._morph_close_numpy(raw_mask)
        raw_mask = self._morph_open_numpy(raw_mask)
        return raw_mask

    @staticmethod
    def _morph_erode_numpy(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Binary erosion using scipy-free numpy sliding minimum."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant', constant_values=255)
        h, w = mask.shape
        out = np.full_like(mask, 255)
        for dr in range(kh):
            for dc in range(kw):
                if kernel[dr, dc]:
                    out = np.minimum(out, padded[dr:dr + h, dc:dc + w])
        return out

    @staticmethod
    def _morph_dilate_numpy(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Binary dilation using numpy sliding maximum."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
        h, w = mask.shape
        out = np.zeros_like(mask)
        for dr in range(kh):
            for dc in range(kw):
                if kernel[dr, dc]:
                    out = np.maximum(out, padded[dr:dr + h, dc:dc + w])
        return out

    def _morph_open_numpy(self, mask: np.ndarray) -> np.ndarray:
        """Morphological open (erode then dilate) without cv2."""
        eroded = self._morph_erode_numpy(mask, self._kernel)
        return self._morph_dilate_numpy(eroded, self._kernel)

    def _morph_close_numpy(self, mask: np.ndarray) -> np.ndarray:
        """Morphological close (dilate then erode) without cv2."""
        dilated = self._morph_dilate_numpy(mask, self._kernel)
        return self._morph_erode_numpy(dilated, self._kernel)


# ---------------------------------------------------------------------------
# Bed-making detector
# ---------------------------------------------------------------------------


class BedMakingDetector:
    """Specialised heuristic detector for assessing bed state.

    Uses depth variance (wrinkle detection) and colour segmentation
    (sheet coverage) to produce a structured assessment dict.
    """

    # Depth variance above this threshold indicates significant wrinkles (metres²)
    _WRINKLE_VAR_THRESHOLD: float = 0.0004
    # Minimum fraction of the bed bbox that must be a consistent colour to
    # count as "sheet coverage"
    _SHEET_COVERAGE_MIN: float = 0.55

    def assess_bed_state(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
    ) -> dict[str, Any]:
        """Return a structured assessment of bed state.

        Returns
        -------
        dict with keys:
            ``is_made`` (bool), ``wrinkle_severity`` (float 0–1),
            ``pillow_count`` (int), ``sheet_coverage`` (float 0–1).
        """
        h, w = frame.shape[:2]

        # --- wrinkle severity via depth variance ---
        valid_depth = depth[depth > 0.0]
        if valid_depth.size > 0:
            depth_var = float(np.var(valid_depth))
            # Normalise to [0, 1]: saturates at 3× the wrinkle threshold
            wrinkle_severity = float(
                np.clip(depth_var / (self._WRINKLE_VAR_THRESHOLD * 3.0), 0.0, 1.0)
            )
        else:
            wrinkle_severity = 0.5  # unknown — assume moderate

        # --- sheet coverage via colour segmentation ---
        sheet_coverage = self._estimate_sheet_coverage(frame)

        # --- pillow detection via blob analysis ---
        pillow_count = self._count_pillows(frame, depth)

        # Heuristic: bed is "made" if sheet coverage is high, wrinkles are
        # low, and at least one pillow is visible.
        is_made = (
            sheet_coverage >= self._SHEET_COVERAGE_MIN
            and wrinkle_severity < 0.4
            and pillow_count >= 1
        )

        return {
            "is_made": is_made,
            "wrinkle_severity": round(wrinkle_severity, 3),
            "pillow_count": pillow_count,
            "sheet_coverage": round(sheet_coverage, 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_sheet_coverage(self, frame: np.ndarray) -> float:
        """Fraction of the frame that appears to be a uniform sheet colour."""
        if _CV2_AVAILABLE:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            sheet_mask = (saturation < 60) & (value > 80)
        else:
            # Numpy approximation: sheets are bright (high mean) and
            # low-saturation (small channel range).
            img = frame.astype(np.float32)
            brightness = img.mean(axis=2)
            channel_range = img.max(axis=2) - img.min(axis=2)
            sheet_mask = (channel_range < 60) & (brightness > 80)

        coverage = float(np.count_nonzero(sheet_mask)) / float(sheet_mask.size)
        return float(np.clip(coverage, 0.0, 1.0))

    def _count_pillows(self, frame: np.ndarray, depth: np.ndarray) -> int:
        """Estimate pillow count by detecting elevated blobs near the bed head."""
        if depth.size == 0:
            return 0

        h, w = frame.shape[:2]
        top_region = depth[: h // 3, :]
        valid = top_region[top_region > 0.0]
        if valid.size == 0:
            return 0

        median_depth = float(np.median(valid))
        elevated = (top_region > 0.0) & (top_region < median_depth - 0.06)
        elevated_mask = elevated.astype(np.uint8) * 255

        min_pillow_area = int((h // 3 * w) * 0.02)

        if _CV2_AVAILABLE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(elevated_mask, cv2.MORPH_OPEN, kernel)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned)
            pillow_blobs = [
                i for i in range(1, num_labels)
                if stats[i, cv2.CC_STAT_AREA] >= min_pillow_area
            ]
        else:
            # Numpy fallback: count disconnected elevated regions via
            # a simple row-projection heuristic (avoids scipy dependency).
            col_presence = elevated_mask.any(axis=0)
            transitions = int(np.count_nonzero(np.diff(col_presence.astype(int)) > 0))
            pillow_blobs = list(range(transitions))

        return min(len(pillow_blobs), 4)
