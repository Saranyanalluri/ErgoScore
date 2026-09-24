"""
Person 1 - Pose detector module.

Thin, testable wrapper around the MediaPipe Pose Landmarker TASK API
(current, supported API — NOT the deprecated `mp.solutions.pose` API).

Responsibilities:
- load/validate the .task model file
- run pose detection in VIDEO mode with monotonically increasing timestamps
- select a single deterministic "primary" pose if MediaPipe unexpectedly
  returns more than one
- extract ONLY the required landmarks (config.REQUIRED_LANDMARKS), applying
  the visibility threshold, and return them as plain-Python LandmarkPoint
  objects (never MediaPipe/NumPy objects)
- compute a documented, non-scientific pose_quality score
- apply optional per-landmark exponential-moving-average smoothing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import PoseConfig, REQUIRED_LANDMARKS, REQUIRED_LANDMARK_NAMES
from app.schemas import LandmarkPoint

logger = logging.getLogger("ergoscore.person1.pose_detector")


class ModelNotFoundError(FileNotFoundError):
    """Raised when the MediaPipe .task model file is missing."""


class PoseDetectorInitError(RuntimeError):
    """Raised when the MediaPipe PoseLandmarker fails to initialize."""


def _select_primary_pose_index(pose_landmarks_list, pose_world_landmarks_list) -> int:
    """Deterministic policy for choosing which detected pose to keep.

    ErgoScore MVP assumes a single worker in frame. If MediaPipe returns
    more than one pose in a frame (num_poses was configured >1, or a
    stray detection slipped through), we always pick the pose whose
    normalized-coordinate bounding box has the largest area — i.e. the
    most prominent / closest-to-camera person. This is deterministic
    given the same input and avoids randomly switching identity between
    frames based on list order.
    """
    if len(pose_landmarks_list) <= 1:
        return 0

    best_index = 0
    best_area = -1.0
    for i, landmarks in enumerate(pose_landmarks_list):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_index = i
    return best_index


class LandmarkSmoother:
    """Per-landmark exponential moving average (EMA) smoother.

    Smooths x, y, z of each *required* landmark independently across
    consecutive frames where that landmark was visible.

    Behaviour (documented per project requirement):
    - Only x, y, z are smoothed. visibility/presence are passed through
      un-smoothed (they describe detection confidence, not position).
    - If a landmark is missing/unreliable on a frame, its EMA state is
      reset (not updated, not interpolated). The next time it becomes
      visible again, smoothing restarts fresh from that raw value rather
      than jumping from a stale old average — this avoids large false
      "teleport" smoothing artifacts after an occlusion gap.
    - alpha (config.smoothing_alpha) is the weight given to the new frame:
      ema = alpha * new + (1 - alpha) * previous_ema
      Higher alpha = less smoothing / more responsive to fast movement.
    """

    def __init__(self, alpha: float, landmark_names: List[str]):
        self.alpha = alpha
        self._state: Dict[str, Optional[np.ndarray]] = {name: None for name in landmark_names}

    def smooth(self, name: str, x: float, y: float, z: float) -> tuple:
        raw = np.array([x, y, z], dtype=np.float64)
        prev = self._state.get(name)
        if prev is None:
            smoothed = raw
        else:
            smoothed = self.alpha * raw + (1.0 - self.alpha) * prev
        self._state[name] = smoothed
        return float(smoothed[0]), float(smoothed[1]), float(smoothed[2])

    def reset_landmark(self, name: str) -> None:
        """Called when a landmark is missing/unreliable on the current frame."""
        self._state[name] = None


class PoseDetector:
    """Wraps a single MediaPipe PoseLandmarker instance (VIDEO mode).

    One instance is created per `process_video()` call and reused for every
    frame of that video (per PERFORMANCE requirement — no repeated init).
    """

    def __init__(self, model_path: Path, config: PoseConfig):
        self.config = config
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"MediaPipe Pose Landmarker model not found at: {self.model_path}\n"
                "Download a pose_landmarker .task model (e.g. 'pose_landmarker_lite.task') "
                "from the MediaPipe model zoo and place it at that exact path. "
                "See README_PERSON1.md 'Model Setup' for the download link and instructions."
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - environment issue
            raise PoseDetectorInitError(
                "Failed to import mediapipe. Ensure it is installed via "
                "`pip install -r requirements.txt` in the active virtual environment."
            ) from exc

        self._mp = mp
        self._mp_vision = mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=config.num_poses,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - surface a clear actionable error
            raise PoseDetectorInitError(
                f"MediaPipe PoseLandmarker failed to initialize with model "
                f"'{self.model_path}'. The model file may be corrupted, incompatible, "
                f"or not a Pose Landmarker task file. Original error: {exc}"
            ) from exc

        self._smoother = (
            LandmarkSmoother(config.smoothing_alpha, REQUIRED_LANDMARK_NAMES)
            if config.smoothing_enabled
            else None
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def detect_for_frame(self, rgb_frame: np.ndarray, timestamp_ms: int):
        """Run detection on one RGB frame. Timestamps must be monotonically
        increasing across calls for the same instance (VIDEO mode requirement).
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def extract_required_landmarks(
        self, result
    ) -> tuple:
        """Extract the required landmarks from a PoseLandmarker result.

        Returns (person_detected, landmarks_dict, world_landmarks_dict, pose_quality).
        landmarks_dict maps name -> LandmarkPoint | None.
        """
        min_vis = self.config.min_landmark_visibility
        empty = {name: None for name in REQUIRED_LANDMARK_NAMES}

        if not result.pose_landmarks:
            if self._smoother is not None:
                for name in REQUIRED_LANDMARK_NAMES:
                    self._smoother.reset_landmark(name)
            return False, empty, None, 0.0

        primary_idx = _select_primary_pose_index(
            result.pose_landmarks,
            result.pose_world_landmarks if result.pose_world_landmarks else None,
        )
        image_landmarks = result.pose_landmarks[primary_idx]
        world_landmarks = (
            result.pose_world_landmarks[primary_idx]
            if result.pose_world_landmarks
            else None
        )

        landmarks: Dict[str, Optional[LandmarkPoint]] = {}
        world_out: Dict[str, Optional[LandmarkPoint]] = {}
        available_count = 0
        visibility_sum = 0.0

        for name, idx in REQUIRED_LANDMARKS.items():
            lm = image_landmarks[idx]
            visibility = getattr(lm, "visibility", None)
            presence = getattr(lm, "presence", None)

            is_reliable = (visibility is None) or (visibility >= min_vis)

            if not is_reliable:
                landmarks[name] = None
                world_out[name] = None
                if self._smoother is not None:
                    self._smoother.reset_landmark(name)
                continue

            x, y, z = float(lm.x), float(lm.y), float(lm.z)
            if self._smoother is not None:
                x, y, z = self._smoother.smooth(name, x, y, z)

            landmarks[name] = LandmarkPoint(
                x=x,
                y=y,
                z=z,
                visibility=float(visibility) if visibility is not None else None,
                presence=float(presence) if presence is not None else None,
            )
            available_count += 1
            visibility_sum += float(visibility) if visibility is not None else 1.0

            if world_landmarks is not None:
                wlm = world_landmarks[idx]
                world_out[name] = LandmarkPoint(
                    x=float(wlm.x),
                    y=float(wlm.y),
                    z=float(wlm.z),
                    visibility=float(visibility) if visibility is not None else None,
                    presence=float(presence) if presence is not None else None,
                )
            else:
                world_out[name] = None

        pose_quality = _compute_pose_quality(available_count, visibility_sum, len(REQUIRED_LANDMARKS))
        return True, landmarks, (world_out if world_landmarks is not None else None), pose_quality


def _compute_pose_quality(available_count: int, visibility_sum: float, total_required: int) -> float:
    """Documented pose-quality metric (NOT an accuracy metric).

    pose_quality = 0.5 * (available_required_landmarks / total_required)
                 + 0.5 * (average_visibility_of_available_landmarks)

    - The first term rewards having more of the required landmarks
      available at all (above the visibility threshold).
    - The second term rewards those available landmarks being detected
      with high confidence.
    - Range: [0.0, 1.0]. This is an engineering heuristic describing how
      trustworthy a frame's landmark set looks, not a ground-truth
      accuracy measurement.
    """
    if total_required == 0:
        return 0.0
    completeness = available_count / total_required
    avg_visibility = (visibility_sum / available_count) if available_count > 0 else 0.0
    quality = 0.5 * completeness + 0.5 * avg_visibility
    return max(0.0, min(1.0, quality))
