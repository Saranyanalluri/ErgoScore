"""
Person 1 - Configuration module.

Centralizes every tunable value used by the pose-detection subsystem so that
no magic numbers are scattered across pose_detector.py / video_processor.py.

These thresholds are ENGINEERING defaults chosen for reasonable behaviour on
a hackathon laptop. They are not scientifically validated optimal values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------
# Project paths (all relative to the project root, Windows-safe via pathlib)
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"

VIDEOS_UPLOAD_DIR = PROJECT_ROOT / "videos" / "uploads"
VIDEOS_PROCESSED_DIR = PROJECT_ROOT / "videos" / "processed"
KEYPOINTS_DIR = PROJECT_ROOT / "data" / "keypoints"

DEFAULT_INPUT_VIDEO = VIDEOS_UPLOAD_DIR / "input.mp4"


def ensure_output_dirs() -> None:
    """Create every output directory Person 1 writes to, if missing."""
    for directory in (VIDEOS_PROCESSED_DIR, KEYPOINTS_DIR, MODEL_DIR, VIDEOS_UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Required landmark names -> MediaPipe Pose landmark index
# --------------------------------------------------------------------------
# These indices come from the standard 33-point MediaPipe Pose topology.
# NAMES MUST STAY STABLE: Person 2 (angle calculator) consumes them directly.

REQUIRED_LANDMARKS: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

REQUIRED_LANDMARK_NAMES: List[str] = list(REQUIRED_LANDMARKS.keys())

# Bone connections between required landmarks only, used for the skeleton
# overlay drawn on the annotated demo video.
SKELETON_CONNECTIONS: List[tuple] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


@dataclass
class PoseConfig:
    """All configurable thresholds for pose detection / quality / smoothing.

    These are engineering thresholds, not scientifically optimal constants.
    """

    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # A landmark below this visibility is treated as unreliable and reported
    # as null in the JSON output, even if MediaPipe returned coordinates.
    min_landmark_visibility: float = 0.5

    # Smoothing (exponential moving average) applied to image-normalized
    # x/y/z of each *visible* required landmark independently across frames.
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.4  # weight given to the new (current) frame

    # Camera view metadata is informational only; never auto-inferred.
    camera_view: str = "unknown"  # one of: "side", "front", "unknown"

    def to_dict(self) -> dict:
        return {
            "num_poses": self.num_poses,
            "min_pose_detection_confidence": self.min_pose_detection_confidence,
            "min_pose_presence_confidence": self.min_pose_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "min_landmark_visibility": self.min_landmark_visibility,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_alpha": self.smoothing_alpha,
            "camera_view": self.camera_view,
        }


SCHEMA_VERSION = "1.0"
