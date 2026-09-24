"""
Person 1 - Data schema module.

Defines the stable, JSON-serializable data model produced by the pose
subsystem. Person 2 consumes the output of `to_dict()` / the JSON files
written by video_processor.py.

Rules enforced here:
- no NumPy scalar types ever reach the output (always cast to float/int)
- no MediaPipe objects are stored
- missing/unreliable landmarks are represented as None, never fabricated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _safe_float(value) -> float:
    """Cast any numeric (including numpy scalars) to a plain Python float."""
    return float(value)


@dataclass
class LandmarkPoint:
    """A single landmark's coordinates and confidence signals."""

    x: float
    y: float
    z: Optional[float] = None
    visibility: Optional[float] = None
    presence: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "x": _safe_float(self.x),
            "y": _safe_float(self.y),
            "z": _safe_float(self.z) if self.z is not None else None,
            "visibility": _safe_float(self.visibility) if self.visibility is not None else None,
            "presence": _safe_float(self.presence) if self.presence is not None else None,
        }


@dataclass
class FrameResult:
    """Per-frame structured result: one entry per processed video frame."""

    frame_index: int
    timestamp_seconds: float
    person_detected: bool
    pose_quality: float
    landmarks: Dict[str, Optional[LandmarkPoint]]
    world_landmarks: Optional[Dict[str, Optional[LandmarkPoint]]] = None

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_seconds": round(_safe_float(self.timestamp_seconds), 4),
            "person_detected": bool(self.person_detected),
            "pose_quality": round(_safe_float(self.pose_quality), 4),
            "landmarks": {
                name: (lm.to_dict() if lm is not None else None)
                for name, lm in self.landmarks.items()
            },
            "world_landmarks": (
                {
                    name: (lm.to_dict() if lm is not None else None)
                    for name, lm in self.world_landmarks.items()
                }
                if self.world_landmarks is not None
                else None
            ),
        }


@dataclass
class VideoMetadata:
    filename: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "fps": round(_safe_float(self.fps), 4),
            "width": int(self.width),
            "height": int(self.height),
            "frame_count": int(self.frame_count),
            "duration_seconds": round(_safe_float(self.duration_seconds), 4),
        }


@dataclass
class ProcessingMetadata:
    """Run-level metadata written to <input_name>_metadata.json."""

    input_path: str
    output_video_path: str
    output_json_path: str
    video_metadata: VideoMetadata
    frames_processed: int
    frames_with_pose: int
    pose_detection_rate: float
    average_pose_quality: float
    processing_time_seconds: float
    processing_fps: float
    camera_view: str
    warnings: List[str] = field(default_factory=list)
    pose_config: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_video_path": self.output_video_path,
            "output_json_path": self.output_json_path,
            "video_metadata": self.video_metadata.to_dict(),
            "frames_processed": int(self.frames_processed),
            "frames_with_pose": int(self.frames_with_pose),
            "pose_detection_rate": round(_safe_float(self.pose_detection_rate), 4),
            "average_pose_quality": round(_safe_float(self.average_pose_quality), 4),
            "processing_time_seconds": round(_safe_float(self.processing_time_seconds), 3),
            "processing_fps": round(_safe_float(self.processing_fps), 3),
            "camera_view": self.camera_view,
            "warnings": list(self.warnings),
            "pose_config": self.pose_config,
        }


@dataclass
class ProcessingResult:
    """Return value of `process_video()` — the Person-1 public API."""

    keypoints_json_path: str
    processed_video_path: str
    metadata_json_path: str
    frames_processed: int
    frames_with_pose: int
    pose_detection_rate: float
    average_pose_quality: float
    warnings: List[str] = field(default_factory=list)
