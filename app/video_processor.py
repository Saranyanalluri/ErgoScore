"""
Person 1 - Video processor module.

Orchestrates: VIDEO -> POSE DETECTION -> LANDMARKS -> ANNOTATED VIDEO +
KEYPOINT JSON + METADATA.

This is the Person 1 public API. Other team members (Person 2, Person 4)
should only need to call `process_video()`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from app.config import (
    DEFAULT_INPUT_VIDEO,
    DEFAULT_MODEL_PATH,
    KEYPOINTS_DIR,
    REQUIRED_LANDMARK_NAMES,
    SCHEMA_VERSION,
    SKELETON_CONNECTIONS,
    VIDEOS_PROCESSED_DIR,
    PoseConfig,
    ensure_output_dirs,
)
from app.pose_detector import ModelNotFoundError, PoseDetector, PoseDetectorInitError
from app.schemas import FrameResult, ProcessingMetadata, ProcessingResult, VideoMetadata

logger = logging.getLogger("ergoscore.person1.video_processor")

# Codec fallback chain: (fourcc, container_extension)
_CODEC_FALLBACKS = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]


class VideoProcessingError(RuntimeError):
    """Raised for any unrecoverable error while processing the input video."""


def _validate_input_video(input_path: Path) -> None:
    if not input_path.exists():
        raise VideoProcessingError(
            f"Input video not found: {input_path}\n"
            "Place the video at that path, or pass --input <path> to run_pose.py."
        )
    if not input_path.is_file():
        raise VideoProcessingError(f"Input path is not a file: {input_path}")


def _open_capture(input_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(
            f"OpenCV could not open video: {input_path}\n"
            "The file may be corrupted, use an unsupported codec, or be a non-video file. "
            "Try re-encoding it to H.264 MP4."
        )
    return cap


def _safe_fps(raw_fps: float, warnings: List[str]) -> float:
    if raw_fps is None or raw_fps <= 0 or np.isnan(raw_fps):
        warnings.append(f"Invalid FPS reported by OpenCV ({raw_fps!r}); defaulting to 30.0 FPS.")
        return 30.0
    return float(raw_fps)


def _create_writer(output_path: Path, fps: float, width: int, height: int, warnings: List[str]) -> tuple:
    """Try codecs in order until one opens successfully.

    Returns (VideoWriter, actual_output_path) with actual_output_path reflecting
    any extension change required by the fallback codec that succeeded.
    """
    last_error = None
    for fourcc_str, ext in _CODEC_FALLBACKS:
        candidate_path = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))
        if writer.isOpened():
            if ext != output_path.suffix:
                warnings.append(
                    f"Preferred codec unavailable; fell back to '{fourcc_str}' "
                    f"writing to {candidate_path.name}."
                )
            return writer, candidate_path
        writer.release()
        last_error = fourcc_str

    raise VideoProcessingError(
        f"Could not open a VideoWriter with any known codec (tried: "
        f"{[c[0] for c in _CODEC_FALLBACKS]}, last attempted: {last_error}). "
        "Your OpenCV build may be missing video codec support."
    )


def _draw_skeleton(frame_bgr: np.ndarray, landmarks: dict, width: int, height: int, person_detected: bool) -> None:
    """Draw joints + bone connections for the required landmarks in-place."""
    status_text = "POSE DETECTED" if person_detected else "NO POSE"
    status_color = (0, 200, 0) if person_detected else (0, 0, 220)
    cv2.putText(
        frame_bgr, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA
    )

    if not person_detected:
        return

    points_px = {}
    for name in REQUIRED_LANDMARK_NAMES:
        lm = landmarks.get(name)
        if lm is None:
            continue
        points_px[name] = (int(lm.x * width), int(lm.y * height))

    for a, b in SKELETON_CONNECTIONS:
        if a in points_px and b in points_px:
            cv2.line(frame_bgr, points_px[a], points_px[b], (255, 180, 0), 2, cv2.LINE_AA)

    for name, (px, py) in points_px.items():
        cv2.circle(frame_bgr, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)


def process_video(
    input_path: Union[str, Path] = DEFAULT_INPUT_VIDEO,
    output_video_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    config: Optional[PoseConfig] = None,
) -> ProcessingResult:
    """Process a video end-to-end: pose detection -> landmarks -> outputs.

    Args:
        input_path: path to the input video file.
        output_video_path: where to write the annotated skeleton video.
            Defaults to videos/processed/<input_stem>_processed.mp4
        output_json_path: where to write the keypoint JSON.
            Defaults to data/keypoints/<input_stem>_keypoints.json
        model_path: path to the MediaPipe pose_landmarker .task model file.
        config: PoseConfig instance; defaults to PoseConfig().

    Returns:
        ProcessingResult with paths to every output file and summary stats.

    Raises:
        VideoProcessingError: for any unrecoverable input/output/video issue.
        ModelNotFoundError: if the .task model file is missing.
        PoseDetectorInitError: if MediaPipe fails to initialize.
    """
    ensure_output_dirs()
    config = config or PoseConfig()
    warnings: List[str] = []

    input_path = Path(input_path)
    _validate_input_video(input_path)

    if output_video_path is None:
        output_video_path = VIDEOS_PROCESSED_DIR / f"{input_path.stem}_processed.mp4"
    else:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if output_json_path is None:
        output_json_path = KEYPOINTS_DIR / f"{input_path.stem}_keypoints.json"
    else:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cap = _open_capture(input_path)
    writer = None
    detector: Optional[PoseDetector] = None
    start_time = time.time()

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = _safe_fps(raw_fps, warnings)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise VideoProcessingError(
                f"Invalid video resolution reported ({width}x{height}) for {input_path}."
            )
        if frame_count_hint <= 0:
            warnings.append(
                "OpenCV reported zero/unknown frame count; will process until read fails."
            )

        writer, output_video_path = _create_writer(output_video_path, fps, width, height, warnings)

        try:
            detector = PoseDetector(Path(model_path), config)
        except (ModelNotFoundError, PoseDetectorInitError):
            raise

        frames: List[FrameResult] = []
        frame_index = 0
        frames_with_pose = 0
        quality_sum = 0.0
        last_timestamp_ms = -1
        ms_per_frame = 1000.0 / fps

        logger.info("=" * 50)
        logger.info("ErgoScore - Person 1 Pose Processing")
        logger.info("=" * 50)
        logger.info("Input: %s", input_path)
        logger.info("Resolution: %dx%d", width, height)
        logger.info("FPS: %.2f", fps)
        logger.info("Frames (hint): %s", frame_count_hint if frame_count_hint > 0 else "unknown")
        logger.info("Processing...")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_bgr is None or frame_bgr.size == 0:
                warnings.append(f"Frame {frame_index} failed to read correctly and was skipped.")
                frame_index += 1
                continue

            timestamp_ms = int(round(frame_index * ms_per_frame))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            try:
                result = detector.detect_for_frame(rgb_frame, timestamp_ms)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Detection failed on frame {frame_index}: {exc}")
                person_detected, landmarks, world_landmarks, pose_quality = (
                    False,
                    {name: None for name in REQUIRED_LANDMARK_NAMES},
                    None,
                    0.0,
                )
            else:
                person_detected, landmarks, world_landmarks, pose_quality = (
                    detector.extract_required_landmarks(result)
                )

            if person_detected:
                frames_with_pose += 1
                quality_sum += pose_quality

            _draw_skeleton(frame_bgr, landmarks, width, height, person_detected)
            writer.write(frame_bgr)

            frames.append(
                FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_ms / 1000.0,
                    person_detected=person_detected,
                    pose_quality=pose_quality,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                )
            )

            frame_index += 1

        frames_processed = frame_index
        if frames_processed == 0:
            raise VideoProcessingError(
                f"Zero frames could be read from {input_path}. The file may be empty or corrupted."
            )

        duration_seconds = frames_processed / fps
        detection_rate = frames_with_pose / frames_processed
        average_quality = (quality_sum / frames_with_pose) if frames_with_pose > 0 else 0.0
        processing_time = time.time() - start_time
        processing_fps = frames_processed / processing_time if processing_time > 0 else 0.0

        video_metadata = VideoMetadata(
            filename=input_path.name,
            fps=fps,
            width=width,
            height=height,
            frame_count=frames_processed,
            duration_seconds=duration_seconds,
        )

        keypoints_payload = {
            "schema_version": SCHEMA_VERSION,
            "video_metadata": video_metadata.to_dict(),
            "pose_config": config.to_dict(),
            "frames": [f.to_dict() for f in frames],
        }
        output_json_path.write_text(json.dumps(keypoints_payload, indent=2), encoding="utf-8")

        metadata = ProcessingMetadata(
            input_path=str(input_path),
            output_video_path=str(output_video_path),
            output_json_path=str(output_json_path),
            video_metadata=video_metadata,
            frames_processed=frames_processed,
            frames_with_pose=frames_with_pose,
            pose_detection_rate=detection_rate,
            average_pose_quality=average_quality,
            processing_time_seconds=processing_time,
            processing_fps=processing_fps,
            camera_view=config.camera_view,
            warnings=warnings,
            pose_config=config.to_dict(),
        )
        metadata_json_path = output_json_path.with_name(f"{input_path.stem}_metadata.json")
        metadata_json_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        logger.info("Pose detection rate: %.1f%%", detection_rate * 100)
        logger.info("Average pose quality: %.2f", average_quality)
        logger.info("Processing FPS: %.1f", processing_fps)
        logger.info("Completed successfully.")
        logger.info("Keypoints: %s", output_json_path)
        logger.info("Processed video: %s", output_video_path)
        logger.info("Metadata: %s", metadata_json_path)
        logger.info("=" * 50)

        return ProcessingResult(
            keypoints_json_path=str(output_json_path),
            processed_video_path=str(output_video_path),
            metadata_json_path=str(metadata_json_path),
            frames_processed=frames_processed,
            frames_with_pose=frames_with_pose,
            pose_detection_rate=detection_rate,
            average_pose_quality=average_quality,
            warnings=warnings,
        )

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if detector is not None:
            detector.close()
