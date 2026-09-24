"""
Person 1 - Minimal CLI entry point.

Usage:
    python run_pose.py
    python run_pose.py --input videos/uploads/my_video.mp4
    python run_pose.py --input videos/uploads/my_video.mp4 --disable-smoothing
    python run_pose.py --input videos/uploads/my_video.mp4 --camera-view side
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import DEFAULT_INPUT_VIDEO, DEFAULT_MODEL_PATH, PoseConfig
from app.pose_detector import ModelNotFoundError, PoseDetectorInitError
from app.video_processor import VideoProcessingError, process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ErgoScore Person 1 - Video pose detection and landmark extraction."
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT_VIDEO),
        help=f"Path to input video (default: {DEFAULT_INPUT_VIDEO})",
    )
    parser.add_argument(
        "--output-video", type=str, default=None,
        help="Path to write the annotated skeleton video (default: videos/processed/<name>_processed.mp4)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write the keypoint JSON (default: data/keypoints/<name>_keypoints.json)",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the MediaPipe pose_landmarker .task model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--disable-smoothing", action="store_true",
        help="Disable exponential-moving-average landmark smoothing.",
    )
    parser.add_argument(
        "--camera-view", type=str, default="unknown", choices=["side", "front", "unknown"],
        help="Informational metadata field only; not auto-detected.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    config = PoseConfig(
        smoothing_enabled=not args.disable_smoothing,
        camera_view=args.camera_view,
    )

    try:
        process_video(
            input_path=Path(args.input),
            output_video_path=Path(args.output_video) if args.output_video else None,
            output_json_path=Path(args.output_json) if args.output_json else None,
            model_path=Path(args.model),
            config=config,
        )
        return 0
    except (VideoProcessingError, ModelNotFoundError, PoseDetectorInitError) as exc:
        logging.error("ErgoScore Person 1 failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
