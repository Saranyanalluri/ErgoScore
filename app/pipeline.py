"""
ErgoScore analysis pipeline — INTEGRATION POINT.

analyze_video() currently returns structured DUMMY data so the Flask app
and dashboard can be built and tested end-to-end before the real analysis
modules exist. Nothing outside this file should know or care that the data
is fake — routes.py and the templates consume the returned dict exactly as
they will once this function calls into real analysis code.

Final pipeline (to be assembled here once the other modules are ready):

    video_path
      -> Person 1: MediaPipe pose extraction (keypoints per frame)
      -> Person 2: joint angles + REBA/RULA scoring (per-frame score)
      -> Person 3: temporal risk analysis + report generation
      -> returns a dict with the exact same shape analyze_video() returns now

When that integration happens, only the body of analyze_video() (and its
private helpers below) should need to change — the return dict's keys and
types are the contract the rest of the app is built against. Keep that
contract stable.
"""

import os
from datetime import datetime, timezone

# (low, high, label) inclusive score bands used to classify overall risk.
RISK_BANDS = [
    (0, 3, "Low"),
    (4, 7, "Medium"),
    (8, 15, "High"),
]

LOAD_BONUS = {"none": 0, "light": 1, "medium": 2, "heavy": 3}
COUPLING_PENALTY = {"good": 0, "fair": 1, "poor": 2}


def _risk_level_for_score(score):
    for low, high, label in RISK_BANDS:
        if low <= score <= high:
            return label
    return "High"


def _dummy_timeline(load, coupling):
    """
    Builds a placeholder per-second score timeline. The shape (list of
    {"time": int, "score": int}) is what Person 3's real temporal analysis
    should also return, so the dashboard and charts need no changes later.

    Load/coupling nudge the dummy scores so the dashboard visibly reacts to
    user input even though the underlying numbers are fake.
    """
    base_scores = [3, 4, 6, 8, 9, 7, 5, 4, 6, 8]
    load_bonus = LOAD_BONUS.get(load, 0)
    coupling_penalty = COUPLING_PENALTY.get(coupling, 0)

    timeline = []
    for t, raw_score in enumerate(base_scores):
        score = raw_score + load_bonus + coupling_penalty
        score = max(1, min(score, 15))
        timeline.append({"time": t, "score": score})
    return timeline


def _dummy_worst_frames(timeline):
    """
    Picks the highest-scoring points in the timeline and fabricates frame
    metadata around them. image_url is left as None — placeholders are
    rendered in the template rather than inventing fake image data here.
    """
    worst_points = sorted(timeline, key=lambda p: p["score"], reverse=True)[:5]
    frames = []
    for point in worst_points:
        frames.append({
            "frame_number": point["time"] * 30,
            "timestamp": f"00:{point['time']:02d}",
            "score": point["score"],
            "trunk_angle": 20 + point["score"] * 3,
            "neck_angle": 10 + point["score"] * 2,
            "knee_angle": 5 + point["score"],
            "image_url": None,
        })
    return frames


def _risk_distribution(timeline):
    total = len(timeline)
    low = sum(1 for p in timeline if p["score"] <= 3)
    medium = sum(1 for p in timeline if 4 <= p["score"] <= 7)
    high = sum(1 for p in timeline if p["score"] >= 8)
    return {
        "low": round(low / total * 100, 1),
        "medium": round(medium / total * 100, 1),
        "high": round(high / total * 100, 1),
    }


def _summary(overall_score, risk_level, distribution):
    return (
        f"This is placeholder analysis. Based on dummy data, the assessed task "
        f"has an overall REBA score of {overall_score} ({risk_level} risk). "
        f"The task spent {distribution['high']}% of the time in high-risk "
        f"postures, {distribution['medium']}% in medium-risk postures, and "
        f"{distribution['low']}% in low-risk postures. Once the pose "
        f"estimation and scoring modules are integrated, this summary will "
        f"reflect real posture analysis instead of placeholder values."
    )


def analyze_video(video_path, load="medium", coupling="fair"):
    """
    Analyze an uploaded video and return a structured result dict.

    Currently returns DUMMY data — see the module docstring for the
    integration plan. Do not add real video analysis here yet; the other
    team members' modules will be wired in through this function later.

    Args:
        video_path: path to the uploaded video file on disk.
        load: one of "none", "light", "medium", "heavy".
        coupling: one of "good", "fair", "poor".

    Returns:
        dict with keys: overall_score, risk_level, average_score,
        maximum_score, duration, risk_distribution, timeline, worst_frames,
        summary, video_filename, generated_at, is_dummy_data.
    """
    timeline = _dummy_timeline(load, coupling)
    scores = [point["score"] for point in timeline]
    average_score = round(sum(scores) / len(scores), 1)
    maximum_score = max(scores)
    overall_score = maximum_score
    risk_level = _risk_level_for_score(overall_score)
    distribution = _risk_distribution(timeline)
    worst_frames = _dummy_worst_frames(timeline)

    return {
        "overall_score": overall_score,
        "risk_level": risk_level,
        "average_score": average_score,
        "maximum_score": maximum_score,
        "duration": f"{len(timeline)} sec (placeholder)",
        "risk_distribution": distribution,
        "timeline": timeline,
        "worst_frames": worst_frames,
        "summary": _summary(overall_score, risk_level, distribution),
        "video_filename": os.path.basename(video_path) if video_path else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_dummy_data": True,
    }
