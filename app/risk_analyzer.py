"""
risk_analyzer.py
=================

Person 3 — Temporal Risk Analysis module for ErgoScore.

This module consumes frame-by-frame REBA scores produced by Person 2's
ergonomic scoring pipeline and turns them into:

    * a smoothed (temporally de-noised) score sequence
    * summary statistics (average / min / max / duration)
    * a percentage breakdown of time spent in each REBA risk category
    * the worst (highest-risk) frames in the clip

This module does **not**:
    * run MediaPipe / any computer vision (that is Person 1)
    * compute joint angles or REBA scores (that is Person 2)
    * build the Flask app / UI (that is Person 4)

It only works with the already-computed REBA scores handed to it.

Public interface
-----------------
The single function Person 4 (or anyone else) needs is:

    analyze_scores(input_path, output_directory) -> dict

See the docstring on that function, and README.md, for full details.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# CONFIGURATION — REBA risk-category thresholds
# ---------------------------------------------------------------------------
# REBA (Rapid Entire Body Assessment) defines five standard "action levels"
# based on the final REBA score (Hignett & McAtamney, 2000):
#
#   Score  | Risk Level    | Action Level
#   -------|---------------|-------------------------------
#   1      | Negligible    | 0 - None necessary
#   2-3    | Low           | 1 - May be necessary
#   4-7    | Medium        | 2 - Necessary
#   8-10   | High          | 3 - Necessary soon
#   11-15  | Very High     | 4 - Necessary now
#
# ErgoScore reports a simplified THREE-bucket risk distribution
# (low / medium / high) for readability on the dashboard and report.
# The buckets below collapse the five standard REBA action levels into
# three groups as follows:
#
#   ErgoScore bucket | REBA score range | Combines action levels
#   ------------------|------------------|------------------------
#   low                | 1 - 3            | Negligible + Low
#   medium              | 4 - 7            | Medium
#   high                | 8 - 15           | High + Very High
#
# If the team decides a different grouping is preferred (e.g. keeping
# "negligible" separate, or a 5-bucket breakdown), only this dictionary
# needs to change — every function below reads from it.
REBA_RISK_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "low": {"min": 1, "max": 3},
    "medium": {"min": 4, "max": 7},
    "high": {"min": 8, "max": 15},
}

# Valid REBA score range (used for input validation).
REBA_MIN_SCORE = 1
REBA_MAX_SCORE = 15

# Default configuration for the pipeline. Can be overridden by callers of
# analyze_scores() / create_analysis().
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_WORST_FRAME_COUNT = 5


# ---------------------------------------------------------------------------
# A. LOAD + VALIDATE INPUT
# ---------------------------------------------------------------------------
def load_scores(input_path: PathLike) -> List[Dict[str, Any]]:
    """Load the raw frame-score JSON produced by Person 2.

    Args:
        input_path: Path to a JSON file containing a list of objects with
            "frame", "timestamp", and "reba_score" keys.

    Returns:
        The parsed JSON content (a list of dicts), unvalidated.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is not valid JSON, or its top-level
            structure is not a list.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input score file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read input file '{path}': {exc}") from exc

    if not raw_text.strip():
        raise ValueError(f"Input file '{path}' is empty.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file '{path}' is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            "Expected the input JSON to be a list of frame-score objects, "
            f"got {type(data).__name__} instead."
        )

    return data


def validate_scores(raw_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize a list of raw frame-score entries.

    Each entry must contain:
        * "frame": an integer (or int-like) frame index
        * "timestamp": a numeric timestamp in seconds
        * "reba_score": a numeric REBA score within
          [REBA_MIN_SCORE, REBA_MAX_SCORE]

    Entries are returned sorted by timestamp (then frame) ascending.

    Args:
        raw_scores: The raw, unvalidated list loaded via load_scores().

    Returns:
        A cleaned list of dicts, each with keys "frame" (int),
        "timestamp" (float), and "reba_score" (float).

    Raises:
        ValueError: If the list is empty, or any entry is missing a
            required field, has a non-numeric value, or has a REBA score
            outside the valid range. The error message identifies the
            offending entry so the problem can be fixed quickly.
    """
    if raw_scores is None or len(raw_scores) == 0:
        raise ValueError(
            "No frame data found in input. The score list is empty."
        )

    required_fields = ("frame", "timestamp", "reba_score")
    cleaned: List[Dict[str, Any]] = []

    for index, entry in enumerate(raw_scores):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Entry at position {index} is not a JSON object "
                f"(got {type(entry).__name__})."
            )

        for field in required_fields:
            if field not in entry:
                raise ValueError(
                    f"Entry at position {index} (frame={entry.get('frame', '?')}) "
                    f"is missing required field '{field}'."
                )

        frame_val = entry["frame"]
        timestamp_val = entry["timestamp"]
        score_val = entry["reba_score"]

        if isinstance(frame_val, bool) or not isinstance(frame_val, (int, float)):
            raise ValueError(
                f"Entry at position {index} has a non-numeric 'frame' value: "
                f"{frame_val!r}"
            )
        if isinstance(timestamp_val, bool) or not isinstance(timestamp_val, (int, float)):
            raise ValueError(
                f"Entry at position {index} (frame={frame_val}) has a "
                f"non-numeric 'timestamp' value: {timestamp_val!r}"
            )
        if isinstance(score_val, bool) or not isinstance(score_val, (int, float)):
            raise ValueError(
                f"Entry at position {index} (frame={frame_val}) has a "
                f"non-numeric 'reba_score' value: {score_val!r}"
            )
        if not (REBA_MIN_SCORE <= score_val <= REBA_MAX_SCORE):
            raise ValueError(
                f"Entry at position {index} (frame={frame_val}) has an "
                f"out-of-range 'reba_score' value: {score_val}. "
                f"Valid REBA scores are {REBA_MIN_SCORE}-{REBA_MAX_SCORE}."
            )

        cleaned.append(
            {
                "frame": int(frame_val),
                "timestamp": float(timestamp_val),
                "reba_score": float(score_val),
            }
        )

    cleaned.sort(key=lambda e: (e["timestamp"], e["frame"]))
    return cleaned


# ---------------------------------------------------------------------------
# B. TEMPORAL SMOOTHING
# ---------------------------------------------------------------------------
def smooth_scores(
    entries: List[Dict[str, Any]], window: int = DEFAULT_SMOOTHING_WINDOW
) -> List[Dict[str, Any]]:
    """Apply a centered moving average to the raw REBA scores.

    The raw scores are never modified or discarded — this function returns
    a NEW list of dicts, each containing both "reba_score" (raw, untouched)
    and "smoothed_score" (the moving-average value).

    Args:
        entries: Validated frame entries (see validate_scores()).
        window: Size of the moving-average window, in frames. Must be a
            positive integer. A window of 1 means no smoothing is applied
            (smoothed_score == reba_score).

    Returns:
        A new list of dicts with an added "smoothed_score" key.

    Raises:
        ValueError: If window is not a positive integer.
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"Smoothing window must be a positive integer, got {window!r}")

    raw_values = [e["reba_score"] for e in entries]
    n = len(raw_values)
    half = window // 2

    smoothed: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_slice = raw_values[lo:hi]
        smoothed.append(sum(window_slice) / len(window_slice))

    result = []
    for entry, smoothed_val in zip(entries, smoothed):
        new_entry = dict(entry)  # copy, so raw entries stay untouched
        new_entry["smoothed_score"] = round(smoothed_val, 3)
        result.append(new_entry)

    return result


# ---------------------------------------------------------------------------
# C. STATISTICS
# ---------------------------------------------------------------------------
def calculate_statistics(smoothed_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics over the (smoothed) frame entries.

    Duration is derived from actual timestamps (max - min), not an
    assumed frame rate, so it stays correct even with variable-FPS input.

    Args:
        smoothed_entries: Output of smooth_scores().

    Returns:
        Dict with frame_count, video_duration, average_score (raw),
        average_smoothed_score, minimum_score, and maximum_score.
    """
    raw_scores = [e["reba_score"] for e in smoothed_entries]
    smoothed_values = [e["smoothed_score"] for e in smoothed_entries]
    timestamps = [e["timestamp"] for e in smoothed_entries]

    duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

    return {
        "frame_count": len(smoothed_entries),
        "video_duration": round(duration, 3),
        "average_score": round(statistics.mean(raw_scores), 3),
        "average_smoothed_score": round(statistics.mean(smoothed_values), 3),
        "minimum_score": min(raw_scores),
        "maximum_score": max(raw_scores),
    }


# ---------------------------------------------------------------------------
# D. RISK DISTRIBUTION
# ---------------------------------------------------------------------------
def calculate_risk_distribution(
    entries: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, float]:
    """Compute the percentage of frames falling into each risk category.

    Uses the RAW reba_score (not the smoothed value) so the distribution
    reflects the actual scored postures, not a smoothed approximation.

    Args:
        entries: Validated (optionally smoothed) frame entries.
        thresholds: Risk category thresholds. Defaults to
            REBA_RISK_THRESHOLDS (see module-level documentation for the
            standard REBA action-level mapping used).

    Returns:
        Dict mapping category name -> percentage of frames (0-100,
        rounded to 1 decimal place). Percentages sum to ~100%.
    """
    thresholds = thresholds or REBA_RISK_THRESHOLDS
    total = len(entries)
    if total == 0:
        return {category: 0.0 for category in thresholds}

    counts = {category: 0 for category in thresholds}
    for entry in entries:
        score = entry["reba_score"]
        for category, bounds in thresholds.items():
            if bounds["min"] <= score <= bounds["max"]:
                counts[category] += 1
                break

    return {
        category: round((count / total) * 100, 1)
        for category, count in counts.items()
    }


# ---------------------------------------------------------------------------
# E. WORST FRAMES
# ---------------------------------------------------------------------------
def find_worst_frames(
    entries: List[Dict[str, Any]], top_n: int = DEFAULT_WORST_FRAME_COUNT
) -> List[Dict[str, Any]]:
    """Identify the highest-risk (highest raw REBA score) frames.

    Args:
        entries: Validated frame entries.
        top_n: How many worst frames to return (3-5 is typical).

    Returns:
        List of dicts with "frame", "timestamp", "score", sorted from
        worst (highest score) to least-bad, longest length = top_n.
    """
    ranked = sorted(
        entries, key=lambda e: (-e["reba_score"], e["frame"])
    )
    worst = ranked[: max(0, top_n)]
    return [
        {
            "frame": e["frame"],
            "timestamp": round(e["timestamp"], 3),
            "score": e["reba_score"],
        }
        for e in worst
    ]


# ---------------------------------------------------------------------------
# F. ASSEMBLE FULL ANALYSIS
# ---------------------------------------------------------------------------
def create_analysis(
    validated_entries: List[Dict[str, Any]],
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    worst_frame_count: int = DEFAULT_WORST_FRAME_COUNT,
) -> Dict[str, Any]:
    """Run the full statistical pipeline and assemble the analysis dict.

    Args:
        validated_entries: Output of validate_scores().
        smoothing_window: Moving-average window size, in frames.
        worst_frame_count: Number of worst frames to report.

    Returns:
        A dict matching the analysis.json schema documented in README.md.
        Includes a "frames" list (raw + smoothed per-frame data) so the
        report generator can plot the timeline without recomputation.
    """
    smoothed_entries = smooth_scores(validated_entries, window=smoothing_window)
    stats = calculate_statistics(smoothed_entries)
    risk_distribution = calculate_risk_distribution(smoothed_entries)
    worst_frames = find_worst_frames(smoothed_entries, top_n=worst_frame_count)

    analysis = {
        "video_duration": stats["video_duration"],
        "frame_count": stats["frame_count"],
        "average_score": stats["average_score"],
        "average_smoothed_score": stats["average_smoothed_score"],
        "minimum_score": stats["minimum_score"],
        "maximum_score": stats["maximum_score"],
        "smoothing_window": smoothing_window,
        "risk_distribution": risk_distribution,
        "risk_thresholds": REBA_RISK_THRESHOLDS,
        "worst_frames": worst_frames,
        "frames": [
            {
                "frame": e["frame"],
                "timestamp": e["timestamp"],
                "raw_score": e["reba_score"],
                "smoothed_score": e["smoothed_score"],
            }
            for e in smoothed_entries
        ],
    }
    return analysis


def save_analysis(analysis: Dict[str, Any], output_path: PathLike) -> Path:
    """Write the analysis dict to a JSON file.

    Args:
        analysis: The dict produced by create_analysis().
        output_path: Destination file path (parent directories are
            created automatically if they don't exist).

    Returns:
        The resolved Path that was written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT — the one function Person 4 needs
# ---------------------------------------------------------------------------
def analyze_scores(
    input_path: PathLike,
    output_directory: PathLike,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    worst_frame_count: int = DEFAULT_WORST_FRAME_COUNT,
    generate_report: bool = True,
) -> Dict[str, Any]:
    """Run the complete Person 3 analysis pipeline.

    This is the single stable function Person 4 (or any other module)
    should call. It loads Person 2's frame-by-frame REBA scores,
    validates them, applies temporal smoothing, computes statistics and
    risk distribution, finds the worst frames, writes analysis.json, and
    (by default) generates the score-timeline graph, risk-distribution
    graph, and an HTML report.

    Args:
        input_path: Path to the JSON file of frame-by-frame REBA scores
            (see README.md for the expected schema).
        output_directory: Directory where analysis.json, the PNG graphs,
            and report.html will be written. Created if it doesn't exist.
        smoothing_window: Moving-average window size, in frames.
        worst_frame_count: Number of worst frames to include.
        generate_report: If True (default), also generate the PNG graphs
            and HTML report via report_generator. Set to False if you
            only need the numeric analysis (e.g. for a fast API response).

    Returns:
        A dict containing:
            video_duration, frame_count, average_score,
            average_smoothed_score, minimum_score, maximum_score,
            risk_distribution, worst_frames, smoothing_window,
            risk_thresholds, and a "paths" sub-dict with the locations of
            all generated files (analysis_json, score_timeline_png,
            risk_distribution_png, report_html — the last three are only
            present if generate_report=True).

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the input is empty, malformed, or fails
            validation (see validate_scores()).
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = load_scores(input_path)
    validated = validate_scores(raw_scores)
    analysis = create_analysis(
        validated,
        smoothing_window=smoothing_window,
        worst_frame_count=worst_frame_count,
    )

    analysis_json_path = save_analysis(analysis, output_dir / "analysis.json")

    paths = {"analysis_json": str(analysis_json_path)}

    if generate_report:
        # Imported here (rather than at module top) so risk_analyzer.py
        # can be imported/tested on its own without requiring matplotlib
        # if a caller only wants generate_report=False.
        from app import report_generator

        report_paths = report_generator.generate_reports(analysis, output_dir)
        paths.update(report_paths)

    result = {
        "video_duration": analysis["video_duration"],
        "frame_count": analysis["frame_count"],
        "average_score": analysis["average_score"],
        "average_smoothed_score": analysis["average_smoothed_score"],
        "minimum_score": analysis["minimum_score"],
        "maximum_score": analysis["maximum_score"],
        "smoothing_window": analysis["smoothing_window"],
        "risk_distribution": analysis["risk_distribution"],
        "risk_thresholds": analysis["risk_thresholds"],
        "worst_frames": analysis["worst_frames"],
        "paths": paths,
    }
    return result


if __name__ == "__main__":
    # Simple manual smoke test / demo run using the bundled sample data.
    here = Path(__file__).parent
    demo_result = analyze_scores(
        here / "sample_scores.json",
        here / "reports",
    )
    print(json.dumps(demo_result, indent=2))
