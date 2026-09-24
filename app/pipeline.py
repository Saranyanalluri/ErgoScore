"""
ErgoScore complete analysis pipeline.

VIDEO
  ↓
Person 1: MediaPipe pose extraction
  ↓
Person 2: Joint angles + REBA scoring
  ↓
Person 3: Temporal smoothing + statistics + reports
  ↓
Flask-compatible result dictionary
"""

import json
import os
import cv2

from datetime import datetime, timezone
from pathlib import Path

from app.video_processor import process_video
from app.person1_adapter import convert_keypoints
from app.scoring_pipeline import process_keypoints
from app.risk_analyzer import analyze_scores


# =========================================================
# RISK BANDS
# =========================================================

RISK_BANDS = [
    (0, 3, "Low"),
    (4, 7, "Medium"),
    (8, 15, "High"),
]


# =========================================================
# INPUT MAPPINGS
# =========================================================

LOAD_VALUES = {
    "none": 0.0,
    "light": 2.0,
    "medium": 5.0,
    "heavy": 10.0,
}


COUPLING_VALUES = {
    "good": 0,
    "fair": 1,
    "poor": 2,
}


# =========================================================
# RISK LEVEL
# =========================================================

def _risk_level_for_score(score):
    """
    Convert a REBA score into the dashboard risk category.
    """

    for low, high, label in RISK_BANDS:

        if low <= score <= high:
            return label

    return "High"


# =========================================================
# LOAD CONVERSION
# =========================================================

def _load_value(load):
    """
    Convert UI load selection into kilograms.
    """

    if isinstance(load, (int, float)):
        return float(load)

    return LOAD_VALUES.get(
        str(load).lower(),
        5.0
    )


# =========================================================
# COUPLING CONVERSION
# =========================================================

def _coupling_value(coupling):
    """
    Convert UI coupling selection into numeric value.
    """

    if isinstance(coupling, (int, float)):
        return int(coupling)

    return COUPLING_VALUES.get(
        str(coupling).lower(),
        1
    )


# =========================================================
# PERSON 2 → PERSON 3 ADAPTER
# =========================================================

def _write_person2_scores(
    scored_frames,
    output_path
):
    """
    Convert Person 2 results into Person 3 input format.

    Person 2 format:

        {
            "frame": 0,
            "timestamp": 0.0,
            "reba": {
                "reba_score": 4
            }
        }

    Person 3 format:

        {
            "frame": 0,
            "timestamp": 0.0,
            "reba_score": 4
        }
    """

    scores = []

    for result in scored_frames:

        reba = result.get(
            "reba",
            {}
        )

        if not isinstance(reba, dict):
            continue

        score = reba.get(
            "reba_score"
        )

        if score is None:
            continue

        scores.append({
            "frame": result.get(
                "frame",
                0
            ),

            "timestamp": result.get(
                "timestamp",
                0.0
            ),

            "reba_score": int(
                round(
                    float(score)
                )
            ),
        })

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            scores,
            indent=2
        ),
        encoding="utf-8"
    )

    return scores


# =========================================================
# FRAME LOOKUP
# =========================================================

def _build_frame_lookup(scored_frames):
    """
    Create a quick frame-number → Person 2 result lookup.
    """

    return {
        result.get("frame"): result
        for result in scored_frames
        if result.get("frame") is not None
    }


# =========================================================
# TIMELINE
# =========================================================

def _build_timeline(analysis):
    """
    Convert Person 3 frame analysis into dashboard format.

    The timeline intentionally uses the smoothed score because
    the timeline is meant to show the temporal trend.

    Dashboard format:

        {
            "time": seconds,
            "score": smoothed_score
        }
    """

    timeline = []

    for frame in analysis.get(
        "frames",
        []
    ):

        score = frame.get(
            "smoothed_score"
        )

        if score is None:
            score = frame.get(
                "score",
                0
            )

        timestamp = frame.get(
            "timestamp",
            0.0
        )

        timeline.append({
            "time": round(
                float(timestamp),
                2
            ),

            "score": round(
                float(score),
                2
            ),
        })

    return timeline


# =========================================================
# WORST FRAMES
# =========================================================

def _build_worst_frames(
    analysis,
    scored_frames
):
    """
    Select the five worst posture moments.

    IMPORTANT:

    The dashboard card labeled "REBA SCORE" uses the
    ORIGINAL FRAME-LEVEL REBA SCORE from Person 2.

    It does NOT use the temporally smoothed score.

    Therefore:

        Raw REBA:
            5, 6, 7, 8, ...

        Smoothed score:
            5.2, 6.2, 7.2, ...

    The raw REBA value is displayed as the actual REBA score.

    The smoothed value is retained separately as
    "smoothed_score".
    """

    frame_lookup = _build_frame_lookup(
        scored_frames
    )

    all_frames = analysis.get(
        "frames",
        []
    )

    if not all_frames:
        return []

    # ---------------------------------------------------------
    # Sort candidates.
    #
    # PRIMARY:
    #     raw frame-level REBA score
    #
    # SECONDARY:
    #     smoothed score
    #
    # TERTIARY:
    #     timestamp
    #
    # This makes the worst-frame cards represent actual
    # REBA scores rather than averages.
    # ---------------------------------------------------------

    def candidate_sort_key(frame):

        raw_score = frame.get(
            "score"
        )

        if raw_score is None:
            raw_score = frame.get(
                "reba_score",
                0
            )

        smoothed_score = frame.get(
            "smoothed_score"
        )

        if smoothed_score is None:
            smoothed_score = raw_score

        timestamp = frame.get(
            "timestamp",
            0
        )

        try:
            raw_score = float(
                raw_score
            )
        except (
            TypeError,
            ValueError
        ):
            raw_score = 0.0

        try:
            smoothed_score = float(
                smoothed_score
            )
        except (
            TypeError,
            ValueError
        ):
            smoothed_score = 0.0

        try:
            timestamp = float(
                timestamp
            )
        except (
            TypeError,
            ValueError
        ):
            timestamp = 0.0

        return (
            raw_score,
            smoothed_score,
            -timestamp
        )

    candidates = sorted(
        all_frames,
        key=candidate_sort_key,
        reverse=True
    )

    selected = []

    # ---------------------------------------------------------
    # Minimum separation between selected frames
    #
    # Video is approximately 30 FPS.
    # 15 frames ≈ 0.5 seconds.
    # ---------------------------------------------------------

    MIN_FRAME_GAP = 15

    for candidate in candidates:

        frame_number = candidate.get(
            "frame"
        )

        if frame_number is None:
            continue

        frame_number = int(
            frame_number
        )

        # -----------------------------------------------------
        # Prevent selecting frames that are too close.
        # -----------------------------------------------------

        too_close = any(
            abs(
                frame_number
                - int(
                    existing.get(
                        "frame_number",
                        -999999
                    )
                )
            ) < MIN_FRAME_GAP
            for existing in selected
        )

        if too_close:
            continue

        # -----------------------------------------------------
        # Find corresponding Person 2 result.
        # -----------------------------------------------------

        person2_result = frame_lookup.get(
            frame_number,
            {}
        )

        angles = person2_result.get(
            "angles",
            {}
        )

        representative = person2_result.get(
            "representative_angles",
            {}
        )

        # -----------------------------------------------------
        # COMPLETE REBA BREAKDOWN
        # -----------------------------------------------------

        reba_breakdown = person2_result.get(
            "reba",
            {}
        )

        if not isinstance(reba_breakdown, dict):
            reba_breakdown = {}

        # -----------------------------------------------------
        # RAW REBA SCORE
        #
        # IMPORTANT:
        # Use the actual Person 2 REBA score.
        # -----------------------------------------------------

        reba_result = person2_result.get(
            "reba",
            {}
        )

        raw_score = reba_result.get(
            "reba_score"
        )

        # -----------------------------------------------------
        # Fallback to Person 3 raw score if necessary.
        # -----------------------------------------------------

        if raw_score is None:

            raw_score = candidate.get(
                "score"
            )

        if raw_score is None:

            raw_score = candidate.get(
                "reba_score",
                0
            )

        try:

            raw_score = int(
                round(
                    float(raw_score)
                )
            )

        except (
            TypeError,
            ValueError
        ):

            raw_score = 0

        # -----------------------------------------------------
        # Smoothed score.
        #
        # This is retained separately and is NOT displayed as
        # the primary REBA SCORE.
        # -----------------------------------------------------

        smoothed_score = candidate.get(
            "smoothed_score"
        )

        if smoothed_score is None:
            smoothed_score = raw_score

        try:

            smoothed_score = round(
                float(smoothed_score),
                2
            )

        except (
            TypeError,
            ValueError
        ):

            smoothed_score = float(
                raw_score
            )

        # -----------------------------------------------------
        # Timestamp.
        # -----------------------------------------------------

        timestamp = candidate.get(
            "timestamp",
            0.0
        )

        try:

            timestamp_float = float(
                timestamp
            )

        except (
            TypeError,
            ValueError
        ):

            timestamp_float = 0.0

        # -----------------------------------------------------
        # Knee angle.
        #
        # Missing knee remains None.
        # Never convert it to 180.
        # -----------------------------------------------------

        observed_knee = representative.get(
            "knee"
        )

        if observed_knee is not None:

            try:

                knee_angle = round(
                    float(observed_knee),
                    1
                )

            except (
                TypeError,
                ValueError
            ):

                knee_angle = None

        else:

            knee_angle = None

        # -----------------------------------------------------
        # Trunk angle.
        # -----------------------------------------------------

        observed_trunk = angles.get(
            "trunk"
        )

        if observed_trunk is not None:

            try:

                trunk_angle = round(
                    float(observed_trunk),
                    1
                )

            except (
                TypeError,
                ValueError
            ):

                trunk_angle = None

        else:

            trunk_angle = None

        # -----------------------------------------------------
        # Neck angle.
        # -----------------------------------------------------

        observed_neck = angles.get(
            "neck"
        )

        if observed_neck is not None:

            try:

                neck_angle = round(
                    float(observed_neck),
                    1
                )

            except (
                TypeError,
                ValueError
            ):

                neck_angle = None

        else:

            neck_angle = None

        # -----------------------------------------------------
        # Add selected frame.
        # -----------------------------------------------------

        selected.append({

            "frame_number": frame_number,

            "timestamp": (
                f"{timestamp_float:.2f}s"
            ),

            # -------------------------------------------------
            # IMPORTANT:
            #
            # This is the REAL integer REBA score.
            # -------------------------------------------------

            "score": raw_score,

            # -------------------------------------------------
            # Keep smoothed score separately.
            # -------------------------------------------------

            "smoothed_score": smoothed_score,

            # -------------------------------------------------
            # Posture measurements.
            # -------------------------------------------------

            "trunk_angle": trunk_angle,

            "neck_angle": neck_angle,

            "knee_angle": knee_angle,

            "knee_detected": (
                knee_angle is not None
            ),

            # -------------------------------------------------
            # COMPLETE REBA BREAKDOWN
            # -------------------------------------------------

            "reba_breakdown": reba_breakdown,

            # -------------------------------------------------
            # Image URL is populated later.
            # -------------------------------------------------

            "image_url": None,
        })

        # -----------------------------------------------------
        # Stop after five frames.
        # -----------------------------------------------------

        if len(selected) >= 5:
            break

    return selected


# =========================================================
# SNAPSHOT GENERATION
# =========================================================

def _generate_worst_frame_snapshots(
    video_path,
    worst_frames,
    output_directory
):
    """
    Extract the selected worst posture frames from the
    original video and save them as JPEG images.
    """

    video_path = Path(
        video_path
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        return worst_frames

    try:

        for frame_info in worst_frames:

            frame_number = frame_info.get(
                "frame_number"
            )

            if frame_number is None:
                continue

            # -------------------------------------------------
            # Seek to requested frame.
            # -------------------------------------------------

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_number)
            )

            success, frame = capture.read()

            if (
                not success
                or frame is None
            ):
                continue

            # -------------------------------------------------
            # Save snapshot.
            # -------------------------------------------------

            filename = (
                f"frame_{int(frame_number)}.jpg"
            )

            snapshot_path = (
                output_directory
                / filename
            )

            cv2.imwrite(
                str(snapshot_path),
                frame
            )

            # -------------------------------------------------
            # URL used by Flask.
            # -------------------------------------------------

            frame_info["image_url"] = (
                f"/reports/"
                f"{video_path.stem}/"
                f"snapshots/"
                f"{filename}"
            )

    finally:

        capture.release()

    return worst_frames


# =========================================================
# SUMMARY
# =========================================================

def _build_summary(
    analysis,
    risk_level
):
    """
    Create a dashboard-friendly factual summary.
    """

    average = analysis.get(
        "average_score",
        0
    )

    maximum = analysis.get(
        "maximum_score",
        0
    )

    duration = analysis.get(
        "video_duration",
        0
    )

    distribution = analysis.get(
        "risk_distribution",
        {}
    )

    return (
        f"The analyzed activity lasted "
        f"{duration:.1f} seconds across "
        f"{analysis.get('frame_count', 0)} "
        f"scored frames. "

        f"The average REBA score was "
        f"{average:.1f}, with a maximum "
        f"score of {maximum:.0f}. "

        f"The overall risk category based "
        f"on the maximum observed REBA "
        f"score is {risk_level}. "

        f"Risk distribution was "
        f"{distribution.get('high', 0):.1f}% high, "
        f"{distribution.get('medium', 0):.1f}% medium, "
        f"and {distribution.get('low', 0):.1f}% low."
    )



# =========================================================
# RECOMMENDATIONS
# =========================================================

def _build_recommendations(worst_frames, risk_level):
    """
    Generate dashboard recommendations from the observed REBA
    component scores and measured posture angles.

    Recommendations are descriptive and rule-based. They do not
    replace an ergonomic assessment by a qualified professional.
    """

    recommendations = []

    if not worst_frames:
        return [{
            "priority": "Info",
            "title": "Insufficient posture detail",
            "detail": "No selected posture frame was available for component-level recommendations.",
            "metric": "No frame data"
        }]

    # Look across the selected worst moments rather than relying
    # on only one frame.
    max_components = {
        "trunk": 0,
        "neck": 0,
        "legs": 0,
        "upper_arm": 0,
        "lower_arm": 0,
        "wrist": 0,
    }

    max_angles = {
        "trunk": None,
        "neck": None,
        "knee": None,
    }

    for frame in worst_frames:
        breakdown = frame.get("reba_breakdown") or {}
        group_a = breakdown.get("group_a") or {}
        group_b = breakdown.get("group_b") or {}

        for key in ("trunk", "neck", "legs"):
            try:
                value = int(group_a.get(key, 0))
            except (TypeError, ValueError):
                value = 0
            max_components[key] = max(max_components[key], value)

        for key in ("upper_arm", "lower_arm"):
            try:
                value = int(group_b.get(key, 0))
            except (TypeError, ValueError):
                value = 0
            max_components[key] = max(max_components[key], value)

        # Wrist may be represented by either wrist or wrist_score
        # depending on the scoring output.
        wrist_value = group_b.get("wrist", group_b.get("wrist_score", 0))
        try:
            wrist_value = int(wrist_value)
        except (TypeError, ValueError):
            wrist_value = 0
        max_components["wrist"] = max(max_components["wrist"], wrist_value)

        # Frame-level angle keys use the *_angle names, while
        # max_angles uses the shorter component names.
        angle_mapping = {
            "trunk": "trunk_angle",
            "neck": "neck_angle",
            "knee": "knee_angle",
        }

        for component, frame_key in angle_mapping.items():
            value = frame.get(frame_key)
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                if max_angles[component] is None or value > max_angles[component]:
                    max_angles[component] = value

    # Component score 3+ is used as the trigger for the
    # component-specific guidance. This keeps the section useful
    # without generating advice for every small score difference.
    if max_components["trunk"] >= 3:
        angle_text = (
            f"Peak observed trunk angle: {max_angles['trunk']:.1f}°."
            if max_angles["trunk"] is not None
            else "Trunk component score reached 3 or higher."
        )
        recommendations.append({
            "priority": "High",
            "title": "Reduce trunk flexion",
            "detail": "Keep the torso closer to an upright position and, where practical, bring the task or work surface closer instead of repeatedly bending forward.",
            "metric": angle_text
        })

    if max_components["neck"] >= 2:
        angle_text = (
            f"Peak observed neck angle: {max_angles['neck']:.1f}°."
            if max_angles["neck"] is not None
            else "Neck component score reached 2."
        )
        recommendations.append({
            "priority": "Medium",
            "title": "Reduce neck bending",
            "detail": "Keep the head and neck closer to a neutral alignment. Raise or reposition frequently viewed work so prolonged downward or extended viewing is reduced.",
            "metric": angle_text
        })

    if max_components["upper_arm"] >= 3:
        recommendations.append({
            "priority": "High",
            "title": "Reduce upper-arm elevation",
            "detail": "Keep frequently used tools and materials within a comfortable working range and avoid holding the upper arm elevated for prolonged periods.",
            "metric": f"Upper-arm component score: {max_components['upper_arm']}"
        })
    elif max_components["upper_arm"] >= 2:
        recommendations.append({
            "priority": "Medium",
            "title": "Monitor arm elevation",
            "detail": "Where possible, keep the work area close to the body and reduce unnecessary shoulder elevation during repeated tasks.",
            "metric": f"Upper-arm component score: {max_components['upper_arm']}"
        })

    if max_components["lower_arm"] >= 2:
        recommendations.append({
            "priority": "Medium",
            "title": "Keep the forearm in a comfortable range",
            "detail": "Reposition the task or tools when possible so the forearm can remain in a more comfortable working range rather than repeatedly reaching or holding awkward positions.",
            "metric": f"Lower-arm component score: {max_components['lower_arm']}"
        })

    if max_components["wrist"] >= 2:
        recommendations.append({
            "priority": "Medium",
            "title": "Reduce awkward wrist posture",
            "detail": "Keep the wrist closer to neutral where practical and adjust the position of tools or input devices to avoid sustained bending.",
            "metric": f"Wrist component score: {max_components['wrist']}"
        })

    # A missing knee angle is deliberately not treated as a posture
    # problem. The current pipeline uses the leg base score when
    # lower-body landmarks are unavailable.
    if max_components["legs"] >= 2 and max_angles["knee"] is not None:
        recommendations.append({
            "priority": "Medium",
            "title": "Review lower-body posture",
            "detail": "Check whether the working position can be adjusted to provide stable weight bearing and reduce prolonged deep knee flexion.",
            "metric": f"Observed knee angle in selected moments: up to {max_angles['knee']:.1f}°"
        })

    if not recommendations:
        if risk_level == "Low":
            recommendations.append({
                "priority": "Info",
                "title": "Maintain the observed posture",
                "detail": "The selected posture components did not trigger a component-specific recommendation. Continue monitoring the task, especially during longer or repetitive work.",
                "metric": "No elevated component threshold detected"
            })
        else:
            recommendations.append({
                "priority": "Info",
                "title": "Review the selected high-score moments",
                "detail": "The overall score indicates that the selected moments should be reviewed together with the measured angles and REBA breakdown.",
                "metric": f"Overall category: {risk_level}"
            })

    # Keep the section concise on the dashboard.
    priority_order = {"High": 0, "Medium": 1, "Info": 2}
    recommendations.sort(
        key=lambda item: priority_order.get(item.get("priority"), 3)
    )

    return recommendations[:5]


# =========================================================
# MAIN PIPELINE
# =========================================================

def analyze_video(
    video_path,
    load="medium",
    coupling="fair"
):
    """
    Run the complete ErgoScore analysis.

    Pipeline:

        Video
          ↓
        Person 1
          ↓
        Keypoints
          ↓
        Person 2
          ↓
        REBA frame scores
          ↓
        Person 3
          ↓
        Temporal analysis + reports
          ↓
        Flask result dictionary
    """

    if not video_path:
        raise ValueError(
            "video_path is required."
        )

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # =====================================================
    # PERSON 1
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "ERGOScore - PERSON 1: POSE EXTRACTION"
    )
    print("=" * 70)

    person1_result = process_video(
        input_path=video_path
    )

    keypoints_path = Path(
        person1_result.keypoints_json_path
    )

    print(
        f"Keypoints: {keypoints_path}"
    )

    # =====================================================
    # LOAD PERSON 1 KEYPOINTS
    # =====================================================

    with keypoints_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        keypoints_data = json.load(f)

    # =====================================================
    # PERSON 1 → PERSON 2
    # =====================================================

    converted = convert_keypoints(
        keypoints_data
    )

    # =====================================================
    # PERSON 2
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "ERGOScore - PERSON 2: REBA SCORING"
    )
    print("=" * 70)

    scored_frames = process_keypoints(
        converted["frames"],
        load_kg=_load_value(load),
        coupling=_coupling_value(coupling),
        sitting=False
    )

    if not scored_frames:
        raise ValueError(
            "No frames could be scored. "
            "The video may not contain enough "
            "usable pose landmarks."
        )

    print(
        f"Successfully scored frames: "
        f"{len(scored_frames)}"
    )

    # =====================================================
    # PERSON 2 → PERSON 3
    # =====================================================

    scores_path = (
        Path("data")
        / "keypoints"
        / f"{video_path.stem}_scores.json"
    )

    _write_person2_scores(
        scored_frames,
        scores_path
    )

    print(
        f"Score file: {scores_path}"
    )

    # =====================================================
    # PERSON 3
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "ERGOScore - PERSON 3: TEMPORAL ANALYSIS"
    )
    print("=" * 70)

    report_directory = (
        Path("reports")
        / video_path.stem
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    analysis = analyze_scores(
        input_path=scores_path,
        output_directory=report_directory,
        smoothing_window=5,
        worst_frame_count=5,
        generate_report=True
    )

    print(
        f"Average REBA: "
        f"{analysis['average_score']:.2f}"
    )

    print(
        f"Maximum REBA: "
        f"{analysis['maximum_score']:.0f}"
    )

    # =====================================================
    # LOAD COMPLETE PERSON 3 ANALYSIS
    # =====================================================

    analysis_json_path = (
        report_directory
        / "analysis.json"
    )

    if analysis_json_path.exists():

        with analysis_json_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            full_analysis = json.load(f)

        # Person 3's public return object does not
        # necessarily contain every frame.

        analysis["frames"] = (
            full_analysis.get(
                "frames",
                []
            )
        )

    else:

        analysis["frames"] = []

    # =====================================================
    # BUILD FINAL DASHBOARD DATA
    # =====================================================

    maximum_score = analysis[
        "maximum_score"
    ]

    risk_level = _risk_level_for_score(
        maximum_score
    )

    # -----------------------------------------------------
    # Timeline
    #
    # Timeline intentionally uses smoothed scores.
    # -----------------------------------------------------

    timeline = _build_timeline(
        analysis
    )

    # -----------------------------------------------------
    # Worst frames
    #
    # Worst-frame cards use RAW integer REBA.
    # -----------------------------------------------------

    worst_frames = _build_worst_frames(
        analysis,
        scored_frames
    )

    # -----------------------------------------------------
    # Generate actual snapshots
    # -----------------------------------------------------

    worst_frames = (
        _generate_worst_frame_snapshots(
            video_path=video_path,
            worst_frames=worst_frames,
            output_directory=(
                report_directory
                / "snapshots"
            )
        )
    )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = _build_recommendations(
        worst_frames,
        risk_level
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = _build_summary(
        analysis,
        risk_level
    )

    # =====================================================
    # PERSON 3 GENERATED FILES
    # =====================================================

    paths = analysis.get(
        "paths",
        {}
    )

    # =====================================================
    # FINAL FLASK RESULT
    # =====================================================

    return {

        # -------------------------------------------------
        # Main score
        # -------------------------------------------------

        "overall_score": maximum_score,

        "risk_level": risk_level,

        "average_score": analysis[
            "average_score"
        ],

        "maximum_score": analysis[
            "maximum_score"
        ],

        # -------------------------------------------------
        # Video information
        # -------------------------------------------------

        "duration": (
            f"{analysis['video_duration']:.1f} sec"
        ),

        # -------------------------------------------------
        # Risk distribution
        # -------------------------------------------------

        "risk_distribution": analysis[
            "risk_distribution"
        ],

        # -------------------------------------------------
        # Timeline
        # -------------------------------------------------

        "timeline": timeline,

        # -------------------------------------------------
        # Worst posture frames
        # -------------------------------------------------

        "worst_frames": worst_frames,

        # -------------------------------------------------
        # Recommendations
        # -------------------------------------------------

        "recommendations": recommendations,

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        "summary": summary,

        # -------------------------------------------------
        # Video filename
        # -------------------------------------------------

        "video_filename": video_path.name,

        # -------------------------------------------------
        # Person 1 outputs
        # -------------------------------------------------

        "processed_video": str(
            person1_result.processed_video_path
        ),

        "keypoints_file": str(
            keypoints_path
        ),

        # -------------------------------------------------
        # Person 2 output
        # -------------------------------------------------

        "scores_file": str(
            scores_path
        ),

        # -------------------------------------------------
        # Person 3 outputs
        # -------------------------------------------------

        "report_file": paths.get(
            "report_html"
        ),

        "timeline_image": paths.get(
            "score_timeline_png"
        ),

        "risk_distribution_image": paths.get(
            "risk_distribution_png"
        ),

        "analysis_file": paths.get(
            "analysis_json"
        ),

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "is_dummy_data": False,

        # -------------------------------------------------
        # Processing statistics
        # -------------------------------------------------

        "frames_processed": (
            person1_result.frames_processed
        ),

        "frames_with_pose": (
            person1_result.frames_with_pose
        ),

        "frames_scored": len(
            scored_frames
        ),
    }