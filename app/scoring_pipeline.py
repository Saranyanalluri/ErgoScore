from app.angle_calculator import calculate_frame_angles
from app.reba import calculate_reba


def _first_available(values):
    """Return the first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def _select_arm_side(angles):
    """
    Select the available arm side.

    If both sides are available, use the side with the
    larger upper-arm angle as the more demanding side.

    Returns:
        side, upper_arm_angle, elbow_angle
    """

    left_upper = angles.get("left_upper_arm")
    right_upper = angles.get("right_upper_arm")

    left_elbow = angles.get("left_elbow")
    right_elbow = angles.get("right_elbow")

    candidates = []

    if left_upper is not None and left_elbow is not None:
        candidates.append(
            ("left", left_upper, left_elbow)
        )

    if right_upper is not None and right_elbow is not None:
        candidates.append(
            ("right", right_upper, right_elbow)
        )

    if not candidates:
        return None, None, None

    if len(candidates) == 1:
        return candidates[0]

    # Select the side with the larger upper-arm angle.
    selected = max(
        candidates,
        key=lambda item: item[1]
    )

    return selected


def process_frame(
    landmarks,
    load_kg=0.0,
    coupling=0,
    wrist_angle=0.0,
    bilateral_weight_bearing=True,
    sitting=False,
    trunk_twisting=False,
    trunk_side_bending=False,
    neck_twisting=False,
    neck_side_bending=False,
    upper_arm_abducted=False,
    shoulder_raised=False,
    arm_supported=False,
    wrist_deviated=False,
    shock_or_rapid_force=False,
    static_posture=False,
    repetitive_movements=False,
    rapid_or_unstable_posture=False
):
    """
    Process one frame of normalized landmark data.

    Missing landmarks are handled safely.

    For a seated worker, knee/ankle data is not required because
    REBA can use the seated leg base score.
    """

    if landmarks is None:
        raise ValueError("landmarks cannot be None.")

    # ---------------------------------------------------------
    # 1. Calculate posture angles
    # ---------------------------------------------------------

    angles = calculate_frame_angles(landmarks)

    # ---------------------------------------------------------
    # 2. Validate essential trunk/neck information
    # ---------------------------------------------------------

    if angles["trunk"] is None:
        raise ValueError(
            "Trunk angle could not be calculated."
        )

    if angles["neck"] is None:
        raise ValueError(
            "Neck angle could not be calculated."
        )

    # ---------------------------------------------------------
    # 3. Select arm side
    # ---------------------------------------------------------

    selected_side, upper_arm_angle, elbow_angle = (
        _select_arm_side(angles)
    )

    if upper_arm_angle is None:
        raise ValueError(
            "No usable upper-arm data was found."
        )

    if elbow_angle is None:
        raise ValueError(
            "No usable elbow data was found."
        )

    # ---------------------------------------------------------
    # 4. Select knee angle when available
    # ---------------------------------------------------------

    knee_candidates = [
        angles.get("left_knee"),
        angles.get("right_knee")
    ]

    knee_candidates = [
        angle for angle in knee_candidates
        if angle is not None
    ]

    if knee_candidates:
        # Smaller angle means greater knee flexion.
        knee_angle = min(knee_candidates)
    else:
        knee_angle = None

    # A missing knee is acceptable for an explicitly seated worker.
    if knee_angle is None and not sitting:
        raise ValueError(
            "Knee angle could not be calculated. "
            "Provide knee/ankle landmarks or use sitting=True."
        )

    # ---------------------------------------------------------
    # 5. Calculate REBA
    # ---------------------------------------------------------

    reba_result = calculate_reba(
        trunk_angle=angles["trunk"],
        neck_angle=angles["neck"],
        knee_angle=knee_angle,
        upper_arm_angle=upper_arm_angle,
        elbow_angle=elbow_angle,
        wrist_angle=wrist_angle,
        load_kg=load_kg,
        coupling=coupling,
        bilateral_weight_bearing=bilateral_weight_bearing,
        sitting=sitting,
        trunk_twisting=trunk_twisting,
        trunk_side_bending=trunk_side_bending,
        neck_twisting=neck_twisting,
        neck_side_bending=neck_side_bending,
        upper_arm_abducted=upper_arm_abducted,
        shoulder_raised=shoulder_raised,
        arm_supported=arm_supported,
        wrist_deviated=wrist_deviated,
        shock_or_rapid_force=shock_or_rapid_force,
        static_posture=static_posture,
        repetitive_movements=repetitive_movements,
        rapid_or_unstable_posture=rapid_or_unstable_posture
    )

    # ---------------------------------------------------------
    # 6. Return clean result
    # ---------------------------------------------------------

    return {
        "angles": angles,

        "representative_angles": {
            "upper_arm": upper_arm_angle,
            "elbow": elbow_angle,
            "knee": knee_angle
        },

        "selected_side": selected_side,

        "reba": reba_result
    }


def process_keypoints(
    keypoints,
    load_kg=0.0,
    coupling=0,
    wrist_angle=0.0,
    bilateral_weight_bearing=True,
    sitting=False,
    trunk_twisting=False,
    trunk_side_bending=False,
    neck_twisting=False,
    neck_side_bending=False,
    upper_arm_abducted=False,
    shoulder_raised=False,
    arm_supported=False,
    wrist_deviated=False,
    shock_or_rapid_force=False,
    static_posture=False,
    repetitive_movements=False,
    rapid_or_unstable_posture=False
):
    """
    Process multiple frames using the current Person 2
    normalized landmark format.

    Expected input:

    [
        {
            "frame": 1,
            "timestamp": 0.0,
            "landmarks": {
                "nose": [0.5, 0.2],
                ...
            }
        }
    ]

    Returns a list of frame results.
    """

    if keypoints is None:
        raise ValueError("keypoints cannot be None.")

    results = []

    for frame_data in keypoints:

        if "landmarks" not in frame_data:
            raise ValueError(
                "Each frame must contain 'landmarks'."
            )

        result = process_frame(
            landmarks=frame_data["landmarks"],
            load_kg=load_kg,
            coupling=coupling,
            wrist_angle=wrist_angle,
            bilateral_weight_bearing=bilateral_weight_bearing,
            sitting=sitting,
            trunk_twisting=trunk_twisting,
            trunk_side_bending=trunk_side_bending,
            neck_twisting=neck_twisting,
            neck_side_bending=neck_side_bending,
            upper_arm_abducted=upper_arm_abducted,
            shoulder_raised=shoulder_raised,
            arm_supported=arm_supported,
            wrist_deviated=wrist_deviated,
            shock_or_rapid_force=shock_or_rapid_force,
            static_posture=static_posture,
            repetitive_movements=repetitive_movements,
            rapid_or_unstable_posture=rapid_or_unstable_posture
        )

        results.append({
            "frame": frame_data.get("frame"),
            "timestamp": frame_data.get("timestamp"),
            **result
        })

    return results