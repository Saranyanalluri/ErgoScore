from app.angle_calculator import calculate_frame_angles
from app.reba import calculate_reba


def _first_available(values):
    """
    Return the first non-None value.

    This helper is kept for compatibility with the existing
    scoring pipeline.
    """

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
        tuple:
            (
                selected_side,
                upper_arm_angle,
                elbow_angle
            )
    """

    left_upper = angles.get(
        "left_upper_arm"
    )

    right_upper = angles.get(
        "right_upper_arm"
    )

    left_elbow = angles.get(
        "left_elbow"
    )

    right_elbow = angles.get(
        "right_elbow"
    )

    candidates = []

    # ---------------------------------------------------------
    # Check left arm
    # ---------------------------------------------------------

    if (
        left_upper is not None
        and left_elbow is not None
    ):

        candidates.append(
            (
                "left",
                left_upper,
                left_elbow
            )
        )

    # ---------------------------------------------------------
    # Check right arm
    # ---------------------------------------------------------

    if (
        right_upper is not None
        and right_elbow is not None
    ):

        candidates.append(
            (
                "right",
                right_upper,
                right_elbow
            )
        )

    # ---------------------------------------------------------
    # No usable arm
    # ---------------------------------------------------------

    if not candidates:
        return None, None, None

    # ---------------------------------------------------------
    # Only one usable arm
    # ---------------------------------------------------------

    if len(candidates) == 1:
        return candidates[0]

    # ---------------------------------------------------------
    # Both arms available
    #
    # Select the side with the larger upper-arm angle.
    # This represents the more demanding detected side.
    # ---------------------------------------------------------

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

    Frames with insufficient essential posture landmarks
    are skipped by returning None.

    Knee handling
    -------------
    If knee landmarks are detected:
        - The measured knee angle is used for REBA.
        - The measured knee angle is shown in the results.

    If knee landmarks are NOT detected:
        - No knee angle is invented.
        - None is passed to the REBA leg scorer.
        - REBA uses the base leg posture score only.
        - The UI displays "N/A".
    """

    # =========================================================
    # 0. Validate landmarks
    # =========================================================

    if landmarks is None:
        return None

    # =========================================================
    # 1. Calculate posture angles
    # =========================================================

    angles = calculate_frame_angles(
        landmarks
    )

    # =========================================================
    # 2. Validate essential trunk information
    # =========================================================

    if angles.get("trunk") is None:
        return None

    # =========================================================
    # 3. Validate essential neck information
    # =========================================================

    if angles.get("neck") is None:
        return None

    # =========================================================
    # 4. Select arm side
    # =========================================================

    (
        selected_side,
        upper_arm_angle,
        elbow_angle
    ) = _select_arm_side(
        angles
    )

    # =========================================================
    # 5. Validate arm information
    # =========================================================

    if upper_arm_angle is None:
        return None

    if elbow_angle is None:
        return None

    # =========================================================
    # 6. Find available knee angles
    # =========================================================

    left_knee_angle = angles.get(
        "left_knee"
    )

    right_knee_angle = angles.get(
        "right_knee"
    )

    knee_candidates = [
        left_knee_angle,
        right_knee_angle
    ]

    knee_candidates = [
        angle
        for angle in knee_candidates
        if angle is not None
    ]

    # =========================================================
    # 7. Determine observed knee angle
    # =========================================================

    if knee_candidates:

        # If both knees are detected, use the more flexed
        # knee because a smaller angle means greater flexion.

        observed_knee_angle = min(
            knee_candidates
        )

        knee_detected = True

    else:

        # -----------------------------------------------------
        # No usable knee landmarks were detected.
        #
        # IMPORTANT:
        # Do not invent a 180-degree measurement.
        # -----------------------------------------------------

        observed_knee_angle = None

        knee_detected = False

    # =========================================================
    # 8. Calculate REBA
    # =========================================================

    reba_result = calculate_reba(
        trunk_angle=angles["trunk"],

        neck_angle=angles["neck"],

        # None means the knee was not actually detected.
        # score_legs() handles this using the base leg score.
        knee_angle=observed_knee_angle,

        upper_arm_angle=upper_arm_angle,

        elbow_angle=elbow_angle,

        wrist_angle=wrist_angle,

        load_kg=load_kg,

        coupling=coupling,

        bilateral_weight_bearing=(
            bilateral_weight_bearing
        ),

        sitting=sitting,

        trunk_twisting=trunk_twisting,

        trunk_side_bending=(
            trunk_side_bending
        ),

        neck_twisting=neck_twisting,

        neck_side_bending=(
            neck_side_bending
        ),

        upper_arm_abducted=(
            upper_arm_abducted
        ),

        shoulder_raised=(
            shoulder_raised
        ),

        arm_supported=(
            arm_supported
        ),

        wrist_deviated=(
            wrist_deviated
        ),

        shock_or_rapid_force=(
            shock_or_rapid_force
        ),

        static_posture=(
            static_posture
        ),

        repetitive_movements=(
            repetitive_movements
        ),

        rapid_or_unstable_posture=(
            rapid_or_unstable_posture
        )
    )

    # =========================================================
    # 9. Build clean result
    # =========================================================

    return {

        # -----------------------------------------------------
        # All calculated angles
        # -----------------------------------------------------

        "angles": angles,

        # -----------------------------------------------------
        # Representative angles
        #
        # These are the values intended for display/reporting.
        #
        # Knee is None when it was not actually detected.
        # -----------------------------------------------------

        "representative_angles": {
            "upper_arm": upper_arm_angle,
            "elbow": elbow_angle,
            "knee": observed_knee_angle
        },

        # -----------------------------------------------------
        # Knee detection metadata
        # -----------------------------------------------------

        "observed_knee_angle": (
            observed_knee_angle
        ),

        "knee_detected": (
            knee_detected
        ),

        # -----------------------------------------------------
        # Internal scoring value
        #
        # This is now the actual observed knee angle or None.
        # There is no fake 180-degree fallback.
        # -----------------------------------------------------

        "knee_angle_for_scoring": (
            observed_knee_angle
        ),

        # -----------------------------------------------------
        # Selected arm
        # -----------------------------------------------------

        "selected_side": selected_side,

        # -----------------------------------------------------
        # REBA result
        # -----------------------------------------------------

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

    Frames that do not contain enough usable posture
    information are skipped.

    Returns:
        List of successfully scored frame results.
    """

    # =========================================================
    # 1. Validate input
    # =========================================================

    if keypoints is None:
        raise ValueError(
            "keypoints cannot be None."
        )

    results = []

    # =========================================================
    # 2. Process every frame
    # =========================================================

    for frame_data in keypoints:

        # -----------------------------------------------------
        # Validate frame structure
        # -----------------------------------------------------

        if "landmarks" not in frame_data:
            raise ValueError(
                "Each frame must contain 'landmarks'."
            )

        # -----------------------------------------------------
        # Process current frame
        # -----------------------------------------------------

        result = process_frame(
            landmarks=frame_data["landmarks"],

            load_kg=load_kg,

            coupling=coupling,

            wrist_angle=wrist_angle,

            bilateral_weight_bearing=(
                bilateral_weight_bearing
            ),

            sitting=sitting,

            trunk_twisting=(
                trunk_twisting
            ),

            trunk_side_bending=(
                trunk_side_bending
            ),

            neck_twisting=(
                neck_twisting
            ),

            neck_side_bending=(
                neck_side_bending
            ),

            upper_arm_abducted=(
                upper_arm_abducted
            ),

            shoulder_raised=(
                shoulder_raised
            ),

            arm_supported=(
                arm_supported
            ),

            wrist_deviated=(
                wrist_deviated
            ),

            shock_or_rapid_force=(
                shock_or_rapid_force
            ),

            static_posture=(
                static_posture
            ),

            repetitive_movements=(
                repetitive_movements
            ),

            rapid_or_unstable_posture=(
                rapid_or_unstable_posture
            )
        )

        # -----------------------------------------------------
        # Skip unusable frames
        # -----------------------------------------------------

        if result is None:
            continue

        # -----------------------------------------------------
        # Add frame metadata
        # -----------------------------------------------------

        results.append({
            "frame": frame_data.get(
                "frame"
            ),

            "timestamp": frame_data.get(
                "timestamp"
            ),

            **result
        })

    # =========================================================
    # 3. Return all successfully scored frames
    # =========================================================

    return results