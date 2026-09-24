import math


# ============================================================
# BASIC GEOMETRY
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point B formed by A-B-C.

    Points can be:
        [x, y]
        (x, y)

    Returns:
        Angle in degrees from 0 to 180.
        Returns None if a point is missing or a vector
        has zero length.
    """

    if (
        point_a is None
        or point_b is None
        or point_c is None
    ):
        return None

    try:
        ax, ay = float(point_a[0]), float(point_a[1])
        bx, by = float(point_b[0]), float(point_b[1])
        cx, cy = float(point_c[0]), float(point_c[1])

    except (
        TypeError,
        ValueError,
        IndexError
    ):
        return None

    ba_x = ax - bx
    ba_y = ay - by

    bc_x = cx - bx
    bc_y = cy - by

    magnitude_ba = math.hypot(
        ba_x,
        ba_y
    )

    magnitude_bc = math.hypot(
        bc_x,
        bc_y
    )

    if (
        magnitude_ba < 1e-6
        or magnitude_bc < 1e-6
    ):
        return None

    dot_product = (
        ba_x * bc_x
        + ba_y * bc_y
    )

    cosine = (
        dot_product
        / (magnitude_ba * magnitude_bc)
    )

    # Protect against floating-point errors.
    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


def midpoint(point_a, point_b):
    """
    Calculate the midpoint between two points.
    """

    if (
        point_a is None
        or point_b is None
    ):
        return None

    try:
        return [
            (
                float(point_a[0])
                + float(point_b[0])
            ) / 2.0,

            (
                float(point_a[1])
                + float(point_b[1])
            ) / 2.0
        ]

    except (
        TypeError,
        ValueError,
        IndexError
    ):
        return None


# ============================================================
# VECTOR HELPERS
# ============================================================

def _normalize_vector(x, y):
    """
    Normalize a 2D vector.

    Returns:
        (x, y) normalized
        None if vector length is too small.
    """

    length = math.hypot(
        x,
        y
    )

    if length < 1e-6:
        return None

    return (
        x / length,
        y / length
    )


def _angle_between_vectors(
    vector_a,
    vector_b
):
    """
    Calculate the smaller angle between two vectors.

    Returns:
        0-180 degrees
        None if invalid.
    """

    ax, ay = vector_a
    bx, by = vector_b

    magnitude_a = math.hypot(
        ax,
        ay
    )

    magnitude_b = math.hypot(
        bx,
        by
    )

    if (
        magnitude_a < 1e-6
        or magnitude_b < 1e-6
    ):
        return None

    cosine = (
        ax * bx
        + ay * by
    ) / (
        magnitude_a
        * magnitude_b
    )

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# TRUNK ANGLE
# ============================================================

def calculate_trunk_angle(landmarks):
    """
    Estimate trunk flexion relative to the lower-body axis.

    Preferred reference:
        hip midpoint -> knee midpoint

    Trunk axis:
        hip midpoint -> shoulder midpoint

    The two vectors point in opposite directions for an
    upright person. Therefore the measured angle is converted
    into the smaller deviation from the lower-body vertical axis.

    If knees are unavailable, image vertical is used as a
    fallback.

    Returns:
        Trunk angle in degrees.
    """

    left_shoulder = landmarks.get(
        "left_shoulder"
    )

    right_shoulder = landmarks.get(
        "right_shoulder"
    )

    left_hip = landmarks.get(
        "left_hip"
    )

    right_hip = landmarks.get(
        "right_hip"
    )

    left_knee = landmarks.get(
        "left_knee"
    )

    right_knee = landmarks.get(
        "right_knee"
    )

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )

    if (
        shoulder_mid is None
        or hip_mid is None
    ):
        return None

    # ---------------------------------------------------------
    # Determine available knee reference.
    # ---------------------------------------------------------

    knee_points = []

    if left_knee is not None:
        knee_points.append(
            left_knee
        )

    if right_knee is not None:
        knee_points.append(
            right_knee
        )

    knee_mid = None

    if len(knee_points) == 1:

        knee_mid = knee_points[0]

    elif len(knee_points) == 2:

        knee_mid = midpoint(
            knee_points[0],
            knee_points[1]
        )

    # ---------------------------------------------------------
    # Trunk vector:
    #
    # hip -> shoulder
    # ---------------------------------------------------------

    trunk_x = (
        shoulder_mid[0]
        - hip_mid[0]
    )

    trunk_y = (
        shoulder_mid[1]
        - hip_mid[1]
    )

    trunk_length = math.hypot(
        trunk_x,
        trunk_y
    )

    if trunk_length < 1e-6:
        return None

    # ---------------------------------------------------------
    # Reference vector.
    #
    # Prefer hip -> knee because this follows the person's
    # actual lower-body orientation in the camera view.
    #
    # If knee is unavailable, use image-downward vertical.
    # ---------------------------------------------------------

    if knee_mid is not None:

        reference_x = (
            knee_mid[0]
            - hip_mid[0]
        )

        reference_y = (
            knee_mid[1]
            - hip_mid[1]
        )

    else:

        reference_x = 0.0
        reference_y = 1.0

    reference_length = math.hypot(
        reference_x,
        reference_y
    )

    if reference_length < 1e-6:
        return None

    # ---------------------------------------------------------
    # Angle between trunk and lower-body reference.
    # ---------------------------------------------------------

    angle = _angle_between_vectors(
        (
            trunk_x,
            trunk_y
        ),
        (
            reference_x,
            reference_y
        )
    )

    if angle is None:
        return None

    # ---------------------------------------------------------
    # Upright posture produces approximately 180 degrees
    # because:
    #
    # hip -> shoulder = upward
    # hip -> knee     = downward
    #
    # Convert to deviation from upright.
    # ---------------------------------------------------------

    if angle > 90.0:
        angle = 180.0 - angle

    return round(
        angle,
        2
    )


# ============================================================
# NECK ANGLE
# ============================================================

def calculate_neck_angle(landmarks):
    """
    Estimate neck flexion/extension relative to the trunk.

    The neck angle is estimated from the difference between:

        1. Trunk direction:
           hip midpoint -> shoulder midpoint

        2. Head direction:
           shoulder midpoint -> nose

    This is a 2D ergonomic screening proxy because the available
    pose landmarks do not provide a direct anatomical neck joint.

    Returns:
        Estimated neck angle in degrees.
        None if required landmarks are unavailable.
    """

    nose = landmarks.get("nose")

    left_shoulder = landmarks.get(
        "left_shoulder"
    )

    right_shoulder = landmarks.get(
        "right_shoulder"
    )

    left_hip = landmarks.get(
        "left_hip"
    )

    right_hip = landmarks.get(
        "right_hip"
    )

    # ---------------------------------------------------------
    # Calculate body midpoints
    # ---------------------------------------------------------

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )

    if (
        nose is None
        or shoulder_mid is None
        or hip_mid is None
    ):
        return None

    # ---------------------------------------------------------
    # Head direction
    #
    # shoulder -> nose
    # ---------------------------------------------------------

    try:

        head_x = (
            float(nose[0])
            - shoulder_mid[0]
        )

        head_y = (
            float(nose[1])
            - shoulder_mid[1]
        )

    except (
        TypeError,
        ValueError,
        IndexError
    ):

        return None

    head_length = math.hypot(
        head_x,
        head_y
    )

    if head_length < 1e-6:
        return None

    # ---------------------------------------------------------
    # Trunk direction
    #
    # hip -> shoulder
    # ---------------------------------------------------------

    trunk_x = (
        shoulder_mid[0]
        - hip_mid[0]
    )

    trunk_y = (
        shoulder_mid[1]
        - hip_mid[1]
    )

    trunk_length = math.hypot(
        trunk_x,
        trunk_y
    )

    if trunk_length < 1e-6:
        return None

    # ---------------------------------------------------------
    # Normalize both vectors
    # ---------------------------------------------------------

    head_x /= head_length
    head_y /= head_length

    trunk_x /= trunk_length
    trunk_y /= trunk_length

    # ---------------------------------------------------------
    # Calculate angle between head and trunk
    # ---------------------------------------------------------

    dot_product = (
        head_x * trunk_x
        + head_y * trunk_y
    )

    dot_product = max(
        -1.0,
        min(1.0, dot_product)
    )

    raw_angle = math.degrees(
        math.acos(dot_product)
    )

    # ---------------------------------------------------------
    # Depending on camera orientation, the two vectors can
    # point in opposite directions.
    #
    # Always use the smaller angular deviation.
    # ---------------------------------------------------------

    if raw_angle > 90.0:
        raw_angle = 180.0 - raw_angle

    # ---------------------------------------------------------
    # DO NOT cap at 45 degrees.
    #
    # The previous:
    #
    #     min(raw_angle, 45)
    #
    # caused repeated 45-degree values.
    #
    # We allow the actual estimated angle to vary.
    # ---------------------------------------------------------

    neck_angle = min(
        raw_angle,
        60.0
    )

    return round(
        neck_angle,
        2
    )


# ============================================================
# UPPER ARM
# ============================================================

def calculate_upper_arm_angle(
    shoulder,
    elbow,
    hip
):
    """
    Calculate upper-arm angle relative to the trunk.

    Approximate interpretation:

        0 degrees   = arm close to trunk
        90 degrees  = arm approximately horizontal
        180 degrees = arm pointing opposite the trunk direction

    Returns:
        Angle in degrees, or None if a point is missing.
    """

    return calculate_angle(
        elbow,
        shoulder,
        hip
    )


# ============================================================
# ELBOW
# ============================================================

def calculate_elbow_angle(
    shoulder,
    elbow,
    wrist
):
    """
    Calculate elbow flexion angle.

    180 degrees is approximately straight.
    Smaller values indicate greater elbow flexion.
    """

    return calculate_angle(
        shoulder,
        elbow,
        wrist
    )


# ============================================================
# KNEE
# ============================================================

def calculate_knee_angle(
    hip,
    knee,
    ankle
):
    """
    Calculate knee angle.

    180 degrees is approximately straight.
    Smaller values indicate greater knee flexion.

    Returns:
        None when required landmarks are missing.
    """

    return calculate_angle(
        hip,
        knee,
        ankle
    )


# ============================================================
# FRAME ANGLES
# ============================================================

def calculate_frame_angles(landmarks):
    """
    Calculate the main posture angles for one frame.

    Missing landmarks are handled safely.

    Returns:

        trunk
        neck

        left_upper_arm
        right_upper_arm

        left_elbow
        right_elbow

        left_knee
        right_knee
    """

    if landmarks is None:
        landmarks = {}

    return {

        # -----------------------------------------------------
        # Trunk
        # -----------------------------------------------------

        "trunk": calculate_trunk_angle(
            landmarks
        ),

        # -----------------------------------------------------
        # Neck
        # -----------------------------------------------------

        "neck": calculate_neck_angle(
            landmarks
        ),

        # -----------------------------------------------------
        # Left upper arm
        # -----------------------------------------------------

        "left_upper_arm": calculate_upper_arm_angle(
            landmarks.get(
                "left_shoulder"
            ),
            landmarks.get(
                "left_elbow"
            ),
            landmarks.get(
                "left_hip"
            )
        ),

        # -----------------------------------------------------
        # Right upper arm
        # -----------------------------------------------------

        "right_upper_arm": calculate_upper_arm_angle(
            landmarks.get(
                "right_shoulder"
            ),
            landmarks.get(
                "right_elbow"
            ),
            landmarks.get(
                "right_hip"
            )
        ),

        # -----------------------------------------------------
        # Left elbow
        # -----------------------------------------------------

        "left_elbow": calculate_elbow_angle(
            landmarks.get(
                "left_shoulder"
            ),
            landmarks.get(
                "left_elbow"
            ),
            landmarks.get(
                "left_wrist"
            )
        ),

        # -----------------------------------------------------
        # Right elbow
        # -----------------------------------------------------

        "right_elbow": calculate_elbow_angle(
            landmarks.get(
                "right_shoulder"
            ),
            landmarks.get(
                "right_elbow"
            ),
            landmarks.get(
                "right_wrist"
            )
        ),

        # -----------------------------------------------------
        # Left knee
        # -----------------------------------------------------

        "left_knee": calculate_knee_angle(
            landmarks.get(
                "left_hip"
            ),
            landmarks.get(
                "left_knee"
            ),
            landmarks.get(
                "left_ankle"
            )
        ),

        # -----------------------------------------------------
        # Right knee
        # -----------------------------------------------------

        "right_knee": calculate_knee_angle(
            landmarks.get(
                "right_hip"
            ),
            landmarks.get(
                "right_knee"
            ),
            landmarks.get(
                "right_ankle"
            )
        )
    }