import math


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point B formed by A-B-C.

    Points can be:
        [x, y]
    or:
        (x, y)

    Returns:
        Angle in degrees from 0 to 180.
        Returns None if a point is missing or a vector has zero length.
    """

    if point_a is None or point_b is None or point_c is None:
        return None

    try:
        ax, ay = float(point_a[0]), float(point_a[1])
        bx, by = float(point_b[0]), float(point_b[1])
        cx, cy = float(point_c[0]), float(point_c[1])
    except (TypeError, ValueError, IndexError):
        return None

    ba_x = ax - bx
    ba_y = ay - by

    bc_x = cx - bx
    bc_y = cy - by

    magnitude_ba = math.hypot(ba_x, ba_y)
    magnitude_bc = math.hypot(bc_x, bc_y)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    dot_product = ba_x * bc_x + ba_y * bc_y

    cosine = dot_product / (magnitude_ba * magnitude_bc)

    # Protect against floating-point errors.
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def midpoint(point_a, point_b):
    """
    Calculate the midpoint between two points.
    """

    if point_a is None or point_b is None:
        return None

    try:
        return [
            (float(point_a[0]) + float(point_b[0])) / 2.0,
            (float(point_a[1]) + float(point_b[1])) / 2.0
        ]
    except (TypeError, ValueError, IndexError):
        return None


def calculate_trunk_angle(landmarks):
    """
    Estimate trunk flexion/extension magnitude relative to vertical.

    Uses the midpoint between the shoulders and the midpoint
    between the hips.

    A straight/upright trunk is approximately 0 degrees.

    Returns:
        0-180 degrees, or None if required landmarks are missing.
    """

    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    if shoulder_mid is None or hip_mid is None:
        return None

    # Point directly above the hip midpoint.
    vertical_reference = [
        hip_mid[0],
        hip_mid[1] - 1.0
    ]

    return calculate_angle(
        vertical_reference,
        hip_mid,
        shoulder_mid
    )


def calculate_neck_angle(landmarks):
    """
    Estimate neck/head angle relative to the trunk.

    The line from the shoulder midpoint to the nose is used
    as an approximation of the head/neck direction.

    The line from the hip midpoint to the shoulder midpoint
    represents the trunk direction.

    An approximately upright head/neck is close to 0 degrees.

    Returns:
        Angle in degrees, or None if required landmarks are missing.
    """

    nose = landmarks.get("nose")

    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    if nose is None or shoulder_mid is None or hip_mid is None:
        return None

    # Vector from shoulder toward the nose.
    head_x = float(nose[0]) - shoulder_mid[0]
    head_y = float(nose[1]) - shoulder_mid[1]

    # Vector from hip toward shoulder.
    trunk_x = shoulder_mid[0] - hip_mid[0]
    trunk_y = shoulder_mid[1] - hip_mid[1]

    head_length = math.hypot(head_x, head_y)
    trunk_length = math.hypot(trunk_x, trunk_y)

    if head_length == 0 or trunk_length == 0:
        return None

    dot_product = head_x * trunk_x + head_y * trunk_y

    cosine = dot_product / (head_length * trunk_length)

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_upper_arm_angle(shoulder, elbow, hip):
    """
    Calculate the upper-arm angle relative to the trunk.

    The angle is measured at the shoulder between:

        shoulder -> elbow
        shoulder -> hip

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


def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Calculate the elbow flexion angle.

    180 degrees is approximately straight.
    Smaller values indicate greater elbow flexion.
    """

    return calculate_angle(
        shoulder,
        elbow,
        wrist
    )


def calculate_knee_angle(hip, knee, ankle):
    """
    Calculate the knee angle.

    180 degrees is approximately straight.
    Smaller values indicate greater knee flexion.

    Returns None when the ankle is not visible.
    """

    return calculate_angle(
        hip,
        knee,
        ankle
    )


def calculate_frame_angles(landmarks):
    """
    Calculate the main posture angles for one frame.

    Missing landmarks are handled safely.

    Returns a dictionary containing:

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
        "trunk": calculate_trunk_angle(landmarks),

        "neck": calculate_neck_angle(landmarks),

        "left_upper_arm": calculate_upper_arm_angle(
            landmarks.get("left_shoulder"),
            landmarks.get("left_elbow"),
            landmarks.get("left_hip")
        ),

        "right_upper_arm": calculate_upper_arm_angle(
            landmarks.get("right_shoulder"),
            landmarks.get("right_elbow"),
            landmarks.get("right_hip")
        ),

        "left_elbow": calculate_elbow_angle(
            landmarks.get("left_shoulder"),
            landmarks.get("left_elbow"),
            landmarks.get("left_wrist")
        ),

        "right_elbow": calculate_elbow_angle(
            landmarks.get("right_shoulder"),
            landmarks.get("right_elbow"),
            landmarks.get("right_wrist")
        ),

        "left_knee": calculate_knee_angle(
            landmarks.get("left_hip"),
            landmarks.get("left_knee"),
            landmarks.get("left_ankle")
        ),

        "right_knee": calculate_knee_angle(
            landmarks.get("right_hip"),
            landmarks.get("right_knee"),
            landmarks.get("right_ankle")
        )
    }