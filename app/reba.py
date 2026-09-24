"""
REBA (Rapid Entire Body Assessment) scoring.

Based on the REBA method developed by Hignett and McAtamney (2000).

This module:
    - converts posture angles into REBA component scores
    - calculates Group A and Group B
    - applies load and coupling adjustments
    - calculates Table C
    - applies activity adjustments
    - returns the final REBA score and action level
"""

from typing import Dict, Optional


# ============================================================
# REBA TABLE A
# ============================================================

# Rows = trunk score 1-5
#
# Columns:
#   neck 1 + legs 1-4
#   neck 2 + legs 1-4
#   neck 3 + legs 1-4

TABLE_A = {
    1: [
        1, 2, 3, 4,
        1, 2, 3, 4,
        3, 3, 5, 6
    ],
    2: [
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
    ],
    3: [
        2, 4, 5, 6,
        4, 5, 6, 7,
        5, 6, 7, 8
    ],
    4: [
        3, 5, 6, 7,
        5, 6, 7, 8,
        6, 7, 8, 9
    ],
    5: [
        4, 6, 7, 8,
        6, 7, 8, 9,
        7, 8, 9, 9
    ],
}


# ============================================================
# REBA TABLE B
# ============================================================

# Rows = upper-arm score 1-6
#
# lower arm 1:
#     wrist 1, wrist 2, wrist 3
#
# lower arm 2:
#     wrist 1, wrist 2, wrist 3

TABLE_B = {
    1: {
        1: {1: 1, 2: 2, 3: 3},
        2: {1: 1, 2: 2, 3: 3},
    },
    2: {
        1: {1: 1, 2: 2, 3: 3},
        2: {1: 2, 2: 3, 3: 4},
    },
    3: {
        1: {1: 3, 2: 4, 3: 5},
        2: {1: 4, 2: 5, 3: 5},
    },
    4: {
        1: {1: 4, 2: 5, 3: 5},
        2: {1: 5, 2: 6, 3: 7},
    },
    5: {
        1: {1: 6, 2: 7, 3: 8},
        2: {1: 7, 2: 8, 3: 8},
    },
    6: {
        1: {1: 7, 2: 8, 3: 8},
        2: {1: 8, 2: 9, 3: 9},
    },
}


# ============================================================
# REBA TABLE C
# ============================================================

# Rows = Score A
# Columns = Score B
#
# Both Score A and Score B range from 1 to 12.

TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


# ============================================================
# VALIDATION
# ============================================================

def _validate_score(
    score: int,
    minimum: int,
    maximum: int,
    name: str
):
    """Validate an integer REBA score."""

    if not isinstance(score, int):
        raise TypeError(f"{name} must be an integer.")

    if not minimum <= score <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )


# ============================================================
# TRUNK
# ============================================================

def score_trunk(
    trunk_angle: float,
    twisting: bool = False,
    side_bending: bool = False
) -> int:
    """
    Calculate REBA trunk score.

    The angle is the magnitude of trunk deviation from upright.

    Base score:
        0 degrees              -> 1
        >0 to 20 degrees       -> 2
        >20 to 60 degrees      -> 3
        >60 degrees            -> 4

    Additional:
        +1 twisting
        +1 side bending

    Maximum trunk score = 5.
    """

    if trunk_angle is None:
        raise ValueError("trunk_angle is required.")

    if trunk_angle < 0:
        raise ValueError("trunk_angle cannot be negative.")

    if trunk_angle == 0:
        score = 1
    elif trunk_angle <= 20:
        score = 2
    elif trunk_angle <= 60:
        score = 3
    else:
        score = 4

    if twisting or side_bending:
        score += 1

    return min(score, 5)


# ============================================================
# NECK
# ============================================================

def score_neck(
    neck_angle: float,
    twisting: bool = False,
    side_bending: bool = False
) -> int:
    """
    Calculate REBA neck score.

    Base score:
        0-20 degrees -> 1
        >20 degrees or extension -> 2

    Additional:
        +1 twisting
        +1 side bending

    Maximum neck score = 3.
    """

    if neck_angle is None:
        raise ValueError("neck_angle is required.")

    if neck_angle < 0:
        raise ValueError("neck_angle cannot be negative.")

    if neck_angle <= 20:
        score = 1
    else:
        score = 2

    if twisting or side_bending:
        score += 1

    return min(score, 3)


# ============================================================
# LEGS
# ============================================================

def score_legs(
    knee_angle: Optional[float],
    bilateral_weight_bearing: bool = True,
    sitting: bool = False
) -> int:
    """
    Calculate REBA leg score.

    Base:
        1 = sitting, walking, or symmetrical bilateral support
        2 = unilateral support, light support, or unstable posture

    Knee adjustment:
        30-60 degrees flexion -> +1
        >60 degrees flexion   -> +2

    Important:
        For a seated worker, missing knee/ankle data is allowed.
        This is useful when the lower legs are outside the camera view.
    """

    # For our seated-worker case, we can score the base leg posture
    # even if the knee is not visible.
    if sitting and knee_angle is None:
        return 1

    if knee_angle is None:
        raise ValueError("knee_angle is required unless sitting=True.")

    if knee_angle < 0 or knee_angle > 180:
        raise ValueError(
            "knee_angle must be between 0 and 180."
        )

    if sitting:
        score = 1
    else:
        score = 1 if bilateral_weight_bearing else 2

        # Convert internal knee angle to flexion.
        knee_flexion = 180 - knee_angle

        if knee_flexion > 60:
            score += 2
        elif knee_flexion >= 30:
            score += 1

    return min(score, 4)


# ============================================================
# UPPER ARM
# ============================================================

def score_upper_arm(
    upper_arm_angle: float,
    abducted_or_rotated: bool = False,
    shoulder_raised: bool = False,
    supported: bool = False
) -> int:
    """
    Calculate REBA upper-arm score.

    Base:
        0-20 degrees             -> 1
        >20-45 degrees           -> 2
        >45-90 degrees           -> 3
        >90 degrees              -> 4

    Additional:
        +1 abducted/rotated
        +1 shoulder raised
        -1 supported / gravity-assisted

    Maximum = 6.
    """

    if upper_arm_angle is None:
        raise ValueError("upper_arm_angle is required.")

    if upper_arm_angle < 0:
        raise ValueError(
            "upper_arm_angle cannot be negative."
        )

    if upper_arm_angle <= 20:
        score = 1
    elif upper_arm_angle <= 45:
        score = 2
    elif upper_arm_angle <= 90:
        score = 3
    else:
        score = 4

    if abducted_or_rotated:
        score += 1

    if shoulder_raised:
        score += 1

    if supported:
        score -= 1

    return max(1, min(score, 6))


# ============================================================
# LOWER ARM / FOREARM
# ============================================================

def score_lower_arm(elbow_angle: float) -> int:
    """
    Calculate REBA forearm/lower-arm score.

    60-100 degrees -> 1
    <60 or >100    -> 2
    """

    if elbow_angle is None:
        raise ValueError("elbow_angle is required.")

    if elbow_angle < 0 or elbow_angle > 180:
        raise ValueError(
            "elbow_angle must be between 0 and 180."
        )

    if 60 <= elbow_angle <= 100:
        return 1

    return 2


# ============================================================
# WRIST
# ============================================================

def score_wrist(
    wrist_angle: float,
    deviated_or_twisted: bool = False
) -> int:
    """
    Calculate REBA wrist score.

    Base:
        0-15 degrees -> 1
        >15 degrees  -> 2

    Additional:
        +1 deviation/torsion

    Maximum = 3.
    """

    if wrist_angle is None:
        raise ValueError("wrist_angle is required.")

    if wrist_angle < 0:
        raise ValueError(
            "wrist_angle cannot be negative."
        )

    score = 1 if wrist_angle <= 15 else 2

    if deviated_or_twisted:
        score += 1

    return min(score, 3)


# ============================================================
# LOAD / FORCE
# ============================================================

def score_load(
    load_kg: float = 0.0,
    shock_or_rapid_force: bool = False
) -> int:
    """
    Calculate REBA load/force adjustment.

    <5 kg       -> 0
    5-10 kg     -> +1
    >10 kg      -> +2
    shock       -> +1 additional
    """

    if load_kg is None:
        load_kg = 0.0

    if load_kg < 0:
        raise ValueError("load_kg cannot be negative.")

    if load_kg < 5:
        score = 0
    elif load_kg <= 10:
        score = 1
    else:
        score = 2

    if shock_or_rapid_force:
        score += 1

    return min(score, 3)


# ============================================================
# COUPLING
# ============================================================

def score_coupling(coupling: int = 0) -> int:
    """
    REBA coupling score.

    0 = Good
    1 = Fair
    2 = Poor
    3 = Unacceptable
    """

    _validate_score(
        coupling,
        0,
        3,
        "coupling"
    )

    return coupling


# ============================================================
# TABLE A
# ============================================================

def calculate_table_a(
    trunk_score: int,
    neck_score: int,
    leg_score: int
) -> int:
    """
    Look up REBA Table A.
    """

    _validate_score(
        trunk_score,
        1,
        5,
        "trunk_score"
    )

    _validate_score(
        neck_score,
        1,
        3,
        "neck_score"
    )

    _validate_score(
        leg_score,
        1,
        4,
        "leg_score"
    )

    index = (
        (neck_score - 1) * 4
        + (leg_score - 1)
    )

    return TABLE_A[trunk_score][index]


# ============================================================
# TABLE B
# ============================================================

def calculate_table_b(
    upper_arm_score: int,
    lower_arm_score: int,
    wrist_score: int
) -> int:
    """
    Look up REBA Table B.
    """

    _validate_score(
        upper_arm_score,
        1,
        6,
        "upper_arm_score"
    )

    _validate_score(
        lower_arm_score,
        1,
        2,
        "lower_arm_score"
    )

    _validate_score(
        wrist_score,
        1,
        3,
        "wrist_score"
    )

    return TABLE_B[
        upper_arm_score
    ][
        lower_arm_score
    ][
        wrist_score
    ]


# ============================================================
# SCORE A
# ============================================================

def calculate_score_a(
    table_a_score: int,
    load_score: int
) -> int:
    """
    Score A = Table A + load/force adjustment.
    """

    _validate_score(
        table_a_score,
        1,
        12,
        "table_a_score"
    )

    _validate_score(
        load_score,
        0,
        3,
        "load_score"
    )

    return min(
        table_a_score + load_score,
        12
    )


# ============================================================
# SCORE B
# ============================================================

def calculate_score_b(
    table_b_score: int,
    coupling_score: int
) -> int:
    """
    Score B = Table B + coupling adjustment.
    """

    _validate_score(
        table_b_score,
        1,
        12,
        "table_b_score"
    )

    _validate_score(
        coupling_score,
        0,
        3,
        "coupling_score"
    )

    return min(
        table_b_score + coupling_score,
        12
    )


# ============================================================
# TABLE C
# ============================================================

def calculate_score_c(
    score_a: int,
    score_b: int
) -> int:
    """
    Look up REBA Table C.

    Rows    = Score A
    Columns = Score B
    """

    _validate_score(
        score_a,
        1,
        12,
        "score_a"
    )

    _validate_score(
        score_b,
        1,
        12,
        "score_b"
    )

    return TABLE_C[
        score_a - 1
    ][
        score_b - 1
    ]


# ============================================================
# ACTIVITY SCORE
# ============================================================

def calculate_activity_score(
    static_posture: bool = False,
    repetitive_movements: bool = False,
    rapid_or_unstable_posture: bool = False
) -> int:
    """
    Calculate REBA activity adjustment.

    +1 static posture
    +1 repetitive movement
    +1 significant posture change / unstable posture

    Maximum = 3.
    """

    score = 0

    if static_posture:
        score += 1

    if repetitive_movements:
        score += 1

    if rapid_or_unstable_posture:
        score += 1

    return score


# ============================================================
# FINAL REBA SCORE
# ============================================================

def calculate_final_reba_score(
    score_c: int,
    activity_score: int = 0
) -> int:
    """
    Final REBA score = Score C + Activity score.

    Maximum = 15.
    """

    _validate_score(
        score_c,
        1,
        12,
        "score_c"
    )

    _validate_score(
        activity_score,
        0,
        3,
        "activity_score"
    )

    return min(
        score_c + activity_score,
        15
    )


# ============================================================
# RISK / ACTION LEVEL
# ============================================================

def get_risk_level(
    reba_score: int
) -> Dict[str, object]:
    """
    Convert the final REBA score into the standard
    REBA action-level information.
    """

    if not 1 <= reba_score <= 15:
        raise ValueError(
            "REBA score must be between 1 and 15."
        )

    if reba_score == 1:
        return {
            "risk_level": "Negligible",
            "action_level": 0,
            "action": "None necessary"
        }

    if 2 <= reba_score <= 3:
        return {
            "risk_level": "Low",
            "action_level": 1,
            "action": "May be necessary"
        }

    if 4 <= reba_score <= 7:
        return {
            "risk_level": "Medium",
            "action_level": 2,
            "action": "Necessary"
        }

    if 8 <= reba_score <= 10:
        return {
            "risk_level": "High",
            "action_level": 3,
            "action": "Necessary soon"
        }

    return {
        "risk_level": "Very high",
        "action_level": 4,
        "action": "Necessary NOW"
    }


# ============================================================
# COMPLETE REBA CALCULATION
# ============================================================

def calculate_reba(
    trunk_angle: float,
    neck_angle: float,
    knee_angle: Optional[float],
    upper_arm_angle: float,
    elbow_angle: float,
    wrist_angle: float,
    load_kg: float = 0.0,
    coupling: int = 0,
    bilateral_weight_bearing: bool = True,
    sitting: bool = False,
    trunk_twisting: bool = False,
    trunk_side_bending: bool = False,
    neck_twisting: bool = False,
    neck_side_bending: bool = False,
    upper_arm_abducted: bool = False,
    shoulder_raised: bool = False,
    arm_supported: bool = False,
    wrist_deviated: bool = False,
    shock_or_rapid_force: bool = False,
    static_posture: bool = False,
    repetitive_movements: bool = False,
    rapid_or_unstable_posture: bool = False
) -> Dict[str, object]:
    """
    Calculate a complete REBA assessment.

    Returns:
        Group A
        Group B
        Score C
        Activity score
        Final REBA score
        Risk/action information
    """

    # --------------------------------------------------------
    # GROUP A
    # --------------------------------------------------------

    trunk = score_trunk(
        trunk_angle,
        twisting=trunk_twisting,
        side_bending=trunk_side_bending
    )

    neck = score_neck(
        neck_angle,
        twisting=neck_twisting,
        side_bending=neck_side_bending
    )

    legs = score_legs(
        knee_angle,
        bilateral_weight_bearing=bilateral_weight_bearing,
        sitting=sitting
    )

    load = score_load(
        load_kg,
        shock_or_rapid_force=shock_or_rapid_force
    )

    table_a = calculate_table_a(
        trunk_score=trunk,
        neck_score=neck,
        leg_score=legs
    )

    score_a = calculate_score_a(
        table_a_score=table_a,
        load_score=load
    )

    # --------------------------------------------------------
    # GROUP B
    # --------------------------------------------------------

    upper_arm = score_upper_arm(
        upper_arm_angle,
        abducted_or_rotated=upper_arm_abducted,
        shoulder_raised=shoulder_raised,
        supported=arm_supported
    )

    lower_arm = score_lower_arm(
        elbow_angle
    )

    wrist = score_wrist(
        wrist_angle,
        deviated_or_twisted=wrist_deviated
    )

    coupling_score = score_coupling(
        coupling
    )

    table_b = calculate_table_b(
        upper_arm_score=upper_arm,
        lower_arm_score=lower_arm,
        wrist_score=wrist
    )

    score_b = calculate_score_b(
        table_b_score=table_b,
        coupling_score=coupling_score
    )

    # --------------------------------------------------------
    # TABLE C
    # --------------------------------------------------------

    score_c = calculate_score_c(
        score_a=score_a,
        score_b=score_b
    )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    activity = calculate_activity_score(
        static_posture=static_posture,
        repetitive_movements=repetitive_movements,
        rapid_or_unstable_posture=rapid_or_unstable_posture
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = calculate_final_reba_score(
        score_c=score_c,
        activity_score=activity
    )

    risk = get_risk_level(
        final_score
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "group_a": {
            "trunk": trunk,
            "neck": neck,
            "legs": legs,
            "load": load,
            "table_a": table_a,
            "score_a": score_a
        },

        "group_b": {
            "upper_arm": upper_arm,
            "lower_arm": lower_arm,
            "wrist": wrist,
            "coupling": coupling_score,
            "table_b": table_b,
            "score_b": score_b
        },

        "score_c": score_c,

        "activity_score": activity,

        "reba_score": final_score,

        "risk": risk
    }