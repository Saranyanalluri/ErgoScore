"""
Adapter for Person 1's keypoint JSON.

Converts Person 1's JSON format into the normalized landmark
format expected by Person 2.
"""


REQUIRED_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def landmark_to_point(landmark):
    """
    Convert one Person 1 landmark object into [x, y].

    Example input:

        {
            "x": 0.45,
            "y": 0.32,
            "z": 0.01,
            "visibility": 0.98,
            "presence": 0.99
        }

    Returns:

        [0.45, 0.32]

    Returns None if the landmark is missing.
    """

    if landmark is None:
        return None

    if not isinstance(landmark, dict):
        return None

    x = landmark.get("x")
    y = landmark.get("y")

    if x is None or y is None:
        return None

    try:
        return [
            float(x),
            float(y)
        ]
    except (TypeError, ValueError):
        return None


def convert_landmarks(person1_landmarks):
    """
    Convert Person 1's landmark dictionary into Person 2 format.

    Missing landmarks remain None.
    """

    if person1_landmarks is None:
        return {}

    converted = {}

    for name in REQUIRED_LANDMARKS:
        converted[name] = landmark_to_point(
            person1_landmarks.get(name)
        )

    return converted


def convert_frame(person1_frame):
    """
    Convert one Person 1 frame.

    Person 1 format:

        {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "person_detected": true,
            "pose_quality": 0.94,
            "landmarks": {...}
        }

    Person 2 format:

        {
            "frame": 0,
            "timestamp": 0.0,
            "person_detected": true,
            "pose_quality": 0.94,
            "landmarks": {...}
        }
    """

    if person1_frame is None:
        raise ValueError("person1_frame cannot be None.")

    return {
        "frame": person1_frame.get("frame_index"),
        "timestamp": person1_frame.get(
            "timestamp_seconds"
        ),
        "person_detected": person1_frame.get(
            "person_detected",
            False
        ),
        "pose_quality": person1_frame.get(
            "pose_quality"
        ),
        "landmarks": convert_landmarks(
            person1_frame.get("landmarks", {})
        )
    }


def convert_keypoints(person1_data):
    """
    Convert the complete Person 1 JSON object.

    Person 1 input:

        {
            "schema_version": "1.0",
            "video_metadata": {...},
            "frames": [...]
        }

    Returns:

        {
            "video_metadata": {...},
            "frames": [...]
        }
    """

    if person1_data is None:
        raise ValueError("person1_data cannot be None.")

    if not isinstance(person1_data, dict):
        raise TypeError(
            "person1_data must be a dictionary."
        )

    frames = person1_data.get("frames")

    if frames is None:
        raise ValueError(
            "Person 1 JSON does not contain 'frames'."
        )

    converted_frames = []

    for frame in frames:
        converted_frames.append(
            convert_frame(frame)
        )

    return {
        "video_metadata": person1_data.get(
            "video_metadata",
            {}
        ),
        "frames": converted_frames
    }


def load_person1_json(file_path):
    """
    Load and convert a Person 1 JSON file.
    """

    import json

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        person1_data = json.load(file)

    return convert_keypoints(person1_data)