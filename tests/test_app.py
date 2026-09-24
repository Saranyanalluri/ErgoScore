"""
Basic tests for the Person 4 (Flask) layer.

These check the routes and the pipeline's dummy-data contract — not real
posture analysis, since that doesn't exist yet. Run with:

    pytest
"""

import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app import create_app


@pytest.fixture
def app(tmp_path):
    test_app = create_app(test_config={
        "TESTING": True,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "PROCESSED_FOLDER": str(tmp_path / "processed"),
    })
    return test_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"ErgoScore" in response.data
    assert b"Analyze Video" in response.data


def test_analyze_with_no_file_redirects_with_flash(client):
    response = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert response.status_code == 302

    follow = client.get("/")
    assert b"choose a video file" in follow.data


def test_analyze_with_invalid_extension_is_rejected(client):
    data = {
        "video": (io.BytesIO(b"not a real video"), "notes.txt"),
        "load": "medium",
        "coupling": "fair",
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 302

    follow = client.get("/")
    assert b"Unsupported file type" in follow.data


def test_analyze_with_valid_video_renders_dashboard(client):
    data = {
        "video": (io.BytesIO(b"fake video bytes"), "test_clip.mp4"),
        "load": "heavy",
        "coupling": "poor",
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Overall Risk" in response.data
    assert b"Score Timeline" in response.data
    assert b"Worst Posture Frames" in response.data


def test_analyze_saves_file_to_upload_folder(app, client):
    data = {
        "video": (io.BytesIO(b"fake video bytes"), "saved_clip.mp4"),
        "load": "medium",
        "coupling": "fair",
    }
    client.post("/analyze", data=data, content_type="multipart/form-data")

    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], "saved_clip.mp4")
    assert os.path.isfile(saved_path)


def test_pipeline_returns_expected_contract(tmp_path):
    from app.pipeline import analyze_video

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")

    result = analyze_video(str(video_path), load="medium", coupling="fair")

    expected_keys = {
        "overall_score", "risk_level", "average_score", "maximum_score",
        "duration", "risk_distribution", "timeline", "worst_frames",
        "summary", "is_dummy_data",
    }
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["timeline"], list)
    assert all({"time", "score"}.issubset(p) for p in result["timeline"])
    assert isinstance(result["worst_frames"], list)
    assert len(result["worst_frames"]) <= 5
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["is_dummy_data"] is True


def test_pipeline_reacts_to_load_and_coupling(tmp_path):
    from app.pipeline import analyze_video

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")

    light_result = analyze_video(str(video_path), load="none", coupling="good")
    heavy_result = analyze_video(str(video_path), load="heavy", coupling="poor")

    assert heavy_result["average_score"] >= light_result["average_score"]
