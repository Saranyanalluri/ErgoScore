"""
ErgoScore routes: the Person 4 (Flask) application layer.

Keeps all HTTP handling here; all analysis logic lives behind
pipeline.analyze_video() so this file never needs to change when the real
analysis modules are integrated.
"""

import os

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from .pipeline import analyze_video

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}
LOAD_OPTIONS = ["none", "light", "medium", "heavy"]
COUPLING_OPTIONS = ["good", "fair", "poor"]


def _allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@main_bp.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        load_options=LOAD_OPTIONS,
        coupling_options=COUPLING_OPTIONS,
    )


@main_bp.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")

    if video is None or video.filename == "":
        flash("Please choose a video file to upload.", "error")
        return redirect(url_for("main.index"))

    if not _allowed_file(video.filename):
        flash(
            "Unsupported file type. Please upload one of: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "error",
        )
        return redirect(url_for("main.index"))

    load = request.form.get("load", "medium")
    coupling = request.form.get("coupling", "fair")
    if load not in LOAD_OPTIONS:
        load = "medium"
    if coupling not in COUPLING_OPTIONS:
        coupling = "fair"

    filename = secure_filename(video.filename)
    if not filename:
        flash("That filename isn't valid. Please rename the file and try again.", "error")
        return redirect(url_for("main.index"))

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    save_path = os.path.join(upload_folder, filename)

    try:
        video.save(save_path)
    except Exception:
        current_app.logger.exception("Failed to save uploaded video")
        flash("Something went wrong saving your video. Please try again.", "error")
        return redirect(url_for("main.index"))

    try:
        result = analyze_video(save_path, load=load, coupling=coupling)
    except Exception:
        current_app.logger.exception("Pipeline analysis failed")
        flash("Analysis failed. Please try a different video.", "error")
        return redirect(url_for("main.index"))

    return render_template(
        "results.html",
        result=result,
        video_filename=filename,
        load=load,
        coupling=coupling,
    )


@main_bp.route("/videos/uploads/<path:filename>")
def uploaded_video(filename):
    """Serves an uploaded video back to the browser for playback."""
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    safe_name = secure_filename(filename)
    if safe_name != filename:
        abort(404)
    file_path = os.path.join(upload_folder, safe_name)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(upload_folder, safe_name)
