"""
ErgoScore Flask routes.

Handles:
- Uploading videos
- Running the complete analysis pipeline
- Displaying results
- Serving uploaded/processed videos
- Serving generated reports and charts
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

ALLOWED_EXTENSIONS = {
    "mp4",
    "mov",
    "avi",
    "webm",
    "mkv",
}

LOAD_OPTIONS = [
    "none",
    "light",
    "medium",
    "heavy",
]

COUPLING_OPTIONS = [
    "good",
    "fair",
    "poor",
]


def _allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def _analysis_error_message(error):
    """
    Convert known pipeline failures into useful user-facing messages.

    Detailed exception information is kept in the Flask log and is not
    exposed directly to the user.
    """

    message = str(error).strip()

    if isinstance(error, FileNotFoundError):
        return (
            "The uploaded video could not be found while processing it. "
            "Please upload the video again."
        )

    if "No frames could be scored" in message:
        return (
            "ErgoScore could not detect enough usable body landmarks "
            "in this video. Please use a video where the person is "
            "clearly visible and the upper body is not heavily occluded."
        )

    if "video" in message.lower() and (
        "open" in message.lower()
        or "read" in message.lower()
        or "decode" in message.lower()
    ):
        return (
            "The video could not be read correctly. "
            "Please try another video file."
        )

    if isinstance(error, ValueError):
        return (
            "The video could not be analyzed with the selected settings. "
            "Please try another video with a clearly visible person."
        )

    return (
        "ErgoScore could not complete the analysis for this video. "
        "Please try another video."
    )


@main_bp.route("/", methods=["GET"])
def index():
    """Render the video upload page."""

    return render_template(
        "index.html",
        load_options=LOAD_OPTIONS,
        coupling_options=COUPLING_OPTIONS,
    )


@main_bp.route("/analyze", methods=["POST"])
def analyze():
    """Upload a video and run the complete ErgoScore pipeline."""

    video = request.files.get("video")

    # ---------------------------------------------------------
    # Validate uploaded file
    # ---------------------------------------------------------

    if video is None or video.filename == "":
        flash(
            "Please choose a video file to upload.",
            "error",
        )
        return redirect(url_for("main.index"))

    if not _allowed_file(video.filename):
        flash(
            "Unsupported file type. Please upload one of: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "error",
        )
        return redirect(url_for("main.index"))

    # ---------------------------------------------------------
    # Read user options
    # ---------------------------------------------------------

    load = request.form.get(
        "load",
        "medium",
    )

    coupling = request.form.get(
        "coupling",
        "fair",
    )

    if load not in LOAD_OPTIONS:
        load = "medium"

    if coupling not in COUPLING_OPTIONS:
        coupling = "fair"

    # ---------------------------------------------------------
    # Secure filename
    # ---------------------------------------------------------

    filename = secure_filename(
        video.filename
    )

    if not filename:
        flash(
            "That filename isn't valid. "
            "Please rename the file and try again.",
            "error",
        )
        return redirect(
            url_for("main.index")
        )

    # ---------------------------------------------------------
    # Save uploaded video
    # ---------------------------------------------------------

    upload_folder = current_app.config[
        "UPLOAD_FOLDER"
    ]

    os.makedirs(
        upload_folder,
        exist_ok=True,
    )

    save_path = os.path.join(
        upload_folder,
        filename,
    )

    try:
        video.save(save_path)

    except Exception:
        current_app.logger.exception(
            "Failed to save uploaded video"
        )

        flash(
            "Something went wrong while saving the video. "
            "Please try again.",
            "error",
        )

        return redirect(
            url_for("main.index")
        )

    # ---------------------------------------------------------
    # Run complete analysis
    #
    # Video
    #   -> Person 1
    #   -> Person 2
    #   -> Person 3
    # ---------------------------------------------------------

    try:
        result = analyze_video(
            save_path,
            load=load,
            coupling=coupling,
        )

    except Exception as error:
        current_app.logger.exception(
            "Pipeline analysis failed"
        )

        flash(
            _analysis_error_message(error),
            "error",
        )

        return redirect(
            url_for("main.index")
        )

    # ---------------------------------------------------------
    # Render results
    # ---------------------------------------------------------

    return render_template(
        "results.html",
        result=result,
        video_filename=filename,
        load=load,
        coupling=coupling,
    )


# =============================================================
# Uploaded video
# =============================================================

@main_bp.route(
    "/videos/uploads/<path:filename>"
)
def uploaded_video(filename):
    """Serve the original uploaded video."""

    upload_folder = current_app.config[
        "UPLOAD_FOLDER"
    ]

    safe_name = secure_filename(filename)

    if safe_name != filename:
        abort(404)

    file_path = os.path.join(
        upload_folder,
        safe_name,
    )

    if not os.path.isfile(file_path):
        abort(404)

    return send_from_directory(
        upload_folder,
        safe_name,
    )


# =============================================================
# Processed video
# =============================================================

@main_bp.route(
    "/videos/processed/<path:filename>"
)
def processed_video(filename):
    """Serve the MediaPipe-annotated processed video."""

    processed_folder = os.path.abspath(
        os.path.join(
            current_app.root_path,
            "..",
            "videos",
            "processed",
        )
    )

    safe_name = secure_filename(filename)

    if safe_name != filename:
        abort(404)

    file_path = os.path.join(
        processed_folder,
        safe_name,
    )

    if not os.path.isfile(file_path):
        abort(404)

    return send_from_directory(
        processed_folder,
        safe_name,
    )


# =============================================================
# Generated reports
# =============================================================

@main_bp.route(
    "/reports/<path:filename>"
)
def report_file(filename):
    """Serve generated HTML reports and report assets."""

    reports_folder = os.path.abspath(
        os.path.join(
            current_app.root_path,
            "..",
            "reports",
        )
    )

    # Normalize the path and prevent traversal.
    requested_path = os.path.normpath(
        os.path.join(
            reports_folder,
            filename,
        )
    )

    if not requested_path.startswith(
        reports_folder
    ):
        abort(404)

    if not os.path.isfile(requested_path):
        abort(404)

    directory = os.path.dirname(
        requested_path
    )

    file_name = os.path.basename(
        requested_path
    )

    return send_from_directory(
        directory,
        file_name,
    )
