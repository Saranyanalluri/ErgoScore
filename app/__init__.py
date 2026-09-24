"""
ErgoScore Flask application factory.

This module wires together configuration, folders, blueprints, and error
handling. It is intentionally simple so other team members' modules can be
plugged into app/pipeline.py without touching this file.
"""

import os

from flask import Flask, render_template


def create_app(test_config=None):
    """
    Application factory.

    Args:
        test_config: optional dict of config overrides, used by the test
            suite to point UPLOAD_FOLDER / PROCESSED_FOLDER at a temp dir.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    app = Flask(
        __name__,
        instance_relative_config=False,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-me"),
        UPLOAD_FOLDER=os.path.join(base_dir, "videos", "uploads"),
        PROCESSED_FOLDER=os.path.join(base_dir, "videos", "processed"),
        MAX_CONTENT_LENGTH=200 * 1024 * 1024,  # 200 MB max upload size
    )

    if test_config:
        app.config.update(test_config)

    # Make sure the folders we depend on actually exist.
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

    from .routes import main_bp
    app.register_blueprint(main_bp)

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", message="Page not found."), 404

    @app.errorhandler(500)
    def server_error(_error):
        # Never leak a traceback to the user.
        app.logger.exception("Unhandled server error")
        return render_template(
            "error.html",
            message="Something went wrong on our end. Please try again.",
        ), 500

    @app.errorhandler(413)
    def file_too_large(_error):
        return render_template(
            "error.html",
            message="That file is too large. Please upload a smaller video.",
        ), 413

    return app
