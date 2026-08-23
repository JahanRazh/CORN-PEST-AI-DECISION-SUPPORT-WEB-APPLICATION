"""
Flask application factory.

Front end and back end are served from this single Flask process: Jinja
templates render the pages, a small JSON API under /api backs the interactive
parts, and Tailwind is delivered from a CDN so there is no build step.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template

from app import config


def create_app() -> Flask:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=True,
    )

    from app.routes.api import api_bp
    from app.routes.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    _register_error_handlers(app)
    _register_template_helpers(app)

    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(error):  # type: ignore[no-untyped-def]
        return render_template(
            "error.html",
            code=404,
            title="Page not found",
            message="That page does not exist. Use the navigation above to get back.",
        ), 404

    @app.errorhandler(413)
    def too_large(error):  # type: ignore[no-untyped-def]
        limit = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        message = f"That image is larger than the {limit} MB limit."
        if _wants_json():
            return jsonify({"ok": False, "error": message}), 413
        return render_template(
            "error.html", code=413, title="Image too large", message=message
        ), 413

    @app.errorhandler(500)
    def server_error(error):  # type: ignore[no-untyped-def]
        app.logger.exception("Unhandled server error")
        message = (
            "Something went wrong while processing that request. Check the "
            "terminal running the server for the full traceback."
        )
        if _wants_json():
            return jsonify({"ok": False, "error": message}), 500
        return render_template(
            "error.html", code=500, title="Server error", message=message
        ), 500


def _wants_json() -> bool:
    from flask import request

    return request.path.startswith("/api") or request.accept_mimetypes.best == "application/json"


def _register_template_helpers(app: Flask) -> None:
    """Expose config vocabularies and small filters to every template."""

    @app.context_processor
    def inject_globals():  # type: ignore[no-untyped-def]
        return {
            "APP_NAME": "CornGuard AI",
            "APP_TAGLINE": "Corn Pest Decision Support System",
        }

    @app.template_filter("pct")
    def percentage(value: float | None, digits: int = 1) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{digits}f}%"

    @app.template_filter("shortclass")
    def short_class(value: str) -> str:
        """'Army Worm-Spodoptera frugiperda' -> 'Army Worm'."""
        return (value or "").split("-")[0].strip()

    @app.template_filter("sciname")
    def scientific_name(value: str) -> str:
        parts = (value or "").split("-", 1)
        return parts[1].strip() if len(parts) > 1 else ""
