"""
Route blueprints for the CornGuard AI web application.

    main_bp   – page routes rendered by Jinja (/, /detect, /result, …)
    api_bp    – JSON API backing the interactive front end (/api/…)
"""

from app.routes.api import api_bp
from app.routes.main import main_bp

__all__ = ["api_bp", "main_bp"]
