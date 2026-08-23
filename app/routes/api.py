"""
JSON API backing the interactive front end.

/api/detect          run the pipeline and return the full record (used by the
                     drag-and-drop analyser for an in-page result)
/api/recompute       re-run only the rule engine with changed field context, so
                     the result page can update recommendations live without
                     re-running the model
/api/metrics         model performance data for the dashboard charts
/api/pests           knowledge base listing for the search box
/api/status          component health for the status indicators
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template, request

from app.services import (
    cloud_store,
    detection_pipeline,
    expert_mapping,
    knowledge_base,
    metrics_service,
    model_service,
    rules_engine,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.route("/detect", methods=["POST"])
def detect():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "No image was supplied."}), 400

    try:
        record = detection_pipeline.run(
            upload.read(),
            upload.filename,
            persist=request.form.get("persist", "true") != "false",
        )
    except detection_pipeline.DetectionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("API detection failed")
        return jsonify({"ok": False, "error": f"Pipeline failure: {exc}"}), 500

    return jsonify({"ok": True, "record": record})




@api_bp.route("/metrics")
def metrics():
    data = metrics_service.load()
    return jsonify(
        {
            "ok": data.get("available", False),
            "error": data.get("error"),
            "headline": data.get("headline", {}),
            "per_class": data.get("per_class", []),
            "averages": data.get("averages", {}),
            "confusion_matrix": data.get("confusion_matrix", []),
            "class_names": data.get("class_names", []),
            "history": data.get("history", []),
            "confusions": metrics_service.confusion_insights(),
        }
    )


@api_bp.route("/pests")
def pests():
    query = (request.args.get("q") or "").strip().lower()
    records = knowledge_base.all_pests()
    if query:
        records = [
            p
            for p in records
            if query in p["common_name"].lower()
            or query in p["scientific_name"].lower()
            or any(query in ai.lower() for ai in p["active_ingredients"])
        ]
    return jsonify(
        {
            "ok": knowledge_base.is_ready(),
            "error": knowledge_base.load_error(),
            "count": len(records),
            "pests": [
                {
                    "common_name": p["common_name"],
                    "scientific_name": p["scientific_name"],
                    "pest_group": p["pest_group"],
                    "slug": p["slug"],
                    "active_ingredients": p["active_ingredients"],
                    "moa_groups": p["moa_groups"],
                    "biological_controls": p["biological_controls"],
                }
                for p in records
            ],
        }
    )


@api_bp.route("/pests/<slug>")
def pest_detail(slug: str):
    pest = knowledge_base.find_pest_by_slug(slug)
    if pest is None:
        return jsonify({"ok": False, "error": "Pest not found."}), 404
    return jsonify({"ok": True, "pest": pest})


@api_bp.route("/status")
def status():
    return jsonify({"ok": True, "status": detection_pipeline.system_status()})


@api_bp.route("/history")
def history():
    limit = request.args.get("limit", type=int) or 50
    result = cloud_store.list_detections(limit=min(limit, 200))
    return jsonify(
        {
            "ok": result["available"],
            "error": result.get("error"),
            "records": result["records"],
            "statistics": cloud_store.detection_statistics(),
        }
    )


@api_bp.route("/reload", methods=["POST"])
def reload_sources():
    """Re-read the Excel workbook and metrics without restarting the server."""
    knowledge_base.reload()
    metrics_service.reload()
    return jsonify(
        {
            "ok": knowledge_base.is_ready(),
            "knowledge_base": knowledge_base.stats() if knowledge_base.is_ready() else {},
            "error": knowledge_base.load_error(),
        }
    )
