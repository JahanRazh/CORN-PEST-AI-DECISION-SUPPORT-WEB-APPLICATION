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

    days_raw = (request.form.get("days_to_harvest") or "").strip()
    try:
        days_value = int(days_raw) if days_raw else None
        if days_value is not None and not 0 <= days_value <= 365:
            days_value = None
    except ValueError:
        days_value = None

    try:
        record = detection_pipeline.run(
            upload.read(),
            upload.filename,
            growth_stage=request.form.get("growth_stage", "vegetative"),
            severity=request.form.get("severity", "moderate"),
            weather=request.form.get("weather", "humid"),
            days_to_harvest=days_value,
            beneficials_present=request.form.get("beneficials") in {"on", "true", "1"},
            field_note=request.form.get("field_note", ""),
            persist=request.form.get("persist", "true") != "false",
        )
    except detection_pipeline.DetectionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("API detection failed")
        return jsonify({"ok": False, "error": f"Pipeline failure: {exc}"}), 500

    return jsonify({"ok": True, "record": record})


@api_bp.route("/recompute", methods=["POST"])
def recompute():
    """Re-run the rule engine for a known pest under different field context.

    This is what makes the result page interactive: changing the growth stage
    or severity slider updates the recommendation instantly, without paying
    for another forward pass through the network.
    """
    payload = request.get_json(silent=True) or {}
    ai_class = payload.get("ai_class")
    if not ai_class:
        return jsonify({"ok": False, "error": "ai_class is required."}), 400

    if ai_class not in model_service.get_class_names():
        return jsonify({"ok": False, "error": f"Unknown class '{ai_class}'."}), 400

    days = payload.get("days_to_harvest")
    if isinstance(days, str):
        days = int(days) if days.strip().isdigit() else None

    try:
        confidence = float(payload.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    mapping = expert_mapping.map_class(ai_class)
    recommendation = rules_engine.generate(
        mapping,
        growth_stage=payload.get("growth_stage", "vegetative"),
        severity=payload.get("severity", "moderate"),
        weather=payload.get("weather", "humid"),
        days_to_harvest=days,
        beneficials_present=bool(payload.get("beneficials_present")),
        confidence=confidence,
    )

    payload_out = recommendation.to_dict()

    # Render the same Jinja partial the result page used on first load, so the
    # live update and the initial render can never drift apart.
    html = render_template("partials/_recommendation.html", recommendation=payload_out)

    return jsonify(
        {
            "ok": True,
            "pest": mapping.to_dict(),
            "recommendation": payload_out,
            "html": html,
        }
    )


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
