"""
Page routes: everything the browser navigates to.

Detection results are held in the session only as an id; the full record is
read back from Firestore so a result link can be shared or reopened later.
"""

from __future__ import annotations

import logging

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app import config
from app.services import (
    cloud_store,
    detection_pipeline,
    expert_mapping,
    knowledge_base,
    metrics_service,
    model_service,
)

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Upload page - the entry point of the workflow."""
    return render_template(
        "index.html",
        status=detection_pipeline.system_status(),
        class_names=model_service.get_class_names(),
        pest_count=len(knowledge_base.all_pests()),
        recent=session.get("last_detection_id"),
    )


@main_bp.route("/detect", methods=["POST"])
def detect():
    """Handle the upload form and redirect to the result page."""
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        flash("Please choose an image before submitting.", "error")
        return redirect(url_for("main.index"))

    raw = upload.read()

    days_to_harvest = request.form.get("days_to_harvest", "").strip()
    try:
        days_value = int(days_to_harvest) if days_to_harvest else None
        if days_value is not None and not 0 <= days_value <= 365:
            days_value = None
    except ValueError:
        days_value = None

    try:
        record = detection_pipeline.run(
            raw,
            upload.filename,
            growth_stage=request.form.get("growth_stage", "vegetative"),
            severity=request.form.get("severity", "moderate"),
            weather=request.form.get("weather", "humid"),
            days_to_harvest=days_value,
            beneficials_present=request.form.get("beneficials") == "on",
            field_note=request.form.get("field_note", ""),
        )
    except detection_pipeline.DetectionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))
    except Exception:
        logger.exception("Detection failed")
        flash(
            "The detection pipeline failed unexpectedly. Check the server "
            "terminal for details.",
            "error",
        )
        return redirect(url_for("main.index"))

    session["last_detection_id"] = record["detection_id"]

    # Firestore is the source of truth for the result page, but if the save
    # failed the record is carried in the session so the user still sees it.
    if not record.get("persistence", {}).get("saved"):
        session["fallback_record"] = record

    return redirect(url_for("main.result", detection_id=record["detection_id"]))


@main_bp.route("/result/<detection_id>")
def result(detection_id: str):
    record = cloud_store.get_detection(detection_id)

    if record is None:
        fallback = session.get("fallback_record")
        if fallback and fallback.get("detection_id") == detection_id:
            record = fallback
        else:
            abort(404)

    return render_template("result.html", record=record)


@main_bp.route("/history")
def history():
    result_set = cloud_store.list_detections(limit=config.HISTORY_PAGE_SIZE)
    return render_template(
        "history.html",
        records=result_set["records"],
        available=result_set["available"],
        error=result_set.get("error"),
        stats=cloud_store.detection_statistics(),
    )


@main_bp.route("/history/<detection_id>/delete", methods=["POST"])
def delete_detection(detection_id: str):
    outcome = cloud_store.delete_detection(detection_id)
    if outcome.get("deleted"):
        flash("Detection record deleted.", "success")
    else:
        flash(f"Could not delete: {outcome.get('error')}", "error")
    return redirect(url_for("main.history"))


@main_bp.route("/knowledge-base")
def knowledge():
    pests = knowledge_base.all_pests()
    query = request.args.get("q", "").strip().lower()
    if query:
        pests = [
            p
            for p in pests
            if query in p["common_name"].lower()
            or query in p["scientific_name"].lower()
            or query in p["pest_group"].lower()
            or any(query in ai.lower() for ai in p["active_ingredients"])
        ]

    return render_template(
        "knowledge.html",
        pests=pests,
        query=request.args.get("q", ""),
        stats=knowledge_base.stats(),
        error=knowledge_base.load_error(),
    )


@main_bp.route("/knowledge-base/<slug>")
def pest_detail(slug: str):
    pest = knowledge_base.find_pest_by_slug(slug)
    if pest is None:
        abort(404)

    # Find the AI class that maps to this pest so the page can show the link
    # between the trained model and the knowledge base entry.
    linked_class = None
    for ai_class, entry in expert_mapping.CLASS_MAPPING.items():
        if entry["kb_name"].lower() == pest["common_name"].lower():
            linked_class = ai_class
            break

    metrics = metrics_service.load()
    class_metrics = None
    if linked_class and metrics.get("available"):
        class_metrics = next(
            (r for r in metrics["per_class"] if r["class_name"] == linked_class), None
        )

    return render_template(
        "pest_detail.html",
        pest=pest,
        linked_class=linked_class,
        class_metrics=class_metrics,
    )


@main_bp.route("/dashboard")
def dashboard():
    metrics = metrics_service.load()
    return render_template(
        "dashboard.html",
        metrics=metrics,
        confusions=metrics_service.confusion_insights(),
        extremes=metrics_service.per_class_extremes(),
        coverage=expert_mapping.coverage_report() if model_service.is_ready() else None,
        usage=cloud_store.detection_statistics(),
        status=detection_pipeline.system_status(),
    )


@main_bp.route("/about")
def about():
    return render_template(
        "about.html",
        status=detection_pipeline.system_status(),
        metrics=metrics_service.load(),
        ood_config=config.OOD_DEFAULTS,
        class_names=model_service.get_class_names(),
    )


@main_bp.route("/model/result/<path:filename>")
def result_asset(filename: str):
    """Serve the training graphs stored outside the static folder."""
    from flask import send_from_directory

    if filename not in {"accuracy_graph.png", "loss_graph.png"}:
        abort(404)
    return send_from_directory(config.RESULT_DIR, filename)
