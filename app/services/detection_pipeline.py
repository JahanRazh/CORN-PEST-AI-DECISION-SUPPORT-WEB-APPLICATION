"""
Detection pipeline: the orchestrator that runs one image through every layer
and assembles the explainable result.

    1. Validate the upload (type, size, decodable image)
    2. Classify with EfficientNetB0            -> model_service
    3. Reject unknown / out-of-distribution    -> ood_service
    4. Validate the pest identity              -> expert_mapping
    5. Persist image and record                -> cloud_store

Each stage appends to a pipeline trace so the result page can show exactly
which component made which decision - the "decision reasoning" requirement.
Stages 4-5 are skipped when stage 3 rejects the image, because naming a pest
for an out-of-distribution photograph is precisely the failure this system
exists to prevent.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app import config
from app.services import (
    cloud_store,
    expert_mapping,
    knowledge_base,
    model_service,
    ood_service,
)

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Raised for user-correctable problems with the submitted image."""


def validate_upload(filename: str, raw: bytes) -> None:
    if not raw:
        raise DetectionError("The uploaded file is empty.")
    if len(raw) > config.MAX_CONTENT_LENGTH:
        limit = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        raise DetectionError(f"Image is larger than the {limit} MB limit.")

    extension = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DetectionError(
            f"'{extension or filename}' is not a supported image type. "
            f"Please upload one of: {allowed}."
        )


def run(
    raw: bytes,
    filename: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Execute the full decision-support pipeline for one image."""
    validate_upload(filename, raw)

    detection_id = uuid.uuid4().hex[:16]
    timestamp = datetime.now(timezone.utc)
    trace: list[dict[str, str]] = []

    # --- Stage 1: decode -------------------------------------------------
    try:
        image = model_service.load_image(raw)
    except Exception as exc:
        raise DetectionError(
            "The file could not be read as an image. It may be corrupted or "
            "saved in an unsupported format."
        ) from exc

    extension = (filename.rsplit(".", 1)[-1] if "." in filename else "unknown").upper()
    trace.append(
        {
            "stage": "Image validation",
            "component": "Format & Size Validator",
            "outcome": f"Accepted {image.width}x{image.height} {extension} image "
                       f"({len(raw) / 1024:.0f} KB)",
        }
    )

    # --- Stage 1.5: Relevance gate (ImageNet first-pass) ------------------
    relevance = ood_service.relevance_score(image)
    if relevance.get("available") and relevance.get("score") is not None:
        if not relevance.get("passed"):
            trace.append(
                {
                    "stage": "ImageNet Validation Gate",
                    "component": "MobileNetV2 (ImageNet)",
                    "outcome": f"Rejected - Not an insect or crop plant (score: {relevance['score']:.4f})",
                }
            )
            
            trace.append({
                "stage": "Pest classification",
                "component": "EfficientNetB0 (transfer learning + fine tuning)",
                "outcome": "Skipped - image failed ImageNet relevance check"
            })
            
            trace.append({
                "stage": "Unknown image rejection",
                "component": "OOD detection layer",
                "outcome": "Skipped - image already rejected by ImageNet gate"
            })
            
            trace.append({
                "stage": "Expert validation",
                "component": "Expert mapping layer",
                "outcome": "Skipped - no pest identity is asserted for a rejected image"
            })
            
            # Fast-fail rejection
            ood_result = ood_service.OODResult(
                is_ood=True,
                status="rejected",
                reason="The image was rejected by the ImageNet relevance gate because it does not appear to contain an insect or crop plant. Pest identification was bypassed.",
                votes=0,
                votes_required=int(config.OOD_DEFAULTS.get("votes_required", 2)),
                relevance=relevance,
                calibrated=False,
            )
            
            record: dict[str, Any] = {
                "detection_id": detection_id,
                "timestamp": timestamp.isoformat(),
                "filename": filename,
                "prediction": None,
                "ood": ood_result.to_dict(),
                "image": {"uploaded": False},
                "status": "rejected",
                "pest": None,
                "rejection": {
                    "title": "Unknown or unrelated image",
                    "message": ood_result.reason,
                    "guidance": _rejection_guidance(ood_result),
                }
            }
            record["trace"] = trace
            
            # Persist and return early
            if persist:
                upload = cloud_store.upload_image(raw, filename)
                record["image"] = upload
                trace.append({
                    "stage": "Image storage", "component": "Cloudinary", 
                    "outcome": "Stored" if upload.get("uploaded") else f"Not stored - {upload.get('error', 'unknown error')}"
                })
                saved = cloud_store.save_detection(record)
                record["persistence"] = saved
                trace.append({
                    "stage": "Record storage", "component": f"Firestore ({config.FIRESTORE_COLLECTION})", 
                    "outcome": "Saved" if saved.get("saved") else f"Not saved - {saved.get('error', 'unknown error')}"
                })
            else:
                record["persistence"] = {"saved": False, "error": "Persistence disabled"}

            return record
        else:
            trace.append(
                {
                    "stage": "ImageNet Validation Gate",
                    "component": "MobileNetV2 (ImageNet)",
                    "outcome": f"Passed - Plausible insect/plant content (score: {relevance['score']:.4f})",
                }
            )
    else:
        trace.append(
            {
                "stage": "ImageNet Validation Gate",
                "component": "MobileNetV2 (ImageNet)",
                "outcome": f"Skipped - Model unavailable ({relevance.get('error', 'Disabled')})",
            }
        )

    # --- Stage 2: classify ------------------------------------------------
    if not model_service.is_ready():
        raise DetectionError(
            f"The detection model is unavailable: {model_service.load_error()}"
        )

    prediction = model_service.predict(image)
    trace.append(
        {
            "stage": "Pest classification",
            "component": "EfficientNetB0 (transfer learning + fine tuning)",
            "outcome": f"Top class '{prediction.class_name}' at "
                       f"{prediction.confidence * 100:.2f}% confidence",
        }
    )

    # --- Stage 3: unknown / OOD rejection ---------------------------------
    ood = ood_service.evaluate(prediction, image=image, precomputed_relevance=relevance)
    trace.append(
        {
            "stage": "Unknown image rejection",
            "component": "OOD detection layer "
                         f"({'calibrated' if ood.calibrated else 'default thresholds'})",
            "outcome": f"{ood.status.title()} - {ood.votes} of "
                       f"{len(ood.signals)} signals flagged",
        }
    )

    band = model_service.confidence_band(prediction.confidence)

    record: dict[str, Any] = {
        "detection_id": detection_id,
        "timestamp": timestamp.isoformat(),
        "filename": filename,
        "prediction": {
            "class_name": prediction.class_name,
            "class_index": prediction.class_index,
            "confidence": round(prediction.confidence, 6),
            "confidence_percentage": round(prediction.confidence * 100, 2),
            "confidence_band": band["label"],
            "confidence_colour": band["colour"],
            "margin": round(prediction.margin, 4),
            "ranked": prediction.ranked(),
        },
        "ood": ood.to_dict(),
        "image": {"uploaded": False},
    }

    # --- Stage 4 and 5: only for accepted predictions ---------------------
    if ood.is_ood:
        record["status"] = "rejected"
        record["pest"] = None
        record["rejection"] = {
            "title": "Unknown or unrelated image",
            "message": ood.reason,
            "guidance": _rejection_guidance(ood),
        }
        trace.append(
            {
                "stage": "Expert validation",
                "component": "Expert mapping layer",
                "outcome": "Skipped - no pest identity is asserted for a "
                           "rejected image",
            }
        )
    else:
        mapping = expert_mapping.map_class(prediction.class_name)
        trace.append(
            {
                "stage": "Expert validation",
                "component": "Expert mapping layer",
                "outcome": f"AI class resolved to '{mapping.display_name}' via "
                           f"{mapping.match_method}",
            }
        )
        record["status"] = "accepted"
        record["pest"] = {
            **mapping.to_dict(),
            "profile": mapping.pest_profile,
        }

    record["trace"] = trace

    # --- Stage 6: persistence --------------------------------------------
    if persist:
        upload = cloud_store.upload_image(raw, filename)
        record["image"] = upload
        trace.append(
            {
                "stage": "Image storage",
                "component": "Cloudinary",
                "outcome": "Stored" if upload.get("uploaded")
                else f"Not stored - {upload.get('error', 'unknown error')}",
            }
        )

        saved = cloud_store.save_detection(record)
        record["persistence"] = saved
        trace.append(
            {
                "stage": "Record storage",
                "component": f"Firestore ({config.FIRESTORE_COLLECTION})",
                "outcome": "Saved" if saved.get("saved")
                else f"Not saved - {saved.get('error', 'unknown error')}",
            }
        )
    else:
        record["persistence"] = {"saved": False, "error": "Persistence disabled"}

    return record


def _rejection_guidance(ood) -> list[str]:
    """Actionable advice tailored to which OOD signals fired."""
    tips: list[str] = []
    relevance = ood.relevance or {}

    if relevance.get("available") and not relevance.get("passed", True):
        top = relevance.get("labels") or []
        if top:
            tips.append(
                f"The image looks more like '{top[0]['label']}' than a corn pest. "
                "Please upload a photograph of the insect or the damaged plant."
            )
        else:
            tips.append(
                "No insect or plant content was detected. Upload a clear "
                "photograph of the pest or the affected crop."
            )

    flagged = {s["abbreviation"] for s in ood.signals if s["flagged"]}
    if "M" in flagged:
        tips.append(
            "The model was torn between two similar pests. A closer, sharper "
            "photograph of the insect's body markings would help separate them."
        )
    if "MSP" in flagged or "H" in flagged:
        tips.append(
            "Confidence was spread thinly across classes. Try better lighting, "
            "a plain background, and filling more of the frame with the pest."
        )
    if "E" in flagged:
        tips.append(
            "The image activates the network weakly, which usually means it is "
            "unlike anything in the training data - it may be a pest species "
            "outside the ten this system was trained on."
        )

    tips.append(
        "The system covers ten corn pests only. Anything else is deliberately "
        "rejected rather than forced into the nearest class."
    )
    return tips


def system_status() -> dict[str, Any]:
    """Component health used by the dashboard and the start-up banner."""
    cloud = cloud_store.service_status()
    return {
        "model": {
            "ready": model_service.is_ready(),
            "error": model_service.load_error(),
            "classes": len(model_service.get_class_names()),
            "path": str(config.MODEL_PATH.name),
        },
        "knowledge_base": {
            "ready": knowledge_base.is_ready(),
            "error": knowledge_base.load_error(),
            **(knowledge_base.stats() if knowledge_base.is_ready() else {}),
        },
        "ood": {
            "ready": True,
            "calibrated": ood_service.load_stats()["calibrated"],
            "relevance_gate": config.RELEVANCE_GATE_ENABLED,
        },
        **cloud,
    }
