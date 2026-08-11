"""
Cloud persistence: Cloudinary for uploaded images, Firestore for detection
records and dashboard aggregates.

Both services are initialised lazily and every call is wrapped so that a
network failure degrades the request rather than breaking it: a detection still
returns its prediction and recommendation even if the record cannot be saved.
The failure is reported back to the caller so the UI can say so honestly
instead of silently dropping data.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app import config

logger = logging.getLogger(__name__)

_cloudinary_state: dict[str, Any] = {"ready": False, "error": None, "checked": False}
_firestore_state: dict[str, Any] = {"client": None, "error": None, "checked": False}
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Cloudinary
# --------------------------------------------------------------------------
def _init_cloudinary() -> None:
    if _cloudinary_state["checked"]:
        return
    with _lock:
        if _cloudinary_state["checked"]:
            return
        _cloudinary_state["checked"] = True
        try:
            import cloudinary

            # app.config imports dotenv at module load, so CLOUDINARY_URL is
            # already in the environment and cloudinary picks it up here.
            cfg = cloudinary.config(secure=True)
            if not cfg.cloud_name:
                raise RuntimeError(
                    "CLOUDINARY_URL is not set. Add it to the .env file as "
                    "cloudinary://<api_key>:<api_secret>@<cloud_name>"
                )
            _cloudinary_state["ready"] = True
            logger.info("Cloudinary configured for cloud '%s'", cfg.cloud_name)
        except Exception as exc:
            logger.warning("Cloudinary unavailable: %s", exc)
            _cloudinary_state["error"] = str(exc)


def cloudinary_ready() -> bool:
    _init_cloudinary()
    return bool(_cloudinary_state["ready"])


def upload_image(raw: bytes, filename: str) -> dict[str, Any]:
    """Upload the submitted image and return its hosted URLs."""
    _init_cloudinary()
    if not _cloudinary_state["ready"]:
        return {
            "uploaded": False,
            "error": _cloudinary_state["error"] or "Cloudinary is not configured",
        }

    try:
        import cloudinary.uploader

        public_id = f"{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:12]}"
        response = cloudinary.uploader.upload(
            raw,
            folder=config.CLOUDINARY_FOLDER,
            public_id=public_id,
            resource_type="image",
            overwrite=False,
            timeout=config.CLOUD_TIMEOUT_SECONDS,
            context={"original_filename": filename},
        )
        return {
            "uploaded": True,
            "url": response.get("secure_url"),
            "public_id": response.get("public_id"),
            "thumbnail_url": _thumbnail_url(response.get("secure_url")),
            "width": response.get("width"),
            "height": response.get("height"),
            "bytes": response.get("bytes"),
            "format": response.get("format"),
        }
    except Exception as exc:
        logger.exception("Cloudinary upload failed")
        return {"uploaded": False, "error": str(exc)}


def _thumbnail_url(secure_url: str | None) -> str | None:
    """Insert a Cloudinary transformation for a 400px square thumbnail."""
    if not secure_url or "/upload/" not in secure_url:
        return secure_url
    return secure_url.replace("/upload/", "/upload/c_fill,w_400,h_400,q_auto,f_auto/", 1)


def delete_image(public_id: str) -> bool:
    _init_cloudinary()
    if not _cloudinary_state["ready"] or not public_id:
        return False
    try:
        import cloudinary.uploader

        cloudinary.uploader.destroy(public_id, timeout=config.CLOUD_TIMEOUT_SECONDS)
        return True
    except Exception:
        logger.exception("Cloudinary delete failed for %s", public_id)
        return False


# --------------------------------------------------------------------------
# Firestore
# --------------------------------------------------------------------------
def _init_firestore() -> None:
    if _firestore_state["checked"]:
        return
    with _lock:
        if _firestore_state["checked"]:
            return
        _firestore_state["checked"] = True
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not config.FIREBASE_CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Firebase service account key not found at "
                    f"{config.FIREBASE_CREDENTIALS_PATH}"
                )

            # initialize_app raises if called twice, which happens under the
            # Flask reloader; reuse the existing app in that case.
            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(str(config.FIREBASE_CREDENTIALS_PATH))
                firebase_admin.initialize_app(cred)

            _firestore_state["client"] = firestore.client()
            logger.info("Firestore client ready")
        except Exception as exc:
            logger.warning("Firestore unavailable: %s", exc)
            _firestore_state["error"] = str(exc)


def firestore_ready() -> bool:
    _init_firestore()
    return _firestore_state["client"] is not None


def firestore_error() -> str | None:
    _init_firestore()
    return _firestore_state["error"]


def cloudinary_error() -> str | None:
    _init_cloudinary()
    return _cloudinary_state["error"]


def save_detection(record: dict[str, Any]) -> dict[str, Any]:
    """Persist one detection record. Returns {saved, id} or {saved, error}."""
    _init_firestore()
    client = _firestore_state["client"]
    if client is None:
        return {
            "saved": False,
            "error": _firestore_state["error"] or "Firestore is not configured",
        }

    try:
        from firebase_admin import firestore

        payload = dict(record)
        payload["created_at"] = firestore.SERVER_TIMESTAMP
        document = client.collection(config.FIRESTORE_COLLECTION).document(
            record["detection_id"]
        )
        document.set(payload)
        return {"saved": True, "id": record["detection_id"]}
    except Exception as exc:
        logger.exception("Firestore save failed")
        return {"saved": False, "error": str(exc)}


def get_detection(detection_id: str) -> dict[str, Any] | None:
    _init_firestore()
    client = _firestore_state["client"]
    if client is None:
        return None
    try:
        snapshot = (
            client.collection(config.FIRESTORE_COLLECTION).document(detection_id).get()
        )
        if not snapshot.exists:
            return None
        return _normalise(snapshot.to_dict())
    except Exception:
        logger.exception("Firestore fetch failed for %s", detection_id)
        return None


def list_detections(limit: int | None = None) -> dict[str, Any]:
    """Most recent detections first."""
    _init_firestore()
    client = _firestore_state["client"]
    if client is None:
        return {
            "available": False,
            "records": [],
            "error": _firestore_state["error"] or "Firestore is not configured",
        }

    limit = limit or config.HISTORY_PAGE_SIZE
    try:
        from firebase_admin import firestore

        query = (
            client.collection(config.FIRESTORE_COLLECTION)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        records = [_normalise(doc.to_dict()) for doc in query.stream()]
        return {"available": True, "records": records, "error": None}
    except Exception as exc:
        # A composite index may be missing on a fresh project; fall back to an
        # unordered read so history still displays.
        logger.warning("Ordered history query failed (%s); falling back", exc)
        try:
            docs = client.collection(config.FIRESTORE_COLLECTION).limit(limit).stream()
            records = [_normalise(doc.to_dict()) for doc in docs]
            records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
            return {"available": True, "records": records, "error": None}
        except Exception as inner:
            logger.exception("Firestore history read failed")
            return {"available": False, "records": [], "error": str(inner)}


def delete_detection(detection_id: str) -> dict[str, Any]:
    """Remove a detection record and its Cloudinary image."""
    _init_firestore()
    client = _firestore_state["client"]
    if client is None:
        return {"deleted": False, "error": "Firestore is not configured"}
    try:
        reference = client.collection(config.FIRESTORE_COLLECTION).document(detection_id)
        snapshot = reference.get()
        if snapshot.exists:
            public_id = (snapshot.to_dict() or {}).get("image", {}).get("public_id")
            if public_id:
                delete_image(public_id)
        reference.delete()
        return {"deleted": True}
    except Exception as exc:
        logger.exception("Firestore delete failed")
        return {"deleted": False, "error": str(exc)}


def detection_statistics(limit: int = 500) -> dict[str, Any]:
    """Aggregate saved detections for the dashboard's usage panel."""
    result = list_detections(limit=limit)
    if not result["available"]:
        return {"available": False, "error": result["error"]}

    records = result["records"]
    by_pest: dict[str, int] = {}
    by_action: dict[str, int] = {}
    confidences: list[float] = []
    ood_rejected = 0

    for record in records:
        if record.get("ood", {}).get("is_ood"):
            ood_rejected += 1
            continue
        pest = record.get("pest", {}).get("display_name") or "Unknown"
        by_pest[pest] = by_pest.get(pest, 0) + 1
        action = record.get("recommendation", {}).get("action_level") or "n/a"
        by_action[action] = by_action.get(action, 0) + 1
        confidence = record.get("prediction", {}).get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    return {
        "available": True,
        "total": len(records),
        "accepted": len(records) - ood_rejected,
        "rejected": ood_rejected,
        "rejection_rate": round(ood_rejected / len(records) * 100, 1) if records else 0.0,
        "mean_confidence": round(sum(confidences) / len(confidences) * 100, 1)
        if confidences
        else None,
        "by_pest": dict(sorted(by_pest.items(), key=lambda kv: kv[1], reverse=True)),
        "by_action": by_action,
    }


def _normalise(data: dict[str, Any] | None) -> dict[str, Any]:
    """Convert Firestore timestamps into ISO strings for JSON/Jinja use."""
    if not data:
        return {}
    record = dict(data)
    created = record.get("created_at")
    if created is not None and hasattr(created, "isoformat"):
        record["created_at"] = created.isoformat()
    elif created is not None:
        record["created_at"] = str(created)
    return record


def service_status() -> dict[str, Any]:
    """Health summary shown on the dashboard."""
    return {
        "firestore": {
            "ready": firestore_ready(),
            "error": firestore_error(),
            "collection": config.FIRESTORE_COLLECTION,
        },
        "cloudinary": {
            "ready": cloudinary_ready(),
            "error": cloudinary_error(),
            "folder": config.CLOUDINARY_FOLDER,
        },
    }
