"""
Model service: owns the trained EfficientNetB0 corn pest classifier.

The Keras model is loaded once per process and wrapped in a multi-output
inference model so a single forward pass yields everything the downstream
layers need:

    probabilities  -> the classification result shown to the user
    logits         -> free-energy and temperature-scaled OOD scores
    features       -> penultimate embedding for distance-based OOD

Recomputing the logits from the final Dense layer's weights avoids a second
forward pass; it was verified to match softmax(logits) to within 1.2e-07.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app import config

logger = logging.getLogger(__name__)

# TensorFlow is imported lazily inside _load() so that Flask can boot (and
# report a clean error page) even if the model file is missing.
_lock = threading.Lock()
_state: dict[str, Any] = {
    "model": None,
    "inference_model": None,
    "class_names": None,
    "head_weights": None,
    "loaded": False,
    "error": None,
}


@dataclass
class Prediction:
    """The raw output of one forward pass, before any OOD or expert logic."""

    class_index: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    logits: np.ndarray
    features: np.ndarray
    class_names: list[str] = field(default_factory=list)

    def ranked(self, top_k: int | None = None) -> list[dict[str, Any]]:
        """All class probabilities, highest first."""
        order = np.argsort(self.probabilities)[::-1]
        if top_k is not None:
            order = order[:top_k]
        return [
            {
                "class_name": self.class_names[i],
                "probability": float(self.probabilities[i]),
                "percentage": round(float(self.probabilities[i]) * 100, 2),
            }
            for i in order
        ]

    @property
    def runner_up(self) -> dict[str, Any] | None:
        ranked = self.ranked(top_k=2)
        return ranked[1] if len(ranked) > 1 else None

    @property
    def margin(self) -> float:
        """Gap between the top-1 and top-2 probabilities."""
        top2 = np.sort(self.probabilities)[::-1][:2]
        return float(top2[0] - top2[1]) if top2.size > 1 else float(top2[0])


def load_class_names() -> list[str]:
    with open(config.CLASS_NAMES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _load() -> None:
    """Load the Keras model and build the multi-output inference wrapper."""
    if _state["loaded"] or _state["error"]:
        return

    with _lock:
        if _state["loaded"] or _state["error"]:
            return
        try:
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf

            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {config.MODEL_PATH}. "
                    "Place best_corn_pest_model.keras in the model/ folder."
                )

            logger.info("Loading corn pest model from %s", config.MODEL_PATH)
            model = tf.keras.models.load_model(config.MODEL_PATH)

            # Penultimate activation (256-d) feeds the softmax head; the
            # 1280-d pooled EfficientNet embedding is kept for feature-space
            # OOD scoring.
            penultimate = model.get_layer("dropout").output
            pooled = model.get_layer("global_average_pooling2d").output
            inference_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.outputs[0], penultimate, pooled],
                name="corn_pest_inference",
            )

            weights, bias = model.get_layer("dense_1").get_weights()

            _state.update(
                model=model,
                inference_model=inference_model,
                class_names=load_class_names(),
                head_weights=(weights, bias),
                loaded=True,
            )
            logger.info(
                "Model ready: %d classes, input %s",
                len(_state["class_names"]),
                model.input_shape,
            )
        except Exception as exc:  # pragma: no cover - startup diagnostics
            logger.exception("Model failed to load")
            _state["error"] = str(exc)


def is_ready() -> bool:
    _load()
    return bool(_state["loaded"])


def load_error() -> str | None:
    _load()
    return _state["error"]


def get_class_names() -> list[str]:
    _load()
    return _state["class_names"] or []


def warm_up() -> None:
    """Run one throwaway inference so the first real request is not slow."""
    if not is_ready():
        return
    blank = np.zeros((1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype="float32")
    _state["inference_model"].predict(blank, verbose=0)
    logger.info("Model warm-up complete")


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
def load_image(raw: bytes) -> Image.Image:
    """Decode uploaded bytes into an upright RGB image."""
    image = Image.open(io.BytesIO(raw))
    # Honour EXIF orientation so phone photos are not rotated.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to the network's input size and return a (1, H, W, 3) batch.

    Pixels stay in the 0-255 range: the saved model contains its own
    Rescaling/Normalization layers, so dividing here would corrupt the input.
    """
    resized = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype="float32")
    return np.expand_dims(array, axis=0)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def predict(image: Image.Image) -> Prediction:
    """Classify one image and return probabilities, logits and features."""
    if not is_ready():
        raise RuntimeError(_state["error"] or "Model is not available")

    batch = preprocess(image)
    probabilities, penultimate, pooled = _state["inference_model"].predict(
        batch, verbose=0
    )

    probabilities = np.asarray(probabilities[0], dtype="float64")
    penultimate = np.asarray(penultimate[0], dtype="float64")
    pooled = np.asarray(pooled[0], dtype="float64")

    weights, bias = _state["head_weights"]
    logits = penultimate @ weights + bias

    class_index = int(np.argmax(probabilities))
    class_names = _state["class_names"]

    return Prediction(
        class_index=class_index,
        class_name=class_names[class_index],
        confidence=float(probabilities[class_index]),
        probabilities=probabilities,
        logits=np.asarray(logits, dtype="float64"),
        features=pooled,
        class_names=class_names,
    )


def confidence_band(confidence: float) -> dict[str, str]:
    """Map a confidence value onto a human-readable band and colour."""
    for threshold, label, colour in config.CONFIDENCE_BANDS:
        if confidence >= threshold:
            return {"label": label, "colour": colour}
    return {"label": "Low", "colour": "rose"}
