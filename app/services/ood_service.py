"""
Out-of-distribution (OOD) detection: the layer responsible for *rejecting*
images that do not belong to the ten known corn pest classes.

A softmax classifier is a closed-set model: given a photograph of a car it
will still emit ten probabilities that sum to one, and one of them will be
the largest. Without this layer the system would confidently label unrelated
images as pests. Two independent stages guard against that:

Stage 1 - Relevance gate
    An ImageNet-pretrained MobileNetV2 asks a domain question the pest model
    cannot: "is this even a photograph of an insect or a plant?" Because it
    was trained on 1000 general classes it recognises cars, faces, documents
    and screenshots, and the probability mass it assigns to arthropod and
    plant classes is a training-free relevance score.

Stage 2 - Distributional scoring
    Four complementary scores are computed from the pest model's own output,
    and a vote is taken. Each captures a different failure mode:

    MSP        Hendrycks & Gimpel (2017). Low peak probability = low certainty.
    Entropy    Flat distribution = the evidence is spread across classes.
    Energy     Liu et al. (2020). -logsumexp(logits); OOD inputs produce
               weaker overall activation than in-distribution ones.
    Margin     A small top1-top2 gap means the model cannot separate two
               visually similar pests, which is itself a reason to abstain.

An optional fifth score (Mahalanobis-style distance to the nearest class
    centroid in feature space) was considered but not implemented.

Voting rather than a single threshold is deliberate: any one score can be
fooled by an adversarial or unusual image, but agreement across independent
signals is far harder to trigger by accident, and the per-signal breakdown is
what the explainability panel shows the user.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app import config

logger = logging.getLogger(__name__)

# ImageNet class-index ranges used by the relevance gate. Verified against the
# official imagenet_class_index.json label map.
IMAGENET_INSECT_RANGE = range(300, 327)      # tiger beetle ... lycaenid butterfly
IMAGENET_ARACHNID_RANGE = range(70, 79)      # harvestman ... tick
IMAGENET_OTHER_RELEVANT = {
    110, 111, 112, 113, 114,                 # flatworm, nematode, conch, snail, slug
    936, 937, 938, 939, 941, 943, 944, 945, 946,  # vegetables / leafy crops
    947, 991, 992, 993, 994, 995, 996, 997,  # fungi (field photographs)
    984, 985, 986, 987, 988, 989, 990,       # rapeseed, daisy, corn, acorn, buckeye
    998,                                     # ear (of corn)
    958,                                     # hay
    580,                                     # greenhouse
}
RELEVANT_IMAGENET_CLASSES = (
    set(IMAGENET_INSECT_RANGE) | set(IMAGENET_ARACHNID_RANGE) | IMAGENET_OTHER_RELEVANT
)

_gate_lock = threading.Lock()
_gate: dict[str, Any] = {"model": None, "loaded": False, "error": None}

_stats_cache: dict[str, Any] | None = None


@dataclass
class OODResult:
    """The verdict of the unknown-image rejection layer."""

    is_ood: bool
    status: str                      # "accepted" | "rejected"
    reason: str
    votes: int
    votes_required: int
    signals: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    relevance: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_in_verdict(self) -> float:
        """How decisive the vote was, in [0, 1]."""
        total = max(len(self.signals), 1)
        agreeing = sum(1 for s in self.signals if s["flagged"] == self.is_ood)
        return round(agreeing / total, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ood": self.is_ood,
            "status": self.status,
            "reason": self.reason,
            "votes": self.votes,
            "votes_required": self.votes_required,
            "signals": self.signals,
            "scores": self.scores,
            "relevance": self.relevance,
            "verdict_agreement": self.confidence_in_verdict,
        }


# --------------------------------------------------------------------------
# Stage 1 - relevance gate
# --------------------------------------------------------------------------
def _load_gate() -> None:
    if _gate["loaded"] or _gate["error"]:
        return
    with _gate_lock:
        if _gate["loaded"] or _gate["error"]:
            return
        try:
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf

            logger.info("Loading MobileNetV2 relevance gate (ImageNet weights)")
            _gate["model"] = tf.keras.applications.MobileNetV2(
                weights="imagenet", include_top=True
            )
            _gate["loaded"] = True
        except Exception as exc:
            # The gate is a bonus safety net. If ImageNet weights cannot be
            # downloaded (offline machine), detection still works using the
            # distributional scores alone.
            logger.warning("Relevance gate unavailable: %s", exc)
            _gate["error"] = str(exc)


def relevance_score(image: Image.Image) -> dict[str, Any]:
    """Estimate whether the image plausibly shows an insect or a plant."""
    if not config.RELEVANCE_GATE_ENABLED:
        return {"available": False, "score": None, "labels": []}

    _load_gate()
    if not _gate["loaded"]:
        return {"available": False, "score": None, "labels": [], "error": _gate["error"]}

    import tensorflow as tf

    resized = image.resize((224, 224), Image.BILINEAR)
    array = np.expand_dims(np.asarray(resized, dtype="float32"), axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    probabilities = _gate["model"].predict(array, verbose=0)[0]
    score = float(sum(probabilities[i] for i in RELEVANT_IMAGENET_CLASSES))

    # Take the top-5 indices directly so each label keeps its true class index;
    # decode_predictions alone would lose it, and matching back by probability
    # value breaks on ties.
    top_indices = np.argsort(probabilities)[::-1][:5]
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        probabilities[np.newaxis, :], top=5
    )[0]
    labels = [
        {
            "label": name.replace("_", " "),
            "probability": round(float(probabilities[index]), 4),
            "relevant": int(index) in RELEVANT_IMAGENET_CLASSES,
        }
        for index, (_, name, _) in zip(top_indices, decoded)
    ]

    return {
        "available": True,
        "score": round(score, 4),
        "threshold": config.RELEVANCE_MIN_SCORE,
        "hard_reject_below": config.RELEVANCE_HARD_REJECT,
        "passed": score >= config.RELEVANCE_MIN_SCORE,
        "labels": labels,
    }


# --------------------------------------------------------------------------
# Stage 2 - distributional scores
# --------------------------------------------------------------------------
def normalised_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy scaled to [0, 1] so it is independent of class count."""
    p = np.clip(probabilities, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    return entropy / float(np.log(len(p)))


def free_energy(logits: np.ndarray) -> float:
    """Energy score E(x) = -logsumexp(logits) (Liu et al., 2020)."""
    max_logit = float(np.max(logits))
    stable = max_logit + float(np.log(np.sum(np.exp(logits - max_logit))))
    return -stable


def cosine_distance_to_nearest_centroid(
    features: np.ndarray, centroids: np.ndarray | None
) -> float | None:
    if centroids is None:
        return None
    norm_f = features / (np.linalg.norm(features) + 1e-12)
    norm_c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarities = norm_c @ norm_f
    return float(1.0 - np.max(similarities))


def evaluate(prediction, image: Image.Image | None = None, precomputed_relevance: dict[str, Any] | None = None) -> OODResult:
    """Decide whether a prediction should be accepted or rejected as Unknown."""
    stats: dict[str, Any] = dict(config.OOD_DEFAULTS)
    probabilities = prediction.probabilities
    logits = prediction.logits

    msp = float(np.max(probabilities))
    entropy = normalised_entropy(probabilities)
    energy = free_energy(logits)
    margin = prediction.margin

    signals: list[dict[str, Any]] = [
        {
            "name": "Maximum Softmax Probability",
            "abbreviation": "MSP",
            "value": round(msp, 4),
            "threshold": round(float(stats["msp_threshold"]), 4),
            "rule": "flag when below threshold",
            "flagged": msp < stats["msp_threshold"],
            "description": "Peak class probability. A low peak means no class "
                           "stood out clearly.",
            "reference": "Hendrycks & Gimpel, ICLR 2017",
        },
        {
            "name": "Predictive Entropy",
            "abbreviation": "H",
            "value": round(entropy, 4),
            "threshold": round(float(stats["entropy_threshold"]), 4),
            "rule": "flag when above threshold",
            "flagged": entropy > stats["entropy_threshold"],
            "description": "Spread of the probability distribution, normalised "
                           "to 0-1. High entropy means evidence is scattered.",
            "reference": "Predictive uncertainty baseline",
        },
        {
            "name": "Free Energy",
            "abbreviation": "E",
            "value": round(energy, 4),
            "threshold": round(float(stats["energy_threshold"]), 4),
            "rule": "flag when above threshold",
            "flagged": energy > stats["energy_threshold"],
            "description": "-logsumexp of the logits. Unfamiliar inputs excite "
                           "the network less, giving higher energy.",
            "reference": "Liu et al., NeurIPS 2020",
        },
        {
            "name": "Top-2 Margin",
            "abbreviation": "M",
            "value": round(margin, 4),
            "threshold": round(float(stats["margin_threshold"]), 4),
            "rule": "flag when below threshold",
            "flagged": margin < stats["margin_threshold"],
            "description": "Gap between the best and second-best class. A "
                           "narrow gap means the two are not separable here.",
            "reference": "Decision-boundary proximity",
        },
    ]

    votes = sum(1 for s in signals if s["flagged"])
    votes_required = int(stats["votes_required"])

    # Relevance gate can veto outright, before the vote is even considered.
    relevance = precomputed_relevance if precomputed_relevance is not None else (relevance_score(image) if image is not None else {"available": False})
    gate_hard_reject = (
        relevance.get("available")
        and relevance.get("score") is not None
        and relevance["score"] < config.RELEVANCE_HARD_REJECT
    )
    gate_soft_reject = (
        relevance.get("available")
        and relevance.get("score") is not None
        and not relevance.get("passed")
    )

    if gate_hard_reject:
        is_ood = True
        reason = (
            "The relevance gate found almost no insect or plant content in this "
            "image, so it was rejected before pest classification was trusted."
        )
    elif gate_soft_reject and votes >= 1:
        is_ood = True
        reason = (
            "The image shows weak insect/plant relevance and "
            f"{votes} of {len(signals)} distributional checks also flagged it as "
            "unfamiliar."
        )
    elif votes >= votes_required:
        flagged_names = ", ".join(s["abbreviation"] for s in signals if s["flagged"])
        is_ood = True
        reason = (
            f"{votes} of {len(signals)} out-of-distribution checks failed "
            f"({flagged_names}). The image does not match the known pest "
            "distribution closely enough to name a species."
        )
    else:
        is_ood = False
        reason = (
            f"Passed the unknown-image check: only {votes} of {len(signals)} "
            f"signals flagged, below the threshold of {votes_required}."
        )

    return OODResult(
        is_ood=is_ood,
        status="rejected" if is_ood else "accepted",
        reason=reason,
        votes=votes,
        votes_required=votes_required,
        signals=signals,
        scores={
            "msp": round(msp, 4),
            "entropy": round(entropy, 4),
            "energy": round(energy, 4),
            "margin": round(margin, 4),
        },
        relevance=relevance,
    )
