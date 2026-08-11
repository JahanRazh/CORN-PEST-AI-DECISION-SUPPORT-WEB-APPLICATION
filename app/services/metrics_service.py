"""
Metrics service: exposes the trained model's evaluation results.

Reads the artefacts produced by the training pipeline in model/result/
(accuracy_report.txt, training_history.csv, training_summary.txt) and turns
them into structured data for the performance dashboard: per-class precision,
recall and F1, the confusion matrix, and the training/validation curves.

Parsing the report rather than re-evaluating means the dashboard shows the
same figures reported in the dissertation, and loads instantly.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
from typing import Any

from app import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, Any] | None = None

REPORT_PATH = config.RESULT_DIR / "accuracy_report.txt"
HISTORY_PATH = config.RESULT_DIR / "training_history.csv"
SUMMARY_PATH = config.RESULT_DIR / "training_summary.txt"


def _parse_headline(text: str) -> dict[str, Any]:
    """Pull the overall test figures out of the report header."""
    def grab(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    return {
        "test_loss": grab(r"Test Loss\s*:\s*([0-9.]+)"),
        "accuracy": grab(r"Accuracy\s*:\s*([0-9.]+)%"),
        "precision": grab(r"Precision\s*:\s*([0-9.]+)%"),
        "recall": grab(r"Recall\s*:\s*([0-9.]+)%"),
        "f1_score": grab(r"F1 Score\s*:\s*([0-9.]+)%"),
        "num_classes": int(m.group(1)) if (m := re.search(r"Classes\s*:\s*(\d+)", text)) else None,
        "image_size": m2.group(1) if (m2 := re.search(r"Image Size\s*:\s*([0-9 x]+)", text)) else None,
    }


def _parse_per_class(text: str) -> list[dict[str, Any]]:
    """Parse the sklearn classification_report block."""
    rows: list[dict[str, Any]] = []
    section = text.split("CLASSIFICATION REPORT")
    if len(section) < 2:
        return rows

    for line in section[1].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=") or stripped.startswith("precision"):
            continue
        if stripped.startswith(("accuracy", "macro avg", "weighted avg")):
            continue
        if "CONFUSION" in stripped:
            break
        # "Class Name   0.9211    0.9211    0.9211       152"
        match = re.match(
            r"^(.*?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", stripped
        )
        if not match:
            continue
        rows.append(
            {
                "class_name": match.group(1).strip(),
                "precision": round(float(match.group(2)) * 100, 2),
                "recall": round(float(match.group(3)) * 100, 2),
                "f1_score": round(float(match.group(4)) * 100, 2),
                "support": int(match.group(5)),
            }
        )
    return rows


def _parse_averages(text: str) -> dict[str, Any]:
    averages: dict[str, Any] = {}
    for key, label in (("macro avg", "macro"), ("weighted avg", "weighted")):
        match = re.search(
            rf"{key}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text
        )
        if match:
            averages[label] = {
                "precision": round(float(match.group(1)) * 100, 2),
                "recall": round(float(match.group(2)) * 100, 2),
                "f1_score": round(float(match.group(3)) * 100, 2),
                "support": int(match.group(4)),
            }
    return averages


def _parse_confusion(text: str) -> list[list[int]]:
    """Parse the numpy-formatted confusion matrix at the end of the report."""
    section = text.split("CONFUSION MATRIX")
    if len(section) < 2:
        return []
    block = section[-1]
    start = block.find("[[")
    if start == -1:
        return []
    end = block.find("]]", start)
    if end == -1:
        return []
    body = block[start + 2 : end]

    matrix: list[list[int]] = []
    for row in body.split("],"):
        numbers = re.findall(r"-?\d+", row)
        if numbers:
            matrix.append([int(n) for n in numbers])
    return matrix


def _parse_history() -> list[dict[str, float]]:
    if not HISTORY_PATH.exists():
        return []
    history: list[dict[str, float]] = []
    try:
        with open(HISTORY_PATH, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                entry: dict[str, float] = {}
                for key, value in row.items():
                    if key is None or value in (None, ""):
                        continue
                    try:
                        entry[key.strip()] = float(value)
                    except ValueError:
                        continue
                if entry:
                    history.append(entry)
    except Exception:
        logger.exception("Could not parse training history")
    return history


def load() -> dict[str, Any]:
    """Load and cache all evaluation artefacts."""
    global _cache
    if _cache is not None:
        return _cache

    with _lock:
        if _cache is not None:
            return _cache

        data: dict[str, Any] = {
            "available": False,
            "error": None,
            "headline": {},
            "per_class": [],
            "averages": {},
            "confusion_matrix": [],
            "class_names": [],
            "history": [],
            "graphs": {},
        }

        try:
            # A JSON export, if scripts/setup_artifacts.py has produced one,
            # takes priority over re-parsing the text report.
            if config.METRICS_PATH.exists():
                with open(config.METRICS_PATH, encoding="utf-8") as handle:
                    data.update(json.load(handle))
                    data["available"] = True
                    data["source"] = "model/model_metrics.json"
            elif REPORT_PATH.exists():
                text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
                data["headline"] = _parse_headline(text)
                data["per_class"] = _parse_per_class(text)
                data["averages"] = _parse_averages(text)
                data["confusion_matrix"] = _parse_confusion(text)
                data["class_names"] = [r["class_name"] for r in data["per_class"]]
                data["available"] = True
                data["source"] = "model/result/accuracy_report.txt"
            else:
                data["error"] = (
                    f"No evaluation report found at {REPORT_PATH}. Run the "
                    "training script to generate it."
                )

            data["history"] = _parse_history()
            data["graphs"] = {
                "accuracy": (config.RESULT_DIR / "accuracy_graph.png").exists(),
                "loss": (config.RESULT_DIR / "loss_graph.png").exists(),
            }
        except Exception as exc:
            logger.exception("Metrics load failed")
            data["error"] = str(exc)

        _cache = data
        return data


def reload() -> dict[str, Any]:
    global _cache
    with _lock:
        _cache = None
    return load()


def confusion_insights() -> list[dict[str, Any]]:
    """Identify the most-confused class pairs.

    Surfacing these on the dashboard connects the model's weaknesses to the
    OOD layer: the pairs listed here are exactly the ones the top-2 margin
    signal is designed to abstain on.
    """
    data = load()
    matrix = data.get("confusion_matrix") or []
    names = data.get("class_names") or []
    if not matrix or len(names) != len(matrix):
        return []

    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(matrix):
        total = sum(row) or 1
        for j, count in enumerate(row):
            if i == j or count == 0:
                continue
            pairs.append(
                {
                    "true_class": names[i],
                    "predicted_class": names[j],
                    "count": count,
                    "rate": round(count / total * 100, 1),
                }
            )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:8]


def per_class_extremes() -> dict[str, Any]:
    """Best and worst performing classes by F1."""
    rows = load().get("per_class") or []
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda r: r["f1_score"])
    return {"weakest": ordered[:3], "strongest": ordered[-3:][::-1]}
