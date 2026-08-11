"""
Build model/model_metrics.json from the training artefacts in model/result/.

The dashboard can parse accuracy_report.txt directly, but doing so on every
cold start means the report format becomes a runtime dependency. This script
does the parsing once and writes a stable JSON export that the metrics service
prefers when present:

    python scripts/setup_artifacts.py

Add --check to validate the artefacts without writing anything, and --force to
overwrite an existing export.

The numbers are *not* recomputed: they are the figures produced by the training
run, so what the dashboard shows is exactly what is reported in the write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow "python scripts/setup_artifacts.py" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.services import metrics_service  # noqa: E402


def _fail(message: str) -> None:
    print(f"  [FAIL] {message}")


def _ok(message: str) -> None:
    print(f"  [ ok ] {message}")


def _warn(message: str) -> None:
    print(f"  [warn] {message}")


def check_inputs() -> list[str]:
    """Report on every artefact the application expects to find."""
    problems: list[str] = []

    print("\nRequired artefacts")
    if config.MODEL_PATH.exists():
        size_mb = config.MODEL_PATH.stat().st_size / (1024 * 1024)
        _ok(f"model             {config.MODEL_PATH.name} ({size_mb:.1f} MB)")
    else:
        problems.append(f"missing {config.MODEL_PATH}")
        _fail(f"model             not found at {config.MODEL_PATH}")

    if config.CLASS_NAMES_PATH.exists():
        names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        _ok(f"class names       {len(names)} classes")
    else:
        problems.append(f"missing {config.CLASS_NAMES_PATH}")
        _fail(f"class names       not found at {config.CLASS_NAMES_PATH}")

    if config.KNOWLEDGE_BASE_PATH.exists():
        _ok(f"knowledge base    {config.KNOWLEDGE_BASE_PATH.name}")
    else:
        problems.append(f"missing {config.KNOWLEDGE_BASE_PATH}")
        _fail(f"knowledge base    not found at {config.KNOWLEDGE_BASE_PATH}")

    if metrics_service.REPORT_PATH.exists():
        _ok(f"accuracy report   {metrics_service.REPORT_PATH.name}")
    else:
        problems.append(f"missing {metrics_service.REPORT_PATH}")
        _fail(f"accuracy report   not found at {metrics_service.REPORT_PATH}")

    print("\nOptional artefacts")
    for label, path in (
        ("training history ", metrics_service.HISTORY_PATH),
        ("accuracy graph   ", config.RESULT_DIR / "accuracy_graph.png"),
        ("loss graph       ", config.RESULT_DIR / "loss_graph.png"),
        ("OOD calibration  ", config.OOD_STATS_PATH),
    ):
        if path.exists():
            _ok(f"{label} {path.name}")
        else:
            _warn(f"{label} not present ({path.name})")

    print("\nCredentials")
    if (config.BASE_DIR / ".env").exists():
        _ok(".env             present")
    else:
        _warn(".env             not found - Cloudinary uploads will be skipped")
    if config.FIREBASE_CREDENTIALS_PATH.exists():
        _ok(f"service account  {config.FIREBASE_CREDENTIALS_PATH.name}")
    else:
        _warn("service account  not found - history will not persist")

    return problems


def build_metrics() -> dict:
    """Parse the text report into the structure the dashboard consumes."""
    text = metrics_service.REPORT_PATH.read_text(encoding="utf-8", errors="replace")

    headline = metrics_service._parse_headline(text)
    per_class = metrics_service._parse_per_class(text)
    averages = metrics_service._parse_averages(text)
    matrix = metrics_service._parse_confusion(text)
    history = metrics_service._parse_history()

    class_names = [row["class_name"] for row in per_class]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "model/result/accuracy_report.txt",
        "model": "EfficientNetB0 (transfer learning + fine tuning)",
        "headline": headline,
        "per_class": per_class,
        "averages": averages,
        "confusion_matrix": matrix,
        "class_names": class_names,
        "history": history,
        "graphs": {
            "accuracy": (config.RESULT_DIR / "accuracy_graph.png").exists(),
            "loss": (config.RESULT_DIR / "loss_graph.png").exists(),
        },
    }


def validate(metrics: dict) -> list[str]:
    """Sanity-check the parse before it is trusted by the dashboard."""
    issues: list[str] = []

    per_class = metrics["per_class"]
    matrix = metrics["confusion_matrix"]
    headline = metrics["headline"]

    if not per_class:
        issues.append("no per-class rows were parsed from the classification report")
    if not matrix:
        issues.append("no confusion matrix was parsed")
    if per_class and matrix and len(per_class) != len(matrix):
        issues.append(
            f"{len(per_class)} classes but a {len(matrix)}x{len(matrix)} confusion matrix"
        )
    if headline.get("num_classes") and per_class:
        if headline["num_classes"] != len(per_class):
            issues.append(
                f"report header says {headline['num_classes']} classes, "
                f"{len(per_class)} were parsed"
            )

    # The confusion matrix diagonal should reproduce the reported accuracy.
    if matrix:
        total = sum(sum(row) for row in matrix)
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        if total:
            derived = correct / total * 100
            reported = headline.get("accuracy")
            if reported is not None and abs(derived - reported) > 0.5:
                issues.append(
                    f"confusion matrix implies {derived:.2f}% accuracy but the "
                    f"report states {reported:.2f}%"
                )

    # Class names in the report must line up with the names the model emits.
    if config.CLASS_NAMES_PATH.exists() and per_class:
        model_names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        report_names = metrics["class_names"]
        if sorted(model_names) != sorted(report_names):
            only_model = set(model_names) - set(report_names)
            only_report = set(report_names) - set(model_names)
            issues.append(
                "class names differ between class_names.json and the report "
                f"(model only: {sorted(only_model)}, report only: {sorted(only_report)})"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare model/model_metrics.json for the dashboard."
    )
    parser.add_argument(
        "--check", action="store_true", help="verify artefacts without writing"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing export"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("CornGuard AI - artefact setup")
    print("=" * 72)

    problems = check_inputs()

    if not metrics_service.REPORT_PATH.exists():
        print("\nCannot build metrics without the accuracy report. Run the training")
        print("script first, or copy accuracy_report.txt into model/result/.")
        return 1

    print("\nParsing the evaluation report")
    metrics = build_metrics()
    headline = metrics["headline"]
    _ok(f"classes           {len(metrics['per_class'])}")
    _ok(f"confusion matrix  {len(metrics['confusion_matrix'])} rows")
    _ok(f"training epochs   {len(metrics['history'])}")
    if headline.get("accuracy") is not None:
        _ok(
            "test performance  "
            f"accuracy {headline['accuracy']:.2f}%  "
            f"precision {headline.get('precision', 0):.2f}%  "
            f"recall {headline.get('recall', 0):.2f}%  "
            f"F1 {headline.get('f1_score', 0):.2f}%"
        )

    issues = validate(metrics)
    if issues:
        print("\nConsistency checks")
        for issue in issues:
            _fail(issue)
    else:
        print("\nConsistency checks")
        _ok("report, confusion matrix and class names agree")

    if args.check:
        print("\n--check given: nothing was written.")
        return 1 if (problems or issues) else 0

    if config.METRICS_PATH.exists() and not args.force:
        print(f"\n{config.METRICS_PATH} already exists. Re-run with --force to replace it.")
        return 0

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {config.METRICS_PATH}")
    print("The dashboard will now read this file instead of re-parsing the report.")

    if problems:
        print("\nOutstanding problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
