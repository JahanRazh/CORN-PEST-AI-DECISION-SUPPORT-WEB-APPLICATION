"""
Calibrate the out-of-distribution thresholds against the real training data.

The application ships with literature-derived defaults (config.OOD_DEFAULTS) so
that rejection works before any calibration set exists. Those defaults are
deliberately conservative: they cannot know how peaked *this* model's softmax
actually is. This script measures the in-distribution behaviour and replaces
them with percentile thresholds, and it computes the class feature centroids
that activate the fifth OOD signal (feature-space distance).

Usage
-----
    python scripts/calibrate_ood.py --data path/to/dataset

`--data` should point at a directory of class sub-folders, the same layout the
training script consumed:

    dataset/
        Army Worm-Spodoptera frugiperda/
            img001.jpg
            ...
        Beet Army Worm-Spodoptera exigua/
            ...

Only images of the known classes are needed. Out-of-distribution images are not
required: the thresholds are set so that a chosen fraction of *in-distribution*
images would be rejected, which bounds the false-rejection rate directly.

Method
------
For every image the pest model's own outputs give four scores plus a feature
embedding. Thresholds are placed at the percentile named by --false-reject
(default 5%), i.e. the point below/above which 5% of genuine pest images fall:

    MSP        5th percentile   (flagged when below)
    Entropy    95th percentile  (flagged when above)
    Energy     95th percentile  (flagged when above)
    Margin     5th percentile   (flagged when below)
    Distance   95th percentile  (flagged when above)

Because the OOD layer requires two signals to agree before rejecting, the
per-signal 5% translates into a much smaller joint false-rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.services import model_service, ood_service  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(root: Path, per_class: int | None, seed: int) -> dict[str, list[Path]]:
    """Map each class folder to the image files it contains."""
    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = {}

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [
            f
            for f in sorted(folder.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not files:
            continue
        if per_class is not None and len(files) > per_class:
            files = rng.sample(files, per_class)
        by_class[folder.name] = files

    return by_class


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate OOD thresholds on the training distribution."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="dataset root containing one sub-folder per class",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=120,
        help="images sampled per class (default 120; 0 means use all)",
    )
    parser.add_argument(
        "--false-reject",
        type=float,
        default=5.0,
        help="target per-signal false-rejection rate in percent (default 5)",
    )
    parser.add_argument(
        "--votes-required",
        type=int,
        default=None,
        help="override the number of signals needed to reject (default: keep current)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dry-run", action="store_true", help="report thresholds without writing"
    )
    args = parser.parse_args()

    if not args.data.is_dir():
        print(f"No such directory: {args.data}")
        return 1
    if not 0 < args.false_reject < 50:
        print("--false-reject must be between 0 and 50")
        return 1

    print("=" * 72)
    print("CornGuard AI - OOD threshold calibration")
    print("=" * 72)

    if not model_service.is_ready():
        print(f"\nModel unavailable: {model_service.load_error()}")
        return 1

    class_names = model_service.get_class_names()
    per_class = None if args.per_class in (0, None) else args.per_class
    by_class = collect_images(args.data, per_class, args.seed)

    if not by_class:
        print(f"\nNo class sub-folders with images were found under {args.data}")
        return 1

    unknown = [name for name in by_class if name not in class_names]
    missing = [name for name in class_names if name not in by_class]
    if unknown:
        print(f"\nIgnoring folders that are not model classes: {unknown}")
    if missing:
        print(f"\nWarning: no images found for {len(missing)} class(es): {missing}")

    total_images = sum(len(v) for k, v in by_class.items() if k in class_names)
    print(f"\nScoring {total_images} images across {len(by_class) - len(unknown)} classes")
    print("(this loads the model once and runs one forward pass per image)\n")

    msp_scores: list[float] = []
    entropy_scores: list[float] = []
    energy_scores: list[float] = []
    margin_scores: list[float] = []
    features_by_class: dict[int, list[np.ndarray]] = {}
    correct = 0
    scored = 0
    failed = 0
    started = time.time()

    for class_name, files in by_class.items():
        if class_name not in class_names:
            continue
        class_index = class_names.index(class_name)

        for path in files:
            try:
                image = model_service.load_image(path.read_bytes())
                prediction = model_service.predict(image)
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"  skipped {path.name}: {exc}")
                continue

            msp_scores.append(float(np.max(prediction.probabilities)))
            entropy_scores.append(ood_service.normalised_entropy(prediction.probabilities))
            energy_scores.append(ood_service.free_energy(prediction.logits))
            margin_scores.append(prediction.margin)
            features_by_class.setdefault(class_index, []).append(prediction.features)

            if prediction.class_index == class_index:
                correct += 1
            scored += 1

            if scored % 50 == 0:
                rate = scored / max(time.time() - started, 1e-6)
                print(
                    f"  {scored}/{total_images} images  "
                    f"({rate:.1f}/s, running accuracy {correct / scored * 100:.1f}%)"
                )

    if scored < 30:
        print(f"\nOnly {scored} images were scored; that is too few to calibrate on.")
        return 1

    elapsed = time.time() - started
    print(f"\nScored {scored} images in {elapsed:.0f}s ({failed} skipped)")
    print(f"Reproduced accuracy on this set: {correct / scored * 100:.2f}%")

    # ---- Class centroids in the pooled EfficientNet feature space ----------
    centroids = np.zeros((len(class_names), len(next(iter(features_by_class.values()))[0])))
    covered = 0
    for index in range(len(class_names)):
        vectors = features_by_class.get(index)
        if vectors:
            centroids[index] = np.mean(np.stack(vectors), axis=0)
            covered += 1
    if covered < len(class_names):
        print(
            f"\nWarning: centroids computed for {covered}/{len(class_names)} classes. "
            "Classes without images get a zero centroid and will never be nearest."
        )

    distances = [
        ood_service.cosine_distance_to_nearest_centroid(vector, centroids)
        for vectors in features_by_class.values()
        for vector in vectors
    ]

    # ---- Percentile thresholds --------------------------------------------
    low = args.false_reject
    high = 100.0 - args.false_reject

    thresholds = {
        "msp_threshold": float(np.percentile(msp_scores, low)),
        "entropy_threshold": float(np.percentile(entropy_scores, high)),
        "energy_threshold": float(np.percentile(energy_scores, high)),
        "margin_threshold": float(np.percentile(margin_scores, low)),
        "feature_distance_threshold": float(np.percentile(distances, high)),
    }

    print(f"\nThresholds at a {args.false_reject:g}% per-signal false-rejection rate")
    print("-" * 72)
    print(f"{'signal':<26}{'default':>12}{'calibrated':>14}{'in-dist median':>18}")
    medians = {
        "msp_threshold": np.median(msp_scores),
        "entropy_threshold": np.median(entropy_scores),
        "energy_threshold": np.median(energy_scores),
        "margin_threshold": np.median(margin_scores),
        "feature_distance_threshold": np.median(distances),
    }
    for key, value in thresholds.items():
        default = config.OOD_DEFAULTS.get(key)
        label = key.replace("_threshold", "").replace("_", " ")
        default_text = f"{default:.4f}" if default is not None else "n/a"
        print(f"{label:<26}{default_text:>12}{value:>14.4f}{medians[key]:>18.4f}")

    # How many in-distribution images the new thresholds would have rejected.
    votes_required = args.votes_required or int(config.OOD_DEFAULTS["votes_required"])
    rejected = 0
    for msp, entropy, energy, margin, distance in zip(
        msp_scores, entropy_scores, energy_scores, margin_scores, distances
    ):
        votes = sum(
            (
                msp < thresholds["msp_threshold"],
                entropy > thresholds["entropy_threshold"],
                energy > thresholds["energy_threshold"],
                margin < thresholds["margin_threshold"],
                distance > thresholds["feature_distance_threshold"],
            )
        )
        rejected += votes >= votes_required

    print("-" * 72)
    print(
        f"With {votes_required} agreeing signals required, {rejected}/{scored} "
        f"({rejected / scored * 100:.2f}%) of these known pest images would be "
        "rejected as Unknown."
    )

    if args.dry_run:
        print("\n--dry-run given: model/ood_stats.npz was not written.")
        return 0

    config.OOD_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.OOD_STATS_PATH,
        centroids=centroids,
        class_names=np.array(class_names, dtype=object),
        calibration_images=scored,
        false_reject_percent=args.false_reject,
        **{k: np.float64(v) for k, v in thresholds.items()},
    )
    print(f"\nWrote {config.OOD_STATS_PATH}")

    # A human-readable companion so the thresholds can be cited without numpy.
    summary_path = config.MODEL_DIR / "ood_calibration.json"
    summary_path.write_text(
        json.dumps(
            {
                "images_scored": scored,
                "classes": len(class_names),
                "accuracy_on_calibration_set": round(correct / scored * 100, 2),
                "false_reject_percent": args.false_reject,
                "votes_required": votes_required,
                "in_distribution_rejection_rate": round(rejected / scored * 100, 2),
                "thresholds": {k: round(v, 6) for k, v in thresholds.items()},
                "defaults": {
                    k: v for k, v in config.OOD_DEFAULTS.items() if k in thresholds
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print("\nRestart the server (or POST /api/reload) to pick up the new thresholds.")
    print("The feature-distance signal is now active and will appear on result pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
