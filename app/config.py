"""
Central configuration for the Corn Pest AI Decision Support System.

Every tunable value lives here so the research thresholds (OOD cut-offs,
confidence bands, severity weights) can be reported in a dissertation without
digging through the service code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project layout -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# .env must be loaded before cloudinary is imported anywhere, because the
# cloudinary package reads CLOUDINARY_URL at import time.
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = MODEL_DIR / "result"

MODEL_PATH = MODEL_DIR / "best_corn_pest_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
OOD_STATS_PATH = MODEL_DIR / "ood_stats.npz"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
KNOWLEDGE_BASE_PATH = DATA_DIR / "Corn_Pest_Rule_Based_Pesticide_Recommendation_System.xlsx"

# Model ---------------------------------------------------------------------
IMAGE_SIZE = 224

# The Keras model already contains Rescaling + Normalization layers, so images
# are fed as raw float32 in the 0-255 range. Do NOT divide by 255 here.
MODEL_EXPECTS_RAW_PIXELS = True

# Uploads -------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Out-of-distribution (OOD) detection thresholds
#
# Defaults are derived from the reported test performance (93.95% accuracy,
# test loss 0.2207) and from standard baselines in the OOD literature:
#   - MSP        Hendrycks & Gimpel (2017), "A Baseline for Detecting
#                Misclassified and Out-of-Distribution Examples"
#   - Energy     Liu et al. (2020), "Energy-based Out-of-distribution Detection"
#   - Entropy    Predictive-uncertainty baseline over the softmax distribution
#
# Run scripts/calibrate_ood.py against the training set to replace these with
# percentile-calibrated values written to model/ood_stats.npz.
# ---------------------------------------------------------------------------
OOD_DEFAULTS = {
    # Maximum softmax probability. Below this the prediction is not trusted.
    "msp_threshold": 0.85,
    # Shannon entropy of the softmax vector, normalised to [0, 1] by log(K).
    # Above this the distribution is too flat to commit to one class.
    "entropy_threshold": 0.45,
    # Free energy = -logsumexp(logits). Higher energy means less in-distribution.
    "energy_threshold": -3.0,
    # Gap between the top-1 and top-2 probabilities. A small margin means the
    # model is torn between two classes.
    "margin_threshold": 0.15,
    # Cosine distance to the nearest class feature centroid (calibrated only).
    "feature_distance_threshold": 0.75,
    # Number of individual OOD signals that must fire before the image is
    # rejected as Unknown.
    "votes_required": 2,
}

# Relevance gate ------------------------------------------------------------
# A pre-filter that answers "is this even a photograph of an insect or a
# plant?" before the pest classifier is consulted. Backed by an
# ImageNet-pretrained MobileNetV2, so it needs no extra training data.
RELEVANCE_GATE_ENABLED = True
RELEVANCE_MIN_SCORE = 0.045   # summed probability mass on insect/plant classes
RELEVANCE_HARD_REJECT = 0.010  # below this the image is rejected outright

# Confidence presentation bands --------------------------------------------
CONFIDENCE_BANDS = [
    (0.90, "Very High", "emerald"),
    (0.75, "High", "green"),
    (0.60, "Moderate", "amber"),
    (0.00, "Low", "rose"),
]


# Flask ---------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "corn-pest-research-dev-key-change-in-production")

# Cloud services ------------------------------------------------------------
FIREBASE_CREDENTIALS_PATH = Path(
    os.getenv("FIREBASE_CREDENTIALS", BASE_DIR / "serviceAccountKey.json")
)
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "detections")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "corn_pest_detections")
CLOUD_TIMEOUT_SECONDS = 30

# Number of history records pulled for the dashboard / history page.
HISTORY_PAGE_SIZE = 24
