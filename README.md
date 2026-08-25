# 🌽 CornGuard AI — Corn Pest Decision Support System

> **A research-grade AI web application that identifies corn pests from uploaded photographs and delivers expert-backed pesticide & IPM recommendations in seconds.**

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [AI & Detection Pipeline](#-ai--detection-pipeline)
4. [Knowledge Base & Expert Mapping](#-knowledge-base--expert-mapping)
5. [Out-of-Distribution (OOD) Detection](#-out-of-distribution-ood-detection)
6. [Technology Stack](#-technology-stack)
7. [Project Structure](#-project-structure)
8. [Supported Corn Pests (10 Classes)](#-supported-corn-pests-10-classes)
9. [Model Performance](#-model-performance)
10. [Cloud Services](#-cloud-services)
11. [Pages & Features](#-pages--features)
12. [REST API Reference](#-rest-api-reference)
13. [Environment Configuration](#-environment-configuration)
14. [Installation & Running Locally](#-installation--running-locally)
15. [Research Context](#-research-context)

---

## 🔍 Project Overview

**CornGuard AI** is a full-stack web application built as a research project at SLIIT (Sri Lanka Institute of Information Technology). It combines deep learning image classification, agricultural domain expertise, and a rule-based knowledge base to assist farmers, agronomists, and researchers in:

- **Identifying corn pests** from a photo of the insect or damaged plant
- **Understanding damage symptoms** and pest behaviour
- **Receiving IPM (Integrated Pest Management) recommendations** with specific active ingredients, biological control options, IRAC mode-of-action groups, treatment thresholds, and application timing
- **Safely rejecting irrelevant images** (cars, faces, documents) rather than misclassifying them

The system is intentionally **explainable**: every result page shows the full pipeline trace — which AI stage made which decision and why — fulfilling the "decision reasoning" requirement of an AI-based decision support system.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser (User)                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │  HTTP (Jinja2-rendered HTML pages + JSON API)
┌──────────────────────▼──────────────────────────────────────────┐
│                    Flask Web Server (run.py)                     │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────────────┐    │
│  │   Page Routes        │   │     JSON API Routes          │    │
│  │   (main.py)          │   │     (api.py)                 │    │
│  │                      │   │                              │    │
│  │  /               ▼   │   │  /api/detect                 │    │
│  │  /detect         ▼   │   │  /api/metrics                │    │
│  │  /result/<id>    ▼   │   │  /api/pests                  │    │
│  │  /history        ▼   │   │  /api/status                 │    │
│  │  /knowledge-base ▼   │   │  /api/history                │    │
│  │  /dashboard      ▼   │   │  /api/reload                 │    │
│  │  /about          ▼   │   │                              │    │
│  └──────────┬───────────┘   └──────────────────────────────┘    │
│             │                                                    │
│  ┌──────────▼──────────────────────────────────────────────┐    │
│  │            Detection Pipeline (detection_pipeline.py)   │    │
│  │                                                         │    │
│  │  Stage 1 → Image Validation (size, format, decodable)   │    │
│  │  Stage 2 → Relevance Gate   (MobileNetV2 + ImageNet)    │    │
│  │  Stage 3 → Pest Classifier  (EfficientNetB0)            │    │
│  │  Stage 4 → OOD Rejection    (4-signal voting system)    │    │
│  │  Stage 5 → Expert Mapping   (KB vocabulary bridge)      │    │
│  │  Stage 6 → Cloud Persistence (Cloudinary + Firestore)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       Cloud Services                             │
│                                                                  │
│   ┌─────────────────────┐    ┌─────────────────────────────┐    │
│   │  Cloudinary CDN     │    │  Google Cloud Firestore      │    │
│   │  (image storage)    │    │  (detection records &        │    │
│   │                     │    │   dashboard aggregates)      │    │
│   └─────────────────────┘    └─────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI & Detection Pipeline

Every image upload runs through a **6-stage sequential pipeline**. Each stage appends to a *pipeline trace* — an audit trail shown to the user on the result page.

### Stage 1 — Image Validation
- Checks file extension (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`)
- Enforces a 10 MB size limit
- Decodes the file as an image using Pillow; rejects corrupted files
- Corrects EXIF orientation so phone photos are upright

### Stage 2 — ImageNet Relevance Gate *(MobileNetV2)*
A pre-trained **MobileNetV2** (ImageNet-1000) answers the domain question the pest classifier cannot: *"Is this even a photograph of an insect or a plant?"*

- Sums the softmax probability mass over ~90 ImageNet classes covering insects, arachnids, arthropods, crops, vegetables, fungi, and corn-specific classes
- If the score is below `RELEVANCE_HARD_REJECT = 0.010`, the image is **immediately rejected** without invoking the pest model — saving compute and preventing a confident wrong label
- Threshold `RELEVANCE_MIN_SCORE = 0.045` distinguishes a *soft pass* from a hard rejection
- This gate is **training-free**: no corn pest data was needed to teach it to reject cars, faces, or documents

### Stage 3 — Pest Classification *(EfficientNetB0)*
The core deep learning model:
- Architecture: **EfficientNetB0** with transfer learning from ImageNet, followed by fine-tuning on the corn pest dataset
- Input: 224×224 RGB image (raw 0–255 pixels; the model contains its own Rescaling/Normalization layers)
- Output: 10-class softmax probability vector
- A **multi-output inference wrapper** extracts three tensors in one forward pass:
  - `probabilities` — softmax vector shown to the user
  - `logits` — raw pre-softmax activations for energy-based OOD scoring
  - `features` — 1280-d pooled EfficientNet embedding for distance-based OOD scoring

### Stage 4 — Out-of-Distribution Rejection *(OOD Service)*
See the dedicated section below.

### Stage 5 — Expert Mapping
Bridges the gap between the model's training folder names and the agricultural vocabulary used in the knowledge base. See the dedicated section below.

### Stage 6 — Cloud Persistence
- Uploads the image to **Cloudinary** (returns a CDN URL stored with the record)
- Saves the full detection record to **Google Cloud Firestore**
- Both operations are **fault-tolerant**: a cloud failure degrades gracefully — the user still sees their result; the UI reports the storage status honestly

---

## 📚 Knowledge Base & Expert Mapping

### Knowledge Base (`knowledge_base.py`)
Backed by a curated **Excel workbook** (`Corn_Pest_Rule_Based_Pesticide_Recommendation_System.xlsx`) authored by agricultural experts. Each row is one pest profile containing:

| Field | Description |
|---|---|
| Common Name | Agronomic common name (e.g., "Fall Armyworm") |
| Scientific Name | Binomial classification |
| Pest Group | Taxonomic group (e.g., Lepidoptera, Hemiptera) |
| Major Damage Symptoms | Field-observable damage description |
| Recommended Active Ingredients | Chemical pesticide options |
| IRAC Mode of Action | Insecticide Resistance Action Committee group codes |
| Biological Control Options | Predators, parasitoids, biopesticides |
| Application Timing | When to apply relative to crop growth stage |
| IPM Recommendation | Integrated Pest Management guidance |
| Environmental Consideration | Environmental safety notes |
| Treatment Guideline | Economic threshold for intervention |

The workbook is loaded **once at start-up** into memory with thread-safe locking. It can be reloaded without restarting the server via `POST /api/reload`.

### Expert Mapping Layer (`expert_mapping.py`)
Three different vocabularies must be reconciled:
1. Model training folder names: `"Army Worm-Spodoptera frugiperda"`
2. Knowledge base common names: `"Fall Armyworm"`
3. Product reference names: `"Armyworm species"`

The **CLASS_MAPPING** table provides a curated bridge between these vocabularies for all 10 trained classes. A **fuzzy fallback matcher** handles future retraining cases by tokenising the class name and finding the best overlap with knowledge base entries.

---

## 🛡 Out-of-Distribution (OOD) Detection

This is the most novel research component. A softmax classifier is a **closed-set model**: given a photo of a car, it still emits 10 probabilities that sum to one. Without OOD detection, the system would confidently mislabel unrelated images.

### Four Independent OOD Signals

| Signal | Abbreviation | Source | What it detects |
|---|---|---|---|
| Maximum Softmax Probability | **MSP** | Hendrycks & Gimpel (2017) | Low peak probability = uncertain prediction |
| Shannon Entropy | **H** | Standard | Flat distribution = evidence spread across all classes |
| Free Energy | **E** | Liu et al. (2020) | `-logsumexp(logits)`; OOD inputs produce weaker overall activation |
| Top-1 / Top-2 Margin | **M** | Standard | Small gap = model torn between two similar pests |

### Voting System
- Each signal casts a binary vote: *flagged* or *not flagged*
- Configurable number of votes required to reject (`votes_required = 1` default)
- **Any single signal** firing is sufficient to abstain — this is deliberately conservative for a safety-critical agricultural context
- The per-signal breakdown is shown in the **explainability panel** on the result page

### Calibrated Thresholds
Running `scripts/calibrate_ood.py` against the training set derives **percentile-calibrated thresholds** (stored in `model/ood_stats.npz`), replacing the literature-default values. A fifth signal — **Mahalanobis-style cosine distance** to the nearest class centroid in feature space — activates only after calibration.

### Configurable Thresholds (in `config.py`)

```python
OOD_DEFAULTS = {
    "msp_threshold":              0.85,   # below → rejected
    "entropy_threshold":          0.45,   # above → rejected
    "energy_threshold":          -3.0,    # above → rejected
    "margin_threshold":           0.15,   # below → rejected
    "feature_distance_threshold": 0.75,   # above → rejected (calibrated only)
    "votes_required":             1,      # votes needed to reject
}
```

---

## 🛠 Technology Stack

### Backend
| Component | Technology |
|---|---|
| Web Framework | **Flask** (Python) |
| Deep Learning | **TensorFlow / Keras** |
| Primary Model | **EfficientNetB0** (transfer learning + fine-tuning) |
| Relevance Gate | **MobileNetV2** (ImageNet pretrained, no additional training) |
| Image Processing | **Pillow (PIL)** |
| Numerical Computation | **NumPy** |
| Knowledge Base Parsing | **Pandas** (reads `.xlsx`) |
| Environment Config | **python-dotenv** |

### Frontend
| Component | Technology |
|---|---|
| Templating Engine | **Jinja2** (server-side rendering) |
| Styling | **Tailwind CSS** (CDN) |
| Interactivity | **Vanilla JavaScript** |
| Charts / Visualisations | Chart.js (model performance dashboard) |

### Cloud & Persistence
| Component | Technology |
|---|---|
| Image Storage | **Cloudinary** (CDN-backed cloud image hosting) |
| Database | **Google Cloud Firestore** (NoSQL document store) |
| Auth / Credentials | **Firebase Admin SDK** (service account key) |

### Development & Tooling
| Component | Technology |
|---|---|
| Language | **Python 3.10+** |
| Package Management | `pip` + `requirements.txt` |
| Entry Point | `run.py` (Werkzeug dev server) |
| OOD Calibration | `scripts/calibrate_ood.py` |
| Training Script | `scripts/corn_pest_detection.py` |

---

## 📁 Project Structure

```
CORN PEST AI DECISION SUPPORT WEB APPLICATION/
│
├── run.py                          # App entry point (CornGuard AI startup banner)
├── requirements.txt                # All Python dependencies
├── .env                            # Secret keys & cloud credentials (not in git)
├── serviceAccountKey.json          # Firebase service account (not in git)
│
├── app/
│   ├── __init__.py                 # Flask app factory (registers blueprints)
│   ├── config.py                   # All tunable thresholds and paths
│   │
│   ├── routes/
│   │   ├── main.py                 # Page routes (/, /detect, /result, /history, etc.)
│   │   └── api.py                  # JSON API (/api/detect, /api/metrics, etc.)
│   │
│   ├── services/
│   │   ├── detection_pipeline.py   # Orchestrates all 6 pipeline stages
│   │   ├── model_service.py        # EfficientNetB0 loader & inference
│   │   ├── ood_service.py          # OOD detection (relevance gate + voting)
│   │   ├── expert_mapping.py       # AI class <-> KB vocabulary bridge
│   │   ├── knowledge_base.py       # Excel workbook loader & pest profiles
│   │   ├── cloud_store.py          # Cloudinary + Firestore persistence
│   │   └── metrics_service.py      # Model performance metrics loader
│   │
│   ├── templates/
│   │   ├── base.html               # Shared layout, nav, flash messages
│   │   ├── index.html              # Upload page (drag-and-drop + preview)
│   │   ├── result.html             # Full result: prediction, OOD, KB, trace
│   │   ├── history.html            # Detection history from Firestore
│   │   ├── dashboard.html          # Model performance metrics & charts
│   │   ├── knowledge.html          # Pest knowledge base listing
│   │   ├── pest_detail.html        # Individual pest profile page
│   │   ├── about.html              # System info, OOD config, class list
│   │   └── error.html              # 404 / 500 error pages
│   │
│   └── static/
│       ├── css/                    # Custom stylesheets
│       ├── js/                     # Client-side scripts
│       ├── images/                 # Static image assets
│       └── videos/                 # Background / demo videos
│
├── model/
│   ├── best_corn_pest_model.keras  # Trained EfficientNetB0 (~20 MB)
│   ├── class_names.json            # Ordered list of 10 class names
│   ├── model_metrics.json          # Test metrics, per-class scores, confusion matrix
│   ├── ood_stats.npz               # Calibrated OOD thresholds (generated by calibrate_ood.py)
│   └── result/
│       ├── accuracy_graph.png      # Training accuracy curve
│       └── loss_graph.png          # Training loss curve
│
├── data/
│   └── Corn_Pest_Rule_Based_Pesticide_Recommendation_System.xlsx
│
└── scripts/
    ├── corn_pest_detection.py      # Model training script (EfficientNetB0)
    ├── calibrate_ood.py            # OOD threshold calibration script
    └── setup_artifacts.py          # Artifact setup helper
```

---

## 🐛 Supported Corn Pests (10 Classes)

| # | Common Name | Scientific Name | Pest Group |
|---|---|---|---|
| 1 | Fall Armyworm | *Spodoptera frugiperda* | Lepidoptera |
| 2 | Beet Armyworm | *Spodoptera exigua* | Lepidoptera |
| 3 | Black Cutworm | *Agrotis ipsilon* | Lepidoptera |
| 4 | Corn Aphid | *Rhopalosiphum maidis* | Hemiptera |
| 5 | Corn Borer (Asian) | *Ostrinia furnacalis* | Lepidoptera |
| 6 | Corn Earworm | *Helicoverpa armigera* | Lepidoptera |
| 7 | Corn Grasshopper | *Oxya chinensis* | Orthoptera |
| 8 | Flea Beetle | *Phyllotreta spp.* | Coleoptera |
| 9 | White Grub | *Holotrichia spp.* | Coleoptera |
| 10 | Wireworm | *Agriotes lineatus* | Coleoptera |

Any image that does not clearly depict one of these ten pests is **deliberately rejected** by the OOD layer rather than forced into the nearest class.

---

## 📊 Model Performance

The EfficientNetB0 classifier was evaluated on a held-out test set of **1,470 images**.

### Headline Metrics

| Metric | Score |
|---|---|
| **Test Accuracy** | **93.95%** |
| Test Loss | 0.2207 |
| Macro Precision | 93.15% |
| Macro Recall | 93.93% |
| Macro F1-Score | 93.45% |
| Weighted F1-Score | 93.93% |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Army Worm (*S. frugiperda*) | 92.11% | 92.11% | 92.11% | 152 |
| Beet Army Worm (*S. exigua*) | 86.72% | 78.72% | 82.53% | 141 |
| Black Cut Worm (*A. ypsilon*) | 77.50% | 92.54% | 84.35% | 67 |
| Corn Aphid (*R. maidis*) | 97.40% | 96.77% | 97.09% | 155 |
| Corn Borer (*O. furnacalis*) | 94.16% | 94.85% | 94.51% | 136 |
| Corn Ear Worm (*H. armigera*) | 95.93% | 97.52% | 96.72% | 121 |
| Corn Grasshopper (*O. chinensis*) | **100.00%** | **100.00%** | **100.00%** | 91 |
| Flea Beetle (*Phyllotreta spp.*) | 96.44% | 98.64% | 97.53% | 220 |
| White Grub (*Holotrichia spp.*) | 96.23% | 95.63% | 95.92% | 160 |
| Wire Worm (*A. lineatus*) | 95.02% | 92.51% | 93.75% | 227 |

---

## ☁️ Cloud Services

### Cloudinary — Image Storage
- Uploaded images are stored in the `corn_pest_detections` folder
- Returns a permanent CDN URL saved with every detection record
- Fault-tolerant: if Cloudinary is unavailable, the detection result is still returned

### Google Cloud Firestore — Detection Records
- Each detection is saved as a Firestore document under the `detections` collection
- Fields stored: `detection_id`, `timestamp`, `filename`, `prediction`, `ood`, `image` (URL), `status`, `pest` (full KB profile), `trace` (pipeline audit trail)
- The **History** page reads the 24 most recent records from Firestore
- The **Dashboard** reads aggregate statistics (total detections, acceptance rate, etc.)
- Detection records can be deleted individually from the History page

---

## 🌐 Pages & Features

### `/` — Upload Page
- Drag-and-drop or click-to-browse image upload
- Live preview of selected image
- System status indicators (model, knowledge base, cloud services)
- Link back to most recent detection

### `/result/<detection_id>` — Result Page
- **Accepted**: Pest name, scientific name, confidence percentage, confidence band (Very High / High / Moderate / Low), all 10 class probabilities ranked
- **Rejected**: Clear rejection message with actionable guidance (which OOD signals fired, why)
- Full **Knowledge Base card**: damage symptoms, active ingredients, IRAC MoA groups, biological controls, application timing, IPM recommendation, treatment threshold
- **OOD explainability panel**: per-signal breakdown (MSP, Entropy, Energy, Margin) with pass/fail status
- **Pipeline trace**: every stage's component and outcome — the full decision reasoning audit trail
- Result URLs are **shareable** (Firestore-backed, not session-dependent)

### `/history` — Detection History
- Grid of all past detections (image thumbnails, pest name, confidence, timestamp)
- Acceptance rate and total count statistics
- Individual record deletion

### `/knowledge-base` — Pest Encyclopedia
- Searchable list of all 10 pest profiles (by common name, scientific name, pest group, or active ingredient)
- Click through to individual pest detail pages

### `/knowledge-base/<slug>` — Pest Detail Page
- Full pest profile from the Excel knowledge base
- Link to the trained AI class that maps to this pest
- Per-class model performance metrics (precision, recall, F1) if the model is ready

### `/dashboard` — Model Performance Dashboard
- Headline metrics (accuracy, precision, recall, F1)
- Per-class performance bar chart
- Confusion matrix
- Training accuracy & loss curves
- Usage statistics from Firestore
- Expert mapping coverage report

### `/about` — System Information
- OOD detection configuration
- Full class list and system status
- Research context and references

---

## 🔌 REST API Reference

All endpoints under `/api/`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Run the full detection pipeline; returns JSON record |
| `GET` | `/api/metrics` | Model performance data (headline + per-class + confusion matrix) |
| `GET` | `/api/pests` | Knowledge base listing; supports `?q=<query>` search |
| `GET` | `/api/pests/<slug>` | Individual pest profile |
| `GET` | `/api/status` | Component health (model, KB, Cloudinary, Firestore) |
| `GET` | `/api/history` | Detection history from Firestore; supports `?limit=N` |
| `POST` | `/api/reload` | Reload knowledge base + metrics without restarting the server |

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root with the following variables:

```dotenv
# Flask
SECRET_KEY=your-secret-key-here

# Cloudinary (get from cloudinary.com dashboard)
CLOUDINARY_URL=cloudinary://<api_key>:<api_secret>@<cloud_name>
CLOUDINARY_FOLDER=corn_pest_detections

# Firebase / Firestore
FIREBASE_CREDENTIALS=serviceAccountKey.json
FIRESTORE_COLLECTION=detections
```

Place your Firebase service account JSON file at `serviceAccountKey.json` in the project root (download from the Firebase Console → Project Settings → Service Accounts).

---

## 🚀 Installation & Running Locally

### Prerequisites
- Python 3.10 or higher
- `pip`
- A trained model file: `model/best_corn_pest_model.keras`

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd "CORN PEST AI DECISION SUPPORT WEB APPLICATION"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env         # then edit .env with your credentials

# 5. Run the application
python run.py

# Optional flags:
python run.py --port 8000       # different port
python run.py --host 0.0.0.0    # accessible on local network
python run.py --debug           # enable Flask hot-reload
python run.py --no-warmup       # skip model warm-up on start
```

The application will be available at `http://127.0.0.1:5000`.

### OOD Calibration (Optional but Recommended)

To replace the literature-default OOD thresholds with percentile-calibrated values derived from the training data:

```bash
python scripts/calibrate_ood.py
```

This writes calibrated thresholds to `model/ood_stats.npz`, which the OOD service loads automatically on next start.

---

## 🔬 Research Context

This system is a research project developed at **SLIIT** as part of an academic dissertation on AI-based agricultural decision support. Key research contributions include:

1. **Multi-stage OOD detection** for an agricultural image classifier — combining a domain-agnostic relevance gate (MobileNetV2 / ImageNet) with four independent distributional signals from the pest model itself
2. **Expert vocabulary reconciliation** — a structured mapping layer that bridges training folder names, agronomic common names, and product reference nomenclature across three different vocabularies
3. **Explainable AI architecture** — every decision in the pipeline is logged to a trace shown to the user, implementing the "glass box" principle for agricultural AI
4. **Rule-based knowledge base integration** — connecting a deep learning classifier to a domain expert-curated Excel workbook via a fault-tolerant service layer

### Key References
- Hendrycks, D. & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR 2017.
- Liu, W. et al. (2020). *Energy-based Out-of-distribution Detection*. NeurIPS 2020.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.

---

## 📄 License

This project is developed for academic research purposes. All rights reserved.

---

*Built with care for smarter, safer corn farming.*
