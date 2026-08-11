"""
Entry point for the CornGuard AI web application.

Front end and back end are one Flask process: Jinja renders the pages, /api
serves the interactive parts, and Tailwind comes from a CDN, so this is the
only thing that needs to be started.

    python run.py                 # http://127.0.0.1:5000
    python run.py --port 8000     # different port
    python run.py --no-warmup     # skip the first-request warm-up
    python run.py --host 0.0.0.0  # reachable from other devices on the LAN
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from app import create_app  # noqa: E402
from app.services import detection_pipeline, model_service  # noqa: E402

app = create_app()


def _banner(host: str, port: int) -> None:
    status = detection_pipeline.system_status()

    def mark(component: dict) -> str:
        return "ready" if component.get("ready") else f"unavailable ({component.get('error')})"

    print()
    print("=" * 68)
    print("  CornGuard AI - Corn Pest Decision Support System")
    print("=" * 68)
    print(f"  Model           : {mark(status['model'])}")
    print(f"  Knowledge base  : {mark(status['knowledge_base'])}")
    print(f"  Image storage   : {mark(status['cloudinary'])}")
    print(f"  Record storage  : {mark(status['firestore'])}")
    print(f"  OOD thresholds  : {'calibrated' if status['ood']['calibrated'] else 'literature defaults'}")
    print("-" * 68)
    print(f"  Open  http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    print("=" * 68)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CornGuard AI web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="enable the Flask reloader")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="do not run a throwaway inference at start-up",
    )
    args = parser.parse_args()

    # The reloader would otherwise load the model twice, which is slow and
    # doubles the memory footprint.
    is_reload_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not args.debug or is_reload_child:
        if not args.no_warmup:
            print("Loading the model (first start takes a few seconds)...")
            model_service.warm_up()
        _banner(args.host, args.port)

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
