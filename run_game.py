"""Launch Circuit Siege in your browser.

    python run_game.py               # http://127.0.0.1:5000, opens a browser tab
    python run_game.py --port 8080 --no-browser
"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from circuit_siege.utils import paths  # noqa: E402
from circuit_siege.web.server import create_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--matches", default=None, help="match-log file (default data/player_games/matches.jsonl)")
    args = ap.parse_args()

    if not paths.MODEL_PATH.exists():
        print("WARNING: no trained model found at", paths.MODEL_PATH)
        print("  Run:  python scripts/generate_dataset.py  &&  python scripts/train_model.py")
        print("  The game UI will open and explain the problem, but matches cannot start.")
    app = create_app(matches_path=args.matches)
    url = f"http://{args.host}:{args.port}"
    print(f"Circuit Siege running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
