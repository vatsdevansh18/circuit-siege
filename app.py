"""WSGI entry point used by Vercel (and any other WSGI host).

For local play, use run_game.py instead; it starts the same app and opens a browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from circuit_siege.web.server import create_app  # noqa: E402

app = create_app()
