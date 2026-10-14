"""Project paths, resolved relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_GAMES = DATA_DIR / "raw" / "selfplay_games.jsonl"
POSITIONS = DATA_DIR / "processed" / "positions.csv.gz"
POSITIONS_SAMPLE = DATA_DIR / "processed" / "positions_sample.csv"
SPLITS = DATA_DIR / "processed" / "splits.json"
PLAYER_MATCHES = DATA_DIR / "player_games" / "matches.jsonl"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "value_model.joblib"
MODEL_CARD = MODELS_DIR / "model_card.json"
ARCHIVE_DIR = MODELS_DIR / "archive"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
