"""Offline retraining with logged human-vs-AI matches (the optional adaptive mode).

This is deliberate, offline retraining, NOT online learning: the AI never changes
during a match. Running the retrain script:
  1. reads finished matches from data/player_games/matches.jsonl,
  2. turns them into extra training rows (both players' perspectives),
  3. trains a CANDIDATE model on the synthetic train+val data plus those rows,
  4. compares candidate vs current on the untouched synthetic test games (and on
     held-out player games when there are enough), then head-to-head,
  5. promotes the candidate only if it is not worse, archiving the current model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..game.engine import IllegalMoveError, P1, P2, replay
from .dataset import game_to_rows, rows_to_frame

PLAYER_GAME_ID_OFFSET = 10_000_000  # keeps player game ids distinct from synthetic ids


def load_player_matches(path: Path) -> tuple[list[dict], int]:
    """Valid, complete matches converted to the raw-game record format. Returns (records, n_skipped)."""
    path = Path(path)
    if not path.exists():
        return [], 0
    records, skipped = [], 0
    for k, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            m = json.loads(line)
            human = int(m["human_side"])
            first = int(m["first_player"])
            moves = [int(x) for x in m["moves"]]
            if human not in (P1, P2) or first not in (P1, P2) or len(moves) != 49:
                raise ValueError("incomplete match")
            final = replay(moves, first)[-1]  # re-checks every move against the rules
            winner = final.winner()
        except (ValueError, KeyError, TypeError, IllegalMoveError, json.JSONDecodeError):
            skipped += 1
            continue
        records.append({"game_id": PLAYER_GAME_ID_OFFSET + k, "first_player": first,
                        "moves": moves, "winner": winner})
    return records, skipped


def player_frame(records: list[dict], seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for rec in records:
        rows.extend(game_to_rows(rec, rng, both_perspectives=True, source="player"))
    return rows_to_frame(rows)


def should_promote(current: dict, candidate: dict, arena: dict, tolerance: float = 0.005) -> tuple[bool, str]:
    """Promotion rule. `current`/`candidate` hold synthetic-test log-loss (and optional player log-loss)."""
    if candidate["test_log_loss"] > current["test_log_loss"] + tolerance:
        return False, (f"candidate is worse on the synthetic test games "
                       f"({candidate['test_log_loss']:.4f} vs {current['test_log_loss']:.4f})")
    if "player_log_loss" in candidate and candidate["player_log_loss"] > current["player_log_loss"] + tolerance:
        return False, "candidate is worse on held-out player games"
    if arena["win_rate"] < 0.5:
        return False, f"candidate lost the head-to-head ({arena['win_rate']:.0%} win rate)"
    return True, "candidate is at least as accurate and wins the head-to-head"


def split_player_games(records: list[dict], seed: int = 0, holdout_frac: float = 0.2, min_holdout_games: int = 10):
    """Hold out some player games for evaluation once there are enough of them."""
    recs = list(records)
    random.Random(seed).shuffle(recs)
    n_hold = int(len(recs) * holdout_frac)
    if n_hold < min_holdout_games // 2 or len(recs) < min_holdout_games:
        return recs, []
    return recs[n_hold:], recs[:n_hold]


def weights_for(df: pd.DataFrame, player_weight: float) -> np.ndarray:
    return np.where(df["source"].to_numpy() == "player", player_weight, 1.0)
