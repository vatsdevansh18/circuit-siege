"""Synthetic dataset generation, validation and leakage-safe splitting.

Pipeline:
  simulate_game()  -> one raw game record (moves, bots, result)       [data/raw]
  game_to_rows()   -> one feature row per non-terminal position        [data/processed]
  validate()       -> missing values, duplicates, ranges, label balance
  make_splits()    -> train / val / test split *by game id*
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..ai.agents import make_agent
from ..ai.features import ENGINEERED_FEATURES, FEATURE_NAMES, extract_features
from ..game.engine import P1, P2, apply_move, new_game, replay

# Mix of data-generator bots. Weighted towards bots that play sensibly so that
# positions resemble real games, with some random play for coverage.
AGENT_MIX = {"random": 0.15, "greedy": 0.30, "cautious": 0.30, "positional": 0.25}
EPSILON_RANGE = (0.0, 0.30)
META_COLUMNS = ["game_id", "ply", "perspective", "source"]
LABEL = "label"


def _game_rng(seed: int, game_id: int) -> random.Random:
    return random.Random(seed * 1_000_003 + game_id)


def simulate_game(game_id: int, seed: int) -> dict:
    """Play one bot-vs-bot game. Deterministic given (game_id, seed)."""
    rng = _game_rng(seed, game_id)
    kinds, weights = zip(*AGENT_MIX.items())
    bots = {}
    for side in (P1, P2):
        kind = rng.choices(kinds, weights=weights)[0]
        eps = round(rng.uniform(*EPSILON_RANGE), 3)
        bots[side] = make_agent(kind, epsilon=eps, rng=random.Random(rng.random()))
    first = rng.choice([P1, P2])
    state = new_game(first)
    while not state.is_terminal():
        state, _ = apply_move(state, bots[state.to_move].choose(state))
    s = state.scores()
    return {
        "game_id": game_id,
        "seed": seed,
        "first_player": first,
        "p1_bot": bots[P1].describe(),
        "p2_bot": bots[P2].describe(),
        "moves": list(state.history),
        "score_p1": s[P1],
        "score_p2": s[P2],
        "winner": state.winner(),
    }


def game_to_rows(record: dict, rng: random.Random, both_perspectives: bool = False,
                 source: str = "selfplay") -> list[list]:
    """Feature rows for every non-terminal position of a recorded game.

    Each position is seen from one randomly chosen player's perspective (or both).
    Label = 1 if that player won the game. Terminal positions are excluded because
    their outcome is just the final score; empty-board positions carry no information.
    """
    states = replay(record["moves"], record["first_player"])
    winner = record["winner"]
    rows = []
    for ply, st in enumerate(states):
        if ply == 0 or st.is_terminal():
            continue
        perspectives = (P1, P2) if both_perspectives else (rng.choice((P1, P2)),)
        for p in perspectives:
            label = 1 if winner == p else 0
            rows.append([record["game_id"], ply, p, source]
                        + extract_features(st.board, st.to_move, p) + [label])
    return rows


def _work(args):
    game_ids, seed = args
    records, rows = [], []
    for gid in game_ids:
        rec = simulate_game(gid, seed)
        records.append(rec)
        rows.extend(game_to_rows(rec, random.Random(seed * 7_919 + gid)))
    return records, rows


def generate(n_games: int, seed: int = 42, workers: int = 1) -> tuple[list[dict], pd.DataFrame]:
    """Simulate `n_games` and return (raw records, feature DataFrame). Deterministic."""
    ids = list(range(n_games))
    chunk = max(1, n_games // max(1, workers * 4))
    jobs = [(ids[i:i + chunk], seed) for i in range(0, n_games, chunk)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_work, jobs))
    else:
        results = [_work(j) for j in jobs]
    records = [r for recs, _ in results for r in recs]
    rows = [row for _, rws in results for row in rws]
    return records, rows_to_frame(rows)


def rows_to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_NAMES + [LABEL])
    df[FEATURE_NAMES] = df[FEATURE_NAMES].astype(np.float32)
    df[LABEL] = df[LABEL].astype(np.int8)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Data-quality checks. Raises ValueError on hard failures, returns a summary."""
    missing_cols = [c for c in META_COLUMNS + FEATURE_NAMES + [LABEL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"dataset is missing columns: {missing_cols[:5]}")
    n_missing = int(df[FEATURE_NAMES + [LABEL]].isna().sum().sum())
    if n_missing:
        raise ValueError(f"dataset has {n_missing} missing values")
    if not set(df[LABEL].unique()) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    cells = df[[c for c in FEATURE_NAMES if c.startswith("cell_")]]
    if not cells.isin([-1, 0, 1]).all().all():
        raise ValueError("cell features must be -1, 0 or +1")

    feature_dupes = int(df.duplicated(subset=FEATURE_NAMES).sum())
    exact_dupes = int(df.duplicated(subset=FEATURE_NAMES + [LABEL]).sum())
    balance = df[LABEL].value_counts(normalize=True).sort_index()
    desc = df[ENGINEERED_FEATURES].describe().T[["mean", "std", "min", "max"]]
    return {
        "rows": int(len(df)),
        "games": int(df["game_id"].nunique()),
        "features": len(FEATURE_NAMES),
        "missing_values": n_missing,
        "duplicate_feature_rows": feature_dupes,
        "duplicate_rows_incl_label": exact_dupes,
        "label_balance": {int(k): round(float(v), 4) for k, v in balance.items()},
        "engineered_feature_stats": desc.round(3).to_dict(orient="index"),
    }


def make_splits(game_ids, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Split *games* (not rows) so no game's positions leak across splits."""
    ids = sorted(set(int(g) for g in game_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    n_val = int(round(len(ids) * val_frac))
    return {
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }
