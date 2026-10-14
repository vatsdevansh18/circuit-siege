"""The AI opponent: picks moves using the trained value model.

How it decides (1-ply lookahead + learned evaluation):
  1. The rules engine lists every legal move (this is the only part done by rules).
  2. For each move, the engine simulates the resulting position.
  3. Each resulting position is converted to features (same code as training).
  4. The model predicts P(AI wins) for each one.
  5. The AI picks according to its difficulty:
       easy   - samples from ALL moves in proportion to exp(logit(p) / T), with a high T,
                so it often plays weaker moves (makes mistakes)
       normal - samples only among the model's 3 best-rated moves with a low T
                (usually the best move, occasionally the 2nd/3rd)
       hard   - re-ranks the top candidates with a 2-ply search: it assumes the
                human replies with the move the *model* rates best for the human,
                then plays the move whose worst case is best.
The model is always the thing that evaluates positions; no hand-written strategy
is used to pick moves.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..game.engine import GameState, apply_move
from .features import extract_features
from .model_io import predict_win_proba

DIFFICULTIES = {
    "easy": {"temperature": 2.0, "lookahead": False,
             "label": "Rookie", "blurb": "Samples loosely from the model's ratings and makes mistakes."},
    "normal": {"temperature": 0.15, "pool": 3, "lookahead": False,
               "label": "Operator", "blurb": "Usually plays the move the model rates best."},
    "hard": {"temperature": 0.0, "lookahead": True, "top_k": 6,
             "label": "Architect", "blurb": "Model-rated moves checked against your best model-rated reply."},
}


@dataclass
class Decision:
    move: int
    win_prob: float                      # model's P(AI wins) after the chosen move
    candidates: list[dict] = field(default_factory=list)  # [{cell, p}] sorted best-first
    method: str = "model"
    elapsed_ms: float = 0.0


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class ModelAgent:
    name = "model"

    def __init__(self, bundle: dict, difficulty: str = "normal", rng: random.Random | None = None):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}; choose from {list(DIFFICULTIES)}")
        self.bundle = bundle
        self.model = bundle["model"]
        self.difficulty = difficulty
        self.cfg = DIFFICULTIES[difficulty]
        self.rng = rng or random.Random()

    def describe(self) -> str:
        return f"model[{self.bundle.get('algorithm', '?')}]({self.difficulty})"

    # -- evaluation -------------------------------------------------------
    def position_win_prob(self, state: GameState, perspective: int) -> float:
        """Model's estimate that `perspective` wins from `state`."""
        if state.is_terminal():
            w = state.winner()
            return 1.0 if w == perspective else 0.0 if w == -perspective else 0.5
        x = extract_features(state.board, state.to_move, perspective)
        return float(predict_win_proba(self.model, [x])[0])

    def evaluate_moves(self, state: GameState) -> tuple[list[int], np.ndarray]:
        """For each legal move: P(side to move wins) after playing it."""
        me = state.to_move
        moves = state.legal_moves()
        if not moves:
            return [], np.zeros(0)
        rows, probs, idx_rows = [], np.zeros(len(moves)), []
        for k, m in enumerate(moves):
            nxt, _ = apply_move(state, m)
            if nxt.is_terminal():  # final move: the outcome is known exactly
                w = nxt.winner()
                probs[k] = 1.0 if w == me else 0.0 if w == -me else 0.5
            else:
                rows.append(extract_features(nxt.board, nxt.to_move, me))
                idx_rows.append(k)
        if rows:
            probs[idx_rows] = predict_win_proba(self.model, rows)
        return moves, probs

    def _two_ply(self, state: GameState, moves: list[int], probs: np.ndarray) -> np.ndarray:
        """Worst-case P(win) for the top-k moves, assuming the opponent's model-best reply."""
        me = state.to_move
        top_k = min(self.cfg.get("top_k", 6), len(moves))
        order = np.argsort(-probs)[:top_k]
        refined = np.full(len(moves), -1.0)
        for k in order:
            nxt, _ = apply_move(state, moves[k])
            if nxt.is_terminal():
                refined[k] = probs[k]
                continue
            replies, reply_probs_opp = self.evaluate_moves(nxt)  # P(opponent wins) per reply
            if not len(replies):
                refined[k] = probs[k]
                continue
            # Opponent picks the reply the model rates best for them; score that line for me.
            worst = int(np.argmax(reply_probs_opp))
            after, _ = apply_move(nxt, replies[worst])
            refined[k] = self.position_win_prob(after, me)
        return refined

    # -- decision ---------------------------------------------------------
    def choose_with_info(self, state: GameState) -> Decision:
        t0 = time.perf_counter()
        moves, probs = self.evaluate_moves(state)
        if not moves:
            raise ValueError("no legal moves: the game is over")

        method = "model"
        if not np.all(np.isfinite(probs)):  # defensive: predict_win_proba already sanitises
            probs = np.where(np.isfinite(probs), probs, 0.5)
            method = "model (sanitised)"

        ranking = probs
        if self.cfg.get("lookahead"):
            ranking = self._two_ply(state, moves, probs)
            method = "model + 2-ply"

        temperature = self.cfg["temperature"]
        if temperature <= 0:
            best = ranking.max()
            pick = self.rng.choice([k for k in range(len(moves)) if ranking[k] >= best - 1e-9])
        else:
            pool = self.cfg.get("pool") or len(moves)
            idx = np.argsort(-probs)[:pool]
            z = _logit(probs[idx]) / temperature
            w = np.exp(z - z.max())
            pick = int(idx[self.rng.choices(range(len(idx)), weights=w.tolist())[0]])

        order = np.argsort(-probs)
        candidates = [{"cell": int(moves[k]), "p": round(float(probs[k]), 4)} for k in order]
        return Decision(
            move=int(moves[pick]),
            win_prob=float(probs[pick]),
            candidates=candidates,
            method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def choose(self, state: GameState) -> int:
        return self.choose_with_info(state).move


def softmax_probabilities(probs, temperature: float) -> list[float]:
    """Exposed for tests: the sampling distribution used by easy/normal."""
    z = _logit(np.asarray(probs, dtype=float)) / temperature
    w = np.exp(z - z.max())
    return (w / w.sum()).tolist()


__all__ = ["DIFFICULTIES", "Decision", "ModelAgent", "softmax_probabilities"]
