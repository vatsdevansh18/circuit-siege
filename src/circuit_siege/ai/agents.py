"""Scripted bots used ONLY to generate synthetic training games and as arena opponents.

None of these is the in-game AI. The in-game opponent is `ModelAgent`
(see model_agent.py), which chooses moves with the trained value model.
The bots are deliberately varied and noisy (epsilon-random), so the dataset
covers many kinds of positions and the labels are not dominated by one style.
"""

from __future__ import annotations

import random

from ..game.engine import CELL_VALUE, CORNERS, EDGES, GameState, apply_move, flips_for


def _swing(board, idx: int, player: int) -> int:
    """Change in (my score - opponent score) from playing idx."""
    return CELL_VALUE[idx] + 2 * sum(CELL_VALUE[j] for j in flips_for(board, idx, player))


def _best_reply_swing(state: GameState) -> int:
    board, player = state.board, state.to_move
    return max((_swing(board, i, player) for i in state.legal_moves()), default=0)


class Agent:
    name = "agent"

    def __init__(self, epsilon: float = 0.0, rng: random.Random | None = None):
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    def score_move(self, state: GameState, idx: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def choose(self, state: GameState) -> int:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves: the game is over")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        scored = [(self.score_move(state, m), m) for m in moves]
        best = max(s for s, _ in scored)
        return self.rng.choice([m for s, m in scored if s >= best - 1e-9])

    def describe(self) -> str:
        return f"{self.name}(eps={self.epsilon:.2f})"


class RandomAgent(Agent):
    name = "random"

    def choose(self, state: GameState) -> int:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("no legal moves: the game is over")
        return self.rng.choice(moves)


class GreedyAgent(Agent):
    """Maximises the immediate score swing."""
    name = "greedy"

    def score_move(self, state, idx):
        return _swing(state.board, idx, state.to_move)


class CautiousAgent(Agent):
    """Immediate swing minus the opponent's best reply swing (a 2-ply heuristic)."""
    name = "cautious"

    def __init__(self, epsilon=0.0, rng=None, caution: float = 0.8):
        super().__init__(epsilon, rng)
        self.caution = caution

    def score_move(self, state, idx):
        gain = _swing(state.board, idx, state.to_move)
        nxt, _ = apply_move(state, idx)
        return gain - self.caution * _best_reply_swing(nxt)


class PositionalAgent(Agent):
    """Immediate swing plus hand-tuned positional bonuses (corners, edges)."""
    name = "positional"

    def score_move(self, state, idx):
        bonus = 3.0 if idx in CORNERS else 1.0 if idx in EDGES else 0.0
        return _swing(state.board, idx, state.to_move) + bonus


AGENT_TYPES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "cautious": CautiousAgent,
    "positional": PositionalAgent,
}


def make_agent(kind: str, epsilon: float = 0.0, rng: random.Random | None = None) -> Agent:
    try:
        return AGENT_TYPES[kind](epsilon=epsilon, rng=rng)
    except KeyError:
        raise ValueError(f"unknown agent kind {kind!r}; choose from {sorted(AGENT_TYPES)}") from None
