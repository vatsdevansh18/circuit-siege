"""Head-to-head matches between agents: the reward-based evaluation of the AI."""

from __future__ import annotations

import math
import time
from typing import Callable

from ..game.engine import P1, P2, apply_move, new_game


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (draws should be counted as half wins)."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def play_game(agent_a, agent_b, a_side: int, first: int = P1) -> dict:
    """agent_a plays `a_side`; returns the result from agent_a's point of view."""
    agents = {a_side: agent_a, -a_side: agent_b}
    state = new_game(first)
    think = {a_side: 0.0, -a_side: 0.0}
    while not state.is_terminal():
        t0 = time.perf_counter()
        move = agents[state.to_move].choose(state)
        think[state.to_move] += time.perf_counter() - t0
        state, _ = apply_move(state, move)
    s = state.scores()
    return {
        "result": state.winner() * a_side,  # +1 win, -1 loss, 0 draw (for agent_a)
        "margin": s[a_side] - s[-a_side],
        "a_think_s": think[a_side],
        "a_moves": sum(1 for i in range(len(state.history)) if (first if i % 2 == 0 else -first) == a_side),
    }


def run_arena(make_a: Callable[[int], object], make_b: Callable[[int], object], n_games: int) -> dict:
    """Play n_games, alternating colours and who moves first. make_x(game_index) -> agent."""
    wins = losses = draws = 0
    margins, think, moves = [], 0.0, 0
    for g in range(n_games):
        a_side = P1 if g % 2 == 0 else P2
        first = P1 if (g // 2) % 2 == 0 else P2
        r = play_game(make_a(g), make_b(g), a_side, first)
        wins += r["result"] == 1
        losses += r["result"] == -1
        draws += r["result"] == 0
        margins.append(r["margin"])
        think += r["a_think_s"]
        moves += r["a_moves"]
    score = wins + 0.5 * draws
    lo, hi = wilson_interval(score, n_games)
    return {
        "games": n_games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": round(score / n_games, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "avg_margin": round(sum(margins) / n_games, 2),
        "avg_ms_per_move": round(1000 * think / max(1, moves), 2),
    }
