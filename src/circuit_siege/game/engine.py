"""Pure game rules for Circuit Siege.

The board is a tuple of 49 ints (row-major, 7x7): 0 = empty, +1 = player one,
-1 = player two. States are immutable; `apply_move` returns a new state. This
module has no ML code, so the same rules drive the data generator, the training
features, the AI and the web game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIZE = 7
N_CELLS = SIZE * SIZE
P1, P2 = 1, -1
EMPTY = 0

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def rc_to_idx(r: int, c: int) -> int:
    return r * SIZE + c


def idx_to_rc(i: int) -> tuple[int, int]:
    return divmod(i, SIZE)


# Cores form a plus shape around the centre: valuable, but exposed.
CORES = frozenset(rc_to_idx(r, c) for r, c in [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3)])
CORE_VALUE = 3
CELL_VALUE = tuple(CORE_VALUE if i in CORES else 1 for i in range(N_CELLS))
TOTAL_POINTS = sum(CELL_VALUE)  # 59: odd, so a full board can never be a tie

CORNERS = frozenset(rc_to_idx(r, c) for r in (0, SIZE - 1) for c in (0, SIZE - 1))
EDGES = frozenset(
    i for i in range(N_CELLS)
    if i not in CORNERS and (idx_to_rc(i)[0] in (0, SIZE - 1) or idx_to_rc(i)[1] in (0, SIZE - 1))
)


def _build_rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """For every cell, the cells visited walking outward in each direction."""
    rays = []
    for i in range(N_CELLS):
        r0, c0 = idx_to_rc(i)
        cell_rays = []
        for dr, dc in DIRECTIONS:
            ray = []
            r, c = r0 + dr, c0 + dc
            while 0 <= r < SIZE and 0 <= c < SIZE:
                ray.append(rc_to_idx(r, c))
                r, c = r + dr, c + dc
            if len(ray) >= 2:  # a capture needs at least one enemy plus an anchor
                cell_rays.append(tuple(ray))
        rays.append(tuple(cell_rays))
    return tuple(rays)


RAYS = _build_rays()
NEIGHBOURS = tuple(
    tuple(
        rc_to_idx(idx_to_rc(i)[0] + dr, idx_to_rc(i)[1] + dc)
        for dr, dc in DIRECTIONS
        if 0 <= idx_to_rc(i)[0] + dr < SIZE and 0 <= idx_to_rc(i)[1] + dc < SIZE
    )
    for i in range(N_CELLS)
)


class IllegalMoveError(ValueError):
    """Raised when a move breaks the rules (occupied cell, bad index, game over)."""


def flips_for(board: tuple[int, ...] | list[int], idx: int, player: int) -> list[int]:
    """Cells that would flip if `player` placed a node on empty cell `idx`."""
    flipped: list[int] = []
    enemy = -player
    for ray in RAYS[idx]:
        line = []
        for j in ray:
            v = board[j]
            if v == enemy:
                line.append(j)
                continue
            if v == player and line:
                flipped.extend(line)
            break
    return flipped


def capture_gain(board, idx: int, player: int) -> int:
    """Points `player` gains by playing `idx`: the cell itself plus captured cells.

    Captured cells count double in the *difference* (they leave the opponent too),
    but this returns only the player's own gain; callers decide how to weigh it.
    """
    return CELL_VALUE[idx] + sum(CELL_VALUE[j] for j in flips_for(board, idx, player))


@dataclass(frozen=True)
class GameState:
    board: tuple[int, ...] = (EMPTY,) * N_CELLS
    to_move: int = P1
    history: tuple[int, ...] = field(default_factory=tuple)  # cells played, in order

    @property
    def ply(self) -> int:
        return len(self.history)

    @property
    def empties(self) -> int:
        return self.board.count(EMPTY)

    def is_terminal(self) -> bool:
        return EMPTY not in self.board

    def legal_moves(self) -> list[int]:
        if self.is_terminal():
            return []
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def scores(self) -> dict[int, int]:
        s = {P1: 0, P2: 0}
        for i, v in enumerate(self.board):
            if v != EMPTY:
                s[v] += CELL_VALUE[i]
        return s

    def winner(self) -> int | None:
        """+1 / -1 for a finished game, 0 for a draw (impossible on a full board), None if ongoing."""
        if not self.is_terminal():
            return None
        s = self.scores()
        return (s[P1] > s[P2]) - (s[P1] < s[P2])


def new_game(first_player: int = P1) -> GameState:
    if first_player not in (P1, P2):
        raise ValueError("first_player must be +1 or -1")
    return GameState(to_move=first_player)


def apply_move(state: GameState, idx: int) -> tuple[GameState, list[int]]:
    """Play `idx` for the side to move. Returns (new_state, flipped_cells)."""
    if state.is_terminal():
        raise IllegalMoveError("the game is already over")
    if isinstance(idx, bool) or not isinstance(idx, int):
        raise IllegalMoveError(f"cell index must be an integer, got {idx!r}")
    if not 0 <= idx < N_CELLS:
        raise IllegalMoveError(f"cell index {idx} is off the board")
    if state.board[idx] != EMPTY:
        raise IllegalMoveError(f"cell {idx} is already occupied")
    player = state.to_move
    flipped = flips_for(state.board, idx, player)
    board = list(state.board)
    board[idx] = player
    for j in flipped:
        board[j] = player
    return GameState(tuple(board), -player, state.history + (idx,)), flipped


def replay(moves, first_player: int = P1) -> list[GameState]:
    """All states of a game, from the empty board to the last move."""
    states = [new_game(first_player)]
    for m in moves:
        states.append(apply_move(states[-1], int(m))[0])
    return states
