"""Turn a game position into the numeric feature vector the value model expects.

Training (dataset building) and live play (the AI opponent) both call
`extract_features`, so preprocessing is identical in both places by construction.
Everything is computed from the *visible* board and the side to move, which is all
a player is entitled to know in this perfect-information game.

Features are *perspective-relative*: "my" means the player whose win probability
we are estimating. That lets one model serve both colours.
"""

from __future__ import annotations

from ..game.engine import (
    CELL_VALUE,
    CORES,
    CORNERS,
    EDGES,
    EMPTY,
    N_CELLS,
    NEIGHBOURS,
    SIZE,
    flips_for,
)

CELL_FEATURES = [f"cell_{r}{c}" for r in range(SIZE) for c in range(SIZE)]

ENGINEERED_FEATURES = [
    "my_points",          # current score of the perspective player
    "opp_points",
    "point_diff",         # my_points - opp_points
    "my_cores",           # cores (3-pt cells) held
    "opp_cores",
    "my_corners",         # corners can never be captured
    "opp_corners",
    "my_edges",           # edge cells (excluding corners): hard to flank
    "opp_edges",
    "my_frontier",        # my nodes next to an empty cell (exposed to attack)
    "opp_frontier",
    "my_vulnerable_pts",  # points of my nodes the opponent could flip next move
    "opp_vulnerable_pts",
    "my_best_gain",       # best immediate swing (placement + captures) available to me
    "opp_best_gain",
    "empties",            # empty cells left
    "phase",              # 0 = empty board, 1 = full board
    "my_turn",            # 1 if it is my move
    "i_move_last",        # 1 if I will make the final placement of the game
]

FEATURE_NAMES = CELL_FEATURES + ENGINEERED_FEATURES
N_FEATURES = len(FEATURE_NAMES)


def _tactics(board, player: int) -> tuple[int, set[int]]:
    """(best point swing for `player` this move, set of enemy cells `player` can flip)."""
    best = 0
    attackable: set[int] = set()
    for i in range(N_CELLS):
        if board[i] != EMPTY:
            continue
        flipped = flips_for(board, i, player)
        captured = sum(CELL_VALUE[j] for j in flipped)
        # Swing in score difference: the new cell, plus captures counted twice
        # (they are added to me and removed from the opponent).
        swing = CELL_VALUE[i] + 2 * captured
        if swing > best:
            best = swing
        attackable.update(flipped)
    return best, attackable


def extract_features(board, to_move: int, perspective: int) -> list[float]:
    """Feature vector for `board` with `to_move` to play, seen by `perspective`."""
    if perspective not in (1, -1) or to_move not in (1, -1):
        raise ValueError("to_move and perspective must be +1 or -1")
    if len(board) != N_CELLS:
        raise ValueError(f"board must have {N_CELLS} cells, got {len(board)}")

    me, opp = perspective, -perspective
    cells = [float(v * perspective) for v in board]

    my_pts = opp_pts = my_cores = opp_cores = 0
    my_corners = opp_corners = my_edges = opp_edges = 0
    my_frontier = opp_frontier = 0
    empties = 0
    for i, v in enumerate(board):
        if v == EMPTY:
            empties += 1
            continue
        mine = v == me
        val = CELL_VALUE[i]
        exposed = any(board[n] == EMPTY for n in NEIGHBOURS[i])
        if mine:
            my_pts += val
            my_cores += i in CORES
            my_corners += i in CORNERS
            my_edges += i in EDGES
            my_frontier += exposed
        else:
            opp_pts += val
            opp_cores += i in CORES
            opp_corners += i in CORNERS
            opp_edges += i in EDGES
            opp_frontier += exposed

    my_best, opp_attackable = _tactics(board, me)
    opp_best, my_attackable = _tactics(board, opp)
    # _tactics(board, me) returns cells *I* can flip, i.e. the opponent's vulnerable cells.
    opp_vuln = sum(CELL_VALUE[j] for j in opp_attackable)
    my_vuln = sum(CELL_VALUE[j] for j in my_attackable)

    my_turn = 1 if to_move == me else 0
    if empties == 0:
        i_last = 0
    else:
        # Players alternate; whoever is to move now also makes moves 3, 5, ...
        last_mover = to_move if empties % 2 == 1 else -to_move
        i_last = 1 if last_mover == me else 0

    engineered = [
        my_pts, opp_pts, my_pts - opp_pts,
        my_cores, opp_cores,
        my_corners, opp_corners,
        my_edges, opp_edges,
        my_frontier, opp_frontier,
        my_vuln, opp_vuln,
        my_best, opp_best,
        empties, 1.0 - empties / N_CELLS,
        my_turn, i_last,
    ]
    return cells + [float(x) for x in engineered]


def features_dict(board, to_move: int, perspective: int) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, extract_features(board, to_move, perspective)))
