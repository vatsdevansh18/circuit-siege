import pytest

from circuit_siege.game.engine import (
    CELL_VALUE,
    CORES,
    CORNERS,
    N_CELLS,
    P1,
    P2,
    TOTAL_POINTS,
    GameState,
    IllegalMoveError,
    apply_move,
    flips_for,
    new_game,
    rc_to_idx,
    replay,
)


def board_with(cells: dict[tuple[int, int], int]) -> tuple[int, ...]:
    b = [0] * N_CELLS
    for (r, c), v in cells.items():
        b[rc_to_idx(r, c)] = v
    return tuple(b)


def test_board_constants():
    assert TOTAL_POINTS == 59 and TOTAL_POINTS % 2 == 1  # no draws on a full board
    assert len(CORES) == 5 and all(CELL_VALUE[i] == 3 for i in CORES)
    assert len(CORNERS) == 4


def test_new_game_is_empty_and_everything_is_legal():
    s = new_game()
    assert s.empties == N_CELLS and s.ply == 0 and s.to_move == P1
    assert len(s.legal_moves()) == N_CELLS
    assert s.winner() is None
    with pytest.raises(ValueError):
        new_game(0)


@pytest.mark.parametrize("dr,dc", [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)])
def test_flank_capture_in_every_direction(dr, dc):
    r0, c0 = 3, 3
    board = board_with({(r0 + dr, c0 + dc): P2, (r0 + 2 * dr, c0 + 2 * dc): P2, (r0 + 3 * dr, c0 + 3 * dc): P1})
    flipped = flips_for(board, rc_to_idx(r0, c0), P1)
    assert sorted(flipped) == sorted([rc_to_idx(r0 + dr, c0 + dc), rc_to_idx(r0 + 2 * dr, c0 + 2 * dc)])


def test_no_capture_without_anchor_or_with_gap():
    # enemy line running to the edge: no anchor, no capture
    assert flips_for(board_with({(0, 1): P2, (0, 2): P2}), rc_to_idx(0, 0), P1) == []
    # gap between enemy and anchor breaks the line
    assert flips_for(board_with({(0, 1): P2, (0, 3): P1}), rc_to_idx(0, 0), P1) == []
    # own node adjacent: nothing to flip
    assert flips_for(board_with({(0, 1): P1, (0, 2): P1}), rc_to_idx(0, 0), P1) == []


def test_multi_direction_capture_and_apply_move():
    board = board_with({(3, 2): P2, (3, 1): P1, (2, 3): P2, (1, 3): P1, (4, 4): P2})
    s = GameState(board, P1, ())
    nxt, flipped = apply_move(s, rc_to_idx(3, 3))
    assert sorted(flipped) == sorted([rc_to_idx(3, 2), rc_to_idx(2, 3)])
    assert nxt.board[rc_to_idx(3, 2)] == P1 and nxt.board[rc_to_idx(2, 3)] == P1
    assert nxt.board[rc_to_idx(4, 4)] == P2  # no anchor on that diagonal
    assert nxt.to_move == P2 and nxt.history == (rc_to_idx(3, 3),)
    assert s.board == board  # original state untouched (immutable)


@pytest.mark.parametrize("bad", [-1, 49, 100, "3", 2.0, None, True])
def test_illegal_indices_rejected(bad):
    with pytest.raises(IllegalMoveError):
        apply_move(new_game(), bad)


def test_occupied_cell_rejected():
    s, _ = apply_move(new_game(), 5)
    with pytest.raises(IllegalMoveError):
        apply_move(s, 5)


def test_full_game_ends_with_correct_scores_and_winner():
    s = new_game(P2)
    for i in range(N_CELLS):
        assert not s.is_terminal()
        s, _ = apply_move(s, i)
    assert s.is_terminal() and s.legal_moves() == []
    sc = s.scores()
    assert sc[P1] + sc[P2] == TOTAL_POINTS
    assert s.winner() == (P1 if sc[P1] > sc[P2] else P2)
    with pytest.raises(IllegalMoveError):
        apply_move(s, 0)


def test_replay_reproduces_states():
    moves = [24, 25, 23, 0, 48, 10]
    states = replay(moves, P1)
    assert len(states) == len(moves) + 1
    assert states[-1].history == tuple(moves)
    assert states[0] == new_game(P1)


def test_restart_gives_fresh_state():
    s, _ = apply_move(new_game(), 3)
    fresh = new_game()
    assert fresh.board != s.board and fresh.ply == 0
