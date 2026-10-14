import random

import numpy as np
import pandas as pd
import pytest

from circuit_siege.ai.features import FEATURE_NAMES, N_FEATURES, extract_features, features_dict
from circuit_siege.game.engine import N_CELLS, P1, P2, apply_move, new_game, replay
from circuit_siege.training.dataset import (
    LABEL,
    game_to_rows,
    generate,
    make_splits,
    simulate_game,
    validate,
)


def random_state(seed, plies=20):
    rng = random.Random(seed)
    s = new_game()
    for _ in range(plies):
        s, _ = apply_move(s, rng.choice(s.legal_moves()))
    return s


def test_feature_vector_shape_and_names():
    s = random_state(1)
    x = extract_features(s.board, s.to_move, P1)
    assert len(x) == N_FEATURES == len(FEATURE_NAMES) == N_CELLS + 19
    assert all(np.isfinite(x))


def test_features_are_perspective_symmetric():
    """P1's view of a board == P2's view of the colour-swapped board."""
    for seed in range(5):
        s = random_state(seed, plies=15 + seed)
        swapped = tuple(-v for v in s.board)
        a = extract_features(s.board, s.to_move, P1)
        b = extract_features(swapped, -s.to_move, P2)
        assert a == b


def test_feature_values_are_sensible():
    s = random_state(3, plies=30)
    f = features_dict(s.board, s.to_move, P1)
    sc = s.scores()
    assert f["my_points"] == sc[P1] and f["opp_points"] == sc[P2]
    assert f["point_diff"] == sc[P1] - sc[P2]
    assert f["empties"] == s.empties
    assert f["my_turn"] == (1.0 if s.to_move == P1 else 0.0)
    assert 0 <= f["phase"] <= 1


def test_bad_feature_inputs_raise():
    with pytest.raises(ValueError):
        extract_features((0,) * 10, P1, P1)
    with pytest.raises(ValueError):
        extract_features((0,) * N_CELLS, 0, P1)


def test_simulation_is_deterministic():
    assert simulate_game(7, seed=5) == simulate_game(7, seed=5)
    assert simulate_game(7, seed=5)["moves"] != simulate_game(8, seed=5)["moves"]


def test_dataset_rows_match_live_feature_extraction(tiny_data):
    """Preprocessing consistency: a training row equals what the live AI computes."""
    records, df = tiny_data
    rec = records[0]
    rows = df[df["game_id"] == rec["game_id"]].sort_values("ply")
    states = replay(rec["moves"], rec["first_player"])
    for _, row in rows.head(10).iterrows():
        st = states[int(row["ply"])]
        live = extract_features(st.board, st.to_move, int(row["perspective"]))
        assert np.allclose(row[FEATURE_NAMES].to_numpy(float), live)
        assert row[LABEL] == (1 if rec["winner"] == row["perspective"] else 0)


def test_dataset_validation_passes_and_reports(tiny_data):
    records, df = tiny_data
    rep = validate(df)
    assert rep["missing_values"] == 0
    assert rep["games"] == len(records) == 40
    assert set(rep["label_balance"]) <= {0, 1}
    # one row per non-terminal, non-empty position: 48 per game
    assert rep["rows"] == 40 * 48


def test_validation_catches_bad_data(tiny_data):
    _, df = tiny_data
    bad = df.copy()
    bad.loc[0, "cell_00"] = np.nan
    with pytest.raises(ValueError):
        validate(bad)
    bad = df.copy()
    bad.loc[0, LABEL] = 2
    with pytest.raises(ValueError):
        validate(bad)
    with pytest.raises(ValueError):
        validate(df.drop(columns=["my_points"]))


def test_splits_are_by_game_with_no_overlap():
    s = make_splits(range(100), seed=3)
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(range(100))
    assert not (set(s["train"]) & set(s["test"]) or set(s["train"]) & set(s["val"]) or set(s["val"]) & set(s["test"]))
    assert make_splits(range(100), seed=3) == s  # reproducible


def test_both_perspectives_mode_doubles_rows():
    rec = simulate_game(1, seed=1)
    one = game_to_rows(rec, random.Random(0))
    two = game_to_rows(rec, random.Random(0), both_perspectives=True)
    assert len(two) == 2 * len(one)


def test_generate_parallel_equals_serial():
    _, a = generate(6, seed=9, workers=1)
    _, b = generate(6, seed=9, workers=2)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
