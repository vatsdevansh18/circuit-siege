import random

import joblib
import numpy as np
import pytest

from circuit_siege.ai.agents import make_agent
from circuit_siege.ai.features import N_FEATURES
from circuit_siege.ai.model_agent import DIFFICULTIES, ModelAgent, softmax_probabilities
from circuit_siege.ai.model_io import (
    ModelIncompatibleError,
    ModelNotFoundError,
    load_bundle,
    predict_win_proba,
)
from circuit_siege.evaluation.arena import play_game, run_arena, wilson_interval
from circuit_siege.evaluation.metrics import classification_metrics
from circuit_siege.game.engine import N_CELLS, P1, GameState, apply_move, new_game
from circuit_siege.training.train import select_model


def test_training_selects_and_scores_models(tiny_data):
    _, df = tiny_data
    ids = sorted(df["game_id"].unique())
    train, val = df[df["game_id"].isin(ids[:30])], df[df["game_id"].isin(ids[30:])]
    results, (algo, params, est) = select_model(train, val, fast=True, log=lambda *_: None)
    assert len(results) == 2
    assert algo == min(results, key=lambda r: r["val_log_loss"])["algorithm"]
    assert all(0 <= r["val_accuracy"] <= 1 for r in results)


def test_model_roundtrip_and_valid_probabilities(tiny_bundle):
    X = np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=(20, N_FEATURES))
    p = predict_win_proba(tiny_bundle["model"], X)
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))
    assert tiny_bundle["algorithm"] == "logistic_regression"


def test_missing_model_file_raises_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="generate_dataset"):
        load_bundle(tmp_path / "nope.joblib")


def test_corrupt_or_wrong_model_file_rejected(tmp_path):
    p = tmp_path / "bad.joblib"
    p.write_bytes(b"not a model")
    with pytest.raises(ModelIncompatibleError):
        load_bundle(p)
    q = tmp_path / "wrong.joblib"
    joblib.dump({"model": object(), "feature_names": ["a", "b"]}, q)
    with pytest.raises(ModelIncompatibleError):
        load_bundle(q)


class NaNModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.full((len(X), 2), np.nan)


def test_invalid_predictions_are_sanitised():
    p = predict_win_proba(NaNModel(), np.zeros((3, N_FEATURES)))
    assert np.all(p == 0.5)
    agent = ModelAgent({"model": NaNModel(), "algorithm": "nan"}, "hard", random.Random(0))
    s = new_game()
    assert agent.choose(s) in s.legal_moves()


@pytest.mark.parametrize("difficulty", list(DIFFICULTIES))
def test_model_agent_always_returns_legal_moves(tiny_bundle, difficulty):
    agent = ModelAgent(tiny_bundle, difficulty, random.Random(1))
    rng = random.Random(2)
    s = new_game()
    while not s.is_terminal():
        if s.to_move == P1:
            d = agent.choose_with_info(s)
            assert d.move in s.legal_moves()
            assert 0 <= d.win_prob <= 1
            assert len(d.candidates) == len(s.legal_moves())
            assert d.candidates == sorted(d.candidates, key=lambda c: -c["p"])
            s, _ = apply_move(s, d.move)
        else:
            s, _ = apply_move(s, rng.choice(s.legal_moves()))


def test_model_agent_with_no_legal_moves(tiny_bundle):
    full = GameState(tuple([1] * N_CELLS), P1, tuple(range(N_CELLS)))
    agent = ModelAgent(tiny_bundle, "normal")
    with pytest.raises(ValueError):
        agent.choose(full)
    assert agent.position_win_prob(full, P1) == 1.0


def test_unknown_difficulty_rejected(tiny_bundle):
    with pytest.raises(ValueError):
        ModelAgent(tiny_bundle, "impossible")


def test_ai_takes_a_decisive_final_move(tiny_bundle):
    """With one empty cell left, the outcome is computed exactly, not guessed."""
    board = [1] * N_CELLS
    board[0] = 0
    s = GameState(tuple(board), P1, tuple(range(1, N_CELLS)))
    moves, probs = ModelAgent(tiny_bundle).evaluate_moves(s)
    assert moves == [0] and probs[0] == 1.0


def test_ai_only_sees_the_board(tiny_bundle, monkeypatch):
    """The model input is built from the public board alone: two identical boards
    reached by different move orders give identical AI ratings."""
    a = new_game()
    for m in (0, 1, 2):
        a, _ = apply_move(a, m)
    b = GameState(a.board, a.to_move, (2, 1, 0))
    agent = ModelAgent(tiny_bundle)
    assert np.allclose(agent.evaluate_moves(a)[1], agent.evaluate_moves(b)[1])


def test_softmax_sampling_prefers_better_moves():
    w = softmax_probabilities([0.2, 0.5, 0.8], temperature=0.5)
    assert w[2] > w[1] > w[0] and abs(sum(w) - 1) < 1e-9


def test_arena_and_wilson():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)
    res = run_arena(lambda g: make_agent("greedy", rng=random.Random(g)),
                    lambda g: make_agent("random", rng=random.Random(100 + g)), 4)
    assert res["wins"] + res["losses"] + res["draws"] == 4
    r = play_game(make_agent("random", rng=random.Random(1)), make_agent("random", rng=random.Random(2)), P1)
    assert r["result"] in (-1, 0, 1)


def test_classification_metrics_basic():
    m = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
