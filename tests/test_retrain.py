import json
import random

from circuit_siege.game.engine import P1, P2
from circuit_siege.training.dataset import simulate_game
from circuit_siege.training.retrain import (
    load_player_matches,
    player_frame,
    should_promote,
    split_player_games,
    weights_for,
)


def write_matches(path, n, extra_lines=()):
    with open(path, "w") as f:
        for k in range(n):
            g = simulate_game(k, seed=3)
            human = random.Random(k).choice([P1, P2])
            f.write(json.dumps({
                "human_side": human, "first_player": g["first_player"], "moves": g["moves"],
                "winner": "human" if g["winner"] == human else "ai",
            }) + "\n")
        for line in extra_lines:
            f.write(line + "\n")


def test_load_player_matches_validates_and_skips_bad_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    bad_moves = json.dumps({"human_side": 1, "first_player": 1, "moves": [0] * 49})  # illegal replay
    write_matches(p, 3, extra_lines=["garbage", '{"human_side": 1}', bad_moves])
    recs, skipped = load_player_matches(p)
    assert len(recs) == 3 and skipped == 3
    assert load_player_matches(tmp_path / "missing.jsonl") == ([], 0)


def test_player_frame_has_both_perspectives_and_weights(tmp_path):
    p = tmp_path / "m.jsonl"
    write_matches(p, 2)
    recs, _ = load_player_matches(p)
    df = player_frame(recs)
    assert len(df) == 2 * 48 * 2
    assert set(df["source"]) == {"player"}
    w = weights_for(df, 3.0)
    assert (w == 3.0).all()


def test_split_only_holds_out_when_enough_games():
    recs = [{"game_id": i} for i in range(6)]
    train, hold = split_player_games(recs)
    assert len(train) == 6 and hold == []
    recs = [{"game_id": i} for i in range(50)]
    train, hold = split_player_games(recs)
    assert len(hold) == 10 and not ({r["game_id"] for r in train} & {r["game_id"] for r in hold})


def test_promotion_rule():
    cur = {"test_log_loss": 0.45}
    good_arena, bad_arena = {"win_rate": 0.55}, {"win_rate": 0.40}
    assert should_promote(cur, {"test_log_loss": 0.449}, good_arena)[0]
    assert not should_promote(cur, {"test_log_loss": 0.47}, good_arena)[0]
    assert not should_promote(cur, {"test_log_loss": 0.44}, bad_arena)[0]
    assert not should_promote({"test_log_loss": 0.45, "player_log_loss": 0.5},
                              {"test_log_loss": 0.45, "player_log_loss": 0.6}, good_arena)[0]


def test_duplicate_match_ids_are_skipped(tmp_path):
    p = tmp_path / "m.jsonl"
    write_matches(p, 1)
    line = json.loads(p.read_text().strip())
    line["match_id"] = "abc"
    p.write_text((json.dumps(line) + "\n") * 2 + "[1, 2]\n")
    recs, skipped = load_player_matches(p)
    assert len(recs) == 1 and skipped == 2
