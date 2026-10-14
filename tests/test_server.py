import json
import random

import pytest

from circuit_siege.web.server import create_app


@pytest.fixture
def client(tiny_model_path, tmp_path):
    app = create_app(model_path=tiny_model_path, matches_path=tmp_path / "matches.jsonl", seed=0)
    app.config["TESTING"] = True
    c = app.test_client()
    c.matches_path = tmp_path / "matches.jsonl"
    return c


def play_out(client, game, rng):
    while not game["over"]:
        if game["to_move"] == "human":
            empties = [i for i, v in enumerate(game["board"]) if v == 0]
            r = client.post("/api/move", json={"game_id": game["game_id"], "cell": rng.choice(empties)})
        else:
            r = client.post("/api/ai_move", json={"game_id": game["game_id"]})
        assert r.status_code == 200, r.json
        game = r.json
    return game


def test_index_and_status(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    s = client.get("/api/status").json
    assert s["model_loaded"] is True and s["error"] is None
    assert set(s["difficulties"]) == {"easy", "normal", "hard"}


def test_full_match_flow_logs_result(client):
    g = client.post("/api/new_game", json={"difficulty": "normal", "first": "human"}).json
    assert g["to_move"] == "human" and g["ply"] == 0 and len(g["preview"]) == 49
    g = play_out(client, g, random.Random(1))
    assert g["over"] and g["winner"] in ("human", "ai") and g["to_move"] is None
    assert g["scores"]["human"] + g["scores"]["ai"] == 59
    assert len(g["prob_trace"]) == 50
    rec = json.loads(client.matches_path.read_text().strip())
    assert rec["winner"] == g["winner"] and len(rec["moves"]) == 49
    stats = client.get("/api/stats").json
    assert stats["played"] == 1 and stats["human_wins"] + stats["ai_wins"] == 1


def test_ai_first_and_decision_payload(client):
    g = client.post("/api/new_game", json={"difficulty": "hard", "first": "ai"}).json
    assert g["to_move"] == "ai" and g["preview"] == {}
    g = client.post("/api/ai_move", json={"game_id": g["game_id"]}).json
    last = g["last"]
    assert last["by"] == "ai" and 0 <= last["cell"] < 49
    assert len(last["all_candidates"]) == 49 and 0 <= last["chosen_p"] <= 1
    assert g["to_move"] == "human" and g["board"][last["cell"]] == -1


def test_wrong_turn_and_illegal_moves(client):
    g = client.post("/api/new_game", json={"first": "human"}).json
    gid = g["game_id"]
    assert client.post("/api/ai_move", json={"game_id": gid}).status_code == 409
    assert client.post("/api/move", json={"game_id": gid, "cell": 10}).status_code == 200
    assert client.post("/api/move", json={"game_id": gid, "cell": 11}).status_code == 409  # AI's turn
    client.post("/api/ai_move", json={"game_id": gid})
    assert client.post("/api/move", json={"game_id": gid, "cell": 10}).status_code == 400  # occupied


@pytest.mark.parametrize("payload", [{"cell": 3}, {"game_id": "nope", "cell": 3}])
def test_unknown_game(client, payload):
    assert client.post("/api/move", json=payload).status_code == 404


@pytest.mark.parametrize("cell", ["3", 3.5, None, -1, 49, True, [1]])
def test_garbage_cells_rejected(client, cell):
    g = client.post("/api/new_game", json={"first": "human"}).json
    r = client.post("/api/move", json={"game_id": g["game_id"], "cell": cell})
    assert r.status_code == 400 and "error" in r.json


def test_bad_new_game_options_and_bodies(client):
    assert client.post("/api/new_game", json={"difficulty": "godlike"}).status_code == 400
    assert client.post("/api/new_game", json={"first": "nobody"}).status_code == 400
    assert client.post("/api/move", data="not json", content_type="application/json").status_code == 400
    assert client.post("/api/move", json=[1, 2]).status_code == 400


def test_moves_after_game_over_rejected(client):
    g = client.post("/api/new_game", json={"first": "human"}).json
    g = play_out(client, g, random.Random(5))
    assert client.post("/api/move", json={"game_id": g["game_id"], "cell": 0}).status_code == 409
    assert client.post("/api/ai_move", json={"game_id": g["game_id"]}).status_code == 409


def test_restart_creates_independent_game(client):
    a = client.post("/api/new_game", json={"first": "human"}).json
    client.post("/api/move", json={"game_id": a["game_id"], "cell": 0})
    b = client.post("/api/new_game", json={"first": "human"}).json
    assert b["game_id"] != a["game_id"] and b["ply"] == 0 and all(v == 0 for v in b["board"])
    assert client.get(f"/api/game/{a['game_id']}").json["ply"] >= 1


def test_missing_model_gives_clear_503(tmp_path):
    app = create_app(model_path=tmp_path / "missing.joblib", matches_path=tmp_path / "m.jsonl")
    c = app.test_client()
    s = c.get("/api/status").json
    assert s["model_loaded"] is False and "train_model.py" in s["error"]
    r = c.post("/api/new_game", json={})
    assert r.status_code == 503 and "train_model.py" in r.json["error"]
    assert c.get("/").status_code == 200  # the UI still loads and explains the problem


def test_corrupted_history_lines_are_skipped(client):
    client.matches_path.write_text('{"winner": "human", "difficulty": "easy"}\nnot json\n{"oops": 1}\n')
    s = client.get("/api/stats").json
    assert s["played"] == 1 and s["human_wins"] == 1
