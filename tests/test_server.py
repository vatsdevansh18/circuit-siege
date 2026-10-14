import json
import random

import pytest

from circuit_siege.web.server import create_app


def make_client(model_path, matches_path, **kw):
    app = create_app(model_path=model_path, matches_path=matches_path, seed=0, **kw)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def client(tiny_model_path, tmp_path):
    c = make_client(tiny_model_path, tmp_path / "matches.jsonl", log_matches=True)
    c.matches_path = tmp_path / "matches.jsonl"
    return c


def step(clients, game, rng):
    """One move by whoever is to play, sent to a randomly chosen server instance."""
    c = rng.choice(clients)
    if game["to_move"] == "human":
        empties = [i for i, v in enumerate(game["board"]) if v == 0]
        r = c.post("/api/move", json={"match": game["match"], "cell": rng.choice(empties)})
    else:
        r = c.post("/api/ai_move", json={"match": game["match"]})
    assert r.status_code == 200, r.json
    return r.json


def play_out(clients, game, rng):
    while not game["over"]:
        game = step(clients, game, rng)
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
    assert g["match"]["moves"] == [] and g["match"]["first"] == "human"
    g = play_out([client], g, random.Random(1))
    assert g["over"] and g["winner"] in ("human", "ai") and g["to_move"] is None
    assert g["scores"]["human"] + g["scores"]["ai"] == 59
    assert len(g["prob_trace"]) == 50 and len(g["events"]) == 49
    rec = json.loads(client.matches_path.read_text().strip())
    assert rec["winner"] == g["winner"] and rec["moves"] == g["match"]["moves"]


def test_stateless_across_server_instances(tiny_model_path, tmp_path):
    """Like serverless hosting: consecutive requests hit different app instances."""
    clients = [make_client(tiny_model_path, tmp_path / f"m{k}.jsonl", log_matches=False) for k in range(3)]
    g = clients[0].post("/api/new_game", json={"first": "ai"}).json
    g = play_out(clients, g, random.Random(3))
    assert g["over"] and len(g["match"]["moves"]) == 49
    assert not any((tmp_path / f"m{k}.jsonl").exists() for k in range(3))  # logging disabled


def test_logging_off_by_default_on_vercel(tiny_model_path, tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    c = make_client(tiny_model_path, tmp_path / "m.jsonl")
    assert c.get("/api/status").json["logging_matches"] is False
    g = play_out([c], c.post("/api/new_game", json={}).json, random.Random(4))
    assert g["over"] and not (tmp_path / "m.jsonl").exists()


def test_ai_first_and_decision_payload(client):
    g = client.post("/api/new_game", json={"difficulty": "hard", "first": "ai"}).json
    assert g["to_move"] == "ai" and g["preview"] == {}
    g = client.post("/api/ai_move", json={"match": g["match"]}).json
    last = g["last"]
    assert last["by"] == "ai" and 0 <= last["cell"] < 49
    assert len(last["all_candidates"]) == 49 and 0 <= last["chosen_p"] <= 1
    assert g["to_move"] == "human" and g["board"][last["cell"]] == -1
    assert g["match"]["moves"] == [last["cell"]]


def test_wrong_turn_and_illegal_moves(client):
    g = client.post("/api/new_game", json={"first": "human"}).json
    assert client.post("/api/ai_move", json={"match": g["match"]}).status_code == 409
    g = client.post("/api/move", json={"match": g["match"], "cell": 10}).json
    assert client.post("/api/move", json={"match": g["match"], "cell": 11}).status_code == 409  # AI's turn
    g = client.post("/api/ai_move", json={"match": g["match"]}).json
    assert client.post("/api/move", json={"match": g["match"], "cell": 10}).status_code == 400  # occupied


def test_tampered_history_rejected(client):
    g = client.post("/api/new_game", json={"first": "human"}).json
    m = dict(g["match"], moves=[5, 5])  # second move on an occupied cell
    r = client.post("/api/ai_move", json={"match": m})
    assert r.status_code == 400 and "invalid" in r.json["error"]


@pytest.mark.parametrize("match", [
    None, [], {"id": "x", "difficulty": "normal", "first": "human"},
    {"id": "x", "difficulty": "godlike", "first": "human", "moves": []},
    {"id": "x", "difficulty": "normal", "first": "nobody", "moves": []},
    {"id": "", "difficulty": "normal", "first": "human", "moves": []},
    {"id": "x", "difficulty": "normal", "first": "human", "moves": ["1"]},
    {"id": "x", "difficulty": "normal", "first": "human", "moves": list(range(50))},
])
def test_bad_match_records(client, match):
    r = client.post("/api/move", json={"match": match, "cell": 3})
    assert r.status_code == 400 and "error" in r.json


@pytest.mark.parametrize("cell", ["3", 3.5, None, -1, 49, True, [1]])
def test_garbage_cells_rejected(client, cell):
    g = client.post("/api/new_game", json={"first": "human"}).json
    r = client.post("/api/move", json={"match": g["match"], "cell": cell})
    assert r.status_code == 400 and "error" in r.json


def test_bad_new_game_options_and_bodies(client):
    assert client.post("/api/new_game", json={"difficulty": "godlike"}).status_code == 400
    assert client.post("/api/new_game", json={"first": "nobody"}).status_code == 400
    assert client.post("/api/move", data="not json", content_type="application/json").status_code == 400
    assert client.post("/api/move", json=[1, 2]).status_code == 400


def test_moves_after_game_over_rejected(client):
    g = client.post("/api/new_game", json={"first": "human"}).json
    g = play_out([client], g, random.Random(5))
    assert client.post("/api/move", json={"match": g["match"], "cell": 0}).status_code == 409
    assert client.post("/api/ai_move", json={"match": g["match"]}).status_code == 409


def test_restart_creates_independent_game(client):
    a = client.post("/api/new_game", json={"first": "human"}).json
    a = client.post("/api/move", json={"match": a["match"], "cell": 0}).json
    b = client.post("/api/new_game", json={"first": "human"}).json
    assert b["match"]["id"] != a["match"]["id"] and b["ply"] == 0 and all(v == 0 for v in b["board"])
    assert a["ply"] == 1


def test_missing_model_gives_clear_503(tmp_path):
    c = make_client(tmp_path / "missing.joblib", tmp_path / "m.jsonl")
    s = c.get("/api/status").json
    assert s["model_loaded"] is False and "train_model.py" in s["error"]
    r = c.post("/api/new_game", json={})
    assert r.status_code == 503 and "train_model.py" in r.json["error"]
    assert c.get("/").status_code == 200  # the UI still loads and explains the problem


def test_wsgi_entry_point_exposes_app():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("vercel_entry", Path(__file__).resolve().parents[1] / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.app.test_client().get("/api/status").status_code == 200
