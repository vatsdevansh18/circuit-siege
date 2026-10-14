"""Local Flask server: hosts the UI and runs authoritative game sessions.

The browser never decides rules or AI moves. It sends the human's chosen cell,
and the server validates it with the engine. The AI move is computed here by
ModelAgent using the saved model.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ..ai.model_agent import DIFFICULTIES, ModelAgent
from ..ai.model_io import ModelIncompatibleError, ModelNotFoundError, load_bundle
from ..game.engine import CORES, P1, P2, GameState, IllegalMoveError, apply_move, flips_for, new_game
from ..utils import paths

STATIC_DIR = Path(__file__).parent / "static"
MAX_SESSIONS = 200


@dataclass
class Match:
    id: str
    difficulty: str
    human: int
    state: GameState
    agent: ModelAgent
    started_at: float = field(default_factory=time.time)
    events: list = field(default_factory=list)       # per-move log for the UI and the match file
    prob_trace: list = field(default_factory=list)   # model P(AI wins) after every move
    logged: bool = False

    @property
    def ai(self) -> int:
        return -self.human


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def create_app(model_path: Path | None = None, matches_path: Path | None = None,
               seed: int | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    model_path = Path(model_path or paths.MODEL_PATH)
    matches_path = Path(matches_path or paths.PLAYER_MATCHES)
    rng = random.Random(seed)
    lock = threading.Lock()
    matches: OrderedDict[str, Match] = OrderedDict()

    try:
        bundle, model_error = load_bundle(model_path), None
    except (ModelNotFoundError, ModelIncompatibleError) as exc:
        bundle, model_error = None, str(exc)

    # ---- helpers ---------------------------------------------------------------------
    def require_model():
        if bundle is None:
            raise ApiError(model_error or "model unavailable", 503)

    def get_match(data) -> Match:
        gid = data.get("game_id")
        if not isinstance(gid, str) or gid not in matches:
            raise ApiError("unknown or expired game_id", 404)
        return matches[gid]

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("request body must be a JSON object")
        return data

    def ai_prob(m: Match) -> float:
        return round(m.agent.position_win_prob(m.state, m.ai), 4)

    def side_name(m: Match, side: int | None) -> str | None:
        if side is None:
            return None
        return "human" if side == m.human else "ai" if side == m.ai else "draw"

    def serialize(m: Match, last=None) -> dict:
        st = m.state
        s = st.scores()
        over = st.is_terminal()
        human_turn = not over and st.to_move == m.human
        preview = ({str(i): flips_for(st.board, i, m.human) for i in st.legal_moves()}
                   if human_turn else {})
        return {
            "game_id": m.id,
            "difficulty": m.difficulty,
            "board": [v * m.human for v in st.board],  # +1 human, -1 AI, 0 empty
            "cores": sorted(CORES),
            "to_move": None if over else side_name(m, st.to_move),
            "ply": st.ply,
            "scores": {"human": s[m.human], "ai": s[m.ai]},
            "over": over,
            "winner": side_name(m, st.winner()) if over else None,
            "preview": preview,
            "ai_win_prob": m.prob_trace[-1] if m.prob_trace else 0.5,
            "prob_trace": m.prob_trace,
            "events": m.events[-60:],
            "last": last,
            "model_version": bundle["version"] if bundle else None,
        }

    def record_move(m: Match, by: str, cell: int, flipped: list[int], extra: dict | None = None) -> dict:
        p = ai_prob(m)
        m.prob_trace.append(p)
        ev = {"ply": m.state.ply, "by": by, "cell": cell, "flipped": flipped, "ai_win_prob": p}
        if extra:
            ev.update(extra)
        m.events.append(ev)
        if m.state.is_terminal():
            log_match(m)
        return ev

    def log_match(m: Match):
        if m.logged:
            return
        m.logged = True
        s = m.state.scores()
        first = m.human if m.events and m.events[0]["by"] == "human" else m.ai
        rec = {
            "match_id": m.id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(time.time() - m.started_at, 1),
            "difficulty": m.difficulty,
            "human_side": m.human,
            "first_player": first,
            "moves": list(m.state.history),
            "score_human": s[m.human],
            "score_ai": s[m.ai],
            "winner": side_name(m, m.state.winner()),
            "model_version": bundle["version"] if bundle else None,
            "ai_win_prob_trace": m.prob_trace,
        }
        try:
            matches_path.parent.mkdir(parents=True, exist_ok=True)
            with open(matches_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as exc:  # logging must never break the game
            app.logger.warning("could not log match: %s", exc)

    def read_history() -> list[dict]:
        if not matches_path.exists():
            return []
        out = []
        for line in matches_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a corrupted line instead of failing the stats panel
            if isinstance(rec, dict) and rec.get("winner") in ("human", "ai", "draw"):
                out.append(rec)
        return out

    # ---- routes ----------------------------------------------------------------------
    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return jsonify({"error": e.message}), e.status

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC_DIR, name)

    @app.get("/api/status")
    def status():
        info = None
        if bundle:
            info = {k: bundle.get(k) for k in ("algorithm", "params", "version", "trained_at",
                                               "train_rows", "train_games", "data_sources")}
            info["metrics"] = bundle.get("metrics", {})
            card = paths.MODEL_CARD
            if model_path == paths.MODEL_PATH and card.exists():
                try:
                    c = json.loads(card.read_text())
                    info["test_metrics"] = c.get("test_metrics")
                    info["arena"] = c.get("arena")
                except json.JSONDecodeError:
                    pass
        return jsonify({
            "model_loaded": bundle is not None,
            "error": model_error,
            "model": info,
            "difficulties": {k: {"label": v["label"], "blurb": v["blurb"]} for k, v in DIFFICULTIES.items()},
        })

    @app.post("/api/new_game")
    def new_match():
        require_model()
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ApiError("request body must be a JSON object")
        difficulty = data.get("difficulty", "normal")
        if difficulty not in DIFFICULTIES:
            raise ApiError(f"difficulty must be one of {list(DIFFICULTIES)}")
        first = data.get("first", "human")
        if first not in ("human", "ai", "random"):
            raise ApiError("first must be 'human', 'ai' or 'random'")
        if first == "random":
            first = rng.choice(["human", "ai"])
        human = P1  # the human always plays colour +1; who moves first varies
        st = new_game(human if first == "human" else P2)
        m = Match(uuid.uuid4().hex, difficulty, human, st,
                  ModelAgent(bundle, difficulty, random.Random(rng.random())))
        m.prob_trace.append(ai_prob(m))
        with lock:
            matches[m.id] = m
            while len(matches) > MAX_SESSIONS:
                matches.popitem(last=False)
        return jsonify(serialize(m))

    @app.get("/api/game/<gid>")
    def get_game(gid):
        require_model()
        return jsonify(serialize(get_match({"game_id": gid})))

    @app.post("/api/move")
    def human_move():
        require_model()
        data = body()
        with lock:
            m = get_match(data)
            cell = data.get("cell")
            if isinstance(cell, bool) or not isinstance(cell, int):
                raise ApiError("cell must be an integer 0-48")
            if m.state.is_terminal():
                raise ApiError("the game is over", 409)
            if m.state.to_move != m.human:
                raise ApiError("it is not your turn", 409)
            try:
                m.state, flipped = apply_move(m.state, cell)
            except IllegalMoveError as exc:
                raise ApiError(str(exc)) from exc
            ev = record_move(m, "human", cell, flipped)
            return jsonify(serialize(m, last=ev))

    @app.post("/api/ai_move")
    def ai_move():
        require_model()
        data = body()
        with lock:
            m = get_match(data)
            if m.state.is_terminal():
                raise ApiError("the game is over", 409)
            if m.state.to_move != m.ai:
                raise ApiError("it is not the AI's turn", 409)
            d = m.agent.choose_with_info(m.state)
            m.state, flipped = apply_move(m.state, d.move)
            ev = record_move(m, "ai", d.move, flipped, {
                "chosen_p": round(d.win_prob, 4),
                "candidates": d.candidates[:5],
                "all_candidates": d.candidates,
                "method": d.method,
                "think_ms": round(d.elapsed_ms, 1),
            })
            # Keep the full candidate list only on the response, not in the stored log.
            stored = dict(ev)
            stored.pop("all_candidates", None)
            m.events[-1] = stored
            return jsonify(serialize(m, last=ev))

    @app.get("/api/stats")
    def stats():
        hist = read_history()
        by_diff = {}
        for d in DIFFICULTIES:
            rs = [r for r in hist if r.get("difficulty") == d]
            by_diff[d] = {"played": len(rs), "human_wins": sum(r["winner"] == "human" for r in rs)}
        streak = 0
        for r in reversed(hist):
            if r["winner"] != "human":
                break
            streak += 1
        recent = [{k: r.get(k) for k in ("finished_at", "difficulty", "score_human", "score_ai", "winner")}
                  for r in hist[-10:]][::-1]
        return jsonify({
            "played": len(hist),
            "human_wins": sum(r["winner"] == "human" for r in hist),
            "ai_wins": sum(r["winner"] == "ai" for r in hist),
            "by_difficulty": by_diff,
            "win_streak": streak,
            "recent": recent,
        })

    return app
