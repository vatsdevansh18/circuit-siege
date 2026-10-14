"""Flask server: hosts the UI and runs the game rules and the AI.

The server is stateless. The browser keeps the match as a small record
(id, difficulty, who moved first, list of moves) and sends it with every
request. The server replays the moves with the engine, which also re-checks
that every move is legal, and then applies the new move. Because nothing is kept
in memory between requests, this works the same on one local process or on
many serverless instances (Vercel).

The browser never decides rules or AI moves. The AI move is computed here by
ModelAgent using the saved model.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ..ai.features import extract_features
from ..ai.model_agent import DIFFICULTIES, ModelAgent
from ..ai.model_io import ModelIncompatibleError, ModelNotFoundError, load_bundle, predict_win_proba
from ..game.engine import CORES, N_CELLS, P1, P2, IllegalMoveError, apply_move, flips_for, new_game
from ..utils import paths

STATIC_DIR = Path(__file__).parent / "static"
HUMAN, AI = P1, P2  # the human always plays colour +1; who moves first varies


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def create_app(model_path: Path | None = None, matches_path: Path | None = None,
               seed: int | None = None, log_matches: bool | None = None) -> Flask:
    """`log_matches` defaults to True locally and False on Vercel (read-only filesystem)."""
    app = Flask(__name__, static_folder=None)
    model_path = Path(model_path or paths.MODEL_PATH)
    matches_path = Path(matches_path or paths.PLAYER_MATCHES)
    if log_matches is None:
        log_matches = not os.environ.get("VERCEL")
    rng = random.Random(seed)

    try:
        bundle, model_error = load_bundle(model_path), None
    except (ModelNotFoundError, ModelIncompatibleError) as exc:
        bundle, model_error = None, str(exc)

    # ---- helpers ---------------------------------------------------------------------
    def require_model():
        if bundle is None:
            raise ApiError(model_error or "model unavailable", 503)

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("request body must be a JSON object")
        return data

    def parse_match(data: dict) -> dict:
        """Validate the match record sent by the browser."""
        m = data.get("match")
        if not isinstance(m, dict):
            raise ApiError("missing 'match' record; start a new game")
        mid, diff, first, moves = m.get("id"), m.get("difficulty"), m.get("first"), m.get("moves")
        if not isinstance(mid, str) or not 0 < len(mid) <= 64:
            raise ApiError("match.id must be a short string")
        if diff not in DIFFICULTIES:
            raise ApiError(f"match.difficulty must be one of {list(DIFFICULTIES)}")
        if first not in ("human", "ai"):
            raise ApiError("match.first must be 'human' or 'ai'")
        if not isinstance(moves, list) or len(moves) > N_CELLS:
            raise ApiError("match.moves must be a list of at most 49 cells")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in moves):
            raise ApiError("match.moves must contain integers")
        return {"id": mid, "difficulty": diff, "first": first, "moves": list(moves)}

    def replay(match: dict):
        """Rebuild every state of the match. Raises ApiError if any move is illegal."""
        state = new_game(HUMAN if match["first"] == "human" else AI)
        states, flips = [state], []
        for cell in match["moves"]:
            try:
                state, flipped = apply_move(state, cell)
            except IllegalMoveError as exc:
                raise ApiError(f"match history is invalid: {exc}") from exc
            states.append(state)
            flips.append(flipped)
        return states, flips

    def prob_trace(states) -> list[float]:
        """Model's P(AI wins) after every move, scored in a single batch."""
        out = [0.0] * len(states)
        rows, idx = [], []
        for k, st in enumerate(states):
            if st.is_terminal():
                w = st.winner()
                out[k] = 1.0 if w == AI else 0.0 if w == HUMAN else 0.5
            else:
                rows.append(extract_features(st.board, st.to_move, AI))
                idx.append(k)
        if rows:
            for k, p in zip(idx, predict_win_proba(bundle["model"], rows)):
                out[k] = round(float(p), 4)
        return out

    def side_name(side: int | None) -> str | None:
        if side is None:
            return None
        return "human" if side == HUMAN else "ai" if side == AI else "draw"

    def serialize(match: dict, states, flips, last=None) -> dict:
        st = states[-1]
        s = st.scores()
        over = st.is_terminal()
        human_turn = not over and st.to_move == HUMAN
        trace = prob_trace(states)
        events = [{"ply": k + 1, "by": side_name(states[k].to_move), "cell": cell, "flipped": flips[k],
                   "ai_win_prob": trace[k + 1]}
                  for k, cell in enumerate(match["moves"])]
        return {
            "match": match,
            "difficulty": match["difficulty"],
            "board": [v * HUMAN for v in st.board],  # +1 human, -1 AI, 0 empty
            "cores": sorted(CORES),
            "to_move": None if over else side_name(st.to_move),
            "ply": st.ply,
            "scores": {"human": s[HUMAN], "ai": s[AI]},
            "over": over,
            "winner": side_name(st.winner()) if over else None,
            "preview": {str(i): flips_for(st.board, i, HUMAN) for i in st.legal_moves()} if human_turn else {},
            "ai_win_prob": trace[-1],
            "prob_trace": trace,
            "events": events,
            "last": last,
            "model_version": bundle["version"],
        }

    def log_match(match: dict, states, trace):
        """Append a finished match to the local log used for retraining."""
        if not log_matches:
            return
        st = states[-1]
        s = st.scores()
        rec = {
            "match_id": match["id"],
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "difficulty": match["difficulty"],
            "human_side": HUMAN,
            "first_player": HUMAN if match["first"] == "human" else AI,
            "moves": match["moves"],
            "score_human": s[HUMAN],
            "score_ai": s[AI],
            "winner": side_name(st.winner()),
            "model_version": bundle["version"],
            "ai_win_prob_trace": trace,
        }
        try:
            matches_path.parent.mkdir(parents=True, exist_ok=True)
            with open(matches_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as exc:  # logging must never break the game
            app.logger.warning("could not log match: %s", exc)

    def respond(match, states, flips, last):
        payload = serialize(match, states, flips, last)
        if payload["over"]:
            log_match(match, states, payload["prob_trace"])
        return jsonify(payload)

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
            "logging_matches": log_matches,
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
        match = {"id": uuid.uuid4().hex, "difficulty": difficulty, "first": first, "moves": []}
        states, flips = replay(match)
        return jsonify(serialize(match, states, flips))

    @app.post("/api/move")
    def human_move():
        require_model()
        data = body()
        match = parse_match(data)
        cell = data.get("cell")
        if isinstance(cell, bool) or not isinstance(cell, int):
            raise ApiError("cell must be an integer 0-48")
        states, flips = replay(match)
        st = states[-1]
        if st.is_terminal():
            raise ApiError("the game is over", 409)
        if st.to_move != HUMAN:
            raise ApiError("it is not your turn", 409)
        try:
            nxt, flipped = apply_move(st, cell)
        except IllegalMoveError as exc:
            raise ApiError(str(exc)) from exc
        match["moves"].append(cell)
        states.append(nxt)
        flips.append(flipped)
        last = {"ply": nxt.ply, "by": "human", "cell": cell, "flipped": flipped}
        return respond(match, states, flips, last)

    @app.post("/api/ai_move")
    def ai_move():
        require_model()
        match = parse_match(body())
        states, flips = replay(match)
        st = states[-1]
        if st.is_terminal():
            raise ApiError("the game is over", 409)
        if st.to_move != AI:
            raise ApiError("it is not the AI's turn", 409)
        agent = ModelAgent(bundle, match["difficulty"], random.Random(rng.random()))
        d = agent.choose_with_info(st)
        nxt, flipped = apply_move(st, d.move)
        match["moves"].append(d.move)
        states.append(nxt)
        flips.append(flipped)
        last = {
            "ply": nxt.ply, "by": "ai", "cell": d.move, "flipped": flipped,
            "chosen_p": round(d.win_prob, 4),
            "candidates": d.candidates[:5],
            "all_candidates": d.candidates,
            "method": d.method,
            "think_ms": round(d.elapsed_ms, 1),
        }
        return respond(match, states, flips, last)

    return app
