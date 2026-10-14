"""Flask server: hosts the UI and runs the game rules and the AI.

The server is stateless. The browser keeps the match as a small record
(id, difficulty, who moved first, list of moves) and sends it with every
request. The server replays the moves with the engine, which also re-checks
that every move is legal, and then applies the new move. Because nothing is kept
in memory between requests, this works the same on one local process or on
many serverless instances (Vercel).

The match record is signed with an HMAC, so the browser can't hand the server a
forged history (for example, "AI" moves it chose itself) that would then be
logged as a real game and used for retraining. The key comes from the
CIRCUIT_SIEGE_SECRET environment variable; without it, a local run uses a random
key per process. On Vercel without the variable, records are unsigned, but
nothing is logged there, so a forged history only affects the forger's own game.

The browser never decides rules or AI moves. The AI move is computed here by
ModelAgent using the saved model.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ..ai.features import extract_features
from ..ai.model_agent import DIFFICULTIES, ModelAgent
from ..ai.model_io import ModelIncompatibleError, ModelNotFoundError, load_bundle, predict_win_proba
from ..game.engine import CORES, N_CELLS, P1, P2, IllegalMoveError, apply_move, flips_for, new_game
from ..utils import paths

STATIC_DIR = Path(__file__).parent / "static"
HUMAN, AI = P1, P2  # the human always plays colour +1; who moves first varies
MAX_TRACKED_MATCHES = 10_000  # newest-record map size (logging mode); oldest matches drop out first


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


_AUTO = object()


def create_app(model_path: Path | None = None, matches_path: Path | None = None,
               seed: int | None = None, log_matches: bool | None = None, secret=_AUTO) -> Flask:
    """`log_matches` defaults to True locally and False on Vercel (read-only filesystem).

    `secret` signs match records. By default it comes from CIRCUIT_SIEGE_SECRET, or a
    random per-process key when running locally. None disables signing (tests only).
    """
    app = Flask(__name__, static_folder=None)
    model_path = Path(model_path or paths.MODEL_PATH)
    matches_path = Path(matches_path or paths.PLAYER_MATCHES)
    on_vercel = bool(os.environ.get("VERCEL"))
    if log_matches is None:
        log_matches = not on_vercel
    if secret is _AUTO:
        secret = os.environ.get("CIRCUIT_SIEGE_SECRET") or (None if on_vercel else secrets.token_hex(32))
    key = secret.encode() if isinstance(secret, str) else secret
    if log_matches and key is None:
        raise ValueError("logging matches requires signed match records; provide a secret")
    rng = random.Random(seed)

    # Match ids already in the log, so a finished match is written only once even
    # if its final request is sent again.
    logged_ids: set[str] = set()
    if log_matches and matches_path.exists():
        for line in matches_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("match_id"), str):
                logged_ids.add(rec["match_id"])

    # Where matches are logged (local runs), only the newest record issued for each
    # match is accepted. Older records still carry valid signatures, so without this a
    # player could rewind (take back moves) and the log would contain games that were
    # never played straight through. The lock makes check-and-update atomic. The map
    # lives in memory, so matches that were in progress when the server restarted (or
    # that dropped out of the map) are refused and the player starts a new game.
    latest: OrderedDict[str, str] = OrderedDict()
    lock = threading.Lock()

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

    def parse_match(data: dict) -> tuple[dict, str | None]:
        """Validate the match record sent by the browser. Returns (match, signature)."""
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
        clean = {"id": mid, "difficulty": diff, "first": first, "moves": list(moves)}
        if key is None:
            return clean, None
        sig = m.get("sig")
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str input.
        if not isinstance(sig, str) or not hmac.compare_digest(sig.encode("utf-8"), sign(clean).encode()):
            raise ApiError("match record was modified or comes from another server; start a new game", 403)
        if log_matches:
            newest = latest.get(mid)
            if newest is None:
                raise ApiError("this game is no longer active on the server (it may have restarted); "
                               "start a new game", 409)
            if not hmac.compare_digest(newest.encode(), sig.encode("utf-8")):
                raise ApiError("this is an old game state (the move was already played); "
                               "start a new game", 409)
        return clean, sig

    def remember(record: dict):
        """Record the newest signed state of a match (logging mode only)."""
        if not log_matches or "sig" not in record:
            return
        latest[record["id"]] = record["sig"]
        latest.move_to_end(record["id"])
        while len(latest) > MAX_TRACKED_MATCHES:
            latest.popitem(last=False)

    def ai_rng(sig: str | None) -> random.Random:
        """The AI's sampling randomness, derived from the signed state when signing is on,
        so sending the same state again can't 're-roll' the AI's choice."""
        if key is None or sig is None:
            return random.Random(rng.random())
        digest = hmac.new(key, b"ai-move:" + sig.encode(), hashlib.sha256).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def sign(match: dict) -> str:
        payload = json.dumps(match, sort_keys=True, separators=(",", ":"))
        return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()

    def signed(match: dict) -> dict:
        out = {k: match[k] for k in ("id", "difficulty", "first", "moves")}
        if key is not None:
            out["sig"] = sign(out)
        return out

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
        record = signed(match)
        remember(record)
        return {
            "match": record,
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
        if not log_matches or match["id"] in logged_ids:
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
            logged_ids.add(match["id"])
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
        with lock:
            return _human_move(data)

    def _human_move(data):
        match, _ = parse_match(data)
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
        data = body()
        with lock:
            return _ai_move(data)

    def _ai_move(data):
        match, sig = parse_match(data)
        states, flips = replay(match)
        st = states[-1]
        if st.is_terminal():
            raise ApiError("the game is over", 409)
        if st.to_move != AI:
            raise ApiError("it is not the AI's turn", 409)
        agent = ModelAgent(bundle, match["difficulty"], ai_rng(sig))
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
