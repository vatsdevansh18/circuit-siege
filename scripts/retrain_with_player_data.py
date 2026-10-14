"""Adaptive mode: retrain the value model with your logged matches (offline).

    python scripts/retrain_with_player_data.py              # needs >= 5 finished matches
    python scripts/retrain_with_player_data.py --dry-run    # evaluate, never promote
    python scripts/retrain_with_player_data.py --min-games 1 --arena-games 20

The current model is only replaced if the candidate passes every check. The old
model is archived in models/archive/ first, so you can always roll back by copying
it back to models/value_model.joblib.
"""

import argparse
import json
import random
import shutil
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from circuit_siege.ai.model_agent import ModelAgent
from circuit_siege.ai.model_io import load_bundle, predict_win_proba, save_bundle
from circuit_siege.evaluation.arena import run_arena
from circuit_siege.training.retrain import (
    load_player_matches,
    player_frame,
    should_promote,
    split_player_games,
    weights_for,
)
from circuit_siege.training.train import load_split_frames, refit, score, xy
from circuit_siege.utils import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--player-weight", type=float, default=3.0,
                    help="sample weight of each player-game row relative to a synthetic row")
    ap.add_argument("--arena-games", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--matches", default=str(paths.PLAYER_MATCHES), help="match log to learn from")
    args = ap.parse_args()

    current = load_bundle(paths.MODEL_PATH)
    records, skipped = load_player_matches(args.matches)
    print(f"Player matches: {len(records)} valid, {skipped} skipped (incomplete or corrupted)")
    if len(records) < args.min_games:
        raise SystemExit(f"Need at least {args.min_games} finished matches to retrain; "
                         f"play more games or pass --min-games.")

    train_recs, hold_recs = split_player_games(records, seed=args.seed)
    frames = load_split_frames(paths.POSITIONS, paths.SPLITS)
    synthetic = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    player_train = player_frame(train_recs, seed=args.seed)
    combined = pd.concat([synthetic, player_train], ignore_index=True)
    print(f"Training rows: {len(synthetic):,} synthetic + {len(player_train):,} from {len(train_recs)} player games "
          f"(weight {args.player_weight}); held-out player games: {len(hold_recs)}")

    t0 = time.time()
    candidate_model = refit(current["algorithm"], current["params"], combined, seed=args.seed,
                            sample_weight=weights_for(combined, args.player_weight))
    print(f"Candidate trained in {time.time() - t0:.0f}s")

    Xt, yt = xy(frames["test"])
    cur = {"test_log_loss": score(current["model"], Xt, yt)["log_loss"]}
    cand = {"test_log_loss": score(candidate_model, Xt, yt)["log_loss"]}
    if hold_recs:
        Xp, yp = xy(player_frame(hold_recs, seed=args.seed))
        cur["player_log_loss"] = score(current["model"], Xp, yp)["log_loss"]
        cand["player_log_loss"] = score(candidate_model, Xp, yp)["log_loss"]
    print(f"Current:   {cur}\nCandidate: {cand}")

    cand_bundle = dict(current, model=candidate_model)
    arena = run_arena(lambda g: ModelAgent(cand_bundle, "normal", random.Random(args.seed + g)),
                      lambda g: ModelAgent(current, "normal", random.Random(10_000 + args.seed + g)),
                      args.arena_games)
    print(f"Head-to-head (candidate vs current, Operator level): {arena['win_rate']:.0%} {arena['ci95']}")

    ok, reason = should_promote(cur, cand, arena)
    promoted = ok and not args.dry_run
    if promoted:
        paths.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archived = paths.ARCHIVE_DIR / f"value_model_{current['version']}.joblib"
        shutil.copy2(paths.MODEL_PATH, archived)
        new = save_bundle(paths.MODEL_PATH, candidate_model, current["algorithm"], current["params"],
                          metrics={"synthetic_test_log_loss": cand["test_log_loss"]},
                          extra={"train_rows": len(combined),
                                 "train_games": int(combined["game_id"].nunique()),
                                 "data_sources": ["selfplay", "player"],
                                 "player_games": len(train_recs),
                                 "previous_version": current["version"], "seed": args.seed})
        card = {k: v for k, v in new.items() if k != "model"}
        card["feature_names"] = f"{len(new['feature_names'])} features"
        paths.MODEL_CARD.write_text(json.dumps(card, indent=2))
        print(f"PROMOTED new model {new['version']}; previous model archived at {archived}")
    else:
        print(f"Not promoted: {reason if not ok else 'dry run'}. Current model unchanged.")

    report = [
        "# Retraining Report", "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Player matches used for training: {len(train_recs)} (held out: {len(hold_recs)}, skipped: {skipped})",
        f"- Player-row weight: {args.player_weight}",
        f"- Synthetic test log-loss: current {cur['test_log_loss']:.4f}, candidate {cand['test_log_loss']:.4f}",
    ]
    if hold_recs:
        report.append(f"- Held-out player log-loss: current {cur['player_log_loss']:.4f}, "
                      f"candidate {cand['player_log_loss']:.4f}")
    report += [f"- Head-to-head candidate vs current: {arena['wins']}-{arena['losses']} "
               f"({arena['win_rate']:.0%}, 95% CI {arena['ci95'][0]:.0%}–{arena['ci95'][1]:.0%})",
               f"- Decision: **{'promoted' if promoted else 'kept current model'}** ({reason}"
               f"{', dry run' if args.dry_run else ''})"]
    paths.REPORTS_DIR.mkdir(exist_ok=True)
    (paths.REPORTS_DIR / "retrain_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
