"""Train the Circuit Siege value model.

    python scripts/train_model.py            # full model selection (several minutes)
    python scripts/train_model.py --fast     # small grid, for quick checks

Steps: load positions -> split by game (splits.json) -> fit candidate models on
train -> pick lowest validation log-loss -> refit winner on train+val -> save
models/value_model.joblib + models/model_card.json + reports/model_selection.md.
The test split is NOT touched here; scripts/evaluate_model.py uses it.
"""

import argparse
import json
import time

import pandas as pd

import _bootstrap  # noqa: F401

from circuit_siege.ai.model_io import save_bundle
from circuit_siege.training.train import load_split_frames, refit, select_model
from circuit_siege.utils import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="tiny grid for smoke tests")
    args = ap.parse_args()

    if not paths.POSITIONS.exists():
        raise SystemExit("Dataset not found. Run: python scripts/generate_dataset.py")
    t0 = time.time()
    frames = load_split_frames(paths.POSITIONS, paths.SPLITS)
    print({k: f"{len(v):,} rows / {v['game_id'].nunique()} games" for k, v in frames.items()})

    print("Model selection on validation split:")
    results, (algo, params, _) = select_model(frames["train"], frames["val"], seed=args.seed, fast=args.fast)
    print(f"Selected: {algo} {params}")

    train_val = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    model = refit(algo, params, train_val, seed=args.seed)
    best_row = min(results, key=lambda r: r["val_log_loss"])
    bundle = save_bundle(
        paths.MODEL_PATH, model, algo, params,
        metrics={"validation": {k: v for k, v in best_row.items() if k.startswith("val_")}},
        extra={"train_rows": len(train_val), "train_games": int(train_val["game_id"].nunique()),
               "seed": args.seed, "data_sources": ["selfplay"]},
    )
    card = {k: v for k, v in bundle.items() if k != "model"}
    card["feature_names"] = f"{len(bundle['feature_names'])} features (see src/circuit_siege/ai/features.py)"
    paths.MODEL_CARD.write_text(json.dumps(card, indent=2))

    sel = pd.DataFrame(results).sort_values("val_log_loss")
    paths.REPORTS_DIR.mkdir(exist_ok=True)
    sel.to_csv(paths.REPORTS_DIR / "model_selection.csv", index=False)
    md = ["# Model Selection", "",
          "Every candidate was fitted on the **train** games and scored on the **validation** games.",
          "The model with the lowest validation log-loss was selected and refitted on train + validation.",
          "`predict_48_ms` is the time to score 48 positions, roughly one full AI turn.", "",
          "| algorithm | params | val log-loss | val accuracy | val ROC-AUC | fit s | predict_48 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in sel.to_dict("records"):
        md.append(f"| {r['algorithm']} | `{r['params']}` | {r['val_log_loss']:.4f} | {r['val_accuracy']:.4f} | "
                  f"{r['val_roc_auc']:.4f} | {r['fit_seconds']} | {r['predict_48_ms']} |")
    md += ["", f"**Selected:** `{algo}` with `{json.dumps(params)}`. Model version `{bundle['version']}`."]
    (paths.REPORTS_DIR / "model_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {paths.MODEL_PATH} ({paths.MODEL_PATH.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
