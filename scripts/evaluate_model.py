"""Evaluate the trained value model on held-out games and in head-to-head play.

    python scripts/evaluate_model.py                 # full evaluation
    python scripts/evaluate_model.py --arena-games 40 # quicker arena

Part 1 - prediction quality on the TEST games (never used for training/selection):
    accuracy, precision, recall, F1, ROC-AUC, log-loss, Brier, confusion matrix,
    accuracy by game phase, calibration, permutation feature importance,
    compared with a majority-class baseline and a "score leader wins" baseline.
Part 2 - playing strength (reward-based): the model-driven AI at every difficulty
    plays each scripted bot; win rate with 95% Wilson confidence interval.
Outputs: reports/evaluation_report.md, reports/metrics.json, reports/figures/*.png
"""

import argparse
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

import _bootstrap  # noqa: F401,E402

from circuit_siege.ai.agents import make_agent  # noqa: E402
from circuit_siege.ai.features import FEATURE_NAMES  # noqa: E402
from circuit_siege.ai.model_agent import ModelAgent  # noqa: E402
from circuit_siege.ai.model_io import load_bundle, predict_win_proba  # noqa: E402
from circuit_siege.evaluation.arena import run_arena  # noqa: E402
from circuit_siege.evaluation.metrics import by_phase, classification_metrics, score_leader_scores  # noqa: E402
from circuit_siege.training.train import load_split_frames, xy  # noqa: E402
from circuit_siege.utils import paths  # noqa: E402

# Chart colours: model = blue, baseline = orange.
C_MODEL, C_BASE, C_MUTED = "#2a78d6", "#eb6834", "#898781"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({v / cm.sum():.1%})", ha="center", va="center",
                color="white" if v > cm.max() * 0.6 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["pred: loses", "pred: wins"])
    ax.set_yticks([0, 1], ["actual: lost", "actual: won"])
    ax.set_title("Confusion matrix (test games)")
    save(fig, "confusion_matrix.png")


def plot_roc(y, p_model, p_base):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for p, c, lab in ((p_model, C_MODEL, "Value model"), (p_base, C_BASE, "Score-leader baseline")):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test games)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "roc_curve.png")


def plot_calibration(y, p):
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(mean_pred, frac, color=C_MODEL, lw=2, marker="o", ms=5, label="Value model")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Calibration (test games)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "calibration.png")


def plot_phase(phase_df):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(phase_df))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, phase_df["model"], w, color=C_MODEL, label="Value model")
    ax.bar(x + w / 2 + 0.01, phase_df["score_leader"], w, color=C_BASE, label="Score-leader baseline")
    for i, (m, b) in enumerate(zip(phase_df["model"], phase_df["score_leader"])):
        ax.text(i - w / 2, m + 0.01, f"{m:.0%}", ha="center", fontsize=8, color=INK2)
        ax.text(i + w / 2, b + 0.01, f"{b:.0%}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0.5, color=C_MUTED, lw=1, ls="--")
    ax.set_xticks(x, [p.split(" (")[0] for p in phase_df["phase"]])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Outcome-prediction accuracy by game phase")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "accuracy_by_phase.png")


def plot_importance(names, means, stds):
    order = np.argsort(means)[-15:]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order],
            color=C_MODEL, height=0.65, error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xlabel("Increase in log-loss when the feature is shuffled")
    ax.set_title("Top 15 features (permutation importance)")
    ax.grid(axis="y", visible=False)
    save(fig, "feature_importance.png")


def plot_arena(arena):
    labels = list(arena.keys())
    rates = [arena[k]["win_rate"] for k in labels]
    lo = [arena[k]["win_rate"] - arena[k]["ci95"][0] for k in labels]
    hi = [arena[k]["ci95"][1] - arena[k]["win_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rates, xerr=[lo, hi], color=C_MODEL, height=0.6, error_kw={"ecolor": INK2, "lw": 1})
    ax.axvline(0.5, color=C_MUTED, lw=1, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AI win rate (95% CI)")
    ax.set_title("Model-driven AI vs scripted bots")
    ax.grid(axis="y", visible=False)
    save(fig, "arena_win_rates.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-games", type=int, default=100, help="games per easy/normal matchup")
    ap.add_argument("--hard-games", type=int, default=60, help="games per hard matchup (slower)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    bundle = load_bundle(paths.MODEL_PATH)
    model = bundle["model"]
    print(f"Loaded {bundle['algorithm']} model {bundle['version']}")
    frames = load_split_frames(paths.POSITIONS, paths.SPLITS)
    test = frames["test"]
    X, y = xy(test)

    # ---- Part 1: prediction quality -------------------------------------------------
    p_model = predict_win_proba(model, X)
    p_lead = score_leader_scores(test)
    # Majority class: always predict the more common training label, with the
    # training base rate as its probability.
    base_rate = float(frames["train"]["label"].mean())
    p_major = np.full(len(y), base_rate)
    results = {
        "value_model": classification_metrics(y, p_model),
        "baseline_score_leader": classification_metrics(y, p_lead),
        "baseline_majority_class": classification_metrics(y, p_major),
    }
    # The score-leader baseline outputs a squashed point difference, not a calibrated probability.
    for k in ("log_loss", "brier"):
        results["baseline_score_leader"].pop(k, None)
    phase = by_phase(test, y, {"model": p_model, "score_leader": p_lead})
    print(json.dumps({k: {m: v for m, v in r.items() if m != "confusion_matrix"} for k, r in results.items()}, indent=1))
    print(phase.to_string(index=False))

    plot_confusion(results["value_model"]["confusion_matrix"])
    plot_roc(y, p_model, p_lead)
    plot_calibration(y, p_model)
    plot_phase(phase)

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(y), size=min(6000, len(y)), replace=False)
    t0 = time.time()
    imp = permutation_importance(model, X[sample], y[sample], scoring="neg_log_loss",
                                 n_repeats=3, random_state=args.seed, n_jobs=1)
    print(f"permutation importance: {time.time() - t0:.0f}s")
    plot_importance(FEATURE_NAMES, imp.importances_mean, imp.importances_std)
    top = sorted(zip(FEATURE_NAMES, imp.importances_mean), key=lambda t: -t[1])[:10]

    # ---- Part 2: playing strength ------------------------------------------------------
    arena = {}
    bots = ["random", "greedy", "cautious", "positional"]
    for diff in ("easy", "normal", "hard"):
        n = args.hard_games if diff == "hard" else args.arena_games
        for bot in bots:
            t0 = time.time()
            res = run_arena(
                lambda g, d=diff: ModelAgent(bundle, d, random.Random(args.seed * 1000 + g)),
                lambda g, b=bot: make_agent(b, epsilon=0.0, rng=random.Random(args.seed * 2000 + g)),
                n,
            )
            arena[f"{diff} vs {bot}"] = res
            print(f"  {diff:6s} vs {bot:10s} win {res['win_rate']:.0%} {res['ci95']} "
                  f"margin {res['avg_margin']:+.1f}  {res['avg_ms_per_move']:.1f} ms/move ({time.time() - t0:.0f}s)")
    res = run_arena(lambda g: ModelAgent(bundle, "hard", random.Random(args.seed * 3000 + g)),
                    lambda g: ModelAgent(bundle, "normal", random.Random(args.seed * 4000 + g)),
                    args.hard_games)
    arena["hard vs normal"] = res
    print(f"  hard vs normal win {res['win_rate']:.0%} {res['ci95']}")
    plot_arena(arena)

    metrics = {
        "model_version": bundle["version"], "algorithm": bundle["algorithm"], "params": bundle["params"],
        "test_rows": int(len(y)), "test_games": int(test["game_id"].nunique()),
        "classification": results, "accuracy_by_phase": phase.to_dict("records"),
        "top_features": [{"feature": f, "importance": round(float(v), 5)} for f, v in top],
        "arena": arena,
    }
    paths.REPORTS_DIR.mkdir(exist_ok=True)
    (paths.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    card = json.loads(paths.MODEL_CARD.read_text()) if paths.MODEL_CARD.exists() else {}
    card["test_metrics"] = {k: v for k, v in results["value_model"].items() if k != "confusion_matrix"}
    card["arena"] = {k: {"win_rate": v["win_rate"], "ci95": v["ci95"]} for k, v in arena.items()}
    paths.MODEL_CARD.write_text(json.dumps(card, indent=2))
    write_report(metrics)
    print(f"Wrote {paths.REPORTS_DIR / 'evaluation_report.md'}")


def write_report(m):
    c = m["classification"]
    vm, sl, mj = c["value_model"], c["baseline_score_leader"], c["baseline_majority_class"]

    def row(name, r):
        def g(k):
            return f"{r[k]:.4f}" if k in r else "n/a"
        return (f"| {name} | {g('accuracy')} | {g('precision')} | {g('recall')} | {g('f1')} | "
                f"{g('roc_auc')} | {g('log_loss')} | {g('brier')} |")

    cm = vm["confusion_matrix"]
    biggest = max(m["accuracy_by_phase"], key=lambda r: r["model"] - r["score_leader"])
    L = [
        "# Evaluation Report", "",
        f"Model: `{m['algorithm']}` `{json.dumps(m['params'])}`, version `{m['model_version']}`.",
        "Generated by `scripts/evaluate_model.py`. All numbers come from real runs.", "",
        "## 1. Outcome prediction on held-out games", "",
        f"The test set has {m['test_rows']:,} positions from **{m['test_games']:,} games** that were never used",
        "for training or model selection. The split is by game, so no position from a test game was seen.", "",
        "| predictor | accuracy | precision | recall | F1 | ROC-AUC | log-loss | Brier |",
        "|---|---|---|---|---|---|---|---|",
        row("**Value model**", vm),
        row("Baseline: score leader wins", sl),
        row("Baseline: majority class", mj), "",
        "**How to read this**", "",
        "- *Accuracy* is the share of positions where the model correctly predicts the eventual winner",
        "  (threshold 0.5). Many positions are genuinely undecided, especially in the opening, so",
        "  100% is impossible. The useful question is how far the model beats the baselines.",
        "- *Log-loss* and *Brier* measure the quality of the **probabilities**. Lower is better,",
        "  and 0.693 is the log-loss of a coin flip. The AI ranks moves by these probabilities,",
        "  so this is the metric that matters most for play.",
        "- *ROC-AUC* is the chance that a randomly picked winning position gets a higher score than",
        "  a randomly picked losing one.", "",
        f"Confusion matrix (rows = actual lost/won, cols = predicted lose/win): `{cm}`", "",
        "![confusion](figures/confusion_matrix.png) ![roc](figures/roc_curve.png)", "",
        "### Accuracy by game phase", "",
        "| phase | positions | value model | score-leader baseline |", "|---|---|---|---|",
    ]
    for r in m["accuracy_by_phase"]:
        L.append(f"| {r['phase']} | {r['rows']:,} | {r['model']:.3f} | {r['score_leader']:.3f} |")
    L += [
        "", "The outcome becomes more predictable as the board fills up. The model's advantage over",
        f"simply counting points is largest in the **{biggest['phase']}** "
        f"({biggest['model']:.1%} vs {biggest['score_leader']:.1%}), where the current score is a poor",
        "guide because nodes keep flipping.", "",
        "![phase](figures/accuracy_by_phase.png) ![calibration](figures/calibration.png)", "",
        "### What the model relies on (permutation importance)", "",
        "| feature | mean increase in log-loss when shuffled |", "|---|---|",
    ]
    for t in m["top_features"]:
        L.append(f"| `{t['feature']}` | {t['importance']:.4f} |")
    L += ["", "![importance](figures/feature_importance.png)", "",
          "## 2. Playing strength (reward-based evaluation)", "",
          "The model-driven AI played full games against each scripted bot (with ε = 0, the bots'",
          "strongest settings). Colours and the first move alternate every game.",
          "Win rate is shown with a 95% Wilson confidence interval.", "",
          "| matchup | games | W-L | win rate | 95% CI | avg margin (pts) | AI ms/move |",
          "|---|---|---|---|---|---|---|"]
    for k, r in m["arena"].items():
        L.append(f"| {k} | {r['games']} | {r['wins']}-{r['losses']} | {r['win_rate']:.1%} | "
                 f"{r['ci95'][0]:.1%}–{r['ci95'][1]:.1%} | {r['avg_margin']:+.1f} | {r['avg_ms_per_move']} |")
    L += ["", "![arena](figures/arena_win_rates.png)", "",
          "Notes: the bots are the same kinds of scripted players that generated the training data.",
          "Beating them shows that the learned evaluation turns into good moves. It does **not** prove",
          "optimal play, and strong humans may find weaknesses the bots never exposed."]
    (paths.REPORTS_DIR / "evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
