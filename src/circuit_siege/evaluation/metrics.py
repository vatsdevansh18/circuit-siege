"""Classification metrics for the value model and its baselines on held-out games."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PHASES = [(1, 12, "opening (ply 1-12)"), (13, 24, "early-mid (13-24)"),
          (25, 36, "late-mid (25-36)"), (37, 48, "endgame (37-48)")]


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(set(y)) > 1:
        out["roc_auc"] = roc_auc_score(y, p)
    is_prob = np.all((p >= 0) & (p <= 1))
    if is_prob:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out["log_loss"] = log_loss(y, pc, labels=[0, 1])
        out["brier"] = brier_score_loss(y, pc)
    return {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in out.items()}


def score_leader_scores(df: pd.DataFrame) -> np.ndarray:
    """Baseline: 'whoever is ahead on points wins'. Ties broken by who moves last.

    Returns a ranking score (for ROC-AUC) whose >0.5 threshold equals the rule.
    """
    diff = df["point_diff"].to_numpy(float)
    tiebreak = np.where(df["i_move_last"].to_numpy() > 0, 0.25, -0.25)
    raw = np.where(diff != 0, diff, tiebreak)
    return 1 / (1 + np.exp(-raw))  # monotone squashing: same ranking and same decisions


def by_phase(df: pd.DataFrame, y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    y = np.asarray(y)
    for lo, hi, name in PHASES:
        mask = ((df["ply"] >= lo) & (df["ply"] <= hi)).to_numpy()
        row = {"phase": name, "rows": int(mask.sum())}
        for label, p in preds.items():
            row[label] = round(float(((p[mask] >= 0.5).astype(int) == y[mask]).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)
