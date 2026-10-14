"""Model selection and training for the value model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..ai.features import FEATURE_NAMES
from ..ai.model_io import predict_win_proba
from .dataset import LABEL


def candidate_grid(seed: int = 42, fast: bool = False) -> list[tuple[str, dict, object]]:
    """(algorithm, params, unfitted estimator). Scaled where the algorithm needs it."""
    if fast:  # tiny grid for tests
        return [
            ("logistic_regression", {"C": 1.0},
             make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
            ("hist_gradient_boosting", {"learning_rate": 0.1, "max_iter": 50},
             HistGradientBoostingClassifier(learning_rate=0.1, max_iter=50, random_state=seed)),
        ]
    grid = []
    for C in (0.1, 1.0):
        grid.append(("logistic_regression", {"C": C},
                     make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))))
    for depth in (12, 20):
        p = {"n_estimators": 150, "max_depth": depth, "min_samples_leaf": 20}
        grid.append(("random_forest", p, RandomForestClassifier(**p, n_jobs=-1, random_state=seed)))
    for lr, leaves in ((0.1, 31), (0.05, 63)):
        p = {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 400, "l2_regularization": 1.0}
        grid.append(("hist_gradient_boosting", p,
                     HistGradientBoostingClassifier(**p, early_stopping=False, random_state=seed)))
    for hidden, alpha in (((128, 64), 1e-4), ((256, 128), 1e-3)):
        p = {"hidden_layer_sizes": hidden, "alpha": alpha}
        grid.append(("mlp", {"hidden_layer_sizes": list(hidden), "alpha": alpha},
                     make_pipeline(StandardScaler(), MLPClassifier(
                         hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True,
                         max_iter=200, n_iter_no_change=8, random_state=seed))))
    return grid


def load_split_frames(positions_path: Path, splits_path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(positions_path)
    splits = json.loads(Path(splits_path).read_text())
    out = {name: df[df["game_id"].isin(set(splits[name]))].reset_index(drop=True)
           for name in ("train", "val", "test")}
    # Hard guarantee against leakage: no game may appear in two splits.
    g = {k: set(v["game_id"]) for k, v in out.items()}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), "split leakage"
    return out


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURE_NAMES].to_numpy(np.float64), df[LABEL].to_numpy(np.int64)


def score(model, X, y) -> dict:
    p = predict_win_proba(model, X)
    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
    }


def select_model(train: pd.DataFrame, val: pd.DataFrame, seed: int = 42, fast: bool = False,
                 log=print) -> tuple[list[dict], tuple[str, dict, object]]:
    Xtr, ytr = xy(train)
    Xva, yva = xy(val)
    results, best, best_ll = [], None, float("inf")
    for algo, params, est in candidate_grid(seed, fast):
        t0 = time.time()
        est.fit(Xtr, ytr)
        fit_s = time.time() - t0
        t1 = time.perf_counter()
        predict_win_proba(est, Xva[:48])  # one AI turn's worth of candidate positions
        latency_ms = (time.perf_counter() - t1) * 1000
        m = score(est, Xva, yva)
        row = {"algorithm": algo, "params": json.dumps(params), **{f"val_{k}": round(v, 4) for k, v in m.items()},
               "fit_seconds": round(fit_s, 1), "predict_48_ms": round(latency_ms, 1)}
        results.append(row)
        log(f"  {algo:24s} {json.dumps(params):70s} val_logloss={m['log_loss']:.4f} "
            f"acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} ({fit_s:.0f}s)")
        if m["log_loss"] < best_ll:
            best_ll, best = m["log_loss"], (algo, params, est)
    return results, best


def refit(algo: str, params: dict, data: pd.DataFrame, seed: int = 42, sample_weight=None):
    """Fresh estimator with the chosen hyper-parameters, fitted on `data`."""
    for a, p, est in candidate_grid(seed) + candidate_grid(seed, fast=True):
        if a == algo and p == params:
            X, y = xy(data)
            if sample_weight is not None:
                last = est.steps[-1][0] if hasattr(est, "steps") else None
                kw = {f"{last}__sample_weight": sample_weight} if last else {"sample_weight": sample_weight}
                try:
                    est.fit(X, y, **kw)
                except TypeError:  # estimator without sample_weight support (MLP)
                    est.fit(X, y)
            else:
                est.fit(X, y)
            return est
    raise ValueError(f"no candidate matches {algo} {params}")
