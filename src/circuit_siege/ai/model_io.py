"""Saving / loading the trained value model as a single self-describing bundle."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES


class ModelNotFoundError(FileNotFoundError):
    """The model file is missing: the training pipeline has not been run yet."""


class ModelIncompatibleError(RuntimeError):
    """The model file exists but was trained on a different feature set."""


def save_bundle(path: Path, model, algorithm: str, params: dict, metrics: dict | None = None,
                extra: dict | None = None) -> dict:
    bundle = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "algorithm": algorithm,
        "params": params,
        "metrics": metrics or {},
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "version": _dt.datetime.now().strftime("v%Y%m%d-%H%M%S"),
    }
    if extra:
        bundle.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


def load_bundle(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model at '{path}'. Run: python scripts/generate_dataset.py "
            f"then python scripts/train_model.py"
        )
    # joblib uses pickle: only load model files this project produced itself.
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupted / truncated file
        raise ModelIncompatibleError(f"Could not read model file '{path}': {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelIncompatibleError(f"'{path}' is not a Circuit Siege model bundle")
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ModelIncompatibleError(
            "Model was trained on a different feature set; retrain with scripts/train_model.py"
        )
    return bundle


def predict_win_proba(model, X) -> np.ndarray:
    """P(label == 1) for each row, robust to class ordering and bad outputs."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    col = classes.index(1) if 1 in classes else proba.shape[1] - 1
    p = np.asarray(proba[:, col], dtype=np.float64)
    # Guard against NaN / out-of-range outputs: treat them as "no information".
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, 0.0, 1.0)
