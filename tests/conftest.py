import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit_siege.ai.model_io import load_bundle, save_bundle  # noqa: E402
from circuit_siege.training.dataset import generate, make_splits  # noqa: E402
from circuit_siege.training.train import refit  # noqa: E402


@pytest.fixture(scope="session")
def tiny_data():
    """A small but real dataset: 40 simulated games."""
    records, df = generate(40, seed=123, workers=1)
    return records, df


@pytest.fixture(scope="session")
def tiny_model_path(tiny_data, tmp_path_factory):
    """A real (small) model trained by the project's own training code."""
    _, df = tiny_data
    splits = make_splits(df["game_id"].unique(), seed=1)
    train = df[df["game_id"].isin(splits["train"])]
    model = refit("logistic_regression", {"C": 1.0}, train)
    path = tmp_path_factory.mktemp("models") / "value_model.joblib"
    save_bundle(path, model, "logistic_regression", {"C": 1.0},
                extra={"train_rows": len(train), "train_games": len(splits["train"]), "data_sources": ["test"]})
    return path


@pytest.fixture(scope="session")
def tiny_bundle(tiny_model_path):
    return load_bundle(tiny_model_path)
