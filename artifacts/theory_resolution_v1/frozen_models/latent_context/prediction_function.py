"""Generated prediction-only entry point for a frozen cognitive theory."""
import pickle
from pathlib import Path


def load_model(directory=None):
    root = Path(directory or Path(__file__).parent)
    with (root / "model.pkl").open("rb") as handle:
        return pickle.load(handle)


def predict(frame, directory=None):
    return load_model(directory).predict(frame)
