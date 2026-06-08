"""Classical supervised baselines (Table 2): SVM / DT / KNN / MLP on flattened
trajectory features. `mode` selects raw real values or seven-hot input.
CNN baselines live in train.train_supervised_classifier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import EncodingConfig
from .dataset import TrajectoryDataset


def _flatten_split(processed_dir: str, split: str, encoding: EncodingConfig, mode: str):
    ds = TrajectoryDataset(processed_dir, split, encoding, mode=mode)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x.numpy().reshape(-1))
        ys.append(y)
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="int64")


def run_classical_baselines(processed_dir: str, encoding: EncodingConfig | None = None,
                            mode: str = "raw") -> dict[str, float]:
    """Train SVM/DT/KNN/MLP on flattened features; report test accuracy."""
    encoding = encoding or EncodingConfig()
    x_train, y_train = _flatten_split(processed_dir, "train", encoding, mode)
    x_test, y_test = _flatten_split(processed_dir, "test", encoding, mode)
    models = {
        "SVM": SVC(kernel="rbf"),
        "DecisionTree": DecisionTreeClassifier(random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=0),
    }
    out: dict[str, float] = {}
    for name, clf in models.items():
        clf.fit(x_train, y_train)
        out[name] = float(accuracy_score(y_test, clf.predict(x_test)))
    return out


def run_mlp(processed_dir: str, encoding: EncodingConfig, mode: str) -> float:
    """Single MLP accuracy for a given encoding mode (raw or seven-hot)."""
    x_train, y_train = _flatten_split(processed_dir, "train", encoding, mode)
    x_test, y_test = _flatten_split(processed_dir, "test", encoding, mode)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=0)
    clf.fit(x_train, y_train)
    return float(accuracy_score(y_test, clf.predict(x_test)))
