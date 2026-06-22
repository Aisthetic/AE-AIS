"""Classification metrics for imbalanced multi-class evaluation."""
from __future__ import annotations

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict:
    """Compute a full metrics dict from label arrays.

    Returns: accuracy, balanced_accuracy, macro_f1, per_class_recall,
             per_class_precision, confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.asarray(y_pred, dtype="int64")

    cm = np.zeros((n_classes, n_classes), dtype="int64")
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_recall = []
    per_class_precision = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_recall.append(recall)
        per_class_precision.append(precision)

    f1_per = []
    for r, p in zip(per_class_recall, per_class_precision):
        denom = r + p
        f1_per.append((2 * r * p / denom) if denom > 0 else 0.0)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    balanced_accuracy = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(f1_per))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "confusion_matrix": cm,
    }
