"""Matplotlib helpers: accuracy-vs-epoch curves (Fig 7) and confusion matrices."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import CLASSES


def plot_accuracy_curves(series: dict[str, list[float]], path: str | Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, ys in series.items():
        ax.plot(range(1, len(ys) + 1), ys, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: str | Path, title: str = "") -> None:
    cm = np.asarray(cm, dtype="float64")
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
