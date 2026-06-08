"""Training loop for SSL-VTC (Algorithm 1) + evaluation utilities."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import PipelineConfig
from .dataset import TrajectoryDataset, stratified_labeled_mask
from .device import get_device
from .loss import total_loss
from .models import SSLVTC


def _cycle(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: SSLVTC, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    """Return (accuracy, confusion_matrix[C,C]) on a loader."""
    model.eval()
    n_classes = model.cfg.n_classes
    cm = np.zeros((n_classes, n_classes), dtype="int64")
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        logits = model.classifier(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y = y.numpy()
        for t, p in zip(y, pred):
            cm[t, p] += 1
        correct += int((pred == y).sum())
        total += len(y)
    return (correct / max(total, 1)), cm


def _datasets(cfg: PipelineConfig, mode: str, fill: str | None, static_avail: float = 1.0):
    mk = lambda split: TrajectoryDataset(
        cfg.paths.processed, split, cfg.encoding, mode=mode, missing_static_fill=fill,
        static_available_fraction=static_avail, withhold_seed=cfg.train.seed,
    )
    return mk("train"), mk("val"), mk("test")


def train(
    cfg: PipelineConfig,
    *,
    supervised_only: bool = False,
    mode: str = "sevenhot",
    missing_static_fill: str | None = None,
    static_available_fraction: float = 1.0,
    progress: bool = True,
) -> dict:
    """Train SSL-VTC. If supervised_only, drops the unlabeled term (baseline)."""
    device = get_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)

    full_train, val_ds, test_ds = _datasets(cfg, mode, missing_static_fill, static_available_fraction)
    t, d = full_train.shape()
    labels = full_train.labels
    lab_mask = stratified_labeled_mask(labels, cfg.train.labeled_fraction, cfg.train.seed)
    lab_idx = np.nonzero(lab_mask)[0]
    unlab_idx = np.nonzero(~lab_mask)[0]

    labeled_ds = Subset(full_train, lab_idx.tolist())
    unlabeled_ds = Subset(full_train, unlab_idx.tolist())

    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    lab_loader = DataLoader(labeled_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw)
    unlab_loader = (
        DataLoader(unlabeled_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw)
        if len(unlabeled_ds) >= bs else None
    )
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw)

    n1, n2 = max(len(lab_idx), 1), max(len(unlab_idx), 1)
    alpha = cfg.train.beta * (n2 / n1)

    model = SSLVTC(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    steps_per_epoch = max(len(lab_loader), 1)
    unlab_iter = _cycle(unlab_loader) if unlab_loader is not None else None

    best_val = -1.0
    best_state = None
    history = {"val_acc": [], "test_acc": []}

    epoch_iter = range(cfg.train.epochs)
    for epoch in epoch_iter:
        model.train()
        lab_iter = iter(lab_loader)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{cfg.train.epochs}", disable=not progress, leave=False)
        for _ in bar:
            x_lab, y_lab = next(lab_iter)
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            if supervised_only or unlab_iter is None:
                x_unlab = None
                eff_alpha = 1.0  # pure cross-entropy weight when no SSL term
            else:
                x_unlab, _ = next(unlab_iter)
                x_unlab = x_unlab.to(device)
                eff_alpha = alpha
            loss, parts = total_loss(model, x_lab, y_lab, x_unlab, cfg.model.n_classes, eff_alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bar.set_postfix(total=f"{parts['total']:.1f}", l_clf=f"{parts['l_clf']:.3f}")

        val_acc, _ = evaluate(model, val_loader, device)
        test_acc, _ = evaluate(model, test_loader, device)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        if progress:
            tqdm.write(f"epoch {epoch+1}: val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test_acc, cm = evaluate(model, test_loader, device)

    return {
        "model": model,
        "best_val_acc": best_val,
        "test_acc": final_test_acc,
        "confusion_matrix": cm,
        "history": history,
        "alpha": alpha,
        "n_labeled": int(n1),
        "n_unlabeled": int(n2),
        "input_shape": (t, d),
    }


def train_supervised_classifier(
    cfg: PipelineConfig,
    *,
    mode: str = "sevenhot",
    missing_static_fill: str | None = None,
    progress: bool = False,
) -> float:
    """Train the CNN Classifier alone (pure cross-entropy, 100% labels) and return
    best-val test accuracy. Used for Table 2 CNN baselines (raw vs seven-hot)."""
    from .models import Classifier

    device = get_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    train_ds, val_ds, test_ds = _datasets(cfg, mode, missing_static_fill)
    t, d = train_ds.shape()

    bs, nw = cfg.train.batch_size, cfg.train.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw)

    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, best_test = -1.0, 0.0
    for epoch in range(cfg.train.epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(clf(x), y).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            def acc(loader):
                c = n = 0
                for x, y in loader:
                    pred = clf(x.to(device)).argmax(1).cpu()
                    c += int((pred == y).sum()); n += len(y)
                return c / max(n, 1)
            v, te = acc(val_loader), acc(test_loader)
        if v > best_val:
            best_val, best_test = v, te
        if progress:
            tqdm.write(f"[clf {mode}] epoch {epoch+1}: val={v:.3f} test={te:.3f}")
    return best_test


def save_result(result: dict, out_dir: str | Path, tag: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), out / f"{tag}_model.pt")
    np.save(out / f"{tag}_confusion.npy", result["confusion_matrix"])
