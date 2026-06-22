"""Training loop for SSL-VTC (Algorithm 1) + evaluation utilities."""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import PipelineConfig
from .dataset import TrajectoryDataset, stratified_labeled_mask
from .device import get_device
from .loss import labeled_loss, unlabeled_loss, classifier_loss, consistency_loss
from .metrics import classification_metrics
from .models import SSLVTC

# Static column indices in the normalized [T,7] matrix (before encoding)
_STATIC_COLS_RAW = [4, 5, 6]  # WID=4, LEN=5, DRA=6


def _static_dropout(x: torch.Tensor, prob: float, mode: str) -> torch.Tensor:
    """Randomly zero static channels in raw/raw_dt tensors during training.

    For seven-hot tensors, zeroing individual bit-positions is non-trivial and
    the effect is handled by the dataset's withheld mechanism instead.
    x: [B, 1, T, W]
    """
    if prob <= 0.0 or mode not in ("raw", "raw_dt"):
        return x
    # Identify the static column indices in the encoded [W] dimension.
    # raw mode: cols 4,5,6 (WID,LEN,DRA) assuming full active_attrs.
    # raw_dt mode: same (Δt is last, not static).
    mask = torch.rand(x.size(0), device=x.device) < prob   # [B] boolean
    if mask.any():
        x = x.clone()
        for col in _STATIC_COLS_RAW:
            if col < x.size(-1):
                x[mask, :, :, col] = 0.0
    return x


def _mmsi_to_bucket(mmsi_list: list[int], n_buckets: int) -> torch.Tensor:
    """Hash MMSIs into [0, n_buckets) buckets."""
    return torch.tensor([m % n_buckets for m in mmsi_list], dtype=torch.long)


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class _EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.shadow.load_state_dict(sd)


class _TrainStep(nn.Module):
    """Bundles the full M2 loss into one forward call.

    Note: nn.DataParallel is incompatible with the M2 sub-module routing.
    Single-GPU only; use 3 GPUs for parallel experiments.
    """

    def __init__(self, sslvtc: SSLVTC, n_classes: int, supervised_only: bool = False):
        super().__init__()
        self.sslvtc = sslvtc
        self.n_classes = n_classes
        self.supervised_only = supervised_only

    def forward(
        self, x_lab: torch.Tensor, y_lab: torch.Tensor, x_unlab: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_lab = x_lab.size(0)
        l1 = labeled_loss(self.sslvtc, x_lab, y_lab, self.n_classes) / n_lab
        l_clf = classifier_loss(self.sslvtc, x_lab, y_lab)
        if self.supervised_only:
            l2 = torch.zeros(1, device=x_lab.device)
        else:
            l2 = unlabeled_loss(self.sslvtc, x_unlab, self.n_classes) / x_unlab.size(0)
        return l1.unsqueeze(0), l2.unsqueeze(0), l_clf.unsqueeze(0)


@torch.no_grad()
def evaluate(
    model: SSLVTC,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return full classification_metrics dict on a loader."""
    model.eval()
    n_classes = model.cfg.n_classes
    all_true, all_pred = [], []
    for batch in loader:
        x, y = batch[0], batch[1]  # ignore mmsi if present
        x = x.to(device)
        logits = model.classifier(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_true.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
        all_pred.append(pred)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype="int64")
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype="int64")
    return classification_metrics(y_true, y_pred, n_classes)


def _resolve_mode(cfg: PipelineConfig, mode: str) -> str:
    """If mode is 'auto', pick based on backbone config."""
    if mode != "auto":
        return mode
    return "raw_dt" if cfg.model.backbone == "temporal_transformer" else "sevenhot"


def _datasets(cfg: PipelineConfig, mode: str, fill: str | None, static_avail: float = 1.0,
               split_column: str = "split", return_mmsi: bool = False):
    mode = _resolve_mode(cfg, mode)
    use_gr = (cfg.model.n_mmsi_buckets > 0 and cfg.model.gr_weight > 0.0) or return_mmsi
    mk = lambda split: TrajectoryDataset(
        cfg.paths.processed, split, cfg.encoding, mode=mode, missing_static_fill=fill,
        static_available_fraction=static_avail, withhold_seed=cfg.train.seed,
        split_column=split_column, return_mmsi=use_gr,
    )
    return mk("train"), mk("val"), mk("test"), mode


def train(
    cfg: PipelineConfig,
    *,
    supervised_only: bool = False,
    mode: str = "sevenhot",
    missing_static_fill: str | None = None,
    static_available_fraction: float = 1.0,
    progress: bool = True,
    tag: str | None = None,
    resume: bool = False,
    split_column: str = "split",
) -> dict:
    """Train SSL-VTC. If supervised_only, drops the unlabeled term (baseline).

    Checkpoints written each epoch to paths.results/checkpoints/{tag}_last.pt.
    Best val macro-F1 checkpoint at {tag}_best.pt.
    History persisted to paths.results/{tag}_history.json each epoch.
    Pass resume=True to continue from last checkpoint.
    """
    device = get_device(cfg.train.device)
    _seed_all(cfg.train.seed)
    pin = device.type == "cuda"

    if torch.cuda.device_count() > 1:
        tqdm.write(
            f"note: {torch.cuda.device_count()} GPUs detected; "
            "single-GPU only (DataParallel incompatible with M2 sub-module routing)"
        )

    use_gr = cfg.model.n_mmsi_buckets > 0 and cfg.model.gr_weight > 0.0
    full_train, val_ds, test_ds, _mode = _datasets(
        cfg, mode, missing_static_fill, static_available_fraction,
        split_column=split_column, return_mmsi=use_gr,
    )
    mode = _mode  # resolved mode (auto → sevenhot / raw_dt)
    t, d = full_train.shape()
    labels = full_train.labels
    lab_mask = stratified_labeled_mask(labels, cfg.train.labeled_fraction, cfg.train.seed)
    lab_idx = np.nonzero(lab_mask)[0]
    unlab_idx = np.nonzero(~lab_mask)[0]

    labeled_ds = Subset(full_train, lab_idx.tolist())
    unlabeled_ds = Subset(full_train, unlab_idx.tolist())

    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    lab_loader = DataLoader(labeled_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=pin)
    unlab_loader = (
        DataLoader(unlabeled_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=pin)
        if len(unlabeled_ds) >= bs else None
    )
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw, pin_memory=pin)

    n1, n2 = max(len(lab_idx), 1), max(len(unlab_idx), 1)
    alpha = cfg.train.beta * (n2 / n1)

    sslvtc = SSLVTC(cfg.model, t, d).to(device)
    train_step = _TrainStep(sslvtc, cfg.model.n_classes, supervised_only)
    opt = torch.optim.Adam(sslvtc.parameters(), lr=cfg.train.lr)

    ema: _EMA | None = None
    if cfg.train.ema_decay is not None:
        ema = _EMA(sslvtc, cfg.train.ema_decay)

    scheduler = None
    if cfg.train.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    steps_per_epoch = max(len(lab_loader), 1)
    unlab_iter = _cycle(unlab_loader) if unlab_loader is not None else None

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_state: dict | None = None
    history: dict = {"val_acc": [], "val_f1": [], "test_acc": [], "test_f1": []}
    start_epoch = 0
    no_improve = 0

    ckpt_dir: Path | None = None
    history_path: Path | None = None
    _tag = tag or ("baseline" if supervised_only else "sslvtc")
    if cfg.paths.results:
        ckpt_dir = Path(cfg.paths.results) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history_path = Path(cfg.paths.results) / f"{_tag}_history.json"

    # Resume from last checkpoint
    if resume and ckpt_dir is not None:
        last_ckpt = ckpt_dir / f"{_tag}_last.pt"
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=device)
            sslvtc.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["opt_state"])
            if ema is not None and "ema_state" in ckpt:
                ema.load_state_dict(ckpt["ema_state"])
            if scheduler is not None and "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_f1 = ckpt.get("best_val_f1", -1.0)
            best_val_acc = ckpt.get("best_val_acc", -1.0)
            history = ckpt.get("history", history)
            no_improve = ckpt.get("no_improve", 0)
            # Restore RNG state for determinism
            rng_state = ckpt.get("rng_state")
            if rng_state is not None:
                torch.set_rng_state(rng_state["torch"])
                np.random.set_state(rng_state["numpy"])
                random.setstate(rng_state["random"])
            tqdm.write(f"resumed from epoch {start_epoch} (best_val_f1={best_val_f1:.4f})")

    for epoch in range(start_epoch, cfg.train.epochs):
        sslvtc.train()
        lab_iter = iter(lab_loader)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{cfg.train.epochs}", disable=not progress, leave=False)
        for _ in bar:
            lab_batch = next(lab_iter)
            x_lab, y_lab = lab_batch[0].to(device), lab_batch[1].to(device)
            mmsi_lab = lab_batch[2] if use_gr and len(lab_batch) > 2 else None

            # Phase 2.3: training-time static channel dropout
            x_lab = _static_dropout(x_lab, cfg.model.static_dropout_prob, mode)

            if supervised_only or unlab_iter is None:
                x_unlab = x_lab
                eff_alpha = 1.0
            else:
                unlab_batch = next(unlab_iter)
                x_unlab = unlab_batch[0].to(device)
                x_unlab = _static_dropout(x_unlab, cfg.model.static_dropout_prob, mode)
                eff_alpha = alpha

            l1_v, l2_v, l_clf_v = train_step(x_lab, y_lab, x_unlab)
            loss = l1_v.mean() + l2_v.mean() + eff_alpha * l_clf_v.mean()

            # Phase 2.2: FixMatch-style consistency on unlabeled
            if cfg.train.consistency_weight > 0.0 and not supervised_only and x_unlab is not x_lab:
                l_con = consistency_loss(
                    sslvtc, x_unlab,
                    threshold=cfg.train.consistency_threshold,
                    mode=mode,
                )
                loss = loss + cfg.train.consistency_weight * l_con

            # Phase 2.3: gradient reversal MMSI adversarial loss
            if use_gr and sslvtc.mmsi_head is not None and mmsi_lab is not None:
                buckets = _mmsi_to_bucket(mmsi_lab.tolist(), cfg.model.n_mmsi_buckets).to(device)
                feat = sslvtc.classifier.forward_features(x_lab)
                gr_logits = sslvtc.mmsi_head(feat)
                l_gr = torch.nn.functional.cross_entropy(gr_logits, buckets)
                loss = loss + cfg.model.gr_weight * l_gr

            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(sslvtc)
            bar.set_postfix(total=f"{loss.item():.1f}", l_clf=f"{l_clf_v.mean().item():.3f}")

        if scheduler is not None:
            scheduler.step()

        # Evaluate with EMA weights if available
        eval_model = ema.shadow if ema is not None else sslvtc
        val_metrics = evaluate(eval_model, val_loader, device)
        test_metrics = evaluate(eval_model, test_loader, device)

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        test_acc = test_metrics["accuracy"]
        test_f1 = test_metrics["macro_f1"]

        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        if progress:
            tqdm.write(
                f"epoch {epoch+1}: val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
            )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            no_improve = 0
            eval_sd = eval_model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in eval_sd.items()}
            if ckpt_dir is not None:
                torch.save(best_state, ckpt_dir / f"{_tag}_best.pt")
        else:
            no_improve += 1

        # Persist history
        if history_path is not None:
            history_path.write_text(json.dumps(history, indent=2))

        # Save last checkpoint
        if ckpt_dir is not None:
            rng_state = {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            }
            ckpt_payload = {
                "epoch": epoch,
                "model_state": sslvtc.state_dict(),
                "opt_state": opt.state_dict(),
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "history": history,
                "no_improve": no_improve,
                "rng_state": rng_state,
            }
            if ema is not None:
                ckpt_payload["ema_state"] = ema.state_dict()
            if scheduler is not None:
                ckpt_payload["scheduler_state"] = scheduler.state_dict()
            torch.save(ckpt_payload, ckpt_dir / f"{_tag}_last.pt")

        # Early stopping on val macro-F1 — only after a warmup floor, so slow-warming
        # low-label SSL runs aren't killed before they start climbing.
        if (cfg.train.patience > 0
                and (epoch + 1) >= cfg.train.min_epochs_before_stop
                and no_improve >= cfg.train.patience):
            if progress:
                tqdm.write(f"early stop at epoch {epoch+1} (no val_f1 improvement for {cfg.train.patience} epochs)")
            break

    # Load best weights
    if best_state is not None:
        sslvtc.load_state_dict(best_state)
    eval_model = ema.shadow if ema is not None else sslvtc
    if ema is not None and best_state is not None:
        ema.shadow.load_state_dict(best_state)

    final_test = evaluate(eval_model, test_loader, device)

    return {
        "model": eval_model,  # EMA shadow (or sslvtc if no EMA)
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "test_acc": final_test["accuracy"],
        "test_f1": final_test["macro_f1"],
        "test_balanced_acc": final_test["balanced_accuracy"],
        "test_per_class_recall": final_test["per_class_recall"],
        "confusion_matrix": final_test["confusion_matrix"],
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
    split_column: str = "split",
    return_metrics: bool = False,
):
    """Train CNN Classifier alone (pure cross-entropy, 100% labels).

    Returns best-val test accuracy (float), or the full test-metrics dict at the
    best-val-F1 epoch if return_metrics=True. Used for Table 2 CNN baselines and
    Table 3 static ablation.
    """
    from .models import Classifier

    device = get_device(cfg.train.device)
    _seed_all(cfg.train.seed)
    pin = device.type == "cuda"
    train_ds, val_ds, test_ds, _mode = _datasets(cfg, mode, missing_static_fill, split_column=split_column)
    t, d = train_ds.shape()

    bs, nw = cfg.train.batch_size, cfg.train.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw, pin_memory=pin)

    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = None
    if cfg.train.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    best_val_f1, best_test_acc = -1.0, 0.0
    best_test_metrics: dict | None = None
    n_classes = cfg.model.n_classes

    for epoch in range(cfg.train.epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(clf(x), y).backward()
            opt.step()
        if scheduler is not None:
            scheduler.step()

        clf.eval()
        with torch.no_grad():
            def _eval(loader):
                yt, yp = [], []
                for batch in loader:
                    x, y = batch[0], batch[1]
                    pred = clf(x.to(device)).argmax(1).cpu().numpy()
                    yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
                    yp.append(pred)
                yt = np.concatenate(yt)
                yp = np.concatenate(yp)
                return classification_metrics(yt, yp, n_classes)

            vm = _eval(val_loader)
            tm = _eval(test_loader)

        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            best_test_acc = tm["accuracy"]
            best_test_metrics = tm
        if progress:
            tqdm.write(f"[clf {mode}] epoch {epoch+1}: val_acc={vm['accuracy']:.3f} val_f1={vm['macro_f1']:.3f}")

    if return_metrics:
        return best_test_metrics if best_test_metrics is not None else {
            "accuracy": best_test_acc, "macro_f1": 0.0, "balanced_accuracy": 0.0,
            "per_class_recall": [], "per_class_precision": [],
        }
    return best_test_acc


def save_result(result: dict, out_dir: str | Path, tag: str, cfg: PipelineConfig | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), out / f"{tag}_model.pt")
    np.save(out / f"{tag}_confusion.npy", result["confusion_matrix"])
    if cfg is not None:
        from .provenance import save_provenance
        save_provenance(cfg, out, tag)
