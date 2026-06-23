"""Consequence experiment on Danish AIS — external validity replication.

Regime A: static-complete Danish cohort  (danishais2019)
Regime B: static-inclusive Danish cohort (danishais2019_inclusive, missing static mean-filled)

Mirrors consequence_experiment.py exactly; only CONFIG_A/B and OUT differ.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/consequence_danish.py
"""
from __future__ import annotations
import dataclasses, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.sslvtc.config import load_config
from src.sslvtc.device import get_device
from src.sslvtc.models import Classifier
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics
from src.sslvtc import CLASS_TO_IDX

CONFIG_A = "configs/danishais2019.yaml"
CONFIG_B = "configs/danishais2019_inclusive.yaml"
OUT      = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results/consequence_danish.csv"
SEEDS    = [42, 43, 44]
EPOCHS   = 50
FINE     = {"WID": 50, "LEN": 100, "DRA": 50}
FISH     = CLASS_TO_IDX["fishing"]
NC       = 4


def _enc(cfg):
    bins = dict(cfg.encoding.bins); bins.update(FINE)
    return dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)


def _loader(ds, bs, nw, pin, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pin)


@torch.no_grad()
def _collect(clf, loader, device):
    clf.eval(); yt, yp = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        yp.append(clf(x.to(device)).argmax(1).cpu().numpy())
        yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    return np.concatenate(yt), np.concatenate(yp)


def train_eval(config, missing_fill, seed):
    cfg = load_config(config)
    device = get_device(cfg.train.device)
    torch.manual_seed(seed); np.random.seed(seed)
    enc = _enc(cfg)
    pin = device.type == "cuda"; bs, nw = cfg.train.batch_size, cfg.train.num_workers

    mk = lambda sp: TrajectoryDataset(
        cfg.paths.processed, sp, enc, mode="sevenhot",
        missing_static_fill=missing_fill, split_column="split")
    train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    t, d = train_ds.shape()

    tl  = _loader(train_ds, bs, nw, pin, shuffle=True)
    vl  = _loader(val_ds,   bs, nw, pin)
    tel = _loader(test_ds,  bs, nw, pin)

    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        clf.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(clf(x), y).backward(); opt.step()
        sched.step()
        yt, yp = _collect(clf, vl, device)
        f1 = classification_metrics(yt, yp, NC)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)

    yt, yp = _collect(clf, tel, device)
    full = classification_metrics(yt, yp, NC)
    return {"clf": clf, "device": device, "cfg": cfg, "enc": enc,
            "test_index": test_ds.index, "yt": yt, "yp": yp, "full": full}


def _row(tag, seed, m, n=None):
    recalls = m.get("per_class_recall", [])
    return {
        "regime": tag, "seed": seed, "n_test": n,
        "accuracy":       round(m["accuracy"] * 100, 2),
        "macro_f1":       round(m["macro_f1"] * 100, 2),
        "balanced_acc":   round(m["balanced_accuracy"] * 100, 2),
        "fishing_recall": round(recalls[FISH] * 100, 2) if FISH < len(recalls) else None,
    }


if __name__ == "__main__":
    import os; os.makedirs(str(pd.Path(OUT).parent) if hasattr(pd, "Path") else OUT.rsplit("/", 1)[0], exist_ok=True)
    from pathlib import Path; Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} | Regime A (Danish static-complete) ===", flush=True)
        t0 = time.time()
        a = train_eval(CONFIG_A, missing_fill=None, seed=seed)
        ra = _row("A_complete", seed, a["full"], len(a["yt"]))
        ra["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(ra); pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  A: acc={ra['accuracy']}  fish_recall={ra['fishing_recall']}", flush=True)

        print(f"=== seed {seed} | Regime B (Danish static-inclusive) ===", flush=True)
        t0 = time.time()
        b = train_eval(CONFIG_B, missing_fill="mean", seed=seed)
        yt, yp = b["yt"], b["yp"]
        rb = _row("B_inclusive_all", seed, b["full"], len(yt))
        rb["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(rb)

        sc = b["test_index"]["static_complete"].to_numpy().astype(bool)
        for name, mask in [("B_test_complete", sc), ("B_test_incomplete", ~sc)]:
            if mask.sum() == 0:
                continue
            m = classification_metrics(yt[mask], yp[mask], NC)
            rows.append(_row(name, seed, m, int(mask.sum())))

        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  B all: acc={rb['accuracy']}  fish_recall={rb['fishing_recall']}", flush=True)

    df = pd.DataFrame(rows)
    print("\n==== DANISH CONSEQUENCE EXPERIMENT (per-seed) ====")
    print(df.to_string(index=False))
    print("\n==== mean over seeds ====")
    summ = df.groupby("regime")[["accuracy", "macro_f1", "fishing_recall"]].agg(["mean", "std"]).round(2)
    print(summ.to_string())
    print(f"\n-> {OUT}")
