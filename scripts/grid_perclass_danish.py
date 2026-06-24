"""Grid experiment on Danish AIS — 3 models × complete/dropped/all populations.

Mirrors grid_perclass_experiment.py exactly; only configs, paths, and OUT differ.

Three models (fine static bins 50/100/50, temporal split):
  A_paper  = size+kinematic, trained on Danish static-COMPLETE cohort
  B_full   = size+kinematic, trained on Danish static-INCLUSIVE cohort (mean-filled)
  B_nosize = kinematic-ONLY, trained on Danish static-inclusive cohort

Evaluations on Danish inclusive test split (complete / dropped / all).

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/grid_perclass_danish.py
"""
from __future__ import annotations
import dataclasses, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.sslvtc.config import load_config
from src.sslvtc.device import get_device
from src.sslvtc.models import Classifier
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics
from src.sslvtc import CLASS_TO_IDX

CONFIG_A   = "configs/danishais2019.yaml"
CONFIG_B   = "configs/danishais2019_inclusive.yaml"
OUT        = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results/grid_perclass_danish.csv"
INCL       = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019_inclusive/processed"
SEEDS      = [42, 43, 44]
EPOCHS     = 50
FINE       = {"WID": 50, "LEN": 100, "DRA": 50}
NC         = 4
IDX2CLS    = {v: k for k, v in CLASS_TO_IDX.items()}


def enc_full(cfg):
    bins = dict(cfg.encoding.bins); bins.update(FINE)
    return dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)

def enc_nosize(cfg):
    return dataclasses.replace(cfg.encoding, use_len=False, use_wid=False, use_dra=False)


def train_clf(config, enc, missing_fill, seed):
    cfg = load_config(config)
    device = get_device(cfg.train.device)
    torch.manual_seed(seed); np.random.seed(seed)
    pin = device.type == "cuda"; bs, nw = cfg.train.batch_size, cfg.train.num_workers
    mk = lambda sp: TrajectoryDataset(cfg.paths.processed, sp, enc, mode="sevenhot",
                                      missing_static_fill=missing_fill, split_column="split")
    tr, va = mk("train"), mk("val")
    t, d = tr.shape()
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    vl = DataLoader(va, batch_size=bs, num_workers=nw, pin_memory=pin)
    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf = torch.nn.CrossEntropyLoss()
    best_f1, best = -1.0, None
    for ep in range(EPOCHS):
        clf.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lf(clf(x), y).backward(); opt.step()
        sched.step()
        yt, yp = _collect(clf, vl, device)
        f1 = classification_metrics(yt, yp, NC)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best)
    return clf, device


@torch.no_grad()
def _collect(clf, loader, device):
    clf.eval(); yt, yp = [], []
    for b in loader:
        x, y = b[0], b[1]
        yp.append(clf(x.to(device)).argmax(1).cpu().numpy())
        yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    return np.concatenate(yt), np.concatenate(yp)


def eval_splits(clf, device, enc, missing_fill):
    cfg = load_config(CONFIG_B)
    ds = TrajectoryDataset(INCL, "test", enc, mode="sevenhot",
                           missing_static_fill=missing_fill, split_column="split")
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                    pin_memory=(device.type == "cuda"))
    yt, yp = _collect(clf, dl, device)
    sc = ds.index["static_complete"].to_numpy().astype(bool)
    return yt, yp, sc


def metrics_row(model, pop, seed, yt, yp):
    m = classification_metrics(yt, yp, NC)
    r = m["per_class_recall"]
    row = {"model": model, "population": pop, "seed": seed, "n": len(yt),
           "accuracy": round(m["accuracy"]*100, 2), "macro_f1": round(m["macro_f1"]*100, 2)}
    for i in range(NC):
        row[f"recall_{IDX2CLS[i]}"] = round(r[i]*100, 2) if i < len(r) else None
    return row


# resume: skip model+seed combos already written
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
rows = pd.read_csv(OUT).to_dict("records") if Path(OUT).exists() else []
done = {(r["model"], r["seed"]) for r in rows if r["population"] == "all"}


def add(model, pop, seed, yt, yp):
    rows.append(metrics_row(model, pop, seed, yt, yp))
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    for seed in SEEDS:
        cfgA = load_config(CONFIG_A); cfgB = load_config(CONFIG_B)
        eF_A, eF_B, eN_B = enc_full(cfgA), enc_full(cfgB), enc_nosize(cfgB)

        if ("A_paper", seed) not in done:
            print(f"\n[seed {seed}] train A_paper (size, complete)", flush=True); t0 = time.time()
            clfA, dev = train_clf(CONFIG_A, eF_A, None, seed)
            yt, yp, sc = eval_splits(clfA, dev, eF_A, "mean")
            add("A_paper", "complete", seed, yt[sc],  yp[sc])
            add("A_paper", "dropped",  seed, yt[~sc], yp[~sc])
            add("A_paper", "all",      seed, yt,      yp)
            print(f"  A_paper done {round((time.time()-t0)/60,1)}min", flush=True)
        else:
            print(f"[seed {seed}] A_paper already done, skipping", flush=True)

        if ("B_full", seed) not in done:
            print(f"[seed {seed}] train B_full (size, inclusive)", flush=True); t0 = time.time()
            clfBf, dev = train_clf(CONFIG_B, eF_B, "mean", seed)
            yt, yp, sc = eval_splits(clfBf, dev, eF_B, "mean")
            add("B_full", "complete", seed, yt[sc],  yp[sc])
            add("B_full", "dropped",  seed, yt[~sc], yp[~sc])
            add("B_full", "all",      seed, yt,      yp)
            print(f"  B_full done {round((time.time()-t0)/60,1)}min", flush=True)
        else:
            print(f"[seed {seed}] B_full already done, skipping", flush=True)

        if ("B_nosize", seed) not in done:
            print(f"[seed {seed}] train B_nosize (kinematic only)", flush=True); t0 = time.time()
            clfBn, dev = train_clf(CONFIG_B, eN_B, None, seed)
            yt, yp, sc = eval_splits(clfBn, dev, eN_B, "zero")
            add("B_nosize", "complete", seed, yt[sc],  yp[sc])
            add("B_nosize", "dropped",  seed, yt[~sc], yp[~sc])
            add("B_nosize", "all",      seed, yt,      yp)
            print(f"  B_nosize done {round((time.time()-t0)/60,1)}min", flush=True)
        else:
            print(f"[seed {seed}] B_nosize already done, skipping", flush=True)

    df = pd.DataFrame(rows)
    print("\n==== DANISH GRID + PER-CLASS (per-seed) ====")
    print(df.to_string(index=False))
    print("\n==== mean over seeds ====")
    cols = ["accuracy","macro_f1","recall_fishing","recall_passenger","recall_cargo","recall_tanker"]
    print(df.groupby(["model","population"])[cols].mean().round(2).to_string())
    print(f"\n-> {OUT}")
