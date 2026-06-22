"""A-grid fix: evaluate the paper's model (A) on dropped vessels with MATCHED normalization.

Model A is trained on the static-complete cohort (fullus2019). Previously we evaluated it on
the inclusive cohort, which used DIFFERENT normalization stats -> distribution-shift confound
(A-complete fell 91.6->88.3). Here we evaluate A on `fullus2019_inclusive_cnorm`, which is the
same inclusive vessels but normalized with the COMPLETE cohort's stats (= A's training norm,
the correct deployment setup). 3 seeds, per-class.

Sanity check: A-on-complete here should recover to ~91.6 (matching the consequence run),
confirming the normalization fix.

Output: fullus2019/results/grid_A_cnorm.csv
Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/grid_A_cnorm.py
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

CONFIG_A = "configs/fullus2019.yaml"                       # A trains here (complete cohort)
CNORM = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019_inclusive_cnorm/processed"
OUT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results/grid_A_cnorm.csv"
SEEDS = [42, 43, 44]
EPOCHS = 50
FINE = {"WID": 50, "LEN": 100, "DRA": 50}
NC = 4
IDX2CLS = {v: k for k, v in CLASS_TO_IDX.items()}


def enc_full(cfg):
    bins = dict(cfg.encoding.bins); bins.update(FINE)
    return dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)


@torch.no_grad()
def collect(clf, loader, device):
    clf.eval(); yt, yp = [], []
    for b in loader:
        x, y = b[0], b[1]
        yp.append(clf(x.to(device)).argmax(1).cpu().numpy())
        yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    return np.concatenate(yt), np.concatenate(yp)


def train_A(enc, seed):
    cfg = load_config(CONFIG_A)
    device = get_device(cfg.train.device)
    torch.manual_seed(seed); np.random.seed(seed)
    pin = device.type == "cuda"; bs, nw = cfg.train.batch_size, cfg.train.num_workers
    mk = lambda sp: TrajectoryDataset(cfg.paths.processed, sp, enc, mode="sevenhot",
                                      missing_static_fill=None, split_column="split")
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
        yt, yp = collect(clf, vl, device)
        f1 = classification_metrics(yt, yp, NC)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best)
    return clf, device


def row(pop, seed, yt, yp):
    m = classification_metrics(yt, yp, NC); r = m["per_class_recall"]
    d = {"model": "A_paper_cnorm", "population": pop, "seed": seed, "n": len(yt),
         "accuracy": round(m["accuracy"]*100, 2), "macro_f1": round(m["macro_f1"]*100, 2)}
    for i in range(NC):
        d[f"recall_{IDX2CLS[i]}"] = round(r[i]*100, 2) if i < len(r) else None
    return d


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        cfg = load_config(CONFIG_A); enc = enc_full(cfg)
        print(f"\n[seed {seed}] train A (complete cohort)", flush=True); t0 = time.time()
        clf, dev = train_A(enc, seed)
        # eval on cnorm inclusive test (matched normalization), mean-fill for missing static
        ds = TrajectoryDataset(CNORM, "test", enc, mode="sevenhot",
                               missing_static_fill="mean", split_column="split")
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                        pin_memory=(dev.type == "cuda"))
        yt, yp = collect(clf, dl, dev)
        sc = ds.index["static_complete"].to_numpy().astype(bool)
        rows.append(row("complete", seed, yt[sc], yp[sc]))
        rows.append(row("dropped", seed, yt[~sc], yp[~sc]))
        rows.append(row("all", seed, yt, yp))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  done {round((time.time()-t0)/60,1)}min  A-complete acc={rows[-3]['accuracy']} (sanity ~91.6)", flush=True)

    df = pd.DataFrame(rows)
    cols = ["accuracy","macro_f1","recall_fishing","recall_passenger","recall_cargo","recall_tanker"]
    print("\n==== A on cnorm populations (mean over seeds) ====")
    print(df.groupby("population")[cols].mean().round(1).to_string())
    print(f"\n-> {OUT}")
