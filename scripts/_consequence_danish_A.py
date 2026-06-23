"""Regime A only — static-complete Danish cohort. Run on GPU 1."""
import sys; sys.path.insert(0, ".")
import dataclasses, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.sslvtc.config import load_config
from src.sslvtc.device import get_device
from src.sslvtc.models import Classifier
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics
from src.sslvtc import CLASS_TO_IDX

CONFIG = "configs/danishais2019.yaml"
OUT    = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results/regime_A.csv"
SEEDS  = [42, 43, 44]; EPOCHS = 50; FINE = {"WID": 50, "LEN": 100, "DRA": 50}
FISH = CLASS_TO_IDX["fishing"]; NC = 4

def _enc(cfg):
    bins = dict(cfg.encoding.bins); bins.update(FINE)
    return dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)

def _loader(ds, bs, nw, pin, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pin)

@torch.no_grad()
def _collect(clf, loader, device):
    clf.eval(); yt, yp = [], []
    for x, y in loader:
        yp.append(clf(x.to(device)).argmax(1).cpu().numpy())
        yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    return np.concatenate(yt), np.concatenate(yp)

def train_eval(seed):
    cfg = load_config(CONFIG)
    device = get_device(cfg.train.device)
    torch.manual_seed(seed); np.random.seed(seed)
    enc = _enc(cfg); pin = device.type == "cuda"; bs, nw = cfg.train.batch_size, cfg.train.num_workers
    mk = lambda sp: TrajectoryDataset(cfg.paths.processed, sp, enc, mode="sevenhot",
                                      missing_static_fill=None, split_column="split")
    tr, va, te = mk("train"), mk("val"), mk("test")
    t, d = tr.shape()
    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        clf.train()
        for x, y in _loader(tr, bs, nw, pin, shuffle=True):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(clf(x), y).backward(); opt.step()
        sched.step()
        yt, yp = _collect(clf, _loader(va, bs, nw, pin), device)
        f1 = classification_metrics(yt, yp, NC)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    yt, yp = _collect(clf, _loader(te, bs, nw, pin), device)
    return classification_metrics(yt, yp, NC), yt, yp

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
rows = []
for seed in SEEDS:
    print(f"=== seed {seed} Regime A ===", flush=True)
    t0 = time.time()
    m, yt, yp = train_eval(seed)
    recalls = m.get("per_class_recall", [])
    rows.append({"regime": "A_complete", "seed": seed, "n_test": len(yt),
                 "accuracy": round(m["accuracy"]*100,2), "macro_f1": round(m["macro_f1"]*100,2),
                 "balanced_acc": round(m["balanced_accuracy"]*100,2),
                 "fishing_recall": round(recalls[FISH]*100,2) if FISH < len(recalls) else None,
                 "minutes": round((time.time()-t0)/60,1)})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  acc={rows[-1]['accuracy']} fish={rows[-1]['fishing_recall']}", flush=True)
print(pd.DataFrame(rows).to_string(index=False))
print(f"-> {OUT}")
