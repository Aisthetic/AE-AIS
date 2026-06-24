"""Does the M2 VAE also collapse on dropped vessels in Danish AIS? (external validity)

Mirrors vae_collapse.py exactly; only configs and paths differ.
Trains full M2 VAE (autoencoder+classifier, 100% labels) on Danish static-complete
cohort, evaluates classifier on Danish inclusive test split by static_complete.

3 seeds, fine static bins 50/100/50.
Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/vae_collapse_danish.py
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
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics
from src.sslvtc.train import train
from src.sslvtc import CLASS_TO_IDX

CONFIG_A = "configs/danishais2019.yaml"
INCL     = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019_inclusive/processed"
OUT      = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results/vae_collapse_danish.csv"
SEEDS    = [42, 43, 44]
FINE     = {"WID": 50, "LEN": 100, "DRA": 50}
NC       = 4
IDX2CLS  = {v: k for k, v in CLASS_TO_IDX.items()}


def enc_full(cfg):
    bins = dict(cfg.encoding.bins); bins.update(FINE)
    return dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)


@torch.no_grad()
def _collect(clf, loader, device):
    clf.eval(); yt, yp = [], []
    for b in loader:
        x, y = b[0], b[1]
        yp.append(clf(x.to(device)).argmax(1).cpu().numpy())
        yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    return np.concatenate(yt), np.concatenate(yp)


def metrics_row(pop, seed, yt, yp):
    m = classification_metrics(yt, yp, NC); r = m["per_class_recall"]
    d = {"model": "A_vae", "population": pop, "seed": seed, "n": len(yt),
         "accuracy": round(m["accuracy"]*100, 2), "macro_f1": round(m["macro_f1"]*100, 2)}
    for i in range(NC):
        d[f"recall_{IDX2CLS[i]}"] = round(r[i]*100, 2) if i < len(r) else None
    return d


Path(OUT).parent.mkdir(parents=True, exist_ok=True)
# resume: skip seeds already written
rows = pd.read_csv(OUT).to_dict("records") if Path(OUT).exists() else []
done_seeds = {r["seed"] for r in rows if r["population"] == "all"}

if __name__ == "__main__":
    for seed in SEEDS:
        if seed in done_seeds:
            print(f"[seed {seed}] already done, skipping", flush=True); continue

        cfg = load_config(CONFIG_A)
        enc = enc_full(cfg)
        cfg = dataclasses.replace(
            cfg, encoding=enc,
            train=dataclasses.replace(cfg.train, labeled_fraction=1.0, seed=seed),
        )
        print(f"\n[seed {seed}] train FULL M2 VAE (Danish, 100% labels)", flush=True)
        t0 = time.time()
        res = train(cfg, supervised_only=False, mode="sevenhot", progress=False,
                    split_column="split", tag=f"vae_collapse_danish_s{seed}")
        model = res["model"]
        device = get_device(cfg.train.device)

        ds = TrajectoryDataset(INCL, "test", enc, mode="sevenhot",
                               missing_static_fill="mean", split_column="split")
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                        pin_memory=(device.type == "cuda"))
        yt, yp = _collect(model.classifier, dl, device)
        sc = ds.index["static_complete"].to_numpy().astype(bool)

        rows.append(metrics_row("complete", seed, yt[sc],  yp[sc]))
        rows.append(metrics_row("dropped",  seed, yt[~sc], yp[~sc]))
        rows.append(metrics_row("all",      seed, yt,      yp))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  done {round((time.time()-t0)/60,1)}min  "
              f"complete={rows[-3]['macro_f1']}  dropped={rows[-2]['macro_f1']}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["accuracy","macro_f1","recall_fishing","recall_passenger","recall_cargo","recall_tanker"]
    print("\n==== Danish M2 VAE on inclusive populations (mean over seeds) ====")
    print(df.groupby("population")[cols].mean().round(1).to_string())
    print(f"\n-> {OUT}")
