"""Does the paper's FULL autoencoder+classifier (M2 VAE) also collapse on dropped vessels?

Findings 1.4/1.5 used the supervised CNN classifier alone (= the paper's static tables). This
tests their actual proposed architecture: the M2 VAE (encoder+decoder+classifier) trained at
100% labels (generative L1 term active = the autoencoder regularizer). We then evaluate its
classifier on the matched-normalization inclusive cohort, split complete / dropped / all.

Compare to grid_A_cnorm.csv (classifier-only A): complete 91.2, dropped 23.4.
  - if VAE also collapses on dropped -> Finding 1 holds for the paper's real architecture
  - if VAE softens the collapse      -> the autoencoder regularizer helps; honest caveat

3 seeds, fine static bins 50/100/50. Output: fullus2019/results/vae_collapse.csv
Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/vae_collapse.py
"""
from __future__ import annotations
import dataclasses, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.sslvtc.config import load_config
from src.sslvtc.device import get_device
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics
from src.sslvtc.train import train
from src.sslvtc import CLASS_TO_IDX

CONFIG_A = "configs/fullus2019.yaml"
CNORM = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019_inclusive_cnorm/processed"
OUT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results/vae_collapse.csv"
SEEDS = [42, 43, 44]
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


def metrics_row(pop, seed, yt, yp):
    m = classification_metrics(yt, yp, NC); r = m["per_class_recall"]
    d = {"model": "A_vae", "population": pop, "seed": seed, "n": len(yt),
         "accuracy": round(m["accuracy"]*100, 2), "macro_f1": round(m["macro_f1"]*100, 2)}
    for i in range(NC):
        d[f"recall_{IDX2CLS[i]}"] = round(r[i]*100, 2) if i < len(r) else None
    return d


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        cfg = load_config(CONFIG_A)
        enc = enc_full(cfg)
        # full M2 VAE at 100% labels (generative L1 + classifier; encoder+decoder active)
        cfg = dataclasses.replace(
            cfg, encoding=enc,
            train=dataclasses.replace(cfg.train, labeled_fraction=1.0, seed=seed),
            paths=dataclasses.replace(cfg.paths, results=cfg.paths.results),
        )
        print(f"\n[seed {seed}] train FULL M2 VAE (autoencoder+classifier), 100% labels", flush=True)
        t0 = time.time()
        res = train(cfg, supervised_only=False, mode="sevenhot", progress=False,
                    split_column="split", tag=f"vae_collapse_s{seed}")
        model = res["model"]
        device = get_device(cfg.train.device)
        # eval its classifier on cnorm inclusive test, split by static_complete
        ds = TrajectoryDataset(CNORM, "test", enc, mode="sevenhot",
                               missing_static_fill="mean", split_column="split")
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                        pin_memory=(device.type == "cuda"))
        yt, yp = collect(model.classifier, dl, device)
        sc = ds.index["static_complete"].to_numpy().astype(bool)
        rows.append(metrics_row("complete", seed, yt[sc], yp[sc]))
        rows.append(metrics_row("dropped", seed, yt[~sc], yp[~sc]))
        rows.append(metrics_row("all", seed, yt, yp))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  done {round((time.time()-t0)/60,1)}min  complete={rows[-3]['macro_f1']} dropped={rows[-2]['macro_f1']}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["accuracy","macro_f1","recall_fishing","recall_passenger","recall_cargo","recall_tanker"]
    print("\n==== M2 VAE on cnorm populations (mean over seeds) ====")
    print(df.groupby("population")[cols].mean().round(1).to_string())
    print("\ncompare classifier-only (grid_A_cnorm): complete 91.2, dropped 23.4")
    print(f"-> {OUT}")
