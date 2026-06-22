"""Static seven-hot bin-count sweep — close the ~86 vs paper-92 ceiling.

WO-LWD (kinematic-only) already matches the paper, so the gap is entirely in how
much the static channels (LEN/WID/DRA) contribute. The paper never disclosed the
static bin resolution; ours (WID10/LEN20/DRA10) may be too coarse. This sweeps
finer static resolutions and reports Full test accuracy to find what recovers ~92.

Kinematic bins (LAT/LON/SOG/COG) held fixed. Supervised CNN, 100% labels, temporal
split (paper protocol). Writes results incrementally.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/sweep_static_bins.py
"""
from __future__ import annotations
import dataclasses, json, time
import pandas as pd
from src.sslvtc.config import load_config
from src.sslvtc.train import train_supervised_classifier

CONFIG = "configs/fullus2019.yaml"
OUT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results/sweep_static_bins.csv"

# (name, WID, LEN, DRA) — kinematic LAT50/LON50/SOG30/COG36 fixed
GRID = [
    ("current_10_20_10", 10, 20, 10),
    ("2x_20_40_20",      20, 40, 20),
    ("3x_30_60_30",      30, 60, 30),
    ("fine_50_100_50",   50, 100, 50),
]


def run_one(wid, length, dra):
    cfg = load_config(CONFIG)
    bins = dict(cfg.encoding.bins)
    bins["WID"], bins["LEN"], bins["DRA"] = wid, length, dra
    enc = dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)
    c = dataclasses.replace(
        cfg,
        encoding=enc,
        train=dataclasses.replace(cfg.train, labeled_fraction=1.0),
    )
    m = train_supervised_classifier(c, mode="sevenhot", split_column="split", return_metrics=True)
    return m


if __name__ == "__main__":
    rows = []
    for name, wid, length, dra in GRID:
        print(f"\n=== {name} (WID{wid}/LEN{length}/DRA{dra}) ===", flush=True)
        t0 = time.time()
        m = run_one(wid, length, dra)
        row = {
            "config": name, "WID": wid, "LEN": length, "DRA": dra,
            "total_static_bins": wid + length + dra,
            "accuracy": round(m["accuracy"] * 100, 2),
            "macro_f1": round(m["macro_f1"] * 100, 2),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)  # incremental
        print(f"  -> Full acc={row['accuracy']} f1={row['macro_f1']} ({row['minutes']}min)  [paper Full=92.22]", flush=True)
    print("\n==== SWEEP DONE ====")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n-> {OUT}")
