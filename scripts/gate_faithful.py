"""Decisive re-test: does the static gain collapse on a PAPER-STRENGTH model?

Uses fine static bins (WID50/LEN100/DRA50 -> Full acc 91.79 ~ paper 92.22) and
runs the Full classifier on BOTH splits with full metrics. WO-LWD is static-bin-
independent (no static), so we reuse the known values: temporal 72.86, vd 72.47.

Static gain (faithful) = Full_fine - WO-LWD, compared temporal vs vessel-disjoint.
  - gain stable across splits  -> no collapse even on strong model (pivot, bulletproof)
  - gain collapses on vd       -> original leakage critique revived

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/gate_faithful.py
"""
from __future__ import annotations
import dataclasses, json, time
import pandas as pd
from src.sslvtc.config import load_config
from src.sslvtc.train import train_supervised_classifier
from src.sslvtc import CLASS_TO_IDX

CONFIG = "configs/fullus2019.yaml"
OUT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results/gate_faithful_fine.csv"
WO_LWD = {"split": 72.86, "split_vd": 72.47}  # static-independent, from prior gate
FISH = CLASS_TO_IDX["fishing"]


def full_fine(split_col):
    cfg = load_config(CONFIG)
    bins = dict(cfg.encoding.bins); bins["WID"], bins["LEN"], bins["DRA"] = 50, 100, 50
    enc = dataclasses.replace(cfg.encoding, bins=bins, use_len=True, use_wid=True, use_dra=True)
    c = dataclasses.replace(cfg, encoding=enc, train=dataclasses.replace(cfg.train, labeled_fraction=1.0))
    return train_supervised_classifier(c, mode="sevenhot", split_column=split_col, return_metrics=True)


if __name__ == "__main__":
    rows = []
    for split_col in ("split", "split_vd"):
        print(f"\n=== Full (fine 50/100/50) on {split_col} ===", flush=True)
        t0 = time.time()
        m = full_fine(split_col)
        recalls = m.get("per_class_recall", [])
        full_acc = m["accuracy"] * 100
        row = {
            "split_protocol": split_col,
            "full_acc": round(full_acc, 2),
            "full_macro_f1": round(m["macro_f1"] * 100, 2),
            "full_fishing_recall": round(recalls[FISH] * 100, 2) if FISH < len(recalls) else None,
            "wo_lwd_acc": WO_LWD[split_col],
            "static_gain_acc": round(full_acc - WO_LWD[split_col], 2),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  -> Full acc={row['full_acc']} static_gain={row['static_gain_acc']} fishing_recall={row['full_fishing_recall']}", flush=True)
    df = pd.DataFrame(rows)
    print("\n==== FAITHFUL-MODEL GATE ====")
    print(df.to_string(index=False))
    if len(df) == 2:
        g = dict(zip(df.split_protocol, df.static_gain_acc))
        delta = g["split"] - g["split_vd"]
        print(f"\nstatic gain: temporal={g['split']}  vessel-disjoint={g['split_vd']}  Δ={delta:.2f}")
        print("VERDICT:", "COLLAPSE (original critique)" if delta > 5 else "NO COLLAPSE (pivot confirmed on strong model)")
