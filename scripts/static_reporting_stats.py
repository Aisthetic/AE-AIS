"""Full-dataset descriptive stats for Finding 1 (replaces the 8-shard sample).

Over ALL interim train shards, computes:
  - message-level class distribution (the 'real ocean' mix)
  - per-class fraction of vessels that EVER report LEN / WID / DRA / all-three
    (a vessel 'reports' a field if any of its messages has it non-null)

Aggregates incrementally (shard by shard, needed columns only) so memory stays small.

Usage: PYTHONPATH=. python scripts/static_reporting_stats.py
Output: fullus2019/results/static_reporting_full.csv  (+ printed summary)
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

INTERIM = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/interim/train"
OUT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results/static_reporting_full.csv"
COLS = ["MMSI", "label", "Length", "Width", "Draft"]


def main():
    shards = sorted(glob.glob(f"{INTERIM}/*.parquet"))
    print(f"{len(shards)} train shards")

    msg_counts: dict[str, int] = {}          # message-level label counts
    # per-MMSI: label + ever-reported flags
    has = {}  # mmsi -> [label, has_len, has_wid, has_dra]

    for k, f in enumerate(shards):
        df = pd.read_parquet(f, columns=COLS)
        vc = df["label"].value_counts()
        for lbl, n in vc.items():
            msg_counts[lbl] = msg_counts.get(lbl, 0) + int(n)
        g = df.groupby("MMSI").agg(
            label=("label", "first"),
            hl=("Length", lambda s: s.notna().any()),
            hw=("Width", lambda s: s.notna().any()),
            hd=("Draft", lambda s: s.notna().any()),
        )
        for mmsi, row in g.iterrows():
            cur = has.get(mmsi)
            if cur is None:
                has[mmsi] = [row.label, bool(row.hl), bool(row.hw), bool(row.hd)]
            else:
                cur[1] |= bool(row.hl); cur[2] |= bool(row.hw); cur[3] |= bool(row.hd)
        if k % 20 == 0:
            print(f"  shard {k}/{len(shards)}", flush=True)

    ves = pd.DataFrame(
        [[l, hl, hw, hd] for (l, hl, hw, hd) in has.values()],
        columns=["label", "has_len", "has_wid", "has_dra"],
    )
    ves["has_all"] = ves.has_len & ves.has_wid & ves.has_dra

    total_msg = sum(msg_counts.values())
    classes = ["cargo", "tanker", "passenger", "fishing"]
    rows = []
    for c in classes:
        sub = ves[ves.label == c]
        n_v = len(sub)
        rows.append({
            "class": c,
            "msg_share_pct": round(100 * msg_counts.get(c, 0) / total_msg, 1),
            "n_vessels": n_v,
            "report_LEN_pct": round(100 * sub.has_len.mean(), 1) if n_v else None,
            "report_WID_pct": round(100 * sub.has_wid.mean(), 1) if n_v else None,
            "report_DRA_pct": round(100 * sub.has_dra.mean(), 1) if n_v else None,
            "report_ALL_pct": round(100 * sub.has_all.mean(), 1) if n_v else None,
        })
    out = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n==== FULL-DATASET STATIC-REPORTING STATS (train) ====")
    print(out.to_string(index=False))
    print(f"\ntotal messages: {total_msg:,}  unique vessels: {len(ves):,}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
