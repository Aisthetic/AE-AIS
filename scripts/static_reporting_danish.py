"""MNAR static-reporting rates by class for Danish AIS (external validity).

Mirrors static_reporting_stats.py exactly; only INTERIM and OUT paths differ.

Usage: PYTHONPATH=. python scripts/static_reporting_danish.py
Output: danishais2019/results/static_reporting_danish.csv
"""
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

INTERIM = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/interim/train"
OUT     = "/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results/static_reporting_danish.csv"
COLS    = ["MMSI", "label", "Length", "Width", "Draft"]


def main():
    shards = sorted(glob.glob(f"{INTERIM}/*.parquet"))
    print(f"{len(shards)} train shards")

    msg_counts: dict[str, int] = {}
    has: dict = {}  # mmsi -> [label, has_len, has_wid, has_dra]

    for k, f in enumerate(shards):
        df = pd.read_parquet(f, columns=COLS)
        for lbl, n in df["label"].value_counts().items():
            msg_counts[lbl] = msg_counts.get(lbl, 0) + int(n)
        g = df.groupby("MMSI").agg(
            label=("label", "first"),
            hl=("Length", lambda s: s.notna().any()),
            hw=("Width",  lambda s: s.notna().any()),
            hd=("Draft",  lambda s: s.notna().any()),
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
    rows = []
    for c in ["cargo", "tanker", "passenger", "fishing"]:
        sub = ves[ves.label == c]
        n_v = len(sub)
        rows.append({
            "class":          c,
            "msg_share_pct":  round(100 * msg_counts.get(c, 0) / total_msg, 1),
            "n_vessels":      n_v,
            "report_LEN_pct": round(100 * sub.has_len.mean(), 1) if n_v else None,
            "report_WID_pct": round(100 * sub.has_wid.mean(), 1) if n_v else None,
            "report_DRA_pct": round(100 * sub.has_dra.mean(), 1) if n_v else None,
            "report_ALL_pct": round(100 * sub.has_all.mean(), 1) if n_v else None,
        })

    out = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n==== DANISH STATIC-REPORTING STATS (train) ====")
    print(out.to_string(index=False))
    print(f"\ntotal messages: {total_msg:,}  unique vessels: {len(ves):,}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
