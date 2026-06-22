"""Add a per-trajectory `static_complete` flag to an inclusive cohort's index.parquet.

static_complete = True iff the stored [T,8] tensor has NO NaN in the static columns
(WID=4, LEN=5, DRA=6) — i.e. the vessel reported all three static fields. This lets the
consequence experiment break down test metrics by complete vs incomplete vessels.

Usage: PYTHONPATH=. python scripts/tag_static_complete.py <processed_dir>
       (default: fullus2019_inclusive/processed)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

STATIC_COLS = [4, 5, 6]  # WID, LEN, DRA in the [T,8] matrix
DEFAULT = "/mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019_inclusive/processed"


def main(processed_dir: str):
    root = Path(processed_dir)
    idx = pd.read_parquet(root / "index.parquet")
    print(f"index rows: {len(idx)}")

    flags = np.empty(len(idx), dtype=bool)
    for i, rel in enumerate(idx["path"].tolist()):
        m = np.load(root / rel)  # [T, 8]
        flags[i] = not np.isnan(m[:, STATIC_COLS]).any()
        if i % 20000 == 0:
            print(f"  {i}/{len(idx)}", flush=True)

    idx["static_complete"] = flags
    idx.to_parquet(root / "index.parquet", index=False)

    n_complete = int(flags.sum())
    print(f"\nstatic_complete: {n_complete}/{len(idx)} ({100*n_complete/len(idx):.1f}%)")
    print("by split:")
    print(idx.groupby("split")["static_complete"].agg(["size", "sum"]))
    print("\nby label x static_complete (counts):")
    print(pd.crosstab(idx["label"], idx["static_complete"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
