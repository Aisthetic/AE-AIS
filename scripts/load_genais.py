"""Adapter: load real AIS from the sibling GenAIS project (NY Harbor, June 2022)
into SSL-VTC's interim clean_{split}.parquet format.

Single month -> the paper's Jan-Apr/May/Jun split is impossible, so we split by
DAY-OF-MONTH (configurable) purely to get real-data numbers locally. The cluster
run will use the proper temporal split on full Jan-Jun 2019.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sslvtc import CLASS_TO_IDX  # noqa: E402
from sslvtc.config import load_config  # noqa: E402
from sslvtc.ingest import vessel_type_to_label  # noqa: E402

SRC = "/Users/zak/Desktop/Code/GenAIS/data/processed/trajectories_clean.parquet"
COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Length", "Width", "Draft", "VesselType"]

# day-of-month boundaries
TRAIN_DAYS = range(1, 19)   # 1-18
VAL_DAYS = range(19, 25)    # 19-24
TEST_DAYS = range(25, 32)   # 25-30


def split_for_day(day: int) -> str | None:
    if day in TRAIN_DAYS:
        return "train"
    if day in VAL_DAYS:
        return "val"
    if day in TEST_DAYS:
        return "test"
    return None


def main(config_path: str):
    cfg = load_config(config_path)
    out = Path(cfg.paths.interim)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC, columns=COLS)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"]).dt.tz_localize(None)
    df["label"] = df["VesselType"].map(vessel_type_to_label)
    df = df[df["label"].notna()].copy()
    df["label_idx"] = df["label"].map(CLASS_TO_IDX).astype("int8")
    df["split"] = df["BaseDateTime"].dt.day.map(split_for_day)
    df = df[df["split"].notna()]

    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        dest = out / f"clean_{split}.parquet"
        part.to_parquet(dest, index=False)
        print(f"{split}: {len(part):>9} msgs, {part.MMSI.nunique()} vessels -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/nyharbor.yaml")
