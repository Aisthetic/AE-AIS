"""Ingest raw MarineCadastre AIS zips/CSVs into filtered, labeled parquet.

Steps: column-name normalization, keep required columns, bbox + range sanity
filter, map VesselType -> 4-class label, tag split by source month, write parquet
partitioned by month.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

from . import CLASS_TO_IDX
from .config import PipelineConfig

# Standard column names we keep downstream.
KEEP = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Length", "Width", "Draft", "VesselType"]

# Lowercase raw header -> standard name (MarineCadastre varies across years).
COLUMN_ALIASES = {
    "mmsi": "MMSI",
    "basedatetime": "BaseDateTime",
    "lat": "LAT", "latitude": "LAT",
    "lon": "LON", "longitude": "LON",
    "sog": "SOG",
    "cog": "COG",
    "length": "Length", "lengthm": "Length",
    "width": "Width", "beam": "Width", "widthm": "Width",
    "draft": "Draft", "draught": "Draft",
    "vesseltype": "VesselType", "shiptype": "VesselType",
}


def vessel_type_to_label(code: float) -> str | None:
    """Map an AIS VesselType code to one of the 4 classes, else None (dropped)."""
    if pd.isna(code):
        return None
    c = int(code)
    if c == 30:
        return "fishing"
    if 60 <= c <= 69:
        return "passenger"
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
    return df.rename(columns=mapping)


def _split_for_month(month: int, cfg: PipelineConfig) -> str | None:
    if month in cfg.ingest.train_months:
        return "train"
    if month in cfg.ingest.val_months:
        return "val"
    if month in cfg.ingest.test_months:
        return "test"
    return None


def _open_csv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks from a .csv or a .zip containing one CSV."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as fh:
                yield from pd.read_csv(fh, chunksize=chunk_size)
    else:
        yield from pd.read_csv(path, chunksize=chunk_size)


def _clean_chunk(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = _normalize_columns(df)
    missing = [c for c in ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselType"] if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns after aliasing: {missing}")
    for col in ["Length", "Width", "Draft"]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[KEEP].copy()

    # numeric coercion
    for col in ["LAT", "LON", "SOG", "COG", "Length", "Width", "Draft", "VesselType"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # range sanity
    df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)]
    df = df[df["SOG"].between(0, cfg.ingest.max_sog_knots)]

    # bbox
    b = cfg.bbox
    if b.lat_min is not None:
        df = df[df["LAT"] >= b.lat_min]
    if b.lat_max is not None:
        df = df[df["LAT"] <= b.lat_max]
    if b.lon_min is not None:
        df = df[df["LON"] >= b.lon_min]
    if b.lon_max is not None:
        df = df[df["LON"] <= b.lon_max]

    # label
    df["label"] = df["VesselType"].map(vessel_type_to_label)
    df = df[df["label"].notna()]
    df["label_idx"] = df["label"].map(CLASS_TO_IDX).astype("int8")
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
    df = df[df["BaseDateTime"].notna()]
    return df


def ingest_file(path: Path, cfg: PipelineConfig) -> pd.DataFrame:
    """Read+clean one raw file; returns concatenated cleaned rows (may be empty)."""
    parts = [_clean_chunk(chunk, cfg) for chunk in _open_csv_chunks(path, cfg.ingest.chunk_size)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=KEEP + ["label", "label_idx"])
    return pd.concat(parts, ignore_index=True)


def _month_from_name(name: str, cfg: PipelineConfig) -> int | None:
    # AIS_2019_01_15.zip -> month 01
    parts = name.split("_")
    try:
        return int(parts[2])
    except (IndexError, ValueError):
        return None


def ingest_all(cfg: PipelineConfig) -> Path:
    """Ingest every raw file, write per-split parquet under paths.interim."""
    raw_dir = Path(cfg.paths.raw)
    out_dir = Path(cfg.paths.interim)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([*raw_dir.glob("*.zip"), *raw_dir.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"no raw .zip/.csv files in {raw_dir}")

    buckets: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for path in tqdm(files, desc="ingest", unit="file"):
        month = _month_from_name(path.name, cfg)
        split = _split_for_month(month, cfg) if month else None
        if split is None:
            tqdm.write(f"skip {path.name}: month {month} not in any split")
            continue
        df = ingest_file(path, cfg)
        if not df.empty:
            df["split"] = split
            buckets[split].append(df)

    written = []
    for split, frames in buckets.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        dest = out_dir / f"clean_{split}.parquet"
        combined.to_parquet(dest, index=False)
        written.append(dest)
    return out_dir
