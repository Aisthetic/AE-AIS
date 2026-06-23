"""Ingest Danish Maritime Authority AIS monthly zips into interim parquets.

Format differences from US MarineCadastre:
  - Ship type is text ("Cargo", "Tanker", "Passenger", "Fishing"), not a numeric code
  - Each monthly zip contains one CSV per day (aisdk-2019-01-01.csv ...)
  - Timestamp format: "01/01/2019 00:00:00"
  - Column names: Latitude/Longitude, Width, Length, Draught (vs Draft)
  - "# Timestamp" header has a leading "#"

Output mirrors the US interim layout:
  interim/{train,val,test}/part_YYYY-MM-DD.parquet

Usage: PYTHONPATH=. python scripts/ingest_danish.py [config]
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sslvtc.config import load_config
from src.sslvtc import CLASS_TO_IDX

DEFAULT_CONFIG = "configs/danishais2019.yaml"

# Danish text "Ship type" -> 4-class label (lowercase strip for robustness)
DANISH_SHIP_TYPE_MAP: dict[str, str] = {
    "fishing":   "fishing",
    "cargo":     "cargo",
    "tanker":    "tanker",
    "passenger": "passenger",
}

KEEP = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG",
        "Length", "Width", "Draft", "VesselType"]

# Map raw Danish column names to standard names
DANISH_ALIASES = {
    "# timestamp":  "BaseDateTime",
    "timestamp":    "BaseDateTime",
    "mmsi":         "MMSI",
    "latitude":     "LAT",
    "longitude":    "LON",
    "sog":          "SOG",
    "cog":          "COG",
    "width":        "Width",
    "length":       "Length",
    "draught":      "Draft",
    "draft":        "Draft",
    "ship type":    "VesselType",
}


def _danish_ship_type_to_label(val) -> str | None:
    if pd.isna(val):
        return None
    return DANISH_SHIP_TYPE_MAP.get(str(val).strip().lower())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in DANISH_ALIASES:
            mapping[col] = DANISH_ALIASES[key]
    return df.rename(columns=mapping)


def _split_for_month(month: int, cfg) -> str | None:
    if month in cfg.ingest.train_months:
        return "train"
    if month in cfg.ingest.val_months:
        return "val"
    if month in cfg.ingest.test_months:
        return "test"
    return None


def _process_daily_csv(fh, cfg, month: int) -> pd.DataFrame:
    """Read one daily CSV file-handle and return cleaned rows."""
    parts = []
    for chunk in pd.read_csv(fh, chunksize=cfg.ingest.chunk_size,
                             dtype=str, low_memory=False):
        chunk = _normalize_columns(chunk)
        missing = [c for c in ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselType"]
                   if c not in chunk.columns]
        if missing:
            continue
        for col in ["Length", "Width", "Draft"]:
            if col not in chunk.columns:
                chunk[col] = pd.NA
        chunk = chunk[KEEP].copy()

        for col in ["LAT", "LON", "SOG", "COG", "Length", "Width", "Draft"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["MMSI"] = pd.to_numeric(chunk["MMSI"], errors="coerce").astype("Int64")

        chunk = chunk[chunk["LAT"].between(-90, 90) & chunk["LON"].between(-180, 180)]
        chunk = chunk[chunk["SOG"].between(0, cfg.ingest.max_sog_knots)]

        b = cfg.bbox
        if b.lat_min is not None: chunk = chunk[chunk["LAT"] >= b.lat_min]
        if b.lat_max is not None: chunk = chunk[chunk["LAT"] <= b.lat_max]
        if b.lon_min is not None: chunk = chunk[chunk["LON"] >= b.lon_min]
        if b.lon_max is not None: chunk = chunk[chunk["LON"] <= b.lon_max]

        chunk["label"] = chunk["VesselType"].map(_danish_ship_type_to_label)
        chunk = chunk[chunk["label"].notna()]
        if chunk.empty:
            continue
        chunk["label_idx"] = chunk["label"].map(CLASS_TO_IDX).astype("int8")
        chunk["BaseDateTime"] = pd.to_datetime(chunk["BaseDateTime"],
                                               format="%d/%m/%Y %H:%M:%S",
                                               errors="coerce")
        chunk = chunk[chunk["BaseDateTime"].notna()]
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ingest_monthly_zip(args: tuple) -> list[tuple[str, int]]:
    """Worker: ingest one monthly zip, write one parquet per day per split."""
    zip_path, cfg_path, interim_dir_str = args
    interim_dir = Path(interim_dir_str)
    cfg = load_config(cfg_path)

    zip_path = Path(zip_path)
    # filename: aisdk-2019-01.zip -> month 1
    try:
        month = int(zip_path.stem.split("-")[2])
    except (IndexError, ValueError):
        return []

    split = _split_for_month(month, cfg)
    if split is None:
        return []

    out_dir = interim_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with zipfile.ZipFile(zip_path) as zf:
        daily_csvs = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for csv_name in daily_csvs:
            # aisdk-2019-01-15.csv -> date string 2019-01-15
            parts = Path(csv_name).stem.split("-")
            try:
                date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
            except IndexError:
                date_str = Path(csv_name).stem

            out_path = out_dir / f"part_{date_str}.parquet"
            if out_path.exists():
                rows = pd.read_parquet(out_path, columns=["MMSI"]).shape[0]
                results.append((split, rows))
                continue

            with zf.open(csv_name) as fh:
                df = _process_daily_csv(fh, cfg, month)

            if df.empty:
                continue
            df.to_parquet(out_path, index=False)
            results.append((split, len(df)))

    return results


def ingest_danish(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    raw_dir = Path(cfg.paths.raw)
    interim_dir = Path(cfg.paths.interim)

    zip_files = sorted(raw_dir.glob("aisdk-2019-*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No aisdk-2019-*.zip files found in {raw_dir}")

    print(f"Found {len(zip_files)} monthly zips in {raw_dir}")

    tasks = [(str(z), config, str(interim_dir)) for z in zip_files]
    totals: dict[str, int] = {}

    for t in tqdm(tasks, desc="ingest monthly zips"):
        for split, n in _ingest_monthly_zip(t):
            totals[split] = totals.get(split, 0) + n

    print("Ingest complete:")
    for split, n in sorted(totals.items()):
        print(f"  {split}: {n:,} rows")
    print(f"  -> {interim_dir}")


if __name__ == "__main__":
    ingest_danish(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
