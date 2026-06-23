"""
MNAR check on Danish Maritime Authority AIS (2019).

Streams the monthly zip from S3 in 10 MB range-request chunks, decompresses
on-the-fly with raw DEFLATE, stops after TARGET_ROWS rows.  No full download needed.

URL structure: http://aisdata.ais.dk/2019/aisdk-2019-{MM}.zip
Column of interest: "Ship type" (text), "Draught", "MMSI"

Output: paper/external_validity_danish.csv
Run: PYTHONPATH=. python scripts/mnar_danish_ais.py
"""
import struct, zlib, sys, io, requests, pandas as pd
from pathlib import Path

URL          = "http://aisdata.ais.dk/2019/aisdk-2019-05.zip"
TARGET_ROWS  = 2_000_000
CHUNK        = 10 * 1024 * 1024  # 10 MB per range request
DATA_DIR     = Path("/mnt/storage_1_10T/zezzahed/AIS_Data/external/danishais_raw")
OUT_CSV      = Path("paper/external_validity_danish.csv")

TEXT2CLS = {
    "fishing":   "fishing",
    "passenger": "passenger",
    "cargo":     "cargo",
    "tanker":    "tanker",
}


def get_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    return r.content


def find_data_offset(url: str) -> int:
    """Parse zip local-file header to find where DEFLATE stream begins."""
    hdr = get_range(url, 0, 65535)
    if hdr[:4] != b'PK\x03\x04':
        raise ValueError("Not a valid zip file")
    compression = struct.unpack_from('<H', hdr, 8)[0]
    if compression != 8:
        raise ValueError(f"Expected DEFLATE (8), got {compression} — cannot stream")
    fname_len = struct.unpack_from('<H', hdr, 26)[0]
    extra_len = struct.unpack_from('<H', hdr, 28)[0]
    return 30 + fname_len + extra_len


def stream_rows(url: str, target: int) -> pd.DataFrame:
    data_offset = find_data_offset(url)
    print(f"DEFLATE stream starts at byte {data_offset}", flush=True)

    decomp  = zlib.decompressobj(-zlib.MAX_WBITS)
    lines   = []
    partial = ""
    offset  = data_offset
    header  = None

    while len(lines) < target + 1:  # +1 for header
        raw = get_range(url, offset, offset + CHUNK - 1)
        if not raw:
            break
        try:
            text = decomp.decompress(raw).decode("utf-8", errors="replace")
        except zlib.error as e:
            print(f"zlib stopped at byte {offset}: {e}", file=sys.stderr)
            break

        text  = partial + text
        parts = text.split("\n")
        partial = parts[-1]
        lines.extend(parts[:-1])

        offset += CHUNK
        data_rows = max(0, len(lines) - 1)
        print(f"  {data_rows:,} rows | {offset//1_000_000:.0f} MB downloaded", flush=True)

    lines.append(partial)
    raw_text = "\n".join(lines[:target + 1])
    df = pd.read_csv(io.StringIO(raw_text), on_bad_lines="skip", low_memory=False)
    print(f"Parsed {len(df):,} rows, columns: {list(df.columns[:6])} ...", flush=True)
    return df


def map_class(s):
    if not isinstance(s, str):
        return None
    sl = s.lower()
    for k, v in TEXT2CLS.items():
        if k in sl:
            return v
    return None


def mnar_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names (DMA uses title-case with spaces)
    df.columns = df.columns.str.strip()
    ship_col    = next(c for c in df.columns if "ship type" in c.lower() or "shiptype" in c.lower())
    draught_col = next(c for c in df.columns if "draught" in c.lower() or "draft" in c.lower())
    mmsi_col    = next(c for c in df.columns if "mmsi" in c.lower())

    df["cls"]     = df[ship_col].apply(map_class)
    df["draught"] = pd.to_numeric(df[draught_col], errors="coerce")
    df = df.dropna(subset=["cls"])

    # Per-vessel: does it EVER report a non-null / non-zero draught?
    ves = (
        df.groupby([mmsi_col, "cls"])
        .agg(has_draft=("draught", lambda s: (s.notna() & (s > 0)).any()))
        .reset_index()
    )
    stats = (
        ves.groupby("cls")["has_draft"]
        .agg(vessels="count", pct_with_draft=lambda x: round(x.mean() * 100, 1))
        .reset_index()
        .sort_values("pct_with_draft")
    )
    return stats


if __name__ == "__main__":
    print(f"Streaming up to {TARGET_ROWS:,} rows from {URL}", flush=True)
    df   = stream_rows(URL, TARGET_ROWS)
    stats = mnar_stats(df)

    print("\n=== MNAR draft-reporting rate by class (Danish AIS 2019-05) ===")
    print(stats.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    print("\n--- US baseline (for comparison) ---")
    print("cargo 89.7% | tanker 84.7% | passenger 23.6% | fishing 12.1%")
    print("If Danish fishing << cargo/tanker -> MNAR confirmed cross-region.")
