"""Download Danish AIS daily zip files from web.ais.dk.

NOTE: web.ais.dk has an expired SSL cert as of 2026. This script disables
SSL verification — only safe because DMA is a known government source.

Usage:
    python scripts/download_danish_ais.py
    python scripts/download_danish_ais.py --dates 2019-05-01 2019-05-02 2019-05-03
"""
from __future__ import annotations
import argparse
import ssl
import urllib.request
from pathlib import Path

DATES = ["2019-05-01", "2019-05-02", "2019-05-03"]
BASE_URL = "https://web.ais.dk/aisdata/aisdk-{date}.zip"
OUT_DIR = Path(__file__).parent.parent / "danishais_raw"


def download(dates: list[str]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx)
    )

    for date in dates:
        url = BASE_URL.format(date=date)
        dest = OUT_DIR / f"aisdk-{date}.zip"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        print(f"downloading {url} ...", flush=True)
        with opener.open(url) as resp, open(dest, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        print(f"  -> {dest.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=DATES)
    args = ap.parse_args()
    download(args.dates)
