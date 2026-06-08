"""Generate synthetic MarineCadastre-format daily AIS CSVs for smoke-testing.

Writes AIS_2019_MM_DD.csv files with the 4 vessel classes into a target dir.
Not real data — only exercises the pipeline wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_CODES = {"fishing": 30, "passenger": 60, "cargo": 70, "tanker": 80}
# distinct kinematic/static signatures per class so the classifier can learn
PROFILES = {
    "fishing":   dict(sog=3.0, length=25, width=8, draft=3, lat0=40.55, lon0=-74.05),
    "passenger": dict(sog=18.0, length=120, width=18, draft=6, lat0=40.65, lon0=-74.00),
    "cargo":     dict(sog=12.0, length=200, width=30, draft=11, lat0=40.50, lon0=-73.95),
    "tanker":    dict(sog=10.0, length=250, width=40, draft=14, lat0=40.70, lon0=-73.85),
}


def make_vessel(mmsi, cls, day, n=200, span_h=8.0, rng=None):
    rng = rng or np.random.default_rng(mmsi)
    p = PROFILES[cls]
    start = pd.Timestamp(f"2019-{day[0]:02d}-{day[1]:02d} 00:00:00")
    secs = np.sort(rng.uniform(0, span_h * 3600, n))
    times = pd.to_datetime(secs, unit="s", origin=start)
    sog = np.clip(rng.normal(p["sog"], 1.0, n), 0.1, 40)
    return pd.DataFrame({
        "MMSI": mmsi,
        "BaseDateTime": times.strftime("%Y-%m-%dT%H:%M:%S"),
        "LAT": p["lat0"] + np.cumsum(rng.normal(0, 0.0005, n)),
        "LON": p["lon0"] + np.cumsum(rng.normal(0, 0.0005, n)),
        "SOG": sog,
        "COG": rng.uniform(0, 360, n),
        "VesselType": CLASS_CODES[cls],
        "Length": p["length"], "Width": p["width"], "Draft": p["draft"],
    })


def main(out_dir: str, vessels_per_class_per_day: int = 6):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # months 1-4 train, 5 val, 6 test; one day each for speed
    days = [(m, 15) for m in (1, 2, 3, 4, 5, 6)]
    mmsi = 100000
    for day in days:
        frames = []
        for cls in CLASS_CODES:
            for _ in range(vessels_per_class_per_day):
                frames.append(make_vessel(mmsi, cls, day, rng=rng))
                mmsi += 1
        df = pd.concat(frames, ignore_index=True)
        name = f"AIS_2019_{day[0]:02d}_{day[1]:02d}.csv"
        df.to_csv(out / name, index=False)
        print(f"wrote {name}: {len(df)} rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
