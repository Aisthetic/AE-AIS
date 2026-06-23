"""Plot sampled vessel trajectories on their geographic regions, colored by class.

Reads the extracted cohort (index.parquet + tensors/*.npy + normalization_stats.json)
produced by `sslvtc` extraction, un-normalizes LAT/LON back to degrees, samples a few
trajectories per class, and draws them as polylines on a lon/lat map per region.

Run on the machine that holds the processed data (the cluster), e.g.:

    python scripts/plot_trajectories.py \
        --region "US 2019" /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/processed \
        --region "Danish 2019" /mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/processed \
        --per-class 40 --out paper/figures/f_traj_maps.png

Then copy paper/figures/f_traj_maps.png into overleaf/figures/ and recompile.

Coastlines are drawn if cartopy is installed; otherwise a clean lon/lat panel with an
aspect correction is used (no extra dependency required).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# class colors match paper/make_figures.py
C = {"cargo": "#1f77b4", "tanker": "#ff7f0e", "passenger": "#2ca02c", "fishing": "#d62728"}
CLASSES = ["cargo", "tanker", "passenger", "fishing"]


def _denorm(col: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return col * (hi - lo) + lo


def _load_region(processed: Path, per_class: int, rng: np.random.Generator):
    """Return list of (label, lat[], lon[]) for sampled trajectories."""
    index = pd.read_parquet(processed / "index.parquet")
    stats = json.loads((processed / "normalization_stats.json").read_text())
    lat_lo, lat_hi = stats["LAT"]["min"], stats["LAT"]["max"]
    lon_lo, lon_hi = stats["LON"]["min"], stats["LON"]["max"]

    # label column may be a string ("fishing") or already lowercase; normalize
    label_col = "label" if "label" in index.columns else "label_idx"
    out = []
    for cls in CLASSES:
        sub = index[index[label_col].astype(str).str.lower() == cls]
        if len(sub) == 0:
            continue
        take = sub.sample(min(per_class, len(sub)), random_state=int(rng.integers(1e9)))
        for _, row in take.iterrows():
            arr = np.load(processed / row["path"])  # [T, 8], normalized
            lat = _denorm(arr[:, 0].astype("float64"), lat_lo, lat_hi)
            lon = _denorm(arr[:, 1].astype("float64"), lon_lo, lon_hi)
            out.append((cls, lat, lon))
    return out


def _draw_panel(ax, trajs, title):
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#888")
        has_cartopy = True
    except Exception:
        has_cartopy = False

    for cls, lat, lon in trajs:
        ax.plot(lon, lat, lw=0.5, alpha=0.6, color=C[cls])
    # one legend handle per class
    for cls in CLASSES:
        ax.plot([], [], color=C[cls], lw=2, label=cls)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    if not has_cartopy and trajs:
        lats = np.concatenate([t[1] for t in trajs])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.nanmean(lats))))
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", nargs=2, action="append", metavar=("NAME", "PROCESSED_DIR"),
                   required=True, help="region label and its processed/ dir; repeatable")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("paper/figures/f_traj_maps.png"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = len(args.region)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, (name, proc) in zip(axes, args.region):
        trajs = _load_region(Path(proc), args.per_class, rng)
        _draw_panel(ax, trajs, f"{name}  ({args.per_class}/class sampled)")

    fig.suptitle("Sampled vessel trajectories by class", fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
