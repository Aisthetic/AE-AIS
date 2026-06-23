"""Plot sampled vessel trajectories on their geographic regions, colored by class.

Reads the extracted cohort (index.parquet + tensors/*.npy + normalization_stats.json)
produced by `sslvtc` extraction, un-normalizes LAT/LON back to degrees, samples a few
trajectories per class, and draws them as polylines on a cartopy map per region.

Run on the machine that holds the processed data (the cluster), e.g.:

    python scripts/plot_trajectories.py \
        --region "US 2019"     /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/processed \
        --region "Danish 2019" /mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/processed \
        --per-class 40 --out paper/figures/f_traj_maps.png

Then copy paper/figures/f_traj_maps.png into overleaf/figures/ and recompile.
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

# Hard-coded extents [lon_min, lon_max, lat_min, lat_max] per region name prefix.
# US data normalization range spans lon -180..+154 (global); clip to conus+gulf.
EXTENTS = {
    "us":     [-130.0, -60.0,  18.0, 52.0],
    "danish": [   7.0,  16.5,  53.5, 58.8],
}


def _denorm(col: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return col * (hi - lo) + lo


def _load_region(processed: Path, per_class: int, rng: np.random.Generator):
    """Return list of (label, lat[], lon[]) for sampled trajectories."""
    index = pd.read_parquet(processed / "index.parquet")
    stats = json.loads((processed / "normalization_stats.json").read_text())
    lat_lo, lat_hi = stats["LAT"]["min"], stats["LAT"]["max"]
    lon_lo, lon_hi = stats["LON"]["min"], stats["LON"]["max"]

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


def _get_extent(name: str, trajs: list) -> list[float]:
    """Return [lon_min, lon_max, lat_min, lat_max]: named override or data bounds + 5% pad."""
    key = name.lower().split()[0]
    if key in EXTENTS:
        return EXTENTS[key]
    lats = np.concatenate([t[1] for t in trajs])
    lons = np.concatenate([t[2] for t in trajs])
    lo, la = np.nanmin(lons), np.nanmin(lats)
    hi, ha = np.nanmax(lons), np.nanmax(lats)
    pw, ph = (hi - lo) * 0.05, (ha - la) * 0.05
    return [lo - pw, hi + pw, la - ph, ha + ph]


def _draw_panel_cartopy(fig, pos, trajs, title, extent):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(pos, projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#eaecee", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555", zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#999", zorder=2)
    ax.add_feature(cfeature.LAKES,     facecolor="#d6eaf8", linewidth=0.2, zorder=2)

    lon_min, lon_max, lat_min, lat_max = extent
    # gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aaa", alpha=0.6, zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    for cls, lat, lon in trajs:
        # clip to extent so cartopy doesn't try to wrap across antimeridian
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if mask.sum() < 2:
            continue
        ax.plot(lon[mask], lat[mask], lw=0.5, alpha=0.55, color=C[cls],
                transform=proj, zorder=4)

    # legend
    handles = [plt.Line2D([0], [0], color=C[c], lw=2, label=c.capitalize())
               for c in CLASSES if any(t[0] == c for t in trajs)]
    ax.legend(handles=handles, fontsize=7, loc="lower left",
              framealpha=0.85, edgecolor="#ccc")
    ax.set_title(title, fontsize=10, pad=4)
    return ax


def _draw_panel_fallback(ax, trajs, title, extent):
    """No-cartopy fallback: plain lon/lat with aspect correction."""
    lon_min, lon_max, lat_min, lat_max = extent
    for cls, lat, lon in trajs:
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if mask.sum() < 2:
            continue
        ax.plot(lon[mask], lat[mask], lw=0.5, alpha=0.55, color=C[cls])

    handles = [plt.Line2D([0], [0], color=C[c], lw=2, label=c.capitalize())
               for c in CLASSES if any(t[0] == c for t in trajs)]
    ax.legend(handles=handles, fontsize=7, loc="lower left")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    mid_lat = (lat_min + lat_max) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)))
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", nargs=2, action="append", metavar=("NAME", "PROCESSED_DIR"),
                   required=True)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("paper/figures/f_traj_maps.png"))
    args = p.parse_args()

    try:
        import cartopy.crs as ccrs  # noqa: F401
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    rng = np.random.default_rng(args.seed)
    n = len(args.region)

    # Load all regions first so we know extents
    all_trajs = []
    all_extents = []
    for name, proc in args.region:
        trajs = _load_region(Path(proc), args.per_class, rng)
        extent = _get_extent(name, trajs)
        all_trajs.append((name, trajs))
        all_extents.append(extent)

    if has_cartopy:
        fig = plt.figure(figsize=(5.2 * n, 4.6))
        for i, ((name, trajs), extent) in enumerate(zip(all_trajs, all_extents)):
            pos = int(f"1{n}{i+1}")  # e.g. 121, 122
            ax = _draw_panel_cartopy(fig, pos, trajs, name, extent)
    else:
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
        if n == 1:
            axes = [axes]
        for ax, (name, trajs), extent in zip(axes, all_trajs, all_extents):
            _draw_panel_fallback(ax, trajs, name, extent)

    fig.suptitle("Sampled vessel trajectories by class", fontsize=12, y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
