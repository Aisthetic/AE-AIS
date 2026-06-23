"""Plot ALL vessel trajectories on their geographic regions, colored by class.

Uses packed_all.npy + LineCollection for fast loading and rendering of the full cohort.

    python scripts/plot_trajectories.py \
        --region "US 2019"     /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/processed \
        --region "Danish 2019" /mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/processed \
        --out paper/figures/f_traj_maps.png

Then copy paper/figures/f_traj_maps.png into overleaf/figures/ and recompile.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import pandas as pd

C      = {"cargo": "#1a6faf", "tanker": "#e07b00", "passenger": "#1a9641", "fishing": "#d7191c"}
ZORDER = {"cargo": 4, "tanker": 5, "passenger": 6, "fishing": 7}
CLASSES = ["cargo", "tanker", "passenger", "fishing"]

EXTENTS = {
    "us":     [-130.0, -60.0,  18.0, 52.0],
    "danish": [   7.0,  16.5,  53.5, 58.8],
}


def _denorm(col: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return col * (hi - lo) + lo


def _build_land_mask(extent: list[float]) -> object:
    """Return a prepared shapely geometry covering land within the extent."""
    import shapely
    import shapely.prepared
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    shpfile = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    reader  = shpreader.Reader(shpfile)
    clip    = box(extent[0], extent[2], extent[1], extent[3])
    polys   = [g.geometry for g in reader.records()
               if g.geometry.intersects(clip)]
    land    = shapely.unary_union(polys)
    return shapely.prepared.prep(land)


def _filter_jumps(cls_arrays: dict, max_step_deg: float = 2.0) -> dict:
    """Drop trajectories with any consecutive step > max_step_deg (GPS teleport artifacts)."""
    out = {}
    for cls, lonlat in cls_arrays.items():
        if len(lonlat) == 0:
            out[cls] = lonlat
            continue
        # diff along time axis; shape [N, T-1, 2]
        steps = np.abs(np.diff(lonlat.astype("float32"), axis=1))
        bad = (steps > max_step_deg).any(axis=(1, 2))
        out[cls] = lonlat[~bad]
        if bad.sum():
            print(f"  {cls}: dropped {bad.sum():,} jump artifacts; {(~bad).sum():,} kept")
    return out


def _filter_ocean_only(cls_arrays: dict, land, threshold: float = 0.20) -> dict:
    """Drop trajectories where > threshold fraction of points fall on land.

    A threshold < 1.0 tolerates port-area points that register as land at 50 m
    resolution, while still removing trajectories that genuinely cross inland.
    """
    import shapely
    out = {}
    for cls, lonlat in cls_arrays.items():
        if len(lonlat) == 0:
            out[cls] = lonlat
            continue
        N, T, _ = lonlat.shape
        lons = lonlat[:, :, 0].ravel().astype("float64")
        lats = lonlat[:, :, 1].ravel().astype("float64")
        on_land = shapely.contains_xy(land.context, lons, lats).reshape(N, T)
        land_frac = on_land.mean(axis=1)          # fraction of points on land per traj
        keep = land_frac <= threshold
        out[cls] = lonlat[keep]
        dropped = N - keep.sum()
        if dropped:
            print(f"  {cls}: dropped {dropped:,} inland trajectories "
                  f"(>{threshold:.0%} pts on land); {keep.sum():,} kept")
    return out


def _load_region(processed: Path, stride: int) -> dict[str, np.ndarray]:
    """Return {cls: float32 array [N, T//stride, 2]} (lat, lon) for all trajectories."""
    index = pd.read_parquet(processed / "index.parquet")
    stats = json.loads((processed / "normalization_stats.json").read_text())
    lat_lo, lat_hi = stats["LAT"]["min"], stats["LAT"]["max"]
    lon_lo, lon_hi = stats["LON"]["min"], stats["LON"]["max"]

    # packed_all.npy: [total_trajs, T, 8]; packed_all_ids.json: {traj_id: row_idx}
    packed = np.load(processed / "packed_all.npy", mmap_mode="r")
    id_map = json.loads((processed / "packed_all_ids.json").read_text())

    label_col = "label" if "label" in index.columns else "label_idx"
    out: dict[str, list] = {c: [] for c in CLASSES}

    for cls in CLASSES:
        sub = index[index[label_col].astype(str).str.lower() == cls]
        if len(sub) == 0:
            continue
        rows = [id_map[tid] for tid in sub["traj_id"].astype(str) if tid in id_map]
        if not rows:
            continue
        # vectorized load: [N, T, 8]
        block = packed[rows, ::stride, :]
        lat = (_denorm(block[:, :, 0].astype("float64"), lat_lo, lat_hi)).astype("float32")
        lon = (_denorm(block[:, :, 1].astype("float64"), lon_lo, lon_hi)).astype("float32")
        out[cls] = np.stack([lon, lat], axis=-1)  # [N, T', 2]
        print(f"  {cls}: {len(rows):,} trajectories")

    return out


def _get_extent(name: str) -> list[float]:
    key = name.lower().split()[0]
    return EXTENTS.get(key, None)


def _alpha_lw(extent: list[float], n_trajs: int) -> tuple[float, float]:
    """Per-panel alpha: scales with area and inversely with trajectory count."""
    lon_min, lon_max, lat_min, lat_max = extent
    area = (lon_max - lon_min) * (lat_max - lat_min)
    # target: ~5 overlapping trajectories per pixel = readable lanes
    # alpha s.t. 1-(1-alpha)^5 ≈ 0.5  →  alpha ≈ 0.13; scale by sqrt(area/n)
    ref_density = 500 / 50.0  # trajs per sq-deg at reference Danish 500-sample setting
    density = n_trajs / area
    alpha = min(0.50, max(0.015, 0.13 * (ref_density / density) ** 0.5))
    lw = min(1.0, max(0.4, 0.55 * (area / 50.0) ** 0.25))
    return alpha, lw


def _make_segments(lonlat: np.ndarray, lon_min, lon_max, lat_min, lat_max):
    """Convert [N, T, 2] to list of (M,2) arrays clipped to extent, one per trajectory."""
    segs = []
    for traj in lonlat:
        lon, lat = traj[:, 0], traj[:, 1]
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if mask.sum() < 2:
            continue
        segs.append(traj[mask])
    return segs


def _draw_panel_cartopy(fig, pos, cls_arrays: dict, title: str, extent: list[float]):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(pos, projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#eaecee", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555", zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#999", zorder=2)
    ax.add_feature(cfeature.LAKES,     facecolor="#d6eaf8", linewidth=0.2, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aaa", alpha=0.6, zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    lon_min, lon_max, lat_min, lat_max = extent
    n_total = sum(len(v) for v in cls_arrays.values() if len(v))
    alpha, lw = _alpha_lw(extent, n_total)

    for cls in CLASSES:
        lonlat = cls_arrays.get(cls)
        if lonlat is None or len(lonlat) == 0:
            continue
        segs = _make_segments(lonlat, lon_min, lon_max, lat_min, lat_max)
        if not segs:
            continue
        lc = mcoll.LineCollection(segs, colors=C[cls], linewidths=lw, alpha=alpha,
                                  transform=proj, zorder=ZORDER[cls])
        ax.add_collection(lc)

    handles = [plt.Line2D([0], [0], color=C[c], lw=2.5, label=c.capitalize(), alpha=0.9)
               for c in CLASSES if len(cls_arrays.get(c, [])) > 0]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9, edgecolor="#bbb")
    ax.set_title(title, fontsize=10, pad=4)
    return ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", nargs=2, action="append", metavar=("NAME", "PROCESSED_DIR"),
                   required=True)
    p.add_argument("--stride", type=int, default=4,
                   help="timestep stride (4 → every 4th point, reduces 160→40 pts/traj)")
    p.add_argument("--out", type=Path, default=Path("paper/figures/f_traj_maps.png"))
    args = p.parse_args()

    try:
        import cartopy.crs as ccrs  # noqa: F401
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        print("WARNING: cartopy not found, falling back to plain axes (no coastlines)")

    n = len(args.region)
    all_data = []
    for name, proc in args.region:
        print(f"Loading {name}...")
        cls_arrays = _load_region(Path(proc), args.stride)
        extent = _get_extent(name)
        print(f"  Filtering GPS jump artifacts...")
        cls_arrays = _filter_jumps(cls_arrays)
        print(f"  Building land mask for {name}...")
        land = _build_land_mask(extent)
        print(f"  Filtering land-crossing trajectories...")
        cls_arrays = _filter_ocean_only(cls_arrays, land, threshold=0.05)
        all_data.append((name, cls_arrays, extent))

    if has_cartopy:
        fig = plt.figure(figsize=(5.5 * n, 4.8))
        for i, (name, cls_arrays, extent) in enumerate(all_data):
            pos = int(f"1{n}{i+1}")
            _draw_panel_cartopy(fig, pos, cls_arrays, name, extent)
    else:
        raise RuntimeError("cartopy required; install with: pip install cartopy")

    fig.suptitle("Vessel trajectories by class (full cohort)", fontsize=12, y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
