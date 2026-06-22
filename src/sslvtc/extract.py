"""Trajectory extraction — paper Section 3, exact 5-step procedure.

Input: cleaned per-split parquet (from ingest). Output: fixed-length seven-hot
trajectory tensors (.npy) + a parquet index (traj_id, label, label_idx, split, path).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SEVEN_ATTRS, PipelineConfig
from .encoding import (
    compute_norm_stats,
    compute_static_means,
    normalize_df_to_matrix,
    save_norm_stats,
)


def _divide(group: pd.DataFrame, max_gap_hours: float) -> list[pd.DataFrame]:
    """Step 1: split one MMSI's messages by calendar day, then on > gap cuts."""
    out: list[pd.DataFrame] = []
    group = group.sort_values("BaseDateTime")
    for _, day_df in group.groupby(group["BaseDateTime"].dt.date):
        dt = day_df["BaseDateTime"]
        gap = dt.diff().dt.total_seconds().div(3600.0)
        cut = (gap > max_gap_hours).cumsum()
        for _, seg in day_df.groupby(cut):
            out.append(seg)
    return out


def _passes_filter(traj: pd.DataFrame, cfg: PipelineConfig) -> bool:
    ex = cfg.extraction
    # Step 2: span and message count
    if len(traj) < ex.min_messages:
        return False
    span_h = (traj["BaseDateTime"].iloc[-1] - traj["BaseDateTime"].iloc[0]).total_seconds() / 3600.0
    if span_h < ex.min_span_hours:
        return False
    # Step 3: abnormal trajectory removal
    sog = traj["SOG"].to_numpy(dtype="float64")
    if np.nanmax(sog) <= ex.abnormal_max_sog:
        return False
    moving_fraction = float(np.mean(sog > ex.moving_sog_threshold))
    if moving_fraction <= ex.min_moving_fraction:
        return False
    # Paper-equivalent selection: require complete static info (LEN/WID/DRA).
    # The paper implicitly dropped vessels lacking static fields (its method
    # consumes them); this disproportionately removes fishing/small Class-B
    # vessels that rarely transmit Draft. Off by default (we keep them and
    # fill via missing_static); on to reproduce the paper's class balance.
    if ex.require_complete_static:
        for col in ("Length", "Width", "Draft"):
            if traj[col].isna().all():
                return False
    return True


def _sample_fixed(traj: pd.DataFrame, length: int) -> pd.DataFrame:
    """Step 5: subsample to `length` messages via evenly spaced indices."""
    n = len(traj)
    if n == length:
        return traj
    idx = np.linspace(0, n - 1, num=length).round().astype("int64")
    return traj.iloc[idx]


_N_MMSI_BUCKETS = 64  # number of MMSI-hash shards for repartition


def _process_mmsi_chunk(args: tuple) -> list[pd.DataFrame]:
    """Worker: run Steps 1-3+5 for a chunk of MMSIs."""
    chunk_df, cfg = args
    trajs: list[pd.DataFrame] = []
    for _, group in chunk_df.groupby("MMSI"):
        for seg in _divide(group, cfg.extraction.max_gap_hours):
            if _passes_filter(seg, cfg):
                trajs.append(_sample_fixed(seg, cfg.extraction.fixed_length))
    return trajs


def _repartition_by_mmsi(split_dir: Path, n_buckets: int) -> Path:
    """Stream day-sharded parquets → MMSI-hash bucket files; idempotent.

    Co-locates all messages for one MMSI (which can span many daily shards)
    so that _divide() sees the full message history per vessel.  Each bucket
    file is written incrementally via pyarrow.ParquetWriter so at most one
    day's data (~785 MB uncompressed) is held in RAM at once.
    """
    bymmsi_dir = split_dir.parent / f"{split_dir.name}_bymmsi"
    if bymmsi_dir.exists() and any(bymmsi_dir.glob("bucket_*.parquet")):
        tqdm.write(f"  reusing bymmsi partition: {bymmsi_dir.name}")
        return bymmsi_dir
    bymmsi_dir.mkdir(exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    day_files = sorted(split_dir.glob("part_*.parquet"))
    if not day_files:
        raise FileNotFoundError(f"no part_*.parquet shards in {split_dir}")

    writers: dict[int, pq.ParquetWriter] = {}
    schema: pa.Schema | None = None

    for day_file in tqdm(day_files, desc=f"partition {split_dir.name}", leave=False):
        day_df = pd.read_parquet(day_file)
        if day_df.empty:
            continue
        table = pa.Table.from_pandas(day_df, preserve_index=False)
        if schema is None:
            schema = table.schema
        buckets = day_df["MMSI"].values.astype(np.int64) % n_buckets
        for b in range(n_buckets):
            mask = buckets == b
            if not mask.any():
                continue
            subset = table.take(np.where(mask)[0])
            if b not in writers:
                writers[b] = pq.ParquetWriter(
                    str(bymmsi_dir / f"bucket_{b:03d}.parquet"), schema
                )
            writers[b].write_table(subset)

    for w in writers.values():
        w.close()
    return bymmsi_dir


def _process_bucket(args: tuple) -> list[pd.DataFrame]:
    """Worker: read one MMSI-hash bucket file, run extraction steps 1-3+5."""
    bucket_path_str, cfg = args
    df = pd.read_parquet(bucket_path_str)
    if df.empty:
        return []
    return _process_mmsi_chunk((df, cfg))


def _compute_dt(traj: pd.DataFrame, max_gap_hours: float) -> np.ndarray:
    """Normalized Δt between consecutive messages; first row = 0. [T] float32 in [0,1]."""
    times = traj["BaseDateTime"].to_numpy(dtype="datetime64[s]").astype("float64")
    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    max_dt = max_gap_hours * 3600.0
    return np.clip(dt / max_dt, 0.0, 1.0).astype("float32")


def _save_tensor(args: tuple) -> dict:
    """Worker: normalize one trajectory and save as .npy [T, 8]; returns index row.

    Column layout: [LAT, LON, SOG, COG, WID, LEN, DRA, DT] where DT = normalized
    inter-message time interval (col 7). Encoding functions consume cols 0:7 for
    seven-hot/raw; col 7 is passed through for the temporal transformer.
    """
    traj, split, i, stats, processed_str, max_gap_hours = args
    from .encoding import normalize_df_to_matrix
    matrix7 = normalize_df_to_matrix(traj, stats)           # [T, 7]
    dt = _compute_dt(traj, max_gap_hours)[:, None]           # [T, 1]
    matrix = np.concatenate([matrix7, dt], axis=1)           # [T, 8]
    traj_id = f"{split}_{i:06d}"
    rel = f"tensors/{traj_id}.npy"
    np.save(Path(processed_str) / rel, matrix)
    return {
        "traj_id": traj_id,
        "split": split,
        "label": traj["label"].iloc[0],
        "label_idx": int(traj["label_idx"].iloc[0]),
        "mmsi": int(traj["MMSI"].iloc[0]),
        "path": rel,
        "matrix": matrix[:, :7] if split == "train" else None,  # stats use cols 0:7
    }


def extract_split(
    source: "Path | pd.DataFrame", cfg: PipelineConfig, n_workers: int = 1
) -> list[pd.DataFrame]:
    """Run Steps 1-3+5 for one split; returns fixed-length trajectory DataFrames.

    source can be:
    - a Path to a directory of day shards (new hardened format)
    - a Path to a single .parquet file (legacy gulf2019 format)
    - a pd.DataFrame (for tests / programmatic use)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    n_workers = n_workers or os.cpu_count() or 4

    if isinstance(source, pd.DataFrame):
        mmsis = source["MMSI"].unique()
        chunks = np.array_split(mmsis, n_workers)
        args = [(source[source["MMSI"].isin(chunk)].copy(), cfg) for chunk in chunks if len(chunk)]
        trajectories: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for result in pool.map(_process_mmsi_chunk, args):
                trajectories.extend(result)
        return trajectories

    if source.is_dir():
        bymmsi_dir = _repartition_by_mmsi(source, _N_MMSI_BUCKETS)
        bucket_files = sorted(bymmsi_dir.glob("bucket_*.parquet"))
        args = [(str(bf), cfg) for bf in bucket_files]
        trajectories: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for result in tqdm(
                pool.map(_process_bucket, args),
                total=len(args),
                desc="extract buckets",
                leave=False,
            ):
                trajectories.extend(result)
        return trajectories

    # Legacy: single parquet file (gulf2019 or similar small regions).
    df = pd.read_parquet(source)
    mmsis = df["MMSI"].unique()
    chunks = np.array_split(mmsis, n_workers)
    args = [(df[df["MMSI"].isin(chunk)].copy(), cfg) for chunk in chunks if len(chunk)]
    trajectories = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_process_mmsi_chunk, args):
            trajectories.extend(result)
    return trajectories


def extract_all(cfg: PipelineConfig, workers: int | None = None) -> Path:
    """Full extraction across train/val/test; writes tensors + index parquet.

    Normalization stats are computed on the TRAIN split only and reused.
    Handles both the new partitioned-shard format (interim/{split}/) and the
    legacy single-file format (interim/clean_{split}.parquet).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = workers or os.cpu_count() or 4

    interim = Path(cfg.paths.interim)
    processed = Path(cfg.paths.processed)
    tensor_dir = processed / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    per_split: dict[str, list[pd.DataFrame]] = {}
    for split in ("train", "val", "test"):
        split_dir = interim / split
        legacy_path = interim / f"clean_{split}.parquet"
        if split_dir.is_dir() and any(split_dir.glob("part_*.parquet")):
            source = split_dir
        elif legacy_path.exists():
            source = legacy_path
        else:
            continue
        tqdm.write(f"extracting {split} ({source})...")
        trajs = extract_split(source, cfg, n_workers=n_workers)
        tqdm.write(f"  -> {len(trajs)} trajectories")
        if trajs:
            per_split[split] = trajs

    if "train" not in per_split or not per_split["train"]:
        raise RuntimeError("no train trajectories extracted; check bbox/thresholds/data")

    norm_from = cfg.extraction.norm_stats_from
    if norm_from:
        from .encoding import load_norm_stats
        stats = load_norm_stats(Path(norm_from) / "normalization_stats.json")
        tqdm.write(f"reusing normalization stats from {norm_from}")
    else:
        train_concat = pd.concat(per_split["train"], ignore_index=True)
        stats = compute_norm_stats(train_concat, SEVEN_ATTRS)
    save_norm_stats(stats, processed / "normalization_stats.json")

    # Parallel tensor save across all splits.
    max_gap_hours = cfg.extraction.max_gap_hours
    all_args = [
        (traj, split, i, stats, str(processed), max_gap_hours)
        for split, trajs in per_split.items()
        for i, traj in enumerate(trajs)
    ]
    rows = []
    train_matrices: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_save_tensor, a): a for a in all_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="encode+save", unit="traj"):
            row = fut.result()
            mat = row.pop("matrix")
            if mat is not None:
                train_matrices.append(mat)
            rows.append(row)

    if norm_from:
        means = json.loads((Path(norm_from) / "static_means.json").read_text())
    else:
        means = compute_static_means(train_matrices)
    (processed / "static_means.json").write_text(json.dumps(means, indent=2))

    index = pd.DataFrame(rows)
    index.to_parquet(processed / "index.parquet", index=False)
    return processed
