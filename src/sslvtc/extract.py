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
    return True


def _sample_fixed(traj: pd.DataFrame, length: int) -> pd.DataFrame:
    """Step 5: subsample to `length` messages via evenly spaced indices."""
    n = len(traj)
    if n == length:
        return traj
    idx = np.linspace(0, n - 1, num=length).round().astype("int64")
    return traj.iloc[idx]


def extract_split(df: pd.DataFrame, cfg: PipelineConfig) -> list[pd.DataFrame]:
    """Run Steps 1-3 + 5 (sampling) for one split; returns fixed-length trajectories."""
    trajectories: list[pd.DataFrame] = []
    for _, group in df.groupby("MMSI"):
        for seg in _divide(group, cfg.extraction.max_gap_hours):
            if _passes_filter(seg, cfg):
                trajectories.append(_sample_fixed(seg, cfg.extraction.fixed_length))
    return trajectories


def extract_all(cfg: PipelineConfig) -> Path:
    """Full extraction across train/val/test; writes tensors + index parquet.

    Normalization stats are computed on the TRAIN split only and reused.
    """
    interim = Path(cfg.paths.interim)
    processed = Path(cfg.paths.processed)
    tensor_dir = processed / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    # First pass: extract raw fixed-length trajectories per split (kept in memory
    # as DataFrames; for the bounded-bbox subset this is tractable).
    per_split: dict[str, list[pd.DataFrame]] = {}
    for split in ("train", "val", "test"):
        path = interim / f"clean_{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        per_split[split] = extract_split(df, cfg)

    if "train" not in per_split or not per_split["train"]:
        raise RuntimeError("no train trajectories extracted; check bbox/thresholds/data")

    # Norm stats from train messages only.
    train_concat = pd.concat(per_split["train"], ignore_index=True)
    stats = compute_norm_stats(train_concat, SEVEN_ATTRS)
    save_norm_stats(stats, processed / "normalization_stats.json")

    # Persist raw normalized [T, 7] matrices; encoding happens at load time.
    rows = []
    train_matrices: list[np.ndarray] = []
    for split, trajs in per_split.items():
        for i, traj in enumerate(tqdm(trajs, desc=f"encode/{split}", unit="traj")):
            matrix = normalize_df_to_matrix(traj, stats)  # (T, 7) normalized, NaN kept
            if split == "train":
                train_matrices.append(matrix)
            traj_id = f"{split}_{i:06d}"
            rel = f"tensors/{traj_id}.npy"
            np.save(processed / rel, matrix)
            rows.append({
                "traj_id": traj_id,
                "split": split,
                "label": traj["label"].iloc[0],
                "label_idx": int(traj["label_idx"].iloc[0]),
                "mmsi": int(traj["MMSI"].iloc[0]),
                "path": rel,
            })

    # Static means (normalized space, train only) for missing-static "mean" fill.
    means = compute_static_means(train_matrices)
    (processed / "static_means.json").write_text(json.dumps(means, indent=2))

    index = pd.DataFrame(rows)
    index.to_parquet(processed / "index.parquet", index=False)
    return processed
