"""Post-hoc resplit utilities operating on index.parquet.

Produces an additional split column ('split_vd') for vessel-disjoint evaluation
without re-extracting tensors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_vessel_disjoint_split(
    processed_dir: str | Path,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    split_column: str = "split_vd",
) -> pd.DataFrame:
    """Assign each unique MMSI to train/val/test disjointly, stratified by vessel class.

    Reads processed_dir/index.parquet, adds `split_column`, writes
    processed_dir/index.parquet in-place, and returns the updated DataFrame.

    Class of a vessel = modal label_idx across its trajectories (vessels have one class).
    Train fraction of total trajectories is approximately `train_frac`; val/test split
    the remainder as val_frac / (1 - train_frac).
    """
    processed = Path(processed_dir)
    index = pd.read_parquet(processed / "index.parquet")

    # Assign each MMSI a class (mode of label_idx)
    vessel_class = (
        index.groupby("mmsi")["label_idx"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"label_idx": "vessel_class"})
    )

    rng = np.random.default_rng(seed)
    test_frac = 1.0 - train_frac - val_frac

    train_mmsi, val_mmsi, test_mmsi = set(), set(), set()

    for cls, grp in vessel_class.groupby("vessel_class"):
        mmsis = grp["mmsi"].to_numpy().copy()
        rng.shuffle(mmsis)
        n = len(mmsis)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        # Ensure at least 1 in test
        n_test = max(1, n - n_train - n_val)
        # Re-adjust n_val if sum exceeds n
        if n_train + n_val + n_test > n:
            n_val = n - n_train - n_test
            if n_val < 1:
                n_val = 1
                n_train = n - n_val - n_test
        train_mmsi.update(mmsis[:n_train].tolist())
        val_mmsi.update(mmsis[n_train:n_train + n_val].tolist())
        test_mmsi.update(mmsis[n_train + n_val:].tolist())

    def _assign(mmsi):
        if mmsi in train_mmsi:
            return "train"
        if mmsi in val_mmsi:
            return "val"
        return "test"

    index[split_column] = index["mmsi"].map(_assign)
    index.to_parquet(processed / "index.parquet", index=False)
    return index


def report_overlap(processed_dir: str | Path) -> dict:
    """Compute and print vessel-identity leakage stats for both split protocols."""
    processed = Path(processed_dir)
    index = pd.read_parquet(processed / "index.parquet")

    results = {}
    for split_col in ("split", "split_vd"):
        if split_col not in index.columns:
            continue
        train_mmsi = set(index.loc[index[split_col] == "train", "mmsi"].unique())
        val_mmsi = set(index.loc[index[split_col] == "val", "mmsi"].unique())
        test_mmsi = set(index.loc[index[split_col] == "test", "mmsi"].unique())

        val_overlap = val_mmsi & train_mmsi
        test_overlap = test_mmsi & train_mmsi

        test_traj_leaked = index[
            (index[split_col] == "test") & (index["mmsi"].isin(train_mmsi))
        ]

        n_test_traj = len(index[index[split_col] == "test"])
        n_val_traj = len(index[index[split_col] == "val"])

        results[split_col] = {
            "n_train_vessels": len(train_mmsi),
            "n_val_vessels": len(val_mmsi),
            "n_test_vessels": len(test_mmsi),
            "val_vessel_overlap_pct": 100 * len(val_overlap) / max(len(val_mmsi), 1),
            "test_vessel_overlap_pct": 100 * len(test_overlap) / max(len(test_mmsi), 1),
            "test_traj_leaked_pct": 100 * len(test_traj_leaked) / max(n_test_traj, 1),
            "n_train_traj": len(index[index[split_col] == "train"]),
            "n_val_traj": n_val_traj,
            "n_test_traj": n_test_traj,
        }

    return results
