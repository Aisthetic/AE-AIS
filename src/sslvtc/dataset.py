"""Dataset + labeled/unlabeled split for SSL-VTC training.

Stored tensors are raw normalized ``[T, 7]`` matrices; this dataset applies the
chosen encoding (seven-hot or raw real values) and missing-static fill at load
time, so ablations / missing-static experiments need no re-extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import SEVEN_ATTRS, EncodingConfig
from .encoding import STATIC_ATTRS, raw_from_matrix, raw_dt_from_matrix, seven_hot_from_matrix

_STATIC_COLS = [SEVEN_ATTRS.index(a) for a in STATIC_ATTRS]


class TrajectoryDataset(Dataset):
    """Returns (x[1,T,W] float32, label_idx int64) for one split.

    mode: "sevenhot" (W=D) or "raw" (W=n_active_attrs).
    missing_static_fill: None | "zero" | "mean".
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        encoding: EncodingConfig | None = None,
        *,
        mode: str = "sevenhot",
        missing_static_fill: str | None = None,
        static_available_fraction: float = 1.0,
        withhold_seed: int = 0,
        indices: np.ndarray | None = None,
        split_column: str = "split",
        return_mmsi: bool = False,
        preload: bool = True,
    ):
        self.root = Path(processed_dir)
        self.encoding = encoding or EncodingConfig()
        self.mode = mode
        self.missing_static_fill = missing_static_fill
        self.return_mmsi = return_mmsi
        self.preload = preload
        means_path = self.root / "static_means.json"
        self.static_means = json.loads(means_path.read_text()) if means_path.exists() else None

        index = pd.read_parquet(self.root / "index.parquet")
        index = index[index[split_column] == split].reset_index(drop=True)
        if indices is not None:
            index = index.iloc[indices].reset_index(drop=True)
        self.index = index

        # Preload all needed tensors via a single packed memmap, eliminating
        # per-sample .npy reads (the per-epoch IO bottleneck on shared storage).
        # The pack is built once over the full master index and memmapped after.
        self._data = None
        self._positions = None
        if preload:
            packed, id_map = self._ensure_packed()
            self._data = packed  # memmap [N_all, T, C], read-only
            self._positions = np.array(
                [id_map[t] for t in self.index["traj_id"].tolist()], dtype="int64"
            )

        # Deterministically withhold static info for (1 - fraction) of trajectories
        # (Table 7). Withheld rows get static columns set to NaN before fill.
        n = len(self.index)
        self._withheld = np.zeros(n, dtype=bool)
        if static_available_fraction < 1.0 and n:
            rng = np.random.default_rng(withhold_seed)
            order = rng.permutation(n)
            keep = max(0, int(round(static_available_fraction * n)))
            self._withheld[order[keep:]] = True

    def _ensure_packed(self):
        """Build (once) and memmap a single packed tensor array for ALL trajectories.

        Returns (memmap[N_all, T, C] float32, {traj_id: row_position}). Packing reads
        every .npy once; subsequent runs just memmap the cache (page-cache fast).
        """
        packed_path = self.root / "packed_all.npy"
        ids_path = self.root / "packed_all_ids.json"
        if not (packed_path.exists() and ids_path.exists()):
            master = pd.read_parquet(self.root / "index.parquet")
            paths = master["path"].tolist()
            ids = master["traj_id"].tolist()
            first = np.load(self.root / paths[0]).astype("float32")
            T, C = first.shape
            arr = np.empty((len(paths), T, C), dtype="float32")
            for k, p in enumerate(paths):
                arr[k] = np.load(self.root / p).astype("float32")
            np.save(packed_path, arr)
            ids_path.write_text(json.dumps({tid: k for k, tid in enumerate(ids)}))
            del arr
        id_map = json.loads(ids_path.read_text())
        data = np.load(packed_path, mmap_mode="r")
        return data, id_map

    def _matrix(self, i: int) -> np.ndarray:
        """Return a writable [T, C] float32 matrix for sample i (RAM or disk)."""
        if self._data is not None:
            return np.array(self._data[self._positions[i]], dtype="float32")  # copy: writable
        return np.load(self.root / self.index.iloc[i]["path"]).astype("float32")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> np.ndarray:
        return self.index["label_idx"].to_numpy()

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        if self.mode == "raw":
            return raw_from_matrix(
                matrix, self.encoding,
                missing_static_fill=self.missing_static_fill or "zero",
                static_means=self.static_means,
            )
        if self.mode == "raw_dt":
            return raw_dt_from_matrix(
                matrix, self.encoding,
                missing_static_fill=self.missing_static_fill or "zero",
                static_means=self.static_means,
            )
        return seven_hot_from_matrix(
            matrix, self.encoding,
            missing_static_fill=self.missing_static_fill,
            static_means=self.static_means,
        )

    def shape(self) -> tuple[int, int]:
        sample = self._encode(self._matrix(0))
        return int(sample.shape[0]), int(sample.shape[1])

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        matrix = self._matrix(i)  # writable [T, C] from RAM pack (or disk fallback)
        if self._withheld[i]:
            matrix[:, _STATIC_COLS] = np.nan  # fill step (zero/mean) handles it
        x = torch.from_numpy(self._encode(matrix)).unsqueeze(0)  # [1, T, W]
        label = int(row["label_idx"])
        if self.return_mmsi:
            mmsi = int(row["mmsi"])
            return x, label, mmsi
        return x, label


def stratified_labeled_mask(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask marking `fraction` of samples as labeled, stratified by class."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        idx = np.nonzero(labels == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(round(fraction * len(idx))))
        mask[idx[:k]] = True
    return mask
