"""Normalization + encoding of AIS attributes (paper Section 3, Step 4).

Extraction persists each trajectory as a raw normalized ``[T, 7]`` float matrix
(columns in SEVEN_ATTRS order; NaN kept for missing static fields). Encoding to
seven-hot — or to raw real values for baselines, with optional missing-static
fill — happens at load time so all experiments (ablations, missing-static,
raw-vs-seven-hot) are config-only and need no re-extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SEVEN_ATTRS, EncodingConfig

# 3-letter attr name (config/paper) -> dataframe column name (from ingest).
ATTR_TO_COL = {
    "LAT": "LAT", "LON": "LON", "SOG": "SOG", "COG": "COG",
    "WID": "Width", "LEN": "Length", "DRA": "Draft",
}
STATIC_ATTRS = ("WID", "LEN", "DRA")
_ATTR_INDEX = {a: i for i, a in enumerate(SEVEN_ATTRS)}


def compute_norm_stats(df, attrs: tuple[str, ...] = SEVEN_ATTRS) -> dict[str, dict[str, float]]:
    """Min/max per attribute, computed on the given (train) frame."""
    stats: dict[str, dict[str, float]] = {}
    for attr in attrs:
        col = ATTR_TO_COL[attr]
        series = df[col].astype("float64").dropna()
        lo = float(series.min()) if len(series) else 0.0
        hi = float(series.max()) if len(series) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        stats[attr] = {"min": lo, "max": hi}
    return stats


def save_norm_stats(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load_norm_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (values.astype("float64") - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def normalize_df_to_matrix(df, stats: dict[str, dict[str, float]]) -> np.ndarray:
    """Trajectory df -> raw normalized [T, 7] matrix (SEVEN_ATTRS order).

    Values clipped to [0,1]; NaN preserved where the source attribute is missing
    (e.g. absent static fields).
    """
    cols = []
    for attr in SEVEN_ATTRS:
        raw = df[ATTR_TO_COL[attr]].to_numpy(dtype="float64")
        norm = _normalize(raw, stats[attr]["min"], stats[attr]["max"])
        norm = np.where(np.isnan(raw), np.nan, norm)
        cols.append(norm)
    return np.stack(cols, axis=1).astype("float32")


def _one_hot_bins(norm: np.ndarray, n_bins: int) -> np.ndarray:
    """norm in [0,1] -> (len, n_bins) one-hot. NaN -> all-zero row."""
    out = np.zeros((len(norm), n_bins), dtype="float32")
    valid = ~np.isnan(norm)
    rows = np.nonzero(valid)[0]
    idx = np.clip(np.floor(norm[rows] * n_bins).astype("int64"), 0, n_bins - 1)
    out[rows, idx] = 1.0
    return out


def _filled(matrix: np.ndarray, cfg: EncodingConfig, fill, means) -> np.ndarray:
    """Apply missing-static fill to a copy of the raw [T,7] matrix."""
    if fill is None:
        return matrix
    out = matrix.copy()
    for attr in STATIC_ATTRS:
        j = _ATTR_INDEX[attr]
        col = out[:, j]
        nan = np.isnan(col)
        if not nan.any():
            continue
        if fill == "zero":
            col[nan] = 0.0
        elif fill == "mean" and means is not None:
            col[nan] = means.get(attr, 0.0)
        out[:, j] = col
    return out


def seven_hot_from_matrix(
    matrix: np.ndarray,
    cfg: EncodingConfig,
    *,
    missing_static_fill: str | None = None,
    static_means: dict[str, float] | None = None,
) -> np.ndarray:
    """Raw [T,7] -> seven-hot [T, D] over the config's active attributes."""
    matrix = _filled(matrix, cfg, missing_static_fill, static_means)
    blocks = []
    for attr in cfg.active_attrs():
        col = matrix[:, _ATTR_INDEX[attr]]
        blocks.append(_one_hot_bins(col, cfg.bins[attr]))
    return np.concatenate(blocks, axis=1).astype("float32")


def raw_from_matrix(
    matrix: np.ndarray,
    cfg: EncodingConfig,
    *,
    missing_static_fill: str | None = "zero",
    static_means: dict[str, float] | None = None,
) -> np.ndarray:
    """Raw [T,7] -> [T, n_active] real-valued features (NaN filled). Baseline input."""
    matrix = _filled(matrix, cfg, missing_static_fill, static_means)
    cols = [matrix[:, _ATTR_INDEX[a]] for a in cfg.active_attrs()]
    out = np.stack(cols, axis=1).astype("float32")
    return np.nan_to_num(out, nan=0.0)


def compute_static_means(matrices: list[np.ndarray]) -> dict[str, float]:
    """Normalized-space mean of each static attr over available (non-NaN) values."""
    means: dict[str, float] = {}
    for attr in STATIC_ATTRS:
        j = _ATTR_INDEX[attr]
        vals = np.concatenate([m[:, j] for m in matrices]) if matrices else np.array([])
        vals = vals[~np.isnan(vals)]
        means[attr] = float(vals.mean()) if len(vals) else 0.0
    return means


def seven_hot_decode(matrix: np.ndarray, cfg: EncodingConfig) -> dict[str, np.ndarray]:
    """Inverse: per-attribute bin index from a seven-hot [T, D] matrix (for tests)."""
    out: dict[str, np.ndarray] = {}
    offset = 0
    for attr in cfg.active_attrs():
        n = cfg.bins[attr]
        out[attr] = matrix[:, offset:offset + n].argmax(axis=1)
        offset += n
    return out


# Backward-compatible df-based helper (used by tests).
def seven_hot_encode(df, stats, cfg: EncodingConfig, **kwargs) -> np.ndarray:
    return seven_hot_from_matrix(normalize_df_to_matrix(df, stats), cfg, **kwargs)
