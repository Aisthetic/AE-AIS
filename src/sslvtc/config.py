"""Frozen-dataclass configuration + YAML loader for the SSL-VTC pipeline.

Mirrors the config style of the sibling GenAIS project but is reimplemented
standalone. Load with ``load_config(path)``; every stage reads its slice.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# The 7 attributes used for seven-hot encoding (paper order).
SEVEN_ATTRS = ("LAT", "LON", "SOG", "COG", "WID", "LEN", "DRA")


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box (degrees). None disables a side."""
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None

    def contains(self, lat: float, lon: float) -> bool:  # pragma: no cover - vectorized in ingest
        if self.lat_min is not None and lat < self.lat_min:
            return False
        if self.lat_max is not None and lat > self.lat_max:
            return False
        if self.lon_min is not None and lon < self.lon_min:
            return False
        if self.lon_max is not None and lon > self.lon_max:
            return False
        return True


@dataclass(frozen=True)
class Paths:
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    results: str = "results"


@dataclass(frozen=True)
class DownloadConfig:
    year: int = 2019
    months: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    url_template: str = (
        "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
    )
    # None = every day in each month (full reproduction). Set an int to cap days
    # per month (sampled evenly) for a lighter run; or list explicit days.
    max_days_per_month: int | None = None
    days: tuple[int, ...] | None = None


@dataclass(frozen=True)
class IngestConfig:
    chunk_size: int = 100_000
    max_sog_knots: float = 60.0
    # AIS VesselType code -> class. fishing=30; passenger 60-69; cargo 70-79; tanker 80-89.
    train_months: tuple[int, ...] = (1, 2, 3, 4)
    val_months: tuple[int, ...] = (5,)
    test_months: tuple[int, ...] = (6,)


@dataclass(frozen=True)
class ExtractionConfig:
    """Paper Section 3 thresholds. Do not relax without noting it."""
    max_gap_hours: float = 2.0          # Step 1: split when adjacent gap > 2h
    min_span_hours: float = 6.0         # Step 2: drop trajectories shorter than 6h
    min_messages: int = 160             # Step 2: drop trajectories with < 160 messages
    abnormal_max_sog: float = 1.0       # Step 3: drop if max SOG <= 1 kn
    moving_sog_threshold: float = 2.0   # Step 3: "moving" defined as SOG > 2 kn
    min_moving_fraction: float = 0.30   # Step 3: drop if moving fraction <= 30%
    fixed_length: int = 160             # Step 5: subsample to this many messages


@dataclass(frozen=True)
class EncodingConfig:
    """Per-attribute bin counts for seven-hot encoding.

    Paper omits the resolution; defaults follow a Nguyen/TrAISformer-style scheme
    (fine for position, coarse for static). All tunable. D = sum(bins).
    """
    bins: dict[str, int] = field(
        default_factory=lambda: {
            "LAT": 50, "LON": 50, "SOG": 30, "COG": 36, "WID": 10, "LEN": 20, "DRA": 10,
        }
    )
    # Static attributes; can be dropped for ablations (Table 3).
    use_len: bool = True
    use_wid: bool = True
    use_dra: bool = True

    def active_attrs(self) -> tuple[str, ...]:
        skip = set()
        if not self.use_len:
            skip.add("LEN")
        if not self.use_wid:
            skip.add("WID")
        if not self.use_dra:
            skip.add("DRA")
        return tuple(a for a in SEVEN_ATTRS if a not in skip)

    def total_dim(self) -> int:
        return sum(self.bins[a] for a in self.active_attrs())


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int = 4
    conv_channels: tuple[int, ...] = (5, 5, 5, 5, 5)   # output channels per conv layer
    conv_kernels: tuple[int, ...] = (10, 10, 10, 5, 3)
    label_embed_dim: int = 50
    latent_dim: int = 20
    decoder_fc_dim: int = 250  # paper reshape target


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 100
    epochs: int = 50
    labeled_fraction: float = 0.20     # fraction of train set treated as labeled
    beta: float = 10.0                 # SSL weight; grid: {5%:10,20%:10,40%:100,60%:1000}
    seed: int = 42
    device: str = "auto"               # auto -> mps/cuda/cpu
    num_workers: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    bbox: BBox = field(default_factory=BBox)
    paths: Paths = field(default_factory=Paths)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct a (possibly nested) frozen dataclass from a dict."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # Resolve string annotations (PEP 563) to real types so nested dataclasses
    # are detected.
    type_hints = typing.get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in field_names:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = type_hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        elif isinstance(value, dict):
            kwargs[key] = value  # plain mapping field (e.g. encoding.bins)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a PipelineConfig from YAML; missing/None path returns defaults."""
    if path is None:
        return PipelineConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build(PipelineConfig, raw)
