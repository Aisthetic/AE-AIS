"""Unit + smoke tests for the SSL-VTC pipeline."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from sslvtc.config import PipelineConfig
from sslvtc.encoding import (
    compute_norm_stats,
    seven_hot_decode,
    seven_hot_encode,
)
from sslvtc.extract import _divide, _passes_filter, _sample_fixed, extract_split
from sslvtc.ingest import vessel_type_to_label
from sslvtc.models import SSLVTC
from sslvtc.loss import total_loss


# ----- ingest: ship-type mapping -----
@pytest.mark.parametrize("code,expected", [
    (30, "fishing"), (60, "passenger"), (69, "passenger"),
    (70, "cargo"), (79, "cargo"), (80, "tanker"), (89, "tanker"),
    (31, None), (50, None), (90, None), (float("nan"), None),
])
def test_vessel_type_mapping(code, expected):
    assert vessel_type_to_label(code) == expected


# ----- helpers to build synthetic AIS trajectories -----
def _make_traj(n=200, span_hours=8.0, base_sog=5.0, label="cargo", label_idx=2, mmsi=1):
    start = pd.Timestamp("2019-01-01 00:00:00")
    times = pd.to_datetime(np.linspace(0, span_hours * 3600, n), unit="s", origin=start)
    return pd.DataFrame({
        "MMSI": mmsi,
        "BaseDateTime": times,
        "LAT": np.linspace(40.5, 40.6, n),
        "LON": np.linspace(-74.0, -73.9, n),
        "SOG": np.full(n, base_sog),
        "COG": np.linspace(0, 180, n),
        "Length": 100.0, "Width": 20.0, "Draft": 5.0,
        "label": label, "label_idx": label_idx,
    })


def test_filter_rejects_short_and_stationary():
    cfg = PipelineConfig()
    assert _passes_filter(_make_traj(n=200, span_hours=8), cfg)
    # too few messages
    assert not _passes_filter(_make_traj(n=100, span_hours=8), cfg)
    # too short span
    assert not _passes_filter(_make_traj(n=200, span_hours=3), cfg)
    # stationary (max SOG <= 1)
    assert not _passes_filter(_make_traj(n=200, span_hours=8, base_sog=0.5), cfg)


def test_divide_splits_on_gap():
    cfg = PipelineConfig()
    a = _make_traj(n=200, span_hours=8)
    b = _make_traj(n=200, span_hours=8)
    b["BaseDateTime"] = b["BaseDateTime"] + pd.Timedelta(hours=11)  # >2h gap, same day-ish
    combined = pd.concat([a, b], ignore_index=True)
    segs = _divide(combined, cfg.extraction.max_gap_hours)
    assert len(segs) >= 2


def test_sample_fixed_length():
    traj = _make_traj(n=237)
    out = _sample_fixed(traj, 160)
    assert len(out) == 160


def test_seven_hot_roundtrip():
    cfg = PipelineConfig()
    traj = _make_traj(n=160)
    attrs = ("LAT", "LON", "SOG", "COG", "WID", "LEN", "DRA")
    stats = compute_norm_stats(traj, attrs)
    mat = seven_hot_encode(traj, stats, cfg.encoding)
    assert mat.shape == (160, cfg.encoding.total_dim())
    # each attribute block is one-hot per row -> row sum == n_active_attrs
    assert np.allclose(mat.sum(axis=1), len(cfg.encoding.active_attrs()))
    decoded = seven_hot_decode(mat, cfg.encoding)
    assert set(decoded) == set(cfg.encoding.active_attrs())


def test_extract_split_end_to_end():
    cfg = PipelineConfig()
    df = pd.concat([_make_traj(n=200, span_hours=8, mmsi=m) for m in range(3)], ignore_index=True)
    trajs = extract_split(df, cfg)
    assert len(trajs) == 3
    assert all(len(t) == cfg.extraction.fixed_length for t in trajs)


# ----- model + loss -----
def test_model_shapes_and_loss():
    cfg = PipelineConfig()
    t, d = 160, cfg.encoding.total_dim()
    model = SSLVTC(cfg.model, t, d)
    x = torch.rand(4, 1, t, d)
    y = torch.tensor([0, 1, 2, 3])

    logits = model.classifier(x)
    assert logits.shape == (4, cfg.model.n_classes)

    mu, logvar = model.encoder(x, torch.nn.functional.one_hot(y, 4).float())
    assert mu.shape == (4, cfg.model.latent_dim)

    x_hat = model.decoder(mu, torch.nn.functional.one_hot(y, 4).float())
    assert x_hat.shape == x.shape
    assert torch.all((x_hat >= 0) & (x_hat <= 1))

    loss, parts = total_loss(model, x, y, x, cfg.model.n_classes, alpha=1.0)
    assert torch.isfinite(loss)
    assert set(parts) == {"l1", "l2", "l_clf", "total"}


def test_smaller_input_flatten_adapts():
    """Conv stack must handle inputs smaller than the paper's via padding."""
    cfg = PipelineConfig()
    small = dataclasses.replace(cfg.model)
    model = SSLVTC(small, t=40, d=60)
    x = torch.rand(2, 1, 40, 60)
    assert model.classifier(x).shape == (2, 4)
