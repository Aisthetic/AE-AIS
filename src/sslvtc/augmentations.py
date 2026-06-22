"""AIS-specific trajectory augmentations for consistency SSL (Phase 2.2).

All augmentations operate on raw [T, d] tensors (float32, values in [0,1]).
They are applied *after* encoding so they work independently of backbone.

Augmentation pair:
  weak  → small jitter only
  strong → jitter + crop/pad + time-warp + static dropout
"""
from __future__ import annotations

import numpy as np
import torch


def gaussian_jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to all channels. x: [B, 1, T, W] or [B, T, W]."""
    return x + sigma * torch.randn_like(x)


def kinematic_jitter(
    x: torch.Tensor,
    sigma: float = 0.02,
    kinematic_cols: list[int] | None = None,
) -> torch.Tensor:
    """Jitter only kinematic columns (LAT=0, LON=1, SOG=2, COG=3).
    x: [B, 1, T, W]
    """
    if kinematic_cols is None:
        kinematic_cols = [0, 1, 2, 3]
    out = x.clone()
    noise = sigma * torch.randn(x.size(0), 1, x.size(2), len(kinematic_cols), device=x.device)
    out[:, :, :, kinematic_cols] = (out[:, :, :, kinematic_cols] + noise).clamp(0.0, 1.0)
    return out


def segment_crop(x: torch.Tensor, min_keep: float = 0.7) -> torch.Tensor:
    """Randomly crop a contiguous segment and pad to original length with boundary values.
    x: [B, 1, T, W]
    """
    T = x.size(2)
    keep = int(T * (min_keep + (1.0 - min_keep) * torch.rand(1).item()))
    keep = max(keep, 1)
    start = torch.randint(0, T - keep + 1, (1,)).item()
    cropped = x[:, :, start:start + keep, :]
    # Pad to T by repeating boundary values
    pad_left = start
    pad_right = T - keep - pad_left
    out = torch.cat([
        cropped[:, :, :1, :].expand(-1, -1, pad_left, -1),
        cropped,
        cropped[:, :, -1:, :].expand(-1, -1, pad_right, -1),
    ], dim=2)
    return out


def time_warp(x: torch.Tensor, max_warp: float = 0.1) -> torch.Tensor:
    """Randomly stretch/compress time axis via 1D grid_sample interpolation.
    x: [B, 1, T, W]
    """
    B, C, T, W = x.shape
    # Random warp: monotonically increasing t-coords with slight jitter
    base = torch.linspace(-1.0, 1.0, T, device=x.device)
    jitter = (torch.rand(T, device=x.device) * 2 - 1) * max_warp
    coords_t = (base + jitter).clamp(-1.0, 1.0)
    # Sort to keep monotonicity
    coords_t, _ = coords_t.sort()
    coords_w = torch.linspace(-1.0, 1.0, W, device=x.device)
    grid_t = coords_t.unsqueeze(1).expand(T, W)  # [T, W]
    grid_w = coords_w.unsqueeze(0).expand(T, W)  # [T, W]
    grid = torch.stack([grid_w, grid_t], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, W, 2]
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")


def static_feature_dropout(x: torch.Tensor, prob: float = 0.5,
                            static_cols: list[int] | None = None) -> torch.Tensor:
    """Randomly zero static feature columns per sample. x: [B, 1, T, W]."""
    if static_cols is None:
        static_cols = [4, 5, 6]  # WID, LEN, DRA in raw/raw_dt layout
    if prob <= 0.0:
        return x
    out = x.clone()
    mask = torch.rand(x.size(0), device=x.device) < prob
    if mask.any():
        for col in static_cols:
            if col < x.size(-1):
                out[mask, :, :, col] = 0.0
    return out


def weak_augment(x: torch.Tensor, mode: str = "sevenhot") -> torch.Tensor:
    """Weak augmentation: small Gaussian jitter only."""
    if mode == "sevenhot":
        # Seven-hot is binary; jitter is less meaningful but still used for consistency
        return gaussian_jitter(x, sigma=0.005)
    return kinematic_jitter(x, sigma=0.01)


def strong_augment(
    x: torch.Tensor,
    mode: str = "sevenhot",
    static_drop_prob: float = 0.5,
) -> torch.Tensor:
    """Strong augmentation: jitter + crop + time-warp + static dropout."""
    if mode == "sevenhot":
        x = gaussian_jitter(x, sigma=0.02)
    else:
        x = kinematic_jitter(x, sigma=0.03)
        x = segment_crop(x, min_keep=0.75)
        x = time_warp(x, max_warp=0.08)
        x = static_feature_dropout(x, prob=static_drop_prob)
    return x
