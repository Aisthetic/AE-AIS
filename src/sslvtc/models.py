"""CNN and Transformer modules for SSL-VTC.

Default backbone: sevenhot_cnn — paper-exact ConvBlock architecture.
Phase 2.1 backbone: temporal_transformer — small transformer over raw [T, d_in]
  feature sequence with Δt-aware positional encoding.

SSLVTC bundles classifier + encoder + decoder and dispatches by cfg.backbone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------------------
# CNN backbone (paper-exact)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """5 Conv2d layers, each followed by ReLU. Padding keeps it robust to small inputs."""

    def __init__(self, kernels: tuple[int, ...], channels: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(channels, kernels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_size(block: ConvBlock, t: int, d: int) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, 1, t, d)
        return int(block(dummy).flatten(1).shape[1])


class Classifier(nn.Module):
    """q(y|x): ConvBlock -> flatten -> FC -> logits over n_classes."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.conv = ConvBlock(cfg.conv_kernels, cfg.conv_channels)
        self.feat_dim = _flatten_size(self.conv, t, d)
        self.fc = nn.Linear(self.feat_dim, cfg.n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(1)  # [B, feat_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))  # logits


class Encoder(nn.Module):
    """q(z|x,y): conv features of x concatenated with label embedding -> mu, logvar."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.conv = ConvBlock(cfg.conv_kernels, cfg.conv_channels)
        flat = _flatten_size(self.conv, t, d)
        self.label_fc = nn.Linear(cfg.n_classes, cfg.label_embed_dim)
        joint = flat + cfg.label_embed_dim
        self.fc_mu = nn.Linear(joint, cfg.latent_dim)
        self.fc_logvar = nn.Linear(joint, cfg.latent_dim)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        ly = torch.relu(self.label_fc(y_onehot))
        joint = torch.cat([h, ly], dim=1)
        return self.fc_mu(joint), self.fc_logvar(joint)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    """p(x|y,z): [z, y] -> FC -> reshape -> ConvTranspose2d stack -> sigmoid -> x_hat."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.t, self.d = t, d
        self.base_ch = cfg.conv_channels[-1]
        self.seed_t = max(t // 8, 1)
        self.seed_d = max(d // 8, 1)
        fc_out = self.base_ch * self.seed_t * self.seed_d
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_classes, cfg.decoder_fc_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder_fc_dim, fc_out),
            nn.ReLU(),
        )
        deconv_kernels = tuple(reversed(cfg.conv_kernels))
        deconv_channels = (5, 5, 5, 5, 1)
        layers: list[nn.Module] = []
        in_ch = self.base_ch
        for j, (out_ch, k) in enumerate(zip(deconv_channels, deconv_kernels)):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            if j < len(deconv_channels) - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, y_onehot], dim=1))
        h = h.view(-1, self.base_ch, self.seed_t, self.seed_d)
        h = self.deconv(h)
        h = F.interpolate(h, size=(self.t, self.d), mode="bilinear", align_corners=False)
        return torch.sigmoid(h)  # [B, 1, T, D]


# ---------------------------------------------------------------------------
# Temporal Transformer backbone (Phase 2.1)
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding added to the token sequence."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerBackbone(nn.Module):
    """Small transformer over raw per-message [T, d_in] feature sequence.

    Input:  [B, 1, T, d_in]  (channel dim = 1, like the CNN path)
    Output: [B, d_model]       CLS token embedding

    d_in should be n_active_attrs + 1 (Δt) for raw_dt mode, or n_active_attrs
    for raw mode. Both work: the input projection handles arbitrary d_in.
    """

    def __init__(self, cfg: ModelConfig, t: int, d_in: int):
        super().__init__()
        d = cfg.tf_d_model
        self.input_proj = nn.Linear(d_in, d)
        nn.init.xavier_uniform_(self.input_proj.weight)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pe = _SinusoidalPE(d, max_len=t + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.tf_nhead,
            dim_feedforward=d * 4,
            dropout=cfg.tf_dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        # enable_nested_tensor requires norm_first=False; suppress harmless warning
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.tf_nlayers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        self.feat_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T, d_in]
        x = x.squeeze(1)                          # [B, T, d_in]
        x = self.input_proj(x)                    # [B, T, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)            # [B, T+1, d_model]
        x = self.pe(x)
        x = self.transformer(x)                   # [B, T+1, d_model]
        return self.norm(x[:, 0])                 # CLS → [B, d_model]


class TFClassifier(nn.Module):
    """Transformer-based q(y|x)."""

    def __init__(self, cfg: ModelConfig, t: int, d_in: int):
        super().__init__()
        self.backbone = TemporalTransformerBackbone(cfg, t, d_in)
        self.feat_dim = self.backbone.feat_dim
        self.fc = nn.Linear(self.feat_dim, cfg.n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, feat_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))  # logits


class TFEncoder(nn.Module):
    """Transformer-based q(z|x,y)."""

    def __init__(self, cfg: ModelConfig, t: int, d_in: int):
        super().__init__()
        self.backbone = TemporalTransformerBackbone(cfg, t, d_in)
        fd = self.backbone.feat_dim
        self.label_fc = nn.Linear(cfg.n_classes, cfg.label_embed_dim)
        joint = fd + cfg.label_embed_dim
        self.fc_mu = nn.Linear(joint, cfg.latent_dim)
        self.fc_logvar = nn.Linear(joint, cfg.latent_dim)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)                                  # [B, d_model]
        ly = torch.relu(self.label_fc(y_onehot))
        joint = torch.cat([h, ly], dim=1)
        return self.fc_mu(joint), self.fc_logvar(joint)


class TFDecoder(nn.Module):
    """MLP decoder for transformer path: [z, y] -> raw [B, 1, T, d_in] in (0,1)."""

    def __init__(self, cfg: ModelConfig, t: int, d_in: int):
        super().__init__()
        self.t, self.d_in = t, d_in
        out_dim = t * d_in
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_classes, cfg.decoder_fc_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder_fc_dim, cfg.decoder_fc_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder_fc_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        out = self.fc(torch.cat([z, y_onehot], dim=1))        # [B, T*d_in]
        return out.view(-1, 1, self.t, self.d_in)             # [B, 1, T, d_in]


# ---------------------------------------------------------------------------
# Phase 2.3 — gradient reversal MMSI head
# ---------------------------------------------------------------------------

class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.alpha)


class MMSIAdversarialHead(nn.Module):
    """Predicts MMSI bucket from representation with reversed gradients.

    Encourages the shared feature extractor to be uninformative about vessel identity
    while retaining class-discriminative information (Phase 2.3 debiasing).
    """

    def __init__(self, feat_dim: int, n_buckets: int, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        hidden = max(feat_dim // 2, n_buckets)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_buckets),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(self.grl(feat))  # logits over buckets


# ---------------------------------------------------------------------------
# SSLVTC bundle
# ---------------------------------------------------------------------------

class SSLVTC(nn.Module):
    """Classifier + Encoder + Decoder jointly trained; dispatches by cfg.backbone."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == "temporal_transformer":
            self.classifier = TFClassifier(cfg, t, d)
            self.encoder = TFEncoder(cfg, t, d)
            self.decoder = TFDecoder(cfg, t, d)
        else:  # sevenhot_cnn (default, paper-exact)
            self.classifier = Classifier(cfg, t, d)
            self.encoder = Encoder(cfg, t, d)
            self.decoder = Decoder(cfg, t, d)

        # Gradient-reversal MMSI head (Phase 2.3)
        if cfg.n_mmsi_buckets > 0 and cfg.gr_weight > 0.0:
            self.mmsi_head: MMSIAdversarialHead | None = MMSIAdversarialHead(
                self.classifier.feat_dim, cfg.n_mmsi_buckets
            )
        else:
            self.mmsi_head = None
