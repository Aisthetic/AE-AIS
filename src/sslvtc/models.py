"""CNN modules for SSL-VTC (paper Section 4.4): Classifier, Encoder, Decoder.

Trajectory is a single-channel image [B, 1, T, D]. The conv stack uses the
paper's channels (1->5->5->5->5) and kernel sizes (10,10,10,5,3). The flattened
feature size is computed dynamically at build time (paper reports 250) so the
model adapts to the chosen T_fixed / D rather than hard-coding it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    """5 Conv2d layers, each followed by ReLU. Padding keeps it robust to small
    inputs (kernel up to 10). Stride 1; 'same'-style padding per layer."""

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
        self.flat = _flatten_size(self.conv, t, d)
        self.fc = nn.Linear(self.flat, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)  # logits; softmax applied in loss/inference


class Encoder(nn.Module):
    """q(z|x,y): conv features of x concatenated with a label embedding -> mu, logvar."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.conv = ConvBlock(cfg.conv_kernels, cfg.conv_channels)
        self.flat = _flatten_size(self.conv, t, d)
        self.label_fc = nn.Linear(cfg.n_classes, cfg.label_embed_dim)
        joint = self.flat + cfg.label_embed_dim
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
    """p(x|y,z): [z, y] -> FC -> reshape -> 5 ConvTranspose2d -> sigmoid -> x_hat.

    The conv-transpose stack mirrors the encoder kernels (3,5,10,10,10). To
    guarantee the reconstruction matches the input (T, D) exactly, the final
    output is bilinearly resized to (T, D); padding choices otherwise make exact
    deconv geometry brittle across arbitrary T/D.
    """

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.t, self.d = t, d
        self.base_ch = cfg.conv_channels[-1]
        # small spatial seed reshaped from FC, then upsample via deconv + final resize
        self.seed_t = max(t // 8, 1)
        self.seed_d = max(d // 8, 1)
        fc_out = self.base_ch * self.seed_t * self.seed_d
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_classes, cfg.decoder_fc_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder_fc_dim, fc_out),
            nn.ReLU(),
        )
        deconv_kernels = tuple(reversed(cfg.conv_kernels))   # (3,5,10,10,10)
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
        h = torch.nn.functional.interpolate(h, size=(self.t, self.d), mode="bilinear", align_corners=False)
        return torch.sigmoid(h)  # [B, 1, T, D] in (0,1)


class SSLVTC(nn.Module):
    """Bundle of classifier + encoder + decoder, jointly trained."""

    def __init__(self, cfg: ModelConfig, t: int, d: int):
        super().__init__()
        self.cfg = cfg
        self.classifier = Classifier(cfg, t, d)
        self.encoder = Encoder(cfg, t, d)
        self.decoder = Decoder(cfg, t, d)
