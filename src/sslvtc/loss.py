"""SSL-VTC loss — Kingma M2 semi-supervised VAE (paper Section 4.3).

L = L1 + L2 + alpha * L_clf,  alpha = beta * (n2 / n1)
  L1   : -ELBO over labeled data
  L2   : -ELBO over unlabeled data (marginalized over labels + classifier entropy)
  L_clf: cross-entropy on labeled data
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import SSLVTC, reparameterize

_LOG_2PI = 1.8378770664093453


def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(y, n_classes).float()


def _recon_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy reconstruction (Bernoulli p(x|y,z)); sum over features, per-sample."""
    bce = F.binary_cross_entropy(x_hat, x, reduction="none")
    return bce.flatten(1).sum(dim=1)


def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x,y) || N(0,I)) per-sample."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _neg_elbo(model: SSLVTC, x: torch.Tensor, y_onehot: torch.Tensor, n_classes: int) -> torch.Tensor:
    """-ELBO J(x,y) per sample (paper Eq. 4): recon + KL - log p(y).

    log p(y) is a uniform categorical prior: log(1/n_classes), constant.
    """
    mu, logvar = model.encoder(x, y_onehot)
    z = reparameterize(mu, logvar)
    x_hat = model.decoder(z, y_onehot)
    recon = _recon_loss(x_hat, x)
    kl = _kl_standard_normal(mu, logvar)
    log_py = torch.log(torch.tensor(1.0 / n_classes, device=x.device))
    return recon + kl - log_py  # higher = worse


def labeled_loss(model: SSLVTC, x: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """L1: sum of -ELBO over the labeled batch (paper Eq. 5)."""
    y_onehot = _one_hot(y, n_classes)
    return _neg_elbo(model, x, y_onehot, n_classes).sum()


def unlabeled_loss(model: SSLVTC, x: torch.Tensor, n_classes: int) -> torch.Tensor:
    """L2: marginalize -ELBO over all labels weighted by q(y|x), minus entropy (Eq. 8-9)."""
    logits = model.classifier(x)
    q_y = F.softmax(logits, dim=1)            # [B, C]
    log_q_y = F.log_softmax(logits, dim=1)

    # weighted -ELBO over every possible label (run encoder/decoder once per class)
    weighted = torch.zeros(x.size(0), device=x.device)
    for c in range(n_classes):
        y_c = torch.full((x.size(0),), c, dtype=torch.long, device=x.device)
        y_onehot = _one_hot(y_c, n_classes)
        j = _neg_elbo(model, x, y_onehot, n_classes)   # [B]
        weighted = weighted + q_y[:, c] * j

    entropy = -torch.sum(q_y * log_q_y, dim=1)          # H(q(y|x)) >= 0
    # U(x) = sum_y q(y|x) J(x,y) - H(q(y|x)); summed over batch
    return (weighted - entropy).sum()


def classifier_loss(model: SSLVTC, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L_clf: mean cross-entropy on the labeled batch (paper Eq. 10)."""
    logits = model.classifier(x)
    return F.cross_entropy(logits, y, reduction="mean")


def consistency_loss(
    model: SSLVTC,
    x_unlab: torch.Tensor,
    *,
    threshold: float = 0.95,
    augment_fn=None,
    mode: str = "sevenhot",
) -> torch.Tensor:
    """FixMatch-style consistency: pseudo-labels from weak aug, CE on strong aug.

    Returns scalar loss (0 if no confident pseudo-labels above threshold).
    """
    from .augmentations import weak_augment, strong_augment
    aug_fn = augment_fn or (lambda x: strong_augment(x, mode=mode))

    with torch.no_grad():
        weak_x = weak_augment(x_unlab, mode=mode)
        logits_weak = model.classifier(weak_x)
        probs = F.softmax(logits_weak, dim=1)
        max_prob, pseudo_labels = probs.max(dim=1)
        mask = max_prob >= threshold

    if mask.sum() == 0:
        return torch.zeros(1, device=x_unlab.device).squeeze()

    strong_x = aug_fn(x_unlab[mask])
    logits_strong = model.classifier(strong_x)
    return F.cross_entropy(logits_strong, pseudo_labels[mask], reduction="mean")


def total_loss(
    model: SSLVTC,
    x_lab: torch.Tensor,
    y_lab: torch.Tensor,
    x_unlab: torch.Tensor | None,
    n_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss L = L1 + L2 + alpha*L_clf. Per-sample normalized for stable scale."""
    n_lab = max(x_lab.size(0), 1)
    l1 = labeled_loss(model, x_lab, y_lab, n_classes) / n_lab
    l_clf = classifier_loss(model, x_lab, y_lab)
    if x_unlab is not None and x_unlab.size(0) > 0:
        l2 = unlabeled_loss(model, x_unlab, n_classes) / x_unlab.size(0)
    else:
        l2 = torch.zeros((), device=x_lab.device)
    loss = l1 + l2 + alpha * l_clf
    parts = {
        "l1": l1.detach().item(),
        "l2": l2.detach().item(),
        "l_clf": l_clf.detach().item(),
        "total": loss.detach().item(),
    }
    return loss, parts
