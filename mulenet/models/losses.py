from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logit: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    y = y.float()
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


def weighted_bce(logit: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logit, y.float(), pos_weight=torch.tensor(pos_weight, device=logit.device)
    )


def edge_loss(logit, y, kind: str = "wce", pos_weight: float = 6.0, gamma: float = 2.0) -> torch.Tensor:
    if kind == "focal":
        return focal_loss(logit, y, gamma)
    if kind == "wce":
        return weighted_bce(logit, y, pos_weight)
    raise ValueError(kind)


def typology_loss(typ_logit: torch.Tensor, typ: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on positive edges only (typ >= 0); zero if none in batch."""
    mask = typ >= 0
    if not mask.any():
        return typ_logit.sum() * 0.0
    return F.cross_entropy(typ_logit[mask], typ[mask])
