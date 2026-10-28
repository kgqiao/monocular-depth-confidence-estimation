"""The confidence head: a small CNN over the inputs of `ConfidenceDataset`, and its loss.

The head outputs one logit per grid cell, the log-odds that the model's depth
there is wrong by more than tau. It standardizes each channel of x first, with
statistics of the train split held as buffers, so a checkpoint carries them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConfidenceHead(nn.Module):
    """A 3x3 conv and a ReLU per width, then a 1x1 conv to one logit per cell."""

    def __init__(self, in_channels: int, widths: tuple[int, ...] = (64, 64, 32)):
        super().__init__()
        self.in_channels = in_channels
        self.widths = tuple(widths)
        self.register_buffer("mean", torch.zeros(in_channels, 1, 1))
        self.register_buffer("std", torch.ones(in_channels, 1, 1))
        layers, c = [], in_channels
        for w in self.widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(inplace=True)]
            c = w
        layers.append(nn.Conv2d(c, 1, 1))
        self.net = nn.Sequential(*layers)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """The per-channel mean and std of x. A channel with std 0 is only centered."""
        self.mean.copy_(mean.reshape(-1, 1, 1))
        self.std.copy_(torch.where(std > 0, std, torch.ones_like(std)).reshape(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, C, h, w) -> logits (B, 1, h, w)."""
        return self.net((x - self.mean) / self.std)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               pos_weight: float | None = None) -> torch.Tensor:
    """Binary cross-entropy averaged over the labeled cells (mask 1); 0 when no cell is labeled."""
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)
