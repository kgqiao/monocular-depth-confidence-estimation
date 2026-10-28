"""The confidence head: a small CNN over the inputs of `ConfidenceDataset`, its loss, and its checkpoints.

The head outputs one logit per grid cell, the log-odds that the model's depth
there is wrong by more than tau. It standardizes each channel of x first, with
statistics of the train split held as buffers, so a checkpoint carries them.
A checkpoint also holds the HeadInputs of the training, so `load_head` gives
back all that is needed to build the head's inputs from a cache.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .dataset import HeadInputs, RadarConfig
from .labels import LabelConfig


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


def save_head(path: str | Path, head: ConfidenceHead, inputs: HeadInputs, channels: list[str], **fields: Any) -> None:
    """Write the head, the inputs it reads, and extra fields of tensors and plain values (epoch, history, ...).

    The file is replaced in one step, so a job stopped while saving leaves the old checkpoint whole.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"head": head.state_dict(), "in_channels": head.in_channels, "widths": list(head.widths),
                "inputs": asdict(inputs), "channels": list(channels), **fields}, tmp)
    os.replace(tmp, path)


def load_head(path: str | Path, device: str | torch.device = "cpu") -> tuple[ConfidenceHead, HeadInputs, dict[str, Any]]:
    """The head of a checkpoint, in eval mode on `device`; the HeadInputs of its training; the other fields.

    `ConfidenceDataset(cache, inputs, ...)` builds the head's inputs. Its `channel_names`
    must equal the checkpoint's "channels".
    """
    fields = torch.load(path, map_location="cpu")
    head = ConfidenceHead(fields.pop("in_channels"), tuple(fields.pop("widths")))
    head.load_state_dict(fields.pop("head"))
    saved = fields.pop("inputs")
    inputs = HeadInputs(**{**saved, "labels": LabelConfig(**saved["labels"]),
                           "radar_cfg": RadarConfig(**saved["radar_cfg"])})
    return head.to(device).eval(), inputs, fields
