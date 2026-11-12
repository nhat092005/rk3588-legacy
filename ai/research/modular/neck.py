"""Neck contract. No architecture here: a new neck subclasses `Neck`.

`__init__(in_channels)` takes the backbone `out_channels`. forward([P3, P4, P5]) returns [N3, N4, N5] at the same strides.
`out_channels` is the tuple of their channel counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Neck(nn.Module):
    out_channels: tuple[int, int, int]

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError
