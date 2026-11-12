"""Backbone contract. No architecture here: a new backbone subclasses `Backbone`.

forward(image) returns [P3, P4, P5] at strides 8, 16, 32. `out_channels` is the tuple of their channel counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Backbone(nn.Module):
    out_channels: tuple[int, int, int]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError
