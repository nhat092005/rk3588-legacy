"""Whole nano-scale detector candidates for the all-core study: backbone family x neck mode x activation, YOLOv6 N/S style head.

Residual blocks put Add last (activation before Add, or none), the form the compiler fuses into the conv. Outputs are raw NCHW
tensors per level (cls logits and box), no sigmoid, no decode. Widths follow P3/P4/P5 = 64/128/256 at strides 8/16/32.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.research.all_core.candidates.blocks import ACTS, Dense3, HeadV6, MBConv, Star, UIB, cb


class Bott(nn.Module):
    """YOLO style bottleneck: two Conv3x3 + act, then Add (measured as ConvReluAdd)."""

    def __init__(self, c, act):
        super().__init__()
        self.a, self.b = cb(c, c, 3, act=act), cb(c, c, 3, act=act)

    def forward(self, x):
        return x + self.b(self.a(x))


FAMILIES = {  # name: (block factory(c, act), activations to try)
    "dense": (Dense3, ("relu", "leakyrelu")),
    "res": (Bott, ("relu", "leakyrelu")),
    "mbconv": (lambda c, a: MBConv(c, a, 3, 4), ("relu6", "relu")),
    "uib": (lambda c, a: UIB(c, a, 3, 3, 4), ("relu",)),  # ExtraDW form. The IB form (no start dw) equals mbconv k3
    "star": (lambda c, a: Star(c, "relu6"), ("relu",)),
}
NECKS = ("resize", "convt", "none")
DEPTHS = (1, 1, 1, 1)  # blocks at stride 4, 8, 16, 32. Not matched to the YOLOv8n budget: the op audit comes before scaling
WIDTHS = (32, 64, 128, 256)


class Neck(nn.Module):
    """resize and convt: top-down (upsample) then bottom-up. none: bottom-up only, no upsample."""

    def __init__(self, mode, act, w=(64, 128, 256)):
        super().__init__()
        self.mode = mode
        c3, c4, c5 = w
        if mode != "none":
            self.up5 = nn.ConvTranspose2d(c5, c5, 2, 2) if mode == "convt" else None
            self.up4 = nn.ConvTranspose2d(c4, c4, 2, 2) if mode == "convt" else None
            self.td4, self.td3 = cb(c5 + c4, c4, 3, act=act), cb(c4 + c3, c3, 3, act=act)
        self.d3, self.d4 = cb(c3, c3, 3, 2, act=act), cb(c4, c4, 3, 2, act=act)
        self.bu4 = cb(c3 + c4, c4, 3, act=act)
        self.bu5 = cb(c4 + c5, c5, 3, act=act)

    def _up(self, x, mod):
        return mod(x) if mod is not None else F.interpolate(x, scale_factor=2.0, mode="nearest")

    def forward(self, p3, p4, p5):
        if self.mode == "none":
            n3 = p3
        else:
            td4 = self.td4(torch.cat([self._up(p5, self.up5), p4], 1))
            n3 = self.td3(torch.cat([self._up(td4, self.up4), p3], 1))
            p4 = td4
        n4 = self.bu4(torch.cat([self.d3(n3), p4], 1))
        n5 = self.bu5(torch.cat([self.d4(n4), p5], 1))
        return n3, n4, n5


class Det(nn.Module):
    def __init__(self, family, neck, act, nc=5, widths=WIDTHS):
        super().__init__()
        block, _ = FAMILIES[family]
        c2, c3, c4, c5 = widths
        stage = lambda c, n: nn.Sequential(*[block(c, act) for _ in range(n)])  # noqa: E731
        self.stem = cb(3, c2, 3, 2, act=act)
        self.s4 = nn.Sequential(cb(c2, c2, 3, 2, act=act), stage(c2, DEPTHS[0]))
        self.p3 = nn.Sequential(cb(c2, c3, 3, 2, act=act), stage(c3, DEPTHS[1]))
        self.p4 = nn.Sequential(cb(c3, c4, 3, 2, act=act), stage(c4, DEPTHS[2]))
        self.p5 = nn.Sequential(cb(c4, c5, 3, 2, act=act), stage(c5, DEPTHS[3]))
        self.neck, self.head = Neck(neck, act, (c3, c4, c5)), HeadV6(act, (c3, c4, c5), nc)

    def forward(self, x):
        p3 = self.p3(self.s4(self.stem(x)))
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return self.head(*self.neck(p3, p4, p5))


def complexity(model: nn.Module, x: torch.Tensor) -> dict:
    """Parameters (with BN) and GFLOPs (2 x MACs of Conv2d and ConvTranspose2d, YOLO convention) for one forward at input x."""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:  # ConvTranspose2d: every input element scatters through the kernel
            macs[0] += inp[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    with torch.no_grad():
        model.eval()(x)
    for h in hs:
        h.remove()
    return dict(params=sum(p.numel() for p in model.parameters()), gflops=round(2 * macs[0] / 1e9, 3))
