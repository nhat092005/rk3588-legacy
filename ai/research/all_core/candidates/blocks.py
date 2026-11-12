"""Candidate blocks for the all-core detector study (deploy form: Conv + BN + activation, BN is folded by the exporter).

Every block is built only from Conv2d, an activation, Add, Mul and Concat unless it is a named control or a variant that
tests one risky op (maxpool, resize, convtranspose). REGISTRY maps a name to (factory, inputs) used by compile_audit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "leakyrelu": lambda: nn.LeakyReLU(0.1), "silu": nn.SiLU, "hardswish": nn.Hardswish}


def cb(c1, c2, k=1, s=1, g=1, act=None):
    layers = [nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(c2)]
    return nn.Sequential(*layers, ACTS[act]()) if act else nn.Sequential(*layers)


class Dense3(nn.Module):
    """RepVGG deploy form: two Conv3x3 + activation."""

    def __init__(self, c, act):
        super().__init__()
        self.a, self.b = cb(c, c, 3, act=act), cb(c, c, 3, act=act)

    def forward(self, x):
        return self.b(self.a(x))


class Res3(nn.Module):
    """ResNet basic block: act(cb3(cba3(x)) + x)."""

    def __init__(self, c, act):
        super().__init__()
        self.a, self.b, self.act = cb(c, c, 3, act=act), cb(c, c, 3), ACTS[act]()

    def forward(self, x):
        return self.act(self.b(self.a(x)) + x)


class MBConv(nn.Module):
    """MobileNetV2 and EfficientNet-Lite inverted residual (pw expand, dw, pw linear, Add)."""

    def __init__(self, c, act, k=3, e=4):
        super().__init__()
        h = c * e
        self.f = nn.Sequential(cb(c, h, 1, act=act), cb(h, h, k, g=h, act=act), cb(h, c, 1))

    def forward(self, x):
        return x + self.f(x)


class UIB(nn.Module):
    """MobileNetV4 universal inverted bottleneck: optional start DW (no activation), pw expand, optional mid DW, pw project."""

    def __init__(self, c, act, start_k=0, mid_k=3, e=4):
        super().__init__()
        h = c * e
        f = [cb(c, c, start_k, g=c)] if start_k else []
        f.append(cb(c, h, 1, act=act))
        if mid_k:
            f.append(cb(h, h, mid_k, g=h, act=act))
        f.append(cb(h, c, 1))
        self.f = nn.Sequential(*f)

    def forward(self, x):
        return x + self.f(x)


class Star(nn.Module):
    """StarNet block: dw, two pw, relu6(a) * b, pw, dw, Add."""

    def __init__(self, c, act="relu6", e=3, k=7):
        super().__init__()
        self.dw1, self.f1, self.f2 = cb(c, c, k, g=c), cb(c, c * e), cb(c, c * e)
        self.act, self.g, self.dw2 = ACTS[act](), cb(c * e, c), cb(c, c, k, g=c)

    def forward(self, x):
        y = self.dw1(x)
        return x + self.dw2(self.g(self.act(self.f1(y)) * self.f2(y)))


class Ghost(nn.Module):
    """GhostNet module: pw primary, dw cheap operation, Concat."""

    def __init__(self, c, act):
        super().__init__()
        self.p, self.q = cb(c, c // 2, 1, act=act), cb(c // 2, c // 2, 3, g=c // 2, act=act)

    def forward(self, x):
        y = self.p(x)
        return torch.cat([y, self.q(y)], 1)


class ConvAct(nn.Module):
    """Control: one Conv + activation."""

    def __init__(self, c, act):
        super().__init__()
        self.f = cb(c, c, 3, act=act)

    def forward(self, x):
        return self.f(x)


class HswishSplit(nn.Module):
    """Probe: HardSwish decomposed into Add, Clip, Mul so the compiler may keep Part A ops."""

    def __init__(self, c, act=None):
        super().__init__()
        self.f = cb(c, c, 3)

    def forward(self, x):
        y = self.f(x)
        return y * torch.clamp(y + 3.0, 0.0, 6.0) * (1.0 / 6.0)


class ActProbe(nn.Module):
    """Probe SiLU-like gates built from Clip, Mul and Relu. A following 1x1 conv is included so a folded constant has somewhere to go."""

    def __init__(self, c, kind):
        super().__init__()
        self.kind, self.f, self.g = kind, cb(c, c, 3), cb(c, c, 1)

    def forward(self, x):
        y = self.f(x)
        if self.kind == "pwl_slope25":  # y * clip(0.25 y + 0.5, 0, 1)
            y = y * torch.clamp(y * 0.25 + 0.5, 0.0, 1.0)
        elif self.kind == "hs_noscale":  # y * clip(y + 3, 0, 6), the 1/6 is left to the next conv weights
            y = y * torch.clamp(y + 3.0, 0.0, 6.0)
        elif self.kind == "relu_sq":  # relu(y) * relu(y)
            r = torch.relu(y)
            y = r * r
        elif self.kind == "relu_gate":  # y * relu6(y) style gate on a ReLU6 branch
            y = y * torch.clamp(y, 0.0, 1.0)
        return self.g(y)


class Down(nn.Module):
    def __init__(self, c, act, mode="conv"):
        super().__init__()
        self.mode, self.f = mode, (cb(c, c, 3, s=2, act=act) if mode == "conv" else nn.MaxPool2d(2))

    def forward(self, x):
        return self.f(x)


class UpCat(nn.Module):
    """Top-down neck step: upsample the low-resolution map, Concat with the high-resolution one, 1x1 fuse. mode none keeps only the high map."""

    def __init__(self, c, act, mode="resize"):
        super().__init__()
        self.mode = mode
        self.up = nn.ConvTranspose2d(c, c, 2, 2) if mode == "convt" else None
        self.fuse = cb(c * (1 if mode == "none" else 2), c, 1, act=act)

    def forward(self, lo, hi):
        if self.mode == "none":
            return self.fuse(hi)
        up = self.up(lo) if self.up is not None else F.interpolate(lo, scale_factor=2.0, mode="nearest")
        return self.fuse(torch.cat([up, hi], 1))


class HeadV6(nn.Module):
    """YOLOv6 N/S style head without DFL: per level stem 1x1, cls (3x3 then 1x1), reg (3x3 then 1x1). Raw NCHW outputs, no sigmoid, no decode."""

    def __init__(self, act, chs=(64, 128, 256), nc=5):
        super().__init__()
        self.stem = nn.ModuleList(cb(c, c, 1, act=act) for c in chs)
        self.cls = nn.ModuleList(nn.Sequential(cb(c, c, 3, act=act), nn.Conv2d(c, nc, 1)) for c in chs)
        self.reg = nn.ModuleList(nn.Sequential(cb(c, c, 3, act=act), nn.Conv2d(c, 4, 1)) for c in chs)

    def forward(self, p3, p4, p5):
        out = []
        for i, x in enumerate((p3, p4, p5)):
            x = self.stem[i](x)
            out += [self.cls[i](x), self.reg[i](x)]
        return tuple(out)


SIZES = {"s80": (64, 80), "s40": (128, 40), "s20": (256, 20)}  # channels aligned to 32 (Conv) and 64 (DW hidden = 4c)


def _one(c, hw):
    return [(1, c, hw, hw)]


REGISTRY = {  # name: (factory(c, act), inputs(c, hw), activations to try, sizes)
    "conv_act": (ConvAct, _one, ("relu", "relu6", "leakyrelu", "silu", "hardswish"), SIZES),
    "hswish_split": (HswishSplit, _one, (None,), SIZES),
    **{f"act_{k}": ((lambda k: lambda c, a: ActProbe(c, k))(k), _one, (None,), {"s40": (128, 40)}) for k in ("pwl_slope25", "hs_noscale", "relu_sq", "relu_gate")},
    "dense3": (Dense3, _one, ("relu", "leakyrelu"), SIZES),
    "res3": (Res3, _one, ("relu", "leakyrelu"), SIZES),
    "mbconv_k3": (lambda c, a: MBConv(c, a, 3), _one, ("relu", "relu6"), SIZES),
    "mbconv_k5": (lambda c, a: MBConv(c, a, 5), _one, ("relu6",), SIZES),
    "uib_ib": (lambda c, a: UIB(c, a, 0, 3), _one, ("relu",), SIZES),
    "uib_convnext": (lambda c, a: UIB(c, a, 7, 0), _one, ("relu",), SIZES),
    "uib_extradw": (lambda c, a: UIB(c, a, 3, 3), _one, ("relu",), SIZES),
    "uib_ffn": (lambda c, a: UIB(c, a, 0, 0), _one, ("relu",), SIZES),
    "star": (lambda c, a: Star(c, a), _one, ("relu6",), SIZES),
    "ghost": (Ghost, _one, ("relu",), SIZES),
    "down_conv": (lambda c, a: Down(c, a, "conv"), _one, ("relu",), SIZES),
    "down_maxpool": (lambda c, a: Down(c, a, "maxpool"), _one, ("relu",), SIZES),
    "neck_resize": (lambda c, a: UpCat(c, a, "resize"), lambda c, hw: [(1, c, hw // 2, hw // 2), (1, c, hw, hw)], ("relu",), SIZES),
    "neck_convt": (lambda c, a: UpCat(c, a, "convt"), lambda c, hw: [(1, c, hw // 2, hw // 2), (1, c, hw, hw)], ("relu",), SIZES),
    "neck_none": (lambda c, a: UpCat(c, a, "none"), lambda c, hw: [(1, c, hw // 2, hw // 2), (1, c, hw, hw)], ("relu",), SIZES),
    "head_v6": (lambda c, a: HeadV6(a), lambda c, hw: [(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)], ("relu", "leakyrelu", "silu"), {"p345": (0, 0)}),
}
