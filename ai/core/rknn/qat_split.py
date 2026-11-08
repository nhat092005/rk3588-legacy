"""QAT building blocks for the split-head YOLOv8n (phase 2d, candidate B): FX-traceable wrapper, prepare_qat_fx with the qconfig of UG V2.3.2 p127.

Ultralytics' DetectionModel.forward is not traceable (C2f unpacks a chunk, the Detect head decodes with shape-dependent code). The wrapper
runs the layers like _predict_once, C2f uses static slices, and the head returns the raw (boxes, scores) of forward_head.
"""
from __future__ import annotations

import copy

import torch


def patch_c2f() -> None:
    from ultralytics.nn.modules import block

    def forward(self, x):
        t = self.cv1(x)
        y = [t[:, : self.c], t[:, self.c :]]  # static slices: FX cannot unpack Tensor.chunk
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    block.C2f.forward = forward


class SplitWrap(torch.nn.Module):
    """Layers of a DetectionModel run like _predict_once, the head returns (boxes, scores)."""

    def __init__(self, det):
        super().__init__()
        self.layers = det.model
        self.save = set(det.save)
        head = det.model[-1]
        head.forward = lambda x: tuple(head.forward_head(list(x), **head.one2many)[k] for k in ("boxes", "scores"))

    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def rknn_qconfig():
    """Activation FakeQuantize with MovingAverageMinMaxObserver (0 to 255, reduce_range False), weight per-channel affine int8 (UG V2.3.2 p127)."""
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, QConfig

    act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    wt = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_affine, ch_axis=0)
    return QConfig(activation=act, weight=wt)


def prepare(det, imgsz: int = 640):
    """Unfused DetectionModel (BN present, train mode) to a QAT GraphModule."""
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    patch_c2f()
    wrap = SplitWrap(copy.deepcopy(det).float().train())
    return prepare_qat_fx(wrap, QConfigMapping().set_global(rknn_qconfig()), (torch.rand(1, 3, imgsz, imgsz),))
