"""Split-head export: the NPU graph ends at the raw head outputs, the box decode runs on the CPU in numpy.

The Ultralytics Detect head returns the raw outputs `boxes` (B x 4*reg_max x A, DFL logits) and `scores` (B x nc x A, class logits)
from `forward_head` (ultralytics/nn/modules/head.py). `export_split_onnx` exports those two tensors. `decode_split` reproduces
Detect._inference (DFL softmax expectation, dist2bbox xywh, multiplied by stride, sigmoid scores) with numpy, tested against it in
tests/test_split_head.py. The decode needs no torch, so the board can run it.
"""
from __future__ import annotations

import copy
from pathlib import Path


from ai.board.split_decode import anchors_and_strides, decode_split  # noqa: F401  (numpy decode shared with the board)


def split_model(weights: str | Path):
    """Fused float model whose forward returns (boxes, scores) through the head's own forward_head."""
    from ultralytics import YOLO

    model = copy.deepcopy(YOLO(str(weights)).model).float().eval()
    model.fuse(verbose=False)
    head = model.model[-1]
    head.forward = lambda x: tuple(head.forward_head(list(x), **head.one2many)[k] for k in ("boxes", "scores"))
    return model


def export_split_onnx(weights: str | Path, out_onnx: str | Path, imgsz: int = 640, opset: int = 19) -> Path:
    import torch

    model = split_model(weights)
    out_onnx = Path(out_onnx)
    torch.onnx.export(model, torch.zeros(1, 3, imgsz, imgsz), str(out_onnx), opset_version=opset, input_names=["images"], output_names=["boxes", "scores"], dynamic_axes=None)
    return out_onnx
