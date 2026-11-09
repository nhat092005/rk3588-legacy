"""Box decode of a split-head model (numpy only, runs on the board and on the PC).

boxes (B, 4*reg_max, A) are the DFL logits and scores (B, nc, A) the class logits of the raw head outputs. decode_split reproduces
Detect._inference of Ultralytics (DFL softmax expectation, dist2bbox xywh, multiplied by stride, sigmoid scores), tested in tests/test_split_head.py.
"""
from __future__ import annotations

import numpy as np


def anchors_and_strides(imgsz: int = 640, strides: tuple[int, ...] = (8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    """Anchor points (A x 2, cell centre offset 0.5) and strides (A,), levels concatenated in row-major order like make_anchors."""
    pts, st = [], []
    for s in strides:
        n = imgsz // s
        sy, sx = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        pts.append(np.stack((sx, sy), -1).reshape(-1, 2))
        st.append(np.full(n * n, s, dtype=np.float32))
    return np.concatenate(pts).astype(np.float32), np.concatenate(st)


def decode_split(boxes: np.ndarray, scores: np.ndarray, imgsz: int = 640, reg_max: int | None = None, strides: tuple[int, ...] = (8, 16, 32)) -> np.ndarray:
    """boxes (B, 4*reg_max, A) and scores (B, nc, A) logits to (B, 4 + nc, A): xywh in pixels and sigmoid scores.

    reg_max defaults to boxes.shape[1] // 4. With reg_max 1 there is no DFL (Detect.dfl is Identity) and boxes already hold ltrb distances.
    """
    b, ch, a = boxes.shape
    reg_max = ch // 4 if reg_max is None else reg_max
    assert ch == 4 * reg_max, f"{ch} box channels, 4 * reg_max = {4 * reg_max}"
    pts, st = anchors_and_strides(imgsz, strides)
    assert pts.shape[0] == a, f"{a} anchors in the output, {pts.shape[0]} expected for imgsz {imgsz}"
    if reg_max == 1:
        dist = boxes.astype(np.float32).reshape(b, 4, a)  # ltrb distances
    else:
        x = boxes.astype(np.float32).reshape(b, 4, reg_max, a)
        x = np.exp(x - x.max(axis=2, keepdims=True))
        dist = (x / x.sum(axis=2, keepdims=True) * np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)).sum(axis=2)  # (B, 4, A) ltrb
    lt, rb = dist[:, :2], dist[:, 2:]
    ap = pts.T[None]  # (1, 2, A)
    x1y1, x2y2 = ap - lt, ap + rb
    xywh = np.concatenate(((x1y1 + x2y2) / 2, x2y2 - x1y1), axis=1) * st[None, None, :]
    return np.concatenate((xywh, 1.0 / (1.0 + np.exp(-scores.astype(np.float32)))), axis=1)
