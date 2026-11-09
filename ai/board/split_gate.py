"""Correctness gate of experiment E2 (run ON THE BOARD): the harness path of a split-head model (two raw outputs, numpy decode, postprocess_raw)
must give the same detections as the standard int8h5 model on the same images. Rule: experiments/npu/throughput/onboard_q2f_split/rule.txt.

Usage: .venv-board/bin/python -m ai.board.split_gate --standard <rknn_int8h5 dir> --split <rknn_int8_split dir> --images <dir> --out <json> [--n 20]
Detections at conf 0.25 (deployment default). A split detection matches a standard detection of the same class with IoU >= 0.5 (greedy by score).
Reports detections per model, matches, F1 = 2 matches / (n_standard + n_split) over all images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ai.board.bench_throughput_rep import load_ctx
from ai.board.common import list_images, postprocess_raw, preprocess


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def match(std: np.ndarray, spl: np.ndarray) -> int:
    used, n = set(), 0
    for d in spl[np.argsort(-spl[:, 4])]:
        cand = [i for i in range(len(std)) if i not in used and int(std[i, 5]) == int(d[5])]
        if not cand:
            continue
        ious = iou(d[:4], std[cand, :4])
        k = int(np.argmax(ious))
        if ious[k] >= 0.5:
            used.add(cand[k])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    frames = [cv2.imread(str(p)) for p in list_images(a.images)[:a.n]]
    s_rknn, s_norm, s_split = load_ctx(a.standard, "CORE_0")
    p_rknn, p_norm, p_split = load_ctx(a.split, "CORE_0")
    assert not s_split and p_split, "the standard model must not be split-head and the split model must be"
    rows, n_std, n_spl, n_match = [], 0, 0, 0
    for i, f in enumerate(frames):
        inp, meta = preprocess(f)
        d_std = postprocess_raw(s_rknn.inference(inputs=[inp]), meta, s_norm, False)
        d_spl = postprocess_raw(p_rknn.inference(inputs=[inp]), meta, p_norm, True)
        m = match(d_std, d_spl) if len(d_std) and len(d_spl) else 0
        rows.append(dict(i=i, n_standard=int(len(d_std)), n_split=int(len(d_spl)), matched=m))
        n_std, n_spl, n_match = n_std + len(d_std), n_spl + len(d_spl), n_match + m
    out = dict(images=len(frames), n_standard=n_std, n_split=n_spl, matched=n_match, f1=2 * n_match / max(n_std + n_spl, 1), per_image=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}))
    s_rknn.release()
    p_rknn.release()


if __name__ == "__main__":
    main()
