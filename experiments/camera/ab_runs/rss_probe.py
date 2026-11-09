"""RSS growth per frame of each camera-pipeline stage, without the camera. Run on the board, one stage per process:

  cd ~/rk3588 && [PIN=1] PYTHONPATH=. taskset -c 4-7 .venv-board/bin/python /tmp/rss_probe.py <stage> [frames] [model_dir]

stages: capture (a bytearray per frame + np.frombuffer), cvt, pre, npu, post (the full step, nothing stored),
        bookkeeping (the full step plus what camera.py stores: records dict, arrivals list, ResourceMonitor thread).
RSS is read after gc.collect() before and after the frames, and once more after malloc_trim(0): growth that trim gives
back is free memory the allocator kept, growth that stays is live memory.
"""
import ctypes
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from ai.board import camera
from ai.board.common import ResourceMonitor, postprocess_raw, preprocess

W, H = 640, 360
STAGES = ("capture", "cvt", "pre", "npu", "post", "bookkeeping")


def rss() -> float:
    gc.collect()
    return psutil.Process().memory_info().rss / 2**20


def main() -> None:
    stage = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    model = Path(sys.argv[3] if len(sys.argv) > 3 else "experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/rknn_int8")
    assert stage in STAGES, STAGES
    if os.environ.get("PIN"):  # PIN=1: apply camera.pin_mmap_threshold() first
        print("mallopt M_MMAP_THRESHOLD set:", camera.pin_mmap_threshold())
    nv = np.full((H * 3 // 2, W), 110, dtype=np.uint8)  # static gray frame: no detections, like the empty room
    rknn, normalized = None, False
    if stage in ("npu", "post", "bookkeeping"):
        from ai.board.bench_npu import load_model

        rknn, normalized = load_model(model, "CORE_0")

    step = camera.make_step(rknn, normalized, False, 0.25, None, (H * 3 // 2, W)) if rknn else None
    records, arrivals = [], []

    def one() -> None:
        if stage == "capture":
            np.frombuffer(bytearray(W * H * 3 // 2), dtype=np.uint8)
            return
        if stage == "bookkeeping":
            frame = np.frombuffer(bytearray(W * H * 3 // 2), dtype=np.uint8)
            t0 = time.perf_counter()
            arrivals.append(t0)
            res = step(frame)
            t1 = time.perf_counter()
            records.append({"t_arr": t0, "t_done": t1, "e2e": (t1 - t0) * 1e3, **res})
            return
        bgr = cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12)
        if stage == "cvt":
            return
        inp, meta = preprocess(bgr)
        if stage == "pre":
            return
        outs = rknn.inference(inputs=[inp])
        if stage == "npu":
            return
        postprocess_raw(outs, meta, normalized, False, 0.25)

    monitor = ResourceMonitor() if stage == "bookkeeping" else None
    for _ in range(300):  # warm-up: allocator and runtime settle
        one()
    records.clear()
    arrivals.clear()
    if monitor:
        monitor.__enter__()
    r0 = rss()
    t0 = time.perf_counter()
    for _ in range(n):
        one()
    dt = time.perf_counter() - t0
    r1 = rss()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    r2 = rss()
    if monitor:
        monitor.__exit__(None, None, None)
    print(f"{stage:11s} frames {n:6d} in {dt:6.1f} s | rss {r0:7.1f} -> {r1:7.1f} MiB (+{r1 - r0:6.2f}, {(r1 - r0) * 1024 / n:6.3f} KiB/frame)"
          f" | after malloc_trim {r2:7.1f} (+{r2 - r0:6.2f}, {(r2 - r0) * 1024 / n:6.3f} KiB/frame)")
    if rknn:
        rknn.release()


if __name__ == "__main__":
    main()
