"""Run ON THE BOARD from ~/rk3588: grab N real frames from the ISP selfpath (16:9 crop, 640x360 NV12), keep every STEP-th, write BGR PNG.
Usage: .venv-board/bin/python tmp_capture_frames.py <out_dir> [n_frames=200] [step=3]"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ai.board.camera import CROP_16_9, CROP_FULL, capture_command, check_frame_size, read_exact, set_crop

DEV, W, H = "/dev/video12", 640, 360
out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
step = int(sys.argv[3]) if len(sys.argv) > 3 else 3
out.mkdir(parents=True, exist_ok=True)
proc = None
try:
    set_crop(DEV, CROP_16_9)
    size = check_frame_size(DEV, W, H)
    proc = subprocess.Popen(capture_command(DEV, W, H), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    buf, saved, i = bytearray(size), 0, 0
    while saved < n and read_exact(proc.stdout, buf):
        if i >= 30 and (i - 30) % step == 0:  # skip the first 30 frames (auto exposure settling)
            bgr = cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
            cv2.imwrite(str(out / f"cam_{saved:04d}.png"), bgr)
            saved += 1
        i += 1
    print(f"saved {saved} frames from {i} read to {out}")
finally:
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=5)
    set_crop(DEV, CROP_FULL)
