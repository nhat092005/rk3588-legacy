"""CLI: build a split-head RKNN (raw boxes and scores outputs, decode on the CPU) of a trained run (phase 2d, candidate A1).

    .venv/bin/python -m ai.automation.build_split <run_id> int8|fp16
Writes runs/<run_id>/weights/rknn_<int8|fp16>_split/{best-rk3588.rknn, metadata.yaml (split_head: true), recipe.json} and the ONNX in
bench/onboard_q2c/split/<run_id>/. INT8 uses the Ultralytics settings (mean 0, std 255, toolkit defaults) and the calibration list of baseline seed 0.
The ONNX is checked in float with onnxruntime: decode_split of the two outputs against output0 of the standard weights/best.onnx.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

RUNS = Path(__file__).resolve().parent / "runs"
CALIB_RUN = "ppe5_yolov8n_baseline_100ep_s0"
SPLIT = RUNS / CALIB_RUN / "bench" / "onboard_q2c" / "split"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("mode", choices=["int8", "fp16"])
    args = ap.parse_args()
    import onnxruntime as ort
    from rknn.api import RKNN

    from ai.board.split_decode import decode_split
    from ai.core.rknn.split_head import export_split_onnx

    run = RUNS / args.run_id
    out_dir = SPLIT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = export_split_onnx(run / "weights" / "best.pt", out_dir / "split.onnx")
    x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    s = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    boxes, scores = s.run(None, {"images": x})
    ref = ort.InferenceSession(str(run / "weights" / "best.onnx"), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    check = float(np.abs(decode_split(boxes, scores) - ref).max())
    rk = run / "weights" / f"rknn_{args.mode}_split"
    rk.mkdir(parents=True, exist_ok=True)
    info = dict(run_id=args.run_id, mode=args.mode, onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(), onnx_outputs=[list(boxes.shape), list(scores.shape)],
                onnx_check_max_abs_vs_standard=check)
    rknn = RKNN(verbose=True, verbose_file=str(rk / "build.log"))
    assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588") == 0
    assert rknn.load_onnx(model=str(onnx)) == 0
    if args.mode == "int8":
        dataset = RUNS / CALIB_RUN / "weights" / "rknn_int8" / "dataset.txt"
        info["dataset_md5"] = hashlib.md5(dataset.read_bytes()).hexdigest()
        assert rknn.build(do_quantization=True, dataset=str(dataset), rknn_batch_size=1) == 0
    else:
        assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(str(rk / "best-rk3588.rknn")) == 0
    rknn.release()
    meta = yaml.safe_load((run / "weights" / "rknn_int8" / "metadata.yaml").read_text())
    meta["split_head"] = True
    (rk / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (rk / "recipe.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"Done: {rk}  float check max abs {check:.2e}")


if __name__ == "__main__":
    main()
