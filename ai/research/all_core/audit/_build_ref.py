"""Child process: export a YOLO yaml (random weights, nc=5) through Ultralytics' own RKNN path, but build with verbose logging on the PC.

    python -m ai.research.all_core.audit._build_ref <model.yaml> <fp16|int8> <workdir>
Mirrors the measured T3 path (ultralytics export format=rknn, opset 19). The board is not used.
"""
import sys
from pathlib import Path

import numpy as np
import ultralytics.utils.export.rknn as urknn
from PIL import Image
from rknn.api import RKNN
from ultralytics import YOLO

yaml_path, precision, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])


def onnx2rknn(onnx_file, output_dir, name="rk3588", quantize=None, dataset=None, metadata=None, prefix="", batch=1):
    kw = {"do_quantization": False}
    if precision == "int8":
        work.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        names = []
        for i in range(40):  # random images: fused op names and targets only, never accuracy
            f = work / f"c{i}.png"
            Image.fromarray(rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)).save(f)
            names.append(str(f))
        (work / "dataset.txt").write_text("\n".join(names) + "\n")
        kw = {"do_quantization": True, "dataset": str(work / "dataset.txt")}
    r = RKNN(verbose=True)
    assert r.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=name) == 0
    assert r.load_onnx(model=onnx_file) == 0
    assert r.build(**kw, rknn_batch_size=batch) == 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert r.export_rknn(str(Path(output_dir) / f"{Path(onnx_file).stem}-{name}.rknn")) == 0  # Ultralytics checks the file size
    r.release()
    return str(output_dir)


urknn.onnx2rknn = onnx2rknn
YOLO(yaml_path).export(format="rknn", name="rk3588", quantize=16, imgsz=640)
