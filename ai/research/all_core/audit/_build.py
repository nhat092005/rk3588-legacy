"""Child process: build an ONNX with rknn-toolkit2 on the PC (no board) and print the compiler log, which holds the layer table.

    python -m ai.research.all_core.audit._build <model.onnx> <fp16|int8> <workdir> [out.rknn]
With out.rknn the built model is also exported for the board.
INT8 uses random calibration data, so it shows the fused op names and targets only, never accuracy.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from rknn.api import RKNN


def main() -> None:
    path, precision, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    kw = {"do_quantization": False}
    if precision == "int8":
        work.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in onnx.load(path).graph.input]
        lines = []
        for n in range(8):
            files = []
            for k, s in enumerate(shapes):
                f = work / f"c{n}_{k}.npy"
                np.save(f, rng.normal(size=s).astype(np.float32))
                files.append(str(f))
            lines.append(" ".join(files))
        (work / "dataset.txt").write_text("\n".join(lines) + "\n")
        kw = {"do_quantization": True, "dataset": str(work / "dataset.txt")}
    r = RKNN(verbose=True)
    assert r.config(target_platform="rk3588") == 0
    assert r.load_onnx(model=path) == 0
    assert r.build(**kw) == 0
    if len(sys.argv) > 4:
        assert r.export_rknn(sys.argv[4]) == 0
    r.release()


if __name__ == "__main__":
    main()
