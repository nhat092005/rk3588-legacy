"""CLI: phase 2e board micro models (G7b Resize attributes, G11 standalone Relu), statements of experiments/npu/operator_support/onboard_q2e/rule_micro.txt.

    .venv/bin/python -m ai.npu.operator_support.micro_ops 192.168.137.49:5555
Writes results/npu/operator_support/q2e_micro.json. The board must be locked (make board-check LOCK=1) and rknn_server running.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ai.core.rknn.benchmark import connect_board, measure_layers
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUN = RUNS_DIR / "ppe5_yolov8n_baseline_100ep_s0"
SETS = {"A": ("asymmetric", "floor"), "B": ("half_pixel", "floor"), "C": ("half_pixel", "round_prefer_floor")}


def resize_model(coord: str, nearest: str, shape):
    from onnx import TensorProto, helper

    n, c, h, w = shape
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest", coordinate_transformation_mode=coord, nearest_mode=nearest)
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, 2 * h, 2 * w])],
                              [helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def relu_model(shape):
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))], [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def ort_run(model, x):
    import onnxruntime as ort

    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def build(model, out: Path) -> str:
    import onnx
    from rknn.api import RKNN

    onnx_path = out.with_suffix(".onnx")
    onnx.save(model, str(onnx_path))
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    assert rknn.load_onnx(model=str(onnx_path)) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(str(out)) == 0
    rknn.release()
    return str(out)


def run_on_board(path: str, device: str, x: np.ndarray, core_mask: str = "CORE_0", perf: bool = False):
    from rknn.api import RKNN  # noqa: F401  (connect_board imports it too)

    with connect_board(path, device, core_mask, **({"perf_debug": True} if perf else {})) as rknn:
        y = rknn.inference(inputs=[x], data_format="nchw")[0]
        layers = measure_layers(rknn, repeats=1)[0]["layers"] if perf else None
    return y, layers


def main() -> None:
    device = sys.argv[1]
    out: dict = {"rule": (experiment_dir(RUN, "onboard_q2e") / "rule_micro.txt").read_text(), "device": device}
    tmp = Path(tempfile.mkdtemp(prefix="micro_ops_"))
    rng = np.random.default_rng(0)
    # ---- G7b
    x = rng.normal(size=(1, 256, 20, 20)).astype(np.float32)
    models = {k: resize_model(*SETS[k], x.shape) for k in SETS}
    ref = ort_run(models["A"], x)
    g7 = {}
    for k, m in models.items():
        own = ort_run(m, x)
        path = build(m, tmp / f"resize_{k}.rknn")
        y, layers = run_on_board(path, device, x, "CORE_0", perf=True)
        g7[k] = dict(attrs=SETS[k], ort_own_vs_ref=float(np.abs(own - ref).max()), npu_vs_own=float(np.abs(y.astype(np.float32) - own).max()), npu_vs_ref=float(np.abs(y.astype(np.float32) - ref).max()),
                     table=[dict(op=r["op"], target=r["target"], dtype=r["dtype"], workload=r["workload"], name=r["name"]) for r in layers])
        g7[k]["match_own"] = g7[k]["npu_vs_own"] < 0.01
        g7[k]["match_ref"] = g7[k]["npu_vs_ref"] < 0.01
    out["g7b"] = g7
    # ---- G11
    xr = rng.normal(size=(1, 64, 80, 80)).astype(np.float32)
    rm = relu_model(xr.shape)
    path = build(rm, tmp / "relu.rknn")
    g11 = {}
    for mask in ("CORE_0", "CORE_0_1_2"):
        y, layers = run_on_board(path, device, xr, mask, perf=True)
        g11[mask] = dict(max_abs_vs_ort=float(np.abs(y.astype(np.float32) - ort_run(rm, xr)).max()),
                         table=[dict(op=r["op"], target=r["target"], dtype=r["dtype"], workload=r["workload"], name=r["name"], time_us=r["time_us"]) for r in layers])
    out["g11"] = g11
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2e_micro.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: {kk: {a: b for a, b in vv.items() if a != "table"} for kk, vv in v.items()} for k, v in out.items() if k in ("g7b", "g11")}, indent=1))
    print("tables:", {k: v["table"] for k, v in g7.items()}, {k: v["table"] for k, v in g11.items()})


if __name__ == "__main__":
    main()
