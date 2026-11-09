"""CLI: phase 2e PC-only analysis (G7a, G10, M1, QAT FP16 rows), statements of experiments/npu/operator_support/onboard_q2e/rule_pc.txt.

    .venv/bin/python -m ai.npu.operator_support.analyze_q2e_pc
Writes results/npu/operator_support/q2e_pc.json and results/npu/operator_support/q2e_pc.md. Descriptive only, no mechanism.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

from ai.core.rknn.benchmark import parse_layers
from ai.paths import REPO_ROOT as REPO, OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUN = RUNS_DIR / "ppe5_yolov8n_baseline_100ep_s0"
LOGS = experiment_dir(RUN, "onboard_q2c") / "logs"
ONNX = experiment_dir(RUN, "onboard_q2c") / "diag" / "ppe5_yolov8n_baseline_100ep_s0" / "int8_input.onnx"
ATTR_SETS = [("asymmetric", "floor"), ("half_pixel", "round_prefer_floor"), ("half_pixel", "floor"), ("pytorch_half_pixel", "round_prefer_floor"), ("align_corners", "round_prefer_floor")]


def first_table(log: Path) -> tuple[list[dict], dict[int, str]]:
    """First Network Layer Information Table rows and the output shape string of each row id."""
    lines = [re.sub(r"^.*?\] ", "", ln) for ln in log.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln]
    text = "\n".join(lines)
    start = text.index("Network Layer Information Table")
    rows = []
    for r in parse_layers(text[start:]):
        if rows and r["id"] <= rows[-1]["id"]:
            break
        rows.append(r)
    shapes: dict[str, str] = {}  # FullName -> output shape of the first table
    ids = {r["id"] for r in rows}
    seen = set()
    for ln in text[start:].split("\n"):
        parts = ln.split()
        if len(parts) > 6 and parts[0].isdigit() and parts[3] in ("NPU", "CPU") and int(parts[0]) in ids and int(parts[0]) not in seen:
            seen.add(int(parts[0]))
            shapes[parts[-1]] = parts[5]
    return rows, shapes


def resize_ort(coord: str, nearest: str, scale: float, x: np.ndarray) -> np.ndarray:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    n, c, h, w = x.shape
    oh, ow = int(h * scale), int(w * scale)
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest", coordinate_transformation_mode=coord, nearest_mode=nearest)
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, oh, ow])],
                              [helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, oh / h, ow / w])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.checker.check_model(model)
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def main() -> None:
    import onnx

    out: dict = {"rule": (experiment_dir(RUN, "onboard_q2e") / "rule_pc.txt").read_text()}
    # ---- G7a
    m = onnx.load(str(ONNX))
    out["g7a"] = {"onnx": str(ONNX.relative_to(REPO)), "resize_attrs": [{a.name: (a.s.decode() if a.s else a.i) for a in n.attribute} | {"name": n.name} for n in m.graph.node if n.op_type == "Resize"]}
    x = np.random.default_rng(0).normal(size=(1, 256, 20, 20)).astype(np.float32)
    ref = {s: resize_ort("asymmetric", "floor", s, x) for s in (2.0, 1.5, 3.0)}
    out["g7a"]["max_abs_diff"] = {f"scale_{s}": {f"{c}|{nm}": float(np.abs(resize_ort(c, nm, s, x) - ref[s]).max()) for c, nm in ATTR_SETS[1:]} for s in (2.0, 1.5, 3.0)}
    # ---- G10
    rows, shapes = first_table(LOGS / "build_ppe5_yolov8n_baseline_100ep_s0_base.log")
    onnx_ops = collections.Counter(n.op_type for n in m.graph.node)
    rknn_ops = collections.Counter(r["op"] for r in rows)
    full = " ".join(r["name"] for r in rows)
    resh = [n.name for n in m.graph.node if n.op_type == "Reshape"]
    exact = {r["name"].split(":", 1)[1] for r in rows if ":" in r["name"]}
    out["g10"] = {"onnx_nodes": len(m.graph.node), "rknn_rows": len(rows), "onnx_ops": dict(onnx_ops), "rknn_ops": dict(rknn_ops),
                  "reshape_nodes": {n: (n in full) for n in resh}, "reshape_nodes_exact": {n: (n in exact) for n in resh},
                  "reshape_rows": [r["name"] for r in rows if r["op"] == "Reshape"],
                  "split_nodes_exact": {n.name: (n.name in exact) for n in m.graph.node if n.op_type in ("Split", "Slice")},
                  "split_rows": [r["name"] for r in rows if r["op"] == "Split"], "slice_nodes": [n.name for n in m.graph.node if n.op_type == "Slice"], "split_nodes": [n.name for n in m.graph.node if n.op_type == "Split"]}
    # ---- M1
    m1 = {}
    for model, run in (("silu", "ppe5_yolov8n_baseline_100ep_s0"), ("relu", "ppe5_yolov8n_relu_100ep_s0")):
        files = sorted((experiment_dir(RUN.parent / run, "onboard_g4b")).glob("tier1_int8_CORE_0_1_2_sess*_layers.json"))
        lay = json.loads(files[0].read_text())["layers"]
        build_rows, build_shapes = first_table(RUN.parent / run / "logs" / sorted(p.name for p in (RUN.parent / run / "logs").glob("export_int8_*.log"))[-1])
        by_id = {r["id"]: r for r in build_rows}
        split, per_op = [], collections.defaultdict(lambda: [0, 0])
        for r in lay:
            if not r["workload"]:
                continue
            pcts = [float(p.rstrip("%")) for p in r["workload"].split("/")]
            is_split = sum(p > 0 for p in pcts) >= 2
            per_op[r["op"]][0 if is_split else 1] += 1
            if is_split:
                split.append(dict(id=r["id"], op=r["op"], name=r["name"], out_shape=build_shapes.get(r["name"]), workload=r["workload"], time_us=r["time_us"]))
        m1[model] = dict(session_file=files[0].name, split_rows=split, per_op={k: dict(split=v[0], not_split=v[1]) for k, v in per_op.items()},
                         workloads=sorted(collections.Counter(s["workload"] for s in split).items(), key=lambda kv: -kv[1]))
    out["m1"] = m1
    # ---- QAT rows
    qrows, _ = first_table(LOGS / "qat_train.log")
    prow, _ = first_table(LOGS / "build_ppe5_yolov8n_baseline_100ep_s0_split_int8.log")
    bad = [i for i, r in enumerate(qrows) if r["target"] == "NPU" and r["dtype"] != "INT8"]
    lo, hi = max(bad[0] - 3, 0), min(bad[-1] + 4, len(qrows))
    out["qat_rows"] = {"window": [dict(id=r["id"], op=r["op"], dtype=r["dtype"], name=r["name"]) for r in qrows[lo:hi]], "fp16_ids": [qrows[i]["id"] for i in bad],
                       "ptq_resize_rows": [dict(id=r["id"], op=r["op"], dtype=r["dtype"], name=r["name"]) for r in prow if r["op"] == "Resize"]}
    out["qat_rows"]["ptq_neighbours"] = [dict(id=r["id"], op=r["op"], dtype=r["dtype"], name=r["name"]) for rr in out["qat_rows"]["ptq_resize_rows"] for r in prow if abs(r["id"] - rr["id"]) <= 2]
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2e_pc.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# Phase 2e PC-only analysis", "", f"G7a ORT max abs diff vs (asymmetric, floor): {json.dumps(out['g7a']['max_abs_diff'])}", "",
          f"G10 onnx nodes {out['g10']['onnx_nodes']}, rknn rows {out['g10']['rknn_rows']}, onnx Reshape {onnx_ops['Reshape']}, rknn Reshape {rknn_ops['Reshape']}, onnx Split {onnx_ops['Split']} Slice {onnx_ops['Slice']}, rknn Split {rknn_ops['Split']}",
          f"Reshape nodes named in a row (contains): {out['g10']['reshape_nodes']}", f"Reshape nodes exact: {out['g10']['reshape_nodes_exact']}", f"Split and Slice nodes exact: {out['g10']['split_nodes_exact']}", "",
          *[f"M1 {k}: per_op {v['per_op']} workloads {v['workloads']}" for k, v in m1.items()], "",
          f"QAT FP16 ids {out['qat_rows']['fp16_ids']}"]
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2e_pc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
