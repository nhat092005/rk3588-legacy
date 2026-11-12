"""Whole candidate detectors on the board, CORE_0 against CORE_0_1_2, measured with the on-board C bench (ai/board/bench/rknn_perf_bench.c).

    .venv/bin/python -m ai.research.all_core.board.board_measure build   --exp a4_board_3core --models-dir DIR   (PC: nets to .rknn, INT8 and FP16, random weights)
    ... copy models, rknn_perf_bench and board_run.sh to the board, lock the clock, run board_run.sh there, copy out/ back to RAW ...
    .venv/bin/python -m ai.research.all_core.board.board_measure analyze --exp a4_board_3core --raw RAW          (PC: results.json and report.md)
Experiments: a4_board_3core (10 nets, 640 input, widths 32/64/128/256) and a5_div3 (dense and mbconv at input 640, 576, 768 and widths 96/192/384, to test even 3-way splits).
The C bench needs no PC link and no root. Timing comes from RKNN_QUERY_PERF_RUN (NPU run duration). The per-layer WorkLoad table comes from
RKNN_QUERY_PERF_DETAIL, the same text as eval_perf with perf_debug=True.
Writes experiments/research/all_core/a4_board_3core/{rule.txt,results.json,report.md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import tempfile
from pathlib import Path

import torch

from ai.core.rknn.benchmark import parse_layers
from ai.research.all_core.audit.compile_audit import RESEARCH_DIR, child
from ai.research.all_core.candidates.nets import WIDTHS, Det

NETS_A4 = [("dense", "none", "relu"), ("dense", "none", "leakyrelu"), ("res", "none", "relu"), ("res", "none", "leakyrelu"),
        ("mbconv", "none", "relu6"), ("mbconv", "none", "relu"), ("uib", "none", "relu"),
        ("dense", "convt", "relu"), ("star", "none", "relu"), ("dense", "resize", "relu")]
VARIANTS_A5 = {"v0_640": (640, WIDTHS), "v1_640_w96": (640, (96, 96, 192, 384)), "v2_576": (576, WIDTHS), "v2_768": (768, WIDTHS), "v3_576_w96": (576, (96, 96, 192, 384))}
NETS_A5 = [("dense", "none", "relu"), ("mbconv", "none", "relu6")]
PRECISIONS = ("int8", "fp16")
MASKS = {1: "CORE_0", 3: "CORE_0_1", 7: "CORE_0_1_2"}  # rknn_core_mask values
RULE_A4 = """Statements measured by a4 (descriptive, no pass or fail threshold is set here).
S1. For each net, which layers of the CORE_0_1_2 per-layer table split over 2 or more cores (WorkLoad rule of analyze_g4.split_row), and which do not.
S2. NPU run duration (RKNN_QUERY_PERF_RUN, median of 3 repetitions x 100 runs after 20 warm-up runs) at core_mask 1, 3 and 7, and the ratios CORE_0 / CORE_0_1 and CORE_0 / CORE_0_1_2.
S3. The same for the YOLOv8n SiLU and ReLU models of the project runs, in the same session, as the reference.
S4. Weights of the nets are random and INT8 calibration data is random, so only structure and latency are read.
Setup: on-board C bench pinned to CPU 6 (taskset), zero input, clocks locked with lock_freq.sh lock before and released after, masks rotated between repetitions, board_run.sh in this folder.
"""


RULE_A5 = """Statements measured by a5 (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8 and FP16, random weights).
S1. Share of each layer over the 3 cores at CORE_0_1_2 for dense and mbconv ReLU nets with output height divisible by 3 at every stage (input 576 and 768) and with output channels in multiples of 96, against the 640 input and 32/64/128/256 widths (v0_640).
S2. Whether the largest share per layer falls toward 33.3 percent, the balance-limited speedup, and the measured CORE_0 / CORE_0_1_2 ratio.
Variants: v0_640 control, v1_640_w96 (widths 96/96/192/384), v2_576 and v2_768 (input only), v3_576_w96 (both). Nets differ in compute, so compare shares first, speedups with that in mind.
"""


RULE_A6 = """Statements measured by a6 (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8 and FP16, random weights).
S1. Same dense bottom-up ReLU backbone and neck at 640x640, three heads: v6 (HeadV6), stock (Ultralytics Detect reg_max=1, DWConv cls branch), stock_legacy (Conv3, Conv3 cls branch).
S2. For each: layers that split over 2 or more cores at CORE_0_1_2, unsplit ops, NPU run duration at the three core masks and the ratios.
"""


RULE_A7 = """Statements measured by a7 (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8 and FP16, random weights).
S1. The exact graphs that Ultralytics builds from ai/models/allcore_dense/allcore_dense.yaml and ai/models/allcore_mbconv/allcore_mbconv.yaml (trainable_audit.py writes the .rknn files): layers that split over 2 or more cores at CORE_0_1_2 and unsplit ops.
S2. NPU run duration at the three core masks and the ratios.
"""


RULE_A9 = """Statements measured by a9 (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8, random weights).
S1. The width x depth grid of scale_grid.py (dense all-core detector, 640x640): layers that split over 2 or more cores at CORE_0_1_2 and unsplit ops.
S2. NPU run duration at the three core masks and the ratios, with YOLOv8n ReLU and SiLU as references in the same session: the project exports (yolov8n_*_int8, DFL and decode included) and the raw-head cuts of raw_refs.py (yolov8n_*_raw, same graph end as the grid).
The budget of the study is the same-session ms of the YOLOv8n ReLU reference, not a stored number. 10.82 ms (a4) is context only. Sanity check of the references: the project exports should show about 111 (ReLU, 80 split) and 117 (SiLU, 29 split) layers as in a4.
"""


RULE_A9B = """Statements measured by a9b (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8, random weights).
S1. YOLOv8n ReLU and SiLU with nc=5 cut at the raw head tensors (raw_refs.py, no DFL, no decode) against the project exports (DFL and decode included), plus dense (1.0, 2) as a drift check against a9.
The raw cuts measured in a9 were built with nc=80 (stock yaml default) and are superseded by these.
"""


RULE_A9C = """Statements measured by a9c (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8, random weights).
S1. Five width x depth points near the raw-head YOLOv8n ReLU budget (scale_grid.py, GRID points added for a9c), with (1.5, 2) and (1.0, 2) of a9 and the YOLOv8n ReLU raw cut (nc=5) and project export in the same session.
S2. Layers that split over 2 or more cores, NPU run duration at the three core masks and the ratios.
"""


RULE_A9D = """Statements measured by a9d (descriptive, no threshold). Same setup as a4 (on-board C bench, CPU 6, clock locked, masks rotated, INT8, random weights).
S1. Dense (1.0, 10) (depth is cheap at nano width in a9c) with (1.5, 2) and the YOLOv8n ReLU raw cut (nc=5) in the same session. Layers that split, ms at the three core masks.
"""


def specs(exp: str) -> list[tuple[str, tuple[str, str, str], int, tuple]]:
    """(model name without precision, (family, neck, act), input size, widths)."""
    if exp == "a6_head":
        return [(f"dense_none_relu_{h}", (h, "none", "relu"), 640, WIDTHS) for h in ("v6", "stock", "stock_legacy")]
    if exp == "a5_div3":
        return [(f"{f}_{n}_{a}_{v}", (f, n, a), s, w) for f, n, a in NETS_A5 for v, (s, w) in VARIANTS_A5.items()]
    return [(f"{f}_{n}_{a}", (f, n, a), 640, WIDTHS) for f, n, a in NETS_A4]


def build(exp: str, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, (fam, neck, act), size, widths in specs(exp):
        if exp == "a6_head":
            from ai.research.all_core.audit.head_audit import build as build_head

            net, x = build_head(fam), torch.randn(1, 3, size, size)
        else:
            net, x = Det(fam, neck, act, widths=widths).eval(), torch.randn(1, 3, size, size)
        for prec in PRECISIONS:
            target = models_dir / f"{name}_{prec}.rknn"
            with tempfile.TemporaryDirectory() as tmp:
                onnx = Path(tmp) / "net.onnx"
                torch.onnx.export(net, x, str(onnx), opset_version=19, input_names=["images"])
                p = child("ai.research.all_core.audit._build", [str(onnx), prec, tmp, str(target)], tmp, 1800)
            if p.returncode != 0 or not target.exists():
                raise RuntimeError(f"build failed for {target.name}:\n{(p.stdout + p.stderr)[-1500:]}")
            print("built", target.name, f"{target.stat().st_size / 1e6:.1f} MB", flush=True)


def split_row(workload: str | None) -> bool:
    return workload is not None and sum(float(x.rstrip("%")) > 0 for x in workload.split("/")) >= 2


def read_latency(raw: Path, name: str, mask: int) -> dict:
    reps = []
    for rep in range(3):
        with (raw / f"{name}_m{mask}_r{rep}.csv").open() as f:
            reps.append([int(r["run_duration_us"]) / 1000 for r in csv.DictReader(f)])
    allv = [v for r in reps for v in r]
    return dict(ms=statistics.median(allv), rep_medians=[statistics.median(r) for r in reps], n=len(allv), min=min(allv), max=max(allv))


def read_layers(raw: Path, name: str, mask: int) -> list[dict]:
    """Per-layer rows of the detail runs: structure of the first run, time_us is the median over runs."""
    text = (raw / f"{name}_m{mask}_detail_detail.txt").read_text()
    tables = [parse_layers(t) for t in re.split(r"=== run \d+ ===", text) if t.strip()]
    rows = tables[0]
    for t in tables[1:]:
        assert [(r["id"], r["op"]) for r in t] == [(r["id"], r["op"]) for r in rows], f"layer table differs between runs ({name} mask {mask})"
    for i, r in enumerate(rows):
        r["time_us"] = statistics.median(t[i]["time_us"] for t in tables)
    return rows


def analyze(exp: str, raw: Path) -> None:
    OUT = RESEARCH_DIR / exp
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rule.txt").write_text({"a5_div3": RULE_A5, "a6_head": RULE_A6, "a7_trainable": RULE_A7, "a9_scale_grid": RULE_A9, "a9b_refs": RULE_A9B, "a9c_fine": RULE_A9C, "a9d_deep": RULE_A9D}.get(exp, RULE_A4))
    names = sorted(p.name[:-len(".done")] for p in raw.glob("*.done"))
    res = {}
    for n in names:
        res[n] = dict(latency={MASKS[m]: read_latency(raw, n, m) for m in MASKS}, layers={MASKS[m]: read_layers(raw, n, m) for m in (1, 7)})
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"# {exp}: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)", "",
             "Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). "
             "Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.", "",
             "| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        lat = {k: v["ms"] for k, v in r["latency"].items()}
        rows = [x for x in r["layers"]["CORE_0_1_2"] if x["op"] not in ("InputOperator", "OutputOperator")]
        split = [x for x in rows if split_row(x["workload"])]
        cyc, tm = sum(x["total_cycles"] or 0 for x in rows) or 1, sum(x["time_us"] or 0 for x in rows) or 1
        one = [x for x in r["layers"]["CORE_0"] if x["op"] not in ("InputOperator", "OutputOperator")]
        big = [max(float(v.rstrip("%")) for v in x["workload"].split("/")) / 100 if x["workload"] else 1.0 for x in rows]
        bal = sum(a["time_us"] * m for a, m in zip(one, big)) or 1
        even = sum(m <= 0.34 for m in big)
        unsplit: dict[str, int] = {}
        for x in rows:
            if not split_row(x["workload"]):
                unsplit[x["op"]] = unsplit.get(x["op"], 0) + 1
        model, prec = n.rsplit("_", 1)
        lines.append(f"| {model} | {prec} | {lat['CORE_0']:.3f} | {lat['CORE_0_1']:.3f} | {lat['CORE_0_1_2']:.3f} | {lat['CORE_0'] / lat['CORE_0_1']:.2f} | {lat['CORE_0'] / lat['CORE_0_1_2']:.2f} | {len(rows)} | {len(split)} | "
                     f"{sum(x['total_cycles'] or 0 for x in split) / cyc:.3f} | {sum(x['time_us'] or 0 for x in split) / tm:.3f} | {sum(a['time_us'] for a in one) / bal:.2f} | {even} of {len(rows)} | {unsplit or '-'} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "analyze"))
    ap.add_argument("--exp", default="a4_board_3core", choices=("a4_board_3core", "a5_div3", "a6_head", "a7_trainable", "a9_scale_grid", "a9b_refs", "a9c_fine", "a9d_deep"))
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--raw", type=Path)
    a = ap.parse_args()
    build(a.exp, a.models_dir) if a.cmd == "build" else analyze(a.exp, a.raw)


if __name__ == "__main__":
    main()
