"""a6: audit of the detection head. Same backbone and neck (dense, bottom-up neck, ReLU, 640x640, nc=5), three heads:
HeadV6 (audited in a3), the Ultralytics Detect(reg_max=1) branches non-legacy (DWConv cls) and legacy (Conv3, Conv3 cls).

    .venv/bin/python -m ai.research.all_core.audit.head_audit [--heads v6,stock,stock_legacy] [--precision fp16,int8] [--models-dir DIR]
Writes experiments/research/all_core/a6_head_audit/{results.json,report.md}. With --models-dir it also writes the ONNX files.
The graph ends at the raw per-level tensors: stock Detect's view and cat of cv2 and cv3 outputs (training only) are not part of the exported graph.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.nn.modules import head as uhead
from ultralytics.nn.modules.conv import Conv

from ai.research.all_core.audit.compile_audit import RESEARCH_DIR, compile_table, summarize
from ai.research.all_core.candidates.nets import Det, complexity
from ai.research.all_core.audit.op_class import measured

OUT = RESEARCH_DIR / "a6_head_audit"
HEADS = ("v6", "stock", "stock_legacy")


class StockRaw(nn.Module):
    """cv2 (box, 4 channels at reg_max=1) and cv3 (cls) of Ultralytics Detect, per level, raw NCHW outputs."""

    def __init__(self, legacy: bool, chs=(64, 128, 256), nc=5):
        super().__init__()
        Conv.default_act = nn.ReLU()
        uhead.Detect.legacy = legacy
        self.det = uhead.Detect(nc=nc, reg_max=1, end2end=False, ch=chs)

    def forward(self, p3, p4, p5):
        out = []
        for i, x in enumerate((p3, p4, p5)):
            out += [self.det.cv3[i](x), self.det.cv2[i](x)]
        return tuple(out)


def build(head: str) -> nn.Module:
    net = Det("dense", "none", "relu")
    if head != "v6":
        net.head = StockRaw(head == "stock_legacy")
    return net.eval()


def audit(head: str, precision: str, table: dict, models_dir: Path | None) -> dict:
    net, x = build(head), torch.randn(1, 3, 640, 640)
    with tempfile.TemporaryDirectory() as tmp:
        path = (models_dir or Path(tmp)) / f"dense_none_relu_{head}.onnx"
        torch.onnx.export(net, x, str(path), opset_version=19, input_names=["images"])
        rows = compile_table(path, precision)
    hp = sum(p.numel() for p in net.head.parameters())
    return dict(head=head, precision=precision, head_params=hp, **complexity(net, x), **summarize(rows, table), rows=rows)


def write_report(results: list[dict]) -> None:
    lines = ["# a6: head audit (PC compile, dense bottom-up ReLU 640x640, nc=5)", "",
             "| head | prec | layers | head params | params M | GFLOPs | cycles M | blockers | ops |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['head']} | {r['precision']} | {r['layers']} | {r['head_params']} | {r['params'] / 1e6:.2f} | {r['gflops']} | {r['total_cycles'] / 1e6:.1f} | {r['blockers'] or '-'} | {r['ops']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--precision", default="fp16,int8")
    ap.add_argument("--models-dir", default="")
    args = ap.parse_args()
    md = Path(args.models_dir) if args.models_dir else None
    if md:
        md.mkdir(parents=True, exist_ok=True)
    table, results = measured(), []
    for h in args.heads.split(","):
        for prec in args.precision.split(","):
            r = audit(h, prec, table, md)
            results.append(r)
            print(f"{h:13s} {prec:5s} layers={r['layers']} head_params={r['head_params']} gflops={r['gflops']} blockers={r['blockers']} ops={r['ops']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    write_report(results)


if __name__ == "__main__":
    main()
