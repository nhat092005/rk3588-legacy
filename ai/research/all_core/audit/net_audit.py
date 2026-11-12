"""Static audit of whole candidate detectors (backbone family x neck x activation), compiled on the PC, compared with YOLOv8n and YOLO26n.

    .venv/bin/python -m ai.research.all_core.audit.net_audit [--families dense,res] [--necks resize,none]
Writes experiments/research/all_core/a3_network_audit/{results.json,report.md}. The board is not used.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from ai.research.all_core.audit.compile_audit import RESEARCH_DIR, compile_table, summarize
from ai.research.all_core.candidates.nets import FAMILIES, NECKS, Det, complexity
from ai.research.all_core.audit.op_class import measured

OUT = RESEARCH_DIR / "a3_network_audit"


def audit_net(family: str, neck: str, act: str, precision: str, table: dict) -> dict:
    net, x = Det(family, neck, act).eval(), torch.randn(1, 3, 640, 640)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "net.onnx"
        torch.onnx.export(net, x, str(path), opset_version=19, input_names=["images"])
        rows = compile_table(path, precision)
    return dict(family=family, neck=neck, act=act, precision=precision, **complexity(net, x), **summarize(rows, table), rows=rows)


def reference_complexity(yaml_path: str) -> dict:
    from ultralytics import YOLO

    return complexity(YOLO(yaml_path).model, torch.randn(1, 3, 640, 640))


def write_report(results: list[dict], refs: list[dict]) -> None:
    lines = ["# Static audit of whole detectors (PC compile, no board, 640x640, nc=5)", "",
             "Params include BN. GFLOPs = 2 x MACs of Conv and ConvTranspose. Cycles are the compiler estimate, not latency.", "",
             "| net | prec | layers | params M | GFLOPs | cycles M | blockers (layers per op) | partial ops |", "|---|---|---|---|---|---|---|---|"]
    for r in refs:
        lines.append(f"| {r['block']} (reference) | {r['precision']} | {r['layers']} | {r['params'] / 1e6:.2f} | {r['gflops']} | {r['total_cycles'] / 1e6:.1f} | {r['blockers']} | {', '.join(sorted(r['by_class'].get('partial', {}))) or '-'} |")
    for r in results:
        lines.append(f"| {r['family']} {r['neck']} {r['act']} | {r['precision']} | {r['layers']} | {r['params'] / 1e6:.2f} | {r['gflops']} | {r['total_cycles'] / 1e6:.1f} | {r['blockers'] or '-'} | {', '.join(sorted(r['by_class'].get('partial', {}))) or '-'} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--necks", default=",".join(NECKS))
    ap.add_argument("--precision", default="fp16,int8")
    ap.add_argument("--reference-yaml", default="", help="comma list of YOLO yaml (same files as compile_audit --reference), for params and GFLOPs")
    args = ap.parse_args()
    table, results = measured(), []
    for fam in args.families.split(","):
        for neck in args.necks.split(","):
            for act in FAMILIES[fam][1]:
                for prec in args.precision.split(","):
                    r = audit_net(fam, neck, act, prec, table)
                    results.append(r)
                    print(f"{fam:7s} {neck:7s} {act:10s} {prec:5s} layers={r['layers']} params={r['params'] / 1e6:.2f}M gflops={r['gflops']} blockers={r['blockers']}", flush=True)
    refs = json.loads((RESEARCH_DIR / "a1_static_audit" / "reference.json").read_text())
    if args.reference_yaml:
        cx = {Path(y).stem: reference_complexity(y) for y in args.reference_yaml.split(",")}
        refs = [{**r, **cx[r["block"]]} for r in refs if r["block"] in cx]
    else:
        refs = []
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(dict(results=results, references=refs), indent=1) + "\n")
    write_report(results, refs)


if __name__ == "__main__":
    main()
