"""a7: compile the exact graph Ultralytics builds from an ai/models yaml (the one that gets trained) and classify it like a3 and a6.

    .venv/bin/python -m ai.research.all_core.audit.trainable_audit [--models allcore_dense,allcore_mbconv] [--precision fp16,int8] [--rknn-dir DIR]
Graph = backbone, neck and the per-level cls and box convs of the Detect head (raw NCHW, no reshape, no sigmoid, no decode).
Writes experiments/research/all_core/a7_trainable_audit/{results.json,report.md}. With --rknn-dir it also writes the .rknn files for the board.
"""
from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

from ai.research.all_core.audit.compile_audit import RESEARCH_DIR, child, compile_table, summarize
from ai.research.all_core.candidates.nets import complexity
from ai.research.all_core.audit.op_class import measured

OUT = RESEARCH_DIR / "a7_trainable_audit"


class Raw(nn.Module):
    """Run the layers of an Ultralytics DetectionModel and return the raw per-level (cls, box) tensors of its Detect head."""

    def __init__(self, dm):
        super().__init__()
        self.dm = dm

    def forward(self, x):
        y = []
        for layer in self.dm.model[:-1]:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x)
        det = self.dm.model[-1]
        feats = [y[i] for i in det.f]
        out = []
        for i, f in enumerate(feats):
            out += [det.cv3[i](f), det.cv2[i](f)]
        return tuple(out)


def audit(name: str, precision: str, table: dict, rknn_dir: Path | None, weights: str | None = None) -> dict:
    """weights: path of a yaml to build instead of ai.models.<name>.<name> (scale_grid.py). The caller must have imported ai.research.all_core.ultra."""
    dm = YOLO(weights or importlib.import_module(f"ai.models.{name}.{name}").WEIGHTS).model.eval()
    net, x = Raw(dm).eval(), torch.randn(1, 3, 640, 640)
    with tempfile.TemporaryDirectory() as tmp:
        onnx = Path(tmp) / f"{name}.onnx"
        torch.onnx.export(net, x, str(onnx), opset_version=19, input_names=["images"])
        if rknn_dir is not None:
            rknn_dir.mkdir(parents=True, exist_ok=True)
            target = rknn_dir / f"{name}_{precision}.rknn"
            p = child("ai.research.all_core.audit._build", [str(onnx), precision, tmp, str(target)], tmp, 1800)
            if p.returncode != 0 or not target.exists():
                raise RuntimeError(f"build failed for {target.name}:\n{(p.stdout + p.stderr)[-1500:]}")
        rows = compile_table(onnx, precision)
    return dict(model=name, precision=precision, **complexity(net, x), **summarize(rows, table), rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="allcore_dense,allcore_mbconv")
    ap.add_argument("--precision", default="fp16,int8")
    ap.add_argument("--rknn-dir", type=Path)
    a = ap.parse_args()
    table, res = measured(), []
    for n in a.models.split(","):
        for prec in a.precision.split(","):
            r = audit(n, prec, table, a.rknn_dir)
            res.append(r)
            print(f"{n:16s} {prec:5s} layers={r['layers']} params={r['params']} gflops={r['gflops']} cycles={r['total_cycles']} blockers={r['blockers']} ops={r['ops']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# a7: exact trainable graphs (PC compile, 640x640, nc=5)", "", "| model | prec | layers | params | GFLOPs | cycles M | blockers | ops |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['model']} | {r['precision']} | {r['layers']} | {r['params']} | {r['gflops']} | {r['total_cycles'] / 1e6:.1f} | {r['blockers'] or '-'} | {r['ops']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
