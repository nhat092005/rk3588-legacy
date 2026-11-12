"""a9: YOLOv8n SiLU and ReLU (stock yaml with nc: 5, legacy head) cut at the raw head tensors, the same graph end as the all-core grid (no DFL, no decode).

    .venv/bin/python -m ai.research.all_core.grid.raw_refs --rknn-dir DIR
Run in its own process: this module must not import ai.research.all_core.ultra, which would swap in the non-legacy head. The layer count is checked
against the legacy graph (YOLOv8n ReLU: DFL and decode removed, so fewer layers than the 111 of the project export).
Writes experiments/research/all_core/a9_scale_grid/raw_refs.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ultralytics
from ai.models.yolov8n_relu_scratch import yolov8n_relu_scratch
from ai.research.all_core.audit.compile_audit import RESEARCH_DIR
from ai.research.all_core.audit.op_class import measured
from ai.research.all_core.audit.trainable_audit import audit

assert "ai.research.all_core.ultra" not in sys.modules, "ultra would swap in the non-legacy head"
STOCK = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
MODELS = {"yolov8n_silu_raw": STOCK, "yolov8n_relu_raw": Path(yolov8n_relu_scratch.WEIGHTS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn-dir", type=Path, required=True)
    a = ap.parse_args()
    table, res = measured(), []
    out = RESEARCH_DIR / "a9_scale_grid"
    (out / "yamls").mkdir(parents=True, exist_ok=True)
    for n, w in MODELS.items():
        src = w.read_text()
        assert "nc: 80 " in src, f"{w} has no nc: 80 line"
        y = out / "yamls" / f"{n.replace('_raw', '')}_raw_nc5.yaml"  # file name keeps 'yolov8n' so Ultralytics guesses scale n
        y.write_text(src.replace("nc: 80 ", "nc: 5 ", 1))
        r = audit(n, "int8", table, a.rknn_dir, weights=str(y))
        res.append(r)
        print(f"{n:18s} layers={r['layers']} params={r['params']} gflops={r['gflops']} blockers={r['blockers']} ops={r['ops']}", flush=True)
    (out / "raw_refs.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
