"""Class of each compiled RKNN op name, from the WorkLoad column measured on the board in this project (YOLOv8n variants).

Classes: multi (every measured row split over 2+ cores), partial (some rows), single (no row split), cpu (Target CPU),
io (Input/Output operator), unmeasured (name never seen in a measured table). No class is guessed from the OPL here.
"""
from __future__ import annotations

import json

from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR

SOURCE = OPERATOR_SUPPORT_RESULTS_DIR / "g4v_latency_workload_2.json"  # by_op: model -> op -> rows, split
# Not in the JSON above: YOLO26n T3 board measurement, docs/contents/09-tien-do/03-khoang-trong/06-yolo.rst
EXTRA = {"exSDPAttention": (2, 0), "ConvAdd": (4, 4)}


def measured() -> dict[str, tuple[int, int]]:
    """op name -> (rows, rows split over 2+ cores), summed over the model variants of the source file."""
    out: dict[str, list[int]] = {}
    for ops in json.loads(SOURCE.read_text())["by_op"].values():
        for op, e in ops.items():
            acc = out.setdefault(op, [0, 0])
            acc[0] += e["rows"]
            acc[1] += e["split"]
    out.update({k: list(v) for k, v in EXTRA.items()})
    return {k: (v[0], v[1]) for k, v in out.items()}


def classify(op: str, target: str, table: dict[str, tuple[int, int]]) -> str:
    if op in ("InputOperator", "OutputOperator"):
        return "io"
    if target == "CPU":
        return "cpu"
    if op not in table:
        return "unmeasured"
    rows, split = table[op]
    return "multi" if split == rows else "single" if split == 0 else "partial"
