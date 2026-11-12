"""Static audit of candidate blocks: compile each on the PC and classify every compiled op with measured board data.

    .venv/bin/python -m ai.research.all_core.audit.compile_audit [--blocks a,b] [--precision fp16,int8]
Writes experiments/research/all_core/a1_static_audit/{results.json,report.md}. The board is not used.
The compile table gives op name, target, shapes and cycles. It does not give the WorkLoad split, the multi-core
granularity rule or DW MAC use, which need the board.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from ai.paths import EXPERIMENTS_DIR, REPO_ROOT
from ai.research.all_core.candidates.blocks import REGISTRY, cb
from ai.research.all_core.audit.op_class import classify, measured

RESEARCH_DIR = EXPERIMENTS_DIR / "research" / "all_core"
ROW = re.compile(r"^(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(\S+)\s+(\S+)\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)\s*$")


def parse_table(text: str) -> list[dict]:
    """First 'Network Layer Information Table' of the compiler log."""
    text = re.sub(r"^[A-Z] RKNN: \[[\d:.]*\] ", "", text, flags=re.M)
    start = text.index("Network Layer Information Table")
    rows: list[dict] = []
    for line in text[start:].splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        i = int(m[1])
        if rows and i <= rows[-1]["id"]:
            break
        rows.append(dict(id=i, op=m[2], dtype=m[3], target=m[4], inp=m[5], out=m[6], ddr=int(m[7]), npu=int(m[8]), total=int(m[9]), name=m[11]))
    return rows


def child(module: str, args: list[str], work: str, timeout: int) -> subprocess.CompletedProcess:
    """Run a build child in the temp dir: rknn-toolkit2 and Ultralytics write check*.onnx and *_rknn_model/ to the working directory."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, cwd=work, env=env, timeout=timeout)


def compile_table(onnx_path: Path, precision: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as work:
        p = child("ai.research.all_core.audit._build", [str(onnx_path), precision, work], work, 1800)
    if p.returncode != 0:
        raise RuntimeError(f"build failed ({onnx_path.name}, {precision}):\n{(p.stdout + p.stderr)[-1500:]}")
    return parse_table(p.stdout + p.stderr)


def summarize(rows: list[dict], table: dict) -> dict:
    cls = [classify(r["op"], r["target"], table) for r in rows]
    for r, c in zip(rows, cls):
        r["class"] = c
    work = [r for r in rows if r["class"] != "io"]
    total = sum(r["total"] for r in work) or 1
    share = {c: sum(r["total"] for r in work if r["class"] == c) / total for c in ("multi", "partial", "single", "cpu", "unmeasured")}
    ops: dict[str, int] = {}
    for r in work:
        ops[r["op"]] = ops.get(r["op"], 0) + 1
    by_class: dict[str, dict[str, int]] = {}
    for r in work:
        by_class.setdefault(r["class"], {}).setdefault(r["op"], 0)
        by_class[r["class"]][r["op"]] += 1
    # cycle_share misses ops with 0 estimated cycles (Concat, Resize, MaxPool), so layers and blockers are the all-core test
    blockers = {op: n for c in ("single", "cpu", "unmeasured") for op, n in by_class.get(c, {}).items()}
    return dict(layers=len(work), cycle_share=share, ops=ops, by_class=by_class, blockers=blockers,
                cpu_ops=sorted(by_class.get("cpu", {})), unmeasured_ops=sorted(by_class.get("unmeasured", {})), total_cycles=sum(r["total"] for r in work))


class Ctx(nn.Module):
    """Put a conv before a single-input block, so a residual Add sees a conv output as in a real network (the compiler fuses it into the conv)."""

    def __init__(self, block, c):
        super().__init__()
        self.pre, self.block = cb(c, c, 1, act="relu"), block

    def forward(self, x):
        return self.block(self.pre(x))


def audit_block(name: str, act, c: int, hw: int, precision: str, table: dict) -> dict:
    factory, inputs, _, _ = REGISTRY[name]
    shapes = inputs(c, hw)
    model = factory(c, act)
    ctx = len(shapes) == 1
    model = (Ctx(model, c) if ctx else model).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.onnx"
        torch.onnx.export(model, tuple(torch.randn(*s) for s in shapes), str(path), opset_version=19, input_names=[f"x{i}" for i in range(len(shapes))])
        rows = [r for r in compile_table(path, precision) if not (ctx and "/pre/" in r["name"])]
    return dict(block=name, act=act, c=c, hw=hw, precision=precision, inputs=shapes, **summarize(rows, table), rows=rows)


def audit_reference(yaml_path: str, precision: str, table: dict) -> dict:
    """Compile a YOLO yaml (random weights) through Ultralytics' RKNN export path and classify the table. Used to check the audit against board counts."""
    with tempfile.TemporaryDirectory() as work:
        p = child("ai.research.all_core.audit._build_ref", [yaml_path, precision, work], work, 3600)
    if p.returncode != 0:
        raise RuntimeError(f"reference build failed ({yaml_path}, {precision}):\n{(p.stdout + p.stderr)[-1500:]}")
    rows = parse_table(p.stdout + p.stderr)
    return dict(block=Path(yaml_path).stem, precision=precision, **summarize(rows, table), rows=rows)


def write_report(results: list[dict], out: Path) -> None:
    """One line per block and activation. stable = same compiled ops at every size and in FP16 and INT8."""
    lines = ["# Static audit of candidate blocks (PC compile, no board)", "",
             "| block | act | compiled ops | blockers (not measured multi-core) | partial ops | stable over sizes and precisions |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["block"], str(r["act"])) for r in results}, key=lambda k: list(REGISTRY).index(k[0]))
    for b, a in keys:
        rs = [r for r in results if r["block"] == b and str(r["act"]) == a]
        first = rs[0]
        stable = all(r["ops"] == first["ops"] for r in rs)
        partial = ", ".join(sorted(first["by_class"].get("partial", {}))) or "-"
        lines.append(f"| {b} | {a} | {first['ops']} | {first['blockers'] or '-'} | {partial} | {'yes' if stable else 'NO'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="", help="comma list of YOLO yaml files (nc=5, scale from file name), audited instead of the blocks")
    ap.add_argument("--blocks", default=",".join(REGISTRY))
    ap.add_argument("--precision", default="fp16,int8")
    ap.add_argument("--exp", default="a1_static_audit", help="experiment folder under experiments/research/all_core/")
    args = ap.parse_args()
    table = measured()
    OUT = RESEARCH_DIR / args.exp
    if args.reference:
        refs = [audit_reference(y, prec, table) for y in args.reference.split(",") for prec in args.precision.split(",")]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "reference.json").write_text(json.dumps(refs, indent=1) + "\n")
        for r in refs:
            print(r["block"], r["precision"], "layers", r["layers"], "ops", r["ops"], "cpu", r["cpu_ops"], flush=True)
        return
    results = []
    for name in args.blocks.split(","):
        _, _, acts, sizes = REGISTRY[name]
        for act in acts:
            for size, (c, hw) in sizes.items():
                for prec in args.precision.split(","):
                    r = audit_block(name, act, c, hw, prec, table)
                    r["size"] = size
                    results.append(r)
                    s = r["cycle_share"]
                    print(f"{name:14s} {str(act):10s} {size:4s} {prec:5s} multi={s['multi']:.3f} partial={s['partial']:.3f} single={s['single']:.3f} cpu={s['cpu']:.3f} unmeasured={s['unmeasured']:.3f} blockers={r['blockers']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(dict(measured_table={k: list(v) for k, v in table.items()}, results=results), indent=1) + "\n")
    write_report(results, OUT)


if __name__ == "__main__":
    main()
