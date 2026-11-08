"""CLI: X5 analysis, run time against input values and weight values in cell a4i4, statements of bench/onboard_x5/rule.txt.

    .venv/bin/python -m ai.automation.analyze_x5 <run_id>
Writes results/onboard_x5_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from ai.automation.analyze_x3 import irq_delta

REPO = Path(__file__).resolve().parents[2]
HALF = ["s0-zeros", "s0-random", "s0-img1", "s0-img2", "s1-zeros", "s1-img1", "s0-zerosb"]


def verdict(diffs, what):
    if all(abs(x) <= 0.05 for pair in diffs for x in pair):
        return f"thời gian không phụ thuộc giá trị {what}"
    if any(all(abs(x) > 0.25 for x in pair) for pair in diffs):
        return f"thời gian phụ thuộc giá trị {what}"
    return "chưa kết luận"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    d = REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "onboard_x5"
    order = HALF + HALF[::-1]
    blocks = []
    for block, c in enumerate(order, 1):
        rows = list(csv.DictReader(open(d / f"b{block}_{c}.csv")))
        ms = [int(r["wall_us"]) / 1000 for r in rows]
        cpus = collections.Counter(int(r["cpu"]) for r in rows)
        top, n = cpus.most_common(1)[0]
        irqd = irq_delta(d / f"irq_pre_{block}_{c}.txt", d / f"irq_post_{block}_{c}.txt")
        realized = top == 4 and n >= 0.99 * len(rows) and irqd[4] >= 0.99 * sum(irqd)
        blocks.append(dict(block=block, cond=c, half=1 if block <= 7 else 2, median=statistics.median(ms), std=statistics.pstdev(ms), realized=realized,
                           api_line=(d / f"b{block}_{c}.log").read_text().splitlines()[0]))
    med = {(b["cond"], b["half"]): b["median"] for b in blocks}
    refm = {h: (med[("s0-zeros", h)] + med[("s0-zerosb", h)]) / 2 for h in (1, 2)}
    drift = [abs(med[("s0-zeros", h)] - med[("s0-zerosb", h)]) for h in (1, 2)]
    diff = lambda c, ref: [med[(c, h)] - (refm[h] if ref == "s0-zeros" else med[(ref, h)]) for h in (1, 2)]  # noqa: E731
    ok = all(b["realized"] for b in blocks) and all(x <= 0.05 for x in drift)
    inputs = {c: diff(c, "s0-zeros") for c in ("s0-random", "s0-img1", "s0-img2")}
    weights = {"s1-zeros vs s0-zeros": diff("s1-zeros", "s0-zeros"), "s1-img1 vs s0-img1": diff("s1-img1", "s0-img1")}
    out = dict(blocks=blocks, all_realized=ok, inputs=inputs, weights=weights,
               inputs_verdict=verdict(list(inputs.values()), "đầu vào") + (" trong các đầu vào đã thử" if verdict(list(inputs.values()), "đầu vào").startswith("thời gian không") else "") if ok else ("không áp dụng (trôi giữa các khối vượt 0.05 ms)" if all(b["realized"] for b in blocks) else "chưa kết luận (khối không thực hiện đúng)"),
               weights_verdict=verdict(list(weights.values()), "trọng số") + (" trong hai mô hình đã thử" if verdict(list(weights.values()), "trọng số").startswith("thời gian không") else "") if ok else ("không áp dụng (trôi giữa các khối vượt 0.05 ms)" if all(b["realized"] for b in blocks) else "chưa kết luận (khối không thực hiện đúng)"),
               drift_ms=drift, progress=(d / "progress.txt").read_text(), rule=(d / "rule.txt").read_text(), inputs_file=(d / "inputs" / "inputs.txt").read_text())
    name = f"onboard_x5_{args.run_id}_int8_CORE_0"
    (REPO / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = [f"# X5: inputs and weights, {args.run_id}", "", "| block | cond | half | median ms | std ms | realized |", "|---|---|---|---|---|---|"]
    md += [f"| {b['block']} | {b['cond']} | {b['half']} | {b['median']:.3f} | {b['std']:.3f} | {b['realized']} |" for b in blocks]
    md += ["", f"Inputs (difference vs s0-zeros, ms, half 1 and 2): {inputs}", f"Verdict: {out['inputs_verdict']}", "",
           f"Weights: {weights}", f"Verdict: {out['weights_verdict']}"]
    (REPO / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
