"""CLI: X3 analysis, C API back to back (gap 0) against paced runs (gap 220 ms) in the six E2e cells, statements of
bench/onboard_x3/rule.txt.

    .venv/bin/python -m ai.automation.analyze_x3 <run_id>
Writes results/onboard_x3_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELLS = ["a4i4", "a0i0", "a4i0", "a0i4", "a1i0", "a1i4"]
E2E = {"a4i4": "s4i4", "a0i0": "s0i0", "a4i0": "s4i0", "a0i4": "s0i4", "a1i0": "s1i0", "a1i4": "s1i4"}


def irq_delta(pre, post):
    def tot(p):
        t = None
        for ln in Path(p).read_text().splitlines():
            v = [int(x) for x in ln.split()[1:9]]
            t = v if t is None else [a + b for a, b in zip(t, v)]
        return t
    return [b - a for a, b in zip(tot(pre), tot(post))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    d = REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "onboard_x3"
    order = CELLS + CELLS[::-1]
    blocks, block = [], 0
    for i, cell in enumerate(order):
        for gap in ((0, 220) if i < 6 else (220, 0)):
            block += 1
            rows = list(csv.DictReader(open(d / f"b{block}_{cell}_g{gap}.csv")))
            ms = [int(r["wall_us"]) / 1000 for r in rows]
            cpus = collections.Counter(int(r["cpu"]) for r in rows)
            app_top, app_n = cpus.most_common(1)[0]
            irqd = irq_delta(d / f"irq_pre_{block}_{cell}_g{gap}.txt", d / f"irq_post_{block}_{cell}_g{gap}.txt")
            irq_top = max(range(8), key=lambda i: irqd[i])
            realized = app_top == int(cell[1]) and app_n >= 0.99 * len(rows) and irq_top == int(cell[3]) and irqd[irq_top] >= 0.99 * sum(irqd)
            blocks.append(dict(block=block, cell=cell, gap=gap, n=len(ms), median=statistics.median(ms), std=statistics.pstdev(ms),
                               slow=sum(1 for x in ms if x >= 20.0) / len(ms), fast=sum(1 for x in ms if x < 17.0) / len(ms), realized=realized))
    onb = json.loads((REPO / "results" / f"onboard_{args.run_id}_int8_CORE_0.json").read_text())
    e2e = {}
    for b in onb["e2e"]["blocks"].values():
        e2e.setdefault(b["cond"], []).append(b["median"])
    e2d = {}
    for b in onb["e2d"]["blocks"].values():
        e2d.setdefault(b["cond"], []).append(b["median"])
    cells = {}
    for c in CELLS:
        bl = [b for b in blocks if b["cell"] == c]
        B = statistics.mean(b["median"] for b in bl if b["gap"] == 0)
        G = statistics.mean(b["median"] for b in bl if b["gap"] == 220)
        E = statistics.mean(e2e[E2E[c]])
        R = statistics.mean(e2d[c])
        comparable = abs(B - R) <= 0.25
        judged = comparable and abs(E - B) >= 0.25
        cells[c] = dict(B=B, G=G, E=E, R=R, comparable=comparable, judged=judged, f=(G - B) / (E - B) if judged else None, realized=all(b["realized"] for b in bl),
                        slow_gap220=[b["slow"] for b in bl if b["gap"] == 220], fast_gap220=[b["fast"] for b in bl if b["gap"] == 220])
    judged = [c for c, v in cells.items() if v["judged"]]
    if not all(v["realized"] for v in cells.values()):
        verdict = "chưa kết luận (khối không thực hiện đúng)"
    elif judged and all(0.5 <= cells[c]["f"] <= 1.5 for c in judged):
        verdict = "nhịp gọi 220 ms đưa thời gian C API về gần thời gian eval_perf ở mọi ô được xét"
    elif judged and all(abs(cells[c]["f"]) <= 0.25 for c in judged):
        verdict = "nhịp gọi 220 ms không giải thích chênh lệch giữa hai đường"
    else:
        verdict = "chưa kết luận"
    out = dict(blocks=blocks, cells=cells, judged=judged, verdict=verdict, progress=(d / "progress.txt").read_text(), rule=(d / "rule.txt").read_text())
    name = f"onboard_x3_{args.run_id}_int8_CORE_0"
    (REPO / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = [f"# X3: C API paced 220 ms against back to back, {args.run_id}", "", "Rule: bench/onboard_x3/rule.txt (written before the data).", "",
          "| cell | B gap 0 ms | G gap 220 ms | E eval_perf (E2e) ms | R E2d ms | comparable | judged | f | realized | slow at gap 220 | fast at gap 220 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {c} | {f(v['B'])} | {f(v['G'])} | {f(v['E'])} | {f(v['R'])} | {v['comparable']} | {v['judged']} | {f(v['f'], 2)} | {v['realized']} | {v['slow_gap220']} | {v['fast_gap220']} |" for c, v in cells.items()]
    md += ["", f"Verdict: {verdict}"]
    (REPO / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
