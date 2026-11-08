"""CLI: latency and coverage of the exported int8h5 against int8, statements of bench/onboard_q2i/rule.txt. The helper log is copied to bench/onboard_q2i/hold/ first.

    .venv/bin/python -m ai.automation.analyze_q2i
Writes results/q2i_latency.{json,md}.
"""
import collections
import json
import statistics
from pathlib import Path

from ai.automation.analyze_g4 import RUNS, build_table, split_row

REPO = Path(__file__).resolve().parents[2]
RUN = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0"}
HOME = RUNS / RUN["silu"] / "bench" / "onboard_q2i"
MASKS = ["CORE_0", "CORE_0_1_2"]
LINES = ("36", "37", "38")


def layer_table(run: str) -> list[dict]:
    from ai.core.rknn.benchmark import parse_layers
    import re

    log = RUNS / RUN["silu"] / "bench" / "onboard_q2c" / "logs" / f"build_{run}_h5.log"
    text = "\n".join(re.sub(r"^.*?\] ", "", ln) for ln in log.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln)
    rows, first = parse_layers(text[text.index("Network Layer Information Table"):]), []
    for r in rows:
        if first and r["id"] <= first[-1]["id"]:
            break
        first.append(r)
    return first


def main() -> None:
    entries = json.loads((HOME / "sessions.json").read_text())
    hlog = [ln.split() for ln in (HOME / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]
    sess = []
    for e in entries:
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        p = e["precision"]
        stem = f"tier1_{p}_{e['mask']}_sess{e['session']}"
        bench = RUNS / e["run"] / "bench" / "onboard_q2i"
        rec = json.loads((bench / f"{stem}.json").read_text())
        lay = json.loads((bench / f"{stem}_layers.json").read_text())
        assert rec["model_sha256"] == e["sha256"]
        sess.append(dict(model=e["model"], precision=p, mask=e["mask"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], layers=lay["layers"]))
    out = dict(sessions=[{k: x for k, x in s.items() if k != "layers"} for s in sess], all_realized=all(s["realized"] for s in sess))
    cond = {}
    for m in RUN:
        for p in ("int8", "int8h5"):
            for k in MASKS:
                ss = sorted((s for s in sess if s["model"] == m and s["precision"] == p and s["mask"] == k), key=lambda s: s["session"])
                assert len(ss) == 2
                meds = [s["latency_ms"] for s in ss]
                cond[(m, p, k)] = dict(session_medians=meds, median=statistics.median(meds), gap_ms=abs(meds[0] - meds[1]), realized=all(s["realized"] for s in ss))
    out["conditions"] = {"|".join(k): v for k, v in cond.items()}
    out["cost"], out["gate_b"], out["gate_a"], out["tables"] = {}, {}, {}, {}
    for m in RUN:
        for k in MASKS:
            b, h = cond[(m, "int8", k)], cond[(m, "int8h5", k)]
            diff = h["median"] - b["median"]
            gap = max(b["gap_ms"], h["gap_ms"])
            ok = b["realized"] and h["realized"]
            out["cost"][f"{m}|{k}"] = dict(int8_ms=b["median"], int8h5_ms=h["median"], cost_ms=diff, cost_pct=100 * diff / b["median"], gap_ms=gap,
                                          changes="chưa kết luận (P2 không thực hiện đúng)" if not ok else ("đổi latency" if abs(diff) > gap else "không đổi"), material=abs(diff) / b["median"] >= 0.05)
        for p in ("int8", "int8h5"):
            c0, c3 = cond[(m, p, "CORE_0")], cond[(m, p, "CORE_0_1_2")]
            drop = c0["median"] - c3["median"]
            gap = max(c0["gap_ms"], c3["gap_ms"])
            out["gate_b"][f"{m}|{p}"] = dict(core0_ms=c0["median"], mask_ms=c3["median"], drop_ms=drop, drop_pct=100 * drop / c0["median"], gap_ms=gap,
                                            verdict="chưa kết luận (P2 không thực hiện đúng)" if not (c0["realized"] and c3["realized"]) else ("nhanh hơn" if 100 * drop / c0["median"] >= 5 and drop > gap else "không nhanh hơn"))
            # the export logs of int8h5 were written with verbose off and hold no layer table: the table of the same graph and recipe is the build_recipe h5 log of onboard_q2c (the exported file has another sha256, same graph)
            build = build_table(RUN[m], p) if p == "int8" else layer_table(RUN[m])
            total = sum(r["total_cycles"] for r in build if r["total_cycles"])
            ss = [s for s in sess if s["model"] == m and s["precision"] == p and s["mask"] == "CORE_0_1_2"]
            cov = []
            for s in ss:
                rows = [r for r in s["layers"] if r["total_cycles"]]
                assert sum(r["total_cycles"] for r in rows) == total, "board cycles differ from the build table"
                cov.append(sum(r["total_cycles"] for r in rows if split_row(r["workload"])) / total)
            out["gate_a"][f"{m}|{p}"] = dict(coverage_per_session=cov, build_total_cycles=total, verdict="đạt" if min(cov) >= 0.5 else "không đạt")
            out["tables"][f"{m}|{p}"] = dict(rows=len(build), counts=dict(collections.Counter(f"{r['target']}|{r['dtype']}" for r in build)),
                                            cpu_ops=sorted({r["op"] for r in build if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}))
    out["rule"] = (HOME / "rule.txt").read_text()
    out["driver_log"] = (HOME / "driver.log").read_text()
    out["hold_progress"] = (HOME / "hold" / "progress.txt").read_text()
    (REPO / "results" / "q2i_latency.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# int8h5 latency cost (P2, interleaved)", "", f"All realized: {out['all_realized']}", ""]
    md += [f"- {k}: int8 {v['int8_ms']:.3f}, int8h5 {v['int8h5_ms']:.3f}, cost {v['cost_ms']:+.3f} ms ({v['cost_pct']:+.2f}%), gap {v['gap_ms']:.3f}, {v['changes']}, material {v['material']}" for k, v in out["cost"].items()]
    md += [f"- B {k}: {v['drop_pct']:.2f}% {v['verdict']}" for k, v in out["gate_b"].items()]
    md += [f"- A {k}: {min(v['coverage_per_session']):.3f} {v['verdict']}" for k, v in out["gate_a"].items()]
    (REPO / "results" / "q2i_latency.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
