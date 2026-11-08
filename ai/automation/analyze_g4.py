"""CLI: gate G4 analysis (A coverage, B latency, S1 masks, G3 table, G6 and G8 workloads), statements of
bench/onboard_g4b/rule.txt. The helper log of the board is copied to bench/onboard_g4b/hold/ first.

    .venv/bin/python -m ai.automation.analyze_g4
Writes results/g4_latency_workload.{json,md}.
"""
from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

from ai.core.rknn.benchmark import parse_layers

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "ai" / "automation" / "runs"
RUN = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0"}
MASKS = ["CORE_0", "CORE_0_1", "CORE_0_1_2", "AUTO"]
SUB = "onboard_g4b"
LINES = ("36", "37", "38")
HOME = RUNS / RUN["silu"] / "bench" / SUB


def split_row(workload: str | None) -> bool:
    """Rule: a row is split across cores when at least 2 of the 3 WorkLoad percentages are above 0.0 percent."""
    return workload is not None and sum(float(x.rstrip("%")) > 0 for x in workload.split("/")) >= 2


def build_table(run: str, prec: str) -> list[dict]:
    logs = sorted((RUNS / run / "logs").glob(f"export_{prec}_*.log")) or sorted((RUNS / run / "logs").glob(f"export_q{8 if prec == 'int8' else 16}_noise.log"))
    text = "\n".join(re.sub(r"^.*?\] ", "", ln) for ln in logs[-1].read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln)
    start = text.index("Network Layer Information Table")
    rows = parse_layers(text[start:])
    # the log prints three tables, keep the first (ids restart at 0 when the second begins)
    first = []
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
        stem = f"tier1_int8_{e['mask']}_sess{e['session']}"
        bench = RUNS / e["run"] / "bench" / SUB
        rec = json.loads((bench / f"{stem}.json").read_text())
        lay = json.loads((bench / f"{stem}_layers.json").read_text())
        assert rec["model_sha256"] == e["sha256"]
        sess.append(dict(model=e["model"], mask=e["mask"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], latency_all=rec["npu_latency_ms_all"],
                         debug_total_ms=rec["debug_total_us"] / 1000, irq_per_cpu=delta, cpu_ops=rec["cpu_ops"], n_layers=rec["n_layers"], layers=lay["layers"]))
    out = dict(sessions=[{k: v for k, v in s.items() if k != "layers"} for s in sess], all_realized=all(s["realized"] for s in sess))

    # latency per condition
    cond = {}
    for m in RUN:
        for k in MASKS:
            ss = sorted((s for s in sess if s["model"] == m and s["mask"] == k), key=lambda s: s["session"])
            assert len(ss) == 2
            meds = [s["latency_ms"] for s in ss]
            cond[(m, k)] = dict(session_medians=meds, median=statistics.median(meds), gap_ms=abs(meds[0] - meds[1]), realized=all(s["realized"] for s in ss))
    out["conditions"] = {f"{m}|{k}": v for (m, k), v in cond.items()}
    out["gate_b"] = {}
    for m in RUN:
        base = cond[(m, "CORE_0")]
        for k in MASKS[1:3]:
            c = cond[(m, k)]
            drop_ms = base["median"] - c["median"]
            drop_pct = 100 * drop_ms / base["median"]
            gap = max(base["gap_ms"], c["gap_ms"])
            ok = base["realized"] and c["realized"]
            verdict = ("chưa kết luận (P2 không thực hiện đúng)" if not ok else ("nhanh hơn" if drop_pct >= 5 and drop_ms > gap else "không nhanh hơn"))
            out["gate_b"][f"{m}|{k}"] = dict(core0_ms=base["median"], mask_ms=c["median"], drop_ms=drop_ms, drop_pct=drop_pct, gap_ms=gap, verdict=verdict)
    auto = {}
    for m in RUN:
        c, base = cond[(m, "AUTO")], cond[(m, "CORE_0")]
        auto[m] = dict(auto_ms=c["median"], core0_ms=base["median"], diff_ms=c["median"] - base["median"], gap_ms=max(c["gap_ms"], base["gap_ms"]))
    out["auto_vs_core0"] = auto
    out["relu_vs_silu"] = {k: dict(silu_ms=cond[("silu", k)]["median"], relu_ms=cond[("relu", k)]["median"], diff_ms=cond[("relu", k)]["median"] - cond[("silu", k)]["median"],
                                   diff_pct=100 * (cond[("relu", k)]["median"] - cond[("silu", k)]["median"]) / cond[("silu", k)]["median"]) for k in MASKS}

    # G4A coverage and op breakdown, CORE_0_1_2 debug sessions
    out["gate_a"], out["by_op"], out["resize"] = {}, {}, {}
    for m in RUN:
        ss = [s for s in sess if s["model"] == m and s["mask"] == "CORE_0_1_2"]
        build = build_table(RUN[m], "int8")
        build_total = sum(r["total_cycles"] for r in build if r["total_cycles"])
        cov = []
        for s in ss:
            rows = [r for r in s["layers"] if r["total_cycles"]]
            assert sum(r["total_cycles"] for r in rows) == build_total, "board cycles differ from the build table"
            cov.append(sum(r["total_cycles"] for r in rows if split_row(r["workload"])) / build_total)
        out["gate_a"][m] = dict(coverage_per_session=cov, build_rows=len(build), build_total_cycles=build_total,
                                build_cpu_ops=sorted({r["op"] for r in build if r["target"] == "CPU"}),
                                verdict="đạt" if min(cov) >= 0.5 else "không đạt")
        by = collections.defaultdict(lambda: dict(rows=0, split=0, cycles=0, workloads=collections.Counter()))
        for r in ss[0]["layers"]:
            b = by[r["op"]]
            b["rows"] += 1
            b["split"] += split_row(r["workload"])
            b["cycles"] += r["total_cycles"] or 0
            b["workloads"][r["workload"]] += 1
        c0 = [x for x in sess if x["model"] == m and x["mask"] == "CORE_0"]
        row_t = {r["id"]: statistics.median(next(y for y in x["layers"] if y["id"] == r["id"])["time_us"] for x in c0) for r in c0[0]["layers"]}
        t_op = collections.Counter()
        for r in c0[0]["layers"]:
            t_op[r["op"]] += row_t[r["id"]]
        out["by_op"][m] = {op: dict(rows=b["rows"], split=b["split"], cycles=b["cycles"], workloads=dict(b["workloads"].most_common(4)),
                                    time_us_core0=t_op[op], time_share_core0=t_op[op] / sum(t_op.values())) for op, b in by.items()}
        rs = []
        for r0 in ss[0]["layers"]:
            if r0["op"] == "Resize":
                t3 = [next(x for x in s["layers"] if x["id"] == r0["id"])["time_us"] for s in ss]
                c0 = [next(x for x in s["layers"] if x["id"] == r0["id"])["time_us"] for s in sess if s["model"] == m and s["mask"] == "CORE_0"]
                rs.append(dict(id=r0["id"], name=r0["name"], workload=r0["workload"], time_us_core012=statistics.median(t3), time_us_core0=statistics.median(c0), cycles=r0["total_cycles"]))
        out["resize"][m] = rs
    out["g3"] = {}
    for m in RUN:
        out["g3"][m] = {f"{p}_cpu_ops": sorted({r["op"] for r in build_table(RUN[m], p) if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}) for p in ("int8", "fp16")}
        out["g3"][m].update({f"{p}_rows": len(build_table(RUN[m], p)) for p in ("int8", "fp16")})
    out["rule"] = (HOME / "rule.txt").read_text()
    out["driver_log"] = (HOME / "driver.log").read_text()
    out["hold_progress"] = (HOME / "hold" / "progress.txt").read_text()
    (REPO / "results" / "g4_latency_workload.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# G4 latency, workload and S1 masks (P2, interleaved)", "", f"All sessions realized: {out['all_realized']}", "", "| model | mask | session medians ms | median | gap ms |", "|---|---|---|---|---|"]
    md += [f"| {m} | {k} | {v['session_medians']} | {v['median']:.3f} | {v['gap_ms']:.3f} |" for (m, k), v in cond.items()]
    md += ["", "## B", ""] + [f"- {k}: drop {v['drop_ms']:.3f} ms ({v['drop_pct']:.2f}%), gap {v['gap_ms']:.3f} ms, {v['verdict']}" for k, v in out["gate_b"].items()]
    md += ["", "## A", ""] + [f"- {m}: coverage {v['coverage_per_session']}, {v['verdict']}" for m, v in out["gate_a"].items()]
    (REPO / "results" / "g4_latency_workload.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
