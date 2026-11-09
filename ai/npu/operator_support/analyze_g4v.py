"""CLI: activation variants (SiLU control, ReLU, LeakyReLU, ReLU6, ReLU 200 epoch) latency, coverage and G3 table, statements of
experiments/npu/operator_support/onboard_g4v/ppe5_yolov8n_baseline_100ep_s0/rule.txt. The helper log of the board is copied to experiments/npu/operator_support/onboard_g4v/ppe5_yolov8n_baseline_100ep_s0/hold/ first.

    .venv/bin/python -m ai.npu.operator_support.analyze_g4v
Writes results/npu/operator_support/g4v_latency_workload.{json,md}. Kept apart from analyze_g4.py so results/npu/operator_support/g4_latency_workload.json is not touched.
"""
from __future__ import annotations

import collections
import json
import statistics

from ai.npu.operator_support.analyze_g4 import RUNS, build_table, split_row
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, experiment_dir

RUN = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0", "leakyrelu": "ppe5_yolov8n_leakyrelu_100ep_s0",
       "relu6": "ppe5_yolov8n_relu6_100ep_s0", "relu200": "ppe5_yolov8n_relu_200ep_s0"}
MASKS = ["CORE_0", "CORE_0_1_2"]
import argparse

SUB = "onboard_g4v"
LINES = ("36", "37", "38")
HOME = experiment_dir(RUNS / RUN["silu"], SUB)


def main() -> None:
    global SUB, HOME
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="onboard_g4v", help="bench subdirectory, onboard_g4v2 is experiment 2")
    SUB = ap.parse_args().sub
    HOME = experiment_dir(RUNS / RUN["silu"], SUB)
    suffix = "" if SUB == "onboard_g4v" else "_" + SUB.removeprefix("onboard_g4v")
    entries = json.loads((HOME / "sessions.json").read_text())
    hlog = [ln.split() for ln in (HOME / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]
    sess = []
    for e in entries:
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        stem = f"tier1_int8_{e['mask']}_sess{e['session']}"
        bench = experiment_dir(RUNS / e["run"], SUB)
        rec = json.loads((bench / f"{stem}.json").read_text())
        lay = json.loads((bench / f"{stem}_layers.json").read_text())
        assert rec["model_sha256"] == e["sha256"]
        sess.append(dict(model=e["model"], mask=e["mask"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], latency_all=rec["npu_latency_ms_all"],
                         irq_per_cpu=delta, sha256=e["sha256"], layers=lay["layers"]))
    out = dict(sessions=[{k: v for k, v in s.items() if k != "layers"} for s in sess], all_realized=all(s["realized"] for s in sess))
    cond = {}
    for m in RUN:
        for k in MASKS:
            ss = sorted((s for s in sess if s["model"] == m and s["mask"] == k), key=lambda s: s["session"])
            assert len(ss) == 2
            meds = [s["latency_ms"] for s in ss]
            cond[(m, k)] = dict(session_medians=meds, median=statistics.median(meds), gap_ms=abs(meds[0] - meds[1]), realized=all(s["realized"] for s in ss))
    out["conditions"] = {f"{m}|{k}": v for (m, k), v in cond.items()}
    out["gate_b"], out["vs_silu"] = {}, {}
    for m in RUN:
        base, c = cond[(m, "CORE_0")], cond[(m, "CORE_0_1_2")]
        drop_ms = base["median"] - c["median"]
        drop_pct = 100 * drop_ms / base["median"]
        gap = max(base["gap_ms"], c["gap_ms"])
        ok = base["realized"] and c["realized"]
        out["gate_b"][m] = dict(core0_ms=base["median"], mask_ms=c["median"], drop_ms=drop_ms, drop_pct=drop_pct, gap_ms=gap,
                                verdict="chưa kết luận (P2 không thực hiện đúng)" if not ok else ("nhanh hơn" if drop_pct >= 5 and drop_ms > gap else "không nhanh hơn"))
    for k in MASKS:
        out["vs_silu"][k] = {m: dict(ms=cond[(m, k)]["median"], diff_ms=cond[(m, k)]["median"] - cond[("silu", k)]["median"],
                                     diff_pct=100 * (cond[(m, k)]["median"] - cond[("silu", k)]["median"]) / cond[("silu", k)]["median"]) for m in RUN}
    out["gate_a"], out["by_op"], out["g3"] = {}, {}, {}
    for m in RUN:
        ss = [s for s in sess if s["model"] == m and s["mask"] == "CORE_0_1_2"]
        build = build_table(RUN[m], "int8")
        build_total = sum(r["total_cycles"] for r in build if r["total_cycles"])
        cov = []
        for s in ss:
            rows = [r for r in s["layers"] if r["total_cycles"]]
            assert sum(r["total_cycles"] for r in rows) == build_total, "board cycles differ from the build table"
            cov.append(sum(r["total_cycles"] for r in rows if split_row(r["workload"])) / build_total)
        out["gate_a"][m] = dict(coverage_per_session=cov, build_rows=len(build), build_total_cycles=build_total, verdict="đạt" if min(cov) >= 0.5 else "không đạt")
        by = collections.defaultdict(lambda: dict(rows=0, split=0, cycles=0))
        for r in ss[0]["layers"]:
            b = by[r["op"]]
            b["rows"] += 1
            b["split"] += split_row(r["workload"])
            b["cycles"] += r["total_cycles"] or 0
        out["by_op"][m] = dict(by)
        out["g3"][m] = {f"{p}_cpu_ops": sorted({r["op"] for r in build_table(RUN[m], p) if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}) for p in ("int8", "fp16")}
        out["g3"][m].update({f"{p}_rows": len(build_table(RUN[m], p)) for p in ("int8", "fp16")})
        out["g3"][m]["fused_ops"] = {op: v["rows"] for op, v in by.items() if op.startswith("Conv") and op != "Conv"}
    out["rule"] = (HOME / "rule.txt").read_text()
    out["driver_log"] = (HOME / "driver.log").read_text()
    out["hold_progress"] = (HOME / "hold" / "progress.txt").read_text()
    (OPERATOR_SUPPORT_RESULTS_DIR / f"g4v_latency_workload{suffix}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# Activation variants latency, coverage, G3 (P2, interleaved)", "", f"All sessions realized: {out['all_realized']}", "", "| model | CORE_0 ms | CORE_0_1_2 ms | drop % | gap ms | B | A coverage | CPU ops |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {m} | {v['core0_ms']:.3f} | {v['mask_ms']:.3f} | {v['drop_pct']:.2f} | {v['gap_ms']:.3f} | {v['verdict']} | {min(out['gate_a'][m]['coverage_per_session']):.3f} | {out['g3'][m]['int8_cpu_ops']} |"
           for m, v in out["gate_b"].items()]
    (OPERATOR_SUPPORT_RESULTS_DIR / f"g4v_latency_workload{suffix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
