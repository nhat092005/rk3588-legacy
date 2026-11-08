"""CLI: phase 2e board results: A2 split head (rule_nodfl_split.txt), micro models G7b and G11 (rule_micro.txt), QAT NPU time and coverage
(onboard_q2e_qat/rule.txt) and RKNNLite probes M4, M6, M7, M16 (rule_probe.txt). The helper log of the QAT run is copied to onboard_q2e_qat/hold/ first.

    .venv/bin/python -m ai.automation.analyze_q2e_board
Writes results/q2e_board.json and results/q2e_board.md.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from ai.automation.analyze_g4 import RUNS, split_row

REPO = Path(__file__).resolve().parents[2]
RUN = "ppe5_yolov8n_baseline_100ep_s0"
A2 = "ppe5_yolov8n_nodfl_100ep_s0"
Q2C = RUNS / RUN / "bench" / "onboard_q2c"
Q2E = RUNS / RUN / "bench" / "onboard_q2e"
QAT = RUNS / RUN / "bench" / "onboard_q2e_qat"
MASKS = ["CORE_0", "CORE_0_1_2"]
LINES = ("36", "37", "38")
NAMES = ["int8h5", "split", "qat"]


def table_from_log(log: Path) -> list[dict]:
    from ai.core.rknn.benchmark import parse_layers

    text = "\n".join(re.sub(r"^.*?\] ", "", ln) for ln in log.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln)
    rows, first = parse_layers(text[text.index("Network Layer Information Table"):]), []
    for r in rows:
        if first and r["id"] <= first[-1]["id"]:
            break
        first.append(r)
    return first


def ev(run: str, name: str) -> dict:
    d = json.loads((RUNS / run / "eval" / f"{name}_css_test.json").read_text())
    return dict(map50=round(100 * d["map50"], 2), map50_95=round(100 * d["map50_95"], 2), NoHelmet=round(100 * d["per_class"]["NoHelmet"]["ap50_95"], 2),
                Vest=round(100 * d["per_class"]["Vest"]["ap50_95"], 2), sha256=d["model_sha256"])


def a2_split() -> dict:
    fp16s, int8s, fp16, int8 = (ev(A2, n) for n in ("rknn_fp16_split", "rknn_int8_split", "rknn_fp16", "rknn_int8"))
    rec = json.loads((RUNS / A2 / "weights" / "rknn_int8_split" / "recipe.json").read_text())
    rec16 = json.loads((RUNS / A2 / "weights" / "rknn_fp16_split" / "recipe.json").read_text())
    a1 = json.loads((RUNS / RUN / "weights" / "rknn_int8_split" / "recipe.json").read_text())
    tab = table_from_log(Q2C / "logs" / f"build_{A2}_split_int8.log")
    crit = round(fp16["map50_95"] - 1.35, 2)
    return dict(fp16_split=fp16s, int8_split=int8s, fp16=fp16, int8=int8,
                v_pc=dict(int8=rec["onnx_check_max_abs_vs_standard"], fp16=rec16["onnx_check_max_abs_vs_standard"], ok=max(rec["onnx_check_max_abs_vs_standard"], rec16["onnx_check_max_abs_vs_standard"]) <= 0.01),
                v_board=dict(diff=round(fp16s["map50_95"] - fp16["map50_95"], 2), ok=abs(fp16s["map50_95"] - fp16["map50_95"]) <= 1.35),
                n2_1=dict(minus_fp16_split=round(int8s["map50_95"] - fp16s["map50_95"], 2), minus_fp16=round(int8s["map50_95"] - fp16["map50_95"], 2), minus_int8=round(int8s["map50_95"] - int8["map50_95"], 2),
                          criterion=crit, meets=int8s["map50_95"] >= crit),
                outputs=rec["onnx_outputs"], same_calibration_list_as_a1=rec["dataset_md5"] == a1["dataset_md5"],
                layers=dict(rows=len(tab), counts={k: sum(1 for r in tab if f"{r['target']}|{r['dtype']}" == k) for k in sorted({f"{r['target']}|{r['dtype']}" for r in tab})},
                            cpu_ops=sorted({r["op"] for r in tab if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"})))


def qat_time() -> dict:
    entries = json.loads((QAT / "sessions.json").read_text())
    hlog = [ln.split() for ln in (QAT / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]
    sess = []
    for e in entries:
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        name = e["variant"] if e["variant"] else e["precision"]
        stem = f"tier1_{e['precision']}{'_' + e['variant'] if e['variant'] else ''}_{e['mask']}_sess{e['session']}"
        rec = json.loads((QAT / f"{stem}.json").read_text())
        lay = json.loads((QAT / f"{stem}_layers.json").read_text())
        assert rec["model_sha256"] == e["sha256"]
        sess.append(dict(name=name, mask=e["mask"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], layers=lay["layers"], sha256=e["sha256"]))
    cond = {}
    for n in NAMES:
        for k in MASKS:
            ss = sorted((s for s in sess if s["name"] == n and s["mask"] == k), key=lambda s: s["session"])
            assert len(ss) == 2
            meds = [s["latency_ms"] for s in ss]
            cond[(n, k)] = dict(session_medians=meds, median=statistics.median(meds), gap_ms=abs(meds[0] - meds[1]), realized=all(s["realized"] for s in ss))
    cmp_ = {}
    for k in MASKS:
        for other in ("split", "int8h5"):
            q, o = cond[("qat", k)], cond[(other, k)]
            diff = q["median"] - o["median"]
            gap = max(q["gap_ms"], o["gap_ms"])
            real = q["realized"] and o["realized"]
            cmp_[f"qat_vs_{other}|{k}"] = dict(qat_ms=q["median"], other_ms=o["median"], diff_ms=diff, diff_pct=100 * diff / o["median"], gap_ms=gap, realized=real,
                                               verdict="chưa kết luận (P2 không thực hiện đúng)" if not real else ("khác" if abs(diff) >= 0.05 * o["median"] and abs(diff) > gap else "không phân biệt được"))
    dec = json.loads((RUNS / RUN / "bench" / "onboard_q2j" / "decode_bench.json").read_text())["median_ms"]
    total = {k: dict(qat_npu_ms=cond[("qat", k)]["median"], decode_ms=dec, qat_total_ms=cond[("qat", k)]["median"] + dec, int8h5_ms=cond[("int8h5", k)]["median"],
                     diff_vs_int8h5_ms=cond[("qat", k)]["median"] + dec - cond[("int8h5", k)]["median"]) for k in MASKS}
    tab = table_from_log(Q2C / "logs" / "qat_train.log")
    qs = [s for s in sess if s["name"] == "qat" and s["mask"] == "CORE_0_1_2"]
    cov = []
    for s in qs:
        rows = [r for r in s["layers"] if r["total_cycles"]]
        tot = sum(r["total_cycles"] for r in rows)
        cov.append(dict(total_cycles=tot, build_total_cycles=sum(r["total_cycles"] or 0 for r in tab), coverage=sum(r["total_cycles"] for r in rows if split_row(r["workload"])) / tot))
    board_counts = {}
    for r in qs[0]["layers"]:
        key = f"{r['target']}|{r['dtype']}"
        board_counts[key] = board_counts.get(key, 0) + 1
    return dict(all_realized=all(s["realized"] for s in sess), sessions=[{k: v for k, v in s.items() if k != "layers"} for s in sess],
                conditions={"|".join(k): v for k, v in cond.items()}, comparisons=cmp_, total=total, coverage_qat=cov, board_table_counts=board_counts,
                rule=(QAT / "rule.txt").read_text(), driver_log=(QAT / "driver.log").read_text(), hold_progress=(QAT / "hold" / "progress.txt").read_text())


def main() -> None:
    out = dict(a2_split=a2_split(), micro=json.loads((REPO / "results" / "q2e_micro.json").read_text()))
    out["a2_split"]["rule"] = (Q2C / "rule_nodfl_split.txt").read_text()
    if (QAT / "sessions.json").exists() and (QAT / "hold" / "log.txt").exists():
        out["qat_time"] = qat_time()
    probes = {}
    for p in ("m4", "m4c", "m6", "m7", "m16"):
        f = Q2E / f"probe_{p}.json"
        if f.exists():
            probes[p] = json.loads(f.read_text())
    out["probes"] = probes
    (REPO / "results" / "q2e_board.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    a = out["a2_split"]
    md = ["# Phase 2e board results", "", f"A2 split: V-PC {a['v_pc']}, V-board {a['v_board']}, N2-1 {a['n2_1']}, layers {a['layers']}"]
    if "qat_time" in out:
        q = out["qat_time"]
        md += ["", f"QAT time: all realized {q['all_realized']}"] + [f"- {k}: {v}" for k, v in q["comparisons"].items()] + [f"- total {k}: {v}" for k, v in q["total"].items()] + [f"- coverage {q['coverage_qat']}", f"- board table {q['board_table_counts']}"]
    for k, v in probes.items():
        md.append(f"{k}: {json.dumps(v['result'], default=float)[:600]}")
    (REPO / "results" / "q2e_board.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
