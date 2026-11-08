"""CLI: split-head accuracy, NPU time, CPU decode time and total, statements S-1 to S-4 of bench/onboard_q2c/rule_split.txt and bench/onboard_q2j/rule.txt.
The helper log is copied to bench/onboard_q2j/hold/ first.

    .venv/bin/python -m ai.automation.analyze_q2j
Writes results/q2j_split.{json,md}.
"""
import json
import re
import statistics
from pathlib import Path

from ai.automation.analyze_g4 import RUNS, split_row

REPO = Path(__file__).resolve().parents[2]
RUN = "ppe5_yolov8n_baseline_100ep_s0"
HOME = RUNS / RUN / "bench" / "onboard_q2j"
Q2C = RUNS / RUN / "bench" / "onboard_q2c"
MASKS = ["CORE_0", "CORE_0_1_2"]
LINES = ("36", "37", "38")
NAMES = ["int8", "int8h5", "split"]


def table_from_log(log: Path) -> list[dict]:
    from ai.core.rknn.benchmark import parse_layers

    text = "\n".join(re.sub(r"^.*?\] ", "", ln) for ln in log.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln)
    rows, first = parse_layers(text[text.index("Network Layer Information Table"):]), []
    for r in rows:
        if first and r["id"] <= first[-1]["id"]:
            break
        first.append(r)
    return first


def ap(tag: str, st: str) -> dict:
    d = json.loads((RUNS / RUN / "eval" / f"{tag}_{st}.json").read_text())
    return dict(map50=100 * d["map50"], map50_95=100 * d["map50_95"], NoHelmet=100 * d["per_class"]["NoHelmet"]["ap50_95"], Vest=100 * d["per_class"]["Vest"]["ap50_95"])


def main() -> None:
    from ai.automation.analyze_g4 import build_table

    out = {"accuracy": {}}
    for st in ("css_test", "ppe2286_ext"):
        out["accuracy"][st] = {k: ap(t, st) for k, t in (("int8", "rknn_int8"), ("int8h5", "rknn_int8h5"), ("split", "rknn_int8_split"), ("fp16", "rknn_fp16"), ("fp16_split", "rknn_fp16_split"))}
    acc = out["accuracy"]["css_test"]
    out["v_board"] = dict(fp16_split=acc["fp16_split"]["map50_95"], fp16=acc["fp16"]["map50_95"], diff=acc["fp16_split"]["map50_95"] - acc["fp16"]["map50_95"], ok=abs(acc["fp16_split"]["map50_95"] - acc["fp16"]["map50_95"]) <= 1.35)
    out["v_pc"] = dict(max_abs=json.loads((RUNS / RUN / "weights" / "rknn_int8_split" / "recipe.json").read_text())["onnx_check_max_abs_vs_standard"])
    out["s1"] = dict(split_minus_fp16=acc["split"]["map50_95"] - acc["fp16"]["map50_95"], split_minus_int8h5=acc["split"]["map50_95"] - acc["int8h5"]["map50_95"], split_minus_int8=acc["split"]["map50_95"] - acc["int8"]["map50_95"],
                     meets=acc["split"]["map50_95"] >= acc["fp16"]["map50_95"] - 1.35)
    dumps = {k: json.loads((RUNS / RUN / "weights" / f"npu_dump_{t}_css_test.json").read_text()) for k, t in (("split", "int8_split"), ("fp16_split", "fp16_split"), ("int8", "int8"), ("int8h5", "int8h5"))}
    out["s4"] = {k: dict(kept_anchors=v["kept_anchors"], decode_ms_median=v.get("decode_ms_median")) for k, v in dumps.items()}
    out["decode"] = json.loads((HOME / "decode_bench.json").read_text())
    # latency
    entries = json.loads((HOME / "sessions.json").read_text())
    hlog = [ln.split() for ln in (HOME / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]
    sess = []
    for e in entries:
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        name = "split" if e["variant"] == "split" else e["precision"]
        stem = f"tier1_{e['precision']}{'_' + e['variant'] if e['variant'] else ''}_{e['mask']}_sess{e['session']}"
        rec = json.loads((RUNS / e["run"] / "bench" / "onboard_q2j" / f"{stem}.json").read_text())
        lay = json.loads((RUNS / e["run"] / "bench" / "onboard_q2j" / f"{stem}_layers.json").read_text())
        assert rec["model_sha256"] == e["sha256"]
        sess.append(dict(name=name, mask=e["mask"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], layers=lay["layers"]))
    out["all_realized"] = all(s["realized"] for s in sess)
    cond = {}
    for n in NAMES:
        for k in MASKS:
            ss = sorted((s for s in sess if s["name"] == n and s["mask"] == k), key=lambda s: s["session"])
            assert len(ss) == 2
            meds = [s["latency_ms"] for s in ss]
            cond[(n, k)] = dict(session_medians=meds, median=statistics.median(meds), gap_ms=abs(meds[0] - meds[1]), realized=all(s["realized"] for s in ss))
    out["conditions"] = {"|".join(k): v for k, v in cond.items()}
    out["cost"] = {}
    for k in MASKS:
        b = cond[("int8", k)]
        for n in ("int8h5", "split"):
            h = cond[(n, k)]
            diff = h["median"] - b["median"]
            gap = max(b["gap_ms"], h["gap_ms"])
            out["cost"][f"{n}|{k}"] = dict(int8_ms=b["median"], ms=h["median"], diff_ms=diff, diff_pct=100 * diff / b["median"], gap_ms=gap, changes="đổi latency" if abs(diff) > gap and b["realized"] and h["realized"] else ("không đổi" if b["realized"] and h["realized"] else "chưa kết luận"))
    dec = out["decode"]["median_ms"]
    out["total"] = {k: dict(split_npu_ms=cond[("split", k)]["median"], decode_ms=dec, split_total_ms=cond[("split", k)]["median"] + dec, int8h5_ms=cond[("int8h5", k)]["median"],
                            diff_vs_int8h5_ms=cond[("split", k)]["median"] + dec - cond[("int8h5", k)]["median"]) for k in MASKS}
    # coverage of the split model
    tab = table_from_log(Q2C / "logs" / f"build_{RUN}_split_int8.log")
    total = sum(r["total_cycles"] or 0 for r in tab)
    ss = [s for s in sess if s["name"] == "split" and s["mask"] == "CORE_0_1_2"]
    cov = []
    for s in ss:
        rows = [r for r in s["layers"] if r["total_cycles"]]
        assert sum(r["total_cycles"] for r in rows) == total, "board cycles differ from the build table"
        cov.append(sum(r["total_cycles"] for r in rows if split_row(r["workload"])) / total)
    out["coverage_split"] = dict(per_session=cov, build_total_cycles=total, rows=len(tab), cpu_ops=sorted({r["op"] for r in tab if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}),
                                 dtypes=sorted({f"{r['target']}|{r['dtype']}" for r in tab}))
    out["rule_split"] = (Q2C / "rule_split.txt").read_text()
    out["rule"] = (HOME / "rule.txt").read_text()
    out["driver_log"] = (HOME / "driver.log").read_text()
    out["hold_progress"] = (HOME / "hold" / "progress.txt").read_text()
    (REPO / "results" / "q2j_split.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# split head", "", f"V-PC {out['v_pc']}", f"V-board {out['v_board']}", f"S-1 {out['s1']}", f"S-4 {out['s4']}", f"decode {out['decode']['median_ms']} ms p95 {out['decode']['p95_ms']}", f"All realized: {out['all_realized']}", ""]
    md += [f"- cost {k}: {v}" for k, v in out["cost"].items()]
    md += [f"- total {k}: {v}" for k, v in out["total"].items()]
    md.append(f"- coverage split {out['coverage_split']}")
    (REPO / "results" / "q2j_split.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
