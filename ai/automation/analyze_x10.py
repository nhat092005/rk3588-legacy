"""CLI: X10, re-analysis of existing B3 data for U11 (small Conv against the compiler estimate) and U12 (time share against GFLOPs
share per stage), statements of bench/analysis_x10/rule.txt. No new measurement.

    .venv/bin/python -m ai.automation.analyze_x10 <run_id>
Writes results/analysis_x10_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    run = REPO / "ai" / "automation" / "runs" / args.run_id
    res = REPO / "results"
    L = json.loads((res / f"layer_analysis_{args.run_id}_int8_CORE_0.json").read_text())
    A = json.loads((res / f"latency_per_block_{args.run_id}_int8_CORE_0.json").read_text())
    a, b = L["conv"]["fit_slope"], L["conv"]["fit_intercept_us"]
    # per-row time: median of the 5 calls of a session, then median of the 3 B3 sessions (same as analyze_layers)
    sessions = [json.loads((run / "bench" / f"tier1_int8_CORE_0_sess{n}_layers.json").read_text())["layers"] for n in (1, 2, 3)]
    rows = []
    for i, r in enumerate(sessions[0]):
        if r["npu_cycles"] <= 0:
            continue
        t = statistics.median(statistics.median(s[i]["time_us_all"]) for s in sessions)
        est = r["total_cycles"] / 1000
        rows.append(dict(name=r.get("name") or r.get("full_name") or str(i), op=r["op"], est_us=est, time_us=t, ratio=t / est if est else None, predicted_us=a * est + b))
    small = [r for r in rows if r["ratio"] and r["ratio"] > 6]
    ok = all(abs(r["time_us"] - r["predicted_us"]) <= 0.2 * r["time_us"] for r in small)
    u11 = "tỉ số lớn của các Conv nhỏ khớp với phần hằng số b của phép khớp" if small and ok else "chưa kết luận"
    stages = {"backbone L0-L9": [f"L{i}" for i in range(10)], "neck L10-L21": [f"L{i}" for i in range(10, 22)], "head L22": ["L22"]}
    st = {}
    for k, blocks in stages.items():
        before = L["stages"][k]["time_share"]
        gfl = L["stages"][k]["gflops_share"]
        after = sum(A["blocks"][x]["share_sens"] for x in blocks)
        st[k] = dict(time_share=before, time_share_without_row_constant=after, gflops_share=gfl, gap_before=before - gfl, gap_after=after - gfl)
    if all(abs(v["gap_after"]) <= 0.5 * abs(v["gap_before"]) for v in st.values()):
        u12 = "phần lớn chênh lệch giữa tỉ lệ thời gian và tỉ lệ GFLOPs theo giai đoạn đi theo hằng số mỗi dòng"
    elif all(abs(v["gap_after"]) >= 0.8 * abs(v["gap_before"]) for v in st.values()):
        u12 = "hằng số mỗi dòng không giải thích chênh lệch"
    else:
        u12 = "chưa kết luận"
    out = dict(fit=dict(a=a, b_us=b), small_rows=small, n_small=len(small), u11=u11, stages=st, u12=u12, rule=(run / "bench" / "analysis_x10" / "rule.txt").read_text())
    name = f"analysis_x10_{args.run_id}_int8_CORE_0"
    (res / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = [f"# X10: re-analysis of B3 data, {args.run_id}", "", f"Fit: time = {a:.3f} x est + {b:.1f} us", "", "| row | op | est us | time us | ratio | predicted us |", "|---|---|---|---|---|---|"]
    md += [f"| {r['name']} | {r['op']} | {r['est_us']:.1f} | {r['time_us']:.1f} | {r['ratio']:.1f} | {r['predicted_us']:.1f} |" for r in small]
    md += ["", f"U11: {u11}", "", "| stage | time share | without row constant | GFLOPs share | gap before | gap after |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['time_share']:.3f} | {v['time_share_without_row_constant']:.3f} | {v['gflops_share']:.3f} | {v['gap_before']:+.3f} | {v['gap_after']:+.3f} |" for k, v in st.items()]
    md += ["", f"U12: {u12}"]
    (res / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
