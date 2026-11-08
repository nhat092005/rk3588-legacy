"""CLI: X4 analysis (tier-1 eval_perf over the LAN, protocols P1-LAN and P2 interleaved, mask and log comparisons), statements of
bench/onboard_x4/rule.txt. Gates come from ai/automation/aggregate_layers.py, run before this script with
--bench-subdir onboard_x4/p1lan --tag p1lan and --bench-subdir onboard_x4/p2 --tag p2. The board helper files are copied to
bench/onboard_x4/hold/ first.

    .venv/bin/python -m ai.automation.analyze_x4 <run_id>
Writes results/onboard_x4_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ai.automation.aggregate_layers import linfit

REPO = Path(__file__).resolve().parents[2]
LINES = ("36", "37", "38")


def row_intercept(layers):
    conv = [r for r in layers["layers"] if r["npu_cycles"] > 0]
    return max(linfit([r["total_cycles"] / 1000 for r in conv], [r["time_us"] for r in conv])[1], 0.0)


def overlap_verdict(ref, other):
    """Rule: 'khác' when every session median of the setting lies outside the min to max range of the reference."""
    lo, hi = min(ref), max(ref)
    return "khác" if all(v < lo or v > hi for v in other) else "không phân biệt được"


def level(ms):
    return "fast" if ms < 17.0 else ("slow" if ms >= 20.0 else "mid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    d = REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "onboard_x4"
    entries = json.loads((d / "sessions.json").read_text())
    hlog = [ln.split() for ln in (d / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]

    def realized(e):
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        lines_used = {ln: sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for c in range(8)) for ln in LINES}
        if e["protocol"] == "p2":
            ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        else:
            ok = bool(win) and all(m == "unpin" for _, m, _, _ in win)
        return dict(ok=ok, log_lines=len(win), irq_per_cpu=delta, irq_lines=lines_used)

    for e in entries:
        e["realized"] = realized(e)
    out = {"protocols": {}}
    for proto in ("p1lan", "p2"):
        rows = []
        for e in [x for x in entries if x["protocol"] == proto and x["kind"] == "benchmark_npu"]:
            sub = d.parent / e["subdir"]
            rec = json.loads((sub / f"tier1_int8_CORE_0_sess{e['session']}.json").read_text())
            lay = json.loads((sub / f"tier1_int8_CORE_0_sess{e['session']}_layers.json").read_text())
            v = rec["npu_latency_ms_all"]
            rows.append(dict(session=e["session"], realized=e["realized"]["ok"], latency_ms=rec["npu_latency_ms"], latency_all=v,
                             level=level(rec["npu_latency_ms"]), first_minus_rest=v[0] - statistics.median(v[1:]), first_lower=v[0] < statistics.median(v[1:]),
                             debug_total_ms=rec["debug_total_us"] / 1000, row_intercept_us=row_intercept(lay), irq_per_cpu=e["realized"]["irq_per_cpu"]))
        all_ok = all(r["realized"] for r in rows)
        gate_ok = all(r["realized"] for r in rows if r["session"] <= 3)
        tag = "P2" if proto == "p2" else "P1"
        bad = f"chưa kết luận ({tag} không thực hiện đúng)"
        firsts = sum(r["first_lower"] for r in rows)
        u6 = (f"lần gọi đầu thấp hơn các lần sau ở {firsts}/{len(rows)} phiên" if firsts >= 9 else ("không ổn định" if firsts <= 6 else f"{firsts}/{len(rows)} phiên"))
        icp = [r["row_intercept_us"] for r in rows]
        gate_file = REPO / "results" / f"latency_per_block_{args.run_id}_int8_CORE_0_{proto}.json"
        gates = {c["code"]: c["ok"] for c in json.loads(gate_file.read_text())["checks"]} if gate_file.exists() else None
        out["protocols"][proto] = dict(sessions=rows, all_realized=all_ok, gate_sessions_realized=gate_ok, gates=gates if gate_ok else bad,
                                       u6=u6 if all_ok else bad, first_lower=firsts, first_minus_rest_median_ms=statistics.median(r["first_minus_rest"] for r in rows),
                                       intercepts_us=icp, intercept_spread_us=max(icp) - min(icp),
                                       debug_below_latency=sum(r["debug_total_ms"] < r["latency_ms"] for r in rows) if all_ok else bad,
                                       levels=[r["level"] for r in rows])
    p1, p2 = out["protocols"]["p1lan"], out["protocols"]["p2"]
    if isinstance(p1["gates"], dict) and isinstance(p2["gates"], dict):
        reading = {}
        for code in ("A5", "A8"):
            a, b = p1["gates"].get(code), p2["gates"].get(code)
            reading[code] = {(False, True): "trên LAN, ghim lõi làm cổng đạt", (True, True): "trên LAN cả hai giao thức đều đạt, không tách được ghim lõi khỏi khoảng cách giữa các phiên",
                             (False, False): "ghim lõi không đủ để cổng đạt", (True, False): "chưa kết luận"}[(a, b)]
        out["gate_reading"] = reading
        labels = {"U25": "CHƯA BIẾT" if reading["A5"] == "chưa kết luận" else "ĐO",
                  "U7": "ĐO" if reading["A5"] == "trên LAN, ghim lõi làm cổng đạt" else "CHƯA BIẾT"}
        out["labels"] = labels
        out["fp16_allowed"] = bool(p2["gates"].get("A8"))
    else:
        out["gate_reading"] = "chưa kết luận (phiên cổng không thực hiện đúng)"
        out["fp16_allowed"] = False
    if p1["all_realized"] and p2["all_realized"]:
        sp1, sp2 = p1["intercept_spread_us"], p2["intercept_spread_us"]
        out["u10"] = ("ghim lõi thu hẹp chênh lệch hằng số mỗi dòng giữa các phiên" if sp2 <= 0.5 * sp1 else
                      ("ghim lõi không thu hẹp" if sp2 >= 0.8 * sp1 else "chưa kết luận"))
    else:
        out["u10"] = "chưa kết luận (phiên không thực hiện đúng)"
    comp = json.loads((d / "comparisons.json").read_text())

    def setting(key, field, val):
        xs = [x for x in comp[key] if x[field] == val]
        return [x["median_ms"] for x in xs], [entries[x["session_index"]]["realized"] for x in xs]

    ref, ref_rs = setting("masks", "mask", "CORE_0")
    ref_ok = all(r["ok"] for r in ref_rs)
    out["masks"] = {}
    for m in ("CORE_0", "AUTO", "CORE_0_1_2"):
        meds, rs = setting("masks", "mask", m)
        ok = all(r["ok"] for r in rs)
        out["masks"][m] = dict(session_medians=meds, diff_vs_core0_ms=statistics.median(meds) - statistics.median(ref), realized=ok,
                               irq_lines_per_session=[r["irq_lines"] for r in rs], irq_cpu_per_session=[r["irq_per_cpu"] for r in rs],
                               verdict=None if m == "CORE_0" else (overlap_verdict(ref, meds) if ok and ref_ok else "chưa kết luận (P2 không thực hiện đúng)"))
    lref, lref_rs = setting("log", "verbose", False)
    lref_ok = all(r["ok"] for r in lref_rs)
    out["log"] = {}
    for v in (False, True):
        meds, rs = setting("log", "verbose", v)
        ok = all(r["ok"] for r in rs)
        out["log"][str(v)] = dict(session_medians=meds, diff_vs_quiet_ms=statistics.median(meds) - statistics.median(lref), realized=ok,
                                  verdict=None if v is False else (overlap_verdict(lref, meds) if ok and lref_ok else "chưa kết luận (P2 không thực hiện đúng)"))
    out["rule"] = (d / "rule.txt").read_text()
    out["driver_log"] = (d / "driver.log").read_text()
    out["hold_progress"] = (d / "hold" / "progress.txt").read_text()
    name = f"onboard_x4_{args.run_id}_int8_CORE_0"
    (REPO / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = [f"# X4: tier-1 eval_perf over the LAN, {args.run_id}", "", "Rule: bench/onboard_x4/rule.txt (written before the data).", ""]
    for proto, v in out["protocols"].items():
        md += [f"## {proto}", "", f"All sessions realized: {v['all_realized']}. Gates (sessions 1 to 3): {v['gates']}",
               "| session | realized | latency ms | level | first minus rest ms | debug total ms | row intercept us |", "|---|---|---|---|---|---|---|"]
        md += [f"| {r['session']} | {r['realized']} | {r['latency_ms']:.3f} | {r['level']} | {r['first_minus_rest']:+.3f} | {r['debug_total_ms']:.3f} | {r['row_intercept_us']:.1f} |" for r in v["sessions"]]
        md += ["", f"U6: {v['u6']} (median first minus rest {v['first_minus_rest_median_ms']:+.3f} ms). U10 spread {v['intercept_spread_us']:.1f} us. U13 debug below latency: {v['debug_below_latency']}", ""]
    md += [f"Gate reading: {out['gate_reading']}", f"Labels: {out.get('labels')}", f"FP16 allowed: {out['fp16_allowed']}", f"U10: {out['u10']}", "",
           f"Masks: {json.dumps(out['masks'], default=float)}", "", f"Log: {json.dumps(out['log'], default=float)}"]
    (REPO / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
