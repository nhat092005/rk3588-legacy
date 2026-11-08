"""CLI: throughput analysis (S1 throughput form, S2, S3, process diagnostic, RSS), statements of bench/onboard_thr/rule.txt.
The helper log of the board is copied to bench/onboard_thr/hold/ first.

    .venv/bin/python -m ai.automation.analyze_thr
Writes results/thr_p3.{json,md}.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = REPO / "ai" / "automation" / "runs" / "ppe5_yolov8n_baseline_100ep_s0" / "bench" / "onboard_thr"
LINE_CPU = {"36": 4, "37": 5, "38": 6}
CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
LABEL = {"T1": "1 context CORE_0", "T2": "1 context CORE_0_1", "T3": "1 context CORE_0_1_2", "T4": "2 context CORE_0 và CORE_1",
         "T5": "3 context (S2)", "T6": "S3 pipeline", "T7": "3 tiến trình (chẩn đoán)"}


def spread(xs):
    return max(xs) - min(xs)


def main() -> None:
    entries = json.loads((HOME / "sessions.json").read_text())
    hlog = [ln.split() for ln in (HOME / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(x[0]), x[1], tuple(x[2:5])) for x in hlog]
    runs = {}
    for e in entries:
        rec = json.loads((HOME / e["file"]).read_text())
        irq = rec["irq_delta_per_cpu"]
        irq_ok = all(sum(d) == 0 or d[LINE_CPU[ln]] >= 0.99 * sum(d) for ln, d in irq.items()) and any(sum(d) > 0 for d in irq.values())
        aff = rec.get("affinity_per_process") or [rec["affinity_process"]]
        aff_ok = all(set(a) <= {4, 5, 6, 7} for a in aff)
        freq_ok = rec["frequency_before"]["locked"] and rec["frequency_after"]["locked"]
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        hold_ok = bool(win) and all(b == "inactive" and a == ("4", "5", "6") for _, b, a in win)
        runs[(e["code"], e["rep"])] = dict(code=e["code"], rep=e["rep"], realized=irq_ok and aff_ok and freq_ok and hold_ok, checks=dict(irq=irq_ok, affinity=aff_ok, freq=freq_ok, hold=hold_ok),
                                           fps=rec["throughput_fps"], wall_s=rec["wall_time_s"], t_pre=rec["t_pre_ms"]["p50"], t_npu=rec["t_npu_ms"]["p50"], t_post=rec["t_post_ms"]["p50"],
                                           cpu_pct=rec["resource"]["avg_process_cpu_pct"], temp_start=rec["resource"]["temp_start_c"], temp_max=rec["resource"]["temp_max_c"],
                                           rss=rec["rss_mib"], peak_rss=rec["resource"]["peak_rss_mib"], frames_per_context=rec["frames_per_context"], cached_images=rec["cached_images"],
                                           pre_workers=rec.get("pre_workers"), post_workers=rec.get("post_workers"), irq=irq)
    cfg = {}
    for c in CODES:
        rs = [runs[(c, r)] for r in (1, 2, 3)]
        fps = [r["fps"] for r in rs]
        cfg[c] = dict(label=LABEL[c], fps_runs=fps, fps=statistics.median(fps), spread=spread(fps), realized=all(r["realized"] for r in rs),
                      cpu_pct=statistics.median(r["cpu_pct"] for r in rs), cpu_spread=spread([r["cpu_pct"] for r in rs]),
                      t_pre=statistics.median(r["t_pre"] for r in rs), t_npu=statistics.median(r["t_npu"] for r in rs), t_post=statistics.median(r["t_post"] for r in rs),
                      temp_max=max(r["temp_max"] for r in rs), irq_line_min=min(sum(v) for r in rs for v in r["irq"].values() if sum(v) > 0),
                      irq_lines_fired=min(sum(1 for v in r["irq"].values() if sum(v) > 0) for r in rs))
    bad = "chưa kết luận (P3 không thực hiện đúng)"
    out = dict(runs=[{k: v for k, v in r.items() if k != "irq"} for r in runs.values()], configs=cfg, all_realized=all(v["realized"] for v in cfg.values()))
    s1 = {c: cfg[c]["fps"] for c in ("T1", "T2", "T3")}
    best = max(s1, key=s1.get)
    out["best_s1"] = best
    ok = cfg[best]["realized"] and cfg["T5"]["realized"]
    gain = cfg["T5"]["fps"] - cfg[best]["fps"]
    out["m12"] = dict(best=best, gain_fps=gain, gain_pct=100 * gain / cfg[best]["fps"], larger_spread=max(cfg["T5"]["spread"], cfg[best]["spread"]),
                      verdict=bad if not ok else ("hơn" if cfg["T5"]["fps"] >= 1.05 * cfg[best]["fps"] and gain > max(cfg["T5"]["spread"], cfg[best]["spread"]) else "không hơn"))
    ok3 = cfg["T6"]["realized"] and cfg["T5"]["realized"]
    g3 = cfg["T6"]["fps"] - cfg["T5"]["fps"]
    sp = max(cfg["T6"]["spread"], cfg["T5"]["spread"])
    cpu_gain = cfg["T5"]["cpu_pct"] - cfg["T6"]["cpu_pct"]
    cpu_sp = max(cfg["T6"]["cpu_spread"], cfg["T5"]["cpu_spread"])
    keep = (cfg["T6"]["fps"] >= 1.05 * cfg["T5"]["fps"] and g3 > sp) or (abs(g3) <= sp and cpu_gain > cpu_sp)
    out["m13"] = dict(gain_fps=g3, gain_pct=100 * g3 / cfg["T5"]["fps"], larger_spread=sp, cpu_t5=cfg["T5"]["cpu_pct"], cpu_t6=cfg["T6"]["cpu_pct"], cpu_gain=cpu_gain, cpu_spread=cpu_sp,
                      verdict=bad if not ok3 else ("đáng giữ" if keep else "không đáng giữ"),
                      workers=dict(pre=runs[("T6", 1)]["pre_workers"], post=runs[("T6", 1)]["post_workers"]))
    ok7 = cfg["T7"]["realized"] and cfg["T5"]["realized"]
    d7 = cfg["T7"]["fps"] - cfg["T5"]["fps"]
    out["m3"] = dict(diff_fps=d7, diff_pct=100 * d7 / cfg["T5"]["fps"], larger_spread=max(cfg["T7"]["spread"], cfg["T5"]["spread"]),
                     verdict=bad if not ok7 else ("khác" if abs(d7) >= 0.05 * cfg["T5"]["fps"] and abs(d7) > max(cfg["T7"]["spread"], cfg["T5"]["spread"]) else "không phân biệt được"))
    r1, r3 = runs[("T1", 1)]["rss"], runs[("T5", 1)]["rss"]
    out["m14"] = dict(t1_after_init=r1["after_context_1"], t5_after_init=r3["after_context_3"], per_added_context=(r3["after_context_3"] - r1["after_context_1"]) / 2,
                      t1_peak=statistics.median(runs[("T1", r)]["peak_rss"] for r in (1, 2, 3)), t5_peak=statistics.median(runs[("T5", r)]["peak_rss"] for r in (1, 2, 3)))
    out["rule"] = (HOME / "rule.txt").read_text()
    out["driver_log"] = (HOME / "driver.log").read_text()
    out["hold_progress"] = (HOME / "hold" / "progress.txt").read_text()
    (REPO / "results" / "thr_p3.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = ["# Throughput under P3", "", f"All runs realized: {out['all_realized']}", "", "| code | config | FPS runs | median | spread | realized |", "|---|---|---|---|---|---|"]
    md += [f"| {c} | {v['label']} | {v['fps_runs']} | {v['fps']:.2f} | {v['spread']:.2f} | {v['realized']} |" for c, v in cfg.items()]
    md += ["", f"M12: {out['m12']}", f"M13: {out['m13']}", f"M3: {out['m3']}", f"M14: {out['m14']}"]
    (REPO / "results" / "thr_p3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
