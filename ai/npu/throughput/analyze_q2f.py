"""CLI: FPS levers under P3, statements of experiments/npu/throughput/onboard_q2f_thr/rule.txt (E1, int8 against int8h5) and experiments/npu/throughput/onboard_q2f_split/rule.txt (E2, split head
with the decode overlapped). The helper logs are copied to the hold/ folders first.

    .venv/bin/python -m ai.npu.throughput.analyze_q2f
Writes results/npu/throughput/q2f_fps.json and results/npu/throughput/q2f_fps.md. Reads only the experiments that have a sessions.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from ai.paths import BASELINE_RUN, THROUGHPUT_RESULTS_DIR, experiment_dir

LINE_CPU = {"36": 4, "37": 5, "38": 6}
TARGET = 60.0


def realized(home: Path, e: dict, rec: dict, hlog: list) -> dict:
    irq = rec["irq_delta_per_cpu"]
    irq_ok = all(sum(d) == 0 or d[LINE_CPU[ln]] >= 0.99 * sum(d) for ln, d in irq.items()) and any(sum(d) > 0 for d in irq.values())
    aff = rec.get("affinity_per_process") or [rec["affinity_process"]]
    aff_ok = all(set(a) <= {4, 5, 6, 7} for a in aff)
    freq_ok = rec["frequency_before"]["locked"] and rec["frequency_after"]["locked"]
    win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
    hold_ok = bool(win) and all(b == "inactive" and a == ("4", "5", "6") for _, b, a in win)
    return dict(irq=irq_ok, affinity=aff_ok, freq=freq_ok, hold=hold_ok, realized=irq_ok and aff_ok and freq_ok and hold_ok)


def load(home: Path, key) -> dict:
    entries = json.loads((home / "sessions.json").read_text())
    hlog = [ln.split() for ln in (home / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(x[0]), x[1], tuple(x[2:5])) for x in hlog]
    runs: dict = {}
    for e in entries:
        rec = json.loads((home / e["file"]).read_text())
        chk = realized(home, e, rec, hlog)
        runs.setdefault(key(e), []).append(dict(rep=e["rep"], fps=rec["throughput_fps"], realized=chk["realized"], checks=chk, cpu_pct=rec["resource"]["avg_process_cpu_pct"],
                                                t_pre=rec["t_pre_ms"]["p50"], t_npu=rec["t_npu_ms"]["p50"], t_post=rec["t_post_ms"]["p50"],
                                                t_post_p95=rec["t_post_ms"].get("p95"), e2e=rec.get("e2e_latency_ms"), temp_max=rec["resource"]["temp_max_c"],
                                                sha256=rec.get("model_sha256")))
    cond = {}
    for k, rs in runs.items():
        assert len(rs) == 3, (k, len(rs))
        fps = [r["fps"] for r in rs]
        cond["|".join(k)] = dict(fps_runs=fps, fps=statistics.median(fps), spread=max(fps) - min(fps), realized=all(r["realized"] for r in rs),
                                 cpu_pct=statistics.median(r["cpu_pct"] for r in rs), t_pre=statistics.median(r["t_pre"] for r in rs), t_npu=statistics.median(r["t_npu"] for r in rs),
                                 t_post=statistics.median(r["t_post"] for r in rs), stage_sum=statistics.median(r["t_pre"] + r["t_npu"] + r["t_post"] for r in rs),
                                 e2e_p50=statistics.median(r["e2e"]["p50"] for r in rs) if rs[0]["e2e"] else None, e2e_p95=statistics.median(r["e2e"]["p95"] for r in rs) if rs[0]["e2e"] else None,
                                 meets_60=statistics.median(fps) >= TARGET, shas=sorted({r["sha256"] for r in rs}))
    return cond


def compare(cond: dict, a: str, b: str) -> dict:
    x, y = cond[a], cond[b]
    d = x["fps"] - y["fps"]
    sp = max(x["spread"], y["spread"])
    ok = x["realized"] and y["realized"]
    return dict(a=a, b=b, diff_fps=d, diff_pct=100 * d / y["fps"], larger_spread=sp,
                verdict="chưa kết luận (P3 không thực hiện đúng)" if not ok else ("khác" if abs(d) >= 0.05 * y["fps"] and abs(d) > sp else "không phân biệt được"))


def main() -> None:
    out: dict = {}
    md = ["# FPS levers under P3", ""]
    e1 = experiment_dir(BASELINE_RUN, "onboard_q2f_thr")
    if (e1 / "sessions.json").exists():
        cond = load(e1, lambda e: (e["model"], e["code"]))
        out["e1"] = dict(conditions=cond, rule=(e1 / "rule.txt").read_text(), driver_log=(e1 / "driver.log").read_text(), hold_progress=(e1 / "hold" / "progress.txt").read_text(),
                         comparisons={c: compare(cond, f"int8h5|{c}", f"int8|{c}") for c in ("T1", "T3", "T4", "T5")}, all_realized=all(v["realized"] for v in cond.values()))
        md += ["## E1 int8h5 against int8", ""] + [f"- {k}: {v['fps']:.2f} FPS spread {v['spread']:.2f} realized {v['realized']} meets60 {v['meets_60']}" for k, v in cond.items()]
        md += [f"- {k}: {v}" for k, v in out["e1"]["comparisons"].items()]
    e2 = experiment_dir(BASELINE_RUN, "onboard_q2f_split")
    if (e2 / "sessions.json").exists():
        cond = load(e2, lambda e: (e["code"],))
        pairs = [("B", "A"), ("C", "A"), ("C", "B"), ("D", "C"), ("E", "C"), ("F", "A")]
        trig = (not cond["C"]["meets_60"]) and cond["C"]["t_post"] >= cond["C"]["t_npu"]
        out["e2"] = dict(conditions=cond, rule=(e2 / "rule.txt").read_text(), driver_log=(e2 / "driver.log").read_text(), hold_progress=(e2 / "hold" / "progress.txt").read_text(),
                         comparisons={f"{a}_vs_{b}": compare(cond, a, b) for a, b in pairs}, e3_triggered=trig, all_realized=all(v["realized"] for v in cond.values()),
                         per_frame_reading={k: (v["e2e_p50"] is not None and v["e2e_p50"] <= 1000 / 60) for k, v in cond.items() if v["e2e_p50"] is not None})
        if (e2 / "split_gate.json").exists():
            out["e2"]["gate"] = {k: v for k, v in json.loads((e2 / "split_gate.json").read_text()).items() if k != "per_image"}
        md += ["", "## E2 split head decode overlap", ""] + [f"- {k}: {v['fps']:.2f} FPS spread {v['spread']:.2f} realized {v['realized']} meets60 {v['meets_60']} e2e_p50 {v['e2e_p50']} stage_sum {v['stage_sum']:.2f}" for k, v in cond.items()]
        md += [f"- {k}: {v}" for k, v in out["e2"]["comparisons"].items()] + [f"- E3 triggered: {trig}"]
    (THROUGHPUT_RESULTS_DIR / "q2f_fps.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (THROUGHPUT_RESULTS_DIR / "q2f_fps.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
