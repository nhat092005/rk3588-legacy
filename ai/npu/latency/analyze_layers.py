"""CLI: deeper analysis of the tier-1 per-layer tables of a run (docs 03-phuong-phap/02-tang-npu-nen-do/03-phan-tich-sau-latency-tung-khoi).

Reads the same runs/<run_id>/bench/tier1_<precision>_<core_mask>_sess<N>* files as aggregate_layers and writes
results/npu/latency/layer_analysis_<run_id>_<precision>_<core_mask>.{md,json}. Everything is in perf_debug=True time, which the
guide says may disable parallel mechanisms (UG232 p.177). "est. cycles time" is Total cycles / 1000 us, i.e. the
compiler's cycle estimate at the locked 1 GHz NPU clock, not a hardware bound.
"""
import argparse
import collections
import json
import statistics
import sys

import numpy as np

from ai.npu.latency.aggregate_layers import OUT_DIR, block_order, layer_gflops, load_sessions, med, spread
from ai.automation.benchmark_npu import RUNS_DIR
from ai.core.rknn.blocks import assign

NPU_MHZ = 1000  # NPU frequency printed by eval_perf after lock_freq.sh lock: 1000000000 Hz


def row_table(sessions):
    """One dict per layer: time = median over sessions of the per-session median over eval_perf calls."""
    rows = []
    n = len(sessions[0][1]["layers"])
    for i in range(n):
        per = [s[1]["layers"][i] for s in sessions]
        first = per[0]
        block, part = assign(first)
        rows.append(dict(id=first["id"], op=first["op"], target=first["target"], name=first["name"], block=block, part=part,
                         ddr=first["ddr_cycles"], npu=first["npu_cycles"], total=first["total_cycles"], rw_kb=first["rw_kb"],
                         mac=first["mac_usage"], time=med([p["time_us"] for p in per]), time_spread=spread([p["time_us"] for p in per])))
    return rows


def linfit(x, y):
    a = np.vstack([x, np.ones(len(x))]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(a, np.array(y, dtype=float), rcond=None)
    pred = slope * np.array(x) + intercept
    ss_res = float(np.sum((np.array(y) - pred) ** 2))
    ss_tot = float(np.sum((np.array(y) - np.mean(y)) ** 2))
    return float(slope), float(intercept), 1 - ss_res / ss_tot if ss_tot else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--precision", default="int8", choices=["fp16", "int8"])
    parser.add_argument("--core-mask", default="CORE_0")
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run_id
    sessions = load_sessions(run_dir / "bench", f"tier1_{args.precision}_{args.core_mask}")
    if not sessions:
        sys.exit("no sessions found")
    rows = row_table(sessions)
    total = sum(r["time"] for r in rows)
    out = {"n_sessions": len(sessions), "total_us": total}

    # 1. Conv rows against the compiler's cycle estimate
    conv = [r for r in rows if r["npu"] > 0]
    for r in conv:
        r["est_us"] = r["total"] / NPU_MHZ
        r["ratio"] = r["time"] / r["est_us"]
    slope, intercept, r2 = linfit([r["est_us"] for r in conv], [r["time"] for r in conv])
    bands = [("<= 1.5", 0, 1.5), ("1.5 - 3", 1.5, 3), ("3 - 6", 3, 6), ("> 6", 6, 1e9)]
    band_rows = [dict(band=n, rows=sum(1 for r in conv if lo < r["ratio"] <= hi), time_us=sum(r["time"] for r in conv if lo < r["ratio"] <= hi),
                      est_us=sum(r["est_us"] for r in conv if lo < r["ratio"] <= hi)) for n, lo, hi in bands]
    out["conv"] = dict(n=len(conv), time_us=sum(r["time"] for r in conv), est_us=sum(r["est_us"] for r in conv),
                       ratio_median=med([r["ratio"] for r in conv]), ratio_min=min(r["ratio"] for r in conv), ratio_max=max(r["ratio"] for r in conv),
                       fit_slope=slope, fit_intercept_us=intercept, fit_r2=r2, bands=band_rows,
                       ddr_gt_npu_rows=[r["id"] for r in conv if r["ddr"] > r["npu"]])

    # 2. rows that are not Conv
    other = [r for r in rows if r["npu"] == 0]
    ops = collections.defaultdict(lambda: dict(calls=0, time=0.0, rw_kb=0))
    for r in other:
        o = ops[r["op"]]
        o["calls"] += 1
        o["time"] += r["time"]
        o["rw_kb"] += r["rw_kb"] or 0
    for o in ops.values():
        o["us_per_mb"] = o["time"] / (o["rw_kb"] / 1024) if o["rw_kb"] else None
    mov = [r for r in other if r["op"] in ("Concat", "Split", "Add") and r["rw_kb"]]
    s2, i2, r22 = linfit([r["rw_kb"] for r in mov], [r["time"] for r in mov])
    out["other"] = dict(time_us=sum(r["time"] for r in other), share=sum(r["time"] for r in other) / total, ops=ops,
                        move_fit=dict(n=len(mov), us_per_mb=s2 * 1024, intercept_us=i2, r2=r22))

    # 3. Detect head and the decode chain
    head = collections.Counter()
    for r in rows:
        if r["block"] in ("L22", "post_detect"):
            head[(r["block"], r["part"])] += r["time"]
    dfl_chain = [r for r in rows if r["part"] == "dfl"]
    decode_ops = collections.Counter()
    for r in rows:
        if r["block"] in ("L22", "post_detect") and r["part"] in ("dfl", "decode", "Mul", "Concat", "Reshape"):
            decode_ops[(r["part"], r["op"], r["target"])] += r["time"]
    out["head"] = dict(parts={f"{b}/{p}": t for (b, p), t in head.items()}, head_total=sum(head.values()),
                       dfl_chain=[dict(id=r["id"], op=r["op"], target=r["target"], time=r["time"]) for r in dfl_chain],
                       decode_ops={f"{p}/{o}/{t}": v for (p, o, t), v in decode_ops.items()})

    # 4. stage share against GFLOPs share
    gf = layer_gflops(run_dir / "weights" / "best.pt")
    stages = {"backbone L0-L9": range(0, 10), "neck L10-L21": range(10, 22), "head L22": [22]}
    gf_total = sum(gf.values())
    stage_rows = {}
    for name, idx in stages.items():
        t = sum(r["time"] for r in rows if r["block"] in {f"L{i}" for i in idx})
        g = sum(gf.get(f"L{i}", 0) for i in idx)
        stage_rows[name] = dict(time_us=t, time_share=t / total, gflops=g, gflops_share=g / gf_total)
    post = sum(r["time"] for r in rows if r["block"] in ("io", "post_detect"))
    stage_rows["io and post_detect"] = dict(time_us=post, time_share=post / total, gflops=0.0, gflops_share=0.0)
    out["stages"] = stage_rows

    # 5. first call against the rest, per session
    warm = []
    for record, layers, _ in sessions:
        lat, dbg = record["npu_latency_ms_all"], layers["debug_total_us_all"]
        warm.append(dict(latency_first=lat[0], latency_rest_median=statistics.median(lat[1:]), debug_first=dbg[0], debug_rest_median=statistics.median(dbg[1:])))
    out["first_vs_rest"] = warm

    # 6. per-row difference between each numbered session and the dry run (B2): constant overhead or size dependent?
    dry_json = run_dir / "bench" / f"tier1_{args.precision}_{args.core_mask}_layers.json"
    delta = []
    if dry_json.exists():
        base = json.loads(dry_json.read_text())["layers"]
        est = np.array([r["total_cycles"] / NPU_MHZ for r in base])
        rw = np.array([r["rw_kb"] for r in base], dtype=float)
        x = np.vstack([np.ones(len(base)), est, rw]).T
        t0 = np.array([r["time_us"] for r in base], dtype=float)
        for i, (_, layers, _) in enumerate(sessions, 1):
            t = np.array([r["time_us"] for r in layers["layers"]], dtype=float)
            d = t - t0
            coef, *_ = np.linalg.lstsq(x, d, rcond=None)
            r2 = 1 - float(np.sum((d - x @ coef) ** 2)) / float(np.sum((d - d.mean()) ** 2))
            groups = collections.defaultdict(list)
            for r, v in zip(layers["layers"], d):
                groups["Conv" if r["npu_cycles"] > 0 else r["op"]].append(v)
            delta.append(dict(session=i, total_delta=float(d.sum()), per_row_mean=float(d.mean()), per_row_median=float(np.median(d)),
                              intercept=float(coef[0]), per_est_us=float(coef[1]), us_per_mb=float(coef[2] * 1024), r2=r2,
                              by_group={k: statistics.mean(v) for k, v in groups.items() if len(v) >= 3}))
    out["delta_vs_b2"] = delta

    OUT_DIR.mkdir(exist_ok=True)
    name = f"layer_analysis_{args.run_id}_{args.precision}_{args.core_mask}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    c = out["conv"]
    md = [f"# Layer analysis: {args.run_id} {args.precision} {args.core_mask}", "",
          f"{len(sessions)} sessions, perf_debug=True time per layer (median over calls, then over sessions). Total {total:.0f} us. "
          f"est. cycles time = Total cycles / {NPU_MHZ} us (compiler estimate at the locked NPU clock, not a hardware bound).", "",
          "## 1. Conv layers against the compiler's cycle estimate", "",
          f"{c['n']} Conv rows use {c['time_us']:.0f} us ({c['time_us'] / total:.1%} of the total) against {c['est_us']:.0f} us of estimated cycles time. "
          f"Per-row ratio time / est.: median {c['ratio_median']:.2f}, min {c['ratio_min']:.2f}, max {c['ratio_max']:.2f}. "
          f"Least squares fit time = {c['fit_slope']:.3f} x est. + {c['fit_intercept_us']:.1f} us (R2 {c['fit_r2']:.3f}, descriptive fit over {c['n']} rows, not a hardware model). "
          f"Rows with DDR cycles above NPU cycles: {c['ddr_gt_npu_rows'] or 'none'}.", "",
          "| ratio band | rows | time us | est. us |", "|---|---|---|---|"]
    md += [f"| {b['band']} | {b['rows']} | {b['time_us']:.0f} | {b['est_us']:.0f} |" for b in c["bands"]]
    md += ["", "| ID | block/part | OpType | cycles DDR/NPU/Total | est. us | time us | ratio | MacUsage | RW KB |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['block']}/{r['part']} | {r['op']} | {r['ddr']}/{r['npu']}/{r['total']} | {r['est_us']:.0f} | {r['time']:.0f} | {r['ratio']:.2f} | {r['mac']} | {r['rw_kb']} |"
           for r in sorted(conv, key=lambda r: -r["time"])]
    o = out["other"]
    md += ["", "## 2. Rows without NPU cycles (not Conv)", "",
           f"{o['time_us']:.0f} us ({o['share']:.1%} of the total). Least squares over {o['move_fit']['n']} Concat, Split and Add rows: "
           f"time = {o['move_fit']['us_per_mb']:.0f} us per MB read and written + {o['move_fit']['intercept_us']:.1f} us (R2 {o['move_fit']['r2']:.3f}).", "",
           "| OpType | calls | time us | share % | RW KB | us per MB |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['calls']} | {v['time']:.0f} | {100 * v['time'] / total:.2f} | {v['rw_kb']} | {'-' if v['us_per_mb'] is None else f'{v['us_per_mb']:.0f}'} |"
           for k, v in sorted(ops.items(), key=lambda kv: -kv[1]["time"])]
    h = out["head"]
    md += ["", "## 3. Detect head and decode chain", "", f"L22 plus post_detect: {h['head_total']:.0f} us ({h['head_total'] / total:.1%}).", "",
           "| group | time us | share % |", "|---|---|---|"]
    md += [f"| {k} | {v:.0f} | {100 * v / total:.1f} |" for k, v in sorted(h["parts"].items(), key=lambda kv: -kv[1])]
    md += ["", "DFL rows (in graph order):", "", "| ID | OpType | Target | time us |", "|---|---|---|---|"]
    md += [f"| {d['id']} | {d['op']} | {d['target']} | {d['time']:.0f} |" for d in h["dfl_chain"]]
    md += ["", "## 4. Time share against GFLOPs share", "", "| stage | time us | time share % | GFLOPs (Conv2d) | GFLOPs share % |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['time_us']:.0f} | {100 * v['time_share']:.1f} | {v['gflops']:.3f} | {100 * v['gflops_share']:.1f} |" for k, v in stage_rows.items()]
    md += ["", "## 5. First eval_perf call against the others (per session)", "", "| session | latency first ms | latency rest median ms | debug total first ms | debug total rest median ms |", "|---|---|---|---|---|"]
    md += [f"| {i + 1} | {w['latency_first']:.3f} | {w['latency_rest_median']:.3f} | {w['debug_first'] / 1000:.3f} | {w['debug_rest_median'] / 1000:.3f} |" for i, w in enumerate(warm)]
    if delta:
        md += ["", "## 6. Difference of each session against the dry run (B2), per layer", "",
               "Regression of (time in session - time in B2) over the 119 layers on 1, est. cycles time (us) and RW (KB). "
               "A constant intercept with slopes near zero means a per-layer overhead that does not depend on size.", "",
               "| session | total delta us | mean delta per layer us | intercept us | per est. us | us per MB | R2 |", "|---|---|---|---|---|---|---|"]
        md += [f"| {d['session']} | {d['total_delta']:.0f} | {d['per_row_mean']:.1f} | {d['intercept']:.1f} | {d['per_est_us']:.3f} | {d['us_per_mb']:.1f} | {d['r2']:.3f} |" for d in delta]
        md += ["", "Mean delta per layer by group (us):", "", "| session | " + " | ".join(delta[0]["by_group"]) + " |", "|---|" + "---|" * len(delta[0]["by_group"])]
        md += [f"| {d['session']} | " + " | ".join(f"{d['by_group'].get(k, float('nan')):.1f}" for k in delta[0]["by_group"]) + " |" for d in delta]
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print((OUT_DIR / f"{name}.md").read_text())


if __name__ == "__main__":
    main()
