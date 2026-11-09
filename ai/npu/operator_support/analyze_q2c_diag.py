"""CLI: 2c-b analysis, statements D1 to D3 of experiments/npu/operator_support/onboard_q2c/rule_diag.txt.

    .venv/bin/python -m ai.npu.operator_support.analyze_q2c_diag
Reads experiments/npu/operator_support/onboard_q2c/diag/<run>/analysis_<k>/ (error_analysis.txt, golden/output0.txt, simulator/output0.txt). Writes results/npu/operator_support/q2c_diag.{json,md}.
"""
import json
import re

import numpy as np
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

HOME = experiment_dir(RUNS_DIR / "ppe5_yolov8n_baseline_100ep_s0", "onboard_q2c")
RUNS = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0"}
ROW = re.compile(r"\[(\w+)\] (\S+)\s+([\d.]+) \| ([\d.]+)\s+([\d.]+) \| ([\d.]+)")


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    out = {}
    for m, r in RUNS.items():
        base = HOME / "diag" / r
        images = [ln.split(" ", 1) for ln in (base / "images.txt").read_text().splitlines()]
        n = len(images)
        layers = {}
        outs = []
        for k in range(n):
            d = base / f"analysis_{k}"
            for ln in (d / "error_analysis.txt").read_text().splitlines():
                mm = ROW.match(ln)
                if mm:
                    layers.setdefault((mm.group(1), mm.group(2)), []).append((float(mm.group(3)), float(mm.group(5))))
            g = np.loadtxt(d / "golden" / "output0.txt").reshape(9, 8400)
            s = np.loadtxt(d / "simulator" / "output0.txt").reshape(9, 8400)
            row = dict(image=images[k][0], class_cos=cos(g[4:], s[4:]), box_cos=cos(g[:4], s[:4]), class_max_abs=float(np.abs(g[4:] - s[4:]).max()))
            gm, sm = g[4:].max(0), s[4:].max(0)
            for t in (0.25, 0.001):
                row[f"t{t}"] = dict(golden_above=int((gm >= t).sum()), missed=int(((gm >= t) & (sm < t)).sum()), extra=int(((sm >= t) & (gm < t)).sum()))
            outs.append(row)
        assert all(len(v) == n for v in layers.values())
        mean_single = {k: float(np.mean([x[1] for x in v])) for k, v in layers.items()}
        mean_entire = {k: float(np.mean([x[0] for x in v])) for k, v in layers.items()}
        lowest = sorted(mean_single.items(), key=lambda kv: kv[1])[:10]
        last = list(layers)[-1]
        cc, bc = [o["class_cos"] for o in outs], [o["box_cos"] for o in outs]
        verdict = "lớp class mất nhiều hơn lớp box" if all(c < b for c, b in zip(cc, bc)) else ("lớp box mất nhiều hơn lớp class" if all(c > b for c, b in zip(cc, bc)) else "chưa kết luận")
        out[m] = dict(n_images=n, images=outs, lowest_single=[dict(op=k[0], name=k[1], mean_single_cos=v) for k, v in lowest], last_row=dict(op=last[0], name=last[1], mean_entire_cos=mean_entire[last]),
                      mean_class_cos=float(np.mean(cc)), mean_box_cos=float(np.mean(bc)), d3=verdict,
                      mean_counts={t: {k: float(np.mean([o[t][k] for o in outs])) for k in ("golden_above", "missed", "extra")} for t in ("t0.25", "t0.001")})
    out["rule"] = (HOME / "rule_diag.txt").read_text()
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2c_diag.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# 2c-b INT8 diagnosis (simulator)", ""]
    for m in RUNS:
        v = out[m]
        md += [f"## {m}", f"class cos {v['mean_class_cos']:.5f}, box cos {v['mean_box_cos']:.5f}, D3: {v['d3']}", f"counts: {v['mean_counts']}", "lowest single cos:"]
        md += [f"- {x['op']} {x['name']} {x['mean_single_cos']:.5f}" for x in v["lowest_single"]]
        md.append("")
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2c_diag.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
