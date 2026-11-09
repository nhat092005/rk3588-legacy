"""CLI: X9 analysis, does irqbalance rewrite the NPU interrupt affinity during a run (E2 condition B with irqbalance active),
statements of experiments/npu/latency/onboard_x9/rule.txt.

    .venv/bin/python -m ai.npu.latency.analyze_x9 <run_id>
Writes results/npu/latency/onboard_x9_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import json

from ai.npu.latency.analyze_x3 import irq_delta
from ai.paths import LATENCY_RESULTS_DIR, RUNS_DIR, experiment_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    d = experiment_dir(RUNS_DIR / args.run_id, "onboard_x9")
    trials = []
    for t in (1, 2, 3):
        lines = [ln.split() for ln in (d / f"aff_{t}.txt").read_text().splitlines() if ln.strip()]
        line36 = [next(x for x in ln[1:] if x.startswith("36:")) for ln in lines]
        changed = [x for x in line36 if x != "36:4/4"]
        irqd = irq_delta(d / f"irq_pre_{t}.txt", d / f"irq_post_{t}.txt")
        trials.append(dict(trial=t, samples=len(line36), changed_samples=len(changed), first_changed=changed[0] if changed else None,
                           irq_per_cpu=irqd, cpu4_share=irqd[4] / sum(irqd)))
    n_changed = sum(1 for x in trials if x["changed_samples"] > 0)
    if n_changed:
        verdict, label = f"irqbalance đổi affinity ngắt NPU trong lúc chạy (đã quan sát ở {n_changed}/3 lần)", "ĐO"
    elif all(x["cpu4_share"] >= 0.99 for x in trials):
        verdict, label = "không quan sát thấy irqbalance đổi affinity trong 3 lần 60 giây", "CHƯA BIẾT"
    else:
        verdict, label = "chưa kết luận", "CHƯA BIẾT"
    out = dict(trials=trials, verdict=verdict, label_U4=label, progress=(d / "progress.txt").read_text(), rule=(d / "rule.txt").read_text())
    name = f"onboard_x9_{args.run_id}_int8_CORE_0"
    (LATENCY_RESULTS_DIR / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    md = [f"# X9: irqbalance and the NPU interrupt affinity, {args.run_id}", "", "| trial | samples | samples not 4/4 | first change | CPU4 share of interrupts |", "|---|---|---|---|---|"]
    md += [f"| {x['trial']} | {x['samples']} | {x['changed_samples']} | {x['first_changed']} | {x['cpu4_share']:.3f} |" for x in trials]
    md += ["", f"Verdict: {verdict}. U4: {label}."]
    (LATENCY_RESULTS_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
