"""CLI: X8 analysis, FP16 against INT8 under protocol P2 (interleaved A B A B A B), statements of experiments/npu/latency/onboard_x8/rule.txt.
Gates come from aggregate_layers.py run before this script with --precision int8|fp16 --bench-subdir onboard_x8/<prec> --tag x8.
The helper log is copied to experiments/npu/latency/onboard_x8/hold/ first.

    .venv/bin/python -m ai.npu.latency.analyze_x8 <run_id>
Writes results/npu/latency/onboard_x8_<run_id>_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import statistics
from ai.paths import LATENCY_RESULTS_DIR, bench_path, experiment_dir

LINES = ("36", "37", "38")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    d = experiment_dir(args.run_id, "onboard_x8")
    entries = json.loads((d / "sessions.json").read_text())
    hlog = [ln.split() for ln in (d / "hold" / "log.txt").read_text().splitlines() if ln.strip()]
    hlog = [(int(e), m, int(nt), int(nv)) for e, m, nt, nv in hlog]
    rows = []
    for e in entries:
        win = [x for x in hlog if e["epoch_start"] <= x[0] <= e["epoch_end"]]
        delta = [sum(e["irq_after"][ln][c] - e["irq_before"][ln][c] for ln in LINES) for c in range(8)]
        ok = bool(win) and all(m == "pin" and nv == 0 for _, m, _, nv in win) and sum(delta) > 0 and delta[4] >= 0.99 * sum(delta)
        rec = json.loads((bench_path(args.run_id, e["subdir"]) / f"tier1_{e['precision']}_CORE_0_sess{e['session']}.json").read_text())
        rows.append(dict(precision=e["precision"], session=e["session"], realized=ok, latency_ms=rec["npu_latency_ms"], latency_all=rec["npu_latency_ms_all"],
                         debug_total_ms=rec["debug_total_us"] / 1000, irq_per_cpu=delta))
    all_ok = all(r["realized"] for r in rows)
    lat = {p: statistics.median(r["latency_ms"] for r in rows if r["precision"] == p) for p in ("int8", "fp16")}
    gates, shares = {}, {}
    for p in ("int8", "fp16"):
        g = json.loads((LATENCY_RESULTS_DIR / f"latency_per_block_{args.run_id}_{p}_CORE_0_x8.json").read_text())
        gates[p] = {c["code"]: c["ok"] for c in g["checks"]}
        shares[p] = {b: v["share"] for b, v in g["blocks"].items()}
    big = [b for b in shares["int8"] if shares["int8"][b] >= 0.02]
    out = dict(sessions=rows, all_realized=all_ok, gates=gates,
               latency_ms=lat if all_ok else "chưa kết luận (P2 không thực hiện đúng)",
               fp16_minus_int8_ms=(lat["fp16"] - lat["int8"]) if all_ok else None, fp16_over_int8=(lat["fp16"] / lat["int8"]) if all_ok else None,
               block_shares={b: dict(int8=shares["int8"][b], fp16=shares["fp16"].get(b)) for b in big},
               rule=(d / "rule.txt").read_text(), driver_log=(d / "driver.log").read_text(), hold_progress=(d / "hold" / "progress.txt").read_text())
    name = f"onboard_x8_{args.run_id}_CORE_0"
    (LATENCY_RESULTS_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = [f"# X8: FP16 against INT8 under P2, {args.run_id}", "", "| precision | session | realized | latency ms | debug total ms |", "|---|---|---|---|---|"]
    md += [f"| {r['precision']} | {r['session']} | {r['realized']} | {r['latency_ms']:.3f} | {r['debug_total_ms']:.3f} |" for r in rows]
    md += ["", f"Gates: {gates}", f"Latency medians: {out['latency_ms']}, FP16 minus INT8 {out['fp16_minus_int8_ms']} ms, ratio {out['fp16_over_int8']}", "",
           "| block | INT8 share | FP16 share |", "|---|---|---|"]
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    md += [f"| {b} | {fmt(v['int8'])} | {fmt(v['fp16'])} |" for b, v in out["block_shares"].items()]
    (LATENCY_RESULTS_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
