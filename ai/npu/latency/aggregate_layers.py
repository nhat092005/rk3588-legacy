"""CLI: per-block latency of a run from its tier-1 sessions (docs 03-phuong-phap/02-tang-npu-nen-do/01-ke-hoach-do).

Reads runs/<run_id>/bench/tier1_<precision>_<core_mask>_sess<N>.json and the matching *_layers.json written by
benchmark_npu with --session N, assigns every RKNN layer to a YOLOv8 layer L0-L22 from its ONNX node name
(ai/core/rknn/blocks.py), and writes results/npu/latency/latency_per_block_<run_id>_<precision>_<core_mask>.{md,json}.
Exits 1 when an acceptance check (A1-A7 of the docs page) fails; the tables are still written.
"""
import argparse
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.automation.bench.benchmark_npu import RUNS_DIR
from ai.core.rknn.benchmark import LAYERS_TIME_RE, parse_frequencies
from ai.core.rknn.blocks import assign
from ai.paths import REPO_ROOT, LATENCY_RESULTS_DIR, RUNS_DIR, bench_path

OUT_DIR = LATENCY_RESULTS_DIR
# Frequencies the board reports after `lock_freq.sh lock` (lock_freq.sh status, 2026-10-30)
LOCKED_FREQ = {"CPU": [1800000, 2256000, 2304000], "NPU": [1000000000], "DDR": [2112000000]}
LATENCY_GAP_MAX = 0.05  # docs check A5, same 5% as gate G4
SHARE_MIN, SHARE_SPREAD_MAX = 0.02, 0.10  # docs check A8: blocks above 2% of the time must keep their share within 10% between sessions
CORE0_WORKLOAD = "100.0%/0.0%/0.0%"


def block_order(block: str) -> tuple:
    return (0, int(block[1:])) if block.startswith("L") else (1, block)


def med(values):
    return statistics.median(values)


def spread(values) -> float:
    return (max(values) - min(values)) / med(values) if med(values) else 0.0


def linfit(x, y):
    """Least squares y = slope * x + intercept; returns (slope, intercept)."""
    a = np.vstack([x, np.ones(len(x))]).T
    (slope, intercept), *_ = np.linalg.lstsq(a, np.array(y, dtype=float), rcond=None)
    return float(slope), float(intercept)


def build_table_rows(log_path: Path) -> list[tuple[str, str]]:
    """(OpType, FullName) of the 'Network Layer Information Table' printed by the RKNN build (RKNN_LOG_LEVEL=3)."""
    lines = [re.sub(r"^.*?\] ", "", ln) for ln in log_path.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("ID") and "OpType" in ln)
    rows = []
    for ln in lines[start + 2:]:
        parts = ln.split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[3] in ("CPU", "NPU", "GPU"):
            rows.append((parts[1], parts[-1]))
        elif rows and ln.startswith("---"):
            break
    return rows


def layer_gflops(weights: Path) -> dict[str, float]:
    """GFLOPs (2 x Conv2d MACs, 640x640) of every YOLOv8 layer, same counting as docs 01-tang-ai/02-kien-truc-model."""
    from ultralytics import YOLO
    model = YOLO(weights).model.float().eval()
    macs, cur = collections.Counter(), [None]
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(lambda m, i, o: macs.update({cur[0]: o.shape[1] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * o.shape[2] * o.shape[3]}))
    for idx, layer in enumerate(model.model):
        layer.register_forward_pre_hook(lambda m, i, idx=idx: cur.__setitem__(0, f"L{idx}"))
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    return {k: 2 * v / 1e9 for k, v in macs.items()}


def load_sessions(bench: Path, stem: str) -> list[tuple[dict, dict, str]]:
    sessions = []
    for path in sorted(glob.glob(str(bench / f"{stem}_sess*.json")), key=lambda p: int(re.search(r"_sess(\d+)\.json$", p).group(1)) if re.search(r"_sess(\d+)\.json$", p) else -1):
        if path.endswith("_layers.json"):
            continue
        record = json.loads(Path(path).read_text())
        layers = json.loads(Path(path.replace(".json", "_layers.json")).read_text())
        text = Path(path.replace(".json", "_layers.txt")).read_text()
        sessions.append((record, layers, text))
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--precision", default="int8", choices=["fp16", "int8"])
    parser.add_argument("--core-mask", default="CORE_0")
    parser.add_argument("--bench-subdir", default="", help="read the sessions from bench/<subdir> (for example onboard_x4/p2) instead of bench/")
    parser.add_argument("--tag", default="", help="suffix of the output name, for example p2")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    stem = f"tier1_{args.precision}_{args.core_mask}"
    bench = bench_path(run_dir, args.bench_subdir)
    sessions = load_sessions(bench, stem)
    if not sessions:
        sys.exit(f"no {stem}_sess*.json in {bench}")

    # Every quantity is computed per eval_perf call first (sum of the rows of one call), then median over the calls of a
    # session, then median over sessions. A sum of per-row medians is not the median of per-call sums.
    def build(record, layers):
        rows, n_calls = layers["layers"], layers["repeats"]
        calls = []
        for j in range(n_calls):
            blk, prt, op = collections.Counter(), collections.Counter(), collections.Counter()
            for r in rows:
                block, part = assign(r)
                blk[block] += r["time_us_all"][j]
                prt[(block, part)] += r["time_us_all"][j]
                op[r["op"]] += r["time_us_all"][j]
            calls.append(dict(block=blk, part=prt, op=op, total=sum(blk.values())))
        cycles = collections.Counter()
        for r in rows:
            cycles[assign(r)[0]] += r["total_cycles"]
        n_rows = collections.Counter(assign(r)[0] for r in rows)
        # per-row additive overhead of the session: intercept of time = a + b * est. cycles time over the Conv rows
        conv = [r for r in rows if r["npu_cycles"] > 0]
        a = max(linfit([r["total_cycles"] / 1000 for r in conv], [r["time_us"] for r in conv])[1], 0.0)
        adj = collections.Counter()
        for r in rows:
            adj[assign(r)[0]] += max(r["time_us"] - a, 0) if r["time_us"] > 0 else 0

        def over_calls(group, key, per_total=False):
            vals = [c[group][key] / (c["total"] if per_total else 1) for c in calls]
            return med(vals), spread(vals)

        return dict(rows=rows, calls=calls, cycles=cycles, over_calls=over_calls, n_rows=n_rows, row_intercept=a,
                    share_adj={b: adj[b] / (sum(adj.values()) or 1) for b in adj},
                    total=med([c["total"] for c in calls]), debug_total=record["debug_total_us"],
                    latency_us=record["npu_latency_ms"] * 1000,
                    sum_ratio=[c["total"] / t for c, t in zip(calls, layers["debug_total_us_all"])])

    # Every quantity is computed per eval_perf call first (sum of the rows of one call), then median over the calls of a
    # session, then median over sessions. A sum of per-row medians is not the median of per-call sums.
    per_session = [build(record, layers) for record, layers, _ in sessions]
    # the unsuffixed dry run (B2) has the same model, lock and repeats: it is a session, reported next to the others
    dry_json = bench / f"{stem}.json"
    dry = build(json.loads(dry_json.read_text()), json.loads((bench / f"{stem}_layers.json").read_text())) if dry_json.exists() else None

    blocks = sorted({b for s in per_session for c in s["calls"] for b in c["block"]}, key=block_order)
    parts = sorted({k for s in per_session for c in s["calls"] for k in c["part"]}, key=lambda k: (block_order(k[0]), k[1]))
    ops = sorted({o for s in per_session for c in s["calls"] for o in c["op"]})
    gflops = layer_gflops(run_dir / "weights" / "best.pt")
    cycles_total = sum(per_session[0]["cycles"].values()) or 1

    def across(group, key, sess=None):
        """us, share and spreads over the sessions (default: the numbered sessions, without the dry run)"""
        sess = sess or per_session
        us = [s["over_calls"](group, key)[0] for s in sess]
        sh = [s["over_calls"](group, key, per_total=True)[0] for s in sess]
        call_spread = max(s["over_calls"](group, key)[1] for s in sess)
        return dict(us=med(us), spread=spread(us), share=med(sh), share_spread=spread(sh), call_spread=call_spread)

    block_stats = {}
    for b in blocks:
        block_stats[b] = across("block", b)
        block_stats[b].update(cycle_share=per_session[0]["cycles"][b] / cycles_total, gflops=gflops.get(b),
                              us_per_gflop=(block_stats[b]["us"] / gflops[b]) if gflops.get(b) else None,
                              rows=per_session[0]["n_rows"][b], share_sens=med([x["share_adj"].get(b, 0) for x in per_session]))
        if dry:
            block_stats[b].update(b2_us=dry["over_calls"]("block", b)[0], b2_share=dry["over_calls"]("block", b, per_total=True)[0])
    part_stats = {}
    for k in parts:
        part_stats[k] = across("part", k)
        part_stats[k]["share_in_block"] = med([s["over_calls"]("part", k)[0] / (s["over_calls"]("block", k[0])[0] or 1) for s in per_session])
    op_stats = {o: dict(across("op", o), calls=sum(1 for r in per_session[0]["rows"] if r["op"] == o)) for o in ops}

    latency_us = [s["latency_us"] for s in per_session]
    debug_totals = [s["debug_total"] for s in per_session]
    sum_ratios = [(min(s["sum_ratio"]), max(s["sum_ratio"])) for s in per_session]
    summary = dict(latency_ms_sessions=[round(v / 1000, 3) for v in latency_us], latency_ms=round(med(latency_us) / 1000, 3),
                   latency_gap=round(spread(latency_us), 4), debug_total_ms_sessions=[round(v / 1000, 3) for v in debug_totals],
                   debug_total_ms=round(med(debug_totals) / 1000, 3), debug_total_gap=round(spread(debug_totals), 4),
                   debug_ratio=round(med(debug_totals) / med(latency_us), 3),
                   sum_ratio_min_max_per_session=[[round(lo, 4), round(hi, 4)] for lo, hi in sum_ratios], n_sessions=len(per_session))
    # reference points kept next to the sessions, not in the statistics: the unsuffixed dry run (B2) and the first probe
    reference = {}
    if dry_json.exists():
        d = json.loads(dry_json.read_text())
        reference["dry_run_B2"] = dict(latency_ms=d["npu_latency_ms"], debug_total_ms=round(d["debug_total_us"] / 1000, 3))
    for probe in sorted(bench.glob("extra_probe_debug_*.txt")):
        m = LAYERS_TIME_RE.search(probe.read_text())
        reference[probe.stem] = dict(debug_total_ms=round(float(m.group(1)) / 1000, 3))
    summary["reference_points"] = reference
    summary["row_intercept_us"] = [round(x["row_intercept"], 1) for x in per_session] + ([round(dry["row_intercept"], 1)] if dry else [])
    tier2 = bench / f"tier2_{args.precision}_{args.core_mask}.json"
    if tier2.exists():
        t2 = json.loads(tier2.read_text())["latency_ms"]["npu"]
        summary["tier2_npu_ms"] = {k: t2[k] for k in ("mean", "p50", "p95") if k in t2}

    # acceptance checks A1-A7 of the docs page
    checks = []
    export_logs = sorted((run_dir / "logs").glob(f"export_{args.precision}_*.log"))
    if export_logs:
        build = build_table_rows(export_logs[-1])
        same = all(build == [(r["op"], r["name"]) for r in s["rows"]] for s in per_session)
        checks.append(("A1", same, f"build table {len(build)} rows vs board table {len(per_session[0]['rows'])} rows, (OpType, FullName) sequence identical: {same}"))
    else:
        checks.append(("A1", False, "no export log to compare with"))
    freqs = [parse_frequencies(text) for _, _, text in sessions] + [r["frequencies_during_eval_perf"] for r, _, _ in sessions]
    checks.append(("A2", all(f == LOCKED_FREQ for f in freqs), f"{len(freqs)} frequency reports equal to {LOCKED_FREQ}"))
    if args.core_mask == "CORE_0":
        bad = [r["id"] for s in per_session for r in s["rows"] if r["target"] == "NPU" and r["op"] != "Reshape" and r["time_us"] > 0 and r["workload"] != CORE0_WORKLOAD]
        checks.append(("A3", not bad, f"NPU rows (not Reshape) with time > 0 and WorkLoad != {CORE0_WORKLOAD}: {sorted(set(bad))}"))
    odd = sorted({(assign(r)[0], r["op"]) for s in per_session for r in s["rows"] if assign(r)[0] in ("io", "post_detect") or assign(r)[1].startswith("other")})
    known = {("io", "InputOperator"), ("io", "OutputOperator"), ("post_detect", "Mul"), ("post_detect", "Concat"), ("post_detect", "Reshape")}
    checks.append(("A4", set(odd) <= known, f"rows outside model.N or with an unknown part: {odd}"))
    if len(per_session) > 1:
        checks.append(("A5", summary["latency_gap"] <= LATENCY_GAP_MAX, f"latency gap between {len(per_session)} sessions {summary['latency_gap']:.2%} (max {LATENCY_GAP_MAX:.0%})"))
    else:
        checks.append(("A5", False, "only one session, gap not evaluable"))
    checks.append(("A6", abs(sum(b["share"] for b in block_stats.values()) - 1) < 0.01, f"sum of block shares {sum(b['share'] for b in block_stats.values()):.4f}, sum_ratio min/max per session {summary['sum_ratio_min_max_per_session']}, debug_ratio {summary['debug_ratio']}"))
    # A8: stability of the quantities that are published (debug-mode total and the block shares), not only of the latency.
    # Evaluated twice: on the numbered sessions, and with the dry run (B2) counted as one more session.
    def a8(sess, totals):
        gap = spread(totals)
        unstable = {}
        for b in blocks:
            st = across("block", b, sess)
            if st["share"] >= SHARE_MIN and st["share_spread"] > SHARE_SPREAD_MAX:
                unstable[b] = round(st["share_spread"], 3)
        return gap, unstable
    if len(per_session) > 1:
        gap, unstable = a8(per_session, debug_totals)
        checks.append(("A8", gap <= LATENCY_GAP_MAX and not unstable,
                       f"sessions only: debug total gap {gap:.2%} (max {LATENCY_GAP_MAX:.0%}), blocks with share >= {SHARE_MIN:.0%} whose share spread exceeds {SHARE_SPREAD_MAX:.0%}: {unstable or 'none'}"))
        if dry:
            gap2, unstable2 = a8(per_session + [dry], debug_totals + [dry["debug_total"]])
            checks.append(("A8+B2", gap2 <= LATENCY_GAP_MAX and not unstable2,
                           f"with the dry run: debug total gap {gap2:.2%}, blocks over {SHARE_SPREAD_MAX:.0%}: {unstable2 or 'none'}"))
    else:
        checks.append(("A8", False, "only one session, stability not evaluable"))
    need = ("model_sha256", "git_commit", "git_dirty", "sdk_version")
    checks.append(("A7", all(k in r for r, _, _ in sessions for k in need), f"provenance keys {need} present in every session file"))

    OUT_DIR.mkdir(exist_ok=True)
    name = f"latency_per_block_{args.run_id}_{args.precision}_{args.core_mask}" + (f"_{args.tag}" if args.tag else "")
    (OUT_DIR / f"{name}.json").write_text(json.dumps(dict(
        summary=summary, checks=[dict(code=c, ok=ok, detail=d) for c, ok, d in checks],
        blocks=block_stats, parts={f"{b}/{p}": v for (b, p), v in part_stats.items()}, ops=op_stats,
        model_sha256=sessions[0][0]["model_sha256"], git_commit=sessions[0][0]["git_commit"]), indent=1) + "\n")

    f = lambda v, n=1: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = [f"# Latency per block: {args.run_id} {args.precision} {args.core_mask}", "",
          f"{summary['n_sessions']} independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions "
          f"(perf_debug=True, so debug-mode time, not latency). model sha256 `{sessions[0][0]['model_sha256'][:12]}`, commit `{sessions[0][0]['git_commit'][:8]}`.", "",
          "## Totals", "",
          f"- latency (perf_debug=False), ms per session: {summary['latency_ms_sessions']}, median {summary['latency_ms']}, gap {summary['latency_gap']:.2%}",
          f"- debug total (Total Operator Elapsed Per Frame Time), ms per session: {summary['debug_total_ms_sessions']}, median {summary['debug_total_ms']}, gap {summary['debug_total_gap']:.2%}",
          f"- debug_ratio (debug total / latency): {summary['debug_ratio']}, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: {summary['sum_ratio_min_max_per_session']}"]
    if "tier2_npu_ms" in summary:
        md.append(f"- tier 2 (RKNNLite inference wall time) npu ms: {summary['tier2_npu_ms']}")
    md.append(f"- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: {summary['row_intercept_us']}. "
              "The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.")
    if summary["reference_points"]:
        md.append(f"- reference points, not in the statistics: {summary['reference_points']}")
    md += ["", "## Table 1: block L0-L22", "", "| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {b} | {s['rows']} | {f(s['us'], 0)} | {f(s.get('b2_us'), 0)} | {s['spread']:.1%} | {s['call_spread']:.1%} | {s['share'] * 100:.1f} | {f(None if s.get('b2_share') is None else s['b2_share'] * 100)} | {s['share_sens'] * 100:.1f} | {s['share_spread']:.1%} | {s['cycle_share'] * 100:.1f} | {f(s['gflops'], 3)} | {f(s['us_per_gflop'], 0)} |" for b, s in block_stats.items()]
    md += ["", "## Table 2: part inside block", "", "| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |", "|---|---|---|---|---|---|---|"]
    md += [f"| {b} | {p} | {f(s['us'], 0)} | {s['spread']:.1%} | {s['call_spread']:.1%} | {s['share'] * 100:.1f} | {s['share_in_block'] * 100:.1f} |" for (b, p), s in part_stats.items()]
    md += ["", "## Table 3: by OpType", "", "| OpType | Calls | us (median) | share % |", "|---|---|---|---|"]
    md += [f"| {o} | {s['calls']} | {f(s['us'], 0)} | {s['share'] * 100:.2f} |" for o, s in sorted(op_stats.items(), key=lambda kv: -kv[1]["us"])]
    md += ["", "## Acceptance checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    md += [f"| {c} | {'pass' if ok else 'FAIL'} | {d} |" for c, ok, d in checks]
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print((OUT_DIR / f"{name}.md").read_text())
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
