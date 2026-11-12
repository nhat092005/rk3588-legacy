"""CLI: C3 versus C2f on YOLOv8n with PPE5, accuracy (FP32 leaderboard) and latency (on-board C bench), read against rule.txt.

Reads experiments/npu/operator_support/c3_vs_c2f/board/ (the board's ~/c3c2f/out copied by the operator: c0_<arm>_r<n>.csv,
c0_<arm>_d<n>_detail.txt, c7_<arm>_r<n>.csv, c7_<arm>_d1_detail.txt, progress.txt) and experiments/leaderboards/ppe5.csv.
Arm A is C2f (yolov8n_scratch), arm B is C3 (yolov8n_c3_scratch). Writes results/npu/operator_support/c3_vs_c2f.{md,json}.
Run: .venv/bin/python -m ai.npu.operator_support.analyze_c3_vs_c2f [--board-dir DIR]
The reading tables are the ones fixed in experiments/npu/operator_support/c3_vs_c2f/rule.txt before any C3 data existed.
"""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

from ai.core.rknn.benchmark import parse_layers

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/npu/operator_support/c3_vs_c2f"
LEADERBOARD = ROOT / "experiments/leaderboards/ppe5.csv"
OUT = ROOT / "results/npu/operator_support"
SEED_SPREAD_PP = 1.35  # rule_split.txt, the floor of the accuracy spread S
ARMS = {"A": "yolov8n_scratch", "B": "yolov8n_c3_scratch"}
EVAL_SETS = ("css_test", "ppe2286_ext")


def med(v):
    return statistics.median(v)


def round_median_ms(path):
    rows = list(csv.DictReader(open(path)))
    return med([int(r["run_duration_us"]) for r in rows]) / 1000, med([int(r["wall_us"]) for r in rows]) / 1000, len(rows)


def latency(board, prefix):
    """Per-round medians (ms) of each arm for tags <prefix>_<arm>_r<n>, in round order."""
    rounds = collections.defaultdict(dict)
    for p in sorted(board.glob(f"{prefix}_?_r*.csv")):
        arm, n = re.fullmatch(rf"{prefix}_([AB])_r(\d+)", p.stem).groups()
        rounds[int(n)][arm] = round_median_ms(p)
    n_rounds = sorted(r for r in rounds if len(rounds[r]) == 2)
    a = [rounds[r]["A"][0] for r in n_rounds]
    b = [rounds[r]["B"][0] for r in n_rounds]
    if not n_rounds:
        return None
    ratios = [y / x for x, y in zip(a, b)]
    noise = max(max(v) / min(v) - 1 for v in (a, b))
    mr = med(ratios)
    reading = ("C3 faster" if mr < 1 - noise else "C3 slower" if mr > 1 + noise else "not distinguishable in this session")
    return dict(rounds=n_rounds, A_ms=a, B_ms=b, ratio=ratios, median_A=med(a), median_B=med(b), median_ratio=mr,
                ratio_min=min(ratios), ratio_max=max(ratios), noise_floor=noise, reading=reading,
                wall_A=[rounds[r]["A"][1] for r in n_rounds], wall_B=[rounds[r]["B"][1] for r in n_rounds])


def detail(board, prefix):
    """Per arm: op counts, median total time per op type and share of the layer-time sum, from the per-layer tables of every file."""
    out = {}
    for arm in "AB":
        files = sorted(board.glob(f"{prefix}_{arm}_d*_detail.txt"))
        per_file = []
        for f in files:
            tables = [t for t in re.split(r"=== run \d+ ===", f.read_text()) if "Network Layer Information Table" in t]
            ops = collections.defaultdict(lambda: [0, []])  # op -> [count, per-run total us]
            totals, wl = [], collections.Counter()
            for t in tables:
                rows = parse_layers(t)
                tot = sum(r["time_us"] or 0 for r in rows)
                totals.append(tot)
                per_run = collections.defaultdict(float)
                cnt = collections.Counter()
                for r in rows:
                    per_run[r["op"]] += r["time_us"] or 0
                    cnt[r["op"]] += 1
                    if r["op"] in ("Split", "Concat", "Add"):
                        shares = [float(x.rstrip("%")) for x in (r["workload"] or "0%/0%/0%").split("/")]
                        wl[(r["op"], "single core" if max(shares) >= 99.0 else "split over cores")] += 1
                for op in per_run:
                    ops[op][0] = cnt[op]
                    ops[op][1].append(per_run[op])
            per_file.append(dict(file=f.name, runs=len(tables), total_us=med(totals),
                                 ops={op: dict(count=c, us=med(v)) for op, (c, v) in ops.items()},
                                 workload={f"{k[0]}|{k[1]}": n for k, n in wl.items()}))
        if per_file:
            agg = {}
            for op in {o for pf in per_file for o in pf["ops"]}:
                agg[op] = dict(count=per_file[0]["ops"].get(op, {}).get("count", 0),
                               us=med([pf["ops"].get(op, {"us": 0})["us"] for pf in per_file]))
            total = med([pf["total_us"] for pf in per_file])
            for op in agg:
                agg[op]["share"] = agg[op]["us"] / total
            out[arm] = dict(files=per_file, total_us=total, ops=agg)
    return out


def accuracy():
    rows = [r for r in csv.DictReader(open(LEADERBOARD))]
    res = {}
    for es in EVAL_SETS:
        res[es] = {}
        for arm, model in ARMS.items():
            sel = sorted((r for r in rows if r["model"] == model and r["epochs"] == "200" and r["eval_set"] == es), key=lambda r: int(r["seed"]))
            res[es][arm] = dict(seeds=[int(r["seed"]) for r in sel],
                                ap50_95=[100 * float(r["map50_95"]) for r in sel],
                                nohelmet=[100 * float(r["ap50_95_NoHelmet"]) for r in sel],
                                vest=[100 * float(r["ap50_95_Vest"]) for r in sel])
    out = {}
    for es in EVAL_SETS:
        a, b = res[es]["A"], res[es]["B"]
        entry = dict(arms=res[es])
        if len(a["ap50_95"]) >= 3 and len(b["ap50_95"]) >= 3:
            spread = max(SEED_SPREAD_PP, max(x["ap50_95"] and (max(x["ap50_95"]) - min(x["ap50_95"])) for x in (a, b)))
            d = statistics.mean(b["ap50_95"]) - statistics.mean(a["ap50_95"])
            entry.update(D=d, S=spread, mean_A=statistics.mean(a["ap50_95"]), mean_B=statistics.mean(b["ap50_95"]),
                         D_nohelmet=statistics.mean(b["nohelmet"]) - statistics.mean(a["nohelmet"]),
                         D_vest=statistics.mean(b["vest"]) - statistics.mean(a["vest"]),
                         reading=("C3 loses accuracy" if d <= -spread else "C3 gains accuracy" if d >= spread else "not distinguishable with three seeds"))
        else:
            entry["reading"] = f"incomplete: {len(a['ap50_95'])} A seeds, {len(b['ap50_95'])} B seeds"
        out[es] = entry
    return out


def complexity():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops
    res = {}
    for arm, path in (("A", "yolov8n.yaml"), ("B", str(ROOT / "ai/models/yolov8n_c3_scratch/yolov8n_c3_scratch.yaml"))):
        m = DetectionModel(path, nc=5, verbose=False)
        res[arm] = dict(params=sum(p.numel() for p in m.parameters()), gflops=get_flops(m, 640))
    return res


def fmt(v, n=2):
    return f"{v:.{n}f}"


def write(out):
    L = ["# C3 versus C2f, YOLOv8n, PPE5", "", "Reading tables fixed in `experiments/npu/operator_support/c3_vs_c2f/rule.txt` before any C3 data. Arm A = C2f (`yolov8n_scratch`), arm B = C3 (`yolov8n_c3_scratch`).", ""]
    c = out.get("complexity")
    if c:
        L += ["## Size", "", "| arm | parameters | GFLOPs (ultralytics get_flops, 640) |", "|---|---|---|",
              f"| A C2f | {c['A']['params']:,} | {fmt(c['A']['gflops'], 3)} |", f"| B C3 | {c['B']['params']:,} | {fmt(c['B']['gflops'], 3)} |", ""]
    L += ["## Accuracy (FP32, AP50-95 in pp)", ""]
    for es, e in out["accuracy"].items():
        L += [f"### {es}", "", "| arm | seeds | AP50-95 | NoHelmet | Vest |", "|---|---|---|---|---|"]
        for arm in "AB":
            a = e["arms"][arm]
            L.append(f"| {arm} | {a['seeds']} | {', '.join(fmt(x) for x in a['ap50_95'])} | {', '.join(fmt(x) for x in a['nohelmet'])} | {', '.join(fmt(x) for x in a['vest'])} |")
        if "D" in e:
            L += ["", f"Mean A {fmt(e['mean_A'])}, mean B {fmt(e['mean_B'])}, D = B minus A = {fmt(e['D'])} pp, S = {fmt(e['S'])} pp, D NoHelmet {fmt(e['D_nohelmet'])}, D Vest {fmt(e['D_vest'])}. Reading: **{e['reading']}**.", ""]
        else:
            L += ["", f"Reading: {e['reading']}.", ""]
    for key, title in (("latency_core0", "Latency, CORE_0, taskset -c 4 (cell a4i4)"), ("latency_core012", "Latency, CORE_0_1_2, taskset -c 4-7 (secondary, no reading)")):
        e = out.get(key)
        if not e:
            continue
        L += [f"## {title}", "", "| round | A median (ms) | B median (ms) | B / A |", "|---|---|---|---|"]
        for r, x, y, q in zip(e["rounds"], e["A_ms"], e["B_ms"], e["ratio"]):
            L.append(f"| {r} | {fmt(x, 3)} | {fmt(y, 3)} | {fmt(q, 3)} |")
        L += ["", f"Median A {fmt(e['median_A'], 3)} ms, median B {fmt(e['median_B'], 3)} ms, median ratio {fmt(e['median_ratio'], 3)} (range {fmt(e['ratio_min'], 3)} to {fmt(e['ratio_max'], 3)}), noise floor {fmt(100 * e['noise_floor'], 2)} percent. Reading: **{e['reading']}**.", ""]
    for key, title in (("detail_core0", "Per-layer detail, CORE_0"), ("detail_core012", "Per-layer detail, CORE_0_1_2")):
        e = out.get(key)
        if not e:
            continue
        L += [f"## {title}", "", "| arm | layer-time sum (us) | op | rows | time (us) | share |", "|---|---|---|---|---|---|"]
        for arm in "AB":
            if arm not in e:
                continue
            for op in sorted(e[arm]["ops"], key=lambda o: -e[arm]["ops"][o]["us"]):
                d = e[arm]["ops"][op]
                L.append(f"| {arm} | {fmt(e[arm]['total_us'], 0)} | {op} | {d['count']} | {fmt(d['us'], 0)} | {fmt(100 * d['share'], 2)} percent |")
        L.append("")
        for arm in "AB":
            if arm in e:
                wl = collections.Counter()
                for pf in e[arm]["files"]:
                    for k, n in pf["workload"].items():
                        wl[k] += n
                L.append(f"WorkLoad of Split, Concat, Add rows, arm {arm} (rows summed over all runs of all files): " + "; ".join(f"{k} x{n}" for k, n in sorted(wl.items())))
        L.append("")
    (OUT / "c3_vs_c2f.md").write_text("\n".join(L) + "\n")
    (OUT / "c3_vs_c2f.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-dir", default=str(EXP / "board"))
    args = ap.parse_args()
    board = Path(args.board_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(accuracy=accuracy(), complexity=complexity())
    for key, prefix in (("latency_core0", "c0"), ("latency_core012", "c7")):
        out[key] = latency(board, prefix)
    for key, prefix in (("detail_core0", "c0"), ("detail_core012", "c7")):
        out[key] = detail(board, prefix)
    write(out)
    print(f"wrote {OUT / 'c3_vs_c2f.md'}")


if __name__ == "__main__":
    main()
