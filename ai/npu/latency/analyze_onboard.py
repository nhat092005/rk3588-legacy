"""CLI: analysis of the on-board C API measurement (ai/board/bench/rknn_perf_bench.c) of a run, set against eval_perf over the PC link.

Reads experiments/npu/latency/onboard/ (copied from the board's ~/onboard/out by the operator, see docs 03-phuong-phap/02-tang-npu-nen-do/02-ket-qua-latency-tung-khoi):
hot_<n>.csv, gap2s.csv, gap10s.csv, detail_<n>_detail.txt, load_hot_1.csv, cpuidle.csv, progress.txt; and the eval_perf data already in
runs/<run_id>/bench (B2, B3 sessions, E1). Writes results/npu/latency/onboard_<run_id>_<precision>_<core_mask>.{md,json}.
"""
import argparse
import collections
import csv
import json
import re
import statistics
import sys

import numpy as np

from ai.npu.latency.aggregate_layers import OUT_DIR, block_order, linfit, med, spread
from ai.automation.bench.benchmark_npu import RUNS_DIR
from ai.paths import experiment_dir
from ai.core.rknn.benchmark import LAYERS_TIME_RE, parse_layers
from ai.core.rknn.blocks import assign


# Level thresholds, read from the valleys of the pooled histogram of the 3000 back-to-back runs (0.25 ms bins): counts 7 at 17.00 ms
# between the fast peak (16.5 ms) and the middle plateau, and 20 at 20.00 ms between the middle plateau (19.25 ms) and the slow peak (21.75 ms).
FAST_BELOW_MS, SLOW_FROM_MS = 17.0, 20.0
# In detail mode the total of the per-layer table has its valley at 19 to 20 ms (11 of 600 runs): two levels.
DETAIL_HIGH_FROM_MS = 20.0


def level(ms):
    return "fast" if ms < FAST_BELOW_MS else ("slow" if ms >= SLOW_FROM_MS else "mid")


def level_shares(values_ms):
    c = collections.Counter(level(v) for v in values_ms)
    return {k: c[k] / len(values_ms) for k in ("fast", "mid", "slow")}


def stats(values):
    v = sorted(values)
    n = len(v)
    q = lambda p: v[min(n - 1, int(round(p * (n - 1))))]  # noqa: E731
    slope = float(np.polyfit(range(n), values, 1)[0]) if n > 2 else 0.0
    return dict(n=n, median=statistics.median(v), mean=statistics.mean(v), min=v[0], max=v[-1], p95=q(0.95), p99=q(0.99),
                cv=statistics.stdev(v) / statistics.mean(v) if n > 1 else 0.0, drift_per_run=slope)


def read_csv(path):
    return list(csv.DictReader(open(path)))


def irq_delta(pre, post):
    """Per-CPU interrupt counts added between two /proc/interrupts snapshots, summed over the NPU rows."""
    def total(path):
        t = None
        for ln in path.read_text().splitlines():
            parts = ln.split()
            vals = [int(x) for x in parts[1:] if x.isdigit()]
            t = vals if t is None else [a + b for a, b in zip(t, vals)]
        return t
    a, b = total(pre), total(post)
    return [y - x for x, y in zip(a, b)]


def idle_deltas(rows, pre, post):
    """Sum over cpus of the cpuidle counters between two snapshots: {state: (entries, time_us)}"""
    def total(label):
        t = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r["label"] == label:
                t[r["state"]][0] += int(r["usage"])
                t[r["state"]][1] += int(r["time_us"])
        return t
    a, b = total(pre), total(post)
    return {s: (b[s][0] - a[s][0], b[s][1] - a[s][1]) for s in b}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--precision", default="int8")
    parser.add_argument("--core-mask", default="CORE_0")
    args = parser.parse_args()
    bench = RUNS_DIR / args.run_id / "bench"
    ob = experiment_dir(args.run_id, "onboard")
    if not (ob / "hot_1.csv").exists():
        sys.exit(f"no on-board data in {ob}")
    out = {}

    # 1. back-to-back runs, one process per batch
    hot = {n: [int(r["wall_us"]) for r in read_csv(ob / f"hot_{n}.csv")] for n in (1, 2, 3)}
    perf_run = {n: [int(r["run_duration_us"]) for r in read_csv(ob / f"hot_{n}.csv")] for n in (1, 2, 3)}
    out["hot"] = {n: stats(v) for n, v in hot.items()}
    out["hot_levels"] = {n: dict(level_shares([x / 1000 for x in v]), longest_run_same_level=max(len(list(g)) for _, g in __import__("itertools").groupby(level(x / 1000) for x in v)),
                                 lag1_autocorr=float(np.corrcoef(v[:-1], v[1:])[0, 1])) for n, v in hot.items()}
    out["hot_gap_between_batches"] = spread([statistics.median(v) for v in hot.values()])
    out["hot_wall_equals_perf_run"] = all(abs(a - b) <= 5 for n in hot for a, b in zip(hot[n], perf_run[n]))

    # 2. NPU activity during batch 1 (rknpu debugfs, 10 Hz)
    load = read_csv(ob / "load_hot_1.csv")
    power_on = sum(1 for r in load if r["power"].strip() == "on")
    loads = [int(r["core0_load"]) for r in load if r["core0_load"].strip().isdigit()]
    out["activity_hot_1"] = dict(samples=len(load), power_on=power_on, power_on_share=power_on / len(load),
                                 core0_load_median=statistics.median(loads), core0_load_max=max(loads), core0_load_min=min(loads))

    # 3. cold start: gaps between runs inside one process
    gap = {}
    for name in ("gap2s", "gap10s"):
        gap[name] = stats([int(r["wall_us"]) for r in read_csv(ob / f"{name}.csv")])
        gap[name]["values"] = [int(r["wall_us"]) for r in read_csv(ob / f"{name}.csv")]
    out["gap"] = gap

    # 4. cpuidle counters around each batch
    idle_rows = read_csv(ob / "cpuidle.csv")
    out["cpuidle"] = {}
    for label in ("hot_1", "hot_2", "hot_3", "gap2s", "gap10s", "detail_1"):
        pre, post = ("pre_" + label, "post_" + label)
        if any(r["label"] == pre for r in idle_rows) and any(r["label"] == post for r in idle_rows):
            out["cpuidle"][label] = {s: dict(entries=e, time_us=t) for s, (e, t) in idle_deltas(idle_rows, pre, post).items()}

    # 5. per-layer tables of the detail batches
    detail = {}
    for n in (1, 2, 3):
        text = (ob / f"detail_{n}_detail.txt").read_text()
        runs = [t for t in re.split(r"=== run \d+ ===\n", text) if t.strip()]
        tables = [parse_layers(t) for t in runs]
        totals = [float(LAYERS_TIME_RE.search(t).group(1)) for t in runs]
        detail[n] = dict(tables=tables, totals=totals)
    ref = detail[1]["tables"][0]
    for n in detail:
        for t in detail[n]["tables"]:
            if [(r["op"], r["name"]) for r in t] != [(r["op"], r["name"]) for r in ref]:
                sys.exit("layer table differs between runs")
    out["detail_totals"] = {n: stats(d["totals"]) for n, d in detail.items()}
    n_rows = len(ref)
    est = [r["total_cycles"] / 1000 for r in ref]
    conv = [i for i, r in enumerate(ref) if r["npu_cycles"] > 0]
    per_batch = {}
    for n, d in detail.items():
        med_rows = [statistics.median(t[i]["time_us"] for t in d["tables"]) for i in range(n_rows)]
        slope, intercept = linfit([est[i] for i in conv], [med_rows[i] for i in conv])
        blocks = collections.Counter()
        for i, r in enumerate(ref):
            blocks[assign(r)[0]] += med_rows[i]
        total = sum(med_rows)
        per_batch[n] = dict(rows=med_rows, total=total, conv_slope=slope, conv_intercept=intercept,
                            block_us=dict(blocks), block_share={b: v / total for b, v in blocks.items()})
    blocks = sorted(per_batch[1]["block_us"], key=block_order)
    out["detail_blocks"] = {b: dict(us=med([per_batch[n]["block_us"][b] for n in per_batch]), share=med([per_batch[n]["block_share"][b] for n in per_batch]),
                                    share_spread=spread([per_batch[n]["block_share"][b] for n in per_batch])) for b in blocks}
    # block shares by level of the run total (detail mode, 600 runs)
    by_level = {"low": [], "high": []}
    for n, d in detail.items():
        for t, tot in zip(d["tables"], d["totals"]):
            by_level["high" if tot / 1000 >= DETAIL_HIGH_FROM_MS else "low"].append(t)
    lvl = {}
    for name, tabs in by_level.items():
        if not tabs:
            continue
        blk_runs = []
        for t in tabs:
            c = collections.Counter()
            for i, r in enumerate(ref):
                c[assign(r)[0]] += t[i]["time_us"]
            blk_runs.append(c)
        lvl[name] = dict(n_runs=len(tabs), total_us_median=med([sum(c.values()) for c in blk_runs]),
                         block_us={b: med([c[b] for c in blk_runs]) for b in blocks},
                         block_share={b: med([c[b] / sum(c.values()) for c in blk_runs]) for b in blocks})
    out["detail_levels"] = lvl
    out["detail_conv_fit"] = {n: dict(slope=p["conv_slope"], intercept=p["conv_intercept"], total=p["total"]) for n, p in per_batch.items()}
    # share spread gate (same rule as A8: blocks above 2% must stay within 10% between batches)
    out["detail_unstable_blocks"] = {b: v["share_spread"] for b, v in out["detail_blocks"].items() if v["share"] >= 0.02 and v["share_spread"] > 0.10}
    # non-Conv rows and the DFL chain from the detail batches
    rows_med = [med([per_batch[n]["rows"][i] for n in per_batch]) for i in range(n_rows)]
    total_med = sum(rows_med)
    by_op = collections.Counter()
    for i, r in enumerate(ref):
        by_op[r["op"]] += rows_med[i]
    out["detail_ops"] = {k: dict(us=v, share=v / total_med) for k, v in by_op.most_common()}
    out["detail_dfl_us"] = sum(rows_med[i] for i, r in enumerate(ref) if assign(r)[0] == "L22" and assign(r)[1] == "dfl")

    # 6. eval_perf over the PC link, for comparison (B2, B3 sessions, E1)
    ev = {}
    b2 = json.loads((bench / f"tier1_{args.precision}_{args.core_mask}.json").read_text())
    ev["B2"] = dict(latency_ms=b2["npu_latency_ms"], debug_total_ms=b2["debug_total_us"] / 1000)
    for n in (1, 2, 3):
        r = json.loads((bench / f"tier1_{args.precision}_{args.core_mask}_sess{n}.json").read_text())
        ev[f"B3-{n}"] = dict(latency_ms=r["npu_latency_ms"], debug_total_ms=r["debug_total_us"] / 1000)
    e1 = json.loads((bench / "extra_e1_evalperf_calls.json").read_text())
    for i, s in enumerate(e1["sessions"], 1):
        vals = [c["us"] / 1000 for c in s["calls"]]
        ev[f"E1-{i}"] = dict(latency_ms=statistics.median(vals), latency_min_ms=min(vals), latency_max_ms=max(vals), n=len(vals))
    out["eval_perf"] = ev
    agg = json.loads((OUT_DIR / f"latency_per_block_{args.run_id}_{args.precision}_{args.core_mask}.json").read_text())
    out["share_compare"] = {b: dict(onboard_all=out["detail_blocks"][b]["share"], onboard_low=out["detail_levels"].get("low", {}).get("block_share", {}).get(b),
                                    onboard_high=out["detail_levels"].get("high", {}).get("block_share", {}).get(b),
                                    evalperf_b3=agg["blocks"][b]["share"] if b in agg["blocks"] else None, evalperf_b2=agg["blocks"][b].get("b2_share") if b in agg["blocks"] else None)
                           for b in blocks}
    # E1: 2 independent eval_perf sessions of 20 calls each (perf_debug=False), with per-call printed frequencies and a 1 Hz board sampler
    e1_stats = []
    for i, sess in enumerate(e1["sessions"], 1):
        vals = [c["us"] / 1000 for c in sess["calls"]]
        st = stats(vals)
        st.update(session=i, range=(max(vals) - min(vals)) / statistics.median(vals), first=vals[0], rest_median=statistics.median(vals[1:]),
                  values=vals, wall_s_median=statistics.median(c["wall_s"] for c in sess["calls"]), init_s=sess["t_init_done"] - sess["t_start"],
                  freq_reports=len({json.dumps(c["freq"], sort_keys=True) for c in sess["calls"]}))
        e1_stats.append(st)
    allv = [v for st in e1_stats for v in st["values"]]
    low = [v for v in allv if v < 19.0]
    sampler = read_csv(bench / "extra_e1_board_sampler.csv")
    out["e1"] = dict(sessions=e1_stats, session_median_gap=spread([st["median"] for st in e1_stats]), low_mode_share=len(low) / len(allv),
                     low_mode_median=statistics.median(low), high_mode_median=statistics.median([v for v in allv if v >= 19.0]),
                     sampler=dict(rows=len(sampler), seconds=float(sampler[-1]["ts"]) - float(sampler[0]["ts"]),
                                  distinct={k: sorted({r[k] for r in sampler}) for k in ("npu_hz", "npu_gov", "dmc_hz", "dmc_gov", "cpu0_khz", "cpu4_khz", "cpu6_khz")},
                                  soc_mC=[min(int(r["soc_mC"]) for r in sampler), max(int(r["soc_mC"]) for r in sampler)],
                                  npu_mC=[min(int(r["npu_mC"]) for r in sampler), max(int(r["npu_mC"]) for r in sampler)]))
    cmpb = {b: [x for x in (v["onboard_all"], v["onboard_low"], v["onboard_high"], v["evalperf_b3"], v["evalperf_b2"]) if x is not None] for b, v in out["share_compare"].items()}
    out["share_agreement_pp"] = {b: (max(v) - min(v)) * 100 for b, v in cmpb.items()}
    out["share_agreement_max_pp_blocks_ge_2pct"] = max(d for b, d in out["share_agreement_pp"].items() if out["detail_blocks"][b]["share"] >= 0.02)
    # E2: controlled test of CPU placement, NPU interrupt placement and cpuidle (ABCD then DCBA)
    e2dir = experiment_dir(args.run_id, "onboard_e2")
    if (e2dir / "b1_A.csv").exists():
        conds = {"A": "baseline", "B": "app and NPU IRQs on CPU4 (big)", "C": "app and NPU IRQs on CPU0 (little)", "D": "cpuidle state1 off"}
        e2 = {}
        for block, cond in enumerate("ABCDDCBA", 1):
            rows = read_csv(e2dir / f"b{block}_{cond}.csv")
            ms = [int(r["wall_us"]) / 1000 for r in rows]
            cpus = collections.Counter(int(r["cpu"]) for r in rows)
            irqd = irq_delta(e2dir / f"irq_pre_{block}_{cond}.txt", e2dir / f"irq_post_{block}_{cond}.txt")
            app_cpu = f"CPU{max(cpus, key=cpus.get)}" if max(cpus.values()) >= 0.99 * len(rows) else "migrating"
            irq_top = max(range(len(irqd)), key=lambda i: irqd[i])
            e2[f"b{block}_{cond}"] = dict(cond=cond, block=block, order="ABCD" if block <= 4 else "DCBA", **stats(ms), **level_shares(ms),
                                          cpu_share={str(c): n / len(rows) for c, n in sorted(cpus.items())}, irq_delta=irqd,
                                          app_cpu=app_cpu, irq_cpu=f"CPU{irq_top}", irq_top_share=irqd[irq_top] / sum(irqd), state1_off=cond == "D",
                                          windows_100=[statistics.median(ms[i:i + 100]) for i in range(0, len(ms), 100)],
                                          fast_runs_before_100=sum(1 for x in ms[:100] if x < FAST_BELOW_MS), fast_runs_total=sum(1 for x in ms if x < FAST_BELOW_MS))
        base = {b["order"]: b for b in e2.values() if b["cond"] == "A"}

        def realized(b):
            """The intended condition was really applied in this block (checked on the per-run CPU and the NPU interrupt counts)."""
            if b["cond"] == "A":
                return True
            if b["cond"] == "D":
                return b["state1_off"]
            pin = "CPU4" if b["cond"] == "B" else "CPU0"
            return b["app_cpu"] == pin and b["irq_cpu"] == pin and b["irq_top_share"] >= 0.99

        verdict = {}
        for c in "BCD":
            shifts, excluded = {}, []
            for b in e2.values():
                if b["cond"] != c:
                    continue
                if realized(b):
                    shifts[b["order"]] = dict(slow=b["slow"] - base[b["order"]]["slow"], fast=b["fast"] - base[b["order"]]["fast"], cv=b["cv"] - base[b["order"]]["cv"])
                else:
                    excluded.append(b["block"])
            sl = [v["slow"] for v in shifts.values()]
            # explicit rule of this analysis (not from a source): needs both orders; supported when the slow-level share moves by more than 0.20 in the same
            # direction in both, not supported when it moves by less than 0.10 in both, otherwise no conclusion. Blocks that did not realize the intended
            # condition are excluded, so a condition with a single realized block cannot be labelled.
            if len(sl) < 2:
                label = "chưa kết luận (chỉ một khối thực hiện đúng điều kiện)"
            else:
                label = "ủng hộ" if all(x < -0.20 for x in sl) or all(x > 0.20 for x in sl) else ("không ủng hộ" if all(abs(x) < 0.10 for x in sl) else "chưa kết luận")
            verdict[c] = dict(shifts=shifts, excluded_blocks=excluded, label=label)
        classes = collections.defaultdict(list)
        for b in e2.values():
            classes[(b["app_cpu"], b["irq_cpu"] + f" ({b['irq_top_share']:.0%})" if b["irq_top_share"] < 0.99 else b["irq_cpu"], "off" if b["state1_off"] else "on")].append(b)
        class_rows = [dict(app=k[0], irq=k[1], state1=k[2], blocks=[x["block"] for x in v], medians=[x["median"] for x in v], slow=[x["slow"] for x in v]) for k, v in classes.items()]
        pooled = {lv: [x / 1000 for n in hot for x in hot[n] if level(x / 1000) == lv] for lv in ("fast", "mid", "slow")}
        irq_per_run = sum(e2["b7_B"]["irq_delta"]) / 1050
        gaps = {"mid_minus_fast": (statistics.median(pooled["mid"]) - statistics.median(pooled["fast"])) * 1000,
                "slow_minus_mid": (statistics.median(pooled["slow"]) - statistics.median(pooled["mid"])) * 1000}
        out["e2"] = dict(blocks=e2, verdict=verdict, progress=(e2dir / "progress.txt").read_text(), classes=class_rows, irq_per_run=irq_per_run,
                         level_medians_ms={k: statistics.median(v) for k, v in pooled.items()}, gap_us=gaps, gap_us_per_interrupt={k: v / irq_per_run for k, v in gaps.items()},
                         board_facts=(e2dir / "board_facts.txt").read_text())
    # E2b: condition B again with irqbalance stopped (order ABBA ABBA, other boot than E2). B is compared only with the A blocks of E2b.
    e2bdir = experiment_dir(args.run_id, "onboard_e2b")
    if (e2bdir / "b1_A.csv").exists():
        e2b = {}
        for block, cond in enumerate("ABBAABBA", 1):
            rows = read_csv(e2bdir / f"b{block}_{cond}.csv")
            ms = [int(r["wall_us"]) / 1000 for r in rows]
            cpus = collections.Counter(int(r["cpu"]) for r in rows)
            irqd = irq_delta(e2bdir / f"irq_pre_{block}_{cond}.txt", e2bdir / f"irq_post_{block}_{cond}.txt")
            irq_top = max(range(len(irqd)), key=lambda i: irqd[i])
            app_top = max(cpus, key=cpus.get)
            e2b[block] = dict(cond=cond, block=block, half=1 if block <= 4 else 2, **stats(ms), **level_shares(ms), std_ms=statistics.pstdev(ms),
                              app_cpu=f"CPU{app_top}" if cpus[app_top] >= 0.99 * len(rows) else "migrating", app_top_share=cpus[app_top] / len(rows),
                              irq_cpu=f"CPU{irq_top}", irq_top_share=irqd[irq_top] / sum(irqd), irq_total=sum(irqd), n_runs_incl_warmup=1050,
                              slow_by_app_cpu={str(c): (sum(1 for r in rows if int(r["cpu"]) == c and level(int(r["wall_us"]) / 1000) == "slow"), n) for c, n in sorted(cpus.items())})
        b_real = {k: v["app_cpu"] == "CPU4" and v["irq_cpu"] == "CPU4" and v["irq_top_share"] >= 0.99 for k, v in e2b.items() if v["cond"] == "B"}
        # pairing was not spelled out in rule.txt (written before the data); every pairing is reported and the half mean is the one used for the label
        slow = lambda ks: statistics.mean(e2b[k]["slow"] for k in ks)  # noqa: E731
        pairings = {
            "half mean of the two A blocks": [slow(bs) - slow(as_) for bs, as_ in (((2, 3), (1, 4)), ((6, 7), (5, 8)))],
            "last A block of each half": [slow((2, 3)) - slow((4,)), slow((6, 7)) - slow((8,))],
            "adjacent A block (b2 and b3 vs b1, b6 and b7 vs b5)": [slow((2, 3)) - slow((1,)), slow((6, 7)) - slow((5,))]}
        sl = pairings["half mean of the two A blocks"]
        label = "chưa kết luận (khối không thực hiện đúng điều kiện)" if not all(b_real.values()) else (
            "ủng hộ" if all(x < -0.20 for x in sl) or all(x > 0.20 for x in sl) else ("không ủng hộ" if all(abs(x) < 0.10 for x in sl) else "chưa kết luận"))
        out["e2b"] = dict(blocks=e2b, realized=b_real, pairings=pairings, label=label, progress=(e2bdir / "progress.txt").read_text(),
                          rule=(e2bdir / "rule.txt").read_text(), rule_note=(e2bdir / "rule_note.txt").read_text() if (e2bdir / "rule_note.txt").exists() else "")
    # E2c: split test of app placement and NPU interrupt placement (irqbalance stopped). Written from experiments/npu/latency/onboard_e2c/rule.txt before the data was opened.
    e2cdir = experiment_dir(args.run_id, "onboard_e2c")
    if (e2cdir / "b1_A.csv").exists():
        pins = {"A": (None, "CPU0"), "B": ("CPU4", "CPU4"), "C": ("CPU0", "CPU0"), "E": ("CPU4", "CPU0"), "F": ("CPU0", "CPU4"), "G": ("CPU1", "CPU0")}
        e2c = {}
        for block, cond in enumerate("ABCEFGGFECBA", 1):
            rows = read_csv(e2cdir / f"b{block}_{cond}.csv")
            ms = [int(r["wall_us"]) / 1000 for r in rows]
            cpus = collections.Counter(int(r["cpu"]) for r in rows)
            irqd = irq_delta(e2cdir / f"irq_pre_{block}_{cond}.txt", e2cdir / f"irq_post_{block}_{cond}.txt")
            irq_top = max(range(len(irqd)), key=lambda i: irqd[i])
            app_top = max(cpus, key=cpus.get)
            app_pin, irq_pin = pins[cond]
            app_ok = app_pin is None or (f"CPU{app_top}" == app_pin and cpus[app_top] >= 0.99 * len(rows))
            irq_ok = f"CPU{irq_top}" == irq_pin and irqd[irq_top] >= 0.99 * sum(irqd)
            e2c[block] = dict(cond=cond, block=block, half=1 if block <= 6 else 2, **stats(ms), **level_shares(ms), std_ms=statistics.pstdev(ms),
                              app_cpu=f"CPU{app_top}", app_top_share=cpus[app_top] / len(rows), irq_cpu=f"CPU{irq_top}", irq_top_share=irqd[irq_top] / sum(irqd),
                              irq_total=sum(irqd), realized=app_ok and irq_ok)
        primary, secondary = {}, {}
        for c in pins:
            bl = [b for b in e2c.values() if b["cond"] == c]
            if not all(b["realized"] for b in bl):
                primary[c] = "chưa kết luận (khối không thực hiện đúng điều kiện)"
            elif all(b["slow"] <= 0.01 for b in bl):
                primary[c] = "mất mức chậm"
            elif all(b["slow"] >= 0.20 for b in bl):
                primary[c] = "còn mức chậm"
            else:
                primary[c] = "lẫn lộn"
            if c != "A" and all(b["realized"] for b in bl):
                sh = [b["slow"] - next(a["slow"] for a in e2c.values() if a["cond"] == "A" and a["half"] == b["half"]) for b in sorted(bl, key=lambda x: x["half"])]
                secondary[c] = dict(shifts=sh, label="ủng hộ" if all(x < -0.20 for x in sh) else ("không ủng hộ" if all(abs(x) < 0.10 for x in sh) else "chưa kết luận"))
        checks = dict(a_has_slow=max(b["slow"] for b in e2c.values() if b["cond"] == "A") >= 0.20, b_lost=primary["B"] == "mất mức chậm", c_lost=primary["C"] == "mất mức chậm")
        valid = all(checks.values())
        gone, still = "mất mức chậm", "còn mức chậm"
        if not valid:
            read_ef = read_eg = "không đọc được yếu tố"
        else:
            read_ef = {(gone, still): "ứng dụng ở CPU4 làm mất mức chậm dù ngắt ở CPU nào", (still, gone): "ngắt ở CPU4 làm mất mức chậm dù ứng dụng ở CPU nào",
                       (gone, gone): "mỗi vị trí đứng riêng đã đủ", (still, still): "cần ứng dụng và ngắt cùng một CPU"}.get((primary["E"], primary["F"]), "không có phát biểu về yếu tố")
            read_eg = {(gone, still): "khi ngắt ở CPU0, ứng dụng ở CPU4 làm mất mức chậm còn ứng dụng ở CPU1 thì không (nói về vị trí ứng dụng, không nói về cùng hay khác CPU)",
                       (still, still): "cần đặt cùng một CPU"}.get((primary["E"], primary["G"]), "không có phát biểu")
        hint = {still: "ủng hộ gợi ý của E2b", gone: "không ủng hộ gợi ý của E2b"}.get(primary["G"], "không có phát biểu") if valid else "không đọc được yếu tố"
        out["e2c"] = dict(blocks=e2c, primary=primary, secondary=secondary, checks=checks, valid=valid, reading_e_f=read_ef, reading_e_g=read_eg, hint_g=hint,
                          progress=(e2cdir / "progress.txt").read_text(), rule=(e2cdir / "rule.txt").read_text())
    # E2d: per-CPU map of app placement x NPU interrupt placement (irqbalance stopped). Written from experiments/npu/latency/onboard_e2d/rule.txt before the data was opened.
    e2ddir = experiment_dir(args.run_id, "onboard_e2d")
    if (e2ddir / "b1_A.csv").exists():
        order = ["A"] + [f"a{a}i0" for a in range(8)] + [f"a{a}i4" for a in range(8)] + ["a5i5", "a6i6", "a7i7"]
        order = order + order[::-1]
        e2d = {}
        for block, cond in enumerate(order, 1):
            rows = read_csv(e2ddir / f"b{block}_{cond}.csv")
            ms = [int(r["wall_us"]) / 1000 for r in rows]
            cpus = collections.Counter(int(r["cpu"]) for r in rows)
            irqd = irq_delta(e2ddir / f"irq_pre_{block}_{cond}.txt", e2ddir / f"irq_post_{block}_{cond}.txt")
            irq_top = max(range(len(irqd)), key=lambda i: irqd[i])
            app_top = max(cpus, key=cpus.get)
            app_pin = None if cond == "A" else int(cond[1])
            irq_pin = 0 if cond == "A" else int(cond[3])
            app_ok = app_pin is None or (app_top == app_pin and cpus[app_top] >= 0.99 * len(rows))
            irq_ok = irq_top == irq_pin and irqd[irq_top] >= 0.99 * sum(irqd)
            e2d[block] = dict(cond=cond, block=block, half=1 if block <= 20 else 2, **stats(ms), **level_shares(ms), std_ms=statistics.pstdev(ms),
                              app_cpu=f"CPU{app_top}", app_top_share=cpus[app_top] / len(rows), irq_cpu=f"CPU{irq_top}", irq_top_share=irqd[irq_top] / sum(irqd),
                              irq_total=sum(irqd), realized=app_ok and irq_ok)
        unreal = "chưa kết luận (khối không thực hiện đúng)"
        primary, fastres = {}, {}
        for c in dict.fromkeys(order):
            bl = [b for b in e2d.values() if b["cond"] == c]
            if not all(b["realized"] for b in bl):
                primary[c] = fastres[c] = unreal
                continue
            primary[c] = "mất mức chậm" if all(b["slow"] <= 0.05 for b in bl) else ("còn mức chậm" if all(b["slow"] >= 0.20 for b in bl) else "lẫn lộn")
            fastres[c] = "đạt mức nhanh" if all(b["fast"] >= 0.95 for b in bl) else "không đạt mức nhanh"
        checks = dict(a_has_slow=max(b["slow"] for b in e2d.values() if b["cond"] == "A") >= 0.20, cell_44=primary["a4i4"] == "mất mức chậm" and fastres["a4i4"] == "đạt mức nhanh")
        valid = all(checks.values())
        rowdef = {r: [a for a in range(8) if primary[f"a{a}i{r}"] in ("mất mức chậm", "còn mức chậm")] for r in (0, 4)}
        S = {r: [a for a in range(8) if primary[f"a{a}i{r}"] == "còn mức chậm"] for r in (0, 4)}
        other = {r: [a for a in range(8) if primary[f"a{a}i{r}"] not in ("mất mức chậm", "còn mức chậm")] for r in (0, 4)}
        common = [a for a in range(8) if a in rowdef[0] and a in rowdef[4]]
        s0c, s4c = [a for a in S[0] if a in common], [a for a in S[4] if a in common]
        if not valid:
            compare = "không đọc được"
        elif len(common) == 8 and s0c == s4c:
            compare = "mức chậm không đổi giữa hai CPU ngắt đã thử"
        elif s0c == s4c:
            compare = f"không đổi trên các CPU {common}, các CPU còn lại không so sánh được"
        else:
            diff = [a for a in common if (a in s0c) != (a in s4c)]
            compare = f"không đổi trên các CPU {[a for a in common if a not in diff]}, khác nhau trên các CPU {diff} (mức chậm ở một hàng và không có ở hàng kia). CPU ngắt 0 và 4 cũng khác cụm và khác xung nhịp, nên không đọc thành vai trò của ngắt"
        cells = [c for c in primary if c != "A"]
        F = [c for c in cells if fastres[c] == "đạt mức nhanh"]
        nonrep = [c for c, want in (("a0i0", "mất mức chậm"), ("a1i0", "còn mức chậm")) if primary[c] != want]
        out["e2d"] = dict(blocks=e2d, primary=primary, fast=fastres, checks=checks, valid=valid, S0=S[0], S4=S[4], other0=other[0], other4=other[4], common=common,
                          compare=compare, F=F if valid else "không đọc được", non_replication_e2c=nonrep, s0_is_123=valid and S[0] == [1, 2, 3],
                          progress=(e2ddir / "progress.txt").read_text(), rule=(e2ddir / "rule.txt").read_text())
    # E2e: the eval_perf / rknn_server path with rknn_server and the NPU interrupts pinned. Written from experiments/npu/latency/onboard_e2e/rule.txt before the data was opened.
    # X6: the same comparison in the big-core cells (experiments/npu/latency/onboard_x6/rule.txt), same code.
    for key, dname, half, cells in (
            ("e2e", "onboard_e2e", ["A", "s4i4", "s0i0", "s4i0", "s0i4", "s1i0", "s1i4"],
             {"s4i4": "a4i4", "s0i0": "a0i0", "s4i0": "a4i0", "s0i4": "a0i4", "s1i0": "a1i0", "s1i4": "a1i4"}),
            ("x6", "onboard_x6", ["s4i4", "s5i4", "s6i4", "s7i4", "s5i5", "s6i6", "s7i7"],
             {"s4i4": "a4i4", "s5i4": "a5i4", "s6i4": "a6i4", "s7i4": "a7i4", "s5i5": "a5i5", "s6i6": "a6i6", "s7i7": "a7i7"})):
        e2edir = experiment_dir(args.run_id, dname)
        if not ((e2edir / "driver_done").exists() and (e2edir / "driver_done").read_text().startswith("DONE")):
            continue
        order = half + half[::-1]
        e2e = {}
        for block, cond in enumerate(order, 1):
            rows = read_csv(e2edir / f"b{block}_{cond}.csv")
            ms = [float(r["total_time_us"]) / 1000 for r in rows]
            irqd = irq_delta(e2edir / f"irq_pre_{block}.txt", e2edir / f"irq_post_{block}.txt")
            irq_top = max(range(len(irqd)), key=lambda i: irqd[i])
            pin = None if cond == "A" else cond[1]
            irq_pin = 0 if cond == "A" else int(cond[3])

            def tasks(path):
                return {ln.split()[0]: ln.split()[1:] for ln in path.read_text().splitlines() if ln.strip()}
            cpu0, cpu1 = tasks(e2edir / f"cpu0_{block}.txt"), tasks(e2edir / f"cpu1_{block}.txt")
            used = {t: (int(cpu1[t][0]) + int(cpu1[t][1])) - (int(cpu0[t][0]) + int(cpu0[t][1])) for t in cpu1 if t in cpu0}
            busy = max(used, key=used.get)
            allowed = [v[0] for k in ("allowed1", "allowed2") for v in tasks(e2edir / f"{k}_{block}.txt").values()]
            allowed_ok = pin is None or all(a == pin for a in allowed)
            busy_cpu = cpu1[busy][2]
            app_ok = pin is None or (allowed_ok and busy_cpu == pin)
            irq_ok = irq_top == irq_pin and irqd[irq_top] >= 0.99 * sum(irqd)
            e2e[block] = dict(cond=cond, block=block, half=1 if block <= len(half) else 2, **stats(ms), **level_shares(ms), std_ms=statistics.pstdev(ms),
                              busy_task=busy, busy_name=cpu1[busy][3], busy_cpu=busy_cpu, tasks=len(cpu1), allowed_ok=allowed_ok,
                              irq_cpu=f"CPU{irq_top}", irq_top_share=irqd[irq_top] / sum(irqd), irq_total=sum(irqd), realized=app_ok and irq_ok)
        unreal = "chưa kết luận (khối không thực hiện đúng)"
        pe, fe = {}, {}
        for c in dict.fromkeys(order):
            bl = [b for b in e2e.values() if b["cond"] == c]
            if not all(b["realized"] for b in bl):
                pe[c] = fe[c] = unreal
                continue
            pe[c] = "mất mức chậm" if all(b["slow"] <= 0.05 for b in bl) else ("còn mức chậm" if all(b["slow"] >= 0.20 for b in bl) else "lẫn lộn")
            fe[c] = "đạt mức nhanh" if all(b["fast"] >= 0.95 for b in bl) else "không đạt mức nhanh"
        d_blocks = out["e2d"]["blocks"] if "e2d" in out else {}
        diffs, cmp_rows = {}, {}
        for c, dc in cells.items():
            med_e = statistics.mean(b["median"] for b in e2e.values() if b["cond"] == c)
            med_d = statistics.mean(b["median"] for b in d_blocks.values() if b["cond"] == dc)
            diffs[c] = med_e - med_d
            cmp_rows[c] = dict(e2e_primary=pe[c], e2e_fast=fe[c], e2d_primary=out["e2d"]["primary"][dc], e2d_fast=out["e2d"]["fast"][dc], median_e2e=med_e, median_e2d=med_d, diff_ms=diffs[c])
        uniform = all(d > 1.25 for d in diffs.values()) or all(d < -1.25 for d in diffs.values())
        for c, r in cmp_rows.items():
            if unreal in (r["e2e_primary"], r["e2d_primary"]):
                r["verdict"] = "không so được (khối không thực hiện đúng)"
            elif uniform:
                r["verdict"] = "không so được (lệch mức đồng đều)"
            else:
                r["verdict"] = "cùng kết quả" if (r["e2e_primary"], r["e2e_fast"]) == (r["e2d_primary"], r["e2d_fast"]) else "khác"
        same_all = all(r["verdict"] == "cùng kết quả" for r in cmp_rows.values())
        out[key] = dict(blocks=e2e, primary=pe, fast=fe, comparison=cmp_rows, uniform_offset=uniform, same_all=same_all, different=[c for c, r in cmp_rows.items() if r["verdict"] == "khác"],
                          progress=(e2edir / "progress.txt").read_text(), rule=(e2edir / "rule.txt").read_text(), driver_log=(e2edir / "driver.log").read_text())
    OUT_DIR.mkdir(exist_ok=True)
    name = f"onboard_{args.run_id}_{args.precision}_{args.core_mask}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    f = lambda v, d=0: f"{v:.{d}f}"  # noqa: E731
    md = [f"# On-board C API measurement: {args.run_id} {args.precision} {args.core_mask}", "",
          "rknn_perf_bench.c on the board (RKNN API 2.3.2, driver 0.9.8, frequencies locked), no PC link. wall_us is the CPU clock around rknn_run, "
          "run_duration_us is RKNN_QUERY_PERF_RUN.", "",
          "## 1. Back-to-back runs (3 independent processes, 1000 runs after 50 warm-up runs)", "",
          "| batch | n | median us | mean us | min | max | p95 | p99 | CV | drift us per run |", "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {s['n']} | {f(s['median'])} | {f(s['mean'])} | {f(s['min'])} | {f(s['max'])} | {f(s['p95'])} | {f(s['p99'])} | {s['cv']:.2%} | {s['drift_per_run']:.4f} |" for n, s in out["hot"].items()]
    md += ["", f"Gap between batch medians: {out['hot_gap_between_batches']:.2%}. wall_us equals run_duration_us within 5 us on every run: {out['hot_wall_equals_perf_run']}.", "",
           "## 2. NPU activity during batch 1 (rknpu debugfs, 0.1 s sampling)", "",
           f"{out['activity_hot_1']['samples']} samples: power on in {out['activity_hot_1']['power_on']} ({out['activity_hot_1']['power_on_share']:.1%}), "
           f"Core0 load median {out['activity_hot_1']['core0_load_median']}%, min {out['activity_hot_1']['core0_load_min']}%, max {out['activity_hot_1']['core0_load_max']}%.", "",
           "## 3. Cold start test (gap between runs inside one process)", "", "| gap | n | median us | min | max | first run us | CV |", "|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {f(v['median'])} | {f(v['min'])} | {f(v['max'])} | {v['values'][0]} | {v['cv']:.2%} |" for k, v in gap.items()]
    md += ["", "## 4. cpuidle counters during each batch (sum over CPUs)", "", "| batch | state | entries | time us |", "|---|---|---|---|"]
    md += [f"| {b} | {s} | {v['entries']} | {v['time_us']} |" for b, st in out["cpuidle"].items() for s, v in st.items()]
    md += ["", "## 5. Per-layer batches (200 runs each after 20 warm-up runs, detail mode)", "",
           "| batch | total us median | CV | Conv fit slope | Conv fit intercept us |", "|---|---|---|---|---|"]
    md += [f"| {n} | {f(out['detail_totals'][n]['median'])} | {out['detail_totals'][n]['cv']:.2%} | {out['detail_conv_fit'][n]['slope']:.3f} | {out['detail_conv_fit'][n]['intercept']:.1f} |" for n in detail]
    md += ["", "| block | us (median over batches) | share % | share spread between batches |", "|---|---|---|---|"]
    md += [f"| {b} | {f(v['us'])} | {v['share'] * 100:.1f} | {v['share_spread']:.1%} |" for b, v in out["detail_blocks"].items()]
    md += ["", f"Blocks above 2% whose share spread between batches exceeds 10%: {out['detail_unstable_blocks'] or 'none'}. DFL (L22/dfl) {out['detail_dfl_us']:.0f} us.", "",
           "| OpType | us | share % |", "|---|---|---|"]
    md += [f"| {k} | {f(v['us'])} | {v['share'] * 100:.2f} |" for k, v in out["detail_ops"].items()]
    md += ["", "## 6. Against eval_perf over the PC link (same .rknn, same locked frequencies)", "", "| source | latency ms | note |", "|---|---|---|"]
    md += [f"| on-board hot batch {n} | {s['median'] / 1000:.3f} | median of {s['n']} runs |" for n, s in out["hot"].items()]
    md += [f"| eval_perf {k} | {v['latency_ms']:.3f} | " + (f"median of {v['n']} calls, min {v['latency_min_ms']:.3f}, max {v['latency_max_ms']:.3f}" if "n" in v else f"debug total {v['debug_total_ms']:.3f} ms") + " |" for k, v in ev.items()]
    e = out["e1"]
    md += ["", "## 7. E1: two eval_perf sessions of 20 calls (perf_debug=False)", "",
           "| session | n | median ms | min | max | range | CV | drift us per call | first call | median of the rest | init_runtime s | wall s per call (median) | distinct printed frequency reports |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {s['session']} | {s['n']} | {s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['range']:.1%} | {s['cv']:.1%} | {s['drift_per_run'] * 1000:.1f} | {s['first']:.3f} | {s['rest_median']:.3f} | {s['init_s']:.0f} | {s['wall_s_median']:.1f} | {s['freq_reports']} |" for s in e["sessions"]]
    sp = e["sampler"]
    md += ["", f"Gap between the two session medians: {e['session_median_gap']:.2%}. {e['low_mode_share']:.0%} of the 40 calls lie below 19.0 ms (median {e['low_mode_median']:.3f} ms), the others have median {e['high_mode_median']:.3f} ms.", "",
           f"Board sampler (1 Hz, {sp['rows']} rows over {sp['seconds']:.0f} s) read only these values, which stayed constant: {sp['distinct']}; SoC temperature {sp['soc_mC'][0] / 1000:.1f} to {sp['soc_mC'][1] / 1000:.1f} C, NPU temperature {sp['npu_mC'][0] / 1000:.1f} to {sp['npu_mC'][1] / 1000:.1f} C. "
           "A flat trace of these values does not prove that the real clocks were stable."]
    md += ["", "## 8. Latency levels of the back-to-back runs", "",
           f"Level thresholds from the valleys of the pooled histogram: fast below {FAST_BELOW_MS} ms, slow from {SLOW_FROM_MS} ms, mid in between.", "",
           "| batch | fast share | mid share | slow share | longest run in one level | lag-1 autocorrelation |", "|---|---|---|---|---|---|"]
    md += [f"| {n} | {v['fast']:.1%} | {v['mid']:.1%} | {v['slow']:.1%} | {v['longest_run_same_level']} | {v['lag1_autocorr']:.2f} |" for n, v in out["hot_levels"].items()]
    md += ["", f"## 9. Block shares of the {sum(v['n_runs'] for v in out['detail_levels'].values())} detail runs by level of the run total (low below {DETAIL_HIGH_FROM_MS} ms, high from it)", "",
           "| level | runs | total us median |", "|---|---|---|"]
    md += [f"| {k} | {v['n_runs']} | {v['total_us_median']:.0f} |" for k, v in out["detail_levels"].items()]
    md += ["", "| block | on-board all % | on-board low % | on-board high % | eval_perf B3 median % | eval_perf B2 % |", "|---|---|---|---|---|---|"]
    fp = lambda v: "-" if v is None else f"{v * 100:.1f}"  # noqa: E731
    md += [f"| {b} | {fp(v['onboard_all'])} | {fp(v['onboard_low'])} | {fp(v['onboard_high'])} | {fp(v['evalperf_b3'])} | {fp(v['evalperf_b2'])} |" for b, v in out["share_compare"].items()]
    md += ["", f"Largest spread (max minus min, in percentage points) of a block share across the five sources (on-board all, low, high, eval_perf B3 median, B2): "
           f"{out['share_agreement_max_pp_blocks_ge_2pct']:.1f} pp over the blocks with at least 2% of the time; per block: "
           + ", ".join(f"{b} {d:.1f}" for b, d in out["share_agreement_pp"].items())]
    if "e2" in out:
        e2o = out["e2"]
        md += ["", "## 10. E2: CPU placement, NPU interrupt placement, cpuidle (1000 runs per block, order ABCD then DCBA)", "",
               "Intended conditions: A baseline, B app and NPU IRQs on CPU4, C app and NPU IRQs on CPU0, D cpuidle state1 (cpu-sleep) off. "
               "The realized factors below come from the per-run CPU and the NPU interrupt counts, and differ from the intended ones in block 2.", "",
               "| block | intended | order | n | median ms | p95 ms | CV | fast | mid | slow | app CPU | NPU interrupts: CPU with most (share) | cpu-sleep off |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {b['block']} | {b['cond']} | {b['order']} | {b['n']} | {b['median']:.3f} | {b['p95']:.3f} | {b['cv']:.1%} | {b['fast']:.1%} | {b['mid']:.1%} | {b['slow']:.1%} | {b['app_cpu']} | {b['irq_cpu']} ({b['irq_top_share']:.0%}) | {b['state1_off']} |" for b in e2o["blocks"].values()]
        md += ["", "Median of each window of 100 runs (ms):", "", "| block | " + " | ".join(f"{i * 100}-{i * 100 + 99}" for i in range(10)) + " |", "|---|" + "---|" * 10]
        md += [f"| {b['block']} {b['cond']} | " + " | ".join(f"{w:.2f}" for w in b["windows_100"]) + " |" for b in e2o["blocks"].values()]
        md += ["", "Blocks grouped by realized factors (app placement, where most NPU interrupts landed, cpu-sleep):", "", "| app | NPU interrupts | cpu-sleep off | blocks | medians ms | slow share |", "|---|---|---|---|---|---|"]
        md += [f"| {c['app']} | {c['irq']} | {c['state1'] == 'off'} | {c['blocks']} | {', '.join(f'{m:.2f}' for m in c['medians'])} | {', '.join(f'{x:.1%}' for x in c['slow'])} |" for c in e2o["classes"]]
        md += ["", "Labels. The rule was fixed before the E2 data existed and is not from a source: a condition is labelled 'ủng hộ' when it moves the slow-level share by more than 0.20 "
               "in the same direction in both orders, 'không ủng hộ' under 0.10 in both, otherwise 'chưa kết luận'. Blocks that did not realize the intended condition are excluded, "
               "so condition B (block 2 not realized) has one valid block and cannot be labelled. 'ủng hộ' here only means that the intervention removes the slow level. It does not identify a mechanism.", "",
               "| intended condition | slow-share shift vs baseline (ABCD, DCBA) | label |", "|---|---|---|"]
        md += [f"| {c} {conds[c]} | " + ", ".join(f"{o}: {v['slow']:+.2f}" for o, v in x["shifts"].items()) + f" | {x['label']} |" for c, x in e2o["verdict"].items()]
        md += ["", "Consistency check only, not a mechanism:", "",
               f"- NPU interrupts per run: {e2o['irq_per_run']:.0f} (66150 interrupts in block 7 over 1050 runs).",
               f"- Medians of the pooled levels: fast {e2o['level_medians_ms']['fast']:.3f} ms, mid {e2o['level_medians_ms']['mid']:.3f} ms, slow {e2o['level_medians_ms']['slow']:.3f} ms. "
               f"Gaps {e2o['gap_us']['mid_minus_fast']:.0f} us and {e2o['gap_us']['slow_minus_mid']:.0f} us, which is {e2o['gap_us_per_interrupt']['mid_minus_fast']:.0f} us and {e2o['gap_us_per_interrupt']['slow_minus_mid']:.0f} us per interrupt.",
               "- Read on the board after E2 (experiments/npu/latency/onboard_e2/board_facts.txt): cpu-sleep exit latency 220 us and minimum residency 1000 us, WFI latency 1 us; irqbalance is active.", "",
               "Board state log of the run (IRQ affinity saved and restored):", "", "```", e2o["progress"].strip(), "```"]
    if "e2b" in out:
        e2bo = out["e2b"]
        md += ["", "## 11. E2b: condition B with irqbalance stopped (1000 runs per block, order ABBA ABBA)", "",
               "Another boot than E2, irqbalance stopped for the whole run, so B is compared only with the A blocks of this run. A is also measured with irqbalance stopped. "
               "B puts the app and all NPU interrupts on CPU4, which changes three things at once (app placement, interrupt placement, both on the same core).", "",
               "| block | cond | half | n | median ms | p95 ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU with most (share) | interrupts |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {b['block']} | {b['cond']} | {b['half']} | {b['n']} | {b['median']:.3f} | {b['p95']:.3f} | {b['std_ms']:.3f} | {b['fast']:.1%} | {b['mid']:.1%} | {b['slow']:.1%} | {b['app_cpu']} ({b['app_top_share']:.0%}) | {b['irq_cpu']} ({b['irq_top_share']:.0%}) | {b['irq_total']} |" for b in e2bo["blocks"].values()]
        md += ["", "Slow-share shift of B versus A under each pairing (halves are blocks 1 to 4 and 5 to 8). The pairing was not spelled out in rule.txt, which was written before the data. "
               "The half mean is used for the label and was chosen after the data existed.", "", "| pairing | half 1 | half 2 |", "|---|---|---|"]
        md += [f"| {k} | {v[0]:+.3f} | {v[1]:+.3f} |" for k, v in e2bo["pairings"].items()]
        md += ["", f"B realized in every B block (app on CPU4 in at least 99% of runs, at least 99% of interrupts on CPU4): {all(e2bo['realized'].values())}. Label for B: {e2bo['label']}. "
               "'ủng hộ' here only means that putting the app and the NPU interrupts on CPU4 removes the slow level. It does not say which of the three changes does it.", "",
               "Slow runs by the CPU the app ended on (post-hoc, exploratory, not used for the label), slow runs / runs on that CPU:", "", "| block | cond | " + " | ".join(f"CPU{c}" for c in range(8)) + " |", "|---|---|" + "---|" * 8]
        md += [f"| {b['block']} | {b['cond']} | " + " | ".join((f"{b['slow_by_app_cpu'][str(c)][0]}/{b['slow_by_app_cpu'][str(c)][1]}" if str(c) in b["slow_by_app_cpu"] else "-") for c in range(8)) + " |" for b in e2bo["blocks"].values()]
        md += ["", "Rule file written before the data (experiments/npu/latency/onboard_e2b/rule.txt):", "", "```", e2bo["rule"].strip(), "```"]
        if e2bo["rule_note"]:
            md += ["", "Note added after the data (experiments/npu/latency/onboard_e2b/rule_note.txt):", "", "```", e2bo["rule_note"].strip(), "```"]
        md += ["", "Board state log of the run (IRQ affinity and irqbalance saved and restored):", "", "```", e2bo["progress"].strip(), "```"]
    if "e2c" in out:
        e2co = out["e2c"]
        md += ["", "## 12. E2c: app placement versus NPU interrupt placement (irqbalance stopped, 1000 runs per block, order ABCEFG GFECBA)", "",
               "Conditions (app CPU / NPU interrupt CPU): A none / default, B 4 / 4, C 0 / 0, E 4 / 0, F 0 / 4, G 1 / 0. Outcomes follow experiments/npu/latency/onboard_e2c/rule.txt, written before the data.", "",
               "| block | cond | half | n | median ms | p95 ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU (share) | interrupts | realized |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {b['block']} | {b['cond']} | {b['half']} | {b['n']} | {b['median']:.3f} | {b['p95']:.3f} | {b['std_ms']:.3f} | {b['fast']:.1%} | {b['mid']:.1%} | {b['slow']:.1%} | {b['app_cpu']} ({b['app_top_share']:.0%}) | {b['irq_cpu']} ({b['irq_top_share']:.0%}) | {b['irq_total']} | {b['realized']} |" for b in e2co["blocks"].values()]
        md += ["", "| condition | primary outcome | secondary: shift vs A of the same half (half 1, half 2) and label |", "|---|---|---|"]
        md += [f"| {c} | {v} | " + (", ".join(f"{x:+.3f}" for x in e2co["secondary"][c]["shifts"]) + f" {e2co['secondary'][c]['label']}" if c in e2co["secondary"] else "-") + " |" for c, v in e2co["primary"].items()]
        md += ["", f"Validity: A has a slow share of at least 0.20 in one block: {e2co['checks']['a_has_slow']}. B lost the slow level: {e2co['checks']['b_lost']}. C lost it: {e2co['checks']['c_lost']}. Valid: {e2co['valid']}.", "",
               f"Reading E and F: {e2co['reading_e_f']}.", f"Reading E and G: {e2co['reading_e_g']}.", f"G against the E2b exploratory hint: {e2co['hint_g']}.", "",
               "Rule file written before the data (experiments/npu/latency/onboard_e2c/rule.txt):", "", "```", e2co["rule"].strip(), "```", "", "Board state log of the run:", "", "```", e2co["progress"].strip(), "```"]
    if "e2d" in out:
        e2do = out["e2d"]
        md += ["", "## 13. E2d: app CPU x NPU interrupt CPU map (irqbalance stopped, 1000 runs per block, 40 blocks)", "",
               "Cell name aXiY: app pinned to CPU X, NPU interrupts on CPU Y. Outcomes follow experiments/npu/latency/onboard_e2d/rule.txt, written before the data.", "",
               "| block | cond | half | median ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU (share) | interrupts | realized |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {b['block']} | {b['cond']} | {b['half']} | {b['median']:.3f} | {b['std_ms']:.3f} | {b['fast']:.1%} | {b['mid']:.1%} | {b['slow']:.1%} | {b['app_cpu']} ({b['app_top_share']:.0%}) | {b['irq_cpu']} ({b['irq_top_share']:.0%}) | {b['irq_total']} | {b['realized']} |" for b in e2do["blocks"].values()]
        md += ["", "| cell | primary outcome | fast outcome |", "|---|---|---|"] + [f"| {c} | {v} | {e2do['fast'][c]} |" for c, v in e2do["primary"].items()]
        md += ["", f"Validity: A has a slow share of at least 0.20 in one block: {e2do['checks']['a_has_slow']}. Cell a4i4 loses the slow level and reaches the fast level: {e2do['checks']['cell_44']}. Valid: {e2do['valid']}.", "",
               f"Not reproducing E2c (a0i0 should lose the slow level, a1i0 should keep it): {e2do['non_replication_e2c'] or 'none'}.", "",
               f"S0 (app CPUs with the slow level, interrupts on CPU0): {e2do['S0']}, ambiguous or unresolved: {e2do['other0']}.", f"S4 (interrupts on CPU4): {e2do['S4']}, ambiguous or unresolved: {e2do['other4']}.",
               f"Comparison on CPUs definite in both rows ({e2do['common']}): {e2do['compare']}.", f"F (cells that reach the fast level): {e2do['F']}.",
               "", "Rule file written before the data (experiments/npu/latency/onboard_e2d/rule.txt):", "", "```", e2do["rule"].strip(), "```", "", "Board state log of the run:", "", "```", e2do["progress"].strip(), "```"]
    if "e2e" in out:
        e2eo = out["e2e"]
        md += ["", "## 14. E2e: eval_perf path (PC Toolkit2 to rknn_server), rknn_server and NPU interrupts pinned (500 timed calls per block, 14 blocks)", "",
               "Cell name sXiY: rknn_server pinned to CPU X, NPU interrupts on CPU Y. Outcomes follow experiments/npu/latency/onboard_e2e/rule.txt, written before the data. Sample = Total Time(us) of one eval_perf call (one inference).", "",
               "| block | cond | half | n | median ms | std ms | fast | mid | slow | busy task (name) final CPU | all tasks pinned | NPU interrupts: CPU (share) | realized |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {b['block']} | {b['cond']} | {b['half']} | {b['n']} | {b['median']:.3f} | {b['std_ms']:.3f} | {b['fast']:.1%} | {b['mid']:.1%} | {b['slow']:.1%} | {b['busy_task']} ({b['busy_name']}) CPU{b['busy_cpu']} | {b['allowed_ok']} | {b['irq_cpu']} ({b['irq_top_share']:.0%}) | {b['realized']} |" for b in e2eo["blocks"].values()]
        md += ["", "| cell | E2e primary | E2e fast | E2d cell primary | E2d cell fast | median E2e ms | median E2d ms | difference ms | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {c} | {r['e2e_primary']} | {r['e2e_fast']} | {r['e2d_primary']} | {r['e2d_fast']} | {r['median_e2e']:.3f} | {r['median_e2d']:.3f} | {r['diff_ms']:+.3f} | {r['verdict']} |" for c, r in e2eo["comparison"].items()]
        md += ["", f"Uniform offset exception (same sign, each difference above 1.25 ms): {e2eo['uniform_offset']}. All six cells give the same result: {e2eo['same_all']}. Cells that differ: {e2eo['different'] or 'none'}.",
               f"A (descriptive): primary {e2eo['primary']['A']}, fast {e2eo['fast']['A']}.", "",
               "Rule file written before the data (experiments/npu/latency/onboard_e2e/rule.txt):", "", "```", e2eo["rule"].strip(), "```", "", "Board helper log:", "", "```", e2eo["progress"].strip(), "```"]
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print((OUT_DIR / f"{name}.md").read_text())


if __name__ == "__main__":
    main()
