"""CLI: X1 analysis, split each C API run into on-CPU, wait, wake and preempted time from an ftrace of the NPU interrupt and the
measuring thread (docs 03-phuong-phap/02-tang-npu-nen-do/06-co-che-cac-muc-do-tre). Definitions and statement rules: bench/onboard_x1/rule.txt.

    .venv/bin/python -m ai.automation.analyze_trace_x1 <run_id>
Writes results/onboard_x1_<run_id>_int8_CORE_0.{json,md}.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = "rknn_perf_bench"
IDLE_EXIT = 4294967295
HALF = ["A", "a1i0", "a0i0", "a4i0", "a1i4", "a4i4"]
ORDER = HALF + HALF[::-1]
PAIRS = [("a1i0", "a4i4"), ("a0i0", "a4i4"), ("a4i0", "a4i4"), ("a1i4", "a4i4"), ("a1i0", "a1i4")]
LINE = re.compile(r"^\s*(.+?)-(\d+)\s+\[(\d+)\]\s+\S+\s+([\d.]+):\s+(\w+):\s+(.*)$")


def parse_trace(path):
    ev = collections.defaultdict(list)
    for ln in gzip.open(path, "rt", errors="replace"):
        if ln.startswith("#"):
            continue
        m = LINE.match(ln)
        if not m:
            continue
        _, _, cpu, ts, name, rest = m.groups()
        t, cpu = float(ts), int(cpu)
        f = dict(kv.split("=", 1) for kv in re.findall(r"(\w+=\S+)", rest))
        if name == "irq_handler_entry" and f.get("irq") == "36":
            ev["irq_in"].append((t, cpu))
        elif name == "irq_handler_exit" and f.get("irq") == "36":
            ev["irq_out"].append((t, cpu))
        elif name == "sched_switch":
            ev["switch"].append((t, cpu, f.get("prev_comm"), f.get("prev_state", ""), f.get("next_comm")))
        elif name == "sched_wakeup" and f.get("comm") == APP:
            ev["wakeup"].append(t)
        elif name == "cpu_idle":
            ev["idle", int(f["cpu_id"])].append((t, int(f["state"])))
    return ev


def deepest_idle(idle, cpu, a, b):
    """Deepest idle state entered by `cpu` in (a, b), -1 when it never entered one."""
    lst = idle.get(("idle", cpu), [])
    i = bisect.bisect_right(lst, (a, IDLE_EXIT + 1))
    best = -1
    while i < len(lst) and lst[i][0] < b:
        if lst[i][1] != IDLE_EXIT:
            best = max(best, lst[i][1])
        i += 1
    return best


def idle_before(idle, cpu, t):
    """State of the idle period of `cpu` that ends right before t (its last entry event before t), -1 if none."""
    lst = idle.get(("idle", cpu), [])
    i = bisect.bisect_left(lst, (t, -1)) - 1
    while i >= 0:
        if lst[i][1] != IDLE_EXIT:
            return lst[i][1]
        i -= 1
    return -1


def runs_of_block(path_csv, ev):
    sw = ev["switch"]
    sw_t = [x[0] for x in sw]
    irq_t = [x[0] for x in ev["irq_in"]]
    irq_cpu = {x[0]: x[1] for x in ev["irq_in"]}
    irq_exit = sorted(ev["irq_out"])
    wake = sorted(ev["wakeup"])
    out = []
    for r in csv.DictReader(open(path_csv)):
        t0 = int(r["t_us"]) / 1e6
        T = int(r["wall_us"]) / 1e6
        t1 = t0 + T
        i, j = bisect.bisect_left(sw_t, t0), bisect.bisect_right(sw_t, t1)
        evs = [x for x in sw[i:j] if APP in (x[2], x[4])]
        C = N = D = P = 0.0
        on, last = True, t0
        out_state, out_t, unpaired, d_list, app_idle, irq_idle, n_sleep = None, None, 0, [], [], [], 0
        handler, to_wake, n_irq = [], [], 0
        for t, cpu, prev, pstate, nxt in evs:
            if prev == APP and on:
                C += t - last
                on, out_t, out_state = False, t, pstate[:1]
            elif nxt == APP and not on:
                if out_state == "R":
                    P += t - out_t
                else:
                    n_sleep += 1
                    k = bisect.bisect_right(irq_t, t) - 1
                    if k >= 0 and irq_t[k] > out_t:
                        e = irq_t[k]
                        N += e - out_t
                        D += t - e
                        d_list.append(t - e)
                        irq_idle.append(idle_before(ev, irq_cpu[e], e))
                        x = bisect.bisect_left(irq_exit, (e, -1))
                        if x < len(irq_exit):
                            handler.append(irq_exit[x][0] - e)
                        w = bisect.bisect_left(wake, e)
                        if w < len(wake) and wake[w] <= t:
                            to_wake.append(wake[w] - e)
                    else:
                        N += t - out_t
                        unpaired += 1
                    app_idle.append(deepest_idle(ev, cpu, out_t, t))
                on, last = True, t
        if on:
            C += t1 - last
        else:  # the window ends while the thread is off CPU (should not happen: rknn_run returns on CPU), counted as a closure failure
            unpaired += 1
        n_irq = bisect.bisect_right(irq_t, t1) - bisect.bisect_left(irq_t, t0)
        out.append(dict(T=T * 1e3, C=C * 1e3, N=N * 1e3, D=D * 1e3, P=P * 1e3, closure=abs(T - (C + N + D + P)) / T, unpaired=unpaired, n_irq=n_irq,
                        n_sleep=n_sleep, app_idle1=(sum(1 for x in app_idle if x == 1) / len(app_idle)) if app_idle else None,
                        irq_idle1=(sum(1 for x in irq_idle if x == 1) / len(irq_idle)) if irq_idle else None,
                        d_us=[x * 1e6 for x in d_list], handler_us=[x * 1e6 for x in handler], to_wake_us=[x * 1e6 for x in to_wake],
                        cpu=int(r["cpu"]), app_idle_raw=app_idle, irq_idle_raw=irq_idle))
    return out


def irq_delta(pre, post):
    def tot(p):
        t = None
        for ln in Path(p).read_text().splitlines():
            v = [int(x) for x in ln.split()[1:9]]
            t = v if t is None else [a + b for a, b in zip(t, v)]
        return t
    return [b - a for a, b in zip(tot(pre), tot(post))]


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def block_summary(d, block, cond, overrun, half):
    ev = parse_trace(d / f"trace_{block}_{cond}.txt.gz")
    runs = runs_of_block(d / f"b{block}_{cond}.csv", ev)
    cpus = collections.Counter(r["cpu"] for r in runs)
    app_top, app_n = cpus.most_common(1)[0]
    irqd = irq_delta(d / f"irq_pre_{block}_{cond}.txt", d / f"irq_post_{block}_{cond}.txt")
    irq_top = max(range(8), key=lambda i: irqd[i])
    app_pin = None if cond == "A" else int(cond[1])
    irq_pin = 0 if cond == "A" else int(cond[3])
    closure_ok = sum(1 for r in runs if r["closure"] <= 0.02) / len(runs)
    realized = ((app_pin is None or (app_top == app_pin and app_n >= 0.99 * len(runs))) and irq_top == irq_pin and irqd[irq_top] >= 0.99 * sum(irqd)
                and closure_ok >= 0.95 and overrun.get(block, -1) == 0)
    all_d = [x for r in runs for x in r["d_us"]]
    all_h = [x for r in runs for x in r["handler_us"]]
    all_w = [x for r in runs for x in r["to_wake_us"]]
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else None  # noqa: E731
    summary = dict(cond=cond, block=block, half=half, n=len(runs), realized=realized, closure_ok_share=closure_ok,
                   overrun_delta=overrun.get(block), app_cpu=app_top, app_share=app_n / len(runs), irq_cpu=irq_top, irq_share=irqd[irq_top] / sum(irqd),
                   median={k: med([r[k] for r in runs]) for k in ("T", "C", "N", "D", "P")}, app_idle1_median=med([r["app_idle1"] for r in runs]),
                   irq_idle1_median=med([r["irq_idle1"] for r in runs]), n_irq_median=med([r["n_irq"] for r in runs]), unpaired_total=sum(r["unpaired"] for r in runs),
                   d_us={"median": med(all_d), "p95": q(all_d, 0.95)}, handler_us={"median": med(all_h), "p95": q(all_h, 0.95)},
                   irq_to_wakeup_us={"median": med(all_w), "p95": q(all_w, 0.95)},
                   idle_counters={"pre": (d / f"idle_pre_{block}_{cond}.txt").read_text(), "post": (d / f"idle_post_{block}_{cond}.txt").read_text()})
    return summary, runs


def main_x2(run_id):
    """X2: X1 setup with one intervention per cell. Statements S1 to S4 and the manipulation checks of bench/onboard_x2/rule.txt."""
    d = REPO / "ai" / "automation" / "runs" / run_id / "bench" / "onboard_x2"
    half = ["a4i4", "a4i0", "a4i0s", "a1i4", "a1i4s", "a4i4f", "a1i0", "a1i0s"]
    order = half + half[::-1]
    progress = (d / "progress.txt").read_text()
    overrun = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"block (\d+) post readback: .* overrun_delta: (-?\d+)", progress)}
    st = re.compile(r"idle_disable=(\d{8}) policy4=(\w+)/(\d+)")
    pre = {int(m.group(1)): st.search(m.group(0)).groups() for m in re.finditer(r"block (\d+) cond \S+ pin=.*", progress)}
    post = {int(m.group(1)): st.search(m.group(0)).groups() for m in re.finditer(r"block (\d+) post readback: .*", progress)}
    base_idle = st.search(next(ln for ln in progress.splitlines() if "restored (read before irqbalance restarts)" in ln)).group(1)
    targets = {"a4i0s": [0], "a1i4s": [1], "a1i0s": [0, 1]}
    blocks, manip = {}, {}
    for block, cond in enumerate(order, 1):
        summ, runs = block_summary(d, block, cond, overrun, 1 if block <= 8 else 2)
        want = list(base_idle)
        for c in targets.get(cond, []):
            want[c] = "1"
        want = "".join(want)
        ok_state = all(x is not None and x[0] == want and (x[1] == "userspace" and x[2] == "1800000" if cond == "a4i4f" else x[1] == "performance")
                       for x in (pre.get(block), post.get(block)))
        shares = {}
        if cond in ("a4i0s", "a1i0s"):
            v = [x for r in runs for x in r["irq_idle_raw"]]
            shares["irq_cpu_state1"] = sum(1 for x in v if x == 1) / len(v)
        if cond in ("a1i4s", "a1i0s"):
            v = [x for r in runs for x in r["app_idle_raw"]]
            shares["app_cpu_state1"] = sum(1 for x in v if x == 1) / len(v)
        ok_trace = all(x <= 0.01 for x in shares.values())
        manip[block] = dict(expected_idle_disable=want, pre=pre.get(block), post=post.get(block), state_ok=ok_state, trace_state1_shares=shares, trace_ok=ok_trace)
        summ["realized_x1_test"] = summ["realized"]
        summ["realized"] = summ["realized"] and ok_state and ok_trace
        blocks[block] = summ
    cell = lambda c, h: next(b for b in blocks.values() if b["cond"] == c and b["half"] == h)  # noqa: E731
    M = lambda c, h, k: cell(c, h)["median"][k]  # noqa: E731

    def removal(part, ctrl, interv):
        cells = (ctrl, interv, "a4i4")
        if not all(cell(c, h)["realized"] for c in cells for h in (1, 2)):
            return dict(verdict="chưa kết luận (khối không thực hiện đúng)")
        hs = [dict(g=M(ctrl, h, part) - M("a4i4", h, part), gs=M(interv, h, part) - M("a4i4", h, part)) for h in (1, 2)]
        if any(x["g"] < 1.25 for x in hs):
            return dict(halves=hs, verdict="không áp dụng (g < 1.25 ms)")
        if all(x["gs"] <= 0.2 * x["g"] for x in hs):
            v = "làm mất"
        elif all(x["gs"] >= 0.8 * x["g"] for x in hs):
            v = "không làm mất"
        else:
            v = "chưa kết luận"
        return dict(halves=hs, verdict=v)

    s1 = removal("N", "a4i0", "a4i0s")
    s2 = removal("D", "a1i4", "a1i4s")
    clk = 2.256 / 1.8
    if all(cell(c, h)["realized"] for c in ("a4i4f", "a4i4", "a1i4") for h in (1, 2)):
        hs = [dict(r=M("a4i4f", h, "C") / M("a4i4", h, "C"), c1=M("a1i4", h, "C") / M("a4i4", h, "C")) for h in (1, 2)]
        if all(abs(x["r"] - x["c1"]) <= 0.1 * x["c1"] for x in hs):
            v = "phần C chênh giữa CPU1 và CPU4 bằng với phần do hạ CPU4 về 1.8 GHz"
        elif all(abs(x["r"] - clk) <= 0.1 * clk and abs(x["r"] - x["c1"]) > 0.1 * x["c1"] for x in hs):
            v = "hạ xung nhịp chỉ giải thích một phần chênh lệch C giữa CPU1 và CPU4"
        else:
            v = "chưa kết luận"
        s3 = dict(halves=hs, clock_ratio=clk, verdict=v)
    else:
        s3 = dict(verdict="chưa kết luận (khối không thực hiện đúng)")
    s4 = {c: [M(c, h, "T") for h in (1, 2)] for c in ("a1i0s", "a1i0", "a4i4")}
    labels = dict(U1b="ĐO (phần tăng mất khi tắt cpu-sleep ở CPU tương ứng)" if s1["verdict"] == "làm mất" and s2["verdict"] == "làm mất" else "CHƯA BIẾT",
                  U3="ĐO (một chiều)" if s3["verdict"].startswith(("phần C", "hạ xung")) else "CHƯA BIẾT")
    out = dict(blocks=blocks, manipulation=manip, s1=s1, s2=s2, s3=s3, s4=s4, labels=labels, progress=progress, rule=(d / "rule.txt").read_text())
    name = f"onboard_x2_{run_id}_int8_CORE_0"
    (REPO / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = [f"# X2: interventions on cpu-sleep and the CPU4 clock, {run_id}", "", "Rule: bench/onboard_x2/rule.txt (written before the data).", "",
          "| block | cond | half | realized (X1 test, manipulation) | T ms | C ms | N ms | D ms | P ms | app idle1 | irq idle1 | D median us | trace state1 shares |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in blocks.values():
        m, mm = b["median"], manip[b["block"]]
        md.append(f"| {b['block']} | {b['cond']} | {b['half']} | {b['realized']} ({b['realized_x1_test']}, {mm['state_ok'] and mm['trace_ok']}) | {f(m['T'])} | {f(m['C'])} | {f(m['N'])} | "
                  f"{f(m['D'])} | {f(m['P'])} | {f(b['app_idle1_median'], 2)} | {f(b['irq_idle1_median'], 2)} | {f(b['d_us']['median'], 1)} | {mm['trace_state1_shares']} |")
    md += ["", f"S1 (N, a4i0 vs a4i0s against a4i4): {s1}", "", f"S2 (D, a1i4 vs a1i4s against a4i4): {s2}", "", f"S3 (C clock): {s3}", "", f"S4 (T ms, half 1 and 2): {s4}", "",
           f"Labels decided by the rule: {labels}"]
    (REPO / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--exp", default="x1", choices=["x1", "x2"])
    args = ap.parse_args()
    if args.exp == "x2":
        return main_x2(args.run_id)
    d = REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "onboard_x1"
    progress = (d / "progress.txt").read_text()
    overrun = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"block (\d+) post readback: .* overrun_delta: (-?\d+)", progress)}
    blocks = {}
    for block, cond in enumerate(ORDER, 1):
        blocks[block], _ = block_summary(d, block, cond, overrun, 1 if block <= 6 else 2)
    cell = lambda c, h: next(b for b in blocks.values() if b["cond"] == c and b["half"] == h)  # noqa: E731
    r1 = {}
    for x, y in PAIRS:
        ok = all(cell(c, h)["realized"] for c in (x, y) for h in (1, 2))
        halves = []
        for h in (1, 2):
            dT = cell(x, h)["median"]["T"] - cell(y, h)["median"]["T"]
            halves.append(dict(dT=dT, ratios={k: (cell(x, h)["median"][k] - cell(y, h)["median"][k]) / dT if dT else None for k in ("C", "N", "D", "P")}))
        if not ok:
            verdict = "chưa kết luận (khối không thực hiện đúng)"
        elif any(abs(hh["dT"]) < 1.25 for hh in halves):
            verdict = "không áp dụng (|dT| < 1.25 ms)"
        else:
            parts = [k for k in ("C", "N", "D", "P") if all(0.8 <= hh["ratios"][k] <= 1.2 for hh in halves)]
            verdict = f"chênh lệch thời gian chạy của cặp này nằm chủ yếu ở phần {parts[0]}" if len(parts) == 1 else "chưa kết luận"
        r1[f"{x} vs {y}"] = dict(halves=halves, verdict=verdict)
    a1 = [cell("a1i0", h)["app_idle1_median"] for h in (1, 2)]
    a4 = [cell("a4i4", h)["app_idle1_median"] for h in (1, 2)]
    r2 = ("ở ô chậm, CPU của tiến trình đo hầu như luôn vào cpu-sleep trong lúc chờ ngắt, ở ô nhanh hầu như không"
          if all(v is not None and v >= 0.9 for v in a1) and all(v is not None and v <= 0.1 for v in a4) else "chỉ báo số")
    onb = json.loads((REPO / "results" / f"onboard_{args.run_id}_int8_CORE_0.json").read_text())
    e2d = {}
    for b in onb["e2d"]["blocks"].values():
        e2d.setdefault(b["cond"], []).append(b["median"])
    r3 = {}
    for c in HALF:
        x1 = statistics.mean(cell(c, h)["median"]["T"] for h in (1, 2))
        ref = statistics.mean(e2d[c]) if c in e2d else None
        r3[c] = dict(x1=x1, e2d=ref, gap=None if ref is None else x1 - ref)
    trace_caveat = any(v["gap"] is not None and abs(v["gap"]) > 1.25 for v in r3.values())
    out = dict(blocks=blocks, r1=r1, r2=dict(a1i0=a1, a4i4=a4, statement=r2), r3=dict(cells=r3, caveat=trace_caveat), progress=progress,
               rule=(d / "rule.txt").read_text())
    res = REPO / "results"
    name = f"onboard_x1_{args.run_id}_int8_CORE_0"
    (res / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    md = [f"# X1: interrupt to run-again decomposition, {args.run_id}", "", "Rule: bench/onboard_x1/rule.txt (written before the data).", "",
          "| block | cond | half | realized | T ms | C ms | N ms | D ms | P ms | app idle1 | irq idle1 | D median us | D p95 us | handler us | irq to wakeup us | closure ok | overrun |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    for b in blocks.values():
        m = b["median"]
        md.append(f"| {b['block']} | {b['cond']} | {b['half']} | {b['realized']} | {f(m['T'])} | {f(m['C'])} | {f(m['N'])} | {f(m['D'])} | {f(m['P'])} | {f(b['app_idle1_median'], 2)} | "
                  f"{f(b['irq_idle1_median'], 2)} | {f(b['d_us']['median'], 1)} | {f(b['d_us']['p95'], 1)} | {f(b['handler_us']['median'], 1)} | {f(b['irq_to_wakeup_us']['median'], 1)} | "
                  f"{b['closure_ok_share']:.3f} | {b['overrun_delta']} |")
    md += ["", "| pair | half 1: dT ms, ratios C N D P | half 2 | verdict |", "|---|---|---|---|"]
    for k, v in r1.items():
        hs = [f"{hh['dT']:+.3f}; " + ", ".join(f"{p} {f(hh['ratios'][p], 2)}" for p in ("C", "N", "D", "P")) for hh in v["halves"]]
        md.append(f"| {k} | {hs[0]} | {hs[1]} | {v['verdict']} |")
    md += ["", f"R2: a1i0 app idle1 {a1}, a4i4 {a4}: {r2}.", "", "R3 (median T, X1 vs E2d, ms): " + ", ".join(f"{c} {f(v['x1'])} vs {f(v['e2d'])}" for c, v in r3.items())
           + f". Trace caveat: {trace_caveat}."]
    (res / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
