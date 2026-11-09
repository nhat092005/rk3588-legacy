"""Which memory mapping of the live camera.py process grows. Runs camera.main() in this process and samples
/proc/self/smaps every SAMPLE s in a thread. At exit prints the mappings whose Rss grew most between first and last sample.

  cd ~/rk3588 && PYTHONPATH=. taskset -c 4-7 .venv-board/bin/python /tmp/live_smaps.py <camera.py args>
"""
import atexit
import collections
import gc
import os
import re
import sys
import threading
import time
import tracemalloc

SAMPLE = float(os.environ.get("SAMPLE", 60))
HDR = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \S+\s*(.*)$")
snaps = []  # (t, {key: rss_kib}, threads)
gcrows, gctypes = [], None  # GCPROBE=1: (rss before collect, unreachable found, rss after collect) per sample
traced, tm_snaps = [], []  # TM=1: tracemalloc current MiB per sample, snapshots of samples 2 and last


def read() -> dict:
    out, key = {}, None
    with open("/proc/self/smaps") as fh:
        for line in fh:
            m = HDR.match(line)
            if m:
                key = f"{m.group(1)}-{m.group(2)} {m.group(3) or '[anon]'}"
            elif line.startswith("Rss:") and key:
                out[key] = out.get(key, 0) + int(line.split()[1])
    return out


def rss_mib() -> float:
    with open("/proc/self/status") as fh:
        return next(int(x.split()[1]) for x in fh if x.startswith("VmRSS:")) / 1024


def gc_probe() -> None:
    global gctypes
    before = rss_mib()
    if len(snaps) == 4:  # once: what kinds of objects are the cyclic garbage
        gc.set_debug(gc.DEBUG_SAVEALL)
        found = gc.collect()
        gctypes = collections.Counter(type(o).__name__ for o in gc.garbage).most_common(8)
        gc.set_debug(0)
        gc.garbage.clear()
    else:
        found = gc.collect()
    gcrows.append((round(before, 1), found, round(rss_mib(), 1)))


def sampler() -> None:
    while True:
        if os.environ.get("GCPROBE"):
            gc_probe()
        snaps.append((time.time(), read(), threading.active_count()))
        if tracemalloc.is_tracing():
            traced.append(round(tracemalloc.get_traced_memory()[0] / 2**20, 2))
            if len(snaps) == 3:
                tm_snaps[:] = [tracemalloc.take_snapshot()]  # baseline, model load is over
            elif len(snaps) > 3:
                tm_snaps[1:] = [tracemalloc.take_snapshot()]  # latest
        time.sleep(SAMPLE)


def report() -> None:
    if len(snaps) < 2:
        return
    print('total Rss per sample (MiB):', [round(sum(x[1].values()) / 1024, 1) for x in snaps])
    (t0, a, th0), (t1, b, th1) = snaps[min(2, len(snaps) - 2)], snaps[-1]  # from the 3rd sample: model load is over
    print(f"smaps samples {len(snaps)} over {t1 - t0:.0f} s, threads {th0} -> {th1}, total Rss {sum(a.values()) / 1024:.1f} -> {sum(b.values()) / 1024:.1f} MiB")
    if gcrows:
        print('gc probe (rss before, unreachable found, rss after):', gcrows)
        print('gc garbage types at sample 5:', gctypes)
    if traced:
        print("tracemalloc traced MiB per sample:", traced)
    if len(tm_snaps) == 2:
        for st in tm_snaps[1].compare_to(tm_snaps[0], "lineno")[:8]:
            print("  tm", st)
    diff = sorted(((b.get(k, 0) - a.get(k, 0), k) for k in set(a) | set(b)), reverse=True)
    for d, k in diff[:8]:
        print(f"  {d / 1024:+8.2f} MiB  {k}  (now {b.get(k, 0) / 1024:.1f})")


if __name__ == "__main__":
    from ai.board import camera

    sys.argv = ["camera"] + sys.argv[1:]
    if os.environ.get('TM'):
        tracemalloc.start(5)
    threading.Thread(target=sampler, daemon=True).start()
    atexit.register(report)
    camera.main()
