"""RKNNLite functional probes (M4 shared context, M6 async_mode, M7 batch size, M16 context count), run ON THE BOARD.

Statements and reading rules: experiments/npu/operator_support/onboard_q2e/rule_probe.txt (written before any result).
Usage: .venv-board/bin/python -m ai.board.probe_runtime <m4|m6|m7|m16> --model <rknn_int8 dir> --out <json> [--b1 <rknn> --b4 <rknn>]
Inputs are 8 fixed random uint8 640x640x3 arrays (seed 0). Timings are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def meminfo() -> dict:
    d = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, v = ln.split(":")
        d[k] = int(v.split()[0]) // 1024
    return d


def rss_mib() -> float:
    for ln in Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmRSS"):
            return int(ln.split()[1]) / 1024
    return -1.0


def inputs(n: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(1, 640, 640, 3), dtype=np.uint8) for _ in range(n)]


def open_ctx(model: str, core=RKNNLite.NPU_CORE_0, **kw) -> RKNNLite:
    r = RKNNLite(verbose=False)
    if r.load_rknn(model) != 0:
        raise RuntimeError("load_rknn failed")
    rc = r.init_runtime(core_mask=core, **kw)
    if rc != 0:
        raise RuntimeError(f"init_runtime returned {rc}")
    return r


def ref_outputs(model: str, xs) -> list[np.ndarray]:
    r = open_ctx(model)
    out = [np.asarray(r.inference(inputs=[x])[0], dtype=np.float32) for x in xs]
    r.release()
    return out


def mad(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float32) - b).max())


def m4(model: str, nthreads: int = 3, calls: int = 100) -> dict:
    xs = inputs()
    ref = ref_outputs(model, xs)
    t0 = time.perf_counter()
    r = open_ctx(model)
    for i in range(nthreads * calls):
        r.inference(inputs=[xs[i % 8]])
    seq_s = time.perf_counter() - t0
    res = dict(exceptions=[], diffs=[], calls=0)
    lock = threading.Lock()

    def work(start: int):
        for i in range(calls):
            k = (start + i) % 8
            try:
                y = np.asarray(r.inference(inputs=[xs[k]])[0], dtype=np.float32)
                d = mad(y, ref[k])
                with lock:
                    res["calls"] += 1
                    res["diffs"].append(d)
            except Exception:
                with lock:
                    res["exceptions"].append(traceback.format_exc()[-400:])

    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, args=(s,)) for s in (0, 3, 6)[:nthreads]]
    [t.start() for t in ts]
    [t.join() for t in ts]
    thr_s = time.perf_counter() - t0
    r.release()
    return dict(threads=nthreads, calls_per_thread=calls, calls_done=res["calls"], exceptions=len(res["exceptions"]), exception_text=res["exceptions"][:2], outputs_with_diff_above_0=sum(d > 0 for d in res["diffs"]),
                max_diff=max(res["diffs"], default=None), wall_s_threads=thr_s, wall_s_sequential_incl_init=seq_s)


def m6(model: str) -> dict:
    xs = inputs()
    ref = ref_outputs(model, xs)
    out = dict(calls=[], init_error=None)
    try:
        r = open_ctx(model, async_mode=True)
    except Exception as e:
        out["init_error"] = str(e)
        return out
    for k in range(8):
        try:
            y = r.inference(inputs=[xs[k]])
            if y is None or y[0] is None:
                out["calls"].append(dict(i=k, output=None))
                continue
            y = np.asarray(y[0], dtype=np.float32)
            out["calls"].append(dict(i=k, vs_same=mad(y, ref[k]), vs_previous=mad(y, ref[k - 1]) if k > 0 else None))
        except Exception:
            out["calls"].append(dict(i=k, exception=traceback.format_exc()[-400:]))
    try:
        r.release()
    except Exception:
        out["release_exception"] = traceback.format_exc()[-300:]
    return out


def m7(b1: str, b4: str) -> dict:
    xs = inputs(4)
    out = {}
    try:
        r1 = open_ctx(b1)
        y1 = [np.asarray(r1.inference(inputs=[x])[0], dtype=np.float32) for x in xs]
        t1 = []
        for _ in range(20):
            t0 = time.perf_counter()
            r1.inference(inputs=[xs[0]])
            t1.append(time.perf_counter() - t0)
        r1.release()
        out["batch1_median_ms"] = 1000 * statistics.median(t1)
    except Exception:
        return dict(error_batch1=traceback.format_exc()[-500:])
    try:
        r4 = open_ctx(b4)
        batch = np.concatenate(xs, axis=0)  # (4, 640, 640, 3)
        y4 = np.asarray(r4.inference(inputs=[batch])[0], dtype=np.float32)
        out["batch4_output_shape"] = list(y4.shape)
        out["max_abs_diff_per_image"] = [mad(y4[i:i + 1], y1[i]) if y4.shape[0] == 4 else None for i in range(4)]
        t4 = []
        for _ in range(20):
            t0 = time.perf_counter()
            r4.inference(inputs=[batch])
            t4.append(time.perf_counter() - t0)
        r4.release()
        out["batch4_median_ms"] = 1000 * statistics.median(t4)
        out["batch4_ms_per_image"] = out["batch4_median_ms"] / 4
    except Exception:
        out["error_batch4"] = traceback.format_exc()[-500:]
    return out


def m16(model: str, cap: int = 32, floor_mib: int = 1024) -> dict:
    xs = inputs(1)
    ref = ref_outputs(model, xs)[0]
    ctxs, rows, stop = [], [], None
    for n in range(1, cap + 1):
        try:
            r = open_ctx(model)
            ctxs.append(r)
            d = mad(r.inference(inputs=[xs[0]])[0], ref)
        except Exception:
            stop = dict(reason="failure", at_context=n, text=traceback.format_exc()[-600:])
            break
        mem = meminfo()
        rows.append(dict(n=n, mem_available_mib=mem["MemAvailable"], rss_mib=round(rss_mib(), 1), max_diff=d))
        if mem["MemAvailable"] < floor_mib:
            stop = dict(reason="memory_floor", at_context=n)
            break
    if stop is None:
        stop = dict(reason="cap_reached", at_context=cap)
    for r in ctxs:
        try:
            r.release()
        except Exception:
            pass
    return dict(contexts_created=len(ctxs), stop=stop, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["m4", "m4c", "m6", "m7", "m16"])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--b1")
    ap.add_argument("--b4")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rknn = str(next(a.model.glob("*.rknn"))) if a.model else None
    rec = dict(probe=a.probe, model=rknn, model_sha256=hashlib.sha256(Path(rknn).read_bytes()).hexdigest() if rknn else None, meminfo_before=meminfo())
    rec["result"] = {"m4": lambda: m4(rknn), "m4c": lambda: m4(rknn, 1, 300), "m6": lambda: m6(rknn), "m7": lambda: m7(a.b1, a.b4), "m16": lambda: m16(rknn)}[a.probe]()
    rec["meminfo_after"] = meminfo()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec["result"], indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
