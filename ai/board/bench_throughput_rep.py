"""One throughput run for the multi-core study (S2, S3 and the process diagnostic), run ON THE BOARD.

One invocation = one independent run (own process, own contexts): the driver repeats it 3 times per configuration. Modes:
  threads   one Python thread per context, each running pre -> NPU -> post (S2, or one context for S1 throughput)
  pipeline  S3: pre workers -> bounded queue -> one NPU worker per context -> bounded queue -> post workers
  procs     diagnostic only: one process per context (compares with threads, tells whether the GIL matters)
Per frame it records T_pre, T_npu and T_post. The record also holds the placement evidence of the run (CPU affinity of every process,
NPU interrupts per CPU before and after, temperatures, frequencies, RSS after each context init).
Usage: taskset -c 4-7 .venv-board/bin/python -m ai.board.bench_throughput_rep --model <dir> --images <dir> --mode threads --core-masks CORE_0,CORE_1,CORE_2 --out <json> --run-id <id> --git-commit <sha> --git-dirty true
"""
import argparse
import datetime
import itertools
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import psutil
import yaml

from ai.board.common import (
    WARMUP, ResourceMonitor, add_provenance_args, frequency_state, list_images, postprocess_raw, preprocess, provenance, summarize, system_info,
)

IRQ_LINES = ("36", "37", "38")


def irq_counts() -> dict:
    out = {}
    for line in Path("/proc/interrupts").read_text().splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") in IRQ_LINES:
            out[parts[0].rstrip(":")] = [int(x) for x in parts[1:1 + os.cpu_count()]]
    return out


def load_ctx(model_dir: Path, mask: str):
    """(rknn, normalized, split): split is True for a split-head model (metadata split_head, two raw outputs decoded in numpy)."""
    from ai.board.bench_npu import load_model

    rknn, normalized = load_model(model_dir, mask)
    meta = yaml.safe_load((model_dir / "metadata.yaml").read_text())
    return rknn, normalized, bool(meta.get("split_head"))


def warm_up(rknn, normalized, split, frames) -> None:
    for i in range(WARMUP):
        inp, meta = preprocess(frames[i % len(frames)])
        postprocess_raw(rknn.inference(inputs=[inp]), meta, normalized, split)


def run_threads(contexts, frames, n_frames):
    """contexts: [(rknn, normalized, split)]. Returns wall seconds and per-stage ms lists."""
    counter, lock = itertools.count(), threading.Lock()
    pre, npu, post = [[] for _ in contexts], [[] for _ in contexts], [[] for _ in contexts]
    start = threading.Barrier(len(contexts) + 1)

    def worker(k):
        rknn, normalized, split = contexts[k]
        start.wait()
        while True:
            with lock:
                i = next(counter)
            if i >= n_frames:
                return
            t0 = time.perf_counter()
            inp, meta = preprocess(frames[i % len(frames)])
            t1 = time.perf_counter()
            out = rknn.inference(inputs=[inp])
            t2 = time.perf_counter()
            postprocess_raw(out, meta, normalized, split)
            t3 = time.perf_counter()
            pre[k].append((t1 - t0) * 1e3)
            npu[k].append((t2 - t1) * 1e3)
            post[k].append((t3 - t2) * 1e3)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(contexts))]
    for t in threads:
        t.start()
    start.wait()
    t_start = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    flat = lambda ll: [x for l in ll for x in l]  # noqa: E731
    return wall, flat(pre), flat(npu), flat(post), [len(l) for l in npu]


def run_pipeline(contexts, frames, n_frames, n_pre, n_post, qsize):
    """S3. Returns wall seconds, per-stage ms lists, end-to-end ms list and frames per NPU worker."""
    counter, lock = itertools.count(), threading.Lock()
    q_in, q_out = queue.Queue(qsize), queue.Queue(qsize)
    pre, npu, post, e2e = [], [], [], []
    per_npu = [0] * len(contexts)
    start = threading.Barrier(n_pre + len(contexts) + n_post + 1)

    def pre_worker():
        start.wait()
        while True:
            with lock:
                i = next(counter)
            if i >= n_frames:
                return
            t0 = time.perf_counter()
            inp, meta = preprocess(frames[i % len(frames)])
            pre.append((time.perf_counter() - t0) * 1e3)
            q_in.put((t0, inp, meta))

    def npu_worker(k):
        rknn, normalized, split = contexts[k]
        start.wait()
        while True:
            item = q_in.get()
            if item is None:
                return
            t0, inp, meta = item
            t1 = time.perf_counter()
            out = rknn.inference(inputs=[inp])
            npu.append((time.perf_counter() - t1) * 1e3)
            per_npu[k] += 1
            q_out.put((t0, out, meta, normalized, split))

    def post_worker():
        start.wait()
        while True:
            item = q_out.get()
            if item is None:
                return
            t0, out, meta, normalized, split = item
            t1 = time.perf_counter()
            postprocess_raw(out, meta, normalized, split)
            t2 = time.perf_counter()
            post.append((t2 - t1) * 1e3)
            e2e.append((t2 - t0) * 1e3)

    pres = [threading.Thread(target=pre_worker) for _ in range(n_pre)]
    npus = [threading.Thread(target=npu_worker, args=(k,)) for k in range(len(contexts))]
    posts = [threading.Thread(target=post_worker) for _ in range(n_post)]
    for t in pres + npus + posts:
        t.start()
    start.wait()
    t_start = time.perf_counter()
    for t in pres:
        t.join()
    for _ in npus:
        q_in.put(None)
    for t in npus:
        t.join()
    for _ in posts:
        q_out.put(None)
    for t in posts:
        t.join()
    return time.perf_counter() - t_start, pre, npu, post, e2e, per_npu


def _proc_worker(args, mask, counter, barrier, conn):
    frames = [cv2.imread(str(p)) for p in list_images(args.images)[:args.cache]]
    rknn, normalized, split = load_ctx(args.model, mask)
    warm_up(rknn, normalized, split, frames)
    rss = psutil.Process().memory_info().rss / 2**20
    pre, npu, post = [], [], []
    barrier.wait()
    while True:
        with counter.get_lock():
            i = counter.value
            counter.value += 1
        if i >= args.frames:
            break
        t0 = time.perf_counter()
        inp, meta = preprocess(frames[i % len(frames)])
        t1 = time.perf_counter()
        out = rknn.inference(inputs=[inp])
        t2 = time.perf_counter()
        postprocess_raw(out, meta, normalized, split)
        t3 = time.perf_counter()
        pre.append((t1 - t0) * 1e3)
        npu.append((t2 - t1) * 1e3)
        post.append((t3 - t2) * 1e3)
    conn.send(dict(pre=pre, npu=npu, post=post, rss_mib=rss, affinity=sorted(os.sched_getaffinity(0))))
    rknn.release()


def run_procs(args, masks):
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    barrier = ctx.Barrier(len(masks) + 1)
    pipes = [ctx.Pipe() for _ in masks]
    procs = [ctx.Process(target=_proc_worker, args=(args, m, counter, barrier, pipes[k][1])) for k, m in enumerate(masks)]
    for p in procs:
        p.start()
    barrier.wait()
    t_start = time.perf_counter()
    results = [pipes[k][0].recv() for k in range(len(masks))]
    for p in procs:
        p.join()
    wall = time.perf_counter() - t_start
    flat = lambda key: [x for r in results for x in r[key]]  # noqa: E731
    return wall, flat("pre"), flat("npu"), flat("post"), [len(r["npu"]) for r in results], [r["rss_mib"] for r in results], [r["affinity"] for r in results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="rknn_<precision>/ folder")
    parser.add_argument("--images", required=True)
    parser.add_argument("--mode", required=True, choices=["threads", "pipeline", "procs"])
    parser.add_argument("--core-masks", required=True, help="one per context, e.g. CORE_0,CORE_1,CORE_2")
    parser.add_argument("--cache", type=int, default=200, help="decoded images kept in RAM")
    parser.add_argument("--frames", type=int, default=1238, help="total frames processed")
    parser.add_argument("--pre-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=1)
    parser.add_argument("--queue-size", type=int, default=6)
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()
    masks = args.core_masks.split(",")

    from ai.board.bench_npu import CORE_MASK

    for m in masks:
        if m not in CORE_MASK:
            parser.error(f"unknown core mask {m}")
    irq_before, freq_before = irq_counts(), frequency_state()
    rss = {"start": psutil.Process().memory_info().rss / 2**20}
    extra = {}
    with ResourceMonitor() as monitor:
        if args.mode == "procs":
            wall, pre, npu, post, per_ctx, rss["per_process_after_init"], aff = run_procs(args, masks)
            extra["affinity_per_process"] = aff
        else:
            frames = [cv2.imread(str(p)) for p in list_images(args.images)[:args.cache]]
            rss["frames_loaded"] = psutil.Process().memory_info().rss / 2**20
            contexts = []
            for m in masks:
                contexts.append(load_ctx(args.model, m))
                rss[f"after_context_{len(contexts)}"] = psutil.Process().memory_info().rss / 2**20
            for rknn, normalized, split in contexts:
                warm_up(rknn, normalized, split, frames)
            extra["affinity_process"] = sorted(os.sched_getaffinity(0))
            if args.mode == "threads":
                wall, pre, npu, post, per_ctx = run_threads(contexts, frames, args.frames)
            else:
                wall, pre, npu, post, e2e, per_ctx = run_pipeline(contexts, frames, args.frames, args.pre_workers, args.post_workers, args.queue_size)
                extra.update(pre_workers=args.pre_workers, post_workers=args.post_workers, queue_size=args.queue_size, e2e_latency_ms=summarize(e2e))
            for rknn, *_ in contexts:
                rknn.release()
    irq_after = irq_counts()
    record = {
        **provenance(args, next(args.model.glob("*.rknn"))),
        "runtime": f"rknnlite, {args.mode}",
        "model": str(args.model),
        "mode": args.mode,
        "contexts": len(masks),
        "core_masks": masks,
        "cached_images": min(args.cache, len(list_images(args.images))),
        "frames": args.frames,
        "warmup_per_context": WARMUP,
        "measured_at": datetime.datetime.now().isoformat(),
        "system": system_info(),
        "frequency_before": freq_before,
        "frequency_after": frequency_state(),
        "wall_time_s": round(wall, 3),
        "throughput_fps": round(args.frames / wall, 2),
        "t_pre_ms": summarize(pre),
        "t_npu_ms": summarize(npu),
        "t_post_ms": summarize(post),
        "frames_per_context": per_ctx,
        "rss_mib": rss,
        "irq_delta_per_cpu": {ln: [a - b for a, b in zip(irq_after[ln], irq_before[ln])] for ln in irq_before},
        "resource": monitor.result(),
        **extra,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Done: {args.out}  throughput {record['throughput_fps']} FPS, mode {args.mode}, masks {masks}")


if __name__ == "__main__":
    main()
