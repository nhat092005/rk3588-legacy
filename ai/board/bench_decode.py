"""CPU time of the split-head decode (numpy), run ON THE BOARD: 200 decodes of random raw outputs, median and p95 in ms.

Usage (pin it like the measurement): taskset -c 4 .venv-board/bin/python -m ai.board.bench_decode --out <json> --run-id <id> --git-commit <sha> --git-dirty true
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ai.board.common import frequency_state
from ai.board.split_decode import decode_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--git-dirty", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    boxes = rng.normal(0, 4, (1, 64, 8400)).astype(np.float32)
    scores = rng.normal(-3, 3, (1, 5, 8400)).astype(np.float32)
    for _ in range(20):
        decode_split(boxes, scores)
    t = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        decode_split(boxes, scores)
        t.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(t)
    rec = {"n": args.n, "median_ms": round(float(np.median(a)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3), "min_ms": round(float(a.min()), 3), "frequency": frequency_state(),
           "run_id": args.run_id, "git_commit": args.git_commit, "git_dirty": args.git_dirty == "true"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"decode median {rec['median_ms']} ms, p95 {rec['p95_ms']} ms")


if __name__ == "__main__":
    main()
