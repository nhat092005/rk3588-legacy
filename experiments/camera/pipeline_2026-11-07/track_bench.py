"""Time BYTETracker.update and the repo's TrackDisplay on synthetic people tracks. Run on the board.

Needs /tmp/trk_pkg (build_trk_pkg.py output) and /tmp/trk_deps (pip --target lapx, provides module `lap`).
Usage: python3 track_bench.py [frames] [comma list of object counts]
Set OPENBLAS_NUM_THREADS=1: with the default BLAS thread count numpy.linalg in the Kalman update is about 100x slower on this board.
"""
import sys, time
from types import SimpleNamespace
sys.path.insert(0, "/tmp/trk_pkg")
sys.path.insert(0, "/tmp/trk_deps")
import numpy as np
import lap  # noqa: F401
from ultralytics.trackers.byte_tracker import BYTETracker
from track_classes import Detections, TrackDisplay

W, H = 1920, 1080
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
WARM = 30


def stats(v):
    v = sorted(v)
    n = len(v)
    return f"median {v[n // 2]:.3f}  p95 {v[int(n * 0.95)]:.3f}  max {v[-1]:.3f} ms"


def scenario(n_obj, frames, rng):
    """Yield (xyxy, conf, cls) per frame: moving people with score dips plus a few false positives."""
    cx = rng.uniform(200, W - 200, n_obj); cy = rng.uniform(250, H - 250, n_obj)
    bw = rng.uniform(60, 200, n_obj); bh = rng.uniform(120, 400, n_obj)
    vx = rng.normal(0, 4, n_obj); vy = rng.normal(0, 3, n_obj)
    base = rng.uniform(0.35, 0.9, n_obj); cls = rng.integers(0, 5, n_obj)
    for _ in range(frames):
        cx += vx; cy += vy
        vx = np.where((cx < 100) | (cx > W - 100), -vx, vx); vy = np.where((cy < 150) | (cy > H - 150), -vy, vy)
        j = rng.normal(0, 1.5, (n_obj, 4))
        x1 = cx - bw / 2 + j[:, 0]; y1 = cy - bh / 2 + j[:, 1]; x2 = cx + bw / 2 + j[:, 2]; y2 = cy + bh / 2 + j[:, 3]
        conf = np.clip(base + rng.normal(0, 0.08, n_obj), 0.1, 0.99)
        dip = rng.random(n_obj) < 0.10
        conf = np.where(dip, rng.uniform(0.12, 0.24, n_obj), conf)
        boxes = np.stack([x1, y1, x2, y2], 1)
        k = max(1, n_obj // 5)
        fx = rng.uniform(0, W - 100, k); fy = rng.uniform(0, H - 100, k)
        fp = np.stack([fx, fy, fx + rng.uniform(30, 120, k), fy + rng.uniform(30, 160, k)], 1)
        boxes = np.concatenate([boxes, fp]); conf = np.concatenate([conf, rng.uniform(0.1, 0.35, k)])
        c = np.concatenate([cls, rng.integers(0, 5, k)])
        yield boxes.astype(np.float32), conf.astype(np.float32), c.astype(np.float32)


print("numpy", np.__version__, "lap", getattr(lap, "__version__", "?"))
NS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [5, 15, 34, 60]
for n_obj in NS:
    rng = np.random.default_rng(0)
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                                          track_buffer=30, match_thresh=0.8, fuse_score=True))
    display = TrackDisplay(5, 15, 3, 5)
    t_trk, t_dis, nin, nout = [], [], [], []
    ids = set()
    for i, (b, c, k) in enumerate(scenario(n_obj, FRAMES + WARM, rng)):
        dets = Detections(b, c, k)
        t0 = time.perf_counter()
        rows = tracker.update(dets)
        t1 = time.perf_counter()
        _, _, draw_ids = display.update(i, rows)
        t2 = time.perf_counter()
        ids.update(int(r[4]) for r in rows)
        if i >= WARM:
            t_trk.append((t1 - t0) * 1e3); t_dis.append((t2 - t1) * 1e3); nin.append(len(b)); nout.append(len(rows))
    tot = [a + b for a, b in zip(t_trk, t_dis)]
    print(f"N={n_obj:3d} dets/frame {sum(nin) / len(nin):5.1f} tracks/frame {sum(nout) / len(nout):5.1f} unique ids {len(ids):4d} | "
          f"update {stats(t_trk)} | display {stats(t_dis)} | total {stats(tot)}")
