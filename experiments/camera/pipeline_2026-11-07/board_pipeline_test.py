"""Time the CPU/NPU stages of the camera pipeline on the board with real NV12 camera frames, sample GPU load meanwhile."""
import sys, time, threading
sys.path.insert(0, "/tmp/bc")
import numpy as np
import cv2
from ai.board.common import preprocess, postprocess

GPU = "/sys/class/devfreq/fb000000.gpu/load"
MODEL = "/home/minhnhat/onboard/best-rk3588.rknn"


def stats(v):
    v = sorted(v)
    n = len(v)
    return f"median {v[n // 2]:.3f}  p95 {v[int(n * 0.95)]:.3f}  mean {sum(v) / n:.3f} ms"


def tm(fn, n=200, warm=20):
    out = []
    for i in range(n + warm):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i >= warm:
            out.append((t1 - t0) * 1e3)
    return out


def cpu_freqs():
    s = []
    for p in (0, 4, 6):
        s.append(f"policy{p}={int(open(f'/sys/devices/system/cpu/cpufreq/policy{p}/scaling_cur_freq').read()) // 1000}MHz")
    return " ".join(s)


frames = {}
for name, w, h in [("4224x3136", 4224, 3136), ("1920x1080", 1920, 1080), ("640x360", 640, 360)]:
    raw = np.fromfile(f"/tmp/f{name}.nv12", dtype=np.uint8)
    sz = w * h * 3 // 2
    assert raw.size % sz == 0, (name, raw.size, sz)
    frames[name] = raw[-sz:].reshape(h * 3 // 2, w)
print("cv2", cv2.__version__, "threads", cv2.getNumThreads(), "|", cpu_freqs())

print("--- NV12 -> BGR (cv2.cvtColor COLOR_YUV2BGR_NV12), real camera frames")
for nthr in (cv2.getNumThreads(), 1):
    cv2.setNumThreads(nthr)
    for name, nv in frames.items():
        print(f"{name:10s} threads={nthr}  {stats(tm(lambda: cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12), n=100, warm=10))}")
cv2.setNumThreads(8)

bgr = {k: cv2.cvtColor(v, cv2.COLOR_YUV2BGR_NV12) for k, v in frames.items()}
print("--- preprocess(bgr) of ai/board/common.py (resize INTER_LINEAR, border 114, BGR->RGB)")
for name, im in bgr.items():
    print(f"{name:10s}  {stats(tm(lambda: preprocess(im), n=200, warm=20))}")

print("--- full CPU+NPU pipeline on the real 1920x1080 frame, GPU load sampled meanwhile")
from rknnlite.api import RKNNLite
r = RKNNLite()
assert r.load_rknn(MODEL) == 0
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
nv = frames["1920x1080"]
gpu = []
stop = False


def sampler():
    while not stop:
        try:
            gpu.append(open(GPU).read().strip())
        except Exception as e:
            gpu.append(f"err {e}")
        time.sleep(0.25)


th = threading.Thread(target=sampler)
th.start()
t_cvt, t_pre, t_npu, t_post, t_all, nbox = [], [], [], [], [], []
for i in range(220):
    a = time.perf_counter()
    b_ = cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12)
    b = time.perf_counter()
    img, meta = preprocess(b_)
    c = time.perf_counter()
    outs = r.inference(inputs=[img], data_format="nhwc")
    d = time.perf_counter()
    det = postprocess(outs[0], meta, True)
    e = time.perf_counter()
    if i >= 20:
        t_cvt.append((b - a) * 1e3); t_pre.append((c - b) * 1e3); t_npu.append((d - c) * 1e3)
        t_post.append((e - d) * 1e3); t_all.append((e - a) * 1e3); nbox.append(len(det))
stop = True
th.join()
print("out shape", [o.shape for o in outs], "dtype", outs[0].dtype, "mean boxes", sum(nbox) / len(nbox))
for nm, v in (("NV12->BGR", t_cvt), ("preprocess", t_pre), ("NPU inference", t_npu), ("postprocess", t_post), ("total", t_all)):
    print(f"{nm:14s} {stats(v)}")
print("GPU load samples:", sorted(set(gpu)), "n =", len(gpu))
print("freq after:", cpu_freqs())
r.release()
