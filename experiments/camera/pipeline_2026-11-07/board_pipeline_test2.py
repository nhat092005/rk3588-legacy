"""Sequential camera pipeline on the board: real 1080p NV12 camera frame and a 1080p image with people (many boxes)."""
import sys, time, threading
sys.path.insert(0, "/tmp/bc")
import numpy as np
import cv2
from ai.board.common import preprocess, postprocess
from rknnlite.api import RKNNLite

MODEL = "/home/minhnhat/onboard/best-rk3588.rknn"
GPU = "/sys/class/devfreq/fb000000.gpu/load"


def stats(v):
    v = sorted(v)
    n = len(v)
    return f"median {v[n // 2]:7.3f}  p95 {v[int(n * 0.95)]:7.3f}  mean {sum(v) / n:7.3f} ms"


def freqs():
    s = []
    for p in (0, 4, 6):
        s.append(f"cpu{p}={int(open(f'/sys/devices/system/cpu/cpufreq/policy{p}/scaling_cur_freq').read()) // 1000}")
    s.append("npu=%d" % (int(open("/sys/class/devfreq/fdab0000.npu/cur_freq").read()) // 10**6))
    s.append("ddr=%d" % (int(open("/sys/class/devfreq/dmc/cur_freq").read()) // 10**6))
    s.append("gov=" + open("/sys/devices/system/cpu/cpufreq/policy6/scaling_governor").read().strip())
    return " ".join(s)


raw = np.fromfile("/tmp/f1920x1080.nv12", dtype=np.uint8)
sz = 1920 * 1080 * 3 // 2
nv = raw[-sz:].reshape(1080 * 3 // 2, 1920)
img = cv2.imread("/tmp/test_img.jpg")
img1080 = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
r = RKNNLite()
assert r.load_rknn(MODEL) == 0
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0


def run(name, make_bgr, n=300, warm=30):
    gpu, stop = [], False

    def sampler():
        while not stop:
            gpu.append(open(GPU).read().strip())
            time.sleep(0.25)

    th = threading.Thread(target=sampler)
    th.start()
    tc, tp, tn, tq, ta, nb = [], [], [], [], [], []
    for i in range(n + warm):
        a = time.perf_counter()
        bgr = make_bgr()
        b = time.perf_counter()
        x, meta = preprocess(bgr)
        c = time.perf_counter()
        outs = r.inference(inputs=[x], data_format="nhwc")
        d = time.perf_counter()
        det = postprocess(outs[0], meta, True)
        e = time.perf_counter()
        if i >= warm:
            tc.append((b - a) * 1e3); tp.append((c - b) * 1e3); tn.append((d - c) * 1e3)
            tq.append((e - d) * 1e3); ta.append((e - a) * 1e3); nb.append(len(det))
    stop = True
    th.join()
    print(f"=== {name} | {freqs()} | mean boxes {sum(nb) / len(nb):.1f} | GPU {sorted(set(gpu))}")
    for nm, v in (("NV12->BGR", tc), ("preprocess", tp), ("NPU", tn), ("postprocess", tq), ("total", ta)):
        print(f"  {nm:12s} {stats(v)}")


print("cv2 threads", cv2.getNumThreads())
run("camera NV12 1080p frame (A)", lambda: cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12))
run("image with people 1080p BGR", lambda: img1080)
run("camera NV12 1080p frame (A again)", lambda: cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12))
r.release()
