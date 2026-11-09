"""Keep the NPU busy (CORE_0) for N seconds with zero input, report latency. Run on the board."""
import sys, time
import numpy as np
from rknnlite.api import RKNNLite

secs = float(sys.argv[1])
r = RKNNLite()
assert r.load_rknn("/home/minhnhat/onboard/best-rk3588.rknn") == 0
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
x = np.zeros((1, 640, 640, 3), dtype=np.uint8)
lat = []
t_end = time.time() + secs
while time.time() < t_end:
    t0 = time.perf_counter()
    r.inference(inputs=[x], data_format="nhwc")
    lat.append((time.perf_counter() - t0) * 1e3)
lat.sort()
print(f"npu_loop n={len(lat)} median {lat[len(lat) // 2]:.3f} p95 {lat[int(len(lat) * 0.95)]:.3f} ms")
r.release()
