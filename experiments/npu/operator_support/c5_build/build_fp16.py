import sys
from rknn.api import RKNN
onnx_path, log_path = sys.argv[1], sys.argv[2]
r = RKNN(verbose=True, verbose_file=log_path)
r.config(target_platform="rk3588")
assert r.load_onnx(model=onnx_path) == 0
assert r.build(do_quantization=False) == 0
r.release()
