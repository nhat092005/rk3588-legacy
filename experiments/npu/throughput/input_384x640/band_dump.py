# ON THE BOARD: raw NPU outputs for the 640x360 bands, padded to (H, 640) like the evaluator letterbox (gray 114).
import sys, glob, os, numpy as np, cv2
from rknnlite.api import RKNNLite
model, H, mask, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
M = {"CORE_0": RKNNLite.NPU_CORE_0, "CORE_0_1_2": RKNNLite.NPU_CORE_0_1_2}
r = RKNNLite(verbose=False); assert r.load_rknn(model) == 0; assert r.init_runtime(core_mask=M[mask]) == 0
names, idx, vals, counts = [], [], [], []
shape = None
dh = (H - 360) / 2; top, bot = round(dh - 0.1), round(dh + 0.1)
for p in sorted(glob.glob("/tmp/band/*.png")):
    img = cv2.copyMakeBorder(cv2.imread(p), top, bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None]
    assert inp.shape == (1, H, 640, 3), inp.shape
    o = np.asarray(r.inference(inputs=[inp])[0], dtype=np.float32)[0]  # (4+nc, N)
    shape = o.shape; keep = np.flatnonzero(o[4:].max(0) > 0.001)
    names.append(os.path.basename(p)); counts.append(len(keep)); idx.append(keep.astype(np.int32)); vals.append(o[:, keep].T)
r.release()
np.savez(out, names=np.array(names), counts=np.array(counts), idx=np.concatenate(idx), vals=np.concatenate(vals), shape=np.array(shape), top=top)
print("done", out, shape, sum(counts))
