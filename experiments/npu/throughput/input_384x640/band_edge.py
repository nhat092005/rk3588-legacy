import sys; sys.path.insert(0, "experiments/npu/throughput/input_384x640")
import numpy as np, torch
from pathlib import Path
from band_int8_eval import LAB, box_iou, non_max_suppression
def recall(path, edge=40, conf_t=0.25):
    d = np.load(path, allow_pickle=True); names = [str(n) for n in d["names"]]; counts = d["counts"]; N = int(d["shape"][1]); top = int(d["top"]); H = 360 + 2 * top
    off = np.concatenate([[0], np.cumsum(counts)]); hit = {"edge": [0, 0], "inner": [0, 0]}
    for i, n in enumerate(names):
        full = np.zeros((9, N), dtype=np.float32); sl = slice(off[i], off[i + 1]); full[:, d["idx"][sl]] = d["vals"][sl].T
        full[[0, 2]] *= 640; full[[1, 3]] *= H
        out = non_max_suppression(torch.from_numpy(full)[None], conf_t, 0.6, max_det=300)[0]
        out[:, [1, 3]] -= top
        lab = LAB / (Path(n).stem + ".txt"); rows = [l.split() for l in lab.read_text().splitlines()] if lab.exists() else []
        for r in rows:
            c, x, y, w, h = int(r[0]), *map(float, r[1:])
            b = torch.tensor([[(x - w / 2) * 640, (y - h / 2) * 360, (x + w / 2) * 640, (y + h / 2) * 360]])
            near = min(b[0, 1], 360 - b[0, 3]) < edge  # box touches the top/bottom band of `edge` px
            k = "edge" if near else "inner"; hit[k][1] += 1
            if len(out) and ((box_iou(b, out[:, :4])[0] >= 0.5) & (out[:, 5] == c)).any(): hit[k][0] += 1
    return {k: (v[0], v[1], round(v[0] / max(v[1], 1), 3)) for k, v in hit.items()}
for p in sys.argv[1:]: print(Path(p).stem, recall(p))
# numeric identity of CORE_0 vs CORE_0_1_2
a = np.load("experiments/npu/throughput/input_384x640/dump_B384.npz"); b = np.load("experiments/npu/throughput/input_384x640/dump_B384_012.npz")
print("CORE_0 vs CORE_0_1_2 identical dumps:", np.array_equal(a["idx"], b["idx"]) and np.array_equal(a["vals"], b["vals"]), "max abs diff", float(np.abs(a["vals"] - b["vals"]).max()) if a["vals"].shape == b["vals"].shape else "shape differs")
