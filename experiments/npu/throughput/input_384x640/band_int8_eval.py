"""Host-side AP of NPU raw dumps on the 640x360 bands (Ultralytics NMS and matching, conf 0.001, iou 0.6, max_det 300)."""
import sys
from pathlib import Path
import numpy as np, torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou

LAB = Path("experiments/npu/throughput/input_384x640/band/labels/val")
iouv = torch.linspace(0.5, 0.95, 10)

def match(pc, tc, iou):  # DetectionValidator.match_predictions (one-to-one, greedy by IoU)
    correct = np.zeros((pc.shape[0], 10), dtype=bool)
    iou = iou.cpu().numpy(); cm = (tc[:, None] == pc[None]).astype(np.float32); iou = iou * cm
    for i, th in enumerate(iouv.cpu().tolist()):
        m = np.nonzero(iou >= th)
        if m[0].shape[0]:
            mt = np.concatenate([np.stack(m, 1), iou[m[0], m[1]][:, None]], 1)
            if m[0].shape[0] > 1:
                mt = mt[mt[:, 2].argsort()[::-1]]; mt = mt[np.unique(mt[:, 1], return_index=True)[1]]; mt = mt[np.unique(mt[:, 0], return_index=True)[1]]
            correct[mt[:, 1].astype(int), i] = True
    return torch.tensor(correct)

def evaluate(path):
    d = np.load(path, allow_pickle=True); names = [str(n) for n in d["names"]]; counts = d["counts"]; nc_a = d["shape"]; N = int(nc_a[1]); H = 360 + 2 * int(d["top"])
    off = np.concatenate([[0], np.cumsum(counts)])
    stats = []
    for i, n in enumerate(names):
        full = np.zeros((9, N), dtype=np.float32); sl = slice(off[i], off[i + 1])
        full[:, d["idx"][sl]] = d["vals"][sl].T
        full[[0, 2]] *= 640; full[[1, 3]] *= H          # boxes normalized per axis (x by 640, y by H)
        pred = torch.from_numpy(full)[None]
        out = non_max_suppression(pred, 0.001, 0.6, max_det=300)[0]  # xyxy, conf, cls in padded coords
        out[:, [1, 3]] -= int(d["top"]); out[:, [0, 2]] = out[:, [0, 2]].clamp(0, 640); out[:, [1, 3]] = out[:, [1, 3]].clamp(0, 360)
        lab = LAB / (Path(n).stem + ".txt"); rows = [l.split() for l in lab.read_text().splitlines()] if lab.exists() else []
        if rows:
            g = np.array(rows, dtype=np.float32); tc = g[:, 0].astype(int); xy = g[:, 1:]
            tb = torch.from_numpy(np.stack([(xy[:, 0] - xy[:, 2] / 2) * 640, (xy[:, 1] - xy[:, 3] / 2) * 360, (xy[:, 0] + xy[:, 2] / 2) * 640, (xy[:, 1] + xy[:, 3] / 2) * 360], 1))
        else: tc = np.zeros(0, dtype=int); tb = torch.zeros((0, 4))
        pc = out[:, 5].cpu().numpy().astype(int)
        correct = match(pc, tc, box_iou(tb, out[:, :4])) if len(out) and len(tc) else torch.zeros((len(out), 10), dtype=torch.bool)
        stats.append((correct.numpy(), out[:, 4].cpu().numpy(), pc, tc))
    tp = np.concatenate([s[0] for s in stats]); conf = np.concatenate([s[1] for s in stats]); pcs = np.concatenate([s[2] for s in stats]); tcs = np.concatenate([s[3] for s in stats])
    r = ap_per_class(tp, conf, pcs, tcs)
    ap = r[5]  # (nc_present, 10)
    return ap.mean(), ap[:, 0].mean(), ap.mean(1), r[0]  # mAP50-95, mAP50, per class, precision per class

if __name__ == "__main__":
    for p in sys.argv[1:]:
        m, m50, pc, _ = evaluate(p)
        print(f"{Path(p).stem:12s} AP50-95 {m:.4f} AP50 {m50:.4f} per-class {np.round(pc, 3).tolist()}")
