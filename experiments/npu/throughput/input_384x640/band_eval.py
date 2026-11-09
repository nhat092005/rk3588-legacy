"""Padding-only A/B: build 640x360 'ISP-like' bands from labeled images, then evaluate the SAME weights with
A = square 640x640 letterbox (pad 140) and B = rect 384x640 (pad 12). Same content, same scale, only the padding differs."""
import math, shutil, sys
from pathlib import Path
import cv2, numpy as np, yaml
from ultralytics import YOLO

SRC = Path("data/ppe5/processed"); OUT = Path("experiments/npu/throughput/input_384x640/band")
splits = sys.argv[1].split(",")  # e.g. test,test_ext
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "images/val").mkdir(parents=True); (OUT / "labels/val").mkdir(parents=True)
kept = dropped = nimg = 0
for sp in splits:
    for ip in sorted((SRC / "images" / sp).glob("*.*")):
        lp = SRC / "labels" / sp / (ip.stem + ".txt")
        im = cv2.imread(str(ip)); h0, w0 = im.shape[:2]
        r = 640 / w0; h1 = max(1, round(h0 * r)); im = cv2.resize(im, (640, h1), interpolation=cv2.INTER_LINEAR)
        if h1 >= 360: top = (h1 - 360) // 2; band = im[top:top + 360]; shift = -top
        else: pad = (360 - h1) // 2; band = cv2.copyMakeBorder(im, pad, 360 - h1 - pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114)); shift = pad
        rows = []
        if lp.exists():
            for ln in lp.read_text().split():
                pass
            for ln in lp.read_text().splitlines():
                c, x, y, w, h = ln.split(); c = int(c); x, y, w, h = map(float, (x, y, w, h))
                x1, x2 = (x - w / 2) * 640, (x + w / 2) * 640; y1, y2 = (y - h / 2) * h1 + shift, (y + h / 2) * h1 + shift
                a0 = (x2 - x1) * (y2 - y1); cy1, cy2 = max(0, y1), min(360, y2)
                if cy2 - cy1 < 2 or (cy2 - cy1) * (x2 - x1) < 0.5 * a0: dropped += 1; continue
                rows.append(f"{c} {(x1 + x2) / 2 / 640:.6f} {(cy1 + cy2) / 2 / 360:.6f} {(x2 - x1) / 640:.6f} {(cy2 - cy1) / 360:.6f}"); kept += 1
        name = f"{sp}_{ip.stem}"
        cv2.imwrite(str(OUT / "images/val" / (name + ".png")), band); (OUT / "labels/val" / (name + ".txt")).write_text("\n".join(rows))
        nimg += 1
print("images", nimg, "boxes kept", kept, "dropped", dropped)
y = OUT / "band.yaml"
y.write_text(yaml.safe_dump({"path": str(OUT.resolve()), "train": "images/val", "val": "images/val", "nc": 5, "names": ["Helmet", "NoHelmet", "NoVest", "Vest", "Person"]}))
model = YOLO("experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/best.pt")
res = {}
for tag, rect in (("A square 640x640", False), ("B rect 384x640", True)):
    m = model.val(data=str(y), split="val", imgsz=640, rect=rect, batch=1, device=0, workers=0, plots=False, verbose=False, conf=0.001, iou=0.6)
    res[tag] = (m.box.map, m.box.map50, m.box.mp, m.box.mr, list(m.box.maps))
    print(tag, "AP50-95 %.4f AP50 %.4f P %.4f R %.4f" % res[tag][:4], "per-class AP50-95", ["%.3f" % v for v in res[tag][4]], "speed", m.speed)
