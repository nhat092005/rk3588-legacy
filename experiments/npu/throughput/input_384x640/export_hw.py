"""Export the ppe5 baseline s0 INT8 model at a non-square input (H, W) for the NPU task-split test.
Calibration: 100 train images, letterboxed to (H, W) like the evaluator (not the project's calibration set)."""
import math, random, shutil, sys, tempfile
from pathlib import Path
import cv2, yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

H, W = int(sys.argv[1]), int(sys.argv[2])
root = Path("experiments/runs/ppe5_yolov8n_baseline_100ep_s0")
out = Path(f"experiments/npu/throughput/input_384x640/rknn_{H}x{W}")
imgs = sorted(Path("data/ppe5/processed/images/train").glob("*.jpg"))
random.Random(0).shuffle(imgs)
calib = out / "calib"; calib.mkdir(parents=True, exist_ok=True)
lb = LetterBox(new_shape=(H, W), scaleup=False)
for p in imgs[:100]:
    im = cv2.imread(str(p)); h0, w0 = im.shape[:2]
    r = min(H / h0, W / w0)
    if r != 1:
        im = cv2.resize(im, (min(math.ceil(w0 * r), W), min(math.ceil(h0 * r), H)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(calib / f"{p.stem}.png"), lb(image=im))
wdir = out / "w"; wdir.mkdir(parents=True, exist_ok=True)
shutil.copy(root / "weights/best.pt", wdir / "best.pt")  # export on a copy: Ultralytics INT8 export deletes <weights>.onnx
model = YOLO(str(wdir / "best.pt"))
with tempfile.TemporaryDirectory() as tmp:
    y = Path(tmp) / "calib.yaml"
    y.write_text(yaml.safe_dump({"path": tmp, "train": str(calib.resolve()), "val": str(calib.resolve()), "calib": str(calib.resolve()), "names": model.names}))
    res = Path(model.export(format="rknn", name="rk3588", quantize=8, imgsz=[H, W], data=str(y), split="calib"))
dst = out / "model"
if dst.exists(): shutil.rmtree(dst)
shutil.move(str(res), str(dst))
print("EXPORTED", dst, list(dst.glob("*.rknn")))
