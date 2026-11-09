"""Accuracy loss from the ISP aspect distortion (gap page row 6), protocol v1 evaluator unchanged, FP32 .pt on the PC.

The ISP scales the full 4224x3136 frame (1.347) without a crop. To 640x360 (1.778) the content is stretched 1.32x
horizontally relative to vertically, to 640x640 (1.0) it is stretched 1.347x vertically relative to horizontally.
Each test image gets that relative stretch, then the normal LetterBox runs. YOLO labels are normalized, so an
axis-aligned stretch leaves them valid: the label files are copied unchanged.

  .venv/bin/python experiments/camera/distort/distort_eval.py [run_id]
Writes experiments/camera/distort/results.json. Reference: the stored eval/pt_fp32_<set>.json must be reproduced first.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import yaml

from ai.core import evaluator
from ai.core.backends import UltralyticsBackend
from ai.core.dataset import dataset_yaml_path

SENSOR = 4224 / 3136
MODES = {"isp_640x360": ((640 / 360) / SENSOR, 1.0), "isp_640x640": (1.0, SENSOR / (640 / 640))}  # (x stretch, y stretch)
SETS = ["css_test", "ppe2286_ext", "ppe2286_heldout"]
ROOT = Path("experiments/camera/distort")


def make_set(eval_set: str, mode: str, sx: float, sy: float) -> Path:
    dataset_name, split = evaluator.EVAL_SETS[eval_set]
    data = yaml.safe_load(dataset_yaml_path(dataset_name).read_text())
    src = Path(data["path"]) / data[split]
    out = ROOT / mode / eval_set
    images = out / "images" / "x"
    labels = out / "labels" / "x"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        target = images / (im.stem + ".png")  # lossless: only the stretch differs from the source
        if not target.exists():
            img = cv2.imread(str(im))
            h, w = img.shape[:2]
            cv2.imwrite(str(target), cv2.resize(img, (round(w * sx), round(h * sy)), interpolation=cv2.INTER_LINEAR))
        label = Path(str(im.parent).replace("/images", "/labels")) / (im.stem + ".txt")
        if label.exists():
            shutil.copyfile(label, labels / label.name)
        n += 1
    data["path"] = str(out.resolve())
    data[split] = "images/x"
    yml = out / "dataset.yaml"
    yml.write_text(yaml.safe_dump(data))
    print(f"{mode} {eval_set}: {n} images, stretch x {sx:.4f} y {sy:.4f}", flush=True)
    return yml


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "ppe5_yolov8n_baseline_100ep_s0"
    run_dir = Path("experiments/runs") / run
    backend = UltralyticsBackend(run_dir / "weights" / "best.pt", device="0")
    results = {"run": run, "protocol": evaluator.PROTOCOL_VERSION, "sets": {}}
    original_yaml_path = evaluator.dataset_yaml_path
    for eval_set in SETS:
        stored = json.loads((run_dir / "eval" / f"pt_fp32_{eval_set}.json").read_text())
        ref = evaluator.evaluate(backend, eval_set)
        row = {"stored": stored, "letterbox": ref}
        for mode, (sx, sy) in MODES.items():
            yml = make_set(eval_set, mode, sx, sy)
            evaluator.dataset_yaml_path = lambda _name, y=yml: y  # this eval set only, the evaluator itself is unchanged
            row[mode] = evaluator.evaluate(backend, eval_set)
            evaluator.dataset_yaml_path = original_yaml_path
        results["sets"][eval_set] = row
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "results.json").write_text(json.dumps(results, indent=1))
        print(eval_set, {k: row[k].get("map50_95") for k in ("stored", "letterbox", *MODES)}, flush=True)


if __name__ == "__main__":
    main()
