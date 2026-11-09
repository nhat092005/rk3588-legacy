"""Letterbox vs stretch-to-square on natural (non-square) images, baseline FP32 .pt, protocol v1 evaluator unchanged.

CSS train/val/test were all preprocessed with "Resize to 640x640 (Stretch)" (data/css/raw/README.roboflow.txt), so css_test
cannot give a natural-aspect reference. ppe2286 test images that are not 640x640 keep their native aspect. On those:
  letterbox: the protocol input (native aspect plus padding)
  stretch:   the same image resized to 640x640 first, what the ISP does for a 640x640 output and what Roboflow did to the train set
A paired image bootstrap gives the confidence interval of the mAP50-95 difference.

  .venv/bin/python experiments/camera/stretch/stretch_eval.py [run_id]
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.utils.metrics import ap_per_class

from ai.core import evaluator
from ai.core.backends import UltralyticsBackend
from ai.core.dataset import dataset_yaml_path

SETS = ["ppe2286_ext", "ppe2286_heldout"]
ROOT = Path("experiments/camera/stretch")
BOOT = 1000


def make(eval_set: str, mode: str) -> tuple[Path, int]:
    dataset_name, split = evaluator.EVAL_SETS[eval_set]
    data = yaml.safe_load(dataset_yaml_path(dataset_name).read_text())
    src = Path(data["path"]) / data[split]
    out = ROOT / mode / eval_set
    images, labels = out / "images" / "x", out / "labels" / "x"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        img = cv2.imread(str(im))
        h, w = img.shape[:2]
        if w == h:  # already square: unknown source aspect, not a natural reference
            continue
        target = images / (im.stem + ".png")  # the same lossless encoding for both modes
        if not target.exists():
            cv2.imwrite(str(target), cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) if mode == "stretch" else img)
        label = Path(str(im.parent).replace("/images", "/labels")) / (im.stem + ".txt")
        if label.exists():
            shutil.copyfile(label, labels / label.name)
        n += 1
    data["path"] = str(out.resolve())
    data[split] = "images/x"
    (out / "dataset.yaml").write_text(yaml.safe_dump(data))
    return out / "dataset.yaml", n


def run(backend, eval_set: str, yml: Path) -> tuple[dict, dict]:
    """evaluate() result plus the per-image stats it fed to DetMetrics, keyed by image name."""
    per_image, original = {}, evaluator.DetMetrics.update_stats

    def capture(self, stat):
        per_image[stat["im_name"]] = stat
        return original(self, stat)

    evaluator.DetMetrics.update_stats = capture
    saved = evaluator.dataset_yaml_path
    evaluator.dataset_yaml_path = lambda _name, y=yml: y
    try:
        return evaluator.evaluate(backend, eval_set), per_image
    finally:
        evaluator.DetMetrics.update_stats = original
        evaluator.dataset_yaml_path = saved


def map5095(stats: list[dict], names: dict) -> float:
    tp = np.concatenate([s["tp"] for s in stats])
    conf = np.concatenate([s["conf"] for s in stats])
    pred = np.concatenate([s["pred_cls"] for s in stats])
    target = np.concatenate([s["target_cls"] for s in stats])
    if not len(target):
        return float("nan")
    return float(ap_per_class(tp, conf, pred, target, names=names)[5].mean())


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "ppe5_yolov8n_baseline_100ep_s0"
    backend = UltralyticsBackend(Path("experiments/runs") / run_id / "weights" / "best.pt", device="0")
    rng = np.random.default_rng(0)
    out = {"run": run_id, "boot": BOOT, "sets": {}}
    for eval_set in SETS:
        row = {}
        stats = {}
        for mode in ("letterbox", "stretch"):
            yml, n = make(eval_set, mode)
            row[mode], stats[mode] = run(backend, eval_set, yml)
            row[mode + "_images"] = n
        names = row["letterbox"]["classes"] if isinstance(row["letterbox"].get("classes"), dict) else dict(enumerate(row["letterbox"]["classes"]))
        keys = sorted(stats["letterbox"])
        a, b = [stats["letterbox"][k] for k in keys], [stats["stretch"][k] for k in keys]
        row["recomputed_map50_95"] = {"letterbox": map5095(a, names), "stretch": map5095(b, names)}
        diffs = []
        for _ in range(BOOT):
            idx = rng.integers(0, len(keys), len(keys))
            diffs.append(map5095([b[i] for i in idx], names) - map5095([a[i] for i in idx], names))
        diffs = np.array(diffs)
        row["delta_stretch_minus_letterbox"] = {"mean": float(np.nanmean(diffs)), "ci95": [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])]}
        out["sets"][eval_set] = row
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "results.json").write_text(json.dumps(out, indent=1))
        print(eval_set, row["letterbox_images"], "lb", row["letterbox"]["map50_95"], "st", row["stretch"]["map50_95"], "recomputed", row["recomputed_map50_95"], row["delta_stretch_minus_letterbox"], flush=True)


if __name__ == "__main__":
    main()
