"""INT8 RKNN build replicating ai/automation/export_npu.py (same calibration, same Ultralytics export), verbose compiler output."""
import sys, shutil, yaml
from pathlib import Path
REPO = Path("/home/nhat/workspace/prjs/rk3588")
sys.path.insert(0, str(REPO))
import rknn.api
_orig = rknn.api.RKNN
rknn.api.RKNN = lambda *a, **k: _orig(*a, **{**k, "verbose": True})
from ai.core.dataset import dataset_yaml_path
from ai.core.rknn.calibration import sample_calibration_images, write_letterboxed_calib
from ai.core.rknn.baseline import ExportConfig, export_rknn
work = Path(sys.argv[1])
data = yaml.safe_load(dataset_yaml_path("sfchd_5class").read_text())
images_dir = Path(data["path"]) / data["train"]
labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
stems = sorted(sample_calibration_images(labels_dir, n=300, min_per_class=20, seed=42))
by_stem = {p.stem: p for p in images_dir.iterdir()}
calib = work / "calib"
shutil.rmtree(calib, ignore_errors=True)
write_letterboxed_calib([by_stem[s] for s in stems], calib)
print("calibration images:", len(stems))
out = export_rknn(ExportConfig(weights=str(work / "best.pt"), quantize=8, calib_images_dir=str(calib / "images")), work / "rknn_int8")
print("OUT", out)
