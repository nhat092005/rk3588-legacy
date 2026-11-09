"""CLI: per-layer INT8 error of a trained run, built exactly like the Ultralytics RKNN INT8 export (phase 2c diagnosis).

Runs YOLO(best.pt).export(format="rknn", quantize=8) with the calibration list of baseline seed 0, but replaces
ultralytics.utils.export.rknn.onnx2rknn by a copy of it that also keeps the normalized ONNX and calls
rknn.accuracy_analysis (simulator, no board) on N images. Config values are the ones of Ultralytics 8.4.154
(mean 0, std 255, default quantization algorithm normal, method channel, optimization level 3).

    .venv/bin/python -m ai.npu.operator_support.diagnose_quant <run_id> --images calib:<png dir> test:<png dir> --n 5 --out <dir>
Writes <out>/<run_id>/analysis_<k>/ (error_analysis.txt and the golden and simulator output0.txt), the .rknn and the ONNX it used.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO
from ai.paths import RUNS_DIR

RUNS = RUNS_DIR
CALIB_RUN = "ppe5_yolov8n_baseline_100ep_s0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--images", nargs="+", required=True, help="tag:dir of letterboxed 640x640 PNG images, repeatable (for example calib:<dir> test:<dir>)")
    ap.add_argument("--n", type=int, default=5, help="images analysed per directory (sorted names, evenly spaced)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-analysis", action="store_true", help="only build and keep the ONNX (used to compare with the deployed .rknn)")
    args = ap.parse_args()

    import ultralytics.utils.export.rknn as urknn
    from rknn.api import RKNN

    out = args.out / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    pick = []
    for spec in args.images:
        tag, d = spec.split(":", 1)
        pngs = sorted(Path(d).glob("*.png"))
        pick += [(tag, pngs[round(i * (len(pngs) - 1) / max(args.n - 1, 1))]) for i in range(args.n)]

    def onnx2rknn(onnx_file, output_dir, name="rk3588", quantize=8, batch=1, dataset=None, metadata=None, prefix=""):
        shutil.copy(onnx_file, out / "int8_input.onnx")
        rknn = RKNN(verbose=False)
        assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=name) == 0
        assert rknn.load_onnx(model=onnx_file) == 0
        assert rknn.build(do_quantization=True, dataset=str(dataset), rknn_batch_size=batch) == 0
        assert rknn.export_rknn(str(out / "model-rk3588.rknn")) == 0
        if not args.no_analysis:
            for k, (_, p) in enumerate(pick):
                d = out / f"analysis_{k}"
                assert rknn.accuracy_analysis(inputs=[str(p)], output_dir=str(d)) == 0
                shutil.rmtree(d / "npy", ignore_errors=True)
                for sub in ("golden", "simulator"):  # 1.2 GB per image: keep only the final output tensor
                    shutil.rmtree(d / sub / "npy", ignore_errors=True)
                    for f in (d / sub).glob("*.txt"):
                        if f.name != "output0.txt":
                            f.unlink()
        return str(out)

    urknn.onnx2rknn = onnx2rknn
    weights = RUNS / args.run_id / "weights" / "best.pt"
    model = YOLO(str(weights))
    calib = (RUNS / CALIB_RUN / "weights" / "calib_int8" / "images").resolve()
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "calib.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": tmp, "train": str(calib), "val": str(calib), "calib": str(calib), "names": model.names}))
        model.export(format="rknn", name="rk3588", quantize=8, imgsz=640, data=str(data_yaml), split="calib")
    shutil.rmtree(weights.parent / "best_rknn_model", ignore_errors=True)  # the exporter leaves its dataset.txt folder next to the weights
    (out / "images.txt").write_text("\n".join(f"{t} {p}" for t, p in pick) + "\n")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
