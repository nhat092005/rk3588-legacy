"""CLI: build an INT8 or hybrid RKNN of a trained run with a named quantization recipe (phase 2c).

Input is the normalized-coordinate ONNX kept by diagnose_quant.py (the one the Ultralytics INT8 export feeds to the toolkit) and the
calibration list of baseline seed 0. Recipes (config names from API V2.3.2 p6 to 10 and p14, p20 to 21, UG p124):
  base    the Ultralytics defaults (same config as the deployed INT8, a rebuild used as control)
  mmse    quantized_algorithm="mmse"
  kl      quantized_algorithm="kl_divergence"
  h1      hybrid: the three final class-branch convolutions (model.22/cv3.N/cv3.N.2/Conv) stay float16
  h2      hybrid: the whole class branch (every quantized tensor under model.22/cv3.) stays float16
  h3      hybrid: only the output tail stays float16 (class-branch Concat_1, Sigmoid and output0 tensors, exact names below)
  h4      hybrid: h2 plus the h3 tail
  h5      hybrid: only the box-decode tail stays float16 (DFL, Slice, Sub, Add, Div, Concat_2, the pixel-to-normalized Div and output0 tensors)
  h6      hybrid: h5 plus the whole box branch (model.22/cv2.)
  c_all   hybrid: every int8 tensor except the network input stays float16 (validity control, expected close to the FP16 model)
  auto    build(auto_hybrid=True) with the default cosine threshold of the toolkit
    .venv/bin/python -m ai.automation.build_recipe <run_id> <recipe>
Writes runs/<run_id>/weights/rknn_int8_<recipe>/{best-rk3588.rknn, metadata.yaml, recipe.json}.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
CALIB_RUN = "ppe5_yolov8n_baseline_100ep_s0"
DIAG = RUNS / CALIB_RUN / "bench" / "onboard_q2c" / "diag"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def layer_dtypes(log: Path) -> dict:
    """First 'Network Layer Information Table' of a verbose build log (RKNN_LOG_LEVEL=3): rows per (Target, DataType) and the non-INT8 NPU rows."""
    from ai.core.rknn.benchmark import parse_layers

    text = "\n".join(re.sub(r"^.*?\] ", "", ln) for ln in log.read_text(errors="replace").replace("\r", "\n").split("\n") if "RKNN: [" in ln)
    rows = parse_layers(text[text.index("Network Layer Information Table"):])
    first = []
    for r in rows:
        if first and r["id"] <= first[-1]["id"]:
            break
        first.append(r)
    counts = {}
    for r in first:
        counts[f"{r['target']}|{r['dtype']}"] = counts.get(f"{r['target']}|{r['dtype']}", 0) + 1
    return dict(rows=len(first), counts=counts, non_int8_npu=[dict(id=r["id"], op=r["op"], dtype=r["dtype"], name=r["name"]) for r in first if r["target"] == "NPU" and r["dtype"] != "INT8"],
                cpu_ops=sorted({r["op"] for r in first if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}), total_cycles=sum(r["total_cycles"] or 0 for r in first))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("recipe", choices=["base", "mmse", "kl", "h1", "h2", "h3", "h4", "h5", "h6", "c_all", "auto"])
    args = ap.parse_args()
    from rknn.api import RKNN

    onnx = DIAG / args.run_id / "int8_input.onnx"
    dataset = RUNS / CALIB_RUN / "weights" / "rknn_int8" / "dataset.txt"
    out = RUNS / args.run_id / "weights" / f"rknn_int8_{args.recipe}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    if args.recipe in ("mmse", "kl"):
        cfg["quantized_algorithm"] = {"mmse": "mmse", "kl": "kl_divergence"}[args.recipe]
    vlog = out / "build.log"
    rknn = RKNN(verbose=True, verbose_file=str(vlog))
    assert rknn.config(**cfg) == 0
    assert rknn.load_onnx(model=str(onnx)) == 0
    info = dict(run_id=args.run_id, recipe=args.recipe, config=cfg, onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(), dataset_md5=md5(dataset))
    if args.recipe in ("base", "mmse", "kl"):
        assert rknn.build(do_quantization=True, dataset=str(dataset), rknn_batch_size=1) == 0
    elif args.recipe == "auto":
        assert rknn.build(do_quantization=True, dataset=str(dataset), rknn_batch_size=1, auto_hybrid=True) == 0
    else:
        from ai.core.rknn.hybrid import build_hybrid

        rknn.release()
        info["float16_layers"] = build_hybrid(onnx, dataset, out / "best-rk3588.rknn", args.recipe, verbose_file=vlog)
        rknn = None
    if rknn is not None:
        assert rknn.export_rknn(str(out / "best-rk3588.rknn")) == 0
        rknn.release()
    shutil.copy(RUNS / args.run_id / "weights" / "rknn_int8" / "metadata.yaml", out / "metadata.yaml")
    try:  # with RKNN_LOG_LEVEL=3 the table goes to stdout, not to the verbose file: the driver keeps the stdout log and analyze_q2c parses it
        info["layer_table"] = layer_dtypes(vlog)
    except ValueError:
        info["layer_table"] = None
    (out / "recipe.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"Done: {out} ({len(info.get('float16_layers', []))} float16 layers)")


if __name__ == "__main__":
    main()
