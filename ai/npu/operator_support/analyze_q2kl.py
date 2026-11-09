"""CLI: phase 2d results of QAT (statements of experiments/npu/operator_support/onboard_q2c/rule_qat.txt) and the no-DFL candidate A2 (rule_nodfl.txt).

    .venv/bin/python -m ai.npu.operator_support.analyze_q2kl
Reads only repo paths (eval JSONs of the runs, export logs, experiments/npu/operator_support/onboard_q2c/logs) and writes results/npu/operator_support/q2k_qat.json,
results/npu/operator_support/q2k_qat_layers.json and results/npu/operator_support/q2l_nodfl.json. The A1 results (q2j_split.json) come from analyze_q2j.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ai.npu.operator_support.build_recipe import layer_dtypes
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUNS = RUNS_DIR
SILU = RUNS / "ppe5_yolov8n_baseline_100ep_s0"
A2 = RUNS / "ppe5_yolov8n_nodfl_100ep_s0"
LOGS = experiment_dir(SILU, "onboard_q2c") / "logs"


def ev(run: Path, name: str) -> dict:
    d = json.loads((run / "eval" / f"{name}_css_test.json").read_text())
    return dict(map50=round(100 * d["map50"], 2), map50_95=round(100 * d["map50_95"], 2),
                per_class={k: round(100 * v["ap50_95"], 2) for k, v in d["per_class"].items()}, sha256=d["model_sha256"], file=f"eval/{name}_css_test.json")


def newest(run: Path, prec: str) -> Path:
    return sorted((run / "logs").glob(f"export_{prec}_*.log"))[-1]


def val(run: Path) -> dict:
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    k = "metrics/mAP50-95(B)"
    return dict(epochs=len(rows), final=float(rows[-1][k]), best=max(float(x[k]) for x in rows))


def main() -> None:
    # --- QAT (rule_qat.txt)
    q = {n: ev(SILU, n) for n in ("rknn_int8_qat", "rknn_int8_split", "rknn_fp16", "rknn_int8h5", "rknn_int8")}
    m = q["rknn_int8_qat"]["map50_95"]
    q["statements"] = {"Q-1_minus_fp16": round(m - q["rknn_fp16"]["map50_95"], 2), "Q-1_minus_split_ptq": round(m - q["rknn_int8_split"]["map50_95"], 2),
                       "Q-1_minus_int8h5": round(m - q["rknn_int8h5"]["map50_95"], 2), "Q-1_minus_int8": round(m - q["rknn_int8"]["map50_95"], 2), "meets_48.41": m >= 48.41}
    q["ppe2286_ext_qat_map50_95"] = json.loads((SILU / "eval" / "rknn_int8_qat_ppe2286_ext.json").read_text())["map50_95"]
    q["note"] = "rule_qat.txt Q-1 evaluated 2026-11-01 on board (RKNN via rknn_server), 1 seed, 1 run. Q-2 layer table in results/npu/operator_support/q2k_qat_layers.json."
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2k_qat.json").write_text(json.dumps(q, indent=1) + "\n")
    ql = dict(note="Q-2 of rule_qat.txt: layer table (first table of the verbose build log), written 2026-11-01 before the QAT mAP exists",
              qat=layer_dtypes(LOGS / "qat_train.log"), split_ptq=layer_dtypes(LOGS / "build_ppe5_yolov8n_baseline_100ep_s0_split_int8.log"),
              qat_train_log=json.loads((SILU / "weights" / "rknn_int8_qat" / "recipe.json").read_text())["log"])
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2k_qat_layers.json").write_text(json.dumps(ql, indent=1) + "\n")
    # --- A2 (rule_nodfl.txt)
    a = {n: ev(A2, n) for n in ("pt_fp32", "rknn_fp16", "rknn_int8")}
    b = {n: ev(SILU, n) for n in ("pt_fp32", "rknn_fp16", "rknn_int8")}
    drop = round(a["rknn_fp16"]["map50_95"] - a["rknn_int8"]["map50_95"], 2)
    ref = round(b["rknn_fp16"]["map50_95"] - b["rknn_int8"]["map50_95"], 2)
    lo, hi = ref - 1.35, ref + 1.35
    reading = ("drop_A2 at most 1.35" if drop <= 1.35 else
               "within 1.35 of SiLU drop: decode arithmetic in INT8 enough to cause the loss without DFL" if lo <= drop <= hi else
               "partial (1.35 to lo)" if drop < lo else "larger than hi")
    exp = json.loads((A2 / "export_int8.json").read_text())
    out = dict(note="N-1 to N-4 of rule_nodfl.txt, evaluated 2026-11-01 on board (RKNN via rknn_server), 1 seed, 1 run, git_dirty true",
               a2=a, silu=b, drop_a2=drop, drop_silu=ref, band=[round(lo, 2), round(hi, 2)], reading=reading, n3_int8_ge_48_41=a["rknn_int8"]["map50_95"] >= 48.41,
               val_map50_95_a2=val(A2), val_map50_95_silu=val(SILU), transferred="285/354 items (nc 5 training log)",
               calibration=dict(stems_md5=exp["calibration"]["stems_md5"], png_md5=exp["calibration"]["png_md5"]), rknn_sha256_int8=exp["rknn_sha256"],
               layers_int8=layer_dtypes(newest(A2, "int8")), layers_fp16=layer_dtypes(newest(A2, "fp16")), second_experiment_run=False)
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2l_nodfl.json").write_text(json.dumps(out, indent=1) + "\n")
    print(q["statements"], drop, ref, reading)


if __name__ == "__main__":
    main()
