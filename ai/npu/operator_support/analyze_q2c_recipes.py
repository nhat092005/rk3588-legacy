"""CLI: 2c-c analysis, statements R-a to R-d of experiments/npu/operator_support/onboard_q2c/rule_recipes.txt.

    .venv/bin/python -m ai.npu.operator_support.analyze_q2c_recipes
Reads runs/<run>/eval/rknn_int8[_<recipe>]_<set>.json, pt_fp32 and rknn_fp16 references, the build logs and recipe.json of each recipe,
and the kept-anchor counts of the board dumps. Writes results/npu/operator_support/q2c_recipes.{json,md}.
"""
import json
from pathlib import Path
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUNS = RUNS_DIR
HOME = experiment_dir(RUNS / "ppe5_yolov8n_baseline_100ep_s0", "onboard_q2c")
RUN = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0"}
RECIPES = ["base", "kl", "h1", "h2", "h3", "h4", "h5", "h6", "c_all", "auto"]
SETS = ("css_test", "ppe2286_ext")
COLS = ("map50_95", "NoHelmet", "Vest")


def ap(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return dict(map50=100 * d["map50"], precision=d["precision"], recall=d["recall"], map50_95=100 * d["map50_95"], NoHelmet=100 * d["per_class"]["NoHelmet"]["ap50_95"], Vest=100 * d["per_class"]["Vest"]["ap50_95"], sha=d["model_sha256"])


def main() -> None:
    from ai.npu.operator_support.build_recipe import layer_dtypes

    out = {"models": {}}
    for m, r in RUN.items():
        ev = RUNS / r / "eval"
        res = {"ref": {}, "recipes": {}}
        for st in SETS:
            res["ref"][st] = dict(int8=ap(ev / f"rknn_int8_{st}.json"), fp16=ap(ev / f"rknn_fp16_{st}.json"), fp32=ap(ev / f"pt_fp32_{st}.json"))
        for rc in RECIPES:
            row = {}
            d = RUNS / r / "weights" / f"rknn_int8_{rc}"
            if (d / "recipe.json").exists():
                info = json.loads((d / "recipe.json").read_text())
                log = HOME / "logs" / f"build_{r}_{rc}.log"
                try:
                    lt = layer_dtypes(log)
                except (ValueError, FileNotFoundError):
                    lt = None
                row["build"] = dict(float16_layers=info.get("float16_layers"), layer_table=lt, onnx_sha256=info["onnx_sha256"], dataset_md5=info["dataset_md5"])
            for st in SETS:
                v = ap(ev / f"rknn_int8_{rc}_{st}.json")
                dump = RUNS / r / "weights" / f"npu_dump_int8_{rc}_{st}.json"
                row[st] = dict(ap=v, kept_anchors=json.loads(dump.read_text())["kept_anchors"] if dump.exists() else None)
            res["recipes"][rc] = row
        out["models"][m] = res
    # R-a rebuild check, R-b recovery, R-c gate C under the recipe
    out["r_a"], out["r_b"], out["r_c"] = {}, {}, {}
    for m in RUN:
        i8, b = out["models"][m]["ref"]["css_test"]["int8"], out["models"][m]["recipes"]["base"]["css_test"]["ap"]
        if b:
            diff = b["map50_95"] - i8["map50_95"]
            out["r_a"][m] = dict(diff_pp=diff, verdict="rebuild equivalent" if abs(diff) <= 0.5 else "rebuild not equivalent")
        f16 = out["models"][m]["ref"]["css_test"]["fp16"]
        for rc in RECIPES:
            v = out["models"][m]["recipes"][rc]["css_test"]["ap"]
            if v and b:
                gap = f16["map50_95"] - b["map50_95"]
                out["r_b"][f"{m}|{rc}"] = dict(diff_vs_base={c: v[c] - b[c] for c in COLS}, recovered_fraction=(v["map50_95"] - b["map50_95"]) / gap if gap else None, fp16_minus_base=gap)
    for rc in RECIPES:
        for st in SETS:
            s_, r_ = out["models"]["silu"]["recipes"][rc][st]["ap"], out["models"]["relu"]["recipes"][rc][st]["ap"]
            if s_ and r_:
                d = {c: r_[c] - s_[c] for c in COLS}
                out["r_c"][f"{rc}|{st}"] = dict(silu=s_, relu=r_, diff_pp=d, passes=all(x >= -1.0 for x in d.values()))
    # ADDENDUM 3 of rule_recipes.txt: gate C under h5 for the deployable variants (R-c-h5, R-e-h5)
    var_runs = {"relu200": "ppe5_yolov8n_relu_200ep_s0", "leakyrelu": "ppe5_yolov8n_leakyrelu_100ep_s0", "relu6": "ppe5_yolov8n_relu6_100ep_s0", "silu200": "ppe5_yolov8n_baseline_200ep_s0"}
    if all((RUNS / r / "eval" / "rknn_int8_h5_css_test.json").exists() for r in var_runs.values()):
        silu_h5 = {st: out["models"]["silu"]["recipes"]["h5"][st]["ap"] for st in SETS}
        vh = {}
        for k, r in var_runs.items():
            vh[k] = {}
            for st in SETS:
                v = ap(RUNS / r / "eval" / f"rknn_int8_h5_{st}.json")
                d = {c: v[c] - silu_h5[st][c] for c in COLS}
                row = dict(ap=v, diff_vs_silu_h5=d, passes=all(x >= -1.0 for x in d.values()))
                if k == "relu200":
                    c2 = ap(RUNS / var_runs["silu200"] / "eval" / f"rknn_int8_h5_{st}.json")
                    row["diff_vs_silu200_h5"] = {c: v[c] - c2[c] for c in COLS}
                vh[k][st] = row
        out["variants_h5"] = vh
    out["rule"] = (HOME / "rule_recipes.txt").read_text()
    out["q2c_diag_note"] = "results/npu/operator_support/q2c_diag.json"
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2c_recipes.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# 2c-c recipes", "", "| model | recipe | AP50-95 | NoHelmet | Vest | kept anchors (css_test) |", "|---|---|---|---|---|---|"]
    for m in RUN:
        i8 = out["models"][m]["ref"]["css_test"]
        md.append(f"| {m} | deployed int8 | {i8['int8']['map50_95']:.2f} | {i8['int8']['NoHelmet']:.2f} | {i8['int8']['Vest']:.2f} | |")
        md.append(f"| {m} | fp16 | {i8['fp16']['map50_95']:.2f} | {i8['fp16']['NoHelmet']:.2f} | {i8['fp16']['Vest']:.2f} | |")
        for rc in RECIPES:
            v = out["models"][m]["recipes"][rc]["css_test"]
            if v["ap"]:
                md.append(f"| {m} | {rc} | {v['ap']['map50_95']:.2f} | {v['ap']['NoHelmet']:.2f} | {v['ap']['Vest']:.2f} | {v['kept_anchors']} |")
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2c_recipes.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
