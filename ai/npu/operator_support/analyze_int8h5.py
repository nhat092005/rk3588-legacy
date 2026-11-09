"""CLI: standard export int8h5, statements V-1 to V-3 of experiments/npu/operator_support/onboard_q2c/rule_int8h5.txt (V-4 latency has its own rule).

    .venv/bin/python -m ai.npu.operator_support.analyze_int8h5
Writes results/npu/operator_support/int8h5.{json,md}.
"""
import hashlib
import json
import statistics
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUNS = RUNS_DIR
RUN = {"silu_s0": "ppe5_yolov8n_baseline_100ep_s0", "silu_s1": "ppe5_yolov8n_baseline_100ep_s1", "silu_s2": "ppe5_yolov8n_baseline_100ep_s2", "relu_s0": "ppe5_yolov8n_relu_100ep_s0"}
HOME = experiment_dir(RUNS / RUN["silu_s0"], "onboard_q2c")
SETS = ("css_test", "ppe2286_ext")
COLS = ("map50_95", "NoHelmet", "Vest")
S0_INT8_SHA = "a501f94ee391bab370c141bbc4853eb78c062635b0a01597fe3a9efb0d99e7dd"


def ap(run: str, tag: str, st: str) -> dict:
    d = json.loads((RUNS / run / "eval" / f"{tag}_{st}.json").read_text())
    return dict(map50=100 * d["map50"], map50_95=100 * d["map50_95"], NoHelmet=100 * d["per_class"]["NoHelmet"]["ap50_95"], Vest=100 * d["per_class"]["Vest"]["ap50_95"], sha=d["model_sha256"])


def main() -> None:
    out = {"runs": {}}
    for k, r in RUN.items():
        ex = json.loads((RUNS / r / "export_int8h5.json").read_text())
        out["runs"][k] = dict(export=dict(sha=ex["rknn_sha256"], recipe=ex["recipe"], n_float16=len(ex["float16_layers"]), stems_md5=ex["calibration"]["stems_md5"], png_md5=ex["calibration"]["png_md5"]),
                              sets={st: dict(int8h5=ap(r, "rknn_int8h5", st), int8=ap(r, "rknn_int8", st), fp16=ap(r, "rknn_fp16", st)) for st in SETS})
        for st in SETS:
            v = out["runs"][k]["sets"][st]
            v["diff_vs_int8"] = {c: v["int8h5"][c] - v["int8"][c] for c in COLS}
            v["diff_vs_fp16"] = {c: v["int8h5"][c] - v["fp16"][c] for c in COLS}
    s0 = RUNS / RUN["silu_s0"]
    h5 = json.loads((s0 / "weights" / "rknn_int8_h5" / "recipe.json").read_text())
    ex0 = json.loads((s0 / "export_int8h5.json").read_text())
    ref = json.loads((RUNS / RUN["silu_s0"] / "eval" / "rknn_int8_h5_css_test.json").read_text())
    new = out["runs"]["silu_s0"]["sets"]["css_test"]["int8h5"]["map50_95"]
    layers_equal = ex0["float16_layers"] == h5["float16_layers"]
    diff = new - 100 * ref["map50_95"]
    out["v1"] = dict(layers_equal=layers_equal, ap_diff_pp=diff, verdict="export reproduces h5" if layers_equal and abs(diff) <= 0.5 else "export does not reproduce h5",
                     sha_export=ex0["rknn_sha256"], sha_build_recipe=hashlib.sha256(next((s0 / "weights" / "rknn_int8_h5").glob("*.rknn")).read_bytes()).hexdigest())
    sha_now = hashlib.sha256(next((s0 / "weights" / "rknn_int8").glob("*.rknn")).read_bytes()).hexdigest()
    out["v2"] = dict(sha_default_int8=sha_now, unchanged=sha_now == S0_INT8_SHA)
    seeds = [out["runs"][f"silu_s{n}"]["sets"]["css_test"] for n in (0, 1, 2)]
    out["v3_range"] = {tag: {c: max(s[tag][c] for s in seeds) - min(s[tag][c] for s in seeds) for c in COLS} for tag in ("int8h5", "int8", "fp16")}
    out["v3_mean"] = {tag: {c: statistics.mean(s[tag][c] for s in seeds) for c in COLS} for tag in ("int8h5", "int8", "fp16")}
    out["rule"] = (HOME / "rule_int8h5.txt").read_text()
    (OPERATOR_SUPPORT_RESULTS_DIR / "int8h5.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# int8h5 standard export", "", f"V-1 {out['v1']}", f"V-2 {out['v2']}", ""]
    for k, v in out["runs"].items():
        c = v["sets"]["css_test"]
        md.append(f"- {k}: int8h5 {c['int8h5']['map50_95']:.2f}, int8 {c['int8']['map50_95']:.2f}, fp16 {c['fp16']['map50_95']:.2f}")
    md.append(f"- range css_test AP50-95: {out['v3_range']}")
    (OPERATOR_SUPPORT_RESULTS_DIR / "int8h5.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
