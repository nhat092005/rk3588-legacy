"""CLI: X7, accuracy of the INT8 model (RKNNLite on the board) against the FP32 PyTorch model of the same run, statements of
bench/eval_x7/rule.txt (numbers only, no threshold).

    .venv/bin/python -m ai.automation.analyze_x7 <run_id>
Writes results/eval_x7_<run_id>_int8.{json,md}.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    ev = REPO / "ai" / "automation" / "runs" / args.run_id / "eval"
    out = {"sets": {}}
    for s in ("css_test", "ppe2286_ext"):
        q, f = json.loads((ev / f"rknn_int8_{s}.json").read_text()), json.loads((ev / f"pt_fp32_{s}.json").read_text())
        out["sets"][s] = dict(n_images=q["n_images"], int8={k: q[k] for k in ("precision", "recall", "map50", "map50_95")}, fp32={k: f[k] for k in ("precision", "recall", "map50", "map50_95")},
                              diff_pp={k: 100 * (q[k] - f[k]) for k in ("precision", "recall", "map50", "map50_95")}, per_class_int8=q["per_class"], per_class_fp32=f["per_class"],
                              int8_backend=q.get("backend"), int8_model_sha256=q.get("model_sha256"))
    out["rule"] = (REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "eval_x7" / "rule.txt").read_text()
    # X7b (bench/eval_x7b/rule.txt): the same RKNNLite path with the FP16 model, to split quantization from the inference path
    x7b = {}
    for st in ("css_test", "ppe2286_ext"):
        h = json.loads((ev / f"rknn_fp16_{st}.json").read_text())
        f = json.loads((ev / f"pt_fp32_{st}.json").read_text())
        x7b[st] = dict(fp16_rknn_map50_95=h["map50_95"], fp32_map50_95=f["map50_95"], d16_pp=100 * (h["map50_95"] - f["map50_95"]), d8_pp=out["sets"][st]["diff_pp"]["map50_95"],
                       model_sha256=h["model_sha256"], backend=h.get("backend"))
    d8, d16 = x7b["css_test"]["d8_pp"], x7b["css_test"]["d16_pp"]
    x7b["verdict"] = ("mức giảm của INT8 chủ yếu đi theo lượng tử hóa" if abs(d16) <= 0.25 * abs(d8) else
                      ("mức giảm chủ yếu đi theo đường suy luận (RKNNLite trên board so với PyTorch trên PC)" if abs(d16) >= 0.75 * abs(d8) else "chưa kết luận"))
    x7b["rule"] = (REPO / "ai" / "automation" / "runs" / args.run_id / "bench" / "eval_x7b" / "rule.txt").read_text()
    out["x7b"] = x7b
    name = f"eval_x7_{args.run_id}_int8"
    (REPO / "results" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    md = [f"# X7: INT8 against FP32 accuracy, {args.run_id}", "", "| set | images | metric | INT8 | FP32 | difference pp |", "|---|---|---|---|---|---|"]
    for s, v in out["sets"].items():
        for k in ("precision", "recall", "map50", "map50_95"):
            md.append(f"| {s} | {v['n_images']} | {k} | {v['int8'][k]:.4f} | {v['fp32'][k]:.4f} | {v['diff_pp'][k]:+.2f} |")
    md += ["", "## X7b: FP16 on the same RKNNLite path", "", "| set | FP32 mAP50-95 | FP16 RKNNLite | D16 pp | INT8 minus FP32 (D8) pp |", "|---|---|---|---|---|"]
    md += [f"| {st} | {v['fp32_map50_95']:.4f} | {v['fp16_rknn_map50_95']:.4f} | {v['d16_pp']:+.2f} | {v['d8_pp']:+.2f} |" for st, v in x7b.items() if st != "verdict" and st != "rule"]
    md += ["", f"Verdict on css_test: {x7b['verdict']}"]
    (REPO / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
