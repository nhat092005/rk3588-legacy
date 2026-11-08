"""CLI: G4 C (ReLU against SiLU accuracy) and G4N (seed spread of the SiLU baseline), statements of bench/eval_g4/rule.txt.

    .venv/bin/python -m ai.automation.analyze_g4c
Reads runs/<run>/eval/{pt_fp32,rknn_int8,rknn_fp16}_<set>.json. Writes results/eval_g4.{json,md}.
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "ai" / "automation" / "runs"
SILU = [f"ppe5_yolov8n_baseline_100ep_s{s}" for s in (0, 1, 2)]
RELU = "ppe5_yolov8n_relu_100ep_s0"
VARIANTS = {"leakyrelu": "ppe5_yolov8n_leakyrelu_100ep_s0", "relu6": "ppe5_yolov8n_relu6_100ep_s0", "relu200": "ppe5_yolov8n_relu_200ep_s0"}
SILU200 = "ppe5_yolov8n_baseline_200ep_s0"
RELU_SEEDS = [f"ppe5_yolov8n_relu_100ep_s{s}" for s in (0, 1, 2)]
SETS = ("css_test", "ppe2286_ext")
TAGS = {"int8": "rknn_int8", "fp16": "rknn_fp16", "fp32": "pt_fp32"}
COLS = ("map50_95", "NoHelmet", "Vest")


def load(run, tag, st):
    d = json.loads((RUNS / run / "eval" / f"{TAGS[tag]}_{st}.json").read_text())
    return dict(map50_95=100 * d["map50_95"], NoHelmet=100 * d["per_class"]["NoHelmet"]["ap50_95"], Vest=100 * d["per_class"]["Vest"]["ap50_95"],
                n_images=d["n_images"], instances_nohelmet=d["per_class"]["NoHelmet"]["instances"], sha=d["model_sha256"])


def main() -> None:
    out = {"noise": {}, "gate_c": {}}
    for st in SETS:
        for tag in TAGS:
            vals = [load(r, tag, st) for r in SILU]
            out["noise"][f"{st}|{tag}"] = {c: dict(values=[v[c] for v in vals], min=min(v[c] for v in vals), max=max(v[c] for v in vals),
                                                   range=max(v[c] for v in vals) - min(v[c] for v in vals), std=statistics.stdev([v[c] for v in vals]),
                                                   verdict="trong nhiễu seed" if max(v[c] for v in vals) - min(v[c] for v in vals) >= 1.0 else "ngoài nhiễu seed")
                                           for c in COLS}
            out["noise"][f"{st}|{tag}"]["n_images"] = vals[0]["n_images"]
    if (RUNS / RELU / "eval" / "rknn_int8_css_test.json").exists():
        for st in SETS:
            for tag in TAGS:
                s, r = load(SILU[0], tag, st), load(RELU, tag, st)
                diff = {c: r[c] - s[c] for c in COLS}
                nz = out["noise"][f"{st}|{tag}"]
                out["gate_c"][f"{st}|{tag}"] = dict(silu=s, relu=r, diff_pp=diff, inside_noise={c: nz[c]["range"] >= abs(diff[c]) for c in COLS},
                                                    passes=all(d >= -1.0 for d in diff.values()))
    if all((RUNS / r / "eval" / "rknn_int8_css_test.json").exists() for r in RELU_SEEDS) and all((RUNS / r / "eval" / "rknn_fp16_ppe2286_ext.json").exists() for r in RELU_SEEDS):
        # bench/eval_g4/rule_relu_seeds.txt (R-1 to R-4)
        out["relu_noise"] = {}
        for st in SETS:
            for tag in TAGS:
                rv = [load(r, tag, st) for r in RELU_SEEDS]
                sv = [load(r, tag, st) for r in SILU]
                row = {}
                for c in COLS:
                    rr, ss = [v[c] for v in rv], [v[c] for v in sv]
                    rng_r, rng_s = max(rr) - min(rr), max(ss) - min(ss)
                    diff = statistics.mean(rr) - statistics.mean(ss)
                    row[c] = dict(values=rr, min=min(rr), max=max(rr), range=rng_r, std=statistics.stdev(rr), silu_range=rng_s, mean_diff=diff,
                                  verdict="lớn hơn nhiễu seed" if abs(diff) > max(rng_r, rng_s) else "trong nhiễu seed",
                                  seed0_diff_inside=abs(rr[0] - ss[0]) <= max(rng_r, rng_s))
                out["relu_noise"][f"{st}|{tag}"] = row
        per = []
        for n in (0, 1, 2):
            s_, r_ = load(SILU[n], "int8", "css_test"), load(RELU_SEEDS[n], "int8", "css_test")
            d = {c: r_[c] - s_[c] for c in COLS}
            per.append(dict(seed=n, diff_pp=d, passes=all(x >= -1.0 for x in d.values())))
        out["relu_seed_gate_c"] = dict(per_seed=per, passing=sum(p_["passes"] for p_ in per))
    present = {k: r for k, r in VARIANTS.items() if (RUNS / r / "eval" / "rknn_fp16_ppe2286_ext.json").exists()}
    if present:
        # bench/eval_g4/rule_variants.txt (V-C, V-E, V-S)
        out["variants"] = {}
        have_ctrl = (RUNS / SILU200 / "eval" / "rknn_fp16_ppe2286_ext.json").exists()
        for k, r in present.items():
            out["variants"][k] = {}
            for st in SETS:
                for tag in TAGS:
                    v, s_, rl = load(r, tag, st), load(SILU[0], tag, st), load(RELU, tag, st)
                    d = {c: v[c] - s_[c] for c in COLS}
                    row = dict(values={c: v[c] for c in COLS}, diff_vs_silu_pp=d, passes=all(x >= -1.0 for x in d.values()), diff_vs_relu_seed0_pp={c: v[c] - rl[c] for c in COLS})
                    if k == "relu200" and have_ctrl:
                        c200 = load(SILU200, tag, st)
                        row["silu200"] = {c: c200[c] for c in COLS}
                        row["diff_vs_silu200_pp"] = {c: v[c] - c200[c] for c in COLS}
                    out["variants"][k][f"{st}|{tag}"] = row
        if have_ctrl:
            out["silu200_vs_silu100"] = {f"{st}|{tag}": {c: load(SILU200, tag, st)[c] - load(SILU[0], tag, st)[c] for c in COLS} for st in SETS for tag in TAGS}
        out["rule_variants"] = (RUNS / SILU[0] / "bench" / "eval_g4" / "rule_variants.txt").read_text()
    out["verdict_c"] = None
    if out["gate_c"]:
        v = out["gate_c"]["css_test|int8"]
        out["verdict_c"] = "đạt" if v["passes"] else "không đạt"
    out["rule"] = (RUNS / SILU[0] / "bench" / "eval_g4" / "rule.txt").read_text()
    out["rule_relu_seeds"] = (RUNS / SILU[0] / "bench" / "eval_g4" / "rule_relu_seeds.txt").read_text()
    (REPO / "results" / "eval_g4.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# G4 C and seed spread", ""]
    for k, v in out["noise"].items():
        md.append(f"- {k}: " + ", ".join(f"{c} range {v[c]['range']:.2f} pp ({v[c]['verdict']})" for c in COLS))
    for k, v in out["gate_c"].items():
        md.append(f"- C {k}: diff " + ", ".join(f"{c} {v['diff_pp'][c]:+.2f}" for c in COLS) + f", passes {v['passes']}")
    for k, v in out.get("relu_noise", {}).items():
        md.append(f"- ReLU spread {k}: " + ", ".join(f"{c} range {v[c]['range']:.2f}, mean diff {v[c]['mean_diff']:+.2f} ({v[c]['verdict']})" for c in COLS))
    for k, v in out.get("variants", {}).items():
        md.append(f"- variant {k} css_test INT8 vs SiLU: " + ", ".join(f"{c} {v['css_test|int8']['diff_vs_silu_pp'][c]:+.2f}" for c in COLS) + f", passes {v['css_test|int8']['passes']}")
    if "relu_seed_gate_c" in out:
        md.append(f"- ReLU seeds passing C against SiLU same seed: {out['relu_seed_gate_c']['passing']}/3")
    (REPO / "results" / "eval_g4.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
