"""CLI: compare training sets on common test sets. Run: make compare-train-sets

Reads runs/<run_id>/eval/pt_fp32_<eval_set>.json for every training dataset x seed x CROSS_EVAL_SETS and writes
results/train_set_comparison.md. The "PPE4" columns average the four classes every training set shares
(Helmet, NoHelmet, NoVest, Vest), so a model without a Person output is not penalised for the Person ground truth
of the test set. A missing file is listed and exits 1; create it with
`python -m ai.automation.evaluate <run_id> --backend pt --eval-set <eval_set>`.
"""
from __future__ import annotations

import json
import statistics
import sys

from ai.core.evaluator import CROSS_EVAL_SETS, PROTOCOL_VERSION
from ai.paths import REPO_ROOT, RESULTS_DIR, RUNS_DIR

OUT = RESULTS_DIR / "train_set_comparison.md"

TRAIN_SETS = ["ppe5", "ppe5_plus", "ppe4", "ppe2286"]
MODEL = "yolov8n_baseline_100ep"
SEEDS = [0, 1, 2]
PPE4 = ["Helmet", "NoHelmet", "NoVest", "Vest"]
PER_CLASS = PPE4 + ["Person"]


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f} (1 seed)"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main() -> None:
    results: dict[tuple[str, str], list[dict]] = {}
    missing, problems = [], []
    fingerprints: dict[str, set[str]] = {e: set() for e in CROSS_EVAL_SETS}
    for train_set in TRAIN_SETS:
        for eval_set in CROSS_EVAL_SETS:
            for seed in SEEDS:
                run_id = f"{train_set}_{MODEL}_s{seed}"
                path = RUNS_DIR / run_id / "eval" / f"pt_fp32_{eval_set}.json"
                if not path.exists():
                    missing.append(f"{run_id} {eval_set}")
                    continue
                data = json.loads(path.read_text())
                if data["protocol_version"] != PROTOCOL_VERSION:
                    problems.append(f"{path.relative_to(REPO_ROOT)}: protocol {data['protocol_version']}")
                fingerprints[eval_set].add(data["dataset_fingerprint"][:12])
                results.setdefault((train_set, eval_set), []).append(data)
    for eval_set, fps in fingerprints.items():
        if len(fps) > 1:
            problems.append(f"{eval_set}: more than one dataset fingerprint {sorted(fps)}")
    if missing or problems:
        print("\n".join([f"missing: {m}" for m in missing] + [f"problem: {p}" for p in problems]))
        sys.exit(1)

    lines = [
        "# Training-set comparison (YOLOv8n, Ultralytics defaults, seeds 0 to 2)",
        "",
        "Mean ± sample std over seeds. AP50-95 per class. PPE4 = mean of Helmet, NoHelmet, NoVest, Vest. "
        "`-` = the model has no such class, or the test set has no ground truth for it.",
        "",
    ]
    for eval_set in CROSS_EVAL_SETS:
        n_images = results[(TRAIN_SETS[0], eval_set)][0]["n_images"]
        lines += [
            f"## Test set `{eval_set}` ({n_images} images, fingerprint {next(iter(fingerprints[eval_set]))})",
            "",
            "| Train set | " + " | ".join(PER_CLASS) + " | PPE4 AP50-95 | PPE4 AP50 |",
            "|---|" + "---|" * (len(PER_CLASS) + 2),
        ]
        for train_set in TRAIN_SETS:
            trained = (REPO_ROOT / "data" / train_set / "classes.txt").read_text().split()
            runs = results[(train_set, eval_set)]
            cells = []
            for c in PER_CLASS:
                values = [r["per_class"][c]["ap50_95"] for r in runs if c in r["per_class"]]
                cells.append(mean_std(values) if c in trained and values else "-")
            ap = [statistics.mean(r["per_class"][c]["ap50_95"] for c in PPE4) for r in runs]
            ap50 = [statistics.mean(r["per_class"][c]["ap50"] for c in PPE4) for r in runs]
            lines.append(f"| {train_set} | " + " | ".join(cells) + f" | {mean_std(ap)} | {mean_std(ap50)} |")
        lines.append("")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
