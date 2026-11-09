"""CLI: M2, what core_mask NPU_CORE_ALL does (experiments/npu/operator_support/onboard_q2e/rule_core_all.txt).

    .venv/bin/python -m ai.npu.operator_support.probe_core_all 192.168.137.49:5555
Writes results/npu/operator_support/q2e_core_all.json. The board must be locked and rknn_server running.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys

from rknn.api import RKNN

from ai.core.rknn.benchmark import connect_board, measure_layers, parse_summary_time_us
from ai.paths import REPO_ROOT as REPO, OPERATOR_SUPPORT_RESULTS_DIR, RUNS_DIR, experiment_dir

RUN = RUNS_DIR / "ppe5_yolov8n_baseline_100ep_s0"
MASKS = {"AUTO": RKNN.NPU_CORE_AUTO, "CORE_0_1_2": RKNN.NPU_CORE_0_1_2, "ALL": RKNN.NPU_CORE_ALL}


def split_names(layers) -> list[str]:
    out = []
    for r in layers:
        if r["workload"]:
            if sum(float(p.rstrip("%")) > 0 for p in r["workload"].split("/")) >= 2:
                out.append(r["name"])
    return out


def main() -> None:
    from ai.core.rknn import benchmark as bm

    device = sys.argv[1]
    model = RUN / "weights" / "rknn_int8" / "best-rk3588.rknn"
    out = dict(rule=(experiment_dir(RUN, "onboard_q2e") / "rule_core_all.txt").read_text(), model=str(model.relative_to(REPO)), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), masks={})
    for name, mask in MASKS.items():
        rec = {}
        try:
            bm.CORE_MASK[name] = mask  # connect_board looks the mask up by name
            with connect_board(str(model), device, name, perf_debug=True) as rknn:
                layers = measure_layers(rknn, repeats=1)[0]["layers"]
            names = split_names(layers)
            rec.update(split_count=len(names), split_names=names, rows=len(layers), cpu_rows=sum(1 for r in layers if r["target"] == "CPU"))
            with connect_board(str(model), device, name) as rknn:
                us = [parse_summary_time_us(str(rknn.eval_perf(is_print=False, fix_freq=True))) for _ in range(5)]
            rec["latency_ms_median"] = statistics.median(us) / 1000
            rec["latency_ms_all"] = [u / 1000 for u in us]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out["masks"][name] = rec
    ref = set(out["masks"].get("CORE_0_1_2", {}).get("split_names", []))
    for name in ("AUTO", "ALL"):
        out["masks"][name]["same_split_set_as_core_0_1_2"] = set(out["masks"][name].get("split_names", [])) == ref if "error" not in out["masks"][name] else None
    (OPERATOR_SUPPORT_RESULTS_DIR / "q2e_core_all.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "split_names"} for k, v in out["masks"].items()}, indent=1))


if __name__ == "__main__":
    main()
