"""Diagnostic E1: two independent init_runtime sessions (perf_debug=False), 20 eval_perf calls each, per-call value, wall time and printed frequencies."""
import json, re, sys, time
from ai.core.rknn.benchmark import connect_board, parse_frequencies, parse_summary_time_us
path = "ai/automation/runs/ppe5_yolov8n_baseline_100ep_s0/weights/rknn_int8/best-rk3588.rknn"
out = {"sessions": []}
for s in range(2):
    sess = {"t_start": time.time(), "calls": []}
    with connect_board(path, "100.67.251.37:5555", core_mask="CORE_0") as rknn:
        sess["t_init_done"] = time.time()
        for i in range(20):
            t = time.time()
            text = str(rknn.eval_perf(is_print=False, fix_freq=True))
            sess["calls"].append({"t": t, "wall_s": round(time.time() - t, 2), "us": parse_summary_time_us(text), "freq": parse_frequencies(text)})
            print(f"s{s} call {i}: {sess['calls'][-1]['us']} us wall {sess['calls'][-1]['wall_s']} s", flush=True)
    out["sessions"].append(sess)
    json.dump(out, open(sys.argv[1], "w"))
print("E1_DONE", flush=True)
