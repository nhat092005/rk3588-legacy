"""Variant driver (PC): SiLU, ReLU, LeakyReLU, ReLU6 and ReLU 200 epoch INT8 sessions interleaved under P2 (helper x4_hold.sh in pin mode, started separately as root in
/home/minhnhat/x4), 45 s rest, then it writes the stop file of the helper. Rule: rule.txt next to this file.

Run from the repo root, detached:  .venv/bin/python <this file>
One session = benchmark_npu.py (5 eval_perf without perf_debug, then 5 eval_perf with perf_debug=True), --skip-memory.
Order, the same in both rounds: for mask in CORE_0, CORE_0_1_2: silu relu leakyrelu relu6 relu200 (all seed 0).
Writes sessions.json (one entry per session). Marker: driver_done contains DONE or ABORT <reason>.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/nhat/workspace/prjs/rk3588")

HERE = Path(__file__).resolve().parent
RUNS = {"silu": "ppe5_yolov8n_baseline_100ep_s0", "relu": "ppe5_yolov8n_relu_100ep_s0", "leakyrelu": "ppe5_yolov8n_leakyrelu_100ep_s0",
        "relu6": "ppe5_yolov8n_relu6_100ep_s0", "relu200": "ppe5_yolov8n_relu_200ep_s0"}
MASKS = ["CORE_0", "CORE_0_1_2"]
HOST = "minhnhat@192.168.137.49"
DEVICE = "192.168.137.49:5555"
REST_S = 45
H = "/home/minhnhat/x4"
SUB = "onboard_g4v"
LOG = []


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True, timeout=60).stdout


def log(msg):
    with open(HERE / "driver.log", "a") as fh:
        fh.write(f"{time.strftime('%T')} {msg}\n")


def set_mode(mode):
    ssh(f"echo {mode} > {H}/mode")
    for _ in range(60):
        if ssh(f"cat {H}/mode_ack").strip() == mode:
            return
        time.sleep(1)
    raise RuntimeError(f"helper did not acknowledge mode {mode}")


def irq_lines():
    out = {}
    for ln in ssh("grep -E '^ *3[678]:' /proc/interrupts").splitlines():
        p = ln.split()
        out[p[0].rstrip(":")] = [int(x) for x in p[1:9]]
    return out


def sha(model):
    rknn = next((Path("experiments/runs") / RUNS[model] / "weights" / "rknn_int8").glob("*.rknn"))
    return subprocess.run(["sha256sum", str(rknn)], capture_output=True, text=True).stdout.split()[0]


def session(model, mask, n, shas):
    entry = dict(protocol="p2", model=model, run=RUNS[model], mask=mask, session=n, subdir=SUB, sha256=sha(model))
    if entry["sha256"] != shas[model]:
        raise RuntimeError(f"{model} model file changed during the run")
    entry["epoch_start"] = int(ssh("date +%s").strip())
    entry["irq_before"] = irq_lines()
    r = subprocess.run([".venv/bin/python", "-m", "ai.automation.benchmark_npu", RUNS[model], DEVICE, "--precision", "int8", "--core-mask", mask,
                        "--session", str(n), "--skip-memory", "--out-subdir", SUB], capture_output=True, text=True, timeout=900)
    entry["irq_after"] = irq_lines()
    entry["epoch_end"] = int(ssh("date +%s").strip())
    LOG.append(entry)
    (HERE / "sessions.json").write_text(json.dumps(LOG, indent=1) + "\n")
    if r.returncode != 0:
        raise RuntimeError(f"{model} {mask} session {n} failed: {r.stderr[-400:]}")
    log(f"{model} {mask} session {n}")


def main():
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        if ssh(f"test -f {H}/ready && echo yes").strip() != "yes":
            raise RuntimeError("helper not ready on the board")
        shas = {m: sha(m) for m in RUNS}
        set_mode("pin")
        for n in (1, 2):
            for mask in MASKS:
                for model in RUNS:
                    session(model, mask, n, shas)
                    time.sleep(REST_S)
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise
    finally:
        ssh(f"touch {H}/stop")


if __name__ == "__main__":
    main()
