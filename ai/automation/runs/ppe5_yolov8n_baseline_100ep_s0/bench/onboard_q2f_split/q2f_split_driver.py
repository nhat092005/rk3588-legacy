"""E2 driver (PC, copy of the E1 driver): split-head model, decode overlapped with the NPU, under protocol P3 (helper p3_hold.sh started separately as root in
/home/minhnhat/p3). Rule: rule.txt next to this file.

Run from the repo root, detached:  .venv/bin/python <this file>
Each run is one ssh call of ai/board/bench_throughput_rep.py under taskset -c 4-7, 3 repetitions of the same 7 configurations in the
same order, 20 s rest. S3 worker counts come from the formula of rule.txt applied to the medians of the first T5 run.
Writes sessions.json. Marker: driver_done contains DONE or ABORT <reason>.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "ppe5_yolov8n_baseline_100ep_s0"
HOST = "minhnhat@192.168.137.49"
H = "/home/minhnhat/p3"
REST_S = 20
FRAMES = 1238
CONFIGS = [  # code, mode, masks, model dir, pre workers, post workers
    ("A", "threads", "CORE_0", "int8h5", 1, 1),
    ("B", "threads", "CORE_0", "int8_split", 1, 1),
    ("C", "pipeline", "CORE_0", "int8_split", 1, 1),
    ("D", "pipeline", "CORE_0", "int8_split", 1, 2),
    ("E", "pipeline", "CORE_0_1_2", "int8_split", 1, 1),
    ("F", "pipeline", "CORE_0", "int8h5", 1, 1),
]
LOG = []
S3 = {}


def ssh(cmd, timeout=900):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True, timeout=timeout)


def log(msg):
    with open(HERE / "driver.log", "a") as fh:
        fh.write(f"{time.strftime('%T')} {msg}\n")


def provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return f"--run-id {RUN} --git-commit {commit} --git-dirty {str(dirty).lower()}"


def run_one(rep, code, mode, masks, model, npre, npost):
    name = f"r{rep}_{code}.json"
    remote = f"ai/automation/runs/{RUN}/bench/onboard_q2f_split/{name}"
    extra = f"--pre-workers {npre} --post-workers {npost}" if mode == "pipeline" else ""
    cmd = (f"cd rk3588 && mkdir -p $(dirname {remote}) && taskset -c 4-7 .venv-board/bin/python -m ai.board.bench_throughput_rep "
           f"--model ai/automation/runs/{RUN}/weights/rknn_{model} --images data/ppe5/processed/images/test --mode {mode} --core-masks {masks} "
           f"--frames {FRAMES} {extra} --out {remote} {provenance()}")
    entry = dict(rep=rep, code=code, mode=mode, masks=masks, model=model, pre_workers=npre, post_workers=npost, file=name, epoch_start=int(ssh("date +%s").stdout.strip()))
    r = ssh(cmd)
    entry["epoch_end"] = int(ssh("date +%s").stdout.strip())
    entry["returncode"] = r.returncode
    LOG.append(entry)
    (HERE / "sessions.json").write_text(json.dumps(LOG, indent=1) + "\n")
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed: {r.stderr[-400:]} {r.stdout[-400:]}")
    subprocess.run(["scp", "-q", f"{HOST}:rk3588/{remote}", str(HERE / name)], check=True)
    log(f"{name} {model} {mode} {masks}: {json.loads((HERE / name).read_text())['throughput_fps']} FPS")


def main():
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        if ssh(f"test -f {H}/ready && echo yes").stdout.strip() != "yes":
            raise RuntimeError("helper not ready on the board")
        for rep in (1, 2, 3):
            for code, mode, masks, model, npre, npost in CONFIGS:
                run_one(rep, code, mode, masks, model, npre, npost)
                time.sleep(REST_S)
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise
    finally:
        ssh(f"touch {H}/stop")


if __name__ == "__main__":
    main()
