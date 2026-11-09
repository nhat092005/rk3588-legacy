"""Camera-frame throughput driver (PC). Rule: rule.txt next to this file. Run from the repo root, detached:  .venv/bin/python <this file>
Locks clocks, starts p3_hold.sh (passwordless sudo helpers), 3 repetitions of C0,C1,C2 x (T1,T5), 20 s rest. Marker: driver_done."""
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "ppe5_yolov8n_baseline_100ep_s0"
HOST = "minhnhat@192.168.137.49"
H = "/home/minhnhat/p3"
REST_S, FRAMES = 20, 1238
IMAGES = {"C0": "data/ppe5/processed/images/test", "C1": "/home/minhnhat/camframes", "C2": "/home/minhnhat/camframes"}
TS = {"T1": "CORE_0", "T5": "CORE_0,CORE_1,CORE_2"}
LOG = []


def ssh(cmd, timeout=900):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True, timeout=timeout)


def log(msg):
    with open(HERE / "driver.log", "a") as fh:
        fh.write(f"{time.strftime('%T')} {msg}\n")


def provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return f"--run-id {RUN} --git-commit {commit} --git-dirty {str(dirty).lower()}"


def isp():
    return ssh("grep -E 'frame:' /proc/rkisp0-vir1").stdout.strip()


def run_one(rep, cond, t):
    name = f"r{rep}_{cond}_{t}.json"
    remote = f"tmp_camfps/{name}"
    entry = dict(rep=rep, cond=cond, t=t, file=name)
    if cond == "C2":
        r = ssh("~/camload_scripts/bg_start.sh")
        if "started" not in r.stdout:
            raise RuntimeError(f"bg_start failed: {r.stdout} {r.stderr}")
        time.sleep(3)
        entry["isp_before"] = isp()
    cmd = (f"cd rk3588 && mkdir -p ../tmp_camfps && taskset -c 4-7 .venv-board/bin/python -m ai.board.bench_throughput_rep "
           f"--model experiments/runs/{RUN}/weights/rknn_int8h5 --images {IMAGES[cond]} --mode threads --core-masks {TS[t]} "
           f"--frames {FRAMES} --out ../{remote} {provenance()}")
    entry["epoch_start"] = int(ssh("date +%s").stdout.strip())
    r = ssh(cmd)
    entry["epoch_end"] = int(ssh("date +%s").stdout.strip())
    entry["returncode"] = r.returncode
    if cond == "C2":
        entry["isp_after"] = isp()
        ssh("~/camload_scripts/bg_stop.sh")
    LOG.append(entry)
    (HERE / "sessions.json").write_text(json.dumps(LOG, indent=1) + "\n")
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed: {r.stderr[-400:]} {r.stdout[-400:]}")
    subprocess.run(["scp", "-q", f"{HOST}:{remote}", str(HERE / name)], check=True)
    log(f"{name}: {json.loads((HERE / name).read_text())['throughput_fps']} FPS")


def main():
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        ssh("rm -rf ~/p3 && mkdir -p ~/p3")
        if ssh("sudo -n /usr/local/sbin/lock_freq.sh lock").returncode != 0:
            raise RuntimeError("lock_freq failed")
        ssh("setsid nohup sudo -n /usr/local/sbin/p3_hold.sh 7200 >/dev/null 2>&1 </dev/null &")
        for _ in range(30):
            if ssh(f"test -f {H}/ready && echo yes").stdout.strip() == "yes":
                break
            time.sleep(1)
        else:
            raise RuntimeError("helper not ready on the board")
        for rep in (1, 2, 3):
            for t in ("T1", "T5"):
                for cond in ("C0", "C1", "C2"):
                    run_one(rep, cond, t)
                    time.sleep(REST_S)
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise
    finally:
        ssh("~/camload_scripts/bg_stop.sh")
        ssh(f"touch {H}/stop")
        time.sleep(3)
        ssh("sudo -n /usr/local/sbin/lock_freq.sh unlock")


if __name__ == "__main__":
    main()
