"""X8 driver (PC): INT8 and FP16 sessions interleaved A B A B A B under P2 (helper x4_hold.sh in pin mode, started separately as
root in /home/minhnhat/x4), 45 s rest, then it writes the stop file of the helper. Rule: rule.txt next to this file.

Run from the repo root, detached:  .venv/bin/python <this file>
Writes sessions.json (one entry per session: protocol, kind, board epoch start and end, interrupt lines before and after).
Marker: driver_done contains DONE or ABORT <reason>.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/nhat/workspace/prjs/rk3588")

HERE = Path(__file__).resolve().parent
RUN = "ppe5_yolov8n_baseline_100ep_s0"
HOST = "minhnhat@192.168.137.49"
DEVICE = "192.168.137.49:5555"
REST_S = 45
H = "/home/minhnhat/x4"
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


def timed(entry, fn):
    entry["epoch_start"] = int(ssh("date +%s").strip())
    entry["irq_before"] = irq_lines()
    result = fn()
    entry["irq_after"] = irq_lines()
    entry["epoch_end"] = int(ssh("date +%s").strip())
    LOG.append(entry)
    (HERE / "sessions.json").write_text(json.dumps(LOG, indent=1) + "\n")
    return result


def npu_session(prec, n):
    sub = f"onboard_x8/{prec}"

    def run():
        r = subprocess.run([".venv/bin/python", "-m", "ai.automation.benchmark_npu", RUN, DEVICE, "--precision", prec, "--core-mask", "CORE_0",
                            "--session", str(n), "--skip-memory", "--out-subdir", sub], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"{prec} session {n} failed: {r.stderr[-400:]}")
    timed(dict(protocol="p2", kind="benchmark_npu", precision=prec, session=n, subdir=sub), run)
    log(f"{prec} session {n} -> {sub}")


def main():
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        if ssh(f"test -f {H}/ready && echo yes").strip() != "yes":
            raise RuntimeError("helper not ready on the board")
        set_mode("pin")
        for n in range(1, 4):
            for prec in ("int8", "fp16"):
                npu_session(prec, n)
                time.sleep(REST_S)
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise
    finally:
        ssh(f"touch {H}/stop")


if __name__ == "__main__":
    main()
