"""X4 driver (PC): P1-LAN and P2 sessions interleaved with a 45 s rest, then mask and log comparisons in P2 mode, then it writes
the stop file of the board helper (x4_hold.sh, started separately as root). Rule: rule.txt next to this file.

Run from the repo root, detached:  .venv/bin/python <this file>
Writes sessions.json (one entry per session: protocol, kind, board epoch start and end, interrupt lines before and after).
Marker: driver_done contains DONE or ABORT <reason>.
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/nhat/workspace/prjs/rk3588")
from rknn.api import RKNN  # noqa: E402

from ai.core.rknn.benchmark import CORE_MASK, connect_board, measure_latency  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "ppe5_yolov8n_baseline_100ep_s0"
MODEL = f"experiments/runs/{RUN}/weights/rknn_int8/best-rk3588.rknn"
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


def npu_session(proto, n):
    sub = f"onboard_x4/{proto}" if n <= 3 else f"onboard_x4/{proto}_extra"

    def run():
        r = subprocess.run([".venv/bin/python", "-m", "ai.automation.benchmark_npu", RUN, DEVICE, "--precision", "int8", "--core-mask", "CORE_0",
                            "--session", str(n), "--skip-memory", "--out-subdir", sub], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"{proto} session {n} failed: {r.stderr[-400:]}")
    timed(dict(protocol=proto, kind="benchmark_npu", session=n, subdir=sub), run)
    log(f"{proto} session {n} -> {sub}")


def latency_session(core_mask="CORE_0", verbose=False):
    if verbose:
        rknn = RKNN(verbose=True)
        if rknn.load_rknn(MODEL) != 0 or rknn.init_runtime(target="rk3588", device_id=DEVICE, core_mask=CORE_MASK[core_mask]) != 0:
            raise RuntimeError("verbose session init failed")
        try:
            return measure_latency(rknn)
        finally:
            rknn.release()
    with connect_board(MODEL, DEVICE, core_mask=core_mask) as rknn:
        return measure_latency(rknn)


def main():
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        if ssh(f"test -f {H}/ready && echo yes").strip() != "yes":
            raise RuntimeError("helper not ready on the board")
        for n in range(1, 11):
            for proto, mode in (("p1lan", "unpin"), ("p2", "pin")):
                set_mode(mode)
                npu_session(proto, n)
                time.sleep(REST_S)
        set_mode("pin")
        comp = {"masks": [], "log": []}
        for rnd in range(1, 5):
            for mask in ("CORE_0", "AUTO", "CORE_0_1_2"):
                e = dict(protocol="p2", kind="mask", mask=mask, round=rnd)
                lat = timed(e, lambda: latency_session(core_mask=mask))
                comp["masks"].append(dict(round=rnd, mask=mask, latency_ms_all=lat["npu_latency_ms_all"], median_ms=statistics.median(lat["npu_latency_ms_all"]),
                                          session_index=len(LOG) - 1))
                log(f"mask {mask} round {rnd}")
                time.sleep(REST_S)
        for rnd in range(1, 4):
            for verbose in (False, True):
                e = dict(protocol="p2", kind="log", verbose=verbose, round=rnd)
                lat = timed(e, lambda: latency_session(verbose=verbose))
                comp["log"].append(dict(round=rnd, verbose=verbose, latency_ms_all=lat["npu_latency_ms_all"], median_ms=statistics.median(lat["npu_latency_ms_all"]),
                                        session_index=len(LOG) - 1))
                log(f"verbose {verbose} round {rnd}")
                time.sleep(REST_S)
        (HERE / "comparisons.json").write_text(json.dumps(comp, indent=1) + "\n")
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise
    finally:
        ssh(f"touch {H}/stop")


if __name__ == "__main__":
    main()
