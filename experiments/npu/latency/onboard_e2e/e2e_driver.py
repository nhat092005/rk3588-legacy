"""E2e driver (PC): 14 blocks of eval_perf calls while the board helper pins rknn_server and the NPU interrupts.

Run from the repo root, detached:  .venv/bin/python <this file>
Writes b<N>_<cond>.csv (run, total_time_us, wall_s, timed) here and copies the helper's files from the board at the end.
Marker: driver_done contains DONE or ABORT <reason>. The helper must already be running on the board (daemon_ready exists).
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/nhat/workspace/prjs/rk3588")
from rknn.api import RKNN  # noqa: E402

from ai.core.rknn.benchmark import CORE_MASK, parse_summary_time_us  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/rknn_int8/best-rk3588.rknn"
HOST = "minhnhat@192.168.137.49"
DEVICE = "192.168.137.49:5555"
WARMUP, CALLS, REST_S = 50, 500, 20
HALF = ["A", "s4i4", "s0i0", "s4i0", "s0i4", "s1i0", "s1i4"]
ORDER = HALF + HALF[::-1]


def sh(cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True, timeout=60).stdout


def wait_board(name: str, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if sh(f"test -f /home/minhnhat/e2e/{name} && echo yes").strip() == "yes":
            return
        time.sleep(0.3)
    raise TimeoutError(f"board file {name} not seen in {timeout_s} s")


def main() -> None:
    marker = HERE / "driver_done"
    marker.unlink(missing_ok=True)
    try:
        for block, cond in enumerate(ORDER, 1):
            sh(f"echo {cond} > /home/minhnhat/e2e/cmd_{block}")
            wait_board(f"ready_{block}", 60)
            if "BAD" in sh(f"cat /home/minhnhat/e2e/ready_{block}"):
                raise RuntimeError(f"helper refused condition {cond}")
            rknn = RKNN(verbose=False)
            try:
                if rknn.load_rknn(MODEL) != 0:
                    raise RuntimeError("load_rknn failed")
                t0 = time.time()
                if rknn.init_runtime(target="rk3588", device_id=DEVICE, core_mask=CORE_MASK["CORE_0"]) != 0:
                    raise RuntimeError("init_runtime failed")
                init_s = time.time() - t0
                sh(f"touch /home/minhnhat/e2e/inited_{block}")
                wait_board(f"started_{block}", 60)
                rows = []
                for i in range(WARMUP + CALLS):
                    t1 = time.time()
                    text = str(rknn.eval_perf(is_print=False, fix_freq=True))
                    wall = time.time() - t1
                    if i >= WARMUP:
                        rows.append((i - WARMUP, parse_summary_time_us(text), wall))
            finally:
                rknn.release()
            sh(f"touch /home/minhnhat/e2e/end_{block}")
            wait_board(f"done_{block}", 60)
            with open(HERE / f"b{block}_{cond}.csv", "w") as fh:
                fh.write("run,total_time_us,wall_s\n")
                fh.writelines(f"{r},{t},{w:.4f}\n" for r, t, w in rows)
            with open(HERE / "driver.log", "a") as fh:
                fh.write(f"{time.strftime('%T')} block {block} {cond} init_runtime {init_s:.2f} s, {len(rows)} calls\n")
            time.sleep(REST_S)
        sh("touch /home/minhnhat/e2e/stop")
        wait_board("daemon_exit", 60)
        subprocess.run(["scp", "-q", "-r", f"{HOST}:e2e/.", str(HERE)], check=True, timeout=300)
        marker.write_text("DONE\n")
    except BaseException as exc:  # noqa: BLE001
        sh("touch /home/minhnhat/e2e/stop")
        marker.write_text(f"ABORT {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
