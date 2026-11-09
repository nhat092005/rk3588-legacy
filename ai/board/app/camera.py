"""Continuous camera pipeline on the board: OV13855 -> ISP -> CPU -> NPU -> post-processing (-> ByteTrack), run ON THE BOARD.

Frames come from `v4l2-ctl` streaming NV12 from the ISP selfpath to stdout. The ISP crops 16:9 out of the 4224x3136
sensor mode (4224x2376) and scales it to WIDTHxHEIGHT, so the aspect ratio is kept. The crop is board state that
outlives this process, so it is set explicitly at start and always restored to the full frame on exit.
Each frame: NV12 -> BGR (cv2), preprocess, RKNN, postprocess_raw, optional BYTETracker (--track). One capture thread
keeps only the newest frame; frames it overwrites before the loop takes them, and gaps between arrivals, are counted.
Per-minute rows (FPS, NPU temperature, CPU) come from this loop's own timestamps.

A --duration-s run is a harness check of the pipeline. The Phase 9 acceptance run is --until-stable: it runs until the
NPU temperature is stable (the per-minute mean of 1 s samples; over 10 minute values none is more than 1 C above the first),
then 30 more full minutes, capped at --max-minutes (reaching the cap means "not stabilised"). Every completed minute is
appended to <out>.rows.jsonl (FPS, e2e mean/p95, NPU median, temperature, lowest NPU/DDR/CPU clocks, CPU, RSS), and the JSON
gets a `phase9` block with the stats of the first 5 minutes and of the stable 30 minutes. Latency here is software only: from the moment a full
frame has been read from the pipe to the end of post-processing, not exposure or ISP time.

--track needs `ultralytics.trackers` importable (the board has no ultralytics): put a copy on PYTHONPATH, together with
`lap` (numpy.linalg in the Kalman filter is ~100x slower with the default BLAS thread count, hence the
OPENBLAS_NUM_THREADS=1 default below).

Usage (on the board): taskset -c 4-7 .venv-board/bin/python -m ai.board.app.camera --model <rknn_dir> --out <json>
       --duration-s 60 --run-id <id> --git-commit <sha> --git-dirty true|false [--core-mask CORE_0] [--track]
       [--save-dir <dir> [--save-max 200] [--save-interval-s 1.0]] [--save-video <file.mp4> [--video-fps 30]]
       [--until-stable [--stable-window-min 10] [--stable-rise-c 1.0] [--stable-after-min 30] [--max-minutes 120]]

--save-dir keeps sample images: annotated frames that have detections, at most one per interval. --save-video writes every
frame, annotated (with the ByteTrack id when --track is on), to an MP4 through ffmpeg (libx264) pinned to the cores this
process may not use. Both do their drawing and encoding in background threads, but still use CPU and memory bandwidth,
so keep them off for timing runs (the measured cost of a run with them on is in the docs).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # must run before anything imports numpy

import argparse
from array import array
import gc
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil

from ai.board.common import (
    IMGSZ, ResourceMonitor, add_provenance_args, frequency_state, postprocess_raw, preprocess, read_text, summarize,
    system_info, thermal_zone,
)

V4L2 = "v4l2-ctl"
# (left, top, width, height) in OV13855 sensor pixels; the board exposes one 4224x3136 mode at 30 fps
MIN_ROW_S = 5.0
CROP_FULL = (0, 0, 4224, 3136)
CROP_16_9 = (0, 380, 4224, 2376)


def v4l2(device: str, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([V4L2, "-d", device, *args], check=True, capture_output=True, text=True, **kw)


def set_crop(device: str, crop: tuple[int, int, int, int]) -> None:
    left, top, w, h = crop
    v4l2(device, f"--set-selection=target=crop,top={top},left={left},width={w},height={h}")


def check_frame_size(device: str, width: int, height: int) -> int:
    """The pipe has no framing, so one frame must be exactly width*height*3/2 bytes (no row padding)."""
    out = v4l2(device, f"--try-fmt-video=width={width},height={height},pixelformat=NV12").stdout
    bpl = int(re.search(r"Bytes per Line\s*:\s*(\d+)", out).group(1))
    size = int(re.search(r"Size Image\s*:\s*(\d+)", out).group(1))
    if bpl != width or size != width * height * 3 // 2:
        raise RuntimeError(f"unexpected frame layout: bytesperline={bpl} sizeimage={size}")
    return size


def capture_command(device: str, width: int, height: int) -> list[str]:
    # v4l2-ctl block-buffers its stdout when it is a pipe: frames reached the reader in late/early pairs every 8 frames
    # although the kernel timestamps were even (measured: 59 late and 60 early intervals in 480 frames). stdbuf -o0 removes it.
    return ["stdbuf", "-o0", V4L2, "-d", device, f"--set-fmt-video=width={width},height={height},pixelformat=NV12",
            "--stream-mmap=4", "--stream-to=-"]


def read_exact(stream, buf: bytearray) -> bool:
    """Fill buf from stream, handling short reads. False when the stream ends first."""
    view = memoryview(buf)
    got = 0
    while got < len(buf):
        n = stream.readinto(view[got:])
        if not n:
            return False
        got += n
    return True


class Capture:
    """Reads frames from `stream` in a thread and keeps only the newest one."""

    def __init__(self, stream, frame_bytes: int):
        self.stream, self.frame_bytes = stream, frame_bytes
        self.cond = threading.Condition()
        self.slot = None  # (seq, arrival time, buffer) of the newest frame not yet taken
        self.seq = 0  # frames read from the pipe
        self.overwritten = 0  # frames replaced in the slot before the loop took them
        self.arrivals = array("d")  # one float per frame for hours: see Records
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            buf = bytearray(self.frame_bytes)
            if not read_exact(self.stream, buf):
                break
            now = time.perf_counter()
            with self.cond:
                if self.slot is not None:
                    self.overwritten += 1
                self.slot = (self.seq, now, buf)
                self.seq += 1
                self.arrivals.append(now)
                self.cond.notify()
        with self.cond:
            self.done = True
            self.cond.notify_all()

    def start(self) -> None:
        self.thread.start()

    def take(self, timeout: float):
        """Newest frame as (seq, arrival time, buffer), or None on timeout or when the stream has ended."""
        with self.cond:
            if self.slot is None and not self.done:
                self.cond.wait(timeout)
            item, self.slot = self.slot, None
            return item


def arrival_gaps(arrivals) -> dict:
    """Intervals between frames read from the pipe. `late_intervals` (above 1.5 periods) is a jitter indicator, not a
    loss count: a late frame is usually followed by an early one. Real upstream loss (V4L2 buffers are not dequeued
    while the pipe is full, so the ISP drops frames) is read from the ISP counters, see parse_isp_counters."""
    if len(arrivals) < 3:
        return {"period_ms": None, "late_intervals": 0, "interval_ms": None}
    dt = np.diff(arrivals)
    period = float(np.median(dt))
    return {
        "period_ms": round(period * 1e3, 3),
        "late_intervals": int((dt > 1.5 * period).sum()),
        "interval_ms": summarize(list(dt * 1e3)),
    }


def parse_isp_counters(line: str | None) -> dict:
    """Frame counter and frameloss from the /proc/rkisp0-vir1 'Output' line (counters restart with each stream)."""
    if not line:
        return {"isp_frames": None, "isp_frameloss": None}
    frame, loss = re.search(r"frame:(\d+)", line), re.search(r"frameloss:(\d+)", line)
    return {"isp_frames": int(frame.group(1)) if frame else None, "isp_frameloss": int(loss.group(1)) if loss else None}


MINUTE_S = 60.0  # one row per MINUTE_S seconds (tests shorten it)
SAMPLE_S = 1.0  # sensor sampling period
EARLY_MIN = 5  # the first minutes that "FPS at the start" is read from, compared with the stable window


def read_sensors(zone) -> dict:
    """One sample of the NPU temperature and of the clocks that thermal throttling would lower (cheap sysfs reads)."""

    def num(path):
        v = read_text(str(path))
        return int(v) if v and v.lstrip("-").isdigit() else None

    temp = num(zone / "temp") if zone else None
    cpu = {p.name: num(p / "scaling_cur_freq") for p in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))}
    return {"temp_c": temp / 1000 if temp is not None else None, "npu_hz": num("/sys/class/devfreq/fdab0000.npu/cur_freq"),
            "ddr_hz": num("/sys/class/devfreq/dmc/cur_freq"), "cpu_khz": cpu}


def _agg(values, fn):
    v = [x for x in values if x is not None]
    return fn(v) if v else None


class MinuteLog:
    """One row per minute from this loop's own timestamps and 1 s sensor samples: FPS, e2e mean and p95, NPU median,
    temperature mean and max, lowest NPU/DDR/CPU clocks of the minute, CPU, RSS. Rows are appended to a .jsonl file as
    they complete, so an interrupted run keeps its data. A last partial minute is reported only if it is at least
    MIN_ROW_S long: a few frames over a fraction of a second would give a meaningless FPS."""

    def __init__(self, path: Path | None = None, sample=None):
        self.path, self.rows, self.gc_ms = path, [], 0.0
        self.proc = psutil.Process()
        zone = thermal_zone()
        self.sample = sample or (lambda: read_sensors(zone))

    def start(self, t0: float) -> None:
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.next_boundary, self.next_sample = t0 + MINUTE_S, t0
        self._reset()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def _reset(self) -> None:
        self.e2e, self.npu, self.samples = [], [], []

    def add(self, rec: dict) -> None:
        self.e2e.append(rec["e2e"])
        self.npu.append(rec["npu"])

    def tick(self, now: float) -> list[dict]:
        """Take a sensor sample when due and return the rows of the minutes that ended."""
        if now >= self.next_sample:
            self.samples.append(self.sample())
            self.next_sample = now + SAMPLE_S
        out = []
        if now >= self.next_boundary:
            # rknnlite leaves ctypes buffers (1.2 MB input each) in reference cycles that the cyclic collector reaches only
            # now and then: RSS swung 80 to 140 MiB (measured). Collected here, after the frame is done, once per minute.
            t = time.perf_counter()
            gc.collect()
            self.gc_ms = (time.perf_counter() - t) * 1e3
        while now >= self.next_boundary:
            out.append(self._finish(MINUTE_S))
            self.next_boundary += MINUTE_S
        return out

    def finish(self, now: float) -> dict | None:
        span = now - (self.next_boundary - MINUTE_S)
        return self._finish(span, partial=True) if span >= MIN_ROW_S and self.e2e else None

    def _finish(self, span: float, partial: bool = False) -> dict:
        n, s = len(self.e2e), self.samples
        temps = [x["temp_c"] for x in s]
        row = {
            "minute": len(self.rows), "frames": n, "window_s": round(span, 1), "fps": round(n / span, 2),
            "e2e_ms_mean": round(float(np.mean(self.e2e)), 3) if n else None,
            "e2e_ms_p95": round(float(np.percentile(self.e2e, 95)), 3) if n else None,
            "npu_ms_p50": round(float(np.percentile(self.npu, 50)), 3) if n else None,
            "temp_c_mean": round(_agg(temps, np.mean), 2) if _agg(temps, np.mean) is not None else None,
            "temp_c_max": _agg(temps, max),
            "npu_hz_min": _agg([x["npu_hz"] for x in s], min), "ddr_hz_min": _agg([x["ddr_hz"] for x in s], min),
            "cpu_khz_min": {k: _agg([x["cpu_khz"].get(k) for x in s], min) for k in (s[0]["cpu_khz"] if s else {})},
            "process_cpu_pct": self.proc.cpu_percent(None), "system_cpu_pct": psutil.cpu_percent(None),
            "rss_mib": round(self.proc.memory_info().rss / 2**20, 1), "gc_collect_ms": round(self.gc_ms, 2),
        }
        if partial:
            row["partial"] = True
        self.rows.append(row)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        self._reset()
        return row


class Stability:
    """Phase 9 stop rule on the per-minute mean NPU temperature. Stable at minute k when, over the last `window`
    minute values, none is more than `rise_c` above the first one of the window (max(window) - window[0] <= rise_c).
    The run then continues for `after_min` full minutes. `max_minutes` caps the run: reaching it means not stabilised."""

    def __init__(self, window: int = 10, rise_c: float = 1.0, after_min: int = 30, max_minutes: int = 120):
        self.window, self.rise_c, self.after_min, self.max_minutes = window, rise_c, after_min, max_minutes
        self.temps: list[float] = []
        self.stable_at: int | None = None
        self.done, self.reason = False, None

    def update(self, row: dict) -> None:
        self.temps.append(row["temp_c_mean"])  # None when a minute had no sensor sample: that window cannot be judged
        if self.stable_at is None and len(self.temps) >= self.window:
            w = self.temps[-self.window:]
            if None not in w and max(w) - w[0] <= self.rise_c:
                self.stable_at = len(self.temps) - 1
        if self.stable_at is not None and len(self.temps) >= self.stable_at + 1 + self.after_min:
            self.done, self.reason = True, "stable_then_tail"
        elif len(self.temps) >= self.max_minutes:
            self.done, self.reason = True, "max_minutes"


class Records:
    """Per-frame measurements, one array('d') per key. A list of dicts cost about 0.5 KiB per frame in RSS, which a real
    pipeline does not have, and the collector walks a long list: gc.collect rose from 0.67 to 10.06 ms in 42 min (measured)."""

    def __init__(self):
        self.cols: dict[str, array] = {}

    def append(self, rec: dict) -> None:
        for key, value in rec.items():
            self.cols.setdefault(key, array("d")).append(value)

    def __len__(self) -> int:
        return len(next(iter(self.cols.values()))) if self.cols else 0

    def col(self, key: str) -> np.ndarray:
        """A copy: a view would block appending to the column."""
        return np.array(self.cols[key])


def run_loop(capture: Capture, step, duration_s: float, warmup: int, log: MinuteLog | None = None,
             stop: Stability | None = None):
    """step(nv12 ndarray) -> dict of stage times in ms and n_det. Returns (records, t_start, warmup frames done).
    With `stop`, the stop rule ends the measured window; otherwise it lasts `duration_s` after the warm-up frames."""
    records, done, t_start = Records(), 0, None
    t_end = time.perf_counter() + duration_s + 5  # until the warm-up frames are done
    while time.perf_counter() < t_end and not (stop and stop.done):
        item = capture.take(timeout=1.0)
        if item is None:
            if capture.done:
                break
            continue
        _, t_arr, buf = item
        res = step(np.frombuffer(buf, dtype=np.uint8))
        t_done = time.perf_counter()
        done += 1
        if done <= warmup:
            continue
        if t_start is None:
            t_start = t_arr
            t_end = float("inf") if stop else t_start + duration_s
            if log:
                log.start(t_start)
        rec = {"t_arr": t_arr, "t_done": t_done, "e2e": (t_done - t_arr) * 1e3, **res}
        records.append(rec)
        if log:
            for row in log.tick(t_done):  # minutes that ended before this frame, then the frame joins the new minute
                if stop:
                    stop.update(row)
            log.add(rec)
    if log and t_start is not None:
        log.finish(time.perf_counter())
    return records, t_start, min(done, warmup)


def window_stats(records: Records, rows: list[dict], t_start: float, first: int, last: int) -> dict | None:
    """Stats over the complete minutes first..last (inclusive) of the measured window, None if they are not all there."""
    sel_rows = rows[first:last + 1]
    if len(sel_rows) != last - first + 1 or any(r.get("partial") for r in sel_rows):
        return None
    lo, hi = t_start + MINUTE_S * first, t_start + MINUTE_S * (last + 1)
    t_done = records.col("t_done")
    sel = (t_done >= lo) & (t_done < hi)
    frames = int(sel.sum())
    if not frames:
        return None

    def vals(key):
        return [r[key] for r in sel_rows if r[key] is not None]

    return {
        "minutes": [first, last], "frames": frames, "fps": round(frames / (MINUTE_S * len(sel_rows)), 3),
        "e2e_ms": summarize(records.col("e2e")[sel]), "npu_ms": summarize(records.col("npu")[sel]),
        "process_cpu_pct_mean": round(float(np.mean(vals("process_cpu_pct"))), 1),
        "system_cpu_pct_mean": round(float(np.mean(vals("system_cpu_pct"))), 1),
        "rss_mib_max": max(vals("rss_mib")),
        "temp_c_mean_first_last": [sel_rows[0]["temp_c_mean"], sel_rows[-1]["temp_c_mean"]],
        "temp_c_max": _agg(vals("temp_c_max"), max),
        "npu_hz_min": _agg(vals("npu_hz_min"), min), "ddr_hz_min": _agg(vals("ddr_hz_min"), min),
        "cpu_khz_min": {k: _agg([r["cpu_khz_min"].get(k) for r in sel_rows], min) for k in sel_rows[0]["cpu_khz_min"]},
    }


def phase9_summary(stop: Stability, records: Records, rows: list[dict], t_start: float) -> dict:
    """Early window (first EARLY_MIN minutes) and the stable window (the `after_min` minutes after stabilisation)."""
    stable = stop.stable_at is not None
    return {
        "rule": {"temperature": "npu-thermal, per-minute mean of 1 s samples", "window_min": stop.window,
                 "rise_c": stop.rise_c, "after_min": stop.after_min, "max_minutes": stop.max_minutes},
        "stable": stable, "stable_at_minute": stop.stable_at,
        "stop_reason": stop.reason or "stream_ended_or_interrupted", "minutes_completed": len(stop.temps),
        "early": window_stats(records, rows, t_start, 0, EARLY_MIN - 1),
        "stable_window": (window_stats(records, rows, t_start, stop.stable_at + 1, stop.stable_at + stop.after_min)
                          if stable else None),
    }


class Detections:
    """Tiny Results-like adapter so numpy detections can feed ultralytics BYTETracker (same as scripts/demo/video/track.py)."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
        w, h = x2 - x1, y2 - y1
        return np.stack([x1 + w / 2, y1 + h / 2, w, h], axis=1)

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, key):
        return Detections(self.xyxy[key], self.conf[key], self.cls[key])


PALETTE = [(0, 170, 0), (0, 0, 220), (220, 120, 0), (0, 160, 220), (180, 0, 180), (160, 160, 0)]  # BGR, by class index


def annotate(bgr: np.ndarray, det: np.ndarray, names: list[str], ids: np.ndarray | None = None) -> np.ndarray:
    """Copy of the frame with a box and 'name conf' per detection (det rows: x1, y1, x2, y2, conf, cls, frame pixels).
    ids: ByteTrack id per detection (-1 = not tracked), appended to the label as '#id'."""
    import cv2

    img = bgr.copy()
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(det):
        cls = int(cls)
        color = PALETTE[cls % len(PALETTE)]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names[cls] if cls < len(names) else cls} {conf:.2f}"
        if ids is not None and ids[i] >= 0:
            label += f" #{int(ids[i])}"
        cv2.putText(img, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


class Saver:
    """Saves annotated frames that have detections, at most one per `interval_s` and `max_images` in total.
    Drawing and JPEG encoding run in a background thread, so only a frame copy happens inside the timed loop."""

    def __init__(self, directory: Path, names: list[str], max_images: int, interval_s: float):
        import queue

        self.dir, self.names, self.max, self.interval = Path(directory), names, max_images, interval_s
        self.dir.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=4)
        self.queued, self.dropped, self.saved, self.last = 0, 0, 0, None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def offer(self, bgr: np.ndarray, det: np.ndarray, ids: np.ndarray | None = None) -> None:
        import queue

        if len(det) == 0 or self.queued >= self.max:
            return
        now = time.perf_counter()
        if self.last is not None and now - self.last < self.interval:
            return
        try:
            self.queue.put_nowait((self.queued, bgr.copy(), det.copy(), None if ids is None else ids.copy()))
        except queue.Full:
            self.dropped += 1
            return
        self.queued += 1
        self.last = now

    def _run(self) -> None:
        import cv2

        while True:
            item = self.queue.get()
            if item is None:
                return
            index, bgr, det, ids = item
            cv2.imwrite(str(self.dir / f"frame_{index:04d}_{len(det)}det.jpg"), annotate(bgr, det, self.names, ids))
            self.saved += 1

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def summary(self) -> dict:
        return {"dir": str(self.dir), "saved": self.saved, "dropped_queue_full": self.dropped,
                "max_images": self.max, "interval_s": self.interval}


def ffmpeg_command(path: Path, width: int, height: int, fps: float) -> list[str]:
    # Fragmented MP4 (empty_moov): the file stays playable up to the last fragment even if the run is killed.
    return ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-framerate", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", str(path)]


class VideoSink:
    """Writes every frame, annotated, to a video through an ffmpeg process, fed from a background thread.
    The loop only puts references on a queue (frames are not modified after the step), so drawing and the pipe write
    stay out of the timed loop. ffmpeg is pinned to the cores this process is not allowed to use, if any. When the
    queue is full the frame is dropped and counted, never waited for."""

    def __init__(self, path: Path, names: list[str], command: list[str], queue_size: int = 60):
        import queue

        self.path, self.names, self.command = Path(path), names, command
        self.queue = queue.Queue(maxsize=queue_size)
        self.written, self.dropped, self.failed, self.proc, self.returncode, self.stderr = 0, 0, False, None, None, ""
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        others = sorted(set(range(os.cpu_count())) - os.sched_getaffinity(0))
        cmd = ["taskset", "-c", ",".join(map(str, others)), *self.command] if others else self.command
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.thread.start()

    def offer(self, bgr: np.ndarray, det: np.ndarray, ids: np.ndarray | None = None) -> None:
        import queue

        if self.failed:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait((bgr, det, ids))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                break
            if self.failed:
                self.dropped += 1
                continue
            try:
                self.proc.stdin.write(annotate(*item[:2], self.names, item[2]).data)
                self.written += 1
            except OSError:  # ffmpeg exited: stop writing, keep draining so the loop never blocks
                self.failed = True
                self.dropped += 1
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def close(self) -> None:
        import queue

        try:
            self.queue.put(None, timeout=10)
        except queue.Full:
            pass
        self.thread.join(120)
        self.stderr = self.proc.stderr.read().decode(errors="replace").strip()[-300:]
        self.proc.stderr.close()
        try:
            self.returncode = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.returncode = self.proc.wait()

    def summary(self) -> dict:
        return {"path": str(self.path), "frames_written": self.written, "dropped_queue_full": self.dropped,
                "ffmpeg_returncode": self.returncode, "ffmpeg_stderr": self.stderr,
                "size_bytes": self.path.stat().st_size if self.path.exists() else None}


def make_tracker(conf: float):
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError as exc:
        raise SystemExit(f"--track needs ultralytics.trackers on PYTHONPATH (and lap): {exc}")
    return BYTETracker(SimpleNamespace(track_high_thresh=conf, track_low_thresh=0.1, new_track_thresh=conf,
                                       track_buffer=30, match_thresh=0.8, fuse_score=True))


def make_step(rknn, normalized: bool, split: bool, conf: float, tracker, shape, saver: Saver | None = None,
              video: VideoSink | None = None):
    import cv2

    def step(nv: np.ndarray) -> dict:
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(nv.reshape(shape), cv2.COLOR_YUV2BGR_NV12)
        t1 = time.perf_counter()
        inp, meta = preprocess(bgr)
        t2 = time.perf_counter()
        outs = rknn.inference(inputs=[inp])
        t3 = time.perf_counter()
        det = postprocess_raw(outs, meta, normalized, split, conf)
        t4 = time.perf_counter()
        ids = None
        if tracker is not None:
            rows = tracker.update(Detections(det[:, :4], det[:, 4], det[:, 5]))
            ids = np.full(len(det), -1, dtype=int)  # tracker id per detection; row[7] is the detection index
            for row in rows:
                ids[int(row[7])] = int(row[4])
        t5 = time.perf_counter()
        # after the stage times are taken: the saver copies the frame for the saved frames only, the video sink copies nothing
        if saver is not None:
            saver.offer(bgr, det, ids)
        if video is not None:
            video.offer(bgr, det, ids)
        return {"cvt": (t1 - t0) * 1e3, "pre": (t2 - t1) * 1e3, "npu": (t3 - t2) * 1e3, "post": (t4 - t3) * 1e3,
                "track": (t5 - t4) * 1e3 if tracker is not None else 0.0, "n_det": len(det)}

    return step


def isp_output_line(device_tag: str = "rkisp0-vir1") -> str | None:
    try:
        for line in Path(f"/proc/{device_tag}").read_text().splitlines():
            if line.startswith("Output"):
                return " ".join(line.split())
    except OSError:
        pass
    return None


def main() -> None:
    import datetime
    import cv2
    import yaml
    from ai.board.bench.bench_npu import CORE_MASK, load_model
    from ai.board.common import provenance

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="rknn_<precision>/ folder")
    parser.add_argument("--core-mask", default="CORE_0", choices=list(CORE_MASK))
    parser.add_argument("--device", default="/dev/video12", help="ISP selfpath node")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-crop", action="store_true", help="scale the full 4224x3136 frame (distorts the aspect ratio)")
    parser.add_argument("--duration-s", type=float, default=60.0, help="measured window, after the warm-up frames")
    parser.add_argument("--until-stable", action="store_true",
                        help="Phase 9: run until the NPU temperature is stable, then --stable-after-min more minutes "
                             "(--duration-s is ignored, --max-minutes caps the run)")
    parser.add_argument("--stable-window-min", type=int, default=10)
    parser.add_argument("--stable-rise-c", type=float, default=1.0)
    parser.add_argument("--stable-after-min", type=int, default=30)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--rows-jsonl", type=Path, help="per-minute rows as they complete (default: <out>.rows.jsonl)")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--save-dir", type=Path, help="save annotated frames that have detections (sample images)")
    parser.add_argument("--save-max", type=int, default=200)
    parser.add_argument("--save-interval-s", type=float, default=1.0, help="at most one saved frame per interval")
    parser.add_argument("--save-video", type=Path, help="write every frame, annotated, to this .mp4 (needs ffmpeg)")
    parser.add_argument("--video-fps", type=float, default=30.0, help="frame rate stored in the video (the camera gives 29.95)")
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()
    if max(args.width, args.height) > IMGSZ:
        raise SystemExit("frame larger than the model input needs a resize step this harness does not have")
    if args.save_video and shutil.which("ffmpeg") is None:  # checked before anything touches the camera
        raise SystemExit("--save-video needs ffmpeg on PATH")
    if args.until_stable and thermal_zone() is None:
        raise SystemExit("--until-stable needs the npu-thermal zone, which was not found")

    meta_yaml = yaml.safe_load((args.model / "metadata.yaml").read_text())
    rknn, normalized = load_model(args.model, args.core_mask)
    split = bool(meta_yaml.get("split_head"))
    tracker = make_tracker(args.conf) if args.track else None
    shape = (args.height * 3 // 2, args.width)
    names = [meta_yaml["names"][i] for i in range(len(meta_yaml["names"]))]
    saver = Saver(args.save_dir, names, args.save_max, args.save_interval_s) if args.save_dir else None
    video = (VideoSink(args.save_video, names, ffmpeg_command(args.save_video, args.width, args.height, args.video_fps))
             if args.save_video else None)
    step = make_step(rknn, normalized, split, args.conf, tracker, shape, saver, video)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so the finally below restores the crop
    crop = CROP_FULL if args.no_crop else CROP_16_9
    proc = None
    try:
        set_crop(args.device, crop)  # explicit in every mode: the previous run may have left another crop
        frame_bytes = check_frame_size(args.device, args.width, args.height)
        proc = subprocess.Popen(capture_command(args.device, args.width, args.height), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
        # Everything alive now (model, libraries) leaves the collector's view, so the per-minute gc.collect in MinuteLog.tick
        # walks only what the loop creates: a full collect took 26 to 29 ms and overwrote a frame each minute (measured).
        gc.collect()
        gc.freeze()
        capture = Capture(proc.stdout, frame_bytes)
        capture.start()
        if saver is not None:
            saver.start()
        if video is not None:
            video.start()
        freq_before = frequency_state()
        log = MinuteLog(args.rows_jsonl or args.out.with_suffix(".rows.jsonl"))
        stop = (Stability(args.stable_window_min, args.stable_rise_c, args.stable_after_min, args.max_minutes)
                if args.until_stable else None)
        with ResourceMonitor() as monitor:
            records, t_start, warm_done = run_loop(capture, step, 60.0 if stop else args.duration_s, args.warmup_frames,
                                                   log, stop)
        isp_line = isp_output_line()  # read while the stream is still running
        freq_after = frequency_state()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        set_crop(args.device, CROP_FULL)
        if saver is not None and saver.thread.is_alive():
            saver.close()  # writes the frames still queued
        if video is not None and video.proc is not None:
            video.close()  # lets ffmpeg finish the file
    if not records:
        raise SystemExit(f"no frames processed after {warm_done} warm-up frames; is the camera streaming?")

    stage = {k: summarize(records.col(k)) for k in ("cvt", "pre", "npu", "post", "e2e")}
    if tracker is not None:
        stage["track"] = summarize(records.col("track"))
    window = float(records.col("t_done")[-1]) - t_start
    result = {
        **provenance(args, next(args.model.glob("*.rknn"))),
        "tag": args.model.name,
        "backend": "rknnlite-on-board",
        "camera": {"device": args.device, "frame": f"{args.width}x{args.height} NV12", "crop_ltwh": list(crop),
                   "sensor_mode": "4224x3136 @30", "isp_output_line": isp_line, **parse_isp_counters(isp_line),
                   "frames_isp_minus_read": (parse_isp_counters(isp_line)["isp_frames"] or 0) - capture.seq,
                   **arrival_gaps(capture.arrivals)},
        "conditions": {"affinity": sorted(os.sched_getaffinity(0)), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                       "cv2_threads": cv2.getNumThreads(), "core_mask": args.core_mask, "tracker": bool(tracker),
                       "freq_before": freq_before, "freq_after": freq_after},
        "frames": {"read_from_pipe": capture.seq, "processed_measured": len(records), "warmup": warm_done,
                   "overwritten_in_slot": capture.overwritten},
        "window_s": round(window, 1),
        "fps": round(len(records) / window, 2),
        "latency_ms": stage,
        "n_det_mean": round(float(np.mean(records.col("n_det"))), 2),
        "saved_images": saver.summary() if saver is not None else None,
        "saved_video": video.summary() if video is not None else None,
        "per_minute": log.rows,
        "rows_jsonl": str(log.path),
        "phase9": phase9_summary(stop, records, log.rows, t_start) if stop else None,
        "resources": monitor.result(),
        "system": system_info(),
        "sdk_version": str(rknn.get_sdk_version()),
        "created_note": datetime.date.today().isoformat(),
    }
    rknn.release()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    cam = result["camera"]
    print(f"Done: {args.out} fps={result['fps']} e2e_p50={stage['e2e']['p50']} e2e_p95={stage['e2e']['p95']} ms frames={len(records)} "
          f"overwritten={capture.overwritten} late_intervals={cam['late_intervals']} isp_frames={cam['isp_frames']} "
          f"read={capture.seq} isp_frameloss={cam['isp_frameloss']}")


if __name__ == "__main__":
    main()
