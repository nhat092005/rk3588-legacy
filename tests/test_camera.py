"""Camera harness logic (frame pipe, newest-frame slot, drop counting, per-minute rows) with fakes, no board needed.

Run: .venv/bin/python -m unittest tests.test_camera
"""
import gc
import io
import time
import unittest
from unittest import mock

import numpy as np

from ai.board import common
from ai.board.app import camera


class Trickle(io.RawIOBase):
    """Gives at most 7 bytes per readinto, like a pipe delivering short reads."""

    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    def readable(self):
        return True

    def readinto(self, b):
        n = min(len(b), 7, len(self.data) - self.pos)
        b[:n] = self.data[self.pos:self.pos + n]
        self.pos += n
        return n


class PacedStream:
    """n frames of `size` bytes, one frame every `delay` s."""

    def __init__(self, n, size, delay):
        self.left, self.size, self.delay, self.cur = n, size, delay, 0

    def readinto(self, b):
        if self.cur == 0:
            if self.left == 0:
                return 0
            time.sleep(self.delay)
            self.left -= 1
            self.cur = self.size
        n = min(len(b), self.cur)
        self.cur -= n
        return n


class FakeRknn:
    def inference(self, inputs):
        return [np.zeros((1, 9, 8400), dtype=np.float32)]


class Camera(unittest.TestCase):
    def test_read_exact_handles_short_reads_and_eof(self):
        data = bytes(range(40))
        buf = bytearray(30)
        self.assertTrue(camera.read_exact(Trickle(data), buf))
        self.assertEqual(bytes(buf), data[:30])
        self.assertFalse(camera.read_exact(Trickle(data[:20]), bytearray(30)))

    def test_capture_keeps_only_the_newest_frame_and_counts_overwrites(self):
        frames = b"".join(bytes([i]) * 12 for i in range(5))
        cap = camera.Capture(Trickle(frames), 12)
        cap.start()
        cap.thread.join(5)
        self.assertEqual((cap.seq, cap.overwritten), (5, 4))
        seq, _, buf = cap.take(0.1)
        self.assertEqual((seq, bytes(buf)), (4, bytes([4]) * 12))
        self.assertIsNone(cap.take(0.1))

    def test_arrival_gaps_counts_late_intervals(self):
        t = [i / 30 for i in range(20)]
        t = t[:10] + [x + 2 / 30 for x in t[10:]]  # one late interval after the 10th frame
        gaps = camera.arrival_gaps(t)
        self.assertEqual(gaps["late_intervals"], 1)
        self.assertAlmostEqual(gaps["period_ms"], 33.333, places=2)
        self.assertNotIn("missing_frames", gaps)  # loss comes from the ISP counters, not from interval jitter

    def test_capture_command_unbuffers_stdout_and_streams_to_the_pipe(self):
        cmd = camera.capture_command("/dev/video12", 640, 360)
        self.assertEqual(cmd[:3], ["stdbuf", "-o0", "v4l2-ctl"])  # without it frames arrive in late/early pairs
        self.assertIn("--stream-to=-", cmd)
        self.assertIn("--set-fmt-video=width=640,height=360,pixelformat=NV12", cmd)

    def test_parse_isp_counters(self):
        line = "Output rkisp_selfpath Format:NV12 Size:640x360 Dcrop(0,380|4224x2376) (frame:84 rate:33ms delay:29ms frameloss:2 bufcnt:4)"
        self.assertEqual(camera.parse_isp_counters(line), {"isp_frames": 84, "isp_frameloss": 2})
        self.assertEqual(camera.parse_isp_counters(None), {"isp_frames": None, "isp_frameloss": None})

    def test_every_frame_is_processed_or_overwritten(self):
        size = 640 * 360 * 3 // 2
        cap = camera.Capture(PacedStream(80, size, 0.002), size)
        cap.start()
        seen = []

        def step(nv):
            seen.append(nv.size)
            time.sleep(0.003)  # slower than the stream, so frames get overwritten
            return {"cvt": 1.0, "pre": 1.0, "npu": 1.0, "post": 1.0, "track": 0.0, "n_det": 0}

        records, t_start, warm = camera.run_loop(cap, step, duration_s=30, warmup=5)
        self.assertEqual(warm, 5)
        self.assertEqual(len(records) + warm + cap.overwritten, cap.seq)
        self.assertEqual(cap.seq, 80)
        self.assertGreater(cap.overwritten, 0)
        self.assertTrue(all(s == size for s in seen))
        self.assertAlmostEqual(records.col("t_arr")[0], t_start)

    def test_saver_writes_only_frames_with_detections_and_respects_interval_and_max(self):
        import tempfile
        from pathlib import Path

        import cv2

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        det = np.array([[100, 80, 220, 300, 0.9, 1], [300, 50, 380, 200, 0.4, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            saver = camera.Saver(Path(tmp) / "out", ["Helmet", "NoHelmet"], max_images=2, interval_s=3600)
            saver.start()
            saver.offer(frame, np.zeros((0, 6), dtype=np.float32))  # no detection: skipped
            saver.offer(frame, det)  # saved
            saver.offer(frame, det)  # inside the interval: skipped
            saver.close()
            files = sorted((Path(tmp) / "out").glob("*.jpg"))
            self.assertEqual([f.name for f in files], ["frame_0000_2det.jpg"])
            img = cv2.imread(str(files[0]))
            self.assertEqual(img.shape, (360, 640, 3))
            self.assertFalse(np.array_equal(img, frame))  # boxes and labels were drawn
            self.assertEqual(saver.summary()["saved"], 1)

            saver = camera.Saver(Path(tmp) / "cap", ["Helmet", "NoHelmet"], max_images=2, interval_s=0)
            saver.start()
            for _ in range(5):
                saver.offer(frame, det)
                time.sleep(0.05)
            saver.close()
            self.assertEqual(len(list((Path(tmp) / "cap").glob("*.jpg"))), 2)  # capped at max_images

    def test_annotate_does_not_modify_the_input_frame(self):
        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        before = frame.copy()
        out = camera.annotate(frame, np.array([[10, 10, 100, 100, 0.5, 7]], dtype=np.float32), ["a"])  # class beyond names
        self.assertTrue(np.array_equal(frame, before))
        self.assertFalse(np.array_equal(out, before))

    def test_annotate_writes_the_tracker_id_into_the_label(self):
        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        det = np.array([[100, 80, 220, 300, 0.9, 0], [300, 50, 380, 200, 0.4, 0]], dtype=np.float32)
        plain = camera.annotate(frame, det, ["Person"])
        with_ids = camera.annotate(frame, det, ["Person"], np.array([7, -1]))
        self.assertFalse(np.array_equal(plain, with_ids))  # '#7' is drawn on the first box only
        self.assertTrue(np.array_equal(plain[60:300, 280:400], with_ids[60:300, 280:400]))

    def test_ffmpeg_command_reads_raw_bgr_from_stdin_and_writes_a_fragmented_mp4(self):
        cmd = camera.ffmpeg_command(__import__("pathlib").Path("/tmp/x.mp4"), 640, 360, 30.0)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("640x360", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], "-")
        self.assertIn("frag_keyframe+empty_moov", cmd)
        self.assertEqual(cmd[-1], "/tmp/x.mp4")

    def test_video_sink_feeds_every_annotated_frame_to_the_process(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "raw.bin"
            sink = camera.VideoSink(out, ["Person"], ["sh", "-c", f"cat > {out}"])
            sink.start()
            frame = np.full((48, 64, 3), 90, dtype=np.uint8)
            for _ in range(5):
                sink.offer(frame, np.array([[5, 5, 30, 40, 0.8, 0]], dtype=np.float32), np.array([3]))
            sink.close()
            self.assertEqual((sink.written, sink.dropped, sink.returncode), (5, 0, 0))
            self.assertEqual(out.stat().st_size, 5 * 64 * 48 * 3)
            self.assertEqual(sink.summary()["size_bytes"], 5 * 64 * 48 * 3)

    def test_video_sink_drops_instead_of_waiting_when_the_encoder_is_slow(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "raw.bin"
            sink = camera.VideoSink(out, ["Person"], ["sh", "-c", f"sleep 2; cat > {out}"], queue_size=2)
            sink.start()
            frame = np.full((100, 100, 3), 90, dtype=np.uint8)
            t0 = time.perf_counter()
            for _ in range(40):
                sink.offer(frame, np.zeros((0, 6), dtype=np.float32))
            self.assertLess(time.perf_counter() - t0, 0.5)  # offer never blocks the loop
            sink.close()
            self.assertGreater(sink.dropped, 0)
            self.assertEqual(sink.written + sink.dropped, 40)

    def test_step_gives_the_saver_and_video_the_tracker_id_of_each_detection(self):
        class Rknn:
            def inference(self, inputs):
                out = np.zeros((1, 9, 8400), dtype=np.float32)
                out[0, :4, 0] = [320, 320, 100, 100]  # one anchor: cx, cy, w, h in input pixels
                out[0, 4, 0] = 0.9  # class 0 score
                return [out]

        class Tracker:
            def update(self, dets):
                return np.array([[0, 0, 10, 10, 5, 0.9, 0, 0]], dtype=np.float32)  # id 5 for detection 0

        class Sink:
            def __init__(self):
                self.calls = []

            def offer(self, bgr, det, ids=None):
                self.calls.append((len(det), None if ids is None else ids.tolist()))

        saver, video = Sink(), Sink()
        step = camera.make_step(Rknn(), False, False, 0.25, Tracker(), (360 * 3 // 2, 640), saver, video)
        res = step(np.full(640 * 360 * 3 // 2, 128, dtype=np.uint8))
        self.assertEqual(res["n_det"], 1)
        self.assertEqual(saver.calls, [(1, [5])])
        self.assertEqual(video.calls, [(1, [5])])

    def test_minute_log_rows_fps_p95_and_jsonl_match_the_completed_minutes(self):
        import json
        import tempfile
        from pathlib import Path

        sample = lambda: {"temp_c": 50.0, "npu_hz": 1_000_000_000, "ddr_hz": 2_112_000_000, "cpu_khz": {"policy4": 2256000}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            log = camera.MinuteLog(path, sample=sample)
            t0 = 1000.0
            log.start(t0)
            rows = []
            for i in range(30 * 130):
                t = t0 + i / 30 + 0.02
                rows += log.tick(t)
                log.add({"e2e": 30.0 if i % 100 == 0 else 20.0, "npu": 18.0})
            partial = log.finish(t0 + 130.0)
            self.assertEqual([r["minute"] for r in rows], [0, 1])
            self.assertEqual((rows[0]["frames"], rows[0]["fps"], rows[1]["fps"]), (1800, 30.0, 30.0))
            self.assertEqual((rows[0]["e2e_ms_mean"], rows[0]["e2e_ms_p95"], rows[0]["npu_ms_p50"]), (20.1, 20.0, 18.0))
            self.assertEqual((rows[0]["temp_c_mean"], rows[0]["npu_hz_min"], rows[0]["cpu_khz_min"]), (50.0, 1_000_000_000, {"policy4": 2256000}))
            self.assertTrue(partial["partial"])
            self.assertEqual((partial["frames"], partial["window_s"]), (300, 10.0))
            lines = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(lines), len(log.rows))  # the file gets one line per completed minute, plus the partial one
            self.assertEqual(lines[0]["fps"], 30.0)

    def test_minute_log_skips_a_trailing_window_shorter_than_five_seconds(self):
        log = camera.MinuteLog(None, sample=lambda: {"temp_c": 50.0, "npu_hz": 1, "ddr_hz": 1, "cpu_khz": {}})
        log.start(0.0)
        log.add({"e2e": 20.0, "npu": 18.0})
        self.assertIsNone(log.finish(3.0))

    def test_minute_log_collects_cyclic_garbage_once_per_minute_and_reports_the_time(self):
        class Cycle:
            pass

        log = camera.MinuteLog(None, sample=lambda: {"temp_c": 50.0, "npu_hz": 1, "ddr_hz": 1, "cpu_khz": {}})
        log.start(0.0)
        log.add({"e2e": 20.0, "npu": 18.0})
        import weakref

        a, b = Cycle(), Cycle()
        a.other, b.other = b, a  # reference cycle: freed by the cyclic collector only
        ref = weakref.ref(a)
        del a, b
        self.assertIsNotNone(ref())
        self.assertEqual(log.tick(1.0), [])  # inside the first minute: no collect
        self.assertIsNotNone(ref())
        rows = log.tick(camera.MINUTE_S + 1.0)
        self.assertIsNone(ref())
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["gc_collect_ms"], 0.0)

    def test_stability_falling_temperature_counts_as_stable(self):
        stop = camera.Stability(window=10, rise_c=1.0, after_min=3, max_minutes=100)
        for k in range(30):
            stop.update({"temp_c_mean": 60.0 - k})
            if stop.done:
                break
        self.assertEqual((stop.stable_at, stop.done, stop.reason), (9, True, "stable_then_tail"))
        self.assertEqual(len(stop.temps), 9 + 1 + 3)  # three full minutes after the stable one

    def test_stability_rise_just_above_the_limit_is_not_stable_and_the_cap_ends_the_run(self):
        slow = camera.Stability(window=10, rise_c=1.0, after_min=3, max_minutes=40)
        for k in range(40):
            slow.update({"temp_c_mean": 50.0 + 0.12 * k})  # 1.08 C over the 9 steps of a window
        self.assertEqual((slow.stable_at, slow.done, slow.reason), (None, True, "max_minutes"))
        ok = camera.Stability(window=10, rise_c=1.0, after_min=3, max_minutes=40)
        for k in range(40):
            ok.update({"temp_c_mean": 50.0 + 0.1 * k})  # 0.9 C over a window
            if ok.done:
                break
        self.assertEqual(ok.stable_at, 9)

    def test_stability_a_minute_without_a_temperature_never_crashes_or_counts_as_stable(self):
        stop = camera.Stability(window=3, rise_c=1.0, after_min=1, max_minutes=20)
        for temp in [50.0, None, 50.0, 50.0]:  # the window [None, 50, 50] cannot be judged
            stop.update({"temp_c_mean": temp})
        self.assertIsNone(stop.stable_at)
        stop.update({"temp_c_mean": 50.0})  # window [50, 50, 50]
        self.assertEqual(stop.stable_at, 4)

    def test_records_keep_per_frame_values_in_columns_the_collector_does_not_walk(self):
        records = camera.Records()
        self.assertEqual(len(records), 0)
        self.assertFalse(records)
        for i in range(3):
            records.append({"t_done": float(i), "e2e": 19.0 + i, "n_det": i})
        self.assertEqual(len(records), 3)
        self.assertEqual(records.col("e2e").tolist(), [19.0, 20.0, 21.0])
        self.assertEqual(records.col("n_det").tolist(), [0.0, 1.0, 2.0])
        records.append({"t_done": 3.0, "e2e": 22.0, "n_det": 3})  # col() returns a copy: the column can still grow
        self.assertEqual(len(records.col("t_done")), 4)
        self.assertTrue(all(len(gc.get_referents(c)) <= 1 for c in records.cols.values()))  # at most its type, not the values

    def test_resource_monitor_keeps_samples_the_collector_does_not_walk_and_reports_the_same_fields(self):
        with mock.patch("ai.board.common.temperature_c", side_effect=[40.0, 50.0, None, 60.0, 45.0] + [55.0] * 100000):
            monitor = common.ResourceMonitor(interval=0.01)
            with monitor:
                time.sleep(0.1)
        cols = (monitor.rss_mib, monitor.proc_cpu, monitor.sys_cpu, monitor.temps)
        self.assertTrue(all(len(gc.get_referents(c)) <= 1 for c in cols))  # at most the type, not the samples
        result = monitor.result()
        self.assertGreater(result["samples"], 0)
        self.assertGreater(result["peak_rss_mib"], 0)
        self.assertIsInstance(result["avg_process_cpu_pct"], float)
        self.assertEqual(result["temp_start_c"], 40.0)
        self.assertGreaterEqual(result["temp_max_c"], 50.0)  # a missing reading (None) does not break the samples

    def test_window_stats_select_the_right_minutes(self):
        sample = lambda: {"temp_c": 50.0, "npu_hz": 1, "ddr_hz": 1, "cpu_khz": {"policy4": 1}}
        log = camera.MinuteLog(None, sample=sample)
        t0 = 1000.0
        log.start(t0)
        records = camera.Records()
        for i in range(30 * 6 * 60 + 30 * 20):  # 6 full minutes and 20 s
            t = t0 + i / 30 + 0.02
            log.tick(t)
            rec = {"t_done": t, "e2e": 20.0 if t - t0 < 180 else 30.0, "npu": 18.0}
            records.append(rec)
            log.add(rec)
        log.finish(t0 + 380.0)
        early = camera.window_stats(records, log.rows, t0, 0, 1)
        late = camera.window_stats(records, log.rows, t0, 3, 4)
        self.assertEqual((early["minutes"], early["frames"], early["e2e_ms"]["p50"]), ([0, 1], 3600, 20.0))
        self.assertEqual((late["minutes"], late["frames"], late["e2e_ms"]["p50"], late["fps"]), ([3, 4], 3600, 30.0, 30.0))
        self.assertIsNone(camera.window_stats(records, log.rows, t0, 5, 6))  # minute 6 is only a partial row

    def test_run_loop_stops_by_itself_when_the_stop_rule_is_met(self):
        from unittest import mock

        size = 640 * 360 * 3 // 2
        cap = camera.Capture(PacedStream(2000, size, 0.002), size)  # 4 s of frames
        cap.start()
        log = camera.MinuteLog(None, sample=lambda: {"temp_c": 50.0, "npu_hz": 1, "ddr_hz": 1, "cpu_khz": {}})
        stop = camera.Stability(window=3, rise_c=1.0, after_min=2, max_minutes=50)

        def step(nv):
            return {"cvt": 1.0, "pre": 1.0, "npu": 1.0, "post": 1.0, "track": 0.0, "n_det": 0}

        with mock.patch.object(camera, "MINUTE_S", 0.2), mock.patch.object(camera, "MIN_ROW_S", 0.05), \
                mock.patch.object(camera, "SAMPLE_S", 0.02):
            records, t_start, warm = camera.run_loop(cap, step, duration_s=60, warmup=3, log=log, stop=stop)
        self.assertEqual((stop.done, stop.reason, stop.stable_at), (True, "stable_then_tail", 2))
        self.assertEqual(len(stop.temps), 5)  # stable at minute 2, then 2 more full minutes
        self.assertLess(cap.seq, 2000)  # it did not wait for the stream to end

    def test_step_runs_cvt_preprocess_and_postprocess_on_a_nv12_frame(self):
        shape = (360 * 3 // 2, 640)
        step = camera.make_step(FakeRknn(), False, False, 0.25, None, shape)
        res = step(np.full(640 * 360 * 3 // 2, 128, dtype=np.uint8))
        self.assertEqual(res["n_det"], 0)
        self.assertTrue(all(res[k] > 0 for k in ("cvt", "pre", "npu", "post")))
        self.assertEqual(res["track"], 0.0)


if __name__ == "__main__":
    unittest.main()
